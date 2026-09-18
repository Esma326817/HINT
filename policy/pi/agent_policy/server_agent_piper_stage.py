

from __future__ import annotations

import base64
import dataclasses
import json
import logging
import os
import sys
import threading
import time
import traceback
from typing import Optional

import cv2
import numpy as np
import requests
from scipy.spatial.transform import Rotation
from PIL import Image
import websockets.exceptions
import websockets.sync.server

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(_root, "src"), _root, _scripts_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from openpi_client import msgpack_numpy
from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _train_config
from scripts.EmaFilter import EmaFilter

logger = logging.getLogger(__name__)


def _squeeze_obs_for_policy(obs: dict) -> dict:
    """Drop leading batch dim of 1 everywhere (matches websocket_policy_server)."""
    def squeeze_first_dim(value):
        if isinstance(value, dict):
            return {k: squeeze_first_dim(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(squeeze_first_dim(v) for v in value)
        if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == 1:
            return value.squeeze(0)
        return value

    obs = squeeze_first_dim(obs)
    if "observation/state" in obs:
        st = obs["observation/state"]
        if isinstance(st, np.ndarray) and st.ndim > 0 and st.shape[0] == 1:
            obs["observation/state"] = st.squeeze(0)
    return obs


# ==============================================================================
# Observation / Action conversion
# ==============================================================================

def _decode_image(raw) -> np.ndarray:
    """Decode base64, raw bytes, numpy, or msgpack-numpy dict → uint8 HWC RGB."""
    if isinstance(raw, dict):
        if "data" in raw and "shape" in raw:
            dtype = np.dtype(raw.get("type", "uint8"))
            arr = np.frombuffer(raw["data"], dtype=dtype).reshape(raw["shape"]).copy()
            return _decode_image(arr)
        return np.zeros((480, 640, 3), dtype=np.uint8)
    if isinstance(raw, list):
        raw = np.asarray(raw, dtype=np.uint8)
    if isinstance(raw, str):
        raw = base64.b64decode(raw)
    if isinstance(raw, np.ndarray) and raw.ndim >= 2:
        img = raw.astype(np.uint8) if raw.dtype != np.uint8 else raw
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        return img
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((480, 640, 3), dtype=np.uint8)


# Camera frames sent to ReasoningAgent: native Piper resolution (width x height).
# RA grounds/tracks at this size, then always returns RGB at output_resolution
# (typically 224x224) and attention maps in model_input_resolution space.
VLM_IMAGE_W, VLM_IMAGE_H = 640, 480
# π05 / attention policy RGB tensors: direct resize (no letterbox) to IMAGE_RESOLUTION.
POLICY_IMAGE_H, POLICY_IMAGE_W = _model.IMAGE_RESOLUTION  # (224, 224)


def _resize_hwc(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Direct-resize an HWC RGB uint8 frame (no padding), matching ReasoningAgent export."""
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[:2] == (height, width):
        return arr
    return np.asarray(Image.fromarray(arr).resize((width, height)))


def _resize_for_vlm(image: np.ndarray) -> np.ndarray:
    """Resize camera frames to the 640x480 size ReasoningAgent expects."""
    return _resize_hwc(image, VLM_IMAGE_W, VLM_IMAGE_H)


def _resize_for_attention_policy(image: np.ndarray) -> np.ndarray:
    """Direct-resize to π05 / attention training resolution (224x224, no pad)."""
    return _resize_hwc(image, POLICY_IMAGE_W, POLICY_IMAGE_H)


# ReasoningAgent modes whose returned frames already carry the mask overlay.
_PAINTED_SEMANTIC_INTENTS = ("highlighting", "both")


def _policy_image(rendered: np.ndarray | None, source: np.ndarray, use_rendered: bool) -> np.ndarray:
    """RA's frame when it painted one, else a local direct resize of the source."""
    if use_rendered and rendered is not None:
        return rendered
    return _resize_for_attention_policy(source)


_CAM_MAP = {
    "camera_front": "observation/global",
    "camera_left": "observation/left_wrist",
    "camera_right": "observation/right_wrist",
}


def _client_prompt(obs: dict, default_prompt: str = "") -> str:
    """Read episode prompt from the robot client observation.

    Looks at top-level and nested ``state`` for ``instruction`` / ``prompt`` /
    ``task``. Falls back to ``default_prompt`` (usually ``--agent-prompt``) when
    the client sends nothing usable.
    """
    sd = obs.get("state", obs)
    if not isinstance(sd, dict):
        sd = {}
    for key in ("instruction", "prompt", "task"):
        val = obs.get(key)
        if val is None:
            val = sd.get(key)
        if val is None:
            continue
        if isinstance(val, np.ndarray):
            val = val.flat[0] if val.size else ""
        if isinstance(val, (list, tuple)):
            val = val[0] if val else ""
        text = str(val).strip()
        if text and text != "None":
            return text
    return str(default_prompt or "").strip()


def _client_to_openpi(obs: dict, default_prompt: str) -> tuple[dict, np.ndarray]:
    sd = obs.get("state", obs)

    f1 = np.array(sd.get("follow1_pos", [0.0] * 7), dtype=np.float32)[:7]
    f2 = np.array(sd.get("follow2_pos", [0.0] * 7), dtype=np.float32)[:7]
    state = np.concatenate([f1, f2]).astype(np.float32)

    prompt = _client_prompt(obs, default_prompt)

    openpi_obs = {"observation/state": state, "prompt": prompt}

    views = obs.get("views", obs)
    for client_key, openpi_key in _CAM_MAP.items():
        raw = views.get(client_key)
        if raw is not None:
            openpi_obs[openpi_key] = _decode_image(raw)
        else:
            openpi_obs[openpi_key] = np.zeros((480, 640, 3), dtype=np.uint8)

    return openpi_obs, state


def _decode_array(raw) -> np.ndarray:
    """Decode state vector: ndarray (OpenPI client), list, or legacy msgpack-numpy dict."""
    if isinstance(raw, np.ndarray):
        return np.asarray(raw, dtype=np.float32).reshape(-1)
    if isinstance(raw, dict) and "data" in raw and "shape" in raw:
        dtype = np.dtype(raw.get("type", "float32"))
        return np.frombuffer(raw["data"], dtype=dtype).reshape(raw["shape"]).copy()
    return np.asarray(raw, dtype=np.float32).reshape(-1)


def _extract_robot_state_14d(obs: dict) -> list[float]:
    """Extract 14D robot state (pos3+euler3+grip1 × 2 arms) from either format."""
    raw_state = obs.get("observation/state")
    if raw_state is not None:
        state = _decode_array(raw_state).astype(np.float32).flatten()[:14]
        if len(state) < 14:
            state = np.pad(state, (0, 14 - len(state)))
        return state.tolist()
    sd = obs.get("state", obs)
    f1 = np.array(sd.get("follow1_pos", [0.0] * 7), dtype=np.float32)[:7]
    f2 = np.array(sd.get("follow2_pos", [0.0] * 7), dtype=np.float32)[:7]
    return np.concatenate([f1, f2]).astype(np.float32).tolist()


def _raw_tcp_pose_vector(obs: dict) -> Optional[np.ndarray]:
    """Piper TCP pose vector if present: native ``observation/tcp_pose`` or legacy ``state.tcp_pose``."""
    raw = obs.get("observation/tcp_pose")
    if raw is None:
        sd = obs.get("state", obs)
        if isinstance(sd, dict):
            raw = sd.get("tcp_pose")
    if raw is None:
        return None
    return _decode_array(raw).astype(np.float64).ravel()


TCP_POSE_PER_ARM = 7
TCP_POSE_DIM = TCP_POSE_PER_ARM * 2


def _tcp_wxyz_to_pos_euler_xyz(tcp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-arm TCP slice (7,): position xyz then quaternion wxyz → pos (3,), euler xyz radians (3,)."""
    v = np.asarray(tcp, dtype=np.float64).ravel()
    if v.size < 7:
        v = np.pad(v, (0, 7 - int(v.size)))
    pos = v[:3].astype(np.float32)
    w, x, y, z = (float(v[3]), float(v[4]), float(v[5]), float(v[6]))
    quat_xyzw = np.array([x, y, z, w], dtype=np.float64)
    n = float(np.linalg.norm(quat_xyzw))
    if n > 1e-9:
        quat_xyzw /= n
    euler = Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=False).astype(np.float32)
    return pos, euler


def _glyph_robot_state_14d(obs: dict) -> list[float]:
    """14D vector sent to GlyphAgent as ``robot_state``.

    Piper (``tcp_pose`` present, 14D): arm1 from tcp_pose[0:7], arm2 from tcp_pose[7:14];
    each arm is pos3+euler3+grip with grippers from joint ``observation/state`` indices 6 and 13.
    If tcp_pose length != 14, log error and fall back to joint-space 14D.
    Otherwise: joint-space 14D from ``_extract_robot_state_14d``.
    """
    raw_tcp = _raw_tcp_pose_vector(obs)
    if raw_tcp is None:
        return _extract_robot_state_14d(obs)

    if raw_tcp.size != TCP_POSE_DIM:
        logger.error(
            "observation/tcp_pose must be %dD (dual-arm xyz+wxyz); got %d, "
            "falling back to observation/state",
            TCP_POSE_DIM,
            raw_tcp.size,
        )
        return _extract_robot_state_14d(obs)

    joint = np.asarray(_extract_robot_state_14d(obs), dtype=np.float32).ravel()
    if joint.size < 14:
        joint = np.pad(joint, (0, 14 - int(joint.size)))
    g_left = float(joint[6])
    g_right = float(joint[13])

    pos1, euler1 = _tcp_wxyz_to_pos_euler_xyz(raw_tcp[0:TCP_POSE_PER_ARM])
    pos2, euler2 = _tcp_wxyz_to_pos_euler_xyz(raw_tcp[TCP_POSE_PER_ARM:TCP_POSE_DIM])
    left7 = np.concatenate([pos1, euler1, np.array([g_left], dtype=np.float32)])
    right7 = np.concatenate([pos2, euler2, np.array([g_right], dtype=np.float32)])
    return np.concatenate([left7, right7]).astype(np.float32).tolist()


def _extract_effort_14d(obs: dict) -> Optional[list[float]]:
    """14D joint effort (torque/current) if the robot reports it.

    The stage classifier was trained on ``state(14) ++ effort(14)``; forwarding
    effort to the VLM service materially improves contact-stage prediction.
    Returns ``None`` when unavailable so the server can zero-pad.
    """
    raw = obs.get("observation/effort")
    if raw is None:
        sd = obs.get("state", obs)
        if isinstance(sd, dict):
            raw = sd.get("effort")
    if raw is None:
        return None
    effort = _decode_array(raw).astype(np.float32).ravel()[:14]
    if effort.size < 14:
        effort = np.pad(effort, (0, 14 - int(effort.size)))
    return effort.tolist()


def _is_openpi_native(obs: dict) -> bool:
    """Detect whether obs uses openpi native format (observation/* keys)."""
    return "observation/global" in obs or "observation/state" in obs


def _smooth_actions(
    actions: np.ndarray,
    interp_factor: int = 3,
    downsample_factor: int = 2,
    ema_alpha: float = 0.4,
) -> np.ndarray:
    N, D = actions.shape
    if N < 2:
        return actions

    old_t = np.arange(N)
    new_n = (N - 1) * interp_factor + 1
    new_t = np.linspace(0, N - 1, new_n)
    upsampled = np.column_stack(
        [np.interp(new_t, old_t, actions[:, d]) for d in range(D)]
    ).astype(np.float32)

    ema = EmaFilter(ema_alpha)
    filtered = np.empty_like(upsampled)
    for i in range(len(upsampled)):
        filtered[i] = ema.run(upsampled[i])

    return filtered[::downsample_factor].copy()


def _truncate_actions_14d(
    actions_14d: np.ndarray, end_ratio: float, z_comp: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Truncate (N, 14) actions; optional z compensation when gripper > 0.5."""
    actions_14d = np.asarray(actions_14d, dtype=np.float32)
    if actions_14d.ndim == 1:
        actions_14d = actions_14d.reshape(1, -1)
    if actions_14d.shape[1] < 14:
        actions_14d = np.pad(actions_14d, ((0, 0), (0, 14 - actions_14d.shape[1])))
    elif actions_14d.shape[1] > 14:
        actions_14d = actions_14d[:, :14]

    end_idx = max(2, int(end_ratio * actions_14d.shape[0]))
    a = np.ascontiguousarray(actions_14d[:end_idx])

    left_7d = a[:, :7]
    right_7d = a[:, 7:14]

    if z_comp != 0.0:
        for i in range(len(left_7d)):
            if left_7d[i, 6] > 0.5:
                left_7d[i, 2] -= z_comp
        for i in range(len(right_7d)):
            if right_7d[i, 6] > 0.5:
                right_7d[i, 2] -= z_comp

    return left_7d, right_7d


def _openpi_to_client(result: dict, end_ratio: float, z_comp: float = 0.0) -> dict:
    """Legacy format: returns {follow1_pos, follow2_pos} lists."""
    actions = np.asarray(result["actions"], dtype=np.float32)
    left_7d, right_7d = _truncate_actions_14d(actions, end_ratio, z_comp)
    return {
        "follow1_pos": left_7d.tolist(),
        "follow2_pos": right_7d.tolist(),
    }


def _openpi_to_native(result: dict, z_comp: float = 0.0) -> dict:
    """OpenPI native: full action chunk (N, 14), same contract as WebsocketPolicyServer.

    Must not truncate along time: ``ActionChunkBroker`` uses metadata ``action_horizon``
    and indexes ``actions[cur_step]`` until cur_step reaches that horizon.
    """
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.shape[1] < 14:
        actions = np.pad(actions, ((0, 0), (0, 14 - actions.shape[1])))
    elif actions.shape[1] > 14:
        actions = actions[:, :14]

    if z_comp != 0.0:
        actions = np.array(actions, copy=True)
        for i in range(len(actions)):
            if actions[i, 6] > 0.5:
                actions[i, 2] -= z_comp
            if actions[i, 13] > 0.5:
                actions[i, 9] -= z_comp

    return {"actions": actions}


# ==============================================================================
# VLM HTTP Client
# ==============================================================================

class VLMClient:
    """Thin HTTP client — calls the VLM service's ``POST /step`` endpoint."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self.last_semantic_grounding: dict | None = None
        self.last_frame_id: int | None = None
        self.last_semantic_intent: str | None = None

    def _encode_image(self, image: np.ndarray) -> bytes:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("Failed to encode image to JPEG")
        return buf.tobytes()

    def jpeg_bytes_for_vlm_upload(self, image: np.ndarray) -> bytes:
        """JPEG bytes identical to the multipart ``image`` file in ``reset`` / ``step``."""
        return self._encode_image(image)

    def _decode_rendered(self, rendered_b64: str) -> np.ndarray:
        rendered_bytes = base64.b64decode(rendered_b64)
        buf = np.frombuffer(rendered_bytes, dtype=np.uint8)
        rendered_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)

    _CAM_FIELDS = (
        ("global", "global_image"),
        ("left_wrist", "left_wrist_image"),
        ("right_wrist", "right_wrist_image"),
    )

    def process(
        self,
        images: dict[str, np.ndarray] | np.ndarray,
        robot_state: list | None = None,
        effort: list | None = None,
        low_dim: list | None = None,
        prompt: str | None = None,
        frame_id: int | None = None,
        model_input_resolution: tuple[int, int] | None = None,
    ) -> dict[str, np.ndarray]:
        """Send the 3 camera frames + state/effort → per-camera renders.

        The VLM service classifies the stage (using the trained stage router),
        routes to the active camera, grounds + renders it, and returns rendered
        images per camera. ``global`` is always present.

        ``robot_state`` (TCP pose, 14) feeds the keyframe rule; ``low_dim``
        (joint state ++ effort) is the stage classifier's input — the stage net
        was trained on joint space, so it takes precedence over ``robot_state``
        for stage prediction. ``effort`` is kept for backward compatibility.

        ``prompt`` is forwarded when provided so the ReasoningAgent API can
        accept it, but the service does not apply system-level prompt rewriting
        on ``/step`` (episode plan is set at ``/reset``).

        Accepts a single ndarray (treated as the global frame) for compatibility.
        Returns ``{camera: rendered_rgb}`` for every camera that was sent.
        """
        if isinstance(images, np.ndarray):
            images = {"global": images}

        files = {
            field: (f"{camera}.jpg", self._encode_image(images[camera]), "image/jpeg")
            for camera, field in self._CAM_FIELDS
            if images.get(camera) is not None
        }
        data = {}
        if robot_state is not None:
            data["robot_state"] = json.dumps(list(robot_state))
        if effort is not None:
            data["effort"] = json.dumps(list(effort))
        if low_dim is not None:
            data["low_dim"] = json.dumps(list(low_dim))
        if prompt:
            data["prompt"] = prompt
        if frame_id is not None:
            data["frame_id"] = str(frame_id)
        if model_input_resolution is not None:
            data["model_input_resolution"] = json.dumps(list(model_input_resolution))

        resp = self._session.post(
            f"{self.base_url}/step", files=files, data=data,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        result = resp.json()
        self.last_semantic_grounding = result.get("semantic_grounding")
        self.last_frame_id = result.get("frame_id")
        self.last_semantic_intent = result.get("semantic_intent_injection")

        rendered_images = result.get("rendered_images")
        if rendered_images:
            return {camera: self._decode_rendered(b64) for camera, b64 in rendered_images.items()}
        return {"global": self._decode_rendered(result["rendered_image"])}

    def reset(
        self,
        image: np.ndarray | None = None,
        robot_state: list | None = None,
        prompt: str | None = None,
    ):
        """Reset VLM session after the first frame is available.

        ``prompt`` becomes the episode TaskContext instruction on the
        ReasoningAgent side (e.g. peg-in-hole color/shape selection). Scene
        handlers may parse it; there is no system-wide prompt pipeline.
        """
        if image is None or robot_state is None:
            return
        files = {"image": ("frame.jpg", self._encode_image(image), "image/jpeg")}
        data = {"robot_state": json.dumps(robot_state)}
        if prompt:
            data["prompt"] = prompt
        self._session.post(
            f"{self.base_url}/reset", files=files, data=data, timeout=self.timeout,
        ).raise_for_status()

    def health(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}/health", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False


# ==============================================================================
# Args
# ==============================================================================

@dataclasses.dataclass
class Args:
    config: str = ""
    """Training config name."""

    checkpoint_dir: str = ""
    """Path to checkpoint directory."""

    port: int = 8001
    host: str = "0.0.0.0"
    default_prompt: Optional[str] = ""
    """Fallback prompt (only if client sends nothing)."""
    action_end_ratio_min: float = 0.3
    action_end_ratio_max: float = 0.7
    gripper_z_compensation: float = 0.002
    smooth_actions: bool = True
    interp_factor: int = 3
    downsample_factor: int = 2
    ema_alpha: float = 0.6
    device: str = "cuda"

    hint_url: str = "http://localhost:8000"
    """Base URL of the HINT FastAPI service."""
    agent_prompt: str = "pick up the black block and place it on the board"
    """Fallback episode prompt when the robot client does not send one.

    Prefer the client's ``instruction`` / ``prompt`` / ``task`` field; that value
    is forwarded to ReasoningAgent ``/reset`` (and accepted on ``/step``) and to
    pi05. Only when the client omits a usable prompt is this fallback used.
    For peg-in-hole, include block color and peg shape in the client prompt (or
    rely on ReasoningAgent YAML defaults when the prompt is incomplete).
    """
    vlm_timeout: float = 30.0
    """HTTP timeout for VLM calls (seconds)."""


# ==============================================================================
# Server
# ==============================================================================

def main(args: Args):
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                "all_proxy", "ALL_PROXY"):
        os.environ.pop(var, None)
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

    # ── Load pi05 policy ──
    print(f"[1/3] Loading policy: {args.config} ...", flush=True)
    t0 = time.time()
    cfg = _train_config.get_config(args.config)
    attention_enabled = bool(getattr(cfg.model, "attention_injection", False))
    policy = _policy_config.create_trained_policy(
        cfg, args.checkpoint_dir, default_prompt=args.default_prompt,
    )
    print(f"[2/3] Model loaded ({time.time() - t0:.1f}s).", flush=True)

    print("  Warming up …", flush=True)
    t0 = time.time()
    dummy = {
        "observation/state": np.zeros(14, dtype=np.float32),
        "observation/global": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/left_wrist": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/right_wrist": np.zeros((224, 224, 3), dtype=np.uint8),
        "prompt": args.agent_prompt or args.default_prompt or "do the task",
    }
    try:
        policy.infer(dummy)
    except Exception:
        pass
    print(f"  Warmup done ({time.time() - t0:.1f}s)", flush=True)

    # ── Init HINT client ──
    print(f"[3/3] HINT service: {args.hint_url}", flush=True)
    vlm_client = VLMClient(args.hint_url, timeout=args.vlm_timeout)

    infer_lock = threading.Lock()
    metadata = {
        "model_type": "agent_policy",
        "state_dim": 14,
        "action_dim": 14,
        "action_horizon": cfg.model.action_horizon,
        "action_end_ratio": f"{args.action_end_ratio_min}~{args.action_end_ratio_max}",
        "smooth_actions": args.smooth_actions,
        "hint_url": args.hint_url,
        "agent_prompt": args.agent_prompt,
        "attention_injection": attention_enabled,
    }

    # ── WebSocket handler ──
    def handler(ws):
        client = ws.remote_address
        print(f"[CONNECT] {client}", flush=True)
        packer = msgpack_numpy.Packer()
        ws.send(packer.pack(metadata))

        glyph_initialized = False
        frame_count = 0
        native_mode = None  # auto-detected on first frame

        try:
            while True:
                raw = ws.recv()
                if isinstance(raw, str):
                    continue
                obs = msgpack_numpy.unpackb(raw)
                obs = _squeeze_obs_for_policy(obs)

                # ── Auto-detect client protocol on first frame ──
                if native_mode is None:
                    native_mode = _is_openpi_native(obs)
                    print(f"  [protocol] {'openpi-native' if native_mode else 'legacy-xrobot'}", flush=True)

                # ── Extract the 3 camera images (global + both wrists) ──
                if native_mode:
                    raw_front = obs.get("observation/global")
                    raw_left = obs.get("observation/left_wrist")
                    raw_right = obs.get("observation/right_wrist")
                else:
                    views = obs.get("views", obs)
                    raw_front = views.get("camera_front")
                    raw_left = views.get("camera_left")
                    raw_right = views.get("camera_right")
                if raw_front is None:
                    continue
                # Always send 640x480 to ReasoningAgent (grounding / tracking source size).
                # RA returns 224x224 RGB via task.output_resolution; attention maps are
                # built from the 640x480 source into model_input_resolution (224x224).
                front_image = _resize_for_vlm(_decode_image(raw_front))
                zeros_img = np.zeros((VLM_IMAGE_H, VLM_IMAGE_W, 3), dtype=np.uint8)
                left_image = (
                    _resize_for_vlm(_decode_image(raw_left)) if raw_left is not None else zeros_img
                )
                right_image = (
                    _resize_for_vlm(_decode_image(raw_right)) if raw_right is not None else zeros_img
                )
                frame_count += 1

                # ── 14D joint state for pi05; GlyphAgent may use TCP-packed 14D when Piper ──
                joint_state_14d = np.asarray(_extract_robot_state_14d(obs), dtype=np.float32)
                glyph_robot_state = _glyph_robot_state_14d(obs)
                glyph_effort = _extract_effort_14d(obs)

                # The stage classifier was trained on JOINT-space state ++ effort (the
                # dataset 'state' column is joint_space; tcp_pose is a separate column used
                # only by the keyframe rule). Send the joint low_dim explicitly so the stage
                # predictor sees joints, while robot_state (TCP pose) still feeds the
                # keyframe rule on the VLM side.
                stage_low_dim = joint_state_14d.tolist()
                if glyph_effort is not None:
                    stage_low_dim = stage_low_dim + list(glyph_effort)

                # Prefer the robot client's episode prompt; fall back to --agent-prompt.
                episode_prompt = _client_prompt(obs, args.agent_prompt)

                # ── GlyphAgent: first frame reset, every frame step ──
                # The VLM service runs stage prediction → routing → render and
                # returns rendered images per camera (active stage camera painted).
                cam_images = {
                    "global": front_image,
                    "left_wrist": left_image,
                    "right_wrist": right_image,
                }
                try:
                    if not glyph_initialized:
                        vlm_client.reset(
                            front_image, glyph_robot_state, prompt=episode_prompt or None,
                        )
                        glyph_initialized = True
                        print(f"  [reset prompt] {episode_prompt!r}", flush=True)
                    rendered = vlm_client.process(
                        cam_images,
                        glyph_robot_state,
                        glyph_effort,
                        low_dim=stage_low_dim,
                        prompt=episode_prompt or None,
                        frame_id=frame_count,
                        model_input_resolution=_model.IMAGE_RESOLUTION,
                    )
                except Exception as e:
                    print(f"  [VLM ERROR] {e}", flush=True)
                    rendered = {}
                    vlm_client.last_semantic_grounding = None
                    vlm_client.last_frame_id = None

                rendered_global = rendered.get("global")
                rendered_left = rendered.get("left_wrist")
                rendered_right = rendered.get("right_wrist")
                # Highlighting / both: use RA's 224x224 overlays. Attention alone:
                # RA returns the frames unpainted, so resizing the 640x480 source
                # locally is equivalent and skips the JPEG round-trip. Either way the
                # geometry is a direct resize, matching the attention maps.
                use_rendered = (
                    not attention_enabled
                    or vlm_client.last_semantic_intent in _PAINTED_SEMANTIC_INTENTS
                )
                policy_global = _policy_image(rendered_global, front_image, use_rendered)
                policy_left = _policy_image(rendered_left, left_image, use_rendered)
                policy_right = _policy_image(rendered_right, right_image, use_rendered)

                # ── Build openpi observation for pi05 ──
                if native_mode:
                    openpi_obs = {
                        "observation/global": policy_global,
                        "observation/left_wrist": policy_left,
                        "observation/right_wrist": policy_right,
                        "observation/state": joint_state_14d,
                        "prompt": episode_prompt,
                    }
                else:
                    views["camera_front"] = policy_global
                    views["camera_left"] = policy_left
                    views["camera_right"] = policy_right
                    openpi_obs, _ = _client_to_openpi(obs, episode_prompt)
                    openpi_obs["prompt"] = episode_prompt

                if attention_enabled:
                    semantic_grounding = (
                        vlm_client.last_semantic_grounding
                        if vlm_client.last_frame_id == frame_count
                        else None
                    )
                    openpi_obs["semantic_grounding"] = semantic_grounding or {}

                print(f"  [frame={frame_count}]", flush=True)

                # ── pi05 inference ──
                with infer_lock:
                    t_infer = time.time()
                    result = policy.infer(openpi_obs)
                    dt = time.time() - t_infer

                actions = np.asarray(
                    result.get("actions", np.zeros((1, 14))), dtype=np.float32,
                )
                raw_shape = actions.shape
                if args.smooth_actions and actions.ndim == 2 and actions.shape[0] >= 2:
                    actions = _smooth_actions(
                        actions, args.interp_factor,
                        args.downsample_factor, args.ema_alpha,
                    )
                    result["actions"] = actions
                print(
                    f"  infer: {dt * 1000:.0f}ms  "
                    f"actions={raw_shape}"
                    f"{'→' + str(actions.shape) if args.smooth_actions else ''}",
                    flush=True,
                )

                if native_mode:
                    output = _openpi_to_native(result, args.gripper_z_compensation)
                else:
                    end_ratio = np.random.uniform(
                        args.action_end_ratio_min, args.action_end_ratio_max,
                    )
                    output = _openpi_to_client(
                        result, end_ratio, args.gripper_z_compensation,
                    )
                ws.send(packer.pack(output))

        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"  [ERROR] {e}", flush=True)
            traceback.print_exc()
            try:
                ws.send(str(e))
            except Exception:
                pass
        finally:
            print(f"[DISCONNECT] {client}  frames={frame_count}", flush=True)

    # ── Start ──
    print(f"\nAGENT POLICY SERVER READY — ws://{args.host}:{args.port}", flush=True)
    print(f"  config: {args.config}", flush=True)
    print(f"  hint_url: {args.hint_url}", flush=True)
    print(
        f"  vlm_camera_size: {VLM_IMAGE_W}x{VLM_IMAGE_H} "
        f"(sent to ReasoningAgent)",
        flush=True,
    )
    print(
        f"  policy_rgb_size: {POLICY_IMAGE_W}x{POLICY_IMAGE_H} "
        f"(RA output_resolution when RA paints the overlay, else local direct-resize)",
        flush=True,
    )
    print(f"  agent_prompt: {args.agent_prompt!r}", flush=True)
    print(f"  action_end_ratio: [{args.action_end_ratio_min}, {args.action_end_ratio_max}]",
          flush=True)
    if args.smooth_actions:
        print(
            f"  smooth: interp={args.interp_factor}× → "
            f"ema(α={args.ema_alpha}) → down={args.downsample_factor}×",
            flush=True,
        )
    print(flush=True)

    server = websockets.sync.server.serve(
        handler, args.host, args.port,
        max_size=None, ping_timeout=120, ping_interval=30,
    )
    server.serve_forever()


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))
