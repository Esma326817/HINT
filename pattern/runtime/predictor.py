"""Online manipulation-pattern classifier runtime.

Loads a trained manipulation-pattern checkpoint and predicts a
per-frame ``stage_id`` (1-6) from the live multi-camera + proprioception stream.

Online inference uses only the current frame from each camera while retaining a
12-step rolling history for state and effort. History indices are clipped at the
start of an episode exactly like the training ``StageWindowDataset``. Output
stabilization is handled downstream by ``ManipulationPatternRouter``.

``/step`` arrives at the (sparse) policy cycle, so a rolling window built here
would be far coarser than the 30 FPS window used in training. Callers that
buffer proprioception at control rate can pass ``low_dim_window`` to
``predict``; it replaces the internal buffer for that frame.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from common.step_profiler import PROFILER
from pattern.data import LOW_HISTORY, image_offsets_for_num_times, resize_and_normalize
from pattern.models import (
    ManipulationPatternRouterNet,
    build_manipulation_pattern_router_config,
)
from pattern.runtime.types import ALL_CAMERAS, CameraName

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StagePrediction:
    """One frame of classifier output, including progress when the ckpt has heads."""

    stage_id: int
    confidence: float
    progress: float | None = None
    progress_all: tuple[float, ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"stage_id": self.stage_id, "confidence": self.confidence}
        if self.progress is not None:
            payload["progress"] = self.progress
        if self.progress_all:
            payload["progress_all"] = list(self.progress_all)
        return payload


class OnlineStagePredictor:
    """Stateful per-frame manipulation-pattern classifier for live inference."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        device: str = "cuda",
        low_history: int = LOW_HISTORY,
        image_offsets: tuple[int, ...] | None = None,
        single_frame: bool = True,
    ) -> None:
        ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        ckpt_config = ckpt.get("config")
        if not isinstance(ckpt_config, dict):
            raise ValueError(f"manipulation-pattern checkpoint missing 'config': {checkpoint}")
        model_cfg = ckpt_config.get("model")
        if not isinstance(model_cfg, dict) or not model_cfg:
            raise ValueError(f"manipulation-pattern checkpoint config missing 'model': {checkpoint}")
        # Old stage-only ckpts omit progress heads; match architecture to weights
        # (model defaults would otherwise force predict_progress=True).
        model_cfg = dict(model_cfg)
        ckpt_state = ckpt["model"]
        has_progress_head = any(
            key.startswith("progress_head") or key.startswith("progress_heads")
            for key in ckpt_state
        )
        model_cfg["predict_progress"] = has_progress_head
        self.num_progress_heads = (
            int(model_cfg.get("num_progress_heads", 1)) if has_progress_head else 0
        )
        dataset_cfg = ckpt_config.get("dataset", {}) or {}
        image_encoder_cfg = model_cfg.get("image_encoder", {}) or {}
        model_num_times = int(image_encoder_cfg.get("num_times", 1))

        self.image_size = (
            int(dataset_cfg.get("image_height", 120)),
            int(dataset_cfg.get("image_width", 160)),
        )
        self.low_history = int(low_history)
        resolved_offsets = (
            image_offsets_for_num_times(model_num_times)
            if image_offsets is None
            else image_offsets
        )
        self.image_offsets = tuple(int(x) for x in resolved_offsets)
        if len(self.image_offsets) != model_num_times:
            raise ValueError(
                f"image_offsets has {len(self.image_offsets)} entries, but checkpoint "
                f"expects num_times={model_num_times}"
            )
        # For legacy four-frame checkpoints, single-frame mode fills all visual
        # slots with the current image. Low-dimensional inputs always retain their
        # real rolling history and are intentionally independent of this flag.
        self.single_frame = bool(single_frame)
        # Offsets are <= 0; we must keep enough frames to reach the furthest back.
        self._frame_window = max(self.low_history, 1 - min(self.image_offsets))

        resolved_device = str(device)
        if resolved_device.startswith("cuda") and not torch.cuda.is_available():
            _logger.warning("CUDA unavailable; stage predictor falling back to CPU")
            resolved_device = "cpu"
        self.device = torch.device(resolved_device)

        self.low_mean = np.asarray(ckpt["low_dim_mean"], dtype=np.float32).ravel()
        self.low_std = np.asarray(ckpt["low_dim_std"], dtype=np.float32).ravel()
        self.low_dim = int(self.low_mean.shape[0])

        self.model = ManipulationPatternRouterNet(
            build_manipulation_pattern_router_config(model_cfg)
        ).to(self.device)
        self.model.load_state_dict(ckpt_state)
        self.model.eval()

        self._frame_buffers: dict[CameraName, deque] = {
            camera: deque(maxlen=self._frame_window) for camera in ALL_CAMERAS
        }
        self._low_buffer: deque = deque(maxlen=self.low_history)
        self._warned_low_dim = False

        _logger.info(
            "OnlineStagePredictor ready: ckpt=%s device=%s image_size=%s low_dim=%d",
            checkpoint,
            self.device,
            self.image_size,
            self.low_dim,
        )

    def reset(self) -> None:
        for buffer in self._frame_buffers.values():
            buffer.clear()
        self._low_buffer.clear()

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        height, width = self.image_size
        return resize_and_normalize(tensor.unsqueeze(0), height, width)[0]

    def _push_images(self, images: dict[CameraName, Image.Image]) -> None:
        for camera in ALL_CAMERAS:
            image = images.get(camera)
            if image is None:
                # Missing wrist view: repeat last seen frame, else fall back to global.
                if self._frame_buffers[camera]:
                    self._frame_buffers[camera].append(self._frame_buffers[camera][-1])
                    continue
                image = images.get("global")
            self._frame_buffers[camera].append(self._preprocess(image))

    def _fit_low_dim(self, low_dim_vec: Any) -> np.ndarray:
        low = np.asarray(low_dim_vec, dtype=np.float32).ravel()
        if low.shape[0] < self.low_dim:
            if not self._warned_low_dim:
                _logger.warning(
                    "low_dim has %d values, expected %d (state+effort); zero-padding. "
                    "Pass effort to the stage predictor for best accuracy.",
                    low.shape[0],
                    self.low_dim,
                )
                self._warned_low_dim = True
            low = np.pad(low, (0, self.low_dim - low.shape[0]))
        elif low.shape[0] > self.low_dim:
            low = low[: self.low_dim]
        return low

    def _build_images(self) -> torch.Tensor:
        frames = []
        for offset in self.image_offsets:
            cameras = []
            for camera in ALL_CAMERAS:
                buffer = self._frame_buffers[camera]
                idx = len(buffer) - 1 if self.single_frame else max(0, len(buffer) - 1 + offset)
                cameras.append(buffer[idx])
            frames.append(torch.stack(cameras, dim=0))  # [cam, 3, H, W]
        # [1, T, cam, 3, H, W]
        return torch.stack(frames, dim=0).unsqueeze(0).to(self.device)

    def _buffer_low_dim(self) -> np.ndarray:
        """Window from the internal rolling buffer, left-clipped like training."""
        buffer = list(self._low_buffer)
        n = len(buffer)
        window = []
        for i in range(self.low_history):
            idx = min(max(n - self.low_history + i, 0), n - 1)
            window.append(buffer[idx])
        return np.stack(window, axis=0)  # [low_history, low_dim]

    def _low_dim_tensor(self, window: np.ndarray) -> torch.Tensor:
        low = (window - self.low_mean) / self.low_std
        return torch.from_numpy(low).float().unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(
        self,
        images: dict[CameraName, Image.Image],
        low_dim_vec: Any,
        *,
        low_dim_window: Any = None,
    ) -> StagePrediction:
        """Push the current frame and classify it.

        ``low_dim_window`` is a ``[low_history, low_dim]`` proprioception window
        (past → current) sampled at the training FPS by the sender; when given it
        is used as-is and ``low_dim_vec`` is ignored. Without it the predictor
        falls back to its own buffer, which only advances once per call.

        ``stage_id`` is 1-based to match the config / registry; the raw model
        output is fed straight into the downstream StageStabilizer for
        debouncing and progress gating, so no smoothing is applied here.
        """
        self._push_images(images)
        if low_dim_window is not None:
            window = np.asarray(low_dim_window, dtype=np.float32)
            # Keep the buffer usable if a later frame arrives without a window.
            self._low_buffer.append(window[-1])
        else:
            self._low_buffer.append(self._fit_low_dim(low_dim_vec))
            window = self._buffer_low_dim()
        low_tensor = self._low_dim_tensor(window)
        image_tensor = self._build_images()
        with PROFILER.section("stage/model", cuda=True):
            out = self.model(low_tensor, image_tensor)
        probs = out["pattern"].softmax(dim=-1)[0].detach().cpu()
        stage_idx = int(probs.argmax().item())
        progress = (
            float(out["progress"][0].detach().cpu().reshape(-1)[0].item())
            if "progress" in out
            else None
        )
        progress_all = (
            tuple(float(value) for value in out["progress_all"][0].detach().cpu().tolist())
            if "progress_all" in out
            else ()
        )
        return StagePrediction(
            stage_id=stage_idx + 1,
            confidence=float(probs[stage_idx].item()),
            progress=progress,
            progress_all=progress_all,
        )


def build_stage_predictor(config: dict[str, Any]) -> OnlineStagePredictor | None:
    """Construct a predictor from the ``stage_source.predict`` config block.

    Returns ``None`` when prediction is not configured (e.g. annotation mode, or
    no checkpoint set), so callers can fall back to externally supplied stages.
    """
    source_cfg = config.get("stage_source", {}) or {}
    if str(source_cfg.get("mode") or "annotation").lower() != "predict":
        return None
    predict_cfg = source_cfg.get("predict", {}) or {}
    checkpoint = predict_cfg.get("checkpoint")
    if not checkpoint:
        _logger.info("stage_source.mode=predict but no checkpoint set; "
                     "expecting stages supplied in the request payload")
        return None
    single_frame = predict_cfg.get("single_frame")
    return OnlineStagePredictor(
        checkpoint=checkpoint,
        device=str(predict_cfg.get("device") or "cuda"),
        single_frame=True if single_frame is None else bool(single_frame),
    )
