"""Local-deploy background SAM2 tracking.

The VLA policy calls ``/step`` only every ~0.5 s, which is far too sparse to keep
a SAM2 box-prompt mask locked on a moving wrist target. In ``infer_mode:
background`` the ReasoningAgent therefore splits the work:

* ``/step`` (driven by the policy, carries ``robot_state``) keeps owning the slow
  decisions: stage classification and VLM grounding. When the stage changes it
  re-grounds and *seeds* the shared tracker state.
* a background thread here pulls camera frames from the dagger image bus
  (``img_url`` e.g. ``http://127.0.0.1:8767``) at camera frame-rate and runs only
  SAM2 propagation + render on the active-stage camera, caching the rendered
  frames.
* ``/step`` then returns the latest cached render immediately instead of blocking
  on SAM2/VLM.

Both paths share ``runtime.pipeline._runtime`` (the same ``TrackingManager``
and ``MaskRenderConfig``) so a box seeded by ``/step`` is visible to the loop, and
both serialize CUDA access through :data:`RUNTIME_LOCK`.
"""

from __future__ import annotations

import logging
import pickle
import threading
import time
from typing import Any

import numpy as np
import requests
import torch
from PIL import Image

from common.runtime_options import injects_highlighting, semantic_intent_injection
from runtime import pipeline as stage_pipeline
from pattern.runtime.types import ALL_CAMERAS, CameraName

_logger = logging.getLogger(__name__)

# Serializes all SAM2 / CUDA / tracker_manager access between the /step handler
# and the background tracking loop (the SAM2 image predictor is a single,
# non-reentrant CUDA model). api_server acquires this around the pipeline call.
RUNTIME_LOCK = threading.RLock()

_IMAGES_ENDPOINT = "/api/dagger/images"


class DaggerImageBusClient:
    """Pulls ``{view: RGB ndarray}`` frames from the dagger UI FastAPI bus."""

    def __init__(self, img_url: str, view_map: dict[str, str] | None = None, timeout: float = 1.0) -> None:
        self.base_url = img_url.rstrip("/")
        # bus view name -> reasoning camera name; default identity (bus already
        # uses global / left_wrist / right_wrist).
        self.view_map = dict(view_map or {})
        self.timeout = timeout
        self._session = requests.Session()

    def fetch(self) -> dict[CameraName, Image.Image] | None:
        try:
            resp = self._session.get(f"{self.base_url}{_IMAGES_ENDPOINT}", timeout=self.timeout)
            resp.raise_for_status()
            payload = pickle.loads(resp.content)
        except Exception as exc:  # network / decode errors: skip this tick
            _logger.debug("dagger image bus fetch failed: %s", exc)
            return None
        images: dict[CameraName, Image.Image] = {}
        for view, array in (payload or {}).items():
            camera = self.view_map.get(view, view)
            if camera not in ALL_CAMERAS or array is None:
                continue
            try:
                images[camera] = Image.fromarray(np.asarray(array)).convert("RGB")
            except Exception:
                continue
        return images or None


class BackgroundTracker:
    """Continuously tracks the active-stage camera off the image bus."""

    def __init__(self, config: dict[str, Any]) -> None:
        deploy_cfg = config.get("deploy", {})
        self.client = DaggerImageBusClient(
            img_url=str(deploy_cfg.get("img_url") or ""),
            view_map=deploy_cfg.get("view_map"),
            timeout=float(deploy_cfg.get("fetch_timeout", 1.0)),
        )
        self.target_fps = float(deploy_cfg.get("track_fps", 30.0))
        self._render_enabled = (
            injects_highlighting(semantic_intent_injection(config))
            and bool(config.get("render", {}).get("enabled", True))
        )

        self._state_lock = threading.Lock()
        self._active_cameras: tuple[CameraName, ...] = ()
        # Latest frames fetched (prefetch thread) and the frames last tracked.
        self._latest_frames: dict[CameraName, Image.Image] | None = None
        self._tracked_frames: dict[CameraName, Image.Image] | None = None
        self._last_track_ms: float = 0.0
        self._frames_tracked: int = 0

        self._stop = threading.Event()
        self._fetch_thread: threading.Thread | None = None
        self._track_thread: threading.Thread | None = None

    # -- called from /step --------------------------------------------------
    def notify_step(self, decision: Any) -> None:
        """Record which camera(s) the current stage routes to."""
        cameras = tuple(getattr(getattr(decision, "route", None), "cameras", ()) or ())
        with self._state_lock:
            self._active_cameras = cameras

    def get_rendered(self) -> dict[CameraName, Image.Image]:
        """Render the freshest frame on demand (called by /step).

        The returned image must *keep up with the tracker* regardless of the
        background loop rate, so this re-tracks the newest fetched frame here and
        renders that — the policy always gets the latest camera view with a mask
        computed on it. The background loop's job is to keep the SAM2 box warm so
        this final step is a small motion from the most recent box, not a 0.5 s
        jump. Rendering (PIL outline/alpha, ~30 ms) stays off the hot loop; it only
        runs when the policy actually pulls a frame (~2 Hz), within its budget.
        """
        with self._state_lock:
            frames = self._latest_frames or self._tracked_frames
            active = self._active_cameras
        runtime = stage_pipeline._runtime
        manager = getattr(runtime, "tracker_manager", None)
        render_config = getattr(runtime, "render_config", None)
        if not frames or not active or manager is None or render_config is None:
            return dict(frames) if frames else {}
        with RUNTIME_LOCK, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            manager.ground_every_step = False
            manager.update_existing(frames, active_cameras=active)
            self._tracked_frames = frames
            return stage_pipeline.render_stage_images(
                images=frames,
                active_cameras=active,
                tracker_manager=manager,
                render_config=render_config,
                render_enabled=self._render_enabled,
                dropout_probability=0.0,
            )

    def reset(self) -> None:
        with self._state_lock:
            self._active_cameras = ()
            self._latest_frames = None
            self._tracked_frames = None

    def stats(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "active_cameras": list(self._active_cameras),
                "last_track_ms": round(self._last_track_ms, 1),
                "frames_tracked": self._frames_tracked,
                "has_frame": self._tracked_frames is not None,
            }

    # -- worker threads -----------------------------------------------------
    def start(self) -> None:
        if self._track_thread is not None:
            return
        self._fetch_thread = threading.Thread(target=self._fetch_loop, name="sam2-bg-fetch", daemon=True)
        self._track_thread = threading.Thread(target=self._track_loop, name="sam2-bg-track", daemon=True)
        self._fetch_thread.start()
        self._track_thread.start()
        _logger.info(
            "[local_deploy] background tracker started (img_url=%s, track_fps=%.0f)",
            self.client.base_url,
            self.target_fps,
        )

    def stop(self) -> None:
        self._stop.set()
        for thread in (self._track_thread, self._fetch_thread):
            if thread is not None:
                thread.join(timeout=2.0)
        self._track_thread = self._fetch_thread = None

    def _fetch_loop(self) -> None:
        """Continuously pull the newest frames so the tracker never waits on HTTP."""
        # Pace slightly above the track rate: enough to always have a fresh frame
        # ready, without spinning HTTP/pickle (CPU-bound, holds the GIL) and stealing
        # cycles from the tracking thread's set_image preprocessing.
        min_period = 0.5 / self.target_fps if self.target_fps > 0 else 0.0
        while not self._stop.is_set():
            tick = time.time()
            images = self.client.fetch()
            if images:
                with self._state_lock:
                    self._latest_frames = images
            wait = max(min_period - (time.time() - tick), 0.0 if images else 0.02)
            if wait:
                self._stop.wait(wait)

    def _track_loop(self) -> None:
        min_period = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        while not self._stop.is_set():
            tick = time.time()
            with self._state_lock:
                active = self._active_cameras
                frames = self._latest_frames
            if active and frames:
                self._track_once(active, frames)
            elapsed = time.time() - tick
            if min_period > elapsed:
                self._stop.wait(min_period - elapsed)

    def _track_once(self, active: tuple[CameraName, ...], frames: dict[CameraName, Image.Image]) -> None:
        runtime = stage_pipeline._runtime
        manager = getattr(runtime, "tracker_manager", None)
        if manager is None:
            return  # runtime not loaded yet (before first /step after reset)
        t0 = time.time()
        # bf16 autocast + inference_mode is what makes SAM2 run its fast kernels
        # (~27 ms vs ~90 ms in fp32 for hiera_small). Both are thread-local, so this
        # does not affect the /step handler running on another thread.
        with RUNTIME_LOCK, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # In background mode VLM grounding only fires on stage change (in /step);
            # SAM2 only propagates cameras the current stage actually uses.
            manager.ground_every_step = False
            manager.update_existing(frames, active_cameras=active)
        with self._state_lock:
            self._tracked_frames = frames
            self._last_track_ms = (time.time() - t0) * 1000.0
            self._frames_tracked += 1
