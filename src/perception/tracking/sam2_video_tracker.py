"""Offline SAM2 video-predictor tracking for pattern-aware dataset rendering.

The online tracker (``src/perception/tracking/manager.py``) re-runs the SAM2 *image*
predictor every frame (``set_image`` = a full image-encoder forward each frame)
and re-prompts with the previous box. For the offline render all frames are
available, so the SAM2 *video* predictor is a better fit: it encodes each frame
once and propagates the mask with memory attention, which both cuts compute and
gives temporally consistent masks instead of per-frame re-detection drift.

A subtask yields a contiguous "segment" of frames on one wrist
camera, grounded once by DINO at the segment start. We feed exactly those frames
to the video predictor (as a small temp JPEG dir, so only tracked frames are
decoded and indices align with our rendered frames) and propagate the box.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from perception.tracking.sam2_config import DEFAULT_SAM2_MODEL_CFG, SAM2Config
from perception.tracking.sam2_video import BBoxXYXY

_logger = logging.getLogger(__name__)


class Sam2VideoSegmentTracker:
    """Build the SAM2 video predictor once, reuse it to track per-segment."""

    def __init__(self, config: dict[str, Any]) -> None:
        import torch

        tracker_cfg = config.get("tracker", {}) or {}
        checkpoint = tracker_cfg.get("sam2_checkpoint")
        sam2_config = SAM2Config(
            model_cfg=str(tracker_cfg.get("sam2_model_cfg") or DEFAULT_SAM2_MODEL_CFG),
            checkpoint_path=Path(checkpoint) if checkpoint else SAM2Config().checkpoint_path,
            device=str(tracker_cfg.get("device") or "cuda"),
        )
        sam2_config.validate()

        from sam2.build_sam import build_sam2_video_predictor

        self.device = sam2_config.device
        self._autocast_dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.predictor = build_sam2_video_predictor(
            sam2_config.model_cfg,
            str(sam2_config.checkpoint_path),
            device=sam2_config.device,
        )

    def track_segment(
        self,
        frames: list[Image.Image],
        box_xyxy: BBoxXYXY,
    ) -> dict[int, np.ndarray]:
        """Track ``box_xyxy`` (given at local frame 0) across ``frames``.

        Returns ``{local_frame_index: boolean mask}`` for every frame the
        predictor produced a mask for.
        """
        import torch

        if not frames:
            return {}

        masks: dict[int, np.ndarray] = {}
        with tempfile.TemporaryDirectory(prefix="sam2_seg_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            for local_idx, frame in enumerate(frames):
                frame.convert("RGB").save(tmp_path / f"{local_idx:05d}.jpg", quality=95)

            with torch.inference_mode(), torch.autocast(
                device_type="cuda" if self.device.startswith("cuda") else "cpu",
                dtype=self._autocast_dtype,
            ):
                state = self.predictor.init_state(
                    video_path=str(tmp_path),
                    offload_video_to_cpu=True,
                )
                self.predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    box=np.asarray(box_xyxy, dtype=np.float32),
                )
                for out_idx, _obj_ids, mask_logits in self.predictor.propagate_in_video(state):
                    mask = (mask_logits[0] > 0.0).squeeze().detach().cpu().numpy().astype(bool)
                    masks[int(out_idx)] = mask
                self.predictor.reset_state(state)

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return masks
