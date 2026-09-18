"""Thin SAM2 video-predictor wrapper for offline wrist-frame perception.tracking.

Grounding-DINO should decide target identity at grounding events. This module only
handles dense mask/bbox propagation between those anchors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from perception.tracking.sam2_config import SAM2Config

BBoxXYXY = tuple[float, float, float, float]


@dataclass(frozen=True)
class SAM2PropagationResult:
    frame_idx: int
    obj_id: int
    mask: np.ndarray
    bbox_xyxy: BBoxXYXY | None
    score: float | None = None


def mask_to_bbox_xyxy(mask: np.ndarray) -> BBoxXYXY | None:
    """Convert a binary mask to xyxy bbox coordinates."""
    mask_bool = np.asarray(mask).astype(bool)
    ys, xs = np.where(mask_bool)
    if xs.size == 0 or ys.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


class SAM2VideoTracker:
    """Prompt SAM2 on a grounding anchor and propagate masks through a frame dir."""

    def __init__(self, config: SAM2Config | None = None) -> None:
        self.config = config or SAM2Config()
        self.config.validate()
        self._predictor = None

    @property
    def predictor(self):
        if self._predictor is None:
            from sam2.build_sam import build_sam2_video_predictor

            self._predictor = build_sam2_video_predictor(
                self.config.model_cfg,
                str(self.config.checkpoint_path),
                device=self.config.device,
                vos_optimized=self.config.vos_optimized,
                apply_postprocessing=self.config.apply_postprocessing,
            )
        return self._predictor

    def init_state(self, frame_dir: str | Path):
        """Initialize SAM2 state from a directory of ordered jpg/jpeg frames."""
        return self.predictor.init_state(video_path=str(frame_dir))

    def add_box_anchor(
        self,
        inference_state,
        *,
        frame_idx: int,
        obj_id: int,
        bbox_xyxy: Iterable[float],
    ) -> None:
        box = np.asarray(tuple(bbox_xyxy), dtype=np.float32)
        if box.shape != (4,):
            raise ValueError(f"bbox_xyxy must contain 4 values, got shape {box.shape}")

        self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=box,
        )

    def add_mask_anchor(
        self,
        inference_state,
        *,
        frame_idx: int,
        obj_id: int,
        mask: np.ndarray,
    ) -> None:
        self.predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            mask=np.asarray(mask).astype(bool),
        )

    def propagate(self, inference_state) -> list[SAM2PropagationResult]:
        """Run SAM2 propagation and return per-frame masks and bboxes."""
        results: list[SAM2PropagationResult] = []
        for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(inference_state):
            masks = (mask_logits > 0.0).detach().cpu().numpy()
            for mask_idx, obj_id in enumerate(obj_ids):
                mask = np.squeeze(masks[mask_idx]).astype(bool)
                results.append(
                    SAM2PropagationResult(
                        frame_idx=int(frame_idx),
                        obj_id=int(obj_id),
                        mask=mask,
                        bbox_xyxy=mask_to_bbox_xyxy(mask),
                    )
                )
        return results

    def track_from_anchor_box(
        self,
        frame_dir: str | Path,
        *,
        anchor_idx: int,
        bbox_xyxy: Iterable[float],
        obj_id: int = 1,
    ) -> list[SAM2PropagationResult]:
        """Convenience path for one anchored object in one wrist-video segment."""
        inference_state = self.init_state(frame_dir)
        self.add_box_anchor(
            inference_state,
            frame_idx=anchor_idx,
            obj_id=obj_id,
            bbox_xyxy=bbox_xyxy,
        )
        return self.propagate(inference_state)
