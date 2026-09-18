"""Per-camera tracker state and mask helpers.

This module is intentionally free of SAM2 and GroundingDINO imports so both the
tracker backends and the online manager can share the same types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pattern.runtime.types import (
    ALL_CAMERAS,
    GLOBAL_CAMERA,
    LEFT_WRIST_CAMERA,
    RIGHT_WRIST_CAMERA,
    CameraName,
)
from perception.tracking.sam2_video import BBoxXYXY

WRIST_CAMERAS: tuple[CameraName, ...] = (LEFT_WRIST_CAMERA, RIGHT_WRIST_CAMERA)
AREA_PLACEMENT_RENDER_MODE = "area_placement"


def global_render_mode(config: dict[str, Any]) -> str:
    return str(config["stage_aware"]["global_render_mode"]).lower()


@dataclass
class CameraTrackerState:
    camera: CameraName
    valid: bool = False
    prompt: str | None = None
    mask: np.ndarray | None = None
    bbox_xyxy: BBoxXYXY | None = None
    score: float | None = None
    last_grounded_frame_idx: int | None = None
    lost_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def invalidate(self) -> None:
        self.valid = False
        self.mask = None
        self.bbox_xyxy = None
        self.score = None
        self.lost_count += 1


def bbox_to_mask(size: tuple[int, int], bbox_xyxy: BBoxXYXY) -> np.ndarray:
    width, height = size
    x1, y1, x2, y2 = bbox_xyxy
    left = max(0, min(width, int(round(x1))))
    top = max(0, min(height, int(round(y1))))
    right = max(0, min(width, int(round(x2))))
    bottom = max(0, min(height, int(round(y2))))
    mask = np.zeros((height, width), dtype=bool)
    if right > left and bottom > top:
        mask[top:bottom, left:right] = True
    return mask


def empty_camera_states() -> dict[CameraName, CameraTrackerState]:
    return {camera: CameraTrackerState(camera=camera) for camera in ALL_CAMERAS}
