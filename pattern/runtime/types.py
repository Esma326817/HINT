"""Shared types for the online stage-aware runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pattern.models.progress import progress_head_for_stage

CameraName = str
StageName = str

GLOBAL_CAMERA: CameraName = "global"
LEFT_WRIST_CAMERA: CameraName = "left_wrist"
RIGHT_WRIST_CAMERA: CameraName = "right_wrist"
ALL_CAMERAS: tuple[CameraName, ...] = (GLOBAL_CAMERA, LEFT_WRIST_CAMERA, RIGHT_WRIST_CAMERA)

FREE_MOVE_STAGE: StageName = "free_move"
PRE_CONTACT_STAGE: StageName = "pre_contact"
DEXTEROUS_CONTACT_STAGE: StageName = "dexterous_contact"
TRANSPORT_CONTACT_STAGE: StageName = "transport_contact"

# Focus identifies which arm/camera the annotated stage is centered on. The
# annotation distinguishes left/right contact, which directly selects the wrist
# camera whose tracker should be enabled for that frame.
FocusName = str
FOCUS_GLOBAL: FocusName = "global"
FOCUS_LEFT: FocusName = "left"
FOCUS_RIGHT: FocusName = "right"

# Default mapping from a focused arm to the camera whose tracker it enables.
# ``global`` keeps None so routing falls back to the stage's configured cameras.
DEFAULT_FOCUS_CAMERAS: dict[FocusName, CameraName | None] = {
    FOCUS_GLOBAL: None,
    FOCUS_LEFT: LEFT_WRIST_CAMERA,
    FOCUS_RIGHT: RIGHT_WRIST_CAMERA,
}

@dataclass(frozen=True)
class StageClassifierOutput:
    """Classifier output consumed by the grounding/rendering layer."""

    stage: StageName
    stage_id: int | None = None
    phase: str | None = None
    focus: str | None = None
    confidence: float | None = None
    progress: float | None = None
    progress_all: tuple[float, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any, default_stage: StageName, default_stage_id: int = 1) -> "StageClassifierOutput":
        if payload is None:
            return cls(stage=default_stage, stage_id=default_stage_id)
        if isinstance(payload, str):
            return cls(stage=payload or default_stage, stage_id=default_stage_id)
        if isinstance(payload, dict):
            stage_id = payload.get("stage_id")
            stage = str(payload.get("stage") or default_stage)
            confidence = payload.get("confidence")
            progress = payload.get("progress")
            progress_all = payload.get("progress_all") or ()
            return cls(
                stage=stage,
                stage_id=int(stage_id) if stage_id is not None else None,
                phase=payload.get("phase"),
                focus=payload.get("focus"),
                confidence=float(confidence) if confidence is not None else None,
                progress=float(progress) if progress is not None else None,
                progress_all=tuple(float(value) for value in progress_all),
                raw=dict(payload),
            )
        raise TypeError(f"unsupported stage payload type: {type(payload).__name__}")

    @property
    def num_progress_heads(self) -> int:
        if self.progress_all:
            return len(self.progress_all)
        return 1 if self.progress is not None else 0

    @property
    def has_progress(self) -> bool:
        return self.num_progress_heads > 0

    def progress_for_stage(self, stage_id: int) -> float | None:
        """Progress of ``stage_id`` on this frame, or None when it is unobservable.

        Multi-head checkpoints score every stage each frame. A single head only
        scores the stage the classifier just predicted.
        """
        if self.progress_all:
            head = progress_head_for_stage(stage_id, len(self.progress_all))
            return None if head is None else self.progress_all[head]
        if self.progress is None or self.stage_id is None:
            return None
        return self.progress if int(stage_id) == int(self.stage_id) else None


@dataclass(frozen=True)
class MultiCameraFrame:
    """Three camera RGB payload passed into stage-aware rendering."""

    images: dict[CameraName, Any]

    def get(self, camera: CameraName) -> Any | None:
        return self.images.get(camera)

    def available_cameras(self) -> tuple[CameraName, ...]:
        return tuple(camera for camera in ALL_CAMERAS if camera in self.images and self.images[camera] is not None)


@dataclass(frozen=True)
class StageRoute:
    stage: StageName
    cameras: tuple[CameraName, ...]
    prompt: str


@dataclass(frozen=True)
class StageDecision:
    stage_id: int
    raw_stage_id: int
    raw_stage: StageName
    confirmed_stage: StageName
    stage_changed: bool
    route: StageRoute
    phase: str | None = None
    focus: str | None = None
    confidence: float | None = None
    progress: float | None = None
