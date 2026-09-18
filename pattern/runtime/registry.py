"""Unified stage-id table loaded from the active task configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pattern.runtime.types import (
    FOCUS_GLOBAL,
    FOCUS_LEFT,
    FOCUS_RIGHT,
    LEFT_WRIST_CAMERA,
    RIGHT_WRIST_CAMERA,
    FocusName,
    StageName,
    StageRoute,
)


def _focus_from_cameras(cameras: tuple[str, ...]) -> FocusName:
    if cameras == (LEFT_WRIST_CAMERA,):
        return FOCUS_LEFT
    if cameras == (RIGHT_WRIST_CAMERA,):
        return FOCUS_RIGHT
    return FOCUS_GLOBAL


@dataclass(frozen=True)
class StageSpec:
    stage_id: int
    name: StageName
    cameras: tuple[str, ...]
    prompt: str

    @property
    def focus(self) -> FocusName:
        return _focus_from_cameras(self.cameras)

    def to_payload(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "stage": self.name,
            "focus": self.focus,
        }

    def to_route(self) -> StageRoute:
        return StageRoute(stage=self.name, cameras=self.cameras, prompt=self.prompt)


@dataclass(frozen=True)
class StageRegistry:
    specs: dict[int, StageSpec]
    default_stage_id: int = 1
    annotation_column: str = "stage_id_gt"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "StageRegistry":
        return cls._from_unified(config["stages"])

    @classmethod
    def _from_unified(cls, stages_cfg: dict[str, Any]) -> "StageRegistry":
        default_stage_id = int(stages_cfg["default_id"])
        annotation_column = str(stages_cfg["column"])
        specs: dict[int, StageSpec] = {}
        for key, payload in stages_cfg["ids"].items():
            stage_id = int(str(key).strip())
            name = str(payload["name"])
            cameras = tuple(str(camera) for camera in payload["cameras"])
            prompt = str(payload["prompt"])
            specs[stage_id] = StageSpec(stage_id=stage_id, name=name, cameras=cameras, prompt=prompt)
        return cls(specs=specs, default_stage_id=default_stage_id, annotation_column=annotation_column)

    def get(self, stage_id: int | None) -> StageSpec:
        if stage_id is None:
            return self.specs[self.default_stage_id]
        resolved = int(stage_id)
        if resolved not in self.specs:
            known = sorted(self.specs)
            raise KeyError(f"unknown stage_id {resolved}; expected one of {known}")
        return self.specs[resolved]

    def ids_for_stages(self, stages: Any) -> frozenset[int]:
        """Every stage_id whose name appears in ``stages``."""
        wanted = {str(stage) for stage in stages}
        return frozenset(spec.stage_id for spec in self.specs.values() if spec.name in wanted)

    def ids_for_transitions(self, transitions: Any) -> frozenset[tuple[int, int]]:
        """Expand name pairs into every matching ``(from_id, to_id)`` edge."""
        edges: set[tuple[int, int]] = set()
        for item in transitions or ():
            from_stage, to_stage = item
            from_ids = self.ids_for_stages([from_stage])
            to_ids = self.ids_for_stages([to_stage])
            for from_id in from_ids:
                for to_id in to_ids:
                    edges.add((from_id, to_id))
        return frozenset(edges)

    def resolve_id(self, stage: StageName, focus: FocusName | None = None) -> int:
        focus = focus or FOCUS_GLOBAL
        for spec in self.specs.values():
            if spec.name == stage and spec.focus == focus:
                return spec.stage_id
        for spec in self.specs.values():
            if spec.name == stage:
                return spec.stage_id
        return self.default_stage_id

    @property
    def default_stage_name(self) -> StageName:
        return self.get(self.default_stage_id).name
