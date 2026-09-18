"""Stage source adapters for annotations and online predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pattern.runtime.registry import StageRegistry


class AnnotationStageSource:
    """Resolve per-frame stage payloads from annotated ``stage_id`` labels."""

    mode = "annotation"

    def __init__(self, *, column: str, registry: StageRegistry) -> None:
        self.column = column
        self.registry = registry

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AnnotationStageSource":
        registry = StageRegistry.from_config(config)
        return cls(column=registry.annotation_column, registry=registry)

    def payload_for_stage_id(self, stage_id: int | None) -> dict[str, Any] | None:
        if stage_id is None:
            return None
        return self.registry.get(int(stage_id)).to_payload()

    def read_episode_stage_ids(self, parquet_path: str | Path) -> list[int]:
        import pandas as pd

        frame = pd.read_parquet(Path(parquet_path), columns=[self.column])
        if self.column not in frame.columns:
            raise ValueError(f"missing stage column {self.column!r} in {parquet_path}")
        return [int(value) for value in frame[self.column].tolist()]

    def episode_payloads(self, parquet_path: str | Path) -> list[dict[str, Any]]:
        return [self.payload_for_stage_id(stage_id) for stage_id in self.read_episode_stage_ids(parquet_path)]

    def resolve_payload(self, *, stage_id: int | None = None, external_payload: Any = None) -> Any:
        del external_payload
        return self.payload_for_stage_id(stage_id)


class PredictStageSource:
    """Pass-through source for online use where a live classifier supplies stages."""

    mode = "predict"

    def __init__(self, registry: StageRegistry) -> None:
        self.registry = registry

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PredictStageSource":
        return cls(registry=StageRegistry.from_config(config))

    def resolve_payload(self, *, stage_id: int | None = None, external_payload: Any = None) -> Any:
        if stage_id is not None:
            return self.registry.get(stage_id).to_payload()
        return external_payload


def build_stage_source(config: dict[str, Any]) -> AnnotationStageSource | PredictStageSource:
    mode = str(config["stage_source"]["mode"]).lower()
    if mode == "annotation":
        return AnnotationStageSource.from_config(config)
    if mode == "predict":
        return PredictStageSource.from_config(config)
    raise ValueError(f"unsupported stage_source.mode: {mode!r} (expected 'annotation' or 'predict')")
