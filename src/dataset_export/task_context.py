"""Config-driven episode task context readers for LeRobot datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from common.task_types import TaskContext


class TaskContextError(ValueError):
    """An episode's configured task context is missing or invalid."""


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _episode_name(episode_index: int) -> str:
    return f"episode_{episode_index:06d}"


@dataclass(frozen=True)
class DatasetTaskContextProvider:
    """Resolve one :class:`TaskContext` from configured dataset metadata."""

    source: str
    required: bool
    instruction_field: str
    fields: tuple[str, ...]
    by_index: Mapping[int, Mapping[str, Any]]
    source_path: Path | None = None
    task_position: int = 0

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        dataset_root: Path,
        *,
        path_override: Path | None = None,
    ) -> "DatasetTaskContextProvider":
        raw_cfg = config["dataset"]["task_context"]
        source = str(raw_cfg["source"]).strip().lower()
        required = bool(raw_cfg["required"])
        instruction_field = str(raw_cfg["instruction_field"])
        raw_fields = raw_cfg["fields"]
        fields = tuple(str(value) for value in raw_fields)
        task_position = int(raw_cfg["task_position"])

        if source == "none":
            return cls(source, required, instruction_field, fields, {})
        if source == "jsonl":
            raw_path = path_override or Path(
                str(raw_cfg["path"])
            )
            source_path = raw_path if raw_path.is_absolute() else dataset_root / raw_path
            rows = _jsonl_rows(source_path)
            by_index = {int(row["episode_index"]): row for row in rows}
            return cls(
                source,
                required,
                instruction_field,
                fields,
                by_index,
                source_path,
                task_position,
            )
        if source == "lerobot":
            episodes_path = dataset_root / "meta" / "episodes.jsonl"
            by_index = {}
            for row in _jsonl_rows(episodes_path):
                index = int(row["episode_index"])
                episode_tasks = row["tasks"]
                selected = str(episode_tasks[task_position])
                payload = {
                    "episode_index": index,
                    "episode_name": _episode_name(index),
                    instruction_field: selected,
                    "tasks": list(episode_tasks),
                }
                by_index[index] = payload
            return cls(
                source,
                required,
                instruction_field,
                fields,
                by_index,
                episodes_path,
                task_position,
            )
        raise TaskContextError(
            f"unsupported dataset.task_context.source={source!r}; "
            "expected jsonl, lerobot, or none"
        )

    def resolve(
        self,
        *,
        episode_index: int | None,
        episode_name: str,
    ) -> TaskContext | None:
        if self.source == "none":
            return None
        row = self.by_index.get(episode_index)
        if row is None:
            if self.required:
                raise TaskContextError(
                    f"missing required task context for {episode_name} "
                    f"(episode_index={episode_index})"
                )
            return None

        instruction = str(row[self.instruction_field]).strip()
        payload = {key: row[key] for key in self.fields}
        return TaskContext(
            instruction=instruction,
            payload=payload,
            source=self.source,
            episode_index=episode_index,
            episode_name=episode_name,
        )
