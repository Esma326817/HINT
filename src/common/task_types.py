"""Shared task/domain types for inference, detection, and dataset_export."""

from dataclasses import dataclass, field
from typing import Any


BBox = list[float] | list[int]


@dataclass(frozen=True)
class TaskContext:
    """Episode-level task input shared by dataset rendering and live reset."""

    instruction: str
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    episode_index: int | None = None
    episode_name: str | None = None


@dataclass
class TargetObject:
    id: int
    bbox_xyxy: BBox
    confidence: float | None = None
    letter: str | None = None
    label: str | None = None
    category: str | None = None


@dataclass
class PlacementCandidate:
    id: int
    bbox_xyxy: BBox
    label: str | None = None


@dataclass
class TaskState:
    """Scene memory for one episode: the objects plus the planned subtasks."""

    task_name: str = "letter"
    task_context: TaskContext | None = None
    blocks: list[TargetObject] = field(default_factory=list)
    placements: list[PlacementCandidate] = field(default_factory=list)
    target_word: str = ""
    """Human-readable task descriptor (the spelled word); scheduling reads the lists below."""
    target_labels: list[str] = field(default_factory=list)
    """One entry per subtask, in the order they are worked through."""
    target_categories: list[str] = field(default_factory=list)
    target_placements: list[str] = field(default_factory=list)
    progress_idx: int = 0
    """Index of the active subtask."""
    target_block_id: int | None = None
    target_placement_id: int | None = None
    picked_block_ids: list[int] = field(default_factory=list)
    placed_block_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
