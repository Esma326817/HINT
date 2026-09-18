"""Shared dataclasses for task specs, scene layout, and reset-time state.

This module is import-safe: it does not load YAML, handlers, or the scene
pipeline. ``task.spec`` parses YAML into these types; ``task.scene`` consumes
them at reset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

SceneObjectRole = Literal["movable", "placement", "landmark"]
BoxCount = Literal["all", "one"]
SubtaskSource = Literal["detected", "word_chars", "fixed"]

DEFAULT_FREE_TARGET_PHRASE = "the {label} object"
DEFAULT_CONTACT_TARGET_PHRASE = "the {placement}"


@dataclass(frozen=True)
class SceneObject:
    """One object or region to locate in the global camera at reset."""

    key: str
    role: SceneObjectRole
    phrase: str = ""
    dino_prompt: str | None = None
    boxes: BoxCount = "one"
    """``all`` keeps every matching box; ``one`` keeps a single box."""
    optional: bool = False
    """When True, reset continues if this object is not found."""
    label: str | None = None
    category: str | None = None
    max_box_area_ratio: float | None = None
    box_threshold: float | None = None
    text_threshold: float | None = None
    keep_top_k: int | None = None
    recognize_crop: bool = True
    """When True and role is movable, run ``recognition.crop`` on each box."""
    classify: str | None = None
    """Optional ``classify.<name>`` key: label each detected box with that set."""
    label_template: str | None = None
    """Format ``{label}`` from classify into a placement/movable name, e.g. ``{label}_basket``."""
    assign_labels: tuple[str, ...] = ()
    """Fixed labels assigned by spatial order when boxes look alike (e.g. bowls)."""
    assign_order: str = "left_to_right"
    """How ``assign_labels`` map onto detections: left/right/top/bottom variants."""
    locator: Any | None = None
    """Resolved robust detection strategy used instead of phrase grounding."""
    select_label: str | None = None
    """Optional label selecting one result from ``locator``."""
    fallback_bbox_from: str | None = None
    """Reuse the first box of another scene object when this object is missing."""
    missing_message: str | None = None
    """Record a reset soft failure when this object cannot be located."""


@dataclass(frozen=True)
class SubtaskStep:
    """One declared subtask of a task whose plan is known before reset."""

    label: str = ""
    category: str = ""
    placement: str = ""
    pick: str | None = None
    """Optional scene-object key used as the movable for this step."""
    place: str | None = None
    """Optional scene-object key used as the destination for this step."""


@dataclass(frozen=True)
class SubtaskPlan:
    """How reset turns one recognized scene into the subtask sequence."""

    source: SubtaskSource = "detected"
    order: str = "detected"
    """``SUBTASK_ORDERINGS`` key used when the order comes from the detections."""
    order_config_key: str = "object_sequence"
    """``task.<name>.<key>`` holding an explicit label order that wins over ``order``."""
    placement_by_category: dict[str, str] = field(default_factory=dict)
    """Destination for each recognized category; an unmapped category has no target."""
    steps: tuple[SubtaskStep, ...] = ()
    """The whole sequence, for ``source: fixed``."""


@dataclass(frozen=True)
class BuiltSubtasks:
    """Resolved per-episode subtask sequence."""

    labels: list[str]
    categories: list[str]
    placements: list[str]
    target_word: str
    source: str


@dataclass
class SceneLayout:
    """Declarative scene layout for one task's reset-time understanding."""

    name: str
    scene_objects: Sequence[SceneObject] = field(default_factory=tuple)
    exclude_movables_inside_placements: bool = True
    exclude_inside: tuple[str, ...] = ()
    """Named containers whose interior should not be re-grasped as a movable."""
    recognize_movables: bool = True
    target_word: str | None = None
    """Used when the subtask source is word_chars, or as TaskState.target_word fallback."""
    metadata: dict[str, Any] = field(default_factory=dict)
    subtasks: SubtaskPlan = field(default_factory=SubtaskPlan)

    placement_for: Callable[..., str | None] | None = None
    """``placement_for(label, category, config) -> placement label``."""

    free_target_phrase: str = DEFAULT_FREE_TARGET_PHRASE
    """Looked at while the hand is empty. Format keys: label, category, placement."""
    contact_target_phrase: str = DEFAULT_CONTACT_TARGET_PHRASE
    """Looked at while the object is held. Format keys: label, category, placement."""
    classify: dict[str, Any] = field(default_factory=dict)
    """``ClassifySpec`` table from the task YAML, keyed by classify name."""


@dataclass
class SceneAccumulator:
    """Mutable reset-time results collected by scene-object role handlers."""

    placements: list[Any] = field(default_factory=list)
    movable_detections: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
