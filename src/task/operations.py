"""Pure functions for recognition, planning, and task progress."""

from __future__ import annotations

from typing import Any

from PIL import Image

from common.vision.bbox import bbox_to_int
from common.vision.crop import crop_xyxy
from common.task_types import TargetObject, PlacementCandidate, TaskState


def recognize_task_objects(
    image: Image.Image,
    detected_objects: list[TargetObject],
    padding: int,
    *,
    task_name: str | None = None,
    config: dict[str, Any] | None = None,
    max_new_tokens: int | None = None,
) -> list[TargetObject]:
    """Recognize generic task objects from detected boxes.

    The dataclass is reused so the renderer/tracker can keep treating each
    target as a bbox-bearing block. Pass ``max_new_tokens`` to keep letter
    crops at the same 8-token cap as the old letter path.
    """
    from task import get_task_handler
    from perception.semantic_grounder.recognizer import recognize_crop_fields

    handler = get_task_handler(task_name, config)
    recognized: list[TargetObject] = []
    for obj in detected_objects:
        crop = crop_xyxy(image, obj.bbox_xyxy, padding=padding)
        fields = recognize_crop_fields(
            crop,
            task_name=handler.prompts.name,
            config=config,
            max_new_tokens=max_new_tokens,
        )
        label = fields["label"]
        category = fields.get("category") or None
        letter = label if len(label) == 1 and label.isalpha() else None
        recognized.append(
            TargetObject(
                id=obj.id,
                bbox_xyxy=list(obj.bbox_xyxy),
                confidence=obj.confidence,
                letter=letter or obj.letter,
                label=label,
                category=category,
            )
        )
    return recognized


def resolve_target_block(task_state: TaskState) -> TargetObject | None:
    return next((block for block in task_state.blocks if block.id == task_state.target_block_id), None)


def resolve_target_placement(task_state: TaskState) -> PlacementCandidate | None:
    return next(
        (placement for placement in task_state.placements if placement.id == task_state.target_placement_id),
        None,
    )


def current_target_label(task_state: TaskState) -> str:
    if 0 <= task_state.progress_idx < len(task_state.target_labels):
        return task_state.target_labels[task_state.progress_idx]
    return ""


def current_target_category(task_state: TaskState) -> str:
    if task_state.target_categories and 0 <= task_state.progress_idx < len(task_state.target_categories):
        return task_state.target_categories[task_state.progress_idx]
    return ""


def current_target_placement_label(task_state: TaskState) -> str:
    if task_state.target_placements and 0 <= task_state.progress_idx < len(task_state.target_placements):
        return task_state.target_placements[task_state.progress_idx]
    return ""


def current_visual_target(task_state: TaskState, stage_name: str | None = None) -> str:
    """What the active subtask looks at under this action pattern."""
    from task.base import action_pattern_group

    if action_pattern_group(stage_name or "") == "contact":
        placement = current_target_placement_label(task_state)
        if placement:
            return placement.replace("_", " ")
    return current_target_label(task_state)


def target_count(task_state: TaskState) -> int:
    return len(task_state.target_labels)


def is_task_complete(task_state: TaskState) -> bool:
    count = target_count(task_state)
    return count > 0 and task_state.progress_idx >= count


def plan_next_target(task_state: TaskState) -> tuple[int | None, int | None]:
    """Pick the object and placement of the subtask at the current progress index."""
    if target_count(task_state) <= 0:
        return None, None
    if task_state.progress_idx < 0 or task_state.progress_idx >= target_count(task_state):
        return None, None

    used_block_ids = set(task_state.picked_block_ids) | set(task_state.placed_block_ids)
    target_label = current_target_label(task_state).lower()
    target_category = current_target_category(task_state).lower()
    placement_label = current_target_placement_label(task_state).lower()

    candidate_blocks = [
        block
        for block in task_state.blocks
        if (block.label or block.letter or "").lower() == target_label and block.id not in used_block_ids
    ]
    if not candidate_blocks and target_category:
        candidate_blocks = [
            block
            for block in task_state.blocks
            if (block.category or "").lower() == target_category and block.id not in used_block_ids
        ]

    # Prefer the leftmost unused match so seed/global boxes stay consistent with a
    # left→right subtask order.
    def _ltr_key(block: TargetObject) -> tuple[float, int, int]:
        x1, y1, x2, _y2 = bbox_to_int(block.bbox_xyxy)
        return (0.5 * (x1 + x2), x1, y1)

    candidate_blocks.sort(key=_ltr_key)
    target_block_id = candidate_blocks[0].id if candidate_blocks else None

    labeled_placements = [
        placement
        for placement in task_state.placements
        if (placement.label or "").lower() == placement_label
    ]
    target_placement_id = labeled_placements[0].id if labeled_placements else None
    return target_block_id, target_placement_id


def summarize_target(task_state: TaskState) -> str:
    return (
        f"target_word={task_state.target_word!r} "
        f"target_labels={task_state.target_labels!r} "
        f"progress_idx={task_state.progress_idx} "
        f"target={current_target_label(task_state)!r} "
        f"target_block_id={task_state.target_block_id} "
        f"target_placement_id={task_state.target_placement_id}"
    )
