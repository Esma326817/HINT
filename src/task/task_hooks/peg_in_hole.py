"""Peg-in-hole/color-shape task rules.

The dataset is a two-subtask task:

1. Pick one colored rectangular block and place it into the black rectangular slot.
2. Pick one colored shape peg from the right holder and insert it into the matching
   shape hole on that colored block.

Prompts, shape labels, scene objects, the two declared subtasks, and the two
robust detection strategies live in ``configs/tasks/peg_in_hole.yaml``, and both
strategies are run through ``TaskSpec.locate``. This module keeps the episode
plan: parsing it, turning it into phrases, and using it to label boxes and pick a
grounding box.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from perception.semantic_grounder.qwen_crop import normalize_compact_box
from task.base import (
    normalize_words,
    qwen_box,
    resolve_episode_task_context,
)
from task.scene import SceneLayout
from task.spec import load_task_spec

SPEC = load_task_spec("peg_in_hole")
PROMPTS = SPEC.prompts()

_PEG_DETECTION = SPEC.detection_method("shape_pegs")
_SHAPE_SPEC = SPEC.classify_spec("peg_shape")
_POSITION_SPEC = SPEC.classify_spec("block_position")
_PEG_SHAPES = frozenset(_SHAPE_SPEC.labels)
_POSITIONS = frozenset(_POSITION_SPEC.labels)
_MISSING_COLORS = frozenset({"", "unknown", "none", "null", "nil"})
_COLOR_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "selected",
        "colored",
        "one",
        "left",
        "leftmost",
        "middle",
        "center",
        "centre",
        "central",
        "right",
        "rightmost",
        "none",
        "null",
    }
)
_TRACKED_PARENT_KEY = "placed_block_tracker_parent"
_HOLE_CACHE_KEY = "placed_block_hole_cache"
_PARENT_BOX_LIMITS = {
    "max_area_ratio": 0.14,
    "max_width_ratio": 0.40,
    "max_height_ratio": 0.60,
}


def _color(value: Any, default: str) -> str:
    color = str(value or "").strip().lower()
    if color in _MISSING_COLORS or color in _COLOR_STOPWORDS:
        return default
    return color


def _color_before(text: str, suffix: str, default: str) -> str:
    match = re.search(rf"\b([a-z]+)\s+{suffix}", normalize_words(text))
    color = match.group(1) if match else ""
    return default if color in ("", * _COLOR_STOPWORDS) else color


def _parse_plan(raw: Any) -> dict[str, str]:
    if isinstance(raw, Mapping):
        return {
            "block_color": _color(
                raw.get("selected_block_color") or raw.get("block_color"), "unknown"
            ),
            "block_position": _POSITION_SPEC.normalize(
                raw.get("selected_block_position") or raw.get("block_position") or "unknown"
            ),
            "peg_color": _color(
                raw.get("selected_peg_color") or raw.get("peg_color"), "white"
            ),
            "peg_shape": _SHAPE_SPEC.normalize(
                raw.get("selected_peg_shape") or raw.get("peg_shape") or "unknown"
            ),
        }
    text = str(raw or "")
    return {
        "block_color": _color_before(text, r"(?:rectangular\s+)?block\b", "unknown"),
        "block_position": _POSITION_SPEC.find_in_text(text),
        "peg_color": _color_before(
            text, r"(?:l shaped|circular|rectangular)\s+peg\b", "white"
        ),
        "peg_shape": _SHAPE_SPEC.find_in_text(text),
    }


def _cached_hole_box(
    task_state,
    shape: str,
    image_size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    raw_boxes = task_state.metadata.get(_HOLE_CACHE_KEY)
    if not isinstance(raw_boxes, dict):
        return None
    return normalize_compact_box(
        raw_boxes.get(shape), image_size, **_PARENT_BOX_LIMITS
    )


def _store_hole_boxes(
    task_state,
    *,
    boxes: dict[str, tuple[float, float, float, float]],
) -> None:
    normalized_boxes = {
        shape: [float(value) for value in box]
        for shape, box in boxes.items()
        if shape in _PEG_SHAPES and len(box) == 4
    }
    if set(normalized_boxes) != set(_PEG_SHAPES):
        return
    task_state.metadata[_HOLE_CACHE_KEY] = normalized_boxes


def capture_tracker_parent_bbox(
    task_state,
    *,
    decision,
    tracker_state,
    image,
) -> None:
    """Keep a post-placement block parent from the outgoing global track.

    The progress and prompt checks are intentional: a green-block track from
    before placement must never seed hole detection. Only an outgoing global
    destination/slot track after subtask 0 completed is eligible.
    """
    if int(getattr(task_state, "progress_idx", 0)) != 1:
        return
    if not bool(getattr(decision, "stage_changed", False)):
        return
    if task_state.metadata.get(_HOLE_CACHE_KEY) is not None:
        return
    if not bool(getattr(tracker_state, "valid", False)):
        return
    prompt = normalize_words(getattr(tracker_state, "prompt", "") or "")
    if "slot" not in prompt and "destination" not in prompt:
        return
    box = normalize_compact_box(
        getattr(tracker_state, "bbox_xyxy", None),
        image.size,
        **_PARENT_BOX_LIMITS,
    )
    if box is None:
        return
    task_state.metadata[_TRACKED_PARENT_KEY] = [float(value) for value in box]


def _locate_matching_hole(
    image,
    target: str,
    task_state,
    config: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    """Reuse or detect the placed block, classify its holes, and take the match.

    The reset-time ``slot_bbox`` is deliberately not reused here: in this scene
    Qwen can confuse the black peg holder with the black destination base. The
    episode-selected block color is unique and remains visible after placement,
    so it is the reliable parent for the hole search.
    """
    target_shape = _SHAPE_SPEC.find_in_text(target)
    if target_shape not in _PEG_SHAPES:
        target_shape = _SHAPE_SPEC.normalize(
            task_state.metadata.get("selected_peg_shape")
        )
    if target_shape not in _PEG_SHAPES:
        return None

    cached = _cached_hole_box(task_state, target_shape, image.size)
    if cached is not None:
        return cached

    block_color = str(
        task_state.metadata.get("selected_block_color") or "unknown"
    ).strip().lower()
    parent_box = None
    tracked_parent = task_state.metadata.get(_TRACKED_PARENT_KEY)
    parent_box = normalize_compact_box(
        tracked_parent, image.size, **_PARENT_BOX_LIMITS
    )

    # A valid post-placement tracker parent skips the first Qwen bbox call.
    # If its hole search fails, retry once with Qwen to avoid making tracker
    # drift a permanent episode failure.
    if parent_box is not None:
        boxes = SPEC.locate(
            "block_holes", image, config=config, parent_box=parent_box
        )
        if target_shape in boxes:
            _store_hole_boxes(task_state, boxes=boxes)
            return boxes[target_shape]
        parent_box = None

    if block_color != "unknown":
        parent_box = qwen_box(
            image,
            (
                f"the {block_color} rectangular block containing three dark "
                "shaped holes, seated in a black destination base"
            ),
        )
    if parent_box is None:
        parent_box = qwen_box(
            image,
            (
                "the one colored rectangular block seated inside the black "
                "rectangular destination base, containing three dark shaped holes; "
                "exclude all unused colored blocks and the black peg holder"
            ),
        )
    parent_box = normalize_compact_box(parent_box, image.size, **_PARENT_BOX_LIMITS)
    if parent_box is None:
        return None
    boxes = SPEC.locate("block_holes", image, config=config, parent_box=parent_box)
    if target_shape not in boxes:
        return None
    _store_hole_boxes(task_state, boxes=boxes)
    return boxes[target_shape]


def _plan_context(plan: Mapping[str, str]) -> dict[str, str]:
    block_color = plan["block_color"]
    block_position = plan.get("block_position") or "unknown"
    peg_color = plan["peg_color"]
    peg_shape = plan["peg_shape"]
    shape_phrase = "L-shaped" if peg_shape == "l shaped" else peg_shape
    qualifiers = [
        part
        for part in (block_position, block_color)
        if part not in ("", "unknown")
    ]
    block = (
        f"the {' '.join(qualifiers)} rectangular block"
        if qualifiers
        else "the selected colored rectangular block"
    )
    slot = "the black rectangular slot"
    peg = f"the {peg_color} {shape_phrase} peg"
    hole_parent = (
        f"the {block_color} rectangular block"
        if block_color != "unknown"
        else "the rectangular block"
    )
    hole = f"the {shape_phrase} hole in {hole_parent}"
    return {
        "block_phrase": block,
        "slot_phrase": slot,
        "peg_phrase": peg,
        "hole_phrase": hole,
        "peg_shape_word": peg_shape,
        "block_locate": "rectangular_blocks" if block_position in _POSITIONS else "",
        **plan,
        "instruction": f"place {block} into {slot}, then insert {peg} into {hole}",
    }


def build_scene_layout(
    config: dict[str, Any],
    *,
    task_context: Any | None = None,
) -> SceneLayout:
    resolved_context = resolve_episode_task_context(
        config,
        task_name=PROMPTS.name,
        task_context=task_context,
        parse_plan=parse_episode_plan,
    )
    plan = _parse_plan(
        dict(getattr(resolved_context, "payload", {}) or {})
        if resolved_context is not None
        else {}
    )
    if plan["peg_shape"] == "unknown" or (
        plan["block_color"] == "unknown" and plan["block_position"] == "unknown"
    ):
        raise ValueError(
            "peg_in_hole requires selected_peg_shape and either "
            "selected_block_color or selected_block_position "
            "from TaskContext or explicit task config"
        )
    return SPEC.scene_layout(
        config,
        context=_plan_context(plan),
        task_context=resolved_context,
    )


def resolve_global_grounding_box(
    *,
    image,
    dino_client,
    config: dict[str, Any],
    task_state,
    stage_name: str,
    max_new_tokens: int = 64,
) -> tuple[float, float, float, float] | None:
    """Compatibility hook for renderers that request a task-specific global box.

    Full-image ``qwen_box("… peg")`` often returns the whole peg holder.
    Prefer the reset-time peg bbox, then crop localization for pick stages.
    """
    if int(getattr(task_state, "progress_idx", 0)) != 1:
        return None

    del dino_client, max_new_tokens

    from task.operations import current_visual_target

    target = current_visual_target(task_state, stage_name)
    if not target:
        return None

    target_l = str(target).lower()
    if "peg" in target_l:
        from task.operations import resolve_target_block

        block = resolve_target_block(task_state)
        if block is not None and block.bbox_xyxy and len(block.bbox_xyxy) == 4:
            known = tuple(float(value) for value in block.bbox_xyxy)
            if _PEG_DETECTION.box_filter(known, image.size):
                return known
        peg_shape = _SHAPE_SPEC.normalize(
            task_state.metadata.get("selected_peg_shape")
        )
        if peg_shape not in _PEG_SHAPES:
            peg_shape = _SHAPE_SPEC.find_in_text(target_l)
        if peg_shape not in _PEG_SHAPES:
            return None
        peg_color = _color(task_state.metadata.get("selected_peg_color"), "white")
        box = SPEC.locate(
            "shape_pegs",
            image,
            config=config,
            context={"peg_color": peg_color},
        ).get(peg_shape)
        return tuple(float(value) for value in box) if box is not None else None

    if "hole" in target_l:
        box = _locate_matching_hole(image, target, task_state, config)
        return tuple(float(value) for value in box) if box is not None else None

    return None


def parse_episode_plan(raw: Any) -> Any | None:
    """Turn payload/text/YAML fields into a complete peg-in-hole ``TaskContext``.

    Returns ``None`` when color or shape is still unknown so the shared
    ``resolve_episode_task_context`` can fall through to the next source.
    """
    from common.task_types import TaskContext

    plan = _parse_plan(raw)
    if plan["peg_shape"] == "unknown" or (
        plan["block_color"] == "unknown" and plan["block_position"] == "unknown"
    ):
        return None
    context = _plan_context(plan)
    payload = {
        "selected_block_color": plan["block_color"],
        "selected_block_position": plan["block_position"],
        "selected_peg_color": plan["peg_color"],
        "selected_peg_shape": plan["peg_shape"],
    }
    return TaskContext(instruction=context["instruction"], payload=payload, source="plan")


HOOKS = {
    "build_scene_layout": build_scene_layout,
    "resolve_global_grounding_box": resolve_global_grounding_box,
    "capture_tracker_parent_bbox": capture_tracker_parent_bbox,
    "parse_episode_plan": parse_episode_plan,
}


def __getattr__(name: str):
    from task.hooks import lazy_handler_attr

    return lazy_handler_attr("peg_in_hole", name)
