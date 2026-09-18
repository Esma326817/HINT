"""Qwen3-VL native grounding: locate a specific target and return its box.

Qwen3-VL emits boxes as ``bbox_2d`` JSON in a 0-1000 normalized coordinate
space; this module parses them, scales to pixels, and can drop boxes that fall
inside an excluded region so an already-placed object is not re-grounded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal

from PIL import Image

from perception.task_manager.qwen_prompt import (
    BBoxXYXY,
    ground_bbox_with_prompt,
    ground_boxes_with_prompt,
    is_center_inside,
    parse_qwen_boxes,
)

if TYPE_CHECKING:
    from perception.task_manager.qwen_runtime import QwenRuntime


def _grounding_prompt(letter: str, *, avoid_board: bool, want_all: bool) -> str:
    if want_all:
        base = (
            f"Locate every letter block showing the letter '{letter}' "
            f"(colored cubes with the single letter '{letter}' on top). "
            "Output all bounding boxes as JSON in the form "
            '[{"bbox_2d":[x1,y1,x2,y2]}, ...] with coordinates normalized to 0-1000. '
            "If there is no such block, output []."
        )
    else:
        base = (
            f"Locate the letter block showing the letter '{letter}' "
            f"(a colored cube with the single letter '{letter}' printed on its top face). "
            "Output only its bounding box as JSON in the form "
            '[{"bbox_2d":[x1,y1,x2,y2]}] with coordinates normalized to 0-1000. '
            "If there is no such block, output []."
        )
    if avoid_board:
        base += (
            " Ignore any block that is already placed on the white cutting board; "
            "only consider blocks resting on the table."
        )
    return base


def _object_grounding_prompt(target: str, *, avoid_region: bool) -> str:
    base = (
        f"Locate exactly one visible object or region described as: '{target}'. "
        "Match the description as closely as you can. Partial views and occlusion "
        "are OK — still return a tight box around the best matching individual "
        "target. Never box its container, holder, or a group of similar objects. "
        "Output only its bounding box as JSON in the form "
        '[{"bbox_2d":[x1,y1,x2,y2]}] with coordinates normalized to 0-1000. '
        "Only output [] when nothing matching the description is visible."
    )
    if avoid_region:
        base += " Ignore any matching object whose center is already inside the destination basket."
    return base


def _flat_array_grounding_prompt(target: str, *, avoid_region: bool) -> str:
    base = f'Locate "{target}". Return only [x1,y1,x2,y2] normalized 0-1000, or [].'
    if avoid_region:
        base += " Ignore any matching object whose center is already inside the destination region."
    return base


_parse_boxes = parse_qwen_boxes
_center_inside = is_center_inside


def qwen_ground_letter(
    image: Image.Image,
    letter: str,
    *,
    runtime: QwenRuntime | None = None,
    max_new_tokens: int = 64,
    exclude_bbox: BBoxXYXY | None = None,
    exclude_pad: float = 6.0,
    output_mode: Literal["json", "flat_array"] = "json",
) -> BBoxXYXY | None:
    """Return the pixel bbox of the block showing ``letter``, or None.

    ``exclude_bbox`` (e.g. the cutting-board region) drops any candidate whose
    center lies inside it, so a letter already placed on the board is skipped.
    """
    return qwen_ground_target(
        image,
        letter,
        runtime=runtime,
        max_new_tokens=max_new_tokens,
        exclude_bbox=exclude_bbox,
        exclude_pad=exclude_pad,
        target_kind="letter_block",
        output_mode=output_mode,
    )


def qwen_ground_target(
    image: Image.Image,
    target: str,
    *,
    runtime: QwenRuntime | None = None,
    max_new_tokens: int = 64,
    exclude_bbox: BBoxXYXY | None = None,
    exclude_pad: float = 6.0,
    target_kind: str = "object",
    output_mode: Literal["json", "flat_array"] = "json",
) -> BBoxXYXY | None:
    """Return the pixel bbox of ``target``, or None."""
    target = (target or "").strip()
    if not target:
        return None
    if output_mode == "flat_array":
        prompt = _flat_array_grounding_prompt(target, avoid_region=exclude_bbox is not None)
    elif output_mode != "json":
        raise ValueError(f"unsupported Qwen grounding output_mode: {output_mode!r}")
    elif target_kind == "letter_block":
        prompt = _grounding_prompt(target, avoid_board=exclude_bbox is not None, want_all=False)
    else:
        prompt = _object_grounding_prompt(target, avoid_region=exclude_bbox is not None)
    return qwen_ground_prompt(
        image,
        prompt,
        runtime=runtime,
        max_new_tokens=max_new_tokens,
        exclude_bbox=exclude_bbox,
        exclude_pad=exclude_pad,
    )


def qwen_ground_phrase(
    image: Image.Image,
    target_phrase: str,
    *,
    runtime: QwenRuntime | None = None,
    max_new_tokens: int = 64,
    exclude_bbox: BBoxXYXY | None = None,
    exclude_pad: float = 6.0,
    output_mode: Literal["json", "flat_array"] = "json",
) -> BBoxXYXY | None:
    """Return one pixel bbox for a task-provided target phrase."""

    target_phrase = (target_phrase or "").strip()
    if not target_phrase:
        return None
    if output_mode == "flat_array":
        prompt = _flat_array_grounding_prompt(
            target_phrase,
            avoid_region=exclude_bbox is not None,
        )
    elif output_mode == "json":
        prompt = _object_grounding_prompt(
            target_phrase,
            avoid_region=exclude_bbox is not None,
        )
    else:
        raise ValueError(f"unsupported Qwen grounding output_mode: {output_mode!r}")
    return qwen_ground_prompt(
        image,
        prompt,
        runtime=runtime,
        max_new_tokens=max_new_tokens,
        exclude_bbox=exclude_bbox,
        exclude_pad=exclude_pad,
    )


def qwen_ground_letter_boxes(
    image: Image.Image,
    letter: str,
    *,
    runtime: QwenRuntime | None = None,
    max_new_tokens: int = 64,
    want_all: bool = False,
    avoid_board: bool = False,
    exclude_bbox: BBoxXYXY | None = None,
    exclude_pad: float = 6.0,
) -> list[BBoxXYXY]:
    """Return one or all pixel bboxes for blocks showing ``letter``."""

    letter = (letter or "").strip().lower()[:1]
    if not letter.isalpha():
        return []
    prompt = _grounding_prompt(letter, avoid_board=avoid_board, want_all=want_all)
    return ground_boxes_with_prompt(
        image,
        prompt,
        runtime=runtime,
        max_new_tokens=max_new_tokens,
        exclude_bbox=exclude_bbox,
        exclude_pad=exclude_pad,
    )


def qwen_ground_prompt(
    image: Image.Image,
    prompt: str,
    *,
    runtime: QwenRuntime | None = None,
    max_new_tokens: int = 64,
    exclude_bbox: BBoxXYXY | None = None,
    exclude_pad: float = 6.0,
    prefer: Literal["largest", "first"] = "largest",
) -> BBoxXYXY | None:
    """Return one pixel bbox using a caller-provided grounding prompt."""

    return ground_bbox_with_prompt(
        image,
        prompt,
        runtime=runtime,
        max_new_tokens=max_new_tokens,
        exclude_bbox=exclude_bbox,
        exclude_pad=exclude_pad,
        prefer=prefer,
    )
