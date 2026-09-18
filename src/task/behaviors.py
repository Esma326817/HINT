"""Reusable scene behaviors declared from task YAML.

These cover the patterns that used to live as letter/peg hooks:

* ``read_from`` — read a plan (word, label) from a crop next to a grounded object
* ``precise_ground`` — split a grounded region into precise placement slots
* ``exclude_inside`` — drop movables whose center sits in a named container
* ``render: area`` — destination is a geometric region (wired in ``task.spec``)
* ``distinguish`` / ``select_by`` — spatial labels and plan-field selection
  (wired in ``task.spec`` / ``SceneObjectSpec`` from the YAML-canonical names)
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from common.vision.bbox import filter_blocks_outside_bbox, is_bbox_center_inside


@dataclass(frozen=True)
class ReadFromSpec:
    """Read the episode plan from a crop relative to a grounded scene object."""

    relative_to: str
    region: str = "above"
    parse: str | None = None


@dataclass(frozen=True)
class PreciseGroundSpec:
    """Split one grounded region into precise placement slots."""

    of: str
    count_from: str = "word_length"
    axis: str = "x"
    max_slots: int | None = None


def bbox_for_scene_key(task_state: Any, key: str) -> list[float] | None:
    """Look up a reset/refresh bbox stored as ``<key>_bbox`` or a placement label."""
    direct = task_state.metadata.get(f"{key}_bbox")
    if _is_bbox(direct):
        return list(direct)
    for placement in task_state.placements:
        if placement.label == key and _is_bbox(placement.bbox_xyxy):
            return list(placement.bbox_xyxy)
    return None


def constrain_recognized_text(text: str, settings: Mapping[str, Any] | None) -> str:
    """Snap a recognized string onto an optional whitelist in task settings."""
    cfg = dict(settings or {})
    normalized = _letters_only(text)
    allowed = [
        candidate
        for candidate in (_letters_only(item) for item in cfg.get("allowed_target_words", []) or [])
        if candidate
    ]
    fallback = _letters_only(cfg.get("fallback_target_word") or "")
    if allowed and normalized not in set(allowed):
        return fallback if fallback in set(allowed) else allowed[0]
    return normalized or fallback


def apply_read_from(
    task_state: Any,
    *,
    spec: Any,
    image: Any,
    config: dict[str, Any] | None = None,
) -> Any:
    """Fill ``task_state.target_word`` from ``spec.read_from``."""
    read_from = spec.read_from
    if read_from is None:
        return task_state
    bbox = bbox_for_scene_key(task_state, read_from.relative_to)
    if not bbox:
        raise RuntimeError(
            f"read_from relative_to {read_from.relative_to!r} was not grounded during reset"
        )
    from common.vision.crop import crop_relative
    from perception.semantic_grounder.recognizer import recognize_landmark_word
    from task.base import task_config

    raw = recognize_landmark_word(
        crop_relative(image, bbox, read_from.region),
        task_name=spec.name,
        config=config,
    )
    settings = dict(spec.settings)
    settings.update(task_config(config or {}, spec.name))
    target_word = constrain_recognized_text(raw, settings)
    if not target_word.strip():
        raise RuntimeError("empty target word from read_from landmark recognition")
    task_state.target_word = target_word
    return task_state


def apply_precise_ground(task_state: Any, *, spec: Any, **_: Any) -> Any:
    """Replace a container box with one geometry slot per planned unit."""
    ground = spec.precise_ground
    if ground is None:
        return task_state
    bbox = bbox_for_scene_key(task_state, ground.of)
    if not bbox:
        raise RuntimeError(f"precise_ground of {ground.of!r} was not grounded during reset")

    count_from = ground.count_from
    target_word = task_state.target_word.strip()
    if count_from in {"word_length", "word"}:
        if not target_word:
            raise RuntimeError("precise_ground count_from word_length needs a target word")
        count = len(target_word)
    else:
        count = int(count_from)

    from common.vision.placement import (
        MAX_PLACEMENT_SLOTS,
        build_placements_from_cutting_board,
    )

    max_slots = int(ground.max_slots or MAX_PLACEMENT_SLOTS)
    if count > max_slots:
        raise RuntimeError(
            f"precise_ground count {count} exceeds max_slots {max_slots}"
        )

    placements = build_placements_from_cutting_board(bbox, count=count)
    task_state.placements = placements
    if target_word and count_from in {"word_length", "word"}:
        category = spec.crop_category
        task_state.target_labels = list(target_word)
        task_state.target_categories = [category] * len(target_word)
        task_state.target_placements = [placement.label or "" for placement in placements]
    else:
        task_state.target_placements = [placement.label or "" for placement in placements]
    return task_state


def exclude_movables_inside(
    movables: Sequence[Any],
    task_state: Any,
    container_keys: Sequence[str],
) -> list[Any]:
    """Drop movables whose bbox center falls inside any named container."""
    kept = list(movables)
    for key in container_keys:
        bbox = bbox_for_scene_key(task_state, key)
        if bbox is None:
            continue
        if kept and hasattr(kept[0], "bbox_xyxy"):
            kept = filter_blocks_outside_bbox(kept, bbox)
        else:
            kept = [
                det
                for det in kept
                if det.get("bbox_xyxy") is None
                or not is_bbox_center_inside(bbox, det["bbox_xyxy"])
            ]
    return kept


def refresh_movables_from_spec(
    *,
    spec: Any,
    image: Any,
    dino_client: Any,
    config: dict[str, Any],
    task_state: Any,
    recognition_padding: int,
) -> Any:
    """Re-detect movables and named containers, then apply ``exclude_inside``."""
    from task.operations import plan_next_target, recognize_task_objects
    from common.task_types import TargetObject
    from task.base import has_crop_identity, movable_detector_labels, single_letter_identity

    meta = dict(task_state.metadata)
    for key in spec.exclude_inside:
        detections = spec.detect(key, image=image, dino_client=dino_client, config=config)
        if detections:
            meta[f"{key}_bbox"] = list(detections[0]["bbox_xyxy"])
    task_state.metadata = meta

    detected: list[Any] = []
    for scene_object in spec.scene_objects:
        if scene_object.role != "movable":
            continue
        for idx, item in enumerate(
            spec.detect(scene_object.key, image=image, dino_client=dino_client, config=config)
        ):
            if item.get("bbox_xyxy") is None:
                continue
            detected.append(
                TargetObject(
                    id=int(item.get("id", idx)),
                    bbox_xyxy=list(item["bbox_xyxy"]),
                    confidence=(
                        float(item["confidence"]) if item.get("confidence") is not None else None
                    ),
                    letter=single_letter_identity(item.get("letter"))
                    or single_letter_identity(item.get("label")),
                    label=str(item.get("label") or item.get("letter") or "") or None,
                    category=str(item.get("category") or "") or None,
                )
            )
    detected = exclude_movables_inside(detected, task_state, spec.exclude_inside)
    generic_labels = movable_detector_labels(spec.scene_objects)
    if spec.recognize_movables and detected and not all(
        has_crop_identity(obj, generic_labels=generic_labels) for obj in detected
    ):
        detected = recognize_task_objects(
            image=image,
            detected_objects=detected,
            padding=recognition_padding,
            task_name=spec.name,
            config=config,
        )
    task_state.blocks = detected
    task_state.target_block_id, task_state.target_placement_id = plan_next_target(task_state)
    return task_state


def _is_bbox(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 4


def _letters_only(value: Any) -> str:
    return "".join(re.findall(r"[a-z]+", str(value or "").lower()))
