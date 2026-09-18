"""Letter spelling: the target word is read from a crop *above* the board.

Prompts, scene objects, and commitment live in ``configs/tasks/letter.yaml``.
``read_from``, ``precise_ground``, and ``exclude_inside`` are generic scene
behaviors. This module snapshots the letter inventory after the board is split
and records used block centers when a subtask completes.
"""

from __future__ import annotations

from typing import Any

from task.spec import load_task_spec

SPEC = load_task_spec("letter")
PROMPTS = SPEC.prompts()

parse_landmark_output = SPEC.landmark_output_parser()
parse_crop_output = SPEC.crop_output_parser()


def detect_cutting_board(dino_client, image, config: dict[str, Any] | None = None) -> dict[str, Any] | None:
    return SPEC.detect_one("cutting_board", image=image, dino_client=dino_client, config=config)


def detect_letter_blocks(dino_client, image, config: dict[str, Any] | None = None):
    """Re-detect the letter blocks on the table."""
    from common.task_types import TargetObject

    from task.base import single_letter_identity

    detections = SPEC.detect(
        "letter_blocks",
        image=image,
        dino_client=dino_client,
        config=config,
    )
    return [
        TargetObject(
            id=int(item.get("id", idx)),
            bbox_xyxy=list(item["bbox_xyxy"]),
            confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
            letter=single_letter_identity(item.get("letter"))
            or single_letter_identity(item.get("label")),
            label=str(item.get("label") or item.get("letter") or "") or None,
            category=str(item.get("category") or "") or None,
        )
        for idx, item in enumerate(detections)
        if item.get("bbox_xyxy") is not None
    ]


def constrain_target_word(word: str, config: dict[str, Any] | None = None) -> str:
    from task.base import task_config
    from task.behaviors import constrain_recognized_text

    settings = dict(SPEC.settings)
    settings.update(task_config(config or {}, PROMPTS.name))
    return constrain_recognized_text(word, settings)


def refine_scene(
    task_state,
    *,
    image,
    dino_client,
    config: dict[str, Any],
    recognition_padding: int,
    task_context=None,
):
    """Snapshot recognized letter instances after ``precise_ground`` splits the board."""
    del image, dino_client, config, recognition_padding, task_context
    from perception.semantic_grounder.letter_instances import snapshot_letter_inventory

    snapshot_letter_inventory(task_state)
    return task_state


def on_subtask_advance(task_state) -> None:
    """Record the used letter-block center when a spelling subtask completes."""
    from perception.semantic_grounder.letter_instances import record_used_letter_center

    record_used_letter_center(task_state)


HOOKS = {
    "refine_scene": refine_scene,
    "on_subtask_advance": on_subtask_advance,
}


def __getattr__(name: str):
    from task.hooks import lazy_handler_attr

    return lazy_handler_attr("letter", name)
