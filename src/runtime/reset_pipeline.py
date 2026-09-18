"""POST /reset: initialize the active task state."""

from __future__ import annotations

import logging
from typing import Any

from PIL import Image

from common.config_loader import load_reasoning_config
from intent.drawing import draw_pick_and_place
from perception.semantic_grounder.dino import DinoClient
from runtime.render_img_output import begin_render_session
from runtime.session_state import clear_task_state, set_session_vis_output_dir, set_task_state
from task.operations import (
    current_target_label,
    resolve_target_block,
    resolve_target_placement,
)
from common.task_types import TaskContext, TaskState
from task import get_task_handler

DEFAULT_LETTER_PADDING = 4
_logger = logging.getLogger(__name__)


def _log_target_status(task_state: TaskState) -> None:
    _logger.info(
        "recognized target word: %s labels=%s | currently picking: %s",
        task_state.target_word,
        task_state.target_labels,
        current_target_label(task_state),
    )


def _save_reset_pick_and_place_visualization(
    image: Image.Image,
    task_state: TaskState,
    output_dir,
) -> None:
    try:
        vis = draw_pick_and_place(
            image=image,
            target_block=resolve_target_block(task_state),
            target_placement=resolve_target_placement(task_state),
        )
        out_path = output_dir / "reset_pick_and_place.png"
        vis.save(out_path, format="PNG")
    except Exception:
        pass


def run_reset_pipeline(
    image: Image.Image,
    robot_state: Any,
    letter_padding: int = DEFAULT_LETTER_PADDING,
    task_context: TaskContext | None = None,
) -> None:
    del robot_state

    clear_task_state()
    session_out = begin_render_session()
    set_session_vis_output_dir(session_out)

    config = load_reasoning_config()
    dino_client = DinoClient(config=config)
    handler = get_task_handler(config=config)
    task_state = handler.build_initial_state(
        image=image,
        dino_client=dino_client,
        config=config,
        recognition_padding=letter_padding,
        task_context=task_context,
    )
    set_task_state(task_state)
    _save_reset_pick_and_place_visualization(image, task_state, session_out)
    _log_target_status(task_state)
