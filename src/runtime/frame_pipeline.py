"""POST /step: pick-and-place overlay per frame (progress owned by SubtaskManager)."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from typing import Any

from PIL import Image

from intent.drawing import draw_pick_and_place
from runtime.render_img_output import save_render_images
from runtime.session_state import get_session_vis_output_dir, get_task_state, set_task_state
from task.operations import (
    current_target_label,
    is_task_complete,
    resolve_target_block,
    resolve_target_placement,
)
from common.task_types import TaskState

_logger = logging.getLogger(__name__)


@dataclass
class _FrameTracker:
    """Per-task counters and paths reset automatically when the task_state identity changes."""

    last_task_state_id: int | None = None
    frame_output_idx: int = 0
    outputs_cleared_for_task_state_id: int | None = None


_tracker = _FrameTracker()


def _log_target_status(task_state: TaskState) -> None:
    _logger.info(
        "recognized target word: %s | currently picking: %s",
        task_state.target_word,
        current_target_label(task_state),
    )


def _advance_frame_index(task_state: TaskState) -> int:
    """Increment the per-task frame counter, resetting state when a new task begins."""
    task_state_id = id(task_state)
    if _tracker.last_task_state_id != task_state_id:
        _tracker.last_task_state_id = task_state_id
        _tracker.frame_output_idx = 0
    _tracker.frame_output_idx += 1
    return _tracker.frame_output_idx


def _clear_outputs_once_for_task(task_state: TaskState) -> None:
    task_state_id = id(task_state)
    if _tracker.outputs_cleared_for_task_state_id == task_state_id:
        return
    out_dir = get_session_vis_output_dir()
    if out_dir is None:
        return
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in out_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        _tracker.outputs_cleared_for_task_state_id = task_state_id
    except Exception:
        pass


def _save_frame_render(rendered: Image.Image, frame_idx: int) -> None:
    save_render_images(rendered, frame_idx)


def _maybe_save_frame_render(rendered: Image.Image, frame_idx: int, *, save_debug: bool) -> None:
    if save_debug:
        _save_frame_render(rendered, frame_idx)


def prepare_frame(task_state: TaskState) -> tuple[int, bool]:
    """Advance the per-task frame index without rule overlay work.

    Used by the stage-aware pipeline when ``SubtaskManager`` owns progress and
    mask-based rendering replaces ``draw_pick_and_place``.
    """
    frame_idx = _advance_frame_index(task_state)
    _clear_outputs_once_for_task(task_state)
    task_complete = is_task_complete(task_state)
    if task_complete:
        _complete_task(task_state)
    return frame_idx, task_complete


def get_current_frame_index() -> int:
    return _tracker.frame_output_idx


def _complete_task(task_state: TaskState) -> None:
    task_state.target_block_id = None
    task_state.target_placement_id = None
    set_task_state(task_state)
    _log_target_status(task_state)


def run_frame_pipeline(
    image: Image.Image,
    robot_state: Any,
    *,
    save_debug: bool = True,
) -> Image.Image:
    """Render the per-frame pick-and-place overlay for the current target.

    Task progress is owned by ``SubtaskManager`` (stage transitions); this
    function only draws the current target block and placement.
    """
    del robot_state
    task_state = get_task_state()
    if task_state is None:
        raise RuntimeError("task_state is not initialized; call reset.run_reset_pipeline first")

    frame_idx = _advance_frame_index(task_state)
    _clear_outputs_once_for_task(task_state)
    if is_task_complete(task_state):
        _complete_task(task_state)
        return image

    _log_target_status(task_state)
    rendered = draw_pick_and_place(
        image=image,
        target_block=resolve_target_block(task_state),
        target_placement=resolve_target_placement(task_state),
    )
    _maybe_save_frame_render(rendered, frame_idx, save_debug=save_debug)
    return rendered
