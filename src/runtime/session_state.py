"""Thread-safe in-memory task state for the online HTTP service."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from threading import Lock

from common.task_types import TaskState
_task_state: TaskState | None = None
_session_vis_output_dir: Path | None = None
_task_state_lock = Lock()


def clear_task_state() -> None:
    global _task_state, _session_vis_output_dir
    with _task_state_lock:
        _task_state = None
        _session_vis_output_dir = None


def get_session_vis_output_dir() -> Path | None:
    with _task_state_lock:
        return _session_vis_output_dir


def get_task_state() -> TaskState | None:
    with _task_state_lock:
        return _task_state


def set_task_state(task_state: TaskState) -> None:
    global _task_state
    with _task_state_lock:
        _task_state = task_state


def set_session_vis_output_dir(output_dir: Path) -> None:
    global _session_vis_output_dir
    with _task_state_lock:
        _session_vis_output_dir = output_dir


def dump_task_state(task_state: TaskState | None) -> str:
    if task_state is None:
        return "null"
    return json.dumps(asdict(task_state), ensure_ascii=False, indent=2)
