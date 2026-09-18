"""Per-task prompts, scene layout, and grounding rules.

Tasks are declared in ``configs/tasks/<name>.yaml`` and auto-registered. A
``src/task/task_hooks/<name>.py`` module is optional and only needed for control flow that
YAML cannot express.
"""

from __future__ import annotations

from task.base import TaskHandler, TaskPrompts, resolve_episode_task_context
from task.hooks import TaskModule
from task.registry import (
    DEFAULT_TASK_NAME,
    available_tasks,
    build_subtask_manager,
    get_task_handler,
    get_task_prompts,
    reload_task_handlers,
    resolve_task_name,
)
from task.scene import (
    SceneLayout,
    define_task,
    understand_scene,
)
from task.spec import clear_spec_cache, load_task_spec
from task.subtask import Subtask, SubtaskManager
from task.types import SceneObject, SubtaskPlan, SubtaskSource, SubtaskStep

__all__ = [
    "DEFAULT_TASK_NAME",
    "SceneObject",
    "SceneLayout",
    "Subtask",
    "SubtaskManager",
    "SubtaskPlan",
    "SubtaskSource",
    "SubtaskStep",
    "TaskHandler",
    "TaskModule",
    "TaskPrompts",
    "available_tasks",
    "build_subtask_manager",
    "clear_spec_cache",
    "define_task",
    "get_task_handler",
    "get_task_prompts",
    "load_task_spec",
    "reload_task_handlers",
    "resolve_episode_task_context",
    "resolve_task_name",
    "understand_scene",
]
