"""Register tasks from YAML specs, attaching optional per-task hooks.

Discovery order:

1. Every ``configs/tasks/*.yaml`` except ``_*`` template files.
2. If the spec sets ``hooks: <name>``, load ``HOOKS`` from
   ``task.task_hooks.<name>``. If ``hooks`` is omitted, ``task.task_hooks.<task_name>``
   is used when that module exists and exports ``HOOKS``.

YAML-only tasks (sorting) need no Python file. Letter and peg-in-hole keep a
module under ``src/task/task_hooks/`` for control flow that YAML cannot express.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from task.base import TaskHandler, TaskPrompts
from task.hooks import validate_hooks
from task.scene import define_task
from task.spec import iter_task_spec_names, load_task_spec
from task.subtask import SubtaskManager

DEFAULT_TASK_NAME = "letter"

_TASK_HANDLERS: dict[str, TaskHandler] = {}


def _load_task_hooks(task_name: str, hooks_module: str | None) -> dict[str, Any]:
    module_name = hooks_module or task_name
    try:
        module = importlib.import_module(f"task.task_hooks.{module_name}")
    except ModuleNotFoundError:
        if hooks_module is not None:
            raise
        return {}
    return validate_hooks(dict(module.HOOKS), plugin=module_name)


def _discover_handlers() -> dict[str, TaskHandler]:
    handlers: dict[str, TaskHandler] = {}
    for name in iter_task_spec_names():
        spec = load_task_spec(name)
        if spec.name in handlers:
            raise ValueError(f"duplicate task name {spec.name!r}")
        handlers[spec.name] = define_task(spec=spec, **_load_task_hooks(name, spec.plugin))
    return handlers


def reload_task_handlers() -> tuple[str, ...]:
    """Re-scan ``configs/tasks/*.yaml`` (tests / hot reload)."""

    from task.spec import clear_spec_cache

    global _TASK_HANDLERS
    clear_spec_cache()
    _TASK_HANDLERS = _discover_handlers()
    return available_tasks()


def available_tasks() -> tuple[str, ...]:
    return tuple(sorted(_TASK_HANDLERS))


def resolve_task_name(config: dict[str, Any] | None = None) -> str:
    if config:
        return str(config.get("task", {}).get("name", DEFAULT_TASK_NAME))
    return os.getenv("REASONING_AGENT_TASK", DEFAULT_TASK_NAME)


def get_task_handler(task_name: str | None = None, config: dict[str, Any] | None = None) -> TaskHandler:
    resolved = task_name or resolve_task_name(config)
    try:
        return _TASK_HANDLERS[resolved]
    except KeyError as exc:
        known = ", ".join(available_tasks()) or "(none)"
        raise KeyError(f"unknown task {resolved!r}; known: {known}") from exc


def get_task_prompts(task_name: str | None = None, config: dict[str, Any] | None = None) -> TaskPrompts:
    return get_task_handler(task_name, config).prompts


def build_subtask_manager(config: dict[str, Any] | None = None) -> SubtaskManager:
    handler = get_task_handler(config=config)
    return SubtaskManager.from_config(
        config or {},
        on_advance=handler.on_subtask_advance,
    )


_TASK_HANDLERS = _discover_handlers()
