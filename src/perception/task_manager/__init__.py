"""Resolve the active subtask and target entity from task context.

VLM runtime and prompt helpers also live here. This package ``__init__`` is
intentionally lazy so ``qwen_prompt`` / ``qwen_runtime`` can be imported during
task-handler discovery without loading ``manager``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["resolve_target_phrase"]

_EXPORTS = {
    "resolve_target_phrase": (
        "perception.task_manager.manager",
        "resolve_target_phrase",
    ),
}


def __getattr__(name: str) -> Any:
    from importlib import import_module

    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    try:
        value = import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) in {f"{__name__}.{name}", name}:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
        raise
    globals()[name] = value
    return value
