"""Optional Python hooks for a YAML-declared task.

A task is registered from ``configs/tasks/<name>.yaml``. Python is only needed
when YAML cannot express control flow. Put those callables in
``src/task/task_hooks/<name>.py`` as ``HOOKS`` (see ``OPTIONAL_HOOKS``).

Typical YAML-only tasks (sorting into named baskets) export nothing. Letter and
peg-in-hole keep a module under ``task_hooks`` for reset snapshots, episode-plan
parsing, and per-frame box lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from task.types import SceneLayout

OPTIONAL_HOOKS: tuple[str, ...] = (
    "parse_landmark_output",
    "parse_crop_output",
    "parse_crop_result",
    "build_scene_layout",
    "build_initial_state",
    "resolve_rule_grounding_prompt",
    "refine_scene",
    "resolve_global_grounding_box",
    "refresh_movables",
    "seed_global_before_grounding",
    "capture_tracker_parent_bbox",
    "resolve_segment_render_mode",
    "parse_episode_plan",
    "resolve_task_context",
    "on_subtask_advance",
)


def lazy_handler_attr(task_name: str, name: str):
    """Lazy ``HANDLER`` / bound methods so task modules can import during registry load."""

    if name in {"HANDLER", "build_initial_state", "resolve_segment_render_mode"}:
        from task.registry import get_task_handler

        handler = get_task_handler(task_name)
        return handler if name == "HANDLER" else getattr(handler, name)
    raise AttributeError(name)


def validate_hooks(hooks: dict[str, Any], *, plugin: str) -> dict[str, Any]:
    """Reject unknown ``HOOKS`` keys so a plugin cannot silently misspell a hook."""

    unknown = sorted(set(hooks) - set(OPTIONAL_HOOKS))
    if unknown:
        known = ", ".join(OPTIONAL_HOOKS)
        raise KeyError(
            f"task.{plugin} has unknown HOOKS {unknown}; known: {known}"
        )
    return {key: value for key, value in hooks.items() if value is not None}


@dataclass
class TaskModule:
    """Runtime handler bound from a YAML spec plus optional plugin hooks.

    Built by ``task.scene.define_task``. Looked up through ``task.get_task_handler``.
    """

    prompts: Any
    parse_landmark_output: Callable[..., str]
    parse_crop_output: Callable[..., str]
    build_initial_state: Callable[..., Any]
    resolve_rule_grounding_prompt: Callable[..., str]
    parse_crop_result: Callable[..., dict[str, str]] | None = None
    build_scene_layout: Callable[..., SceneLayout] | None = None
    refine_scene: Callable[..., Any] | None = None
    resolve_global_grounding_box: Callable[..., Any] | None = None
    refresh_movables: Callable[..., Any] | None = None
    seed_global_before_grounding: Callable[..., bool] | None = None
    capture_tracker_parent_bbox: Callable[..., Any] | None = None
    resolve_segment_render_mode: Callable[..., str] | None = None
    parse_episode_plan: Callable[..., Any] | None = None
    resolve_task_context: Callable[..., Any] | None = None
    on_subtask_advance: Callable[..., Any] | None = None
