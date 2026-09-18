"""Resolve the target phrase committed by the active manipulation pattern."""

from __future__ import annotations

from typing import Any

from PIL import Image

from pattern.runtime.types import StageDecision
from common.task_types import TaskState
from task import get_task_handler


def resolve_target_phrase(
    *,
    decision: StageDecision,
    images: dict[str, Image.Image],
    task_state: TaskState | None,
    task_instruction: str | None,
    config: dict[str, Any],
) -> str:
    """Return the semantic target phrase for the current stage.

    By default this is config/rule selected, so the expensive VLM path is not
    called every frame. Set ``vlm.generate_prompt: true`` in YAML to ask Qwen on
    grounding events.
    """

    handler = get_task_handler(config=config)
    base_prompt = decision.route.prompt
    stored_context = getattr(task_state, "task_context", None) if task_state is not None else None
    stored_instruction = (
        str(getattr(stored_context, "instruction", "") or "").strip()
        if stored_context is not None
        else ""
    )
    effective_instruction = task_instruction or stored_instruction or None

    vlm_cfg = config.get("vlm", {})
    if not bool(vlm_cfg.get("generate_prompt", False)):
        return handler.resolve_rule_grounding_prompt(
            decision=decision,
            task_state=task_state,
            task_instruction=effective_instruction,
            base_prompt=base_prompt,
            config=config,
        )

    from perception.task_manager.qwen_prompt import generate_text
    from perception.task_manager.qwen_runtime import get_qwen_runtime

    primary_image = next((images[camera] for camera in decision.route.cameras if camera in images), None)
    if primary_image is None:
        return handler.resolve_rule_grounding_prompt(
            decision=decision,
            task_state=task_state,
            task_instruction=effective_instruction,
            base_prompt=base_prompt,
            config=config,
        )

    instruction_text = effective_instruction or (task_state.target_word if task_state is not None else "")
    generation_prompt = handler.prompts.grounding_generation
    active_target = ""
    if task_state is not None:
        from task.operations import current_target_label

        active_target = current_target_label(task_state)
    prompt = (
        f"{generation_prompt}\n"
        f"Task instruction: {instruction_text}\n"
        f"Stage: {decision.confirmed_stage}\n"
        f"Phase: {decision.phase or ''}\n"
        f"Focus: {decision.focus or ''}\n"
        f"Active target: {active_target or 'none'}\n"
        f"Default target phrase: {base_prompt}\n"
    )
    raw_text = generate_text(
        primary_image,
        prompt,
        get_qwen_runtime(),
        max_new_tokens=int(vlm_cfg.get("max_new_tokens", 48)),
    )
    cleaned = " ".join(raw_text.strip().split())
    return cleaned or handler.resolve_rule_grounding_prompt(
        decision=decision,
        task_state=task_state,
        task_instruction=effective_instruction,
        base_prompt=base_prompt,
        config=config,
    )
