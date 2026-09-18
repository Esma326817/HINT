from __future__ import annotations

from typing import Any

from PIL import Image

from task import get_task_handler
from perception.task_manager.qwen_prompt import generate_text
from perception.task_manager.qwen_runtime import QwenRuntime, get_qwen_runtime


def recognize_landmark_word(
    image: Image.Image,
    runtime: QwenRuntime | None = None,
    max_new_tokens: int = 32,
    *,
    task_name: str | None = None,
    config: dict[str, Any] | None = None,
) -> str:
    handler = get_task_handler(task_name, config)
    runtime = runtime or get_qwen_runtime()
    raw_text = generate_text(
        image,
        handler.prompts.landmark_recognition,
        runtime,
        max_new_tokens=max_new_tokens,
    )
    return handler.parse_landmark_output(raw_text)


def recognize_crop(
    image_crop: Image.Image,
    runtime: QwenRuntime | None = None,
    max_new_tokens: int = 8,
    *,
    task_name: str | None = None,
    config: dict[str, Any] | None = None,
) -> str:
    return recognize_crop_fields(
        image_crop,
        runtime=runtime,
        max_new_tokens=max_new_tokens,
        task_name=task_name,
        config=config,
    )["label"]


def recognize_crop_fields(
    image_crop: Image.Image,
    runtime: QwenRuntime | None = None,
    max_new_tokens: int | None = None,
    *,
    task_name: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, str]:
    handler = get_task_handler(task_name, config)
    crop_prompt = handler.prompts.crop_recognition
    if not crop_prompt:
        raise ValueError(f"task {handler.prompts.name!r} has no crop_recognition prompt")
    from task.spec import load_task_spec
    from perception.semantic_grounder.qwen_crop import _resize_small_crop

    # Tiny native crops leave too few visual tokens for toy-food identity.
    # This is task configured; letter and peg recognition retain their inputs.
    image_crop = _resize_small_crop(
        image_crop, load_task_spec(handler.prompts.name).crop_min_side
    )
    runtime = runtime or get_qwen_runtime()
    parse_crop_result = getattr(handler, "parse_crop_result", None)
    if max_new_tokens is not None:
        token_limit = int(max_new_tokens)
    elif callable(parse_crop_result) and parse_crop_result.__name__ == "parse_label_category":
        token_limit = 48
    else:
        token_limit = 8
    raw_text = generate_text(image_crop, crop_prompt, runtime, max_new_tokens=token_limit)
    if callable(parse_crop_result):
        return parse_crop_result(raw_text)
    return {"label": handler.parse_crop_output(raw_text), "category": ""}


def recognize_letter(
    image_crop: Image.Image,
    runtime: QwenRuntime | None = None,
    max_new_tokens: int = 8,
    *,
    task_name: str | None = None,
    config: dict[str, Any] | None = None,
) -> str:
    return recognize_crop(
        image_crop,
        runtime=runtime,
        max_new_tokens=max_new_tokens,
        task_name=task_name,
        config=config,
    )


def clean_landmark_word(text: str, *, task_name: str | None = None, config: dict[str, Any] | None = None) -> str:
    handler = get_task_handler(task_name, config)
    return handler.parse_landmark_output(text)


def clean_letter(text: str, *, task_name: str | None = None, config: dict[str, Any] | None = None) -> str:
    handler = get_task_handler(task_name, config)
    return handler.parse_crop_output(text)


# Backward-compatible aliases for the default letter task.
LANDMARK_RECOGNITION_PROMPT = get_task_handler("letter").prompts.landmark_recognition
LETTER_RECOGNITION_PROMPT = get_task_handler("letter").prompts.crop_recognition or ""
