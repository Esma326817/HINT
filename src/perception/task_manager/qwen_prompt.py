"""Generic Qwen-VL prompt execution helpers.

This module is intentionally task-agnostic: callers provide the prompt and the
expected output shape, and task-specific modules only decide which prompt to use.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from PIL import Image

from common.step_profiler import PROFILER

if TYPE_CHECKING:
    from perception.task_manager.qwen_runtime import QwenRuntime

_logger = logging.getLogger(__name__)

BBoxXYXY = tuple[float, float, float, float]
VlmOutputKind = Literal["text", "bbox", "boxes"]

_JSON_ARRAY = re.compile(r"\[.*\]", re.S)
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_FLAT_BOX_ARRAY = re.compile(
    rf"\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})"
)


@dataclass(frozen=True)
class VlmPromptResult:
    """Output from a generic VLM prompt call."""

    prompt: str
    raw_text: str
    text: str = ""
    boxes: list[BBoxXYXY] = field(default_factory=list)

    @property
    def bbox(self) -> BBoxXYXY | None:
        return self.boxes[0] if self.boxes else None


def generate_text(
    image: Image.Image,
    prompt: str,
    runtime: QwenRuntime,
    max_new_tokens: int,
) -> str:
    """Execute one task-agnostic Qwen image/text generation request."""

    from qwen_vl_utils import process_vision_info

    from perception.task_manager.qwen_runtime import qwen_generate_context

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    chat_text = runtime.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    model_inputs = runtime.processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(runtime.model.device)

    started = time.time()
    with PROFILER.section("grounding/qwen", cuda=True):
        with qwen_generate_context(runtime):
            generated_ids = runtime.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                # Grounding and recognition are measurements. Qwen checkpoints
                # ship with sampling enabled; inheriting that default changes
                # object identity/category across identical scene resets.
                do_sample=False,
            )
    input_tokens = int(model_inputs.input_ids.shape[1])
    _logger.info(
        "[qwen-timing] generate: %.0f ms (in=%d new=%d/max=%d tok)",
        (time.time() - started) * 1000.0,
        input_tokens,
        int(generated_ids.shape[1]) - input_tokens,
        max_new_tokens,
    )
    trimmed_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return runtime.processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def run_vlm_prompt(
    image: Image.Image,
    prompt: str,
    *,
    output: VlmOutputKind = "text",
    runtime: QwenRuntime | None = None,
    max_new_tokens: int | None = None,
    exclude_bbox: BBoxXYXY | None = None,
    exclude_pad: float = 6.0,
    prefer: Literal["largest", "first"] = "largest",
) -> VlmPromptResult:
    """Run a VLM prompt and parse the requested output shape.

    For ``output="bbox"`` / ``"boxes"``, the prompt should instruct the model to
    return JSON like ``[{"bbox_2d":[x1,y1,x2,y2]}]`` in Qwen's 0-1000 normalized
    coordinate space.
    """

    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("VLM prompt must be non-empty")

    if runtime is None:
        from perception.task_manager.qwen_runtime import get_qwen_runtime

        runtime = get_qwen_runtime()
    token_limit = int(max_new_tokens if max_new_tokens is not None else (64 if output != "text" else 32))
    raw_text = generate_text(image.convert("RGB"), prompt, runtime, max_new_tokens=token_limit)

    if output == "text":
        return VlmPromptResult(prompt=prompt, raw_text=raw_text, text=raw_text.strip())

    width, height = image.size
    boxes = parse_qwen_boxes(raw_text, width, height)
    if exclude_bbox is not None:
        boxes = [box for box in boxes if not is_center_inside(box, exclude_bbox, exclude_pad)]
    if output == "bbox" and boxes:
        boxes = [select_box(boxes, prefer=prefer)]
    return VlmPromptResult(prompt=prompt, raw_text=raw_text, boxes=boxes)


def ground_bbox_with_prompt(
    image: Image.Image,
    prompt: str,
    *,
    runtime: QwenRuntime | None = None,
    max_new_tokens: int = 64,
    exclude_bbox: BBoxXYXY | None = None,
    exclude_pad: float = 6.0,
    prefer: Literal["largest", "first"] = "largest",
) -> BBoxXYXY | None:
    """Return one bbox from a caller-provided grounding prompt."""

    try:
        result = run_vlm_prompt(
            image,
            prompt,
            output="bbox",
            runtime=runtime,
            max_new_tokens=max_new_tokens,
            exclude_bbox=exclude_bbox,
            exclude_pad=exclude_pad,
            prefer=prefer,
        )
    except Exception:
        _logger.exception("VLM bbox prompt failed")
        return None
    return result.bbox


def ground_boxes_with_prompt(
    image: Image.Image,
    prompt: str,
    *,
    runtime: QwenRuntime | None = None,
    max_new_tokens: int = 64,
    exclude_bbox: BBoxXYXY | None = None,
    exclude_pad: float = 6.0,
) -> list[BBoxXYXY]:
    """Return all parsed bboxes from a caller-provided grounding prompt."""

    try:
        result = run_vlm_prompt(
            image,
            prompt,
            output="boxes",
            runtime=runtime,
            max_new_tokens=max_new_tokens,
            exclude_bbox=exclude_bbox,
            exclude_pad=exclude_pad,
        )
    except Exception:
        _logger.exception("VLM boxes prompt failed")
        return []
    return list(result.boxes)


def _is_numeric_box(item: object) -> bool:
    return (
        isinstance(item, (list, tuple))
        and len(item) == 4
        and all(not isinstance(value, (dict, list, tuple)) for value in item)
    )


def parse_qwen_boxes(text: str, width: int, height: int) -> list[BBoxXYXY]:
    """Parse Qwen normalized boxes into pixel ``xyxy`` boxes.

    Accepts stock ``[{"bbox_2d":[x1,y1,x2,y2]}, ...]``, a single flat
    ``[x1,y1,x2,y2]``, or a nested flat ``[[x1,y1,x2,y2], ...]``. Closing
    brackets may be missing when generation stops at a low token limit.
    """

    match = _JSON_ARRAY.search(text)
    data = None
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raw_boxes: list[object] = []
    if isinstance(data, list):
        if _is_numeric_box(data):
            raw_boxes.append(data)
        else:
            for item in data:
                if isinstance(item, dict):
                    raw = item.get("bbox_2d") or item.get("bbox")
                    if raw and len(raw) == 4:
                        raw_boxes.append(raw)
                elif _is_numeric_box(item):
                    raw_boxes.append(item)

    # A malformed bbox_2d object must not be reinterpreted as a flat array.
    if not raw_boxes and "bbox" not in str(text or "").lower():
        raw_boxes.extend(match.groups() for match in _FLAT_BOX_ARRAY.finditer(str(text or "")))

    boxes: list[BBoxXYXY] = []
    for raw in raw_boxes:
        try:
            x1, y1, x2, y2 = (float(value) for value in raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        box = (
            x1 * width / 1000.0,
            y1 * height / 1000.0,
            x2 * width / 1000.0,
            y2 * height / 1000.0,
        )
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
    return boxes


def is_center_inside(box: BBoxXYXY, region: BBoxXYXY, pad: float = 0.0) -> bool:
    cx = 0.5 * (box[0] + box[2])
    cy = 0.5 * (box[1] + box[3])
    rx1, ry1, rx2, ry2 = region
    return (rx1 - pad) <= cx <= (rx2 + pad) and (ry1 - pad) <= cy <= (ry2 + pad)


def select_box(boxes: list[BBoxXYXY], *, prefer: Literal["largest", "first"] = "largest") -> BBoxXYXY:
    if not boxes:
        raise ValueError("cannot select from an empty box list")
    if prefer == "first":
        return boxes[0]
    return max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
