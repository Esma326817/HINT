"""Reusable Qwen-VL grounding helpers for full images and image crops.

Full-image grounding often fails for small / crowded objects (shape pegs,
hole openings). These helpers can separate instances first and use isolated
crops for fine-grained recognition:

1. Ask Qwen for one or more instance boxes.
2. Crop each instance when fine-grained recognition is needed.
3. Preserve or map all boxes back to full-image pixel coordinates.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable, Sequence
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from perception.task_manager.qwen_prompt import BBoxXYXY, run_vlm_prompt

_logger = logging.getLogger(__name__)

CropXYXY = tuple[int, int, int, int]
LabelNormalizer = Callable[[Any], str]
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.S)


def qwen_max_new_tokens(config: dict[str, Any] | None, default: int = 128) -> int:
    """Read ``grounding.qwen.max_new_tokens`` from a ReasoningAgent config."""
    grounding = config.get("grounding", {}) if isinstance(config, dict) else {}
    if not isinstance(grounding, dict):
        return default
    qwen_cfg = grounding.get("qwen", {})
    if not isinstance(qwen_cfg, dict):
        return default
    return int(qwen_cfg.get("max_new_tokens", default))


def pad_xyxy_crop(
    box: Sequence[float],
    image_size: tuple[int, int],
    *,
    pad: int = 0,
) -> CropXYXY:
    """Expand ``box`` by ``pad`` pixels and clamp to the image bounds."""
    width, height = image_size
    x1, y1, x2, y2 = (float(value) for value in box)
    return (
        max(0, int(x1 - pad)),
        max(0, int(y1 - pad)),
        min(width, int(x2 + pad)),
        min(height, int(y2 + pad)),
    )


def _resize_small_crop(image: Image.Image, min_side: int) -> Image.Image:
    """Upscale a small recognition crop while preserving its aspect ratio."""
    if min_side <= 0 or min(image.size) >= min_side:
        return image
    scale = min_side / max(1, min(image.size))
    size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    return image.resize(size, Image.Resampling.BICUBIC)


def ground_labeled_instances(
    image: Image.Image,
    *,
    prompt: str,
    search_crop: CropXYXY | Sequence[int] | None = None,
    normalize_label: LabelNormalizer | None = None,
    max_new_tokens: int = 192,
    expected_labels: set[str] | frozenset[str] | None = None,
) -> dict[str, BBoxXYXY]:
    """Ground multiple labeled instances in one Qwen call.

    The prompt must request a JSON array whose objects contain ``shape``,
    ``label``, or ``name`` plus ``bbox_2d``. Returned boxes use full-image
    coordinates even when ``search_crop`` is supplied.
    """
    working_image = image
    offset_x = 0
    offset_y = 0
    if search_crop is not None:
        crop_xyxy = pad_xyxy_crop(search_crop, image.size)
        offset_x, offset_y = crop_xyxy[:2]
        working_image = image.crop(crop_xyxy)

    try:
        raw_text = run_vlm_prompt(
            working_image,
            prompt,
            output="text",
            max_new_tokens=max_new_tokens,
        ).raw_text
    except Exception:
        _logger.exception("labeled multi-instance Qwen grounding failed")
        return {}

    match = _JSON_ARRAY_RE.search(raw_text or "")
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, list):
        return {}

    labeled: dict[str, BBoxXYXY] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        raw_label = next(
            (item.get(key) for key in ("shape", "label", "name") if item.get(key) is not None),
            "",
        )
        label = (
            normalize_label(raw_label)
            if normalize_label is not None
            else str(raw_label).strip().lower()
        )
        raw_box = item.get("bbox_2d") or item.get("bbox")
        if not label or label == "unknown" or not raw_box or len(raw_box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in raw_box)
        except (TypeError, ValueError):
            continue
        box = (
            offset_x + x1 * working_image.width / 1000.0,
            offset_y + y1 * working_image.height / 1000.0,
            offset_x + x2 * working_image.width / 1000.0,
            offset_y + y2 * working_image.height / 1000.0,
        )
        if box[2] <= box[0] or box[3] <= box[1] or label in labeled:
            return {}
        labeled[label] = box

    if expected_labels is not None and set(labeled) != set(expected_labels):
        return {}
    return labeled


def ground_and_classify_instances(
    image: Image.Image,
    *,
    instance_prompt: str,
    classification_prompt: str,
    search_crop: CropXYXY | Sequence[int] | None = None,
    normalize_label: LabelNormalizer | None = None,
    instance_max_new_tokens: int = 192,
    classification_max_new_tokens: int = 16,
    min_crop_side: int = 256,
    classification_pad: int = 0,
    expected_count: int | None = None,
) -> dict[str, BBoxXYXY]:
    """Ground separate instances, then classify each tight crop with Qwen.

    This separates the two VLM decisions that are commonly entangled by direct
    text grounding: instance separation ("where are the individual objects?")
    and fine-grained recognition ("what is this one object?"). The prompts,
    labels, optional search region, object count, and label normalization remain
    task-provided. Returned boxes are always in full-image coordinates.
    """
    boxes = ground_instances(
        image,
        prompt=instance_prompt,
        search_crop=search_crop,
        max_new_tokens=instance_max_new_tokens,
        expected_count=expected_count,
    )
    if not boxes:
        return {}

    classified_instances: list[tuple[str, BBoxXYXY]] = []
    for box in boxes:
        crop = image.crop(
            pad_xyxy_crop(box, image.size, pad=max(0, int(classification_pad)))
        )
        crop = _resize_small_crop(crop, min_crop_side)
        try:
            classified = run_vlm_prompt(
                crop,
                classification_prompt,
                output="text",
                max_new_tokens=classification_max_new_tokens,
            ).text
        except Exception:
            _logger.exception("Qwen instance classification failed box=%s", box)
            continue
        _logger.info(
            "Qwen instance classification box=%s raw=%r normalized=%r",
            tuple(round(float(value), 1) for value in box),
            classified,
            normalize_label(classified) if normalize_label is not None else classified.strip().lower(),
        )
        label = normalize_label(classified) if normalize_label is not None else classified.strip().lower()
        if not label or label == "unknown":
            continue
        classified_instances.append((label, box))

    labeled = {label: box for label, box in classified_instances}
    if len(labeled) != len(classified_instances):
        duplicate_labels = sorted(
            {
                label
                for label, _box in classified_instances
                if sum(item_label == label for item_label, _ in classified_instances) > 1
            }
        )
        _logger.warning(
            "duplicate Qwen instance labels %s; rejecting ambiguous result",
            duplicate_labels,
        )
        return {}
    return labeled


def ground_and_jointly_classify_instances(
    image: Image.Image,
    *,
    instance_prompt: str,
    classification_prompt: str,
    expected_labels: Sequence[str],
    search_crop: CropXYXY | Sequence[int] | None = None,
    normalize_label: LabelNormalizer | None = None,
    instance_max_new_tokens: int = 192,
    classification_max_new_tokens: int = 96,
    panel_size: int = 384,
    classification_pad: int = 8,
    expected_count: int | None = None,
) -> dict[str, BBoxXYXY]:
    """Detect candidates, then classify all enlarged crops in one VLM call.

    Fine shape differences are easier to judge when the model can compare all
    candidates side by side. Candidate panels are created only to present model
    inputs; no pixel geometry or fixed spatial order is used for recognition.
    """
    labels = tuple(str(label) for label in expected_labels)
    count = expected_count if expected_count is not None else len(labels)
    boxes = ground_instances(
        image,
        prompt=instance_prompt,
        search_crop=search_crop,
        max_new_tokens=instance_max_new_tokens,
        expected_count=count,
    )
    if len(boxes) != count or count != len(labels):
        return {}

    size = max(128, int(panel_size))
    header = 36
    canvas = Image.new("RGB", (size * count, size + header), (238, 238, 238))
    draw = ImageDraw.Draw(canvas)
    for index, box in enumerate(boxes, start=1):
        crop = image.crop(
            pad_xyxy_crop(box, image.size, pad=max(0, int(classification_pad)))
        ).convert("RGB")
        crop = ImageOps.contain(crop, (size - 12, size - 12), Image.Resampling.BICUBIC)
        panel_x = (index - 1) * size
        paste_x = panel_x + (size - crop.width) // 2
        paste_y = header + (size - crop.height) // 2
        canvas.paste(crop, (paste_x, paste_y))
        draw.text((panel_x + 10, 10), f"CANDIDATE {index}", fill=(0, 0, 0))

    try:
        raw_text = run_vlm_prompt(
            canvas,
            classification_prompt,
            output="text",
            max_new_tokens=classification_max_new_tokens,
        ).raw_text
    except Exception:
        _logger.exception("joint Qwen instance classification failed")
        return {}

    match = _JSON_ARRAY_RE.search(raw_text or "")
    if not match:
        _logger.warning("joint Qwen classification returned no JSON array: %r", raw_text)
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        # Qwen occasionally emits ``"candidate":2"`` with one stray quote.
        # Recover only explicit candidate/shape pairs; this is output parsing,
        # not a pixel-based recognition fallback.
        payload = []
        allowed = "|".join(re.escape(label) for label in labels)
        for object_text in re.findall(r"\{[^{}]*\}", match.group(0)):
            candidate_match = re.search(
                r"""candidate["'\s:]*([1-9]\d*)""",
                object_text,
                flags=re.I,
            )
            label_match = re.search(
                rf"""shape["'\s:]*["']?({allowed})\b""",
                object_text,
                flags=re.I,
            )
            if candidate_match and label_match:
                payload.append(
                    {
                        "candidate": int(candidate_match.group(1)),
                        "shape": label_match.group(1).lower(),
                    }
                )
        if len(payload) != count:
            _logger.warning("joint Qwen classification returned invalid JSON: %r", raw_text)
            return {}
        _logger.info("recovered joint classification from malformed JSON: %r", raw_text)
    if not isinstance(payload, list) or len(payload) != count:
        return {}

    labeled: dict[str, BBoxXYXY] = {}
    seen_candidates: set[int] = set()
    for item in payload:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            raw_candidate, raw_label = item
        elif isinstance(item, dict):
            raw_candidate = next(
                (item.get(key) for key in ("candidate", "panel", "index", "id") if key in item),
                None,
            )
            raw_label = next(
                (item.get(key) for key in ("shape", "label", "name") if key in item),
                None,
            )
        else:
            return {}
        try:
            candidate = int(raw_candidate)
        except (TypeError, ValueError):
            return {}
        label = (
            normalize_label(raw_label)
            if normalize_label is not None
            else str(raw_label or "").strip().lower()
        )
        if (
            candidate < 1
            or candidate > count
            or candidate in seen_candidates
            or label not in labels
            or label in labeled
        ):
            return {}
        seen_candidates.add(candidate)
        labeled[label] = boxes[candidate - 1]

    if set(labeled) != set(labels):
        return {}
    _logger.info("joint Qwen classification raw=%r labels=%s", raw_text, sorted(labeled))
    return labeled


def ground_instances(
    image: Image.Image,
    *,
    prompt: str,
    search_crop: CropXYXY | Sequence[int] | None = None,
    max_new_tokens: int = 192,
    expected_count: int | None = None,
) -> list[BBoxXYXY]:
    """Ground unlabeled instances and return full-image pixel boxes."""
    working_image = image
    offset_x = 0
    offset_y = 0
    if search_crop is not None:
        crop_xyxy = pad_xyxy_crop(search_crop, image.size)
        offset_x, offset_y = crop_xyxy[:2]
        working_image = image.crop(crop_xyxy)

    try:
        result = run_vlm_prompt(
            working_image,
            prompt,
            output="boxes",
            max_new_tokens=max_new_tokens,
        )
    except Exception:
        _logger.exception("multi-instance Qwen grounding failed")
        return []

    boxes = [
        (
            float(box[0]) + offset_x,
            float(box[1]) + offset_y,
            float(box[2]) + offset_x,
            float(box[3]) + offset_y,
        )
        for box in result.boxes
        if len(box) == 4 and float(box[2]) > float(box[0]) and float(box[3]) > float(box[1])
    ]
    if expected_count is not None and len(boxes) != expected_count:
        _logger.warning(
            "multi-instance Qwen grounding returned %d boxes; expected %d",
            len(boxes),
            expected_count,
        )
        return []
    return boxes


def is_compact_box(
    box: BBoxXYXY,
    image_size: tuple[int, int],
    *,
    max_area_ratio: float = 0.045,
    max_width_ratio: float = 0.28,
    max_height_ratio: float = 0.35,
    min_center_x_ratio: float | None = None,
) -> bool:
    """Heuristic for rejecting whole-holder / whole-block boxes."""
    width, height = image_size
    x1, y1, x2, y2 = box
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw * bh > max_area_ratio * width * height:
        return False
    if bw > max_width_ratio * width or bh > max_height_ratio * height:
        return False
    if min_center_x_ratio is not None:
        center_x = 0.5 * (x1 + x2)
        if center_x < min_center_x_ratio * width:
            return False
    return True


def normalize_compact_box(
    raw_box: Any,
    image_size: tuple[int, int],
    *,
    max_area_ratio: float = 0.045,
    max_width_ratio: float = 0.28,
    max_height_ratio: float = 0.35,
    min_center_x_ratio: float | None = None,
) -> BBoxXYXY | None:
    """Validate an external box before using it as a crop or tracker seed."""
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        return None
    try:
        box = tuple(float(value) for value in raw_box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in box):
        return None

    width, height = image_size
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = box
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
        return None
    if not is_compact_box(
        box,
        image_size,
        max_area_ratio=max_area_ratio,
        max_width_ratio=max_width_ratio,
        max_height_ratio=max_height_ratio,
        min_center_x_ratio=min_center_x_ratio,
    ):
        return None
    return box
