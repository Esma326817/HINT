"""BBox geometry helpers used across tasks."""

from __future__ import annotations

from typing import Any


def bbox_iou(a: list[float] | list[int], b: list[float] | list[int]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def bbox_intersection_over_min_area(a: list[float] | list[int], b: list[float] | list[int]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    min_area = min(area_a, area_b)
    return inter / min_area if min_area > 0 else 0.0


def dedupe_detections(detections: list[dict[str, Any]], iou_threshold: float = 0.55) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for det in sorted(detections, key=lambda item: float(item.get("confidence") or 0.0), reverse=True):
        bbox = det.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue
        if any(
            bbox_iou(list(bbox), list(other["bbox_xyxy"])) >= iou_threshold
            or bbox_intersection_over_min_area(list(bbox), list(other["bbox_xyxy"])) >= 0.85
            for other in kept
        ):
            continue
        kept.append(det)
    for idx, det in enumerate(kept):
        det["id"] = idx
    return kept


def classify_crop_label(image, bbox_xyxy: list[float], classify_spec: Any) -> str:
    """Run the classify prompt on one crop and map the answer through aliases."""
    from common.vision.crop import crop_xyxy
    from perception.task_manager.qwen_prompt import generate_text
    from perception.task_manager.qwen_runtime import get_qwen_runtime

    prompt = classify_spec.prompt or (
        f"Choose one label from {list(classify_spec.labels)}. Output only the label."
    )
    raw = generate_text(
        crop_xyxy(image, bbox_xyxy, padding=4),
        prompt,
        get_qwen_runtime(),
        max_new_tokens=8,
    )
    try:
        return classify_spec.require(raw)
    except ValueError:
        label = classify_spec.find_in_text(raw)
        if label == classify_spec.unknown_label:
            raise ValueError(f"failed to extract {classify_spec.key} from Qwen output: {raw!r}")
        return label
