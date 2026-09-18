"""Detection strategies for objects that plain text grounding gets wrong.

Some objects cannot be located by asking Qwen or GroundingDINO for a phrase, and
the reason is a property of the *object*, not of the task. Each strategy below
names that property so a new task can reuse a proven pipeline instead of
inventing one:

``separate_and_classify_instances``
    Several near-identical objects that are small relative to the frame and
    differ only in fine geometry (shape pegs: L-shaped vs circular vs rectangular). Text
    grounding returns the container or one merged box, so instance separation
    and fine-grained recognition are split into two VLM calls and every crop is
    upscaled before classification.

``classify_regions_in_parent``
    Small labeled sub-features inside a known parent region (the three shaped
    holes in the placed block). The parent box is the search crop, and all
    labels must come back exactly once or the result is rejected.

``recognize_with_padding_vote``
    Objects placed at random with an open label set (letter blocks: any letter,
    any pose). A single crop is unstable, so the same box is recognized at
    several paddings and the majority answer wins.

``recognize_target_candidate`` / ``detect_with_category_fallback``
    Open-set object names that may disagree across views while their task class
    comes from a finite vocabulary. Existing boxes are ranked by exact name,
    then category, then detector order; a task must explicitly opt into this
    best-effort behavior by declaring its category fallback.

``merge_host_part_detections``
    One physical object split into host/part detections with known labels
    (radish body + leaf mislabeled as lettuce/spinach, or two radish
    fragments). Adjacent matching boxes are unioned; the host label is kept.

Same-shape color pairs (blue/pink basket) use a generic DINO detect plus
``scene_object.classify`` in the task YAML rather than a dedicated strategy.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from perception.semantic_grounder.qwen_crop import (
    CropXYXY,
    LabelNormalizer,
    ground_and_classify_instances,
    ground_and_jointly_classify_instances,
    ground_instances,
    ground_labeled_instances,
    is_compact_box,
    pad_xyxy_crop,
    qwen_max_new_tokens,
)
from perception.task_manager.qwen_prompt import BBoxXYXY

BoxFilter = Callable[[BBoxXYXY, tuple[int, int]], bool]

DEFAULT_VOTE_PADDING_DELTAS = (0, 6, 12, 18, 24, 30)


@dataclass(frozen=True)
class RecognitionMatch:
    """How one candidate relates to the requested target."""

    observed_label: str
    observed_category: str = ""
    mode: str = "label"


@dataclass(frozen=True)
class DetectionFallbackResult:
    """Result of strict grounding followed by one category-constrained retry."""

    value: Any
    prompt: str
    mode: str


def recognize_target_candidate(
    image_crop: Any,
    *,
    recognize_fields: Callable[[Any], dict[str, str]],
    expected_label: str,
    expected_category: str = "",
    allowed_categories: Sequence[str] = (),
) -> RecognitionMatch | None:
    """Recognize one existing candidate and assign a semantic preference level.

    Name and finite-category matches remain useful ranking signals, but a valid
    crop always returns an observation.  This lets the caller choose the best of
    the boxes it already has instead of discarding every candidate because a
    partially visible object received the wrong open-set name/category.
    """
    from task.base import normalize_label, normalize_words

    fields = recognize_fields(image_crop)
    observed_label = str(fields.get("label") or "")
    observed_category = normalize_words(fields.get("category") or "")
    try:
        if normalize_label(expected_label) == normalize_label(observed_label):
            return RecognitionMatch(observed_label=observed_label, mode="label")
    except ValueError:
        pass

    expected = normalize_words(expected_category)
    allowed = {normalize_words(category) for category in allowed_categories}
    if expected and expected in allowed and observed_category == expected:
        return RecognitionMatch(
            observed_label=observed_label,
            observed_category=observed_category,
            mode="category",
        )
    return RecognitionMatch(
        observed_label=observed_label,
        observed_category=observed_category,
        mode="best_effort",
    )


def recognize_strict_then_category(
    image_crop: Any,
    *,
    recognize_fields: Callable[[Any], dict[str, str]],
    expected_label: str,
    expected_category: str = "",
    allowed_categories: Sequence[str] = (),
) -> RecognitionMatch | None:
    """Compatibility helper that keeps the former hard-match semantics."""
    match = recognize_target_candidate(
        image_crop,
        recognize_fields=recognize_fields,
        expected_label=expected_label,
        expected_category=expected_category,
        allowed_categories=allowed_categories,
    )
    return match if match.mode != "best_effort" else None


def detect_with_category_fallback(
    *,
    strict_prompt: str,
    detect_fn: Callable[[str], Any | None],
    expected_category: str = "",
    allowed_categories: Sequence[str] = (),
    category_prompt_template: str = "{category} object",
) -> DetectionFallbackResult | None:
    """Try strict grounding once, then retry with one allowed category prompt."""
    from task.base import normalize_words

    value = detect_fn(strict_prompt)
    if value is not None:
        return DetectionFallbackResult(value=value, prompt=strict_prompt, mode="strict")

    category = normalize_words(expected_category)
    allowed = {normalize_words(item) for item in allowed_categories}
    if not category or category not in allowed:
        return None
    category_prompt = category_prompt_template.format(category=category).strip()
    if not category_prompt or category_prompt == strict_prompt.strip():
        return None
    value = detect_fn(category_prompt)
    if value is None:
        return None
    return DetectionFallbackResult(value=value, prompt=category_prompt, mode="category")


def compact_box_filter(
    *,
    max_area_ratio: float = 0.045,
    max_width_ratio: float = 0.28,
    max_height_ratio: float = 0.35,
    min_center_x_ratio: float | None = None,
) -> BoxFilter:
    """Build a predicate that rejects container-sized boxes.

    Instance grounding for a small object frequently returns the rack, tray, or
    holder that carries it. The accepted size envelope is task data, so it stays
    a parameter.
    """

    def _accept(box: BBoxXYXY, image_size: tuple[int, int]) -> bool:
        return is_compact_box(
            box,
            image_size,
            max_area_ratio=max_area_ratio,
            max_width_ratio=max_width_ratio,
            max_height_ratio=max_height_ratio,
            min_center_x_ratio=min_center_x_ratio,
        )

    return _accept


def separate_and_classify_instances(
    image: Any,
    *,
    instance_prompt: str,
    classification_prompt: str,
    normalize_label: LabelNormalizer | None = None,
    expected_labels: Sequence[str] | frozenset[str] | set[str] | None = None,
    expected_count: int | None = None,
    box_filter: BoxFilter | None = None,
    search_crop: CropXYXY | Sequence[int] | None = None,
    search_pad: int = 0,
    min_crop_side: int = 256,
    classification_pad: int = 0,
    instance_min_new_tokens: int = 192,
    classification_max_new_tokens: int = 16,
    config: dict[str, Any] | None = None,
) -> dict[str, tuple[float, float, float, float]]:
    """Separate small look-alike instances, then classify every isolated crop.

    Returns full-image boxes keyed by label. The result is all-or-nothing: an
    empty dict means the scene was ambiguous (wrong instance count, duplicate
    label, container-sized box, or a missing expected label), which is safer
    than handing a wrong box to the tracker.
    """
    token_limit = max(instance_min_new_tokens, qwen_max_new_tokens(config))
    padded_search_crop = (
        pad_xyxy_crop(search_crop, image.size, pad=search_pad)
        if search_crop is not None
        else None
    )
    boxes = ground_and_classify_instances(
        image,
        instance_prompt=instance_prompt,
        classification_prompt=classification_prompt,
        search_crop=padded_search_crop,
        normalize_label=normalize_label,
        instance_max_new_tokens=token_limit,
        classification_max_new_tokens=classification_max_new_tokens,
        min_crop_side=min_crop_side,
        classification_pad=classification_pad,
        expected_count=expected_count,
    )
    located = {
        label: tuple(float(value) for value in box)
        for label, box in boxes.items()
        if box_filter is None or box_filter(box, image.size)
    }
    if expected_labels is not None and set(located) != set(expected_labels):
        return {}
    return located


def separate_and_jointly_classify_instances(
    image: Any,
    *,
    instance_prompt: str,
    classification_prompt: str,
    normalize_label: LabelNormalizer | None,
    expected_labels: Sequence[str] | frozenset[str] | set[str],
    expected_count: int | None = None,
    box_filter: BoxFilter | None = None,
    search_crop: CropXYXY | Sequence[int] | None = None,
    search_pad: int = 0,
    panel_size: int = 384,
    classification_pad: int = 8,
    instance_min_new_tokens: int = 192,
    classification_max_new_tokens: int = 96,
    config: dict[str, Any] | None = None,
) -> dict[str, tuple[float, float, float, float]]:
    """Detect instances, then ask the VLM for a joint one-to-one assignment.

    All shape recognition is performed with the vision-language model. The
    method does not assume that candidates have a fixed order in the source
    image.
    """
    labels = tuple(expected_labels)
    padded_search_crop = (
        pad_xyxy_crop(search_crop, image.size, pad=search_pad)
        if search_crop is not None
        else None
    )
    boxes = ground_and_jointly_classify_instances(
        image,
        instance_prompt=instance_prompt,
        classification_prompt=classification_prompt,
        expected_labels=labels,
        search_crop=padded_search_crop,
        normalize_label=normalize_label,
        instance_max_new_tokens=max(
            instance_min_new_tokens, qwen_max_new_tokens(config)
        ),
        classification_max_new_tokens=classification_max_new_tokens,
        panel_size=panel_size,
        classification_pad=classification_pad,
        expected_count=expected_count,
    )
    located = {
        label: tuple(float(value) for value in box)
        for label, box in boxes.items()
        if box_filter is None or box_filter(box, image.size)
    }
    return located if set(located) == set(labels) else {}


def locate_instances_by_order(
    image: Any,
    *,
    instance_prompt: str,
    labels_in_order: Sequence[str],
    order_axis: str = "y",
    reverse: bool = False,
    expected_count: int | None = None,
    box_filter: BoxFilter | None = None,
    search_crop: CropXYXY | Sequence[int] | None = None,
    search_pad: int = 0,
    instance_min_new_tokens: int = 192,
    config: dict[str, Any] | None = None,
) -> dict[str, tuple[float, float, float, float]]:
    """Locate instances whose physical order is fixed, without shape classification."""
    labels = tuple(str(label) for label in labels_in_order)
    count = expected_count if expected_count is not None else len(labels)
    if count != len(labels) or order_axis not in {"x", "y"}:
        raise ValueError("ordered detection needs one label per instance and order_axis x or y")
    padded_search_crop = (
        pad_xyxy_crop(search_crop, image.size, pad=search_pad)
        if search_crop is not None
        else None
    )
    boxes = ground_instances(
        image,
        prompt=instance_prompt,
        search_crop=padded_search_crop,
        max_new_tokens=max(instance_min_new_tokens, qwen_max_new_tokens(config)),
        expected_count=count,
    )
    boxes = [
        tuple(float(value) for value in box)
        for box in boxes
        if box_filter is None or box_filter(box, image.size)
    ]
    if len(boxes) != count:
        return {}
    coordinate = 0 if order_axis == "x" else 1
    boxes.sort(
        key=lambda box: 0.5 * (box[coordinate] + box[coordinate + 2]),
        reverse=reverse,
    )
    return dict(zip(labels, boxes))


def classify_regions_in_parent(
    image: Any,
    *,
    prompt: str,
    parent_box: Sequence[float],
    normalize_label: LabelNormalizer | None = None,
    expected_labels: Sequence[str] | frozenset[str] | set[str] | None = None,
    search_pad: int = 0,
    min_new_tokens: int = 192,
    config: dict[str, Any] | None = None,
) -> dict[str, tuple[float, float, float, float]]:
    """Label small sub-features inside a known parent region in one VLM call.

    ``parent_box`` is the enclosing box (a slot, tray, or block) in full-image
    pixels; it is padded and used as the search crop so the model never sees the
    rest of the table.
    """
    expected = frozenset(expected_labels) if expected_labels is not None else None
    return ground_labeled_instances(
        image,
        search_crop=pad_xyxy_crop(parent_box, image.size, pad=search_pad),
        prompt=prompt,
        normalize_label=normalize_label,
        max_new_tokens=max(min_new_tokens, qwen_max_new_tokens(config)),
        expected_labels=expected,
    )


def crop_paddings_for_vote(
    padding: int,
    deltas: Sequence[int] = DEFAULT_VOTE_PADDING_DELTAS,
) -> tuple[int, ...]:
    """Crop paddings to majority-vote one recognition over."""
    return tuple(sorted({max(0, padding + delta) for delta in deltas}))


def recognize_with_padding_vote(
    image: Any,
    bbox: Sequence[float],
    recognize_fn: Callable[[Any], str],
    padding: int,
    *,
    padding_deltas: Sequence[int] = DEFAULT_VOTE_PADDING_DELTAS,
) -> str:
    """Majority-vote one crop recognition across several crop paddings.

    For randomly placed objects with an open label set, how much context the
    crop includes changes the answer. Voting over paddings removes most of that
    sensitivity. Falls back to the plain single-padding call when every vote
    raised.
    """
    from common.vision.crop import crop_xyxy

    votes: list[str] = []
    for pad in crop_paddings_for_vote(padding, padding_deltas):
        try:
            votes.append(recognize_fn(crop_xyxy(image, bbox, padding=pad)))
        except Exception:  # noqa: BLE001 - a bad crop just doesn't get a vote
            continue
    if not votes:
        return recognize_fn(crop_xyxy(image, bbox, padding=padding))
    return Counter(votes).most_common(1)[0][0]


def _boxes_look_attached(a: Sequence[float], b: Sequence[float]) -> bool:
    """True when two boxes likely cover parts of the same physical object."""
    from common.vision.geometry import bbox_intersection_over_min_area

    if bbox_intersection_over_min_area(list(a), list(b)) >= 0.12:
        return True
    aw, ah = max(0.0, float(a[2]) - float(a[0])), max(0.0, float(a[3]) - float(a[1]))
    bw, bh = max(0.0, float(b[2]) - float(b[0])), max(0.0, float(b[3]) - float(b[1]))
    max_w, max_h = max(aw, bw), max(ah, bh)
    if max_w <= 0 or max_h <= 0:
        return False
    h_overlap = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    min_w = min(aw, bw)
    v_gap = max(0.0, max(float(a[1]), float(b[1])) - min(float(a[3]), float(b[3])))
    cx_gap = abs(0.5 * (float(a[0]) + float(a[2])) - 0.5 * (float(b[0]) + float(b[2])))
    return h_overlap / min_w >= 0.35 and cx_gap <= max_w * 0.45 and v_gap <= max_h * 0.35


def _normalize_merge_label(label: Any) -> str:
    text = " ".join(str(label or "").lower().split())
    if text.startswith("toy "):
        text = text[4:].strip()
    return text


def merge_host_part_detections(
    *,
    blocks: Sequence[Any],
    host_labels: Sequence[str],
    part_labels: Sequence[str],
    host_category: str | None = None,
) -> list[Any]:
    """Union adjacent host/part (or host/host) detections and keep the host label.

    Both boxes must match the declared host/part vocabulary, and the result keeps
    a host label (preferring the larger host member) so a radish leaf called
    lettuce/spinach does not survive as its own subtask.
    """
    if len(blocks) < 2:
        return list(blocks)

    from common.task_types import TargetObject
    from task.base import detection_area

    hosts = {_normalize_merge_label(label) for label in host_labels if str(label).strip()}
    parts = {_normalize_merge_label(label) for label in part_labels if str(label).strip()}
    if not hosts:
        return list(blocks)
    allowed = hosts | parts

    def role(block: Any) -> str | None:
        label = _normalize_merge_label(getattr(block, "label", None))
        if label in hosts:
            return "host"
        if label in parts:
            return "part"
        return None

    parent = list(range(len(blocks)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, left in enumerate(blocks):
        left_role = role(left)
        if left_role is None or _normalize_merge_label(left.label) not in allowed:
            continue
        for j in range(i + 1, len(blocks)):
            right = blocks[j]
            right_role = role(right)
            if right_role is None:
                continue
            # host+part, part+host, or host+host (split radish both labeled radish)
            if left_role == "part" and right_role == "part":
                continue
            if not _boxes_look_attached(left.bbox_xyxy, right.bbox_xyxy):
                continue
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

    groups: dict[int, list[int]] = {}
    for i in range(len(blocks)):
        groups.setdefault(find(i), []).append(i)
    if all(len(idxs) == 1 for idxs in groups.values()):
        return list(blocks)

    merged: list[TargetObject] = []
    for idxs in sorted(groups.values(), key=min):
        members = [blocks[i] for i in idxs]
        if len(members) == 1:
            block = members[0]
            merged.append(
                TargetObject(
                    id=len(merged),
                    bbox_xyxy=list(block.bbox_xyxy),
                    confidence=block.confidence,
                    letter=block.letter,
                    label=block.label,
                    category=block.category,
                )
            )
            continue

        boxes = [list(map(float, m.bbox_xyxy)) for m in members]
        bbox = [
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        ]
        host_members = [m for m in members if role(m) == "host"]
        canonical = max(
            host_members or members,
            key=lambda item: (detection_area({"bbox_xyxy": item.bbox_xyxy}), item.confidence or 0.0),
        )
        label = _normalize_merge_label(canonical.label)
        if label not in hosts and host_members:
            label = _normalize_merge_label(host_members[0].label)
        # Prefer the first declared host spelling when aliases vary.
        for preferred in host_labels:
            if _normalize_merge_label(preferred) == label:
                label = str(preferred).strip().lower()
                break
        category = host_category or canonical.category or "vegetable"
        merged.append(
            TargetObject(
                id=len(merged),
                bbox_xyxy=bbox,
                confidence=max((m.confidence or 0.0) for m in members),
                letter=None,
                label=label,
                category=category,
            )
        )
    return merged
