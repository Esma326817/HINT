"""Disambiguate duplicate letter blocks during free_move grounding.

Reset stores a letter inventory. Each completed subtask records the used block
center. Free-move grounding then:

* ``>= 2`` unused inventory instances of the active letter → ask Qwen for all
  matches, drop board/used hits, take leftmost.
* ``< 2`` → single-box detect; if the hit is filtered, exclude it and retry once
  (asking for all remaining matches); if still empty, fall back to reset pose.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from PIL import Image

from task.operations import current_target_label
from pattern.runtime.types import FREE_MOVE_STAGE
from perception.task_manager.qwen_prompt import BBoxXYXY, is_center_inside

USED_CENTERS_KEY = "used_letter_centers"
LETTER_INVENTORY_KEY = "letter_inventory"
_USED_RADIUS_PX = 48.0
_MAX_SINGLE_RETRIES = 2


def snapshot_letter_inventory(task_state: Any) -> None:
    """Freeze recognized blocks at reset for later instance fallback."""
    if task_state is None:
        return
    inventory = []
    for block in getattr(task_state, "blocks", None) or ():
        letter = (getattr(block, "letter", None) or getattr(block, "label", None) or "").strip().lower()[:1]
        bbox = getattr(block, "bbox_xyxy", None)
        if not letter or bbox is None or len(bbox) != 4:
            continue
        inventory.append(
            {
                "id": int(block.id),
                "letter": letter,
                "bbox_xyxy": [float(v) for v in bbox],
            }
        )
    meta = getattr(task_state, "metadata", None)
    if not isinstance(meta, dict):
        task_state.metadata = {}
        meta = task_state.metadata
    meta[LETTER_INVENTORY_KEY] = inventory


def record_used_letter_center(task_state: Any) -> None:
    """Mark the active target block as consumed when a subtask completes."""
    if task_state is None:
        return
    from task.operations import resolve_target_block

    block = resolve_target_block(task_state)
    bbox = getattr(block, "bbox_xyxy", None) if block is not None else None
    if bbox is None or len(bbox) != 4:
        return
    meta = getattr(task_state, "metadata", None)
    if not isinstance(meta, dict):
        task_state.metadata = {}
        meta = task_state.metadata
    centers = list(meta.get(USED_CENTERS_KEY) or [])
    centers.append([*_center(bbox)])
    meta[USED_CENTERS_KEY] = centers


def maybe_ground_letter_free_move(
    *,
    image: Image.Image,
    task_state: Any,
    stage_name: str | None,
    config: dict[str, Any],
    qwen_boxes_fn: Callable[..., list[BBoxXYXY]] | None = None,
) -> Any | None:
    """Letter-only free_move path; ``None`` means caller should use the default."""
    from perception.semantic_grounder.selection import GroundingSelection

    if (
        stage_name != FREE_MOVE_STAGE
        or task_state is None
        or str(getattr(task_state, "task_name", "") or "") != "letter"
    ):
        return None

    letter = (current_target_label(task_state) or "").strip().lower()[:1]
    if not letter.isalpha():
        return None

    qwen_cfg = (config.get("grounding", {}) or {}).get("qwen", {}) or {}
    exclude_pad = float(qwen_cfg.get("board_pad", 6.0))
    max_new_tokens = int(qwen_cfg.get("max_new_tokens", 64))
    board = _exclude_board_region(task_state)
    used_centers = _used_centers(task_state)
    inventory = _remaining_inventory(task_state, letter, used_centers=used_centers)

    if qwen_boxes_fn is None:
        from perception.semantic_grounder.qwen import qwen_ground_letter_boxes

        qwen_boxes_fn = qwen_ground_letter_boxes

    def _detect(*, want_all: bool) -> list[BBoxXYXY]:
        return list(
            qwen_boxes_fn(
                image,
                letter,
                max_new_tokens=max_new_tokens,
                want_all=want_all,
                avoid_board=board is not None,
            )
            or ()
        )

    def _finish(box: BBoxXYXY, source: str) -> Any:
        _bind_target_block(task_state, box)
        return GroundingSelection(box, source)

    if len(inventory) >= 2:
        kept = _filter_boxes(
            _detect(want_all=True),
            board=board,
            used_centers=used_centers,
            rejected=(),
            pad=exclude_pad,
        )
        if kept:
            return _finish(_leftmost(kept), "qwen_letter_multi")
        return _finish(inventory[0], "inventory_letter")

    rejected: list[BBoxXYXY] = []
    for attempt in range(_MAX_SINGLE_RETRIES):
        # After a filtered hit, ask for all matches so the next pick can skip it.
        raw = _detect(want_all=attempt > 0 or bool(rejected))
        if not raw:
            break
        kept = _filter_boxes(
            raw,
            board=board,
            used_centers=used_centers,
            rejected=rejected,
            pad=exclude_pad,
        )
        if kept:
            return _finish(_leftmost(kept) if len(kept) > 1 else kept[0], "qwen_letter_single")
        rejected.extend(box for box in raw if not any(_same_instance(box, prior) for prior in rejected))

    if inventory:
        return _finish(inventory[0], "inventory_letter")
    return GroundingSelection(None, "letter_missing")


def _exclude_board_region(task_state: Any) -> BBoxXYXY | None:
    meta = getattr(task_state, "metadata", None) or {}
    board = meta.get("cutting_board_bbox") if isinstance(meta, dict) else None
    if board is not None and len(board) == 4:
        return (float(board[0]), float(board[1]), float(board[2]), float(board[3]))
    from perception.semantic_grounder.selection import _board_region_bbox

    return _board_region_bbox(task_state)


def _used_centers(task_state: Any) -> list[tuple[float, float]]:
    meta = getattr(task_state, "metadata", None) or {}
    raw = meta.get(USED_CENTERS_KEY) if isinstance(meta, dict) else None
    centers: list[tuple[float, float]] = []
    for item in raw or ():
        if item is None or len(item) < 2:
            continue
        centers.append((float(item[0]), float(item[1])))
    return centers


def _remaining_inventory(
    task_state: Any,
    letter: str,
    *,
    used_centers: Sequence[tuple[float, float]],
) -> list[BBoxXYXY]:
    meta = getattr(task_state, "metadata", None) or {}
    inventory = meta.get(LETTER_INVENTORY_KEY) if isinstance(meta, dict) else None
    boxes: list[BBoxXYXY] = []
    for item in inventory or ():
        if str(item.get("letter") or "").lower()[:1] != letter:
            continue
        bbox = item.get("bbox_xyxy")
        if bbox is None or len(bbox) != 4:
            continue
        box: BBoxXYXY = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        if _near_any(box, used_centers):
            continue
        boxes.append(box)
    return _sort_ltr(boxes)


def _filter_boxes(
    boxes: Sequence[BBoxXYXY],
    *,
    board: BBoxXYXY | None,
    used_centers: Sequence[tuple[float, float]],
    rejected: Sequence[BBoxXYXY],
    pad: float,
) -> list[BBoxXYXY]:
    kept: list[BBoxXYXY] = []
    for box in boxes:
        if board is not None and is_center_inside(box, board, pad):
            continue
        if _near_any(box, used_centers):
            continue
        if any(_same_instance(box, prior) for prior in rejected):
            continue
        kept.append(box)
    return kept


def _bind_target_block(task_state: Any, box: BBoxXYXY) -> None:
    """Keep subtask used-id bookkeeping aligned with the grounded instance."""
    meta = getattr(task_state, "metadata", None) or {}
    inventory = meta.get(LETTER_INVENTORY_KEY) if isinstance(meta, dict) else None
    best_id = None
    best_dist = float("inf")
    cx, cy = _center(box)
    for item in inventory or ():
        bbox = item.get("bbox_xyxy")
        if bbox is None or len(bbox) != 4:
            continue
        ix, iy = _center(bbox)
        dist = (cx - ix) ** 2 + (cy - iy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_id = int(item["id"])
    if best_id is not None:
        task_state.target_block_id = best_id


def _center(box: Sequence[float]) -> tuple[float, float]:
    return 0.5 * (float(box[0]) + float(box[2])), 0.5 * (float(box[1]) + float(box[3]))


def _near_any(box: BBoxXYXY, centers: Sequence[tuple[float, float]], radius: float = _USED_RADIUS_PX) -> bool:
    cx, cy = _center(box)
    r2 = radius * radius
    return any((cx - x) ** 2 + (cy - y) ** 2 <= r2 for x, y in centers)


def _same_instance(a: BBoxXYXY, b: BBoxXYXY, radius: float = _USED_RADIUS_PX) -> bool:
    return _near_any(a, [_center(b)], radius=radius)


def _sort_ltr(boxes: Sequence[BBoxXYXY]) -> list[BBoxXYXY]:
    return sorted(boxes, key=lambda box: (_center(box)[0], box[0], box[1]))


def _leftmost(boxes: Sequence[BBoxXYXY]) -> BBoxXYXY:
    return _sort_ltr(boxes)[0]
