"""Canonical mapping between stage ids and the router's progress heads."""

from __future__ import annotations

PROGRESS_HEAD_BY_STAGE_ID: tuple[int, ...] = (0, 1, 1, 3, 3, 2)


def progress_head_for_stage(stage_id: int, num_heads: int) -> int | None:
    """Return the progress-head index for a one-based stage id."""
    stage_index = int(stage_id) - 1
    if not 0 <= stage_index < len(PROGRESS_HEAD_BY_STAGE_ID):
        return None
    if num_heads <= 1:
        return 0
    if num_heads == 4:
        return PROGRESS_HEAD_BY_STAGE_ID[stage_index]
    if num_heads == 6:
        return stage_index
    return None
