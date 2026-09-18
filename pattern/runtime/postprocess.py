"""Offline stage-id smoothing shared by annotation and evaluation.

The online counterpart is ``StageStabilizer`` in
``pattern.runtime.manipulation_pattern_router``.
"""

from __future__ import annotations


def smooth_predictions(predictions: list[int], stable_frames: int) -> list[int]:
    """Debounce stage changes until a candidate persists for ``stable_frames``."""
    if not predictions:
        return []
    if stable_frames <= 1:
        return list(predictions)

    current = predictions[0]
    candidate: int | None = None
    candidate_count = 0
    smoothed: list[int] = []
    for prediction in predictions:
        if prediction == current:
            candidate = None
            candidate_count = 0
        elif prediction == candidate:
            candidate_count += 1
        else:
            candidate = prediction
            candidate_count = 1

        if candidate_count >= stable_frames:
            current = candidate
            candidate = None
            candidate_count = 0
        smoothed.append(current)
    return smoothed
