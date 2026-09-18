"""Appearance descriptors + similarity for target re-identification.

Shared by the offline robustness harness (``eval_similarity_reacquire``) and the
online wrist ``TrackingManager``. The idea: between sparse inference steps
the wrist view jumps, so instead of re-prompting SAM2 with a stale box we detect
the candidate letter blocks in the current frame and pick the one whose
appearance is most similar to the previously tracked target.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
from PIL import Image

from common.vision.crop import crop_xyxy

BBox = Sequence[float]
Descriptor = dict


def hsv_histogram(crop: Image.Image, h_bins: int = 50, s_bins: int = 60) -> np.ndarray:
    """Normalised H-S histogram of a crop — robust to scale/small rotation."""
    rgb = np.asarray(crop.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()


def gray_descriptor(crop: Image.Image, size: int = 32) -> np.ndarray:
    """Flattened, mean-subtracted, L2-normalised grayscale patch (for NCC)."""
    gray = np.asarray(crop.convert("L").resize((size, size)), dtype=np.float32)
    gray -= gray.mean()
    norm = float(np.linalg.norm(gray))
    return (gray / norm).flatten() if norm > 1e-6 else gray.flatten()


def describe(crop: Image.Image) -> Descriptor:
    """Build the descriptor bundle for a crop."""
    return {"hsv": hsv_histogram(crop), "gray": gray_descriptor(crop)}


def describe_box(frame: Image.Image, box: BBox, padding: int = 0) -> Descriptor | None:
    """Describe the crop of ``box`` from ``frame``; None if the crop is degenerate."""
    try:
        return describe(crop_xyxy(frame, list(box), padding=padding))
    except ValueError:
        return None


def similarity(ref: Descriptor, cand: Descriptor, kind: str = "combined") -> float:
    """Similarity in [0, 1] between two descriptor bundles."""
    hsv_sim = max(0.0, float(cv2.compareHist(ref["hsv"], cand["hsv"], cv2.HISTCMP_CORREL)))
    gray_sim = max(0.0, float(np.dot(ref["gray"], cand["gray"])))
    if kind == "hsv":
        return hsv_sim
    if kind == "gray":
        return gray_sim
    return 0.5 * hsv_sim + 0.5 * gray_sim  # combined


def pick_most_similar(
    reference: Descriptor,
    candidate_boxes: list[BBox],
    frame: Image.Image,
    kind: str = "combined",
    padding: int = 0,
) -> tuple[int, float, Descriptor | None]:
    """Pick the candidate box most similar to ``reference``.

    Returns ``(index, score, picked_descriptor)``; index = -1 if no usable crop.
    """
    best_idx, best_score, best_desc = -1, -1.0, None
    for idx, box in enumerate(candidate_boxes):
        desc = describe_box(frame, box, padding=padding)
        if desc is None:
            continue
        score = similarity(reference, desc, kind)
        if score > best_score:
            best_idx, best_score, best_desc = idx, score, desc
    return best_idx, best_score, best_desc
