"""Model-space ViT patch attention maps derived from tracked source-image boxes."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_MODEL_INPUT_RESOLUTION = (224, 224)  # (height, width)
VIT_PATCH_SIZE = 14


def normalize_model_input_resolution(value: Any) -> tuple[int, int]:
    """Parse ``height,width`` and require a complete 14px ViT patch grid."""
    if value is None:
        resolution = DEFAULT_MODEL_INPUT_RESOLUTION
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            resolution = DEFAULT_MODEL_INPUT_RESOLUTION
        elif stripped.startswith("["):
            resolution = normalize_model_input_resolution(json.loads(stripped))
        else:
            parts = [part for part in re.split(r"[xX, ]+", stripped) if part]
            resolution = normalize_model_input_resolution([int(part) for part in parts])
    elif isinstance(value, (int, np.integer)):
        resolution = (int(value), int(value))
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = list(value)
        if len(items) == 1:
            resolution = (int(items[0]), int(items[0]))
        elif len(items) == 2:
            resolution = (int(items[0]), int(items[1]))
        else:
            raise ValueError(
                "model_input_resolution must contain [height, width], "
                f"got {items!r}"
            )
    else:
        raise TypeError(
            "model_input_resolution must be an int, 'HxW', or [height, width], "
            f"got {type(value).__name__}"
        )

    height, width = resolution
    if height <= 0 or width <= 0:
        raise ValueError(f"model_input_resolution must be positive, got {resolution}")
    if height % VIT_PATCH_SIZE or width % VIT_PATCH_SIZE:
        raise ValueError(
            f"model_input_resolution {resolution} must be divisible by the fixed "
            f"ViT patch size {VIT_PATCH_SIZE}"
        )
    return height, width


def resolve_model_input_resolution(
    config: Mapping[str, Any] | None,
    override: Any = None,
) -> tuple[int, int]:
    """Use the request override or the configured policy input resolution."""
    if override is not None:
        return normalize_model_input_resolution(override)
    return normalize_model_input_resolution(
        config.get("task", {}).get("model_input_resolution")
    )


def resolve_output_resolution(
    config: Mapping[str, Any] | None,
) -> tuple[int, int]:
    """Resolve policy RGB output size, falling back to model input geometry."""
    configured = config.get("task", {}).get("output_resolution")
    return (
        normalize_model_input_resolution(configured)
        if configured is not None
        else resolve_model_input_resolution(config)
    )


def patch_grid_shape(
    model_input_resolution: tuple[int, int],
) -> tuple[int, int]:
    height, width = normalize_model_input_resolution(model_input_resolution)
    return height // VIT_PATCH_SIZE, width // VIT_PATCH_SIZE


def bbox_to_model_space(
    bbox_xyxy: Sequence[float],
    *,
    source_size: tuple[int, int],
    model_input_resolution: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    """Map source ``xyxy`` through the policy's direct resize (no padding)."""
    source_width, source_height = source_size
    target_height, target_width = normalize_model_input_resolution(model_input_resolution)
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"source_size must be positive, got {source_size}")
    if len(bbox_xyxy) != 4:
        raise ValueError(f"bbox_xyxy must contain four values, got {bbox_xyxy!r}")

    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    x1, x2 = np.clip([x1, x2], 0.0, float(source_width))
    y1, y2 = np.clip([y1, y2], 0.0, float(source_height))
    if x2 <= x1 or y2 <= y1:
        return None

    scale_x = target_width / source_width
    scale_y = target_height / source_height

    mapped = (
        float(x1 * scale_x),
        float(y1 * scale_y),
        float(x2 * scale_x),
        float(y2 * scale_y),
    )
    if mapped[2] <= mapped[0] or mapped[3] <= mapped[1]:
        return None
    return mapped


def resize_policy_image(
    image: Image.Image,
    model_input_resolution: tuple[int, int] = DEFAULT_MODEL_INPUT_RESOLUTION,
) -> Image.Image:
    """Directly resize a source RGB image to the policy tensor geometry.

    Grounding and tracking operate on the original camera image. This conversion
    is deliberately the final boundary before dataset encoding or API response,
    so the VLM retains all 640x480 pixels while π receives a dense 224x224 image
    with no letterbox padding.
    """
    target_height, target_width = normalize_model_input_resolution(model_input_resolution)
    rgb = image.convert("RGB")
    if rgb.size == (target_width, target_height):
        return rgb
    return rgb.resize((target_width, target_height), Image.Resampling.BILINEAR)


def bbox_to_patch_attention_map(
    bbox_xyxy: Sequence[float] | None,
    *,
    source_size: tuple[int, int],
    model_input_resolution: tuple[int, int] = DEFAULT_MODEL_INPUT_RESOLUTION,
) -> tuple[np.ndarray, tuple[float, float, float, float] | None]:
    """Return per-patch bbox coverage weights and its model-space bbox."""
    target_height, target_width = normalize_model_input_resolution(model_input_resolution)
    grid_height, grid_width = patch_grid_shape((target_height, target_width))
    empty = np.zeros((grid_height, grid_width), dtype=np.float32)
    if bbox_xyxy is None:
        return empty, None

    model_bbox = bbox_to_model_space(
        bbox_xyxy,
        source_size=source_size,
        model_input_resolution=(target_height, target_width),
    )
    if model_bbox is None:
        return empty, None

    x1, y1, x2, y2 = model_bbox
    patch_left = np.arange(grid_width, dtype=np.float32) * VIT_PATCH_SIZE
    patch_top = np.arange(grid_height, dtype=np.float32) * VIT_PATCH_SIZE
    overlap_x = np.maximum(
        0.0,
        np.minimum(patch_left + VIT_PATCH_SIZE, x2) - np.maximum(patch_left, x1),
    )
    overlap_y = np.maximum(
        0.0,
        np.minimum(patch_top + VIT_PATCH_SIZE, y2) - np.maximum(patch_top, y1),
    )
    weights = np.outer(overlap_y, overlap_x) / float(VIT_PATCH_SIZE**2)
    return np.clip(weights, 0.0, 1.0).astype(np.float32), model_bbox


def patch_attention_payload(
    bbox_xyxy: Sequence[float] | None,
    *,
    source_size: tuple[int, int],
    model_input_resolution: tuple[int, int],
    grounding_method: str,
    temporal_age: int = 0,
) -> dict[str, Any]:
    """Build the online JSON representation consumed directly by policies."""
    resolution = normalize_model_input_resolution(model_input_resolution)
    attention_map, model_bbox = bbox_to_patch_attention_map(
        bbox_xyxy,
        source_size=source_size,
        model_input_resolution=resolution,
    )
    return {
        "spatial_encoding": "vit_patch_attention",
        "attention_map": attention_map.tolist(),
        "grid_size": list(attention_map.shape),
        "model_input_resolution": list(resolution),
        "patch_size": [VIT_PATCH_SIZE, VIT_PATCH_SIZE],
        "model_bbox_xyxy": list(model_bbox) if model_bbox is not None else None,
        "valid": model_bbox is not None,
        "source_size": list(source_size),
        "grounding_method": str(grounding_method),
        "temporal_age": int(temporal_age),
    }
