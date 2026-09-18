"""Write current-frame ViT patch attention maps into a LeRobot dataset copy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from intent.attention import (
    VIT_PATCH_SIZE,
    bbox_to_patch_attention_map,
    normalize_model_input_resolution,
    patch_grid_shape,
)
from pattern.runtime.types import ALL_CAMERAS
from perception.tracking.sam2_video import mask_to_bbox_xyxy

DEFAULT_SEMANTIC_GROUNDING_COLUMNS = {
    "global": "global_semantic_grounding",
    "left_wrist": "left_wrist_semantic_grounding",
    "right_wrist": "right_wrist_semantic_grounding",
}


def semantic_grounding_columns(config: Mapping[str, Any]) -> dict[str, str]:
    """Resolve the three fixed-size parquet output columns."""
    dataset_config = config.get("dataset") or {}
    configured = dataset_config.get("semantic_grounding_columns") or {}
    columns = {
        camera: str(configured.get(camera) or DEFAULT_SEMANTIC_GROUNDING_COLUMNS[camera])
        for camera in ALL_CAMERAS
    }
    if len(set(columns.values())) != len(columns):
        raise ValueError(f"semantic grounding column names must be unique: {columns}")
    return columns


def masks_to_patch_attention_maps(
    *,
    tracked_masks: Mapping[str, Mapping[int, np.ndarray]],
    image_sizes: Mapping[str, tuple[int, int]],
    frame_count: int,
    model_input_resolution: tuple[int, int],
) -> dict[str, list[list[list[float]]]]:
    """Convert each current-frame tracker bbox to model-space patch weights."""
    resolution = normalize_model_input_resolution(model_input_resolution)
    output: dict[str, list[list[list[float]]]] = {}
    for camera in ALL_CAMERAS:
        width, height = image_sizes[camera]
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid source image size for {camera}: {(width, height)}")
        camera_masks = tracked_masks.get(camera) or {}
        rows: list[list[list[float]]] = []
        for frame_index in range(frame_count):
            mask = camera_masks.get(frame_index)
            bbox = mask_to_bbox_xyxy(mask) if mask is not None else None
            # mask_to_bbox_xyxy returns inclusive maxima; convert to the
            # half-open pixel rectangle used for area coverage.
            if bbox is not None:
                bbox = (bbox[0], bbox[1], bbox[2] + 1.0, bbox[3] + 1.0)
            attention_map, _ = bbox_to_patch_attention_map(
                bbox,
                source_size=(width, height),
                model_input_resolution=resolution,
            )
            rows.append(attention_map.tolist())
        output[camera] = rows
    return output


def write_semantic_grounding_parquet(
    parquet_path: Path,
    *,
    maps_by_camera: Mapping[str, Sequence[Sequence[Sequence[float]]]],
    columns: Mapping[str, str],
    model_input_resolution: tuple[int, int],
) -> None:
    """Atomically add or replace fixed-size patch-map columns."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    resolution = normalize_model_input_resolution(model_input_resolution)
    grid_height, grid_width = patch_grid_shape(resolution)
    map_type = pa.list_(pa.list_(pa.float32(), grid_width), grid_height)
    table = pq.read_table(parquet_path)
    for camera in ALL_CAMERAS:
        rows = maps_by_camera[camera]
        if len(rows) != table.num_rows:
            raise ValueError(
                f"attention-map row count mismatch for {camera}: "
                f"maps={len(rows)} parquet={table.num_rows}"
            )
        values = pa.array(rows, type=map_type)
        column_name = columns[camera]
        if column_name in table.column_names:
            table = table.set_column(table.column_names.index(column_name), column_name, values)
        else:
            table = table.append_column(column_name, values)

    temp_path = parquet_path.with_name(f".{parquet_path.name}.attention.tmp")
    try:
        pq.write_table(table, temp_path)
        temp_path.replace(parquet_path)
    finally:
        temp_path.unlink(missing_ok=True)


def update_info_semantic_grounding_features(
    dataset_root: Path,
    *,
    columns: Mapping[str, str],
    model_input_resolution: tuple[int, int],
) -> None:
    """Register patch attention maps in LeRobot ``meta/info.json``."""
    resolution = normalize_model_input_resolution(model_input_resolution)
    grid_height, grid_width = patch_grid_shape(resolution)
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.setdefault("features", {})
    for camera in ALL_CAMERAS:
        features[columns[camera]] = {
            "dtype": "float32",
            "shape": [grid_height, grid_width],
            "info": {
                "spatial_encoding": "vit_patch_attention",
                "model_input_resolution": list(resolution),
                "patch_size": [VIT_PATCH_SIZE, VIT_PATCH_SIZE],
            },
        }
    temp_path = info_path.with_name(f".{info_path.name}.attention.tmp")
    try:
        temp_path.write_text(json.dumps(info, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
        temp_path.replace(info_path)
    finally:
        temp_path.unlink(missing_ok=True)
