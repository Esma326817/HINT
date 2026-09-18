"""Configurable LeRobot fields, ordered by the model's camera names."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


DEFAULT_CAMERA_KEYS = {
    "global": "images.global",
    "left_wrist": "images.left_wrist",
    "right_wrist": "images.right_wrist",
}


@dataclass(frozen=True)
class LeRobotDataKeys:
    camera_keys: tuple[str, ...] = tuple(DEFAULT_CAMERA_KEYS.values())
    state_key: str = "state"
    effort_key: str = "effort"


def resolve_data_keys(
    *configs: Mapping[str, Any] | None,
    camera_names: Sequence[str] | None = None,
) -> LeRobotDataKeys:
    """Merge dataset configs in order; absent/null values inherit earlier keys.

    Defaults preserve existing datasets and checkpoints. Camera mappings are
    resolved by model camera name, independent of the YAML mapping order.
    """
    names = tuple(DEFAULT_CAMERA_KEYS) if camera_names is None else tuple(camera_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("camera_names must be nonempty and unique")
    cameras = dict(DEFAULT_CAMERA_KEYS)
    lowdim = {"state_key": "state", "effort_key": "effort"}
    for cfg in configs:
        if cfg is None:
            continue
        camera_keys = cfg.get("camera_keys")
        if camera_keys is not None:
            if not isinstance(camera_keys, Mapping):
                raise ValueError("dataset.camera_keys must map model camera names to LeRobot video keys")
            unknown = set(camera_keys) - set(names)
            if unknown:
                raise ValueError(f"dataset.camera_keys contains unknown model camera names: {sorted(unknown)}")
            cameras.update(camera_keys)
        for key in lowdim:
            if cfg.get(key) is not None:
                lowdim[key] = cfg[key]
    for name in names:
        value = cameras.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"dataset.camera_keys.{name} must be a nonempty LeRobot video key")
    for key, value in lowdim.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"dataset.{key} must be a nonempty Parquet column name")
    return LeRobotDataKeys(tuple(cameras[name] for name in names), **lowdim)
