"""Configuration helpers shared by pattern command-line workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from easydict import EasyDict

from common.config_loader import load_yaml_file


def _to_easydict(value: Any) -> Any:
    if isinstance(value, dict):
        return EasyDict({key: _to_easydict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_easydict(item) for item in value]
    return value


def load_pattern_config(path: str | Path) -> EasyDict:
    """Load a pattern YAML file with attribute-style access."""
    return _to_easydict(load_yaml_file(Path(path)))


def to_plain_dict(value: Any) -> Any:
    """Convert nested EasyDict values into serialization-friendly containers."""
    if isinstance(value, dict):
        return {key: to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_dict(item) for item in value]
    return value
