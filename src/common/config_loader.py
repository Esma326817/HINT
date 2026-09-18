"""YAML loading and the default HINT experiment configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "data" / "reasoning_agent_letter.yaml"


def deep_merge_mappings(
    base: Mapping[str, Any], overlay: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge mappings; nested dicts merge, lists and scalars in ``overlay`` replace."""
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge_mappings(current, value)
        else:
            merged[key] = value
    return merged


def _resolve_extends_path(child: Path, extends: str) -> Path:
    parent = Path(extends)
    if not parent.is_absolute():
        parent = child.parent / parent
    if parent.suffix == "":
        parent = parent.with_suffix(".yaml")
    return parent


def load_yaml_file(
    path: str | Path,
    *,
    _stack: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Read a YAML mapping, resolving ``extends`` relative to the file."""
    path = Path(path)
    resolved = path.resolve()
    if resolved in _stack:
        raise ValueError(f"circular config extends involving {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    extends = config.pop("extends", None)
    if not extends:
        return config
    parent = load_yaml_file(
        _resolve_extends_path(path, str(extends)),
        _stack=(*_stack, resolved),
    )
    return deep_merge_mappings(parent, config)


def load_reasoning_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load an explicit path, ``REASONING_AGENT_CONFIG``, or the default YAML.

    Explicit and environment-supplied relative paths use the working directory.
    The default path is anchored to the repository, independent of that directory.
    """
    config_path = Path(path or os.getenv("REASONING_AGENT_CONFIG", DEFAULT_CONFIG_PATH))
    return load_yaml_file(config_path)
