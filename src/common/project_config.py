"""Cached experiment settings and environment-aware resource paths for HINT."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from common.config_loader import PROJECT_ROOT, load_reasoning_config


def _is_configured(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return text not in ("", "null", "none", "~")


def _resolve_path(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


@lru_cache(maxsize=1)
def load_project_config() -> dict[str, Any]:
    """Cache the default experiment config; call ``cache_clear()`` to reload.

    Set ``REASONING_AGENT_CONFIG`` before the first call. Treat the returned
    dictionary as read-only; use ``load_reasoning_config`` for an uncached copy.
    """
    return load_reasoning_config()


def resolve_output_root(config: dict[str, Any] | None = None) -> Path:
    """Resolve the output root from the environment, YAML, or ``outputs/``.

    Relative YAML paths use the repository root. Environment paths retain
    their working-directory-relative semantics.
    """
    env_value = os.getenv("REASONING_AGENT_OUTPUT_ROOT")
    if _is_configured(env_value):
        return Path(str(env_value))

    cfg = config if config is not None else load_project_config()
    root = cfg["output"]["root"]
    if _is_configured(root):
        return _resolve_path(root)
    return PROJECT_ROOT / "outputs"


def resolve_render_image_dir(config: dict[str, Any] | None = None) -> Path:
    """Directory that hosts online render sessions (``session_*`` / ``live``).

    Resolution order:
      1. ``REASONING_AGENT_RENDER_IMAGE_DIR``
      2. ``output.render_image`` (absolute, or relative to ``output.root``)
      3. ``<output.root>/render_image``
    """
    env_value = os.getenv("REASONING_AGENT_RENDER_IMAGE_DIR")
    if _is_configured(env_value):
        return Path(str(env_value))

    cfg = config if config is not None else load_project_config()
    output_cfg = cfg["output"]
    configured = output_cfg.get("render_image")
    if _is_configured(configured):
        path = Path(str(configured))
        if path.is_absolute():
            return path
        return (resolve_output_root(cfg) / path).resolve()
    return resolve_output_root(cfg) / "render_image"


def resolve_dino_settings(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve GroundingDINO checkpoint / device from YAML + env overrides.

    ``config_path`` is optional: when unset, the config shipped inside the pip
    ``groundingdino`` package is used (no repo clone required).
    For existing experiment configs, relative custom config paths are resolved
    against the repository's parent; checkpoint paths use the repository root.
    """
    cfg = config if config is not None else load_project_config()
    dino_cfg = cfg["grounding"]["dino"]

    checkpoint = os.getenv("GROUNDING_DINO_WEIGHTS") or dino_cfg.get("checkpoint")
    config_path = os.getenv("GROUNDING_DINO_CONFIG") or dino_cfg.get("config_path")

    if _is_configured(config_path):
        resolved_config = _resolve_path(config_path, base=PROJECT_ROOT.parent)
    else:
        from perception.semantic_grounder.dino_backend import package_default_config_path

        resolved_config = package_default_config_path()

    device = os.getenv("GROUNDING_DINO_DEVICE") or dino_cfg.get("device") or "cuda"

    return {
        "checkpoint": _resolve_path(checkpoint),
        "config_path": resolved_config,
        "device": str(device),
        "box_threshold": float(dino_cfg.get("box_threshold", 0.25)),
        "text_threshold": float(dino_cfg.get("text_threshold", 0.2)),
        "max_box_area_ratio": float(dino_cfg.get("max_box_area_ratio", 0.5)),
    }
