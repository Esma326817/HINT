"""Semantic grounding backends and shared result-selection helpers.

Imports are lazy so task-handler discovery and tests can load lightweight
submodules (attention maps, robust strategies) without pulling GroundingDINO
or the Qwen runtime.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DetectorSettings",
    "DinoClient",
    "DinoClientError",
    "DinoPromptOptions",
    "GroundingDinoDetectAgent",
    "GroundingSelection",
    "get_grounding_dino_agent",
    "ground_and_classify_instances",
    "ground_labeled_instances",
    "is_compact_box",
    "normalize_compact_box",
    "package_default_config_path",
    "pad_xyxy_crop",
    "qwen_max_new_tokens",
    "reset_grounding_dino_agent",
    "resolve_grounding_selection",
]

_EXPORTS = {
    "DinoClient": ("perception.semantic_grounder.dino", "DinoClient"),
    "DinoClientError": ("perception.semantic_grounder.dino", "DinoClientError"),
    "DinoPromptOptions": ("perception.semantic_grounder.dino", "DinoPromptOptions"),
    "GroundingSelection": ("perception.semantic_grounder.selection", "GroundingSelection"),
    "resolve_grounding_selection": (
        "perception.semantic_grounder.selection",
        "resolve_grounding_selection",
    ),
    "ground_and_classify_instances": (
        "perception.semantic_grounder.qwen_crop",
        "ground_and_classify_instances",
    ),
    "ground_labeled_instances": (
        "perception.semantic_grounder.qwen_crop",
        "ground_labeled_instances",
    ),
    "is_compact_box": ("perception.semantic_grounder.qwen_crop", "is_compact_box"),
    "normalize_compact_box": (
        "perception.semantic_grounder.qwen_crop",
        "normalize_compact_box",
    ),
    "pad_xyxy_crop": ("perception.semantic_grounder.qwen_crop", "pad_xyxy_crop"),
    "qwen_max_new_tokens": ("perception.semantic_grounder.qwen_crop", "qwen_max_new_tokens"),
    "DetectorSettings": ("perception.semantic_grounder.dino_backend", "DetectorSettings"),
    "GroundingDinoDetectAgent": (
        "perception.semantic_grounder.dino_backend",
        "GroundingDinoDetectAgent",
    ),
    "get_grounding_dino_agent": (
        "perception.semantic_grounder.dino_backend",
        "get_grounding_dino_agent",
    ),
    "package_default_config_path": (
        "perception.semantic_grounder.dino_backend",
        "package_default_config_path",
    ),
    "reset_grounding_dino_agent": (
        "perception.semantic_grounder.dino_backend",
        "reset_grounding_dino_agent",
    ),
}


def __getattr__(name: str) -> Any:
    from importlib import import_module

    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        try:
            value = getattr(import_module(module_name), attr_name)
        except Exception:
            if module_name.endswith("dino_backend"):
                return None
            raise
        globals()[name] = value
        return value
    try:
        value = import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) in {f"{__name__}.{name}", name}:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
        raise
    globals()[name] = value
    return value
