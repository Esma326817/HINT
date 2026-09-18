"""SAM2 / bbox-mask tracking between grounding events.

Imports are lazy so online code can load ``TrackingManager`` without pulling the
offline SAM2 video tracker, and offline scripts can load ``SAM2VideoTracker``
without constructing the online manager.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AREA_PLACEMENT_RENDER_MODE",
    "CameraTrackerState",
    "SAM2Config",
    "SAM2PropagationResult",
    "SAM2VideoTracker",
    "TrackingManager",
    "bbox_to_mask",
    "global_render_mode",
    "mask_to_bbox_xyxy",
]

_EXPORTS = {
    "TrackingManager": ("perception.tracking.manager", "TrackingManager"),
    "SAM2Config": ("perception.tracking.sam2_config", "SAM2Config"),
    "SAM2PropagationResult": (
        "perception.tracking.sam2_video",
        "SAM2PropagationResult",
    ),
    "SAM2VideoTracker": ("perception.tracking.sam2_video", "SAM2VideoTracker"),
    "mask_to_bbox_xyxy": ("perception.tracking.sam2_video", "mask_to_bbox_xyxy"),
    "CameraTrackerState": ("perception.tracking.state", "CameraTrackerState"),
    "bbox_to_mask": ("perception.tracking.state", "bbox_to_mask"),
    "global_render_mode": ("perception.tracking.state", "global_render_mode"),
    "AREA_PLACEMENT_RENDER_MODE": (
        "perception.tracking.state",
        "AREA_PLACEMENT_RENDER_MODE",
    ),
}


def __getattr__(name: str) -> Any:
    from importlib import import_module

    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
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
