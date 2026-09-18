"""In-process GroundingDINO inference (pip-installed ``groundingdino`` package).

Loaded lazily like SAM2 — no separate HTTP server or repo clone required.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert

import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict

from common.step_profiler import PROFILER

DEFAULT_BOX_THRESHOLD = 0.25
DEFAULT_TEXT_THRESHOLD = 0.2
DEFAULT_MAX_BOX_AREA_RATIO = 0.5
DEFAULT_PACKAGE_CONFIG_NAME = "GroundingDINO_SwinT_OGC.py"

_agent_lock = threading.Lock()
_agent: GroundingDinoDetectAgent | None = None
_agent_key: tuple[str, str, str] | None = None


def package_default_config_path() -> Path:
    """Config shipped inside the installed ``groundingdino`` package."""
    import groundingdino

    package_root = Path(groundingdino.__file__).resolve().parent
    candidate = package_root / "config" / DEFAULT_PACKAGE_CONFIG_NAME
    if candidate.exists():
        return candidate
    # Older layouts used groundingdino/config under the package root.
    alt = package_root / "GroundingDINO_SwinT_OGC.py"
    if alt.exists():
        return alt
    raise FileNotFoundError(
        f"GroundingDINO package config not found under {package_root}. "
        "Reinstall with: bash install.sh"
    )


@dataclass
class DetectorSettings:
    config_path: str
    weights_path: str
    device: str = "cuda"
    box_threshold: float = DEFAULT_BOX_THRESHOLD
    text_threshold: float = DEFAULT_TEXT_THRESHOLD
    max_box_area_ratio: float = DEFAULT_MAX_BOX_AREA_RATIO


class GroundingDinoDetectAgent:
    """Prompt-mode GroundingDINO detector for local (in-process) use."""

    def __init__(self, settings: DetectorSettings):
        self.settings = settings
        self.model = load_model(
            model_config_path=settings.config_path,
            model_checkpoint_path=settings.weights_path,
            device=settings.device,
        )

    @staticmethod
    def settings_from_resolved(dino: dict[str, Any]) -> DetectorSettings:
        return DetectorSettings(
            config_path=str(dino["config_path"]),
            weights_path=str(dino["checkpoint"]),
            device=str(os.environ.get("GROUNDING_DINO_DEVICE") or dino.get("device") or "cuda"),
            box_threshold=float(
                os.environ.get("GROUNDING_DINO_BOX_THRESHOLD", dino.get("box_threshold", DEFAULT_BOX_THRESHOLD))
            ),
            text_threshold=float(
                os.environ.get(
                    "GROUNDING_DINO_TEXT_THRESHOLD", dino.get("text_threshold", DEFAULT_TEXT_THRESHOLD)
                )
            ),
            max_box_area_ratio=float(
                os.environ.get(
                    "GROUNDING_DINO_MAX_BOX_AREA_RATIO",
                    dino.get("max_box_area_ratio", DEFAULT_MAX_BOX_AREA_RATIO),
                )
            ),
        )

    @staticmethod
    def default_settings() -> DetectorSettings:
        from common.project_config import resolve_dino_settings

        return GroundingDinoDetectAgent.settings_from_resolved(resolve_dino_settings())

    @staticmethod
    def _transform():
        return T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _to_detections(
        boxes: torch.Tensor,
        logits: torch.Tensor,
        width: int,
        height: int,
        max_box_area_ratio: float,
        keep_top_k: int | None,
    ) -> list[dict[str, Any]]:
        boxes_pixel = boxes * torch.Tensor([width, height, width, height])
        boxes_xyxy = box_convert(boxes=boxes_pixel, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        image_area = float(width * height)

        detections: list[dict[str, Any]] = []
        for box, logit in zip(boxes_xyxy, logits):
            x1, y1, x2, y2 = box.tolist()
            box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if image_area > 0 and box_area / image_area > max_box_area_ratio:
                continue
            detections.append(
                {
                    "confidence": round(float(logit.item()), 4),
                    "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                }
            )

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        if keep_top_k is not None:
            detections = detections[:keep_top_k]
        for idx, detection in enumerate(detections):
            detection["id"] = idx
        return detections

    def detect_pil(
        self,
        image: Image.Image,
        prompt: str,
        *,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
        max_box_area_ratio: float | None = None,
        keep_top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        caption = str(prompt or "").strip()
        if not caption:
            raise ValueError("prompt mode requires a non-empty prompt")

        rgb = image.convert("RGB")
        width, height = rgb.size
        image_tensor, _ = self._transform()(rgb, None)

        with PROFILER.section("grounding/dino", cuda=True):
            boxes, logits, _phrases = predict(
                model=self.model,
                image=image_tensor,
                caption=caption,
                box_threshold=(
                    float(box_threshold)
                    if box_threshold is not None
                    else self.settings.box_threshold
                ),
                text_threshold=(
                    float(text_threshold)
                    if text_threshold is not None
                    else self.settings.text_threshold
                ),
                device=self.settings.device,
            )
        return self._to_detections(
            boxes,
            logits,
            width,
            height,
            float(max_box_area_ratio)
            if max_box_area_ratio is not None
            else self.settings.max_box_area_ratio,
            keep_top_k,
        )


def get_grounding_dino_agent(config: dict[str, Any] | None = None) -> GroundingDinoDetectAgent:
    """Lazy singleton agent keyed by checkpoint / config / device."""
    from common.project_config import resolve_dino_settings

    global _agent, _agent_key

    dino = resolve_dino_settings(config)
    settings = GroundingDinoDetectAgent.settings_from_resolved(dino)
    key = (settings.weights_path, settings.config_path, settings.device)

    with _agent_lock:
        if _agent is not None and _agent_key == key:
            return _agent
        _agent = GroundingDinoDetectAgent(settings)
        _agent_key = key
        return _agent


def reset_grounding_dino_agent() -> None:
    """Drop the cached agent (tests / device switch)."""
    global _agent, _agent_key
    with _agent_lock:
        _agent = None
        _agent_key = None
