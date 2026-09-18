"""Local GroundingDINO client (in-process, no HTTP).

Public API mirrors the former HTTP client so task / tracking call sites stay stable.
The ``groundingdino`` package is imported lazily on first detect / health_check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


class DinoClientError(RuntimeError):
    """Raised when GroundingDINO inference fails or returns an invalid payload."""


_UNSET = object()


@dataclass(frozen=True)
class DinoPromptOptions:
    """Prompt-detection knobs shared across tasks."""

    box_threshold: float | None = None
    text_threshold: float | None = None
    max_box_area_ratio: float | None = None
    keep_top_k: int | None = 1


class DinoClient:
    """In-process GroundingDINO wrapper.

    The model is loaded lazily on first ``detect_*`` / ``health_check`` call.
    """

    def __init__(self, config: dict[str, Any] | None = None, **_ignored: Any) -> None:
        # ``base_url`` / HTTP kwargs are ignored for backward compatibility with
        # older call sites that still pass them.
        self.config = config
        self._agent = None

    def _resolve_agent(self):
        if self._agent is None:
            from perception.semantic_grounder.dino_backend import get_grounding_dino_agent

            self._agent = get_grounding_dino_agent(self.config)
        return self._agent

    def health_check(self) -> dict[str, Any]:
        agent = self._resolve_agent()
        return {
            "status": "ok",
            "backend": "inprocess",
            "device": agent.settings.device,
            "checkpoint": agent.settings.weights_path,
        }

    def detect_prompt(
        self,
        image: Image.Image,
        prompt: str,
        *,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
        max_box_area_ratio: float | None = None,
        keep_top_k: int | None = 1,
    ) -> list[dict[str, Any]]:
        try:
            detections = self._resolve_agent().detect_pil(
                image,
                prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                max_box_area_ratio=max_box_area_ratio,
                keep_top_k=keep_top_k,
            )
        except Exception as exc:
            raise DinoClientError(f"GroundingDINO detection failed: {exc}") from exc
        if not isinstance(detections, list):
            raise DinoClientError("DINO detections must be a list")
        return detections

    def detect_objects(
        self,
        image: Image.Image,
        prompt: str,
        *,
        options: DinoPromptOptions | None = None,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
        max_box_area_ratio: float | None = None,
        keep_top_k: int | None | object = _UNSET,
    ) -> list[dict[str, Any]]:
        """Detect prompt-described objects with task-provided settings.

        New tasks should prefer this method over adding task-specific methods to
        ``DinoClient``. Keyword arguments override values from ``options``.
        """

        opts = options or DinoPromptOptions()
        resolved_keep_top_k = opts.keep_top_k if keep_top_k is _UNSET else keep_top_k
        return self.detect_prompt(
            image,
            prompt,
            box_threshold=box_threshold if box_threshold is not None else opts.box_threshold,
            text_threshold=text_threshold if text_threshold is not None else opts.text_threshold,
            max_box_area_ratio=(
                max_box_area_ratio if max_box_area_ratio is not None else opts.max_box_area_ratio
            ),
            keep_top_k=resolved_keep_top_k if isinstance(resolved_keep_top_k, int) else None,
        )

    def ground_bbox(
        self,
        image: Image.Image,
        prompt: str,
        *,
        options: DinoPromptOptions | None = None,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
        max_box_area_ratio: float | None = None,
    ) -> tuple[float, float, float, float] | None:
        """Return the highest-confidence bbox for a prompt, or ``None``."""

        detections = self.detect_objects(
            image,
            prompt,
            options=options,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            max_box_area_ratio=max_box_area_ratio,
            keep_top_k=1,
        )
        if not detections:
            return None
        raw_box = detections[0].get("bbox_xyxy", ())
        if len(raw_box) != 4:
            return None
        return tuple(float(value) for value in raw_box)
