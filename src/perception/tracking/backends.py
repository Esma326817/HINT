"""SAM2 / bbox tracker backends used between grounding events."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from perception.tracking.state import CameraTrackerState, bbox_to_mask
from perception.tracking.sam2_config import SAM2Config
from perception.tracking.sam2_video import BBoxXYXY, mask_to_bbox_xyxy


class BBoxMaskTrackerBackend:
    """Keep the grounded bbox mask until the next grounding event.

    This backend does not need SAM2, so the online loop still works in tests
    and on machines that only have GroundingDINO installed.
    """

    name = "bbox_mask"

    def init_from_bbox(
        self,
        *,
        image: Image.Image,
        bbox_xyxy: BBoxXYXY,
        prompt: str,
        state: Any = None,
    ) -> tuple[np.ndarray, BBoxXYXY]:
        del prompt, state
        return bbox_to_mask(image.size, bbox_xyxy), bbox_xyxy

    def update(
        self,
        *,
        image: Image.Image,
        state: CameraTrackerState,
    ) -> tuple[np.ndarray | None, BBoxXYXY | None]:
        del image
        if state.mask is None:
            return None, None
        return state.mask, mask_to_bbox_xyxy(state.mask)


class SAM2BoxTrackerBackend:
    """Online SAM2 image predictor, prompted by the latest bbox."""

    name = "sam2_box"

    def __init__(self, config: dict[str, Any]) -> None:
        tracker_cfg = config.get("tracker", {})
        checkpoint = tracker_cfg.get("sam2_checkpoint")
        sam2_config = SAM2Config(
            model_cfg=str(tracker_cfg.get("sam2_model_cfg") or "configs/sam2.1/sam2.1_hiera_s.yaml"),
            checkpoint_path=Path(checkpoint) if checkpoint else SAM2Config().checkpoint_path,
            device=str(tracker_cfg.get("device") or "cuda"),
        )
        sam2_config.validate()

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(
            sam2_config.model_cfg,
            str(sam2_config.checkpoint_path),
            device=sam2_config.device,
            apply_postprocessing=sam2_config.apply_postprocessing,
        )
        self.predictor = SAM2ImagePredictor(model)

    def init_from_bbox(
        self,
        *,
        image: Image.Image,
        bbox_xyxy: BBoxXYXY,
        prompt: str,
        state: Any = None,
    ) -> tuple[np.ndarray, BBoxXYXY]:
        del prompt, state
        return self._predict_mask(image=image, bbox_xyxy=bbox_xyxy)

    def update(
        self,
        *,
        image: Image.Image,
        state: CameraTrackerState,
    ) -> tuple[np.ndarray | None, BBoxXYXY | None]:
        if state.bbox_xyxy is None:
            return None, None
        return self._predict_mask(image=image, bbox_xyxy=state.bbox_xyxy)

    def _predict_mask(self, *, image: Image.Image, bbox_xyxy: BBoxXYXY) -> tuple[np.ndarray, BBoxXYXY]:
        self.predictor.set_image(np.asarray(image.convert("RGB")))
        masks, _scores, _logits = self.predictor.predict(
            box=np.asarray(bbox_xyxy, dtype=np.float32),
            multimask_output=False,
        )
        mask = np.squeeze(masks[0]).astype(bool)
        return mask, mask_to_bbox_xyxy(mask) or bbox_xyxy


def backend_from_config(config: dict[str, Any]):
    backend = str(config["tracker"]["backend"]).lower()
    builders = {
        "bbox_mask": BBoxMaskTrackerBackend,
        "sam2": lambda: SAM2BoxTrackerBackend(config),
        "sam2_memory": lambda: _sam2_memory_backend(config),
    }
    return builders[backend]()


def _sam2_memory_backend(config: dict[str, Any]):
    from perception.tracking.sam2_memory import SAM2MemoryTrackerBackend

    return SAM2MemoryTrackerBackend(config)
