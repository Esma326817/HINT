"""Streaming SAM2 *video-predictor* tracker (memory propagation).

The default ``SAM2BoxTrackerBackend`` is stateless: every frame it re-runs the
image predictor with the previous bbox as a fresh prompt. It has no temporal
memory, so on occlusion / motion (e.g. the gripper covering the global target on
approach) the mask collapses, the camera state is invalidated, and grounding
(VLM) re-fires — a re-grounding storm.

This backend instead drives ``SAM2VideoPredictor`` in a streaming fashion: the
box seeds frame 0, and each subsequent live frame is tracked with
``_run_single_frame_inference(run_mem_encoder=True)`` using the accumulated
per-frame memory (maskmem features + object pointers). That is far more robust to
occlusion, so the mask stays valid and grounding stops re-firing.

Same interface as ``SAM2BoxTrackerBackend`` (``init_from_bbox`` / ``update``) plus
a ``state`` arg so per-camera memory streams are kept independent. Enable with
``tracker.backend: sam2_memory``.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import numpy as np
import torch
from PIL import Image

from perception.tracking.sam2_video import BBoxXYXY, mask_to_bbox_xyxy

_logger = logging.getLogger(__name__)


class SAM2MemoryTrackerBackend:
    """SAM2 video-predictor streaming tracker with per-camera memory."""

    name = "sam2_memory"

    def __init__(self, config: dict[str, Any]) -> None:
        tracker_cfg = config.get("tracker", {})
        checkpoint = tracker_cfg.get("sam2_checkpoint")
        self.device = str(tracker_cfg.get("device") or "cuda")
        from sam2.build_sam import build_sam2_video_predictor

        self.predictor = build_sam2_video_predictor(
            str(tracker_cfg.get("sam2_model_cfg") or "configs/sam2.1/sam2.1_hiera_s.yaml"),
            str(checkpoint),
            device=self.device,
        )
        self.image_size = int(self.predictor.image_size)
        # Match load_video_frames normalization (ImageNet mean/std).
        self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        self._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
        # How many recent frames to keep in memory / image buffer (bounds GPU use;
        # SAM2 only attends to the last ~num_maskmem memories anyway).
        self._keep = max(8, int(tracker_cfg.get("memory_frames", 24)))
        self._states: dict[str, dict] = {}  # camera -> inference_state

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        arr = np.array(image.convert("RGB").resize((self.image_size, self.image_size)))
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return (t - self._mean) / self._std

    def _mask_from_logits(self, inf: dict, pred_masks: torch.Tensor) -> np.ndarray:
        _, video_res = self.predictor._get_orig_video_res_output(
            inf, pred_masks.to(self.device)
        )
        return (video_res[0, 0] > 0.0).cpu().numpy()

    @torch.inference_mode()
    def init_from_bbox(
        self,
        *,
        image: Image.Image,
        bbox_xyxy: BBoxXYXY,
        prompt: str,
        state: Any = None,
    ) -> tuple[np.ndarray | None, BBoxXYXY | None]:
        del prompt
        cam = str(getattr(state, "camera", "default"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with tempfile.TemporaryDirectory() as d:
                image.convert("RGB").save(os.path.join(d, "00000.jpg"), quality=95)
                inf = self.predictor.init_state(video_path=d)
            # Replace the stacked image tensor with a small rolling dict so the buffer
            # does not grow ~12 MB/frame. _get_image_feature does images[frame_idx].
            inf["images"] = {0: inf["images"][0]}
            inf["_img_dtype"] = inf["images"][0].dtype
            box = np.asarray(bbox_xyxy, dtype=np.float32)
            self.predictor.add_new_points_or_box(
                inference_state=inf, frame_idx=0, obj_id=1, box=box
            )
            self.predictor.propagate_in_video_preflight(inf)
            cond = inf["output_dict_per_obj"][0]["cond_frame_outputs"][0]
            mask = self._mask_from_logits(inf, cond["pred_masks"])
        self._states[cam] = inf
        return mask, (mask_to_bbox_xyxy(mask) or tuple(float(v) for v in bbox_xyxy))

    @torch.inference_mode()
    def update(
        self,
        *,
        image: Image.Image,
        state: Any,
    ) -> tuple[np.ndarray | None, BBoxXYXY | None]:
        cam = str(getattr(state, "camera", "default"))
        inf = self._states.get(cam)
        if inf is None:
            return None, None
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                frame_idx = int(inf["num_frames"])
                inf["images"][frame_idx] = self._preprocess(image).to(inf["_img_dtype"])
                inf["num_frames"] = frame_idx + 1
                obj_out = inf["output_dict_per_obj"][0]
                current_out, pred_masks = self.predictor._run_single_frame_inference(
                    inference_state=inf,
                    output_dict=obj_out,
                    frame_idx=frame_idx,
                    batch_size=1,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=False,
                    run_mem_encoder=True,
                )
                obj_out["non_cond_frame_outputs"][frame_idx] = current_out
                inf.setdefault("frames_tracked_per_obj", {}).setdefault(0, {})[frame_idx] = {
                    "reverse": False
                }
                mask = self._mask_from_logits(inf, pred_masks)
            self._prune(inf, frame_idx)
        except Exception:
            _logger.exception("sam2_memory update failed for camera %s; dropping stream", cam)
            self._states.pop(cam, None)
            return None, None
        return mask, mask_to_bbox_xyxy(mask)

    def _prune(self, inf: dict, frame_idx: int) -> None:
        """Bound GPU memory: drop image frames and non-cond memory beyond the window."""
        cutoff = frame_idx - self._keep
        if cutoff < 0:
            return
        images = inf["images"]
        for i in [k for k in images if isinstance(k, int) and k <= cutoff]:
            images.pop(i, None)
        nc = inf["output_dict_per_obj"][0]["non_cond_frame_outputs"]
        for i in [k for k in nc if k <= cutoff]:
            nc.pop(i, None)
