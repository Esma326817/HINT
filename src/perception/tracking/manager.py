"""Online SAM2 tracking loop.

The manager owns per-camera SAM2/bbox-mask state. Bounding-box *selection*
lives in ``perception.semantic_grounder``; this module only decides
when to ask the grounder and then propagates the chosen box.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
from PIL import Image

from common.step_profiler import PROFILER
from pattern.runtime.types import (
    ALL_CAMERAS,
    FREE_MOVE_STAGE,
    GLOBAL_CAMERA,
    CameraName,
    StageDecision,
)
from intent.attention import (
    DEFAULT_MODEL_INPUT_RESOLUTION,
    patch_attention_payload,
)
from perception.semantic_grounder.dino import DinoClient
from perception.semantic_grounder.selection import (
    free_move_grounding_delay_frames,
    resolve_grounding_selection,
    should_defer_free_move_grounding,
    transport_area_placement_enabled,
)
from perception.tracking.backends import backend_from_config
from perception.tracking.sam2_video import mask_to_bbox_xyxy
from perception.tracking.state import (
    AREA_PLACEMENT_RENDER_MODE,
    WRIST_CAMERAS,
    CameraTrackerState,
    bbox_to_mask,
    empty_camera_states,
    global_render_mode,
)

DEFAULT_RECOGNITION_PADDING = 4

_logger = logging.getLogger(__name__)


class TrackingManager:
    """Per-camera SAM2/bbox-mask state, with grounding calls when a track is lost."""

    def __init__(self, config: dict[str, Any], dino_client: DinoClient | None = None) -> None:
        self.config = config
        grounding_cfg = config["grounding"]
        self.dino_client = dino_client or DinoClient(config)
        self.backend = backend_from_config(config)
        # Per-frame grounder after reset: "dino" (GroundingDINO + optional Qwen
        # verify) or "qwen" (Qwen3-VL native box from the task prompt, DINO as
        # fallback).
        # Ground every step: re-run the grounder on each primary camera every frame
        # to draw a fresh, exact box, instead of letting SAM2 propagate the previous
        # mask between sparse (~0.5-1s) inference steps. Online the wrist moves
        # continuously, so propagation drifts/lags and distorts the mask; a fresh
        # grounding per frame keeps the box locked on the target. Costs one grounder
        # call per primary camera per step.
        self.ground_every_step = bool(grounding_cfg.get("ground_every_step", False))
        self.states = empty_camera_states()
        # Frames spent in the current free_move segment (0 on entry). Used to defer
        # grounding/VLM until the arm clears the global view after transport.
        self._free_move_frames_in_stage = 0
        self._free_move_grounding_pending = False

    def reset(self) -> None:
        self.states = empty_camera_states()
        self._free_move_frames_in_stage = 0
        self._free_move_grounding_pending = False

    def _free_move_grounding_delay_frames(self) -> int:
        return free_move_grounding_delay_frames(self.config)

    def _defer_free_move_grounding(self, decision: StageDecision) -> bool:
        return should_defer_free_move_grounding(
            config=self.config,
            stage_name=decision.confirmed_stage,
            frames_in_stage=self._free_move_frames_in_stage,
        )

    def _advance_free_move_frame_counter(self, decision: StageDecision) -> None:
        if decision.confirmed_stage != FREE_MOVE_STAGE:
            self._free_move_frames_in_stage = 0
            return
        if decision.stage_changed:
            self._free_move_frames_in_stage = 0
        self._free_move_frames_in_stage += 1

    @staticmethod
    def _freeze_global_state(state: CameraTrackerState, reason: str) -> None:
        """Keep rendering the last valid global mask after tracking is lost."""
        state.valid = state.mask is not None and state.bbox_xyxy is not None
        state.lost_count += 1
        state.metadata = {
            **state.metadata,
            "tracking_frozen": True,
            "tracking_lost_reason": reason,
        }

    def update_existing(
        self,
        images: dict[CameraName, Image.Image],
        *,
        active_cameras: Iterable[CameraName],
    ) -> None:
        """Propagate tracker state only for cameras routed by the current stage.

        Inactive camera states are deliberately left untouched. Stage-transition
        logic remains responsible for re-grounding or replacing them when they
        become active again.
        """
        min_area_ratio = float(self.config["grounding"]["min_mask_area_ratio"])
        active = frozenset(active_cameras)
        for camera, state in self.states.items():
            if camera not in active or not state.valid:
                continue
            image = images.get(camera)
            # A placement area describes a fixed image-space destination, which
            # may contain no segmentable object. Re-rasterize its original bbox
            # on every frame and never pass it through SAM2.
            if state.metadata.get("render_mode") == AREA_PLACEMENT_RENDER_MODE:
                if image is None or state.bbox_xyxy is None:
                    state.invalidate()
                    continue
                state.mask = bbox_to_mask(image.size, state.bbox_xyxy)
                state.valid = self._mask_is_valid(state.mask, image.size, min_area_ratio)
                if not state.valid:
                    state.invalidate()
                else:
                    state.lost_count = 0
                continue
            # Once global tracking is lost, keep the last trustworthy mask/bbox
            # fixed instead of letting an occluder pull the tracker elsewhere.
            # A stage/prompt change will explicitly ground_camera() again and replace
            # this state with the new visual target.
            if camera == GLOBAL_CAMERA and state.metadata.get("tracking_frozen"):
                state.lost_count += 1
                continue
            # Wrist cameras are re-grounded from scratch every step in
            # apply_stage_decision when ground_every_step is on, so propagating the
            # previous (drifting) mask here is wasted work. Global keeps propagating:
            # the table target is ~static, so the previous box is still good.
            if self.ground_every_step and camera in WRIST_CAMERAS:
                continue
            if image is None:
                if camera == GLOBAL_CAMERA:
                    self._freeze_global_state(state, "missing_image")
                else:
                    state.invalidate()
                continue
            with PROFILER.section(f"sam/update/{camera}", cuda=True):
                mask, bbox = self.backend.update(image=image, state=state)
            if mask is None or bbox is None or not self._mask_is_valid(mask, image.size, min_area_ratio):
                if camera == GLOBAL_CAMERA:
                    self._freeze_global_state(state, "invalid_tracking_update")
                else:
                    state.invalidate()
                continue
            state.mask = mask
            state.bbox_xyxy = bbox
            state.valid = True
            state.lost_count = 0
            state.metadata.pop("tracking_frozen", None)
            state.metadata.pop("tracking_lost_reason", None)

    def apply_stage_decision(
        self,
        *,
        frame_idx: int,
        images: dict[CameraName, Image.Image],
        decision: StageDecision,
        prompt: str,
        target_letter: str | None = None,
        verify_target: str | None = None,
        letter_padding: int = DEFAULT_RECOGNITION_PADDING,
        task_state: Any = None,
    ) -> None:
        # verify_target is preferred; target_letter kept as a compatibility alias.
        if verify_target is None:
            verify_target = target_letter
        target_letter = verify_target
        # Give task-specific logic one chance to preserve the *previous* global
        # track before stage-entry grounding replaces it. Peg-in-hole uses this
        # only after block placement, when the slot track has converged onto the
        # colored block now seated in the slot.
        if task_state is not None and decision.stage_changed:
            from task import get_task_handler

            handler = get_task_handler(
                getattr(task_state, "task_name", None), self.config
            )
            capture_parent = getattr(handler, "capture_tracker_parent_bbox", None)
            global_image = images.get(GLOBAL_CAMERA)
            if callable(capture_parent) and global_image is not None:
                capture_parent(
                    task_state,
                    decision=decision,
                    tracker_state=self.states[GLOBAL_CAMERA],
                    image=global_image,
                )
        grounding_cfg = self.config["grounding"]
        block_mask_global = global_render_mode(self.config) == "block_mask"
        if block_mask_global:
            # Match offline block_mask: global tracks the object while the hand is
            # empty and the destination once it is held, which is exactly the route.
            primary_cameras = set(decision.route.cameras)
        else:
            # Legacy pick_place_fill: global uses bbox fill, wrists use SAM2 masks.
            primary_cameras = set(decision.route.cameras) - {GLOBAL_CAMERA}
        prompt_changed = any(
            self.states[camera].prompt is not None and self.states[camera].prompt != prompt
            for camera in primary_cameras
        )
        defer_free_move_grounding = self._defer_free_move_grounding(decision)
        if decision.stage_changed and decision.confirmed_stage == FREE_MOVE_STAGE:
            self._free_move_grounding_pending = True
        if decision.confirmed_stage != FREE_MOVE_STAGE:
            self._free_move_grounding_pending = False
        if defer_free_move_grounding:
            delay = self._free_move_grounding_delay_frames()
            _logger.info(
                "defer free_move grounding/VLM: frame %d/%d after stage entry (arm clearing view)",
                self._free_move_frames_in_stage,
                delay,
            )

        for camera in ALL_CAMERAS:
            state = self.states[camera]
            if camera not in primary_cameras:
                if bool(grounding_cfg.get("invalidate_non_primary_on_lost", True)) and not state.valid:
                    state.invalidate()
                continue

            image = images.get(camera)
            if image is None:
                if camera == GLOBAL_CAMERA and state.valid:
                    self._freeze_global_state(state, "missing_image")
                else:
                    state.invalidate()
                continue

            # Ground every step (wrist only): the wrist target moves continuously, so
            # draw a fresh box each frame instead of relying on the propagated mask.
            # Global is left to the needs_grounding path below (its target is static).
            if self.ground_every_step and camera in WRIST_CAMERAS:
                if defer_free_move_grounding:
                    continue
                self.ground_camera(
                    camera=camera,
                    image=image,
                    prompt=prompt,
                    frame_idx=frame_idx,
                    stage_name=decision.confirmed_stage,
                    target_letter=target_letter,
                    letter_padding=letter_padding,
                    task_state=task_state,
                )
                continue

            needs_grounding = (
                decision.stage_changed
                or not state.valid
                or (
                    bool(grounding_cfg.get("reground_on_prompt_change", True))
                    and state.prompt is not None
                    and state.prompt != prompt
                )
                or prompt_changed
            )
            if (
                self._free_move_grounding_pending
                and not defer_free_move_grounding
                and camera in primary_cameras
            ):
                needs_grounding = True
            if not needs_grounding:
                continue
            if defer_free_move_grounding:
                continue

            if not state.valid or decision.stage_changed or prompt_changed:
                # Global (block_mask, free_move) re-acquires from the known reset
                # bbox without a VLM call when SAM2 merely lost the (static) target
                # mid-stage — e.g. the gripper occludes it on approach. A fresh Qwen
                # call only fires on stage entry / target-letter change, so transient
                # mask drops don't trigger a re-grounding storm.
                allow_vlm = (
                    camera != GLOBAL_CAMERA
                    or decision.stage_changed
                    or prompt_changed
                    or self._free_move_grounding_pending
                )
                grounded = self.ground_camera(
                    camera=camera,
                    image=image,
                    prompt=prompt,
                    frame_idx=frame_idx,
                    stage_name=decision.confirmed_stage,
                    target_letter=target_letter,
                    letter_padding=letter_padding,
                    task_state=task_state,
                    allow_vlm=allow_vlm,
                )
                if self._free_move_grounding_pending and grounded and camera in primary_cameras:
                    self._free_move_grounding_pending = False

        self._advance_free_move_frame_counter(decision)

    def ground_camera(
        self,
        *,
        camera: CameraName,
        image: Image.Image,
        prompt: str,
        frame_idx: int,
        stage_name: str | None = None,
        target_letter: str | None = None,
        verify_target: str | None = None,
        letter_padding: int = DEFAULT_RECOGNITION_PADDING,
        task_state: Any = None,
        allow_vlm: bool = True,
    ) -> bool:
        if verify_target is None:
            verify_target = target_letter
        target_letter = verify_target
        state = self.states[camera]
        grounding_cfg = self.config["grounding"]
        selection = resolve_grounding_selection(
            camera=camera,
            image=image,
            prompt=prompt,
            task_state=task_state,
            stage_name=stage_name,
            config=self.config,
            dino_client=self.dino_client,
            verify_target=target_letter,
            recognition_padding=letter_padding,
            allow_vlm=allow_vlm,
        )
        PROFILER.mark_grounded(camera, selection.source)
        bbox = selection.bbox_xyxy
        if bbox is None:
            state.invalidate()
            state.prompt = prompt
            state.last_grounded_frame_idx = frame_idx
            state.metadata = {"grounding_source": selection.source}
            return False
        render_mode = self._resolve_render_mode(
            task_state=task_state,
            camera=camera,
            stage_name=stage_name,
            prompt=prompt,
        )
        if render_mode == AREA_PLACEMENT_RENDER_MODE:
            # The bbox itself is the intended visual cue. SAM2 would instead
            # segment a surrounding surface (for letter, usually the full board).
            mask, bbox_xyxy = bbox_to_mask(image.size, bbox), bbox
        else:
            with PROFILER.section(f"sam/init/{camera}", cuda=True):
                mask, bbox_xyxy = self.backend.init_from_bbox(
                    image=image,
                    bbox_xyxy=bbox,
                    prompt=prompt,
                    state=state,
                )
        if not self._mask_is_valid(
            mask,
            image.size,
            float(grounding_cfg["min_mask_area_ratio"]),
        ):
            state.invalidate()
            state.prompt = prompt
            state.last_grounded_frame_idx = frame_idx
            return False

        state.valid = True
        state.prompt = prompt
        state.mask = mask
        state.bbox_xyxy = bbox_xyxy
        state.score = (
            float(selection.detection["confidence"])
            if selection.detection is not None
            and selection.detection.get("confidence") is not None
            else None
        )
        state.last_grounded_frame_idx = frame_idx
        state.lost_count = 0
        state.metadata = {
            "detection": selection.detection,
            "backend": (
                AREA_PLACEMENT_RENDER_MODE
                if render_mode == AREA_PLACEMENT_RENDER_MODE
                else self.backend.name
            ),
            "render_mode": render_mode,
            "grounding_source": selection.source,
            "verified_label": selection.matched_label,
            "verified_category": selection.matched_category,
            "verification_mode": selection.verification_mode,
        }
        return True

    def _resolve_render_mode(
        self,
        *,
        task_state: Any,
        camera: CameraName,
        stage_name: str | None,
        prompt: str,
    ) -> str:
        if transport_area_placement_enabled(
            config=self.config,
            task_state=task_state,
            camera=camera,
            stage_name=stage_name,
        ):
            return AREA_PLACEMENT_RENDER_MODE
        if task_state is None:
            return "sam2"
        from task import get_task_handler

        handler = get_task_handler(getattr(task_state, "task_name", None), self.config)
        hook = getattr(handler, "resolve_segment_render_mode", None)
        if not callable(hook):
            return "sam2"
        mode = str(
            hook(
                task_state,
                camera=camera,
                stage_name=stage_name or "",
                prompt=prompt,
            )
            or "sam2"
        ).lower()
        if mode not in {"sam2", "static_shape", AREA_PLACEMENT_RENDER_MODE}:
            raise ValueError(f"unsupported task render mode: {mode}")
        return mode

    @staticmethod
    def _mask_is_valid(mask: np.ndarray, size: tuple[int, int], min_area_ratio: float) -> bool:
        width, height = size
        area = int(np.count_nonzero(np.asarray(mask).astype(bool)))
        return area >= max(1, int(width * height * min_area_ratio))

    def valid_mask(self, camera: CameraName) -> np.ndarray | None:
        state = self.states.get(camera)
        if state is None or not state.valid:
            return None
        return state.mask

    def valid_flags(self) -> dict[CameraName, bool]:
        return {camera: state.valid for camera, state in self.states.items()}

    def current_semantic_grounding(
        self,
        *,
        images: dict[CameraName, Image.Image],
        active_cameras: tuple[CameraName, ...],
        model_input_resolution: tuple[int, int] = DEFAULT_MODEL_INPUT_RESOLUTION,
    ) -> dict[CameraName, dict[str, Any] | None]:
        """Return model-space ViT patch weights for the current frame.

        SAM2 remains the temporal tracker. The current mask is reduced to its
        source-pixel bbox here; ReasoningAgent then applies the same direct resize
        geometry as the policy RGB image and patch-pools before returning it.
        """
        active = set(active_cameras)
        result: dict[CameraName, dict[str, Any] | None] = {
            camera: None for camera in ALL_CAMERAS
        }
        for camera in ALL_CAMERAS:
            image = images.get(camera)
            state = self.states.get(camera)
            if image is None or state is None:
                continue

            width, height = image.size
            if width <= 0 or height <= 0:
                continue
            bbox = None
            if camera in active and state.valid and state.mask is not None:
                if state.metadata.get("render_mode") == AREA_PLACEMENT_RENDER_MODE:
                    bbox = state.bbox_xyxy
                else:
                    inclusive_bbox = mask_to_bbox_xyxy(state.mask)
                    if inclusive_bbox is not None:
                        bbox = (
                            inclusive_bbox[0],
                            inclusive_bbox[1],
                            inclusive_bbox[2] + 1.0,
                            inclusive_bbox[3] + 1.0,
                        )
                        state.bbox_xyxy = bbox

            result[camera] = patch_attention_payload(
                bbox,
                source_size=(width, height),
                model_input_resolution=model_input_resolution,
                grounding_method=str(state.metadata.get("backend") or self.backend.name),
                temporal_age=state.lost_count,
            )
        return result
