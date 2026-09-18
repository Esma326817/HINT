"""Online stage-aware multi-camera grounding and rendering runtime."""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from common.config_loader import DEFAULT_CONFIG_PATH, load_reasoning_config
from common.runtime_options import (
    SEMANTIC_INTENT_ATTENTION,
    injects_attention,
    injects_highlighting,
    semantic_intent_injection,
    skipped_stages,
)
from intent.highlighting import MaskRenderConfig, render_mask_overlay
from runtime.frame_pipeline import get_current_frame_index, prepare_frame, run_frame_pipeline
from runtime.render_img_output import save_render_images
from runtime.session_state import get_task_state
from common.step_profiler import PROFILER
from task.operations import current_visual_target
from intent.attention import resolve_model_input_resolution, resolve_output_resolution
from pattern.runtime.predictor import OnlineStagePredictor, StagePrediction, build_stage_predictor
from pattern.runtime.manipulation_pattern_router import ManipulationPatternRouter
from pattern.runtime.source import AnnotationStageSource, PredictStageSource, build_stage_source
from pattern.runtime.types import (
    ALL_CAMERAS,
    GLOBAL_CAMERA,
    CameraName,
    MultiCameraFrame,
    StageClassifierOutput,
    StageDecision,
)
from task import SubtaskManager, build_subtask_manager
from perception.tracking import TrackingManager
from perception.tracking.state import global_render_mode
from perception.task_manager.manager import resolve_target_phrase

_logger = logging.getLogger(__name__)


@dataclass
class StageAwareStepResult:
    rendered_images: dict[CameraName, Image.Image]
    decision: StageDecision
    prompt: str
    tracker_valid: dict[CameraName, bool]
    semantic_grounding: dict[CameraName, dict[str, Any] | None]
    frame_idx: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Runtime:
    config_path: Path | None = None
    config_mtime: float | None = None
    config: dict[str, Any] = field(default_factory=dict)
    router: ManipulationPatternRouter | None = None
    tracker_manager: TrackingManager | None = None
    render_config: MaskRenderConfig | None = None
    stage_source: AnnotationStageSource | PredictStageSource | None = None
    stage_predictor: OnlineStagePredictor | None = None
    subtask_manager: SubtaskManager | None = None
    last_task_state_id: int | None = None

    def ensure_loaded(self, config_path: str | Path | None = None) -> None:
        resolved_path = Path(config_path or os.getenv("REASONING_AGENT_CONFIG", DEFAULT_CONFIG_PATH))
        mtime = resolved_path.stat().st_mtime if resolved_path.exists() else None
        if self.config and self.config_path == resolved_path and self.config_mtime == mtime:
            return
        self.config = load_reasoning_config(resolved_path)
        self.router = ManipulationPatternRouter(self.config)
        self.tracker_manager = TrackingManager(self.config)
        self.render_config = MaskRenderConfig.from_config(self.config)
        self.stage_source = build_stage_source(self.config)
        # In predict mode with a configured checkpoint, the runtime classifies the
        # incoming frame itself (no externally supplied stage required).
        self.stage_predictor = build_stage_predictor(self.config)
        self.subtask_manager = build_subtask_manager(self.config)
        self.config_path = resolved_path
        self.config_mtime = mtime
        self.last_task_state_id = None

    def reset_for_task_if_needed(self, task_state: Any) -> None:
        task_state_id = id(task_state) if task_state is not None else None
        if self.last_task_state_id == task_state_id:
            return
        if self.router is not None:
            self.router.reset()
        if self.tracker_manager is not None:
            self.tracker_manager.reset()
        if self.stage_predictor is not None:
            self.stage_predictor.reset()
        if self.subtask_manager is not None:
            self.subtask_manager.reset()
        self.last_task_state_id = task_state_id


_runtime = _Runtime()


def reset_stage_aware_runtime() -> None:
    global _runtime
    _runtime = _Runtime()


def run_stage_aware_frame_pipeline(
    *,
    frames: MultiCameraFrame,
    robot_state: Any,
    stage_payload: Any = None,
    stage_id: int | None = None,
    low_dim: Any = None,
    low_dim_window: Any = None,
    task_instruction: str | None = None,
    model_input_resolution: tuple[int, int] | None = None,
    config_path: str | Path | None = None,
) -> StageAwareStepResult:
    _runtime.ensure_loaded(config_path)
    if (
        _runtime.router is None
        or _runtime.tracker_manager is None
        or _runtime.render_config is None
        or _runtime.stage_source is None
        or _runtime.subtask_manager is None
    ):
        raise RuntimeError("stage-aware runtime failed to initialize")

    images = {
        camera: frames.images[camera].convert("RGB")
        for camera in frames.available_cameras()
    }
    if "global" not in images:
        raise ValueError("stage-aware pipeline requires a global camera image")

    task_state = get_task_state()
    if task_state is None:
        raise RuntimeError("task_state is not initialized; call reset.run_reset_pipeline first")
    _runtime.reset_for_task_if_needed(task_state)
    semantic_intent_mode = semantic_intent_injection(_runtime.config)
    resolved_model_input_resolution = resolve_model_input_resolution(
        _runtime.config,
        model_input_resolution,
    )
    resolved_output_resolution = resolve_output_resolution(_runtime.config)

    needs_rule_render = (
        injects_highlighting(semantic_intent_mode)
        and global_render_mode(_runtime.config) != "block_mask"
    )
    rule_render = None
    if needs_rule_render:
        rule_render = run_frame_pipeline(
            images["global"],
            robot_state,
            save_debug=False,
        )
        frame_idx = get_current_frame_index()
    else:
        frame_idx, task_complete = prepare_frame(task_state)
        if task_complete:
            completed_decision = _runtime.router.decide(
                StageClassifierOutput.from_payload(
                    None,
                    default_stage=_runtime.router.registry.default_stage_name,
                    default_stage_id=_runtime.router.registry.default_stage_id,
                )
            )
            PROFILER.set_stage(
                completed_decision.confirmed_stage,
                completed_decision.stage_id,
            )
            return StageAwareStepResult(
                rendered_images=dict(images),
                decision=completed_decision,
                prompt="",
                tracker_valid=_runtime.tracker_manager.valid_flags(),
                semantic_grounding={camera: None for camera in ALL_CAMERAS},
                frame_idx=frame_idx,
                metadata={
                    "fallback": "task_complete",
                    "semantic_intent_injection": semantic_intent_mode,
                    "model_input_resolution": list(resolved_model_input_resolution),
                    "output_resolution": list(resolved_output_resolution),
                },
            )

    default_stage = _runtime.router.registry.default_stage_name
    default_stage_id = _runtime.router.registry.default_stage_id
    # Predict mode: when a manipulation-pattern classifier is loaded and no pattern was supplied
    # externally, run it on the live frame to obtain the raw stage_id. The
    # predictor keeps its own rolling window, so it must see every frame; when the
    # sender buffers proprioception at control rate it passes a dense
    # ``low_dim_window`` instead, which supersedes that buffer.
    prediction: StagePrediction | None = None
    if _runtime.stage_predictor is not None and stage_id is None and stage_payload is None:
        prediction = _runtime.stage_predictor.predict(
            images,
            robot_state if low_dim is None else low_dim,
            low_dim_window=low_dim_window,
        )
        stage_id = prediction.stage_id
    predict_confidence = prediction.confidence if prediction is not None else None
    # Resolve the per-frame stage from the configured source. In annotation mode
    # this maps the dataset's annotated stage_id; in predict mode the predicted
    # (or externally supplied) stage_id / payload passes through.
    resolved_payload = _runtime.stage_source.resolve_payload(
        stage_id=stage_id,
        external_payload=stage_payload,
    )
    if resolved_payload is None:
        resolved_payload = stage_payload
    # Confidence and progress ride along with the stage so the router can gate
    # transitions on how far the current and candidate stages are.
    if prediction is not None and isinstance(resolved_payload, dict):
        resolved_payload = {**resolved_payload, **prediction.to_payload()}
    stage_output = StageClassifierOutput.from_payload(
        resolved_payload,
        default_stage=default_stage,
        default_stage_id=default_stage_id,
    )
    decision = _runtime.router.decide(stage_output)
    PROFILER.set_stage(decision.confirmed_stage, decision.stage_id)
    stage_skipped = decision.confirmed_stage in skipped_stages(_runtime.config)
    output_cameras = () if stage_skipped else decision.route.cameras
    # Schedule the active subtask from the confirmed pattern *before* resolving the
    # prompt, so a contact -> free transition advances to the next subtask and the
    # prompt reflects it immediately. Within one subtask the target stays locked,
    # keeping pre_contact consistent with the free_move choice.
    subtask_advanced = _runtime.subtask_manager.observe(
        decision=decision,
        task_state=task_state,
    )
    active_subtask = _runtime.subtask_manager.current_subtask(task_state)
    prompt = resolve_target_phrase(
        decision=decision,
        images=images,
        task_state=task_state,
        task_instruction=task_instruction,
        config=_runtime.config,
    )
    confirmed_progress = decision.progress
    progress_text = "n/a" if confirmed_progress is None else f"{confirmed_progress:.3f}"
    print(
        f"stage={decision.confirmed_stage} (id={decision.stage_id}) "
        f"progress={progress_text} | prompt={prompt!r}",
        flush=True,
    )
    subtask_metadata = {
        "subtask_advanced": subtask_advanced,
        "active_subtask": active_subtask.description if active_subtask else None,
        "active_label": active_subtask.label if active_subtask else None,
        "active_target": (
            current_visual_target(task_state, decision.confirmed_stage)
            if task_state is not None
            else None
        ),
        "progress_idx": task_state.progress_idx if task_state is not None else None,
        "target_word": task_state.target_word if task_state is not None else None,
        "predicted_stage_id": stage_id if predict_confidence is not None else None,
        "predict_confidence": predict_confidence,
        "predict_progress": prediction.progress if prediction is not None else None,
        "confirmed_progress": confirmed_progress,
    }

    with torch.inference_mode():
        if torch.cuda.is_available():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _runtime.tracker_manager.update_existing(
                    images,
                    active_cameras=output_cameras,
                )
                _runtime.tracker_manager.apply_stage_decision(
                    frame_idx=frame_idx,
                    images=images,
                    decision=decision,
                    prompt=prompt,
                    target_letter=current_visual_target(task_state, decision.confirmed_stage)
                    if task_state is not None
                    else None,
                    task_state=task_state,
                )
        else:
            _runtime.tracker_manager.update_existing(
                images,
                active_cameras=output_cameras,
            )
            _runtime.tracker_manager.apply_stage_decision(
                frame_idx=frame_idx,
                images=images,
                decision=decision,
                prompt=prompt,
                target_letter=current_visual_target(task_state, decision.confirmed_stage),
                task_state=task_state,
            )

    render_cfg = _runtime.config["render"]
    # Highlighting (and both) feeds overlays into the policy; attention alone returns
    # raw RGB and injects semantic_grounding instead. debug_overlay only paints for
    # disk debug, so it is redundant once the policy render is already on.
    policy_render_enabled = (
        injects_highlighting(semantic_intent_mode)
        and bool(render_cfg["enabled"])
    )
    with PROFILER.section("render"):
        rendered = render_stage_images(
            images=images,
            active_cameras=output_cameras,
            tracker_manager=_runtime.tracker_manager,
            render_config=_runtime.render_config,
            render_enabled=policy_render_enabled,
            dropout_probability=float(render_cfg.get("dropout_probability", 0.0)),
        )
    if (
        injects_highlighting(semantic_intent_mode)
        and global_render_mode(_runtime.config) != "block_mask"
    ):
        # Legacy pick_place_fill: global uses the known target-block / placement fill.
        rendered[GLOBAL_CAMERA] = rule_render
    debug_images = rendered
    if (
        semantic_intent_mode == SEMANTIC_INTENT_ATTENTION
        and bool(render_cfg.get("debug_overlay", False))
    ):
        debug_images = render_stage_images(
            images=images,
            active_cameras=output_cameras,
            tracker_manager=_runtime.tracker_manager,
            render_config=_runtime.render_config,
            render_enabled=True,
            dropout_probability=0.0,
        )
    _save_stage_outputs(
        debug_images,
        frame_idx,
        stage=decision.confirmed_stage,
        prompt=prompt,
    )
    semantic_grounding = (
        _runtime.tracker_manager.current_semantic_grounding(
            images=images,
            active_cameras=output_cameras,
            model_input_resolution=resolved_model_input_resolution,
        )
        if injects_attention(semantic_intent_mode)
        else {camera: None for camera in ALL_CAMERAS}
    )
    return StageAwareStepResult(
        rendered_images=rendered,
        decision=decision,
        prompt=prompt,
        tracker_valid=_runtime.tracker_manager.valid_flags(),
        semantic_grounding=semantic_grounding,
        frame_idx=frame_idx,
        metadata={
            "fallback": None,
            "semantic_intent_injection": semantic_intent_mode,
            "semantic_intent_stage_skipped": stage_skipped,
            "model_input_resolution": list(resolved_model_input_resolution),
            "output_resolution": list(resolved_output_resolution),
            **subtask_metadata,
        },
    )


def render_stage_images(
    *,
    images: dict[CameraName, Image.Image],
    active_cameras: tuple[CameraName, ...],
    tracker_manager: TrackingManager,
    render_config: MaskRenderConfig,
    render_enabled: bool,
    dropout_probability: float,
) -> dict[CameraName, Image.Image]:
    if not render_enabled:
        return dict(images)

    use_dropout = dropout_probability > 0.0 and random.random() < dropout_probability
    if use_dropout:
        return dict(images)

    rendered: dict[CameraName, Image.Image] = dict(images)
    for camera in active_cameras:
        image = images.get(camera)
        if image is None:
            continue
        mask = tracker_manager.valid_mask(camera)
        if mask is None:
            continue
        rendered[camera] = render_mask_overlay(image, mask, render_config)
    return rendered


def _save_stage_outputs(
    rendered: dict[CameraName, Image.Image],
    frame_idx: int,
    *,
    stage: str | None = None,
    prompt: str | None = None,
) -> None:
    save_render_images(rendered, frame_idx, stage=stage, prompt=prompt)
