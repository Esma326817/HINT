"""FastAPI service: /reset, /step, /health.

Run from project root::

    python -m inference.api_server
"""

from __future__ import annotations

import base64
import json
import logging
import os
from io import BytesIO
from typing import Literal

from common.logging_setup import configure_project_logging

configure_project_logging()

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from perception.semantic_grounder.dino import DinoClient
from runtime.reset_pipeline import run_reset_pipeline
from common.step_profiler import PROFILER
from common.task_types import TaskContext
from runtime.frame_pipeline import run_frame_pipeline
from runtime.pipeline import reset_stage_aware_runtime, run_stage_aware_frame_pipeline
from pattern.runtime.types import MultiCameraFrame
from common.config_loader import load_reasoning_config
from perception.task_manager.qwen_runtime import is_model_ready, initialize_qwen_runtime_from_config
from runtime.background import RUNTIME_LOCK
from intent.attention import (
    normalize_model_input_resolution,
    resize_policy_image,
    resolve_output_resolution,
)
from task.base import plan_from_prompt_enabled
from task import get_task_handler

router = APIRouter()

# Set in create_app when deploy.infer_mode == "background": a thread that tracks
# the active-stage camera off the dagger image bus between sparse /step calls.
_bg_tracker = None


class SemanticGroundingResponse(BaseModel):
    spatial_encoding: Literal["vit_patch_attention"] = "vit_patch_attention"
    attention_map: list[list[float]]
    grid_size: tuple[int, int]
    model_input_resolution: tuple[int, int]
    patch_size: tuple[int, int]
    model_bbox_xyxy: tuple[float, float, float, float] | None = None
    valid: bool
    source_size: tuple[int, int]
    grounding_method: str
    temporal_age: int = 0


class StepResponse(BaseModel):
    rendered_image: str = Field(
        ...,
        description="Base64 encoded PNG bytes, resized to task.output_resolution (typically 224x224).",
    )
    rendered_images: dict[str, str] | None = Field(
        default=None,
        description=(
            "Optional per-camera base64 encoded PNG bytes for stage-aware multi-camera calls. "
            "Always resized to task.output_resolution before encode."
        ),
    )
    stage: str | None = None
    prompt: str | None = None
    tracker_valid: dict[str, bool] | None = None
    frame_id: int | None = None
    semantic_grounding: dict[str, SemanticGroundingResponse | None] | None = None
    semantic_intent_injection: str | None = Field(
        default=None,
        description=(
            "How this frame injects semantic intent: 'highlighting' and 'both' return "
            "painted RGB, 'attention' and 'both' populate semantic_grounding."
        ),
    )


class ResetResponse(BaseModel):
    status: str = Field(default="ok")


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    model_ready: bool
    dino_healthy: bool


def _load_image(file: UploadFile) -> Image.Image:
    try:
        payload = file.file.read()
        return Image.open(BytesIO(payload)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid image payload: {exc}") from exc



def _parse_robot_state(robot_state: str) -> list[float]:
    try:
        payload = json.loads(robot_state)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"robot_state must be valid JSON: {exc}") from exc

    if not isinstance(payload, list):
        raise HTTPException(status_code=400, detail="robot_state must decode to a list")
    if len(payload) != 14:
        raise HTTPException(status_code=400, detail="robot_state must contain exactly 14 floats")

    try:
        return [float(value) for value in payload]
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"robot_state must contain numeric values: {exc}") from exc


def _parse_float_list(name: str, raw: str | None, expected_len: int | None = None) -> list[float] | None:
    if raw is None or not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"{name} must be valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise HTTPException(status_code=400, detail=f"{name} must decode to a list")
    if expected_len is not None and len(payload) != expected_len:
        raise HTTPException(status_code=400, detail=f"{name} must contain exactly {expected_len} floats")
    try:
        return [float(value) for value in payload]
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{name} must contain numeric values: {exc}") from exc


def _resolve_low_dim(
    robot_state: list[float],
    effort: str | None,
    low_dim: str | None,
) -> list[float] | None:
    """Build the predictor's low_dim vector (state ++ effort).

    Preference order: explicit ``low_dim`` (already concatenated) > ``state ++
    effort`` > None (caller lets the predictor zero-pad effort).
    """
    parsed_low_dim = _parse_float_list("low_dim", low_dim)
    if parsed_low_dim is not None:
        return parsed_low_dim
    parsed_effort = _parse_float_list("effort", effort, expected_len=14)
    if parsed_effort is not None:
        return list(robot_state) + parsed_effort
    return None


def _parse_low_dim_window(raw: str | None) -> list[list[float]] | None:
    """Dense ``[low_history, state ++ effort]`` window (past → current), as sent.

    The sender owns the window contract: it buffers at control rate, subsamples to
    the training FPS and pads the episode start, so it arrives ready for the
    predictor. ``None`` means the predictor falls back to its own rolling buffer.
    """
    if raw is None or not raw.strip():
        return None
    return json.loads(raw)


def _parse_stage_payload(stage: str | None, stage_output: str | None) -> str | dict | None:
    if stage_output and stage_output.strip():
        try:
            payload = json.loads(stage_output)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"stage_output must be valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="stage_output must decode to an object")
        return payload
    if stage and stage.strip():
        return stage.strip()
    return None


def _resolve_episode_prompt(
    prompt: str | None,
    task_instruction: str | None,
) -> str | None:
    """Unify ``prompt`` / ``task_instruction`` form fields into one episode string.

    Accepted and stored for scene handlers that need an episode plan (e.g.
    peg_in_hole color/shape). Not interpreted by a system-wide prompt pipeline.
    """
    prompt_text = str(prompt or "").strip()
    instruction_text = str(task_instruction or "").strip()
    if prompt_text and instruction_text and prompt_text != instruction_text:
        raise HTTPException(
            status_code=400,
            detail="prompt conflicts with task_instruction",
        )
    return prompt_text or instruction_text or None


def _parse_task_context(
    raw_context: str | None,
    raw_instruction: str | None,
) -> TaskContext | None:
    instruction = str(raw_instruction or "").strip()
    payload: dict = {}
    if raw_context and raw_context.strip():
        try:
            parsed = json.loads(raw_context)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"task_context must be valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail="task_context must decode to an object")
        context_instruction = str(parsed.pop("instruction", "") or "").strip()
        if instruction and context_instruction and instruction != context_instruction:
            raise HTTPException(
                status_code=400,
                detail="task_context.instruction conflicts with task_instruction/prompt",
            )
        instruction = context_instruction or instruction
        payload = parsed
    if not instruction and not payload:
        return None
    if not instruction:
        raise HTTPException(status_code=400, detail="task context requires a non-empty instruction")
    return TaskContext(instruction=instruction, payload=payload, source="api")


# Wire format for /step rendered images. Frames are resized to task.output_resolution
# (default 224x224) before encode. PIL PNG encode of a 640x480 frame is
# ~46 ms; JPEG q95 is ~0.8 ms (≈60x faster) and the policy is trained on AV1-
# compressed (lossy) frames anyway, so JPEG is in-distribution. Override with
# REASONING_WIRE_FORMAT=PNG to restore lossless encoding.
_WIRE_FORMAT = os.getenv("REASONING_WIRE_FORMAT", "JPEG").upper()
_WIRE_QUALITY = int(os.getenv("REASONING_WIRE_QUALITY", "95"))


def _encode_image(image: Image.Image) -> str:
    buffer = BytesIO()
    if _WIRE_FORMAT in ("JPEG", "JPG"):
        image.save(buffer, format="JPEG", quality=_WIRE_QUALITY)
    else:
        image.save(buffer, format=_WIRE_FORMAT)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@router.post("/step", response_model=StepResponse)
@PROFILER.profile_frame
def step_route(
    image: UploadFile | None = File(None),
    global_image: UploadFile | None = File(None),
    left_wrist_image: UploadFile | None = File(None),
    right_wrist_image: UploadFile | None = File(None),
    robot_state: str = Form(...),
    stage: str | None = Form(None),
    stage_output: str | None = Form(None),
    effort: str | None = Form(None),
    low_dim: str | None = Form(None),
    low_dim_window: str | None = Form(None),
    prompt: str | None = Form(None),
    task_instruction: str | None = Form(None),
    frame_id: int | None = Form(None),
    model_input_resolution: str | None = Form(None),
) -> StepResponse:
    # Accept prompt/task_instruction for API compatibility, but do not feed them
    # into system-level grounding here. Episode plan is resolved at /reset and
    # kept on TaskState.task_context for scene handlers that need it.
    _resolve_episode_prompt(prompt, task_instruction)
    parsed_robot_state = _parse_robot_state(robot_state)
    resolved_low_dim = _resolve_low_dim(parsed_robot_state, effort, low_dim)
    resolved_low_dim_window = _parse_low_dim_window(low_dim_window)
    try:
        requested_model_input_resolution = (
            normalize_model_input_resolution(model_input_resolution)
            if model_input_resolution is not None
            else None
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    named_images = {
        "global": global_image,
        "left_wrist": left_wrist_image,
        "right_wrist": right_wrist_image,
    }
    if any(file is not None for file in named_images.values()):
        if global_image is None:
            raise HTTPException(status_code=400, detail="global_image is required for multi-camera /step")
        decoded_images = {
            camera: _load_image(file)
            for camera, file in named_images.items()
            if file is not None
        }
        # RUNTIME_LOCK serializes CUDA/tracker access with the background tracking
        # thread (no-op contention in sync mode where no thread runs).
        RUNTIME_LOCK.acquire()
        try:
            result = run_stage_aware_frame_pipeline(
                frames=MultiCameraFrame(decoded_images),
                robot_state=parsed_robot_state,
                stage_payload=_parse_stage_payload(stage, stage_output),
                low_dim=resolved_low_dim,
                low_dim_window=resolved_low_dim_window,
                model_input_resolution=requested_model_input_resolution,
            )
        finally:
            RUNTIME_LOCK.release()
        # Background (local_deploy) mode: /step owns stage + VLM grounding (it just
        # seeded the shared tracker); the rendered frames come from the background
        # loop that tracks at camera frame-rate, so return its latest cache.
        source_images = result.rendered_images
        if _bg_tracker is not None:
            _bg_tracker.notify_step(result.decision)
            cached = _bg_tracker.get_rendered()
            if cached:
                source_images = {**result.rendered_images, **cached}
        output_resolution = normalize_model_input_resolution(
            result.metadata.get("output_resolution")
            or result.metadata.get("model_input_resolution")
        )
        policy_images = {
            camera: resize_policy_image(rendered, output_resolution)
            for camera, rendered in source_images.items()
        }
        rendered_images = {
            camera: _encode_image(rendered)
            for camera, rendered in policy_images.items()
        }
        # Reuse the already-encoded global instead of encoding it a second time.
        global_b64 = rendered_images.get("global")
        if global_b64 is None:
            global_b64 = _encode_image(next(iter(policy_images.values())))
        return StepResponse(
            rendered_image=global_b64,
            rendered_images=rendered_images,
            stage=result.decision.confirmed_stage,
            prompt=result.prompt,
            tracker_valid=result.tracker_valid,
            frame_id=frame_id,
            semantic_grounding=result.semantic_grounding,
            semantic_intent_injection=result.metadata.get("semantic_intent_injection"),
        )

    if image is None:
        raise HTTPException(status_code=400, detail="image or global_image must be provided")
    decoded_image = _load_image(image)
    rendered_image = run_frame_pipeline(decoded_image, parsed_robot_state)
    output_resolution = resolve_output_resolution(load_reasoning_config())
    policy_image = resize_policy_image(rendered_image, output_resolution)
    encoded_image = _encode_image(policy_image)
    return StepResponse(
        rendered_image=encoded_image
    )


@router.post("/reset", response_model=ResetResponse)
def reset_route(
    image: UploadFile = File(...),
    robot_state: str = Form(...),
    prompt: str | None = Form(None),
    task_context: str | None = Form(None),
    task_instruction: str | None = Form(None),
) -> ResetResponse:
    decoded_image = _load_image(image)
    parsed_robot_state = _parse_robot_state(robot_state)
    episode_prompt = _resolve_episode_prompt(prompt, task_instruction)
    config = load_reasoning_config()
    # Structured task_context JSON is always accepted. Free-text prompts are
    # folded in when plan_from_prompt is enabled; incomplete prompts fall back
    # to task.<name> YAML plan fields via the shared episode-context resolver.
    task_name = str((config.get("task") or {}).get("name") or "").strip() or None
    use_prompt = plan_from_prompt_enabled(config, task_name=task_name)
    parsed_context = _parse_task_context(
        task_context,
        episode_prompt if use_prompt else None,
    )
    handler = get_task_handler(task_name, config)
    resolve_task_context = getattr(handler, "resolve_task_context", None)
    if resolve_task_context is not None:
        resolved_context = resolve_task_context(
            config=config,
            task_context=parsed_context,
            episode_prompt=episode_prompt,
        )
    else:
        resolved_context = parsed_context
    with RUNTIME_LOCK:
        reset_stage_aware_runtime()
        run_reset_pipeline(
            decoded_image,
            parsed_robot_state,
            task_context=resolved_context,
        )
    if _bg_tracker is not None:
        _bg_tracker.reset()
    return ResetResponse()


@router.get("/deploy/stats")
def deploy_stats_route() -> dict:
    """Background tracker stats (empty dict in sync mode)."""
    if _bg_tracker is None:
        return {"infer_mode": "sync"}
    return {"infer_mode": "background", **_bg_tracker.stats()}


@router.get("/profiling/stats")
def profiling_stats_route() -> dict:
    """Return and persist the current per-stage /step latency report."""
    PROFILER.flush()
    return PROFILER.snapshot()


@router.get("/health", response_model=HealthResponse)
def health_route() -> HealthResponse:
    dino_healthy = False
    try:
        DinoClient().health_check()
        dino_healthy = True
    except Exception:
        dino_healthy = False

    return HealthResponse(
        status="ok",
        model_ready=is_model_ready(),
        dino_healthy=dino_healthy,
    )


def _maybe_start_background_tracker(config: dict) -> None:
    """Start the local_deploy SAM2 tracking thread when configured."""
    global _bg_tracker
    deploy_cfg = config.get("deploy", {})
    mode = str(deploy_cfg.get("infer_mode") or "sync").lower()
    img_url = str(deploy_cfg.get("img_url") or "")
    if mode not in ("background", "local", "local_deploy"):
        return
    if not img_url:
        logging.getLogger("reasoning_agent").warning(
            "[local_deploy] infer_mode=%s but deploy.img_url is empty; staying in sync mode", mode
        )
        return
    from runtime.background import BackgroundTracker
    from runtime.pipeline import _runtime

    _runtime.ensure_loaded()  # build tracker_manager/render_config before the loop runs
    _bg_tracker = BackgroundTracker(config)
    _bg_tracker.start()


def create_app(*, preload_models: bool = True) -> FastAPI:
    """构建 FastAPI 应用。preload_models=False 可用于仅测路由、不加载 Qwen。"""
    config = load_reasoning_config()
    PROFILER.configure(
        config,
        config_path=os.getenv("REASONING_AGENT_CONFIG"),
    )
    if preload_models:
        initialize_qwen_runtime_from_config(config)

    app = FastAPI(title="ReasoningAgent", version="0.1.0")
    app.include_router(router)
    if preload_models:
        _maybe_start_background_tracker(config)
    return app


app = create_app()


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(
        "inference.api_server:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
