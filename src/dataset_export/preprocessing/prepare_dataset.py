"""Batch-render a LeRobot dataset with stage-aware three-camera export.

Global camera: pick-and-place overlay via ``dataset_utils.render_pick_and_place_fill``;
recognition artifacts use shared helpers from ``dataset_utils``.

Wrist cameras: GroundingDINO + tracker mask overlay on the active camera selected
by either annotated ``stage_id_gt`` routing or direct stage-model prediction.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from common.config_loader import DEFAULT_CONFIG_PATH, load_reasoning_config
from common.runtime_options import (
    SEMANTIC_INTENT_ATTENTION,
    SEMANTIC_INTENT_BOTH,
    SEMANTIC_INTENT_HIGHLIGHTING,
    semantic_intent_injection,
    skipped_stages,
)
from dataset_export.attention_bbox import (
    semantic_grounding_columns,
    masks_to_patch_attention_maps,
    update_info_semantic_grounding_features,
    write_semantic_grounding_parquet,
)
from dataset_export.preprocessing.dataset_utils import (
    DEFAULT_LOG_DIR,
    DEFAULT_RECOGNITION_PADDING,
    DEFAULT_STATE_COLUMN,
    EpisodeResult,
    TeeStream,
    build_artifact_dir,
    build_log_path,
    choose_output_fps,
    configure_process_logging,
    dump_task_state,
    expand_per_gpu_worker_assignments,
    get_video_metadata,
    iter_video_frames,
    log_target_switch,
    merge_worker_summaries,
    parse_physical_gpu_ids,
    prepare_output_dataset,
    save_recognition_artifact,
    split_jobs_for_workers,
    validate_gpu_environment,
)
from dataset_export.lerobot_metadata import (
    prune_to_contiguous_episode_subset,
    update_episode_image_stats,
    update_info_video_metadata,
    validate_common_dataset_files,
)
from dataset_export.task_context import DatasetTaskContextProvider, TaskContextError
from intent.highlighting import MaskRenderConfig, render_mask_overlay
from perception.semantic_grounder.dino import DinoClient
from pattern.runtime.predictor import OnlineStagePredictor, build_stage_predictor
from pattern.runtime.manipulation_pattern_router import ManipulationPatternRouter
from pattern.runtime.source import AnnotationStageSource, build_stage_source
from pattern.runtime.types import (
    ALL_CAMERAS,
    FREE_MOVE_STAGE,
    GLOBAL_CAMERA,
    LEFT_WRIST_CAMERA,
    RIGHT_WRIST_CAMERA,
    StageClassifierOutput,
    TRANSPORT_CONTACT_STAGE,
)
from intent.attention import (
    VIT_PATCH_SIZE,
    patch_grid_shape,
    resize_policy_image,
    resolve_model_input_resolution,
    resolve_output_resolution,
)
from task.operations import plan_next_target
from task.operations import (
    current_target_label,
    is_task_complete,
    resolve_target_block,
    resolve_target_placement,
    summarize_target,
    target_count,
)
from task.base import resolve_reset_detector
from common.task_types import TaskContext, TaskState
from task import build_subtask_manager, get_task_handler
from perception.semantic_grounder.selection import (
    free_move_grounding_delay_frames,
    resolve_grounding_selection,
    should_defer_free_move_grounding,
)
from perception.tracking.sam2_video_tracker import Sam2VideoSegmentTracker
from perception.tracking.state import AREA_PLACEMENT_RENDER_MODE, bbox_to_mask
from perception.task_manager.manager import resolve_target_phrase

DEFAULT_SRC_ROOT = Path(
    "/dataset/robot/real_world/piper/lerobot/letter/piper_letter_v2_big_annotation_merged"
)
DEFAULT_DST_ROOT = Path(
    "/dataset/robot/real_world/piper/lerobot/letter/piper_letter_v2_big_annotation_merged_stage_rendered"
)
DEFAULT_GPU_IDS = "0"
WRIST_CAMERAS = (LEFT_WRIST_CAMERA, RIGHT_WRIST_CAMERA)


@dataclass
class WristTrackSegment:
    """A contiguous run of frames where one wrist camera is the active route.

    Grounded once by DINO at ``start``; SAM2 video propagation fills the rest.
    """

    camera: str
    start: int
    end: int  # inclusive
    box: tuple[float, ...] | None
    prompt: str
    render_mode: str = "sam2"


def requires_dino(config: dict[str, Any]) -> bool:
    if resolve_reset_detector(config) == "dino":
        return True
    return str(config["grounding"]["grounder"]).lower() != "qwen"


def current_subtask_label(task_state: TaskState) -> str | None:
    """Label of the active subtask: a letter, or an object such as "apple"."""
    target = current_target_label(task_state)
    return target.lower() if target else None


def segment_grounding_prompt(camera: str, route_prompt: str, task_state: TaskState) -> str | None:
    """Prompt to ground a new segment on ``camera``.

    The route prompt already names the active subtask's target for this action
    pattern, on every camera.
    """
    del camera, task_state
    return route_prompt


def ground_track_segment_box(
    *,
    camera: str,
    image: Image.Image,
    seg_prompt: str,
    task_state: TaskState,
    stage_name: str,
    grounder: str,
    grounding_cfg: dict[str, Any],
    dino_client: DinoClient | None,
    config: dict[str, Any],
    qwen_ground_phrase: Any,
    qwen_max_new_tokens: int,
    qwen_board_pad: int,
    recognition_padding: int,
) -> tuple[float, ...] | None:
    """Resolve one SAM2 seed through the same selector used by online inference."""
    del grounder, grounding_cfg, qwen_max_new_tokens, qwen_board_pad
    selection = resolve_grounding_selection(
        camera=camera,
        image=image,
        prompt=seg_prompt,
        task_state=task_state,
        stage_name=stage_name,
        config=config,
        dino_client=dino_client,
        verify_target=current_subtask_label(task_state),
        recognition_padding=recognition_padding,
        allow_vlm=True,
        qwen_ground_fn=qwen_ground_phrase,
    )
    return selection.bbox_xyxy


def overlay_masked_frames(
    frames: list[Image.Image],
    masks: dict[int, np.ndarray],
    frame_count: int,
    render_enabled: bool,
    render_config: MaskRenderConfig,
) -> list[Image.Image]:
    rendered: list[Image.Image] = []
    for frame_index in range(frame_count):
        frame = frames[frame_index].convert("RGB")
        mask = masks.get(frame_index)
        if render_enabled and mask is not None:
            rendered.append(render_mask_overlay(frame, mask, render_config))
        else:
            rendered.append(frame)
    return rendered


def summarize_segment_masks(segment: WristTrackSegment, masks: dict[int, np.ndarray]) -> dict[str, Any]:
    """Measure actual masks, rather than counting empty SAM2 outputs as coverage."""
    expected = segment.end - segment.start + 1
    areas = [int(np.count_nonzero(masks[i])) if i in masks else 0 for i in range(expected)]
    produced = sum(i in masks for i in range(expected))
    nonempty = sum(area > 0 for area in areas)
    return {
        **asdict(segment),
        "expected_frames": expected,
        "produced_frames": produced,
        "nonempty_frames": nonempty,
        "empty_or_missing_frames": expected - nonempty,
        "nonempty_fraction": nonempty / expected,
        "mask_area_min": min(areas),
        "mask_area_max": max(areas),
    }


def static_shape_mask(
    image_size: tuple[int, int],
    bbox_xyxy: tuple[float, ...],
    prompt: str,
) -> np.ndarray:
    """Rasterize a small recognized target without asking SAM to segment its parent."""
    from PIL import ImageDraw

    width, height = image_size
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    xyxy = (
        max(0, min(width, round(x1))),
        max(0, min(height, round(y1))),
        max(0, min(width, round(x2))),
        max(0, min(height, round(y2))),
    )
    canvas = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(canvas)
    normalized = str(prompt).lower()
    if "circular" in normalized:
        draw.ellipse(xyxy, fill=255)
    elif "l-shaped" in normalized or "l shaped" in normalized:
        left, top, right, bottom = xyxy
        arm_x = left + max(1, round((right - left) * 0.42))
        arm_y = top + max(1, round((bottom - top) * 0.58))
        draw.polygon(
            [
                (left, top),
                (arm_x, top),
                (arm_x, arm_y),
                (right, arm_y),
                (right, bottom),
                (left, bottom),
            ],
            fill=255,
        )
    else:
        draw.rectangle(xyxy, fill=255)
    return np.asarray(canvas, dtype=np.uint8) > 0


@dataclass(frozen=True)
class StageAwareEpisodeJob:
    episode_name: str
    episode_index: int
    parquet_path: Path
    output_parquet_path: Path
    semantic_grounding_parquet_path: Path | None
    video_paths: dict[str, Path]
    output_video_paths: dict[str, Path]
    attention_output_video_paths: dict[str, Path] | None = None
    task_context: TaskContext | None = None
    task_context_error: str | None = None
    verbose_task_state: bool = False


def normalize_episode_name(token: str) -> str:
    token = token.strip()
    if not token:
        raise ValueError("empty episode id")
    if token.endswith(".mp4"):
        token = Path(token).stem
    if token.startswith("episode_"):
        return token
    if token.isdigit():
        return f"episode_{int(token):06d}"
    raise ValueError(f"invalid episode id {token!r}; use 19, 000019, or episode_000019")


def parse_episode_names(raw: str | None) -> set[str]:
    if not raw:
        return set()
    names: set[str] = set()
    for token in raw.replace(",", " ").split():
        names.add(normalize_episode_name(token))
    return names


def failed_episode_names_from_summary(summary_path: Path) -> set[str]:
    if not summary_path.is_file():
        raise FileNotFoundError(f"failed summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for episode in summary.get("failed_episodes", []):
        name = episode.get("episode_name")
        if name:
            names.add(str(name))
    for episode_id in summary.get("failed_episode_ids", []):
        names.add(normalize_episode_name(str(episode_id)))
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-aware batch render for LeRobot datasets (global pick/place + wrist masks).",
    )
    parser.add_argument("--src-root", default=str(DEFAULT_SRC_ROOT))
    parser.add_argument("--dst-root", default=str(DEFAULT_DST_ROOT))
    parser.add_argument(
        "--attention-dst-root",
        default=None,
        help=(
            "Compatibility option for both mode; must equal --dst-root. "
            "Both mode writes highlighted videos and grounding columns into one dataset."
        ),
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--semantic-intent-injection",
        choices=(
            SEMANTIC_INTENT_HIGHLIGHTING,
            SEMANTIC_INTENT_ATTENTION,
            SEMANTIC_INTENT_BOTH,
        ),
        default=None,
        help="Override task.semantic_intent_injection for this offline export.",
    )
    parser.add_argument("--episode-pattern", default="episode_*.mp4")
    parser.add_argument(
        "--episode-ids",
        default=None,
        help=(
            "Comma/space separated episodes to render, e.g. '19,episode_000023'. "
            "This filters after --episode-pattern."
        ),
    )
    parser.add_argument(
        "--failed-from-summary",
        default=None,
        help="Render only failed episodes listed in an existing render summary JSON.",
    )
    parser.add_argument("--state-column", default=DEFAULT_STATE_COLUMN)
    parser.add_argument("--stage-column", default=None, help="Override stage_source.annotation.column.")
    parser.add_argument(
        "--task-context-file",
        default=None,
        help="Override dataset.task_context.path for a JSONL task-context source.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-copy", action="store_true")
    parser.add_argument(
        "--export-semantic-grounding",
        dest="export_semantic_grounding",
        action="store_true",
        help=(
            "Assert that task.semantic_intent_injection is attention. Attention "
            "export is selected by the task config; this flag is kept for compatibility."
        ),
    )
    parser.add_argument(
        "--grounder",
        choices=("dino", "qwen"),
        default=None,
        help=(
            "Per-frame detector for grounding new segments (overrides grounding.grounder). "
            "'qwen': Qwen3-VL boxes the active target prompt and fails closed when "
            "no box is returned. 'dino': GroundingDINO detect + Qwen verify."
        ),
    )
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument(
        "--recognition-padding",
        "--letter-padding",
        dest="recognition_padding",
        type=int,
        default=DEFAULT_RECOGNITION_PADDING,
        help="Padding around detected object crops for Qwen recognition.",
    )
    parser.add_argument(
        "--re-recognize-on-stage-switch",
        action="store_true",
        help="Refresh movable-object recognition when a confirmed stage switch occurs.",
    )
    parser.add_argument("--fps-mode", choices=("source", "fixed"), default="source")
    parser.add_argument("--fixed-fps", type=float, default=30.0)
    parser.add_argument(
        "--global-render-mode",
        choices=("pick_place_fill", "block_mask"),
        default="block_mask",
        help=(
            "Global camera rendering: 'block_mask' tracks the active target "
            "with a SAM2 mask like the wrists (default); 'pick_place_fill' draws the "
            "red target-block + blue placement fill instead."
        ),
    )
    parser.add_argument("--gpu-ids", default=DEFAULT_GPU_IDS)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument(
        "--merge-summary",
        default=None,
        help=(
            "After a partial rerender, replace matching episode results in this "
            "existing full summary JSON and write the merged summary back to it."
        ),
    )
    parser.add_argument(
        "--instant-stage-switch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force stage_stability_frames=1 for annotated GT (default: true).",
    )
    return parser.parse_args()


def write_rendered_video_stream(output_path: Path, frames: list[Image.Image], fps: float) -> None:
    """Encode frames by piping raw RGB straight into ffmpeg (no temp PNG files).

    The PNG-sequence writer in dataset_utils writes one PNG per frame per
    camera (~6k disk writes/episode), which dominated render time. Streaming raw
    frames to ffmpeg is ~5-6x faster for the same libx264/yuv420p output.
    """
    if not frames:
        raise ValueError("rendered frames must not be empty")
    width, height = frames[0].size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            if frame.size != (width, height):
                raise RuntimeError("all rendered frames must have the same size")
            process.stdin.write(np.asarray(frame.convert("RGB"), dtype=np.uint8).tobytes())
        process.stdin.close()
    except BrokenPipeError as exc:  # ffmpeg died early
        process.wait()
        raise RuntimeError(f"ffmpeg pipe broke for {output_path}") from exc
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed for {output_path} with exit code {return_code}")


def camera_video_keys(config: dict[str, Any]) -> dict[str, str]:
    cameras_cfg = config["cameras"]
    return {
        GLOBAL_CAMERA: str(cameras_cfg["global"]),
        LEFT_WRIST_CAMERA: str(cameras_cfg["left_wrist"]),
        RIGHT_WRIST_CAMERA: str(cameras_cfg["right_wrist"]),
    }


def collect_stage_aware_jobs(
    src_root: Path,
    dst_root: Path,
    attention_dst_root: Path | None,
    camera_keys: dict[str, str],
    episode_pattern: str,
    limit: int | None,
    episode_names: set[str] | None = None,
    task_context_provider: DatasetTaskContextProvider | None = None,
) -> list[StageAwareEpisodeJob]:
    global_key = camera_keys[GLOBAL_CAMERA]
    video_roots = sorted(src_root.glob(f"videos/chunk-*/{global_key}"))
    jobs: list[StageAwareEpisodeJob] = []

    for global_video_root in video_roots:
        chunk_name = global_video_root.parent.name
        data_root = src_root / "data" / chunk_name
        for global_video_path in sorted(global_video_root.glob(episode_pattern)):
            episode_name = global_video_path.stem
            try:
                episode_index = int(episode_name.removeprefix("episode_"))
            except ValueError as exc:
                raise ValueError(f"invalid LeRobot episode filename: {global_video_path.name}") from exc
            if episode_names is not None and episode_name not in episode_names:
                continue
            parquet_path = data_root / f"{episode_name}.parquet"
            if not parquet_path.is_file():
                raise FileNotFoundError(f"missing parquet for {global_video_path}: {parquet_path}")
            output_parquet_path = dst_root / "data" / chunk_name / f"{episode_name}.parquet"
            semantic_grounding_parquet_path = (
                attention_dst_root / "data" / chunk_name / f"{episode_name}.parquet"
                if attention_dst_root is not None
                else None
            )

            video_paths: dict[str, Path] = {}
            output_video_paths: dict[str, Path] = {}
            attention_output_video_paths: dict[str, Path] | None = (
                {} if attention_dst_root is not None else None
            )
            for camera_name, video_key in camera_keys.items():
                video_path = src_root / "videos" / chunk_name / video_key / global_video_path.name
                if not video_path.is_file():
                    raise FileNotFoundError(f"missing {camera_name} video for {episode_name}: {video_path}")
                video_paths[camera_name] = video_path
                output_video_paths[camera_name] = (
                    dst_root / "videos" / chunk_name / video_key / global_video_path.name
                )
                output_video_paths[camera_name].parent.mkdir(parents=True, exist_ok=True)
                if attention_output_video_paths is not None:
                    attention_output_video_paths[camera_name] = (
                        attention_dst_root
                        / "videos"
                        / chunk_name
                        / video_key
                        / global_video_path.name
                    )
                    attention_output_video_paths[camera_name].parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

            task_context = None
            task_context_error = None
            if task_context_provider is not None:
                try:
                    task_context = task_context_provider.resolve(
                        episode_index=episode_index,
                        episode_name=episode_name,
                    )
                except TaskContextError as exc:
                    task_context_error = str(exc)

            jobs.append(
                StageAwareEpisodeJob(
                    episode_name=episode_name,
                    episode_index=episode_index,
                    parquet_path=parquet_path,
                    output_parquet_path=output_parquet_path,
                    semantic_grounding_parquet_path=semantic_grounding_parquet_path,
                    video_paths=video_paths,
                    output_video_paths=output_video_paths,
                    attention_output_video_paths=attention_output_video_paths,
                    task_context=task_context,
                    task_context_error=task_context_error,
                )
            )

    if limit is not None:
        jobs = jobs[:limit]
    if not jobs:
        requested = f" and episode names {sorted(episode_names)}" if episode_names else ""
        raise RuntimeError(f"no episodes matched the requested pattern{requested}")

    return [
        StageAwareEpisodeJob(
            episode_name=job.episode_name,
            episode_index=job.episode_index,
            parquet_path=job.parquet_path,
            output_parquet_path=job.output_parquet_path,
            semantic_grounding_parquet_path=job.semantic_grounding_parquet_path,
            video_paths=job.video_paths,
            output_video_paths=job.output_video_paths,
            attention_output_video_paths=job.attention_output_video_paths,
            task_context=job.task_context,
            task_context_error=job.task_context_error,
            verbose_task_state=idx < 2,
        )
        for idx, job in enumerate(jobs)
    ]


def load_episode_states(parquet_path: Path, state_column: str) -> list[Sequence[float]]:
    import pandas as pd

    dataframe = pd.read_parquet(parquet_path)
    if state_column not in dataframe.columns:
        available_columns = ", ".join(str(column) for column in dataframe.columns)
        raise ValueError(
            f"missing required column in {parquet_path}: {state_column}; "
            f"available columns: {available_columns}"
        )
    states = dataframe[state_column].tolist()
    if not states:
        raise ValueError(f"parquet contains no states: {parquet_path}")
    return states


def load_episode_low_dim(
    parquet_path: Path,
    state_column: str,
    effort_column: str = "effort",
) -> list[np.ndarray]:
    """State + effort inputs used by offline stage prediction."""
    import pandas as pd

    dataframe = pd.read_parquet(parquet_path, columns=[state_column, effort_column])
    return [
        np.concatenate(
            [np.asarray(state, dtype=np.float32), np.asarray(effort, dtype=np.float32)]
        )
        for state, effort in zip(
            dataframe[state_column], dataframe[effort_column], strict=True
        )
    ]


def align_camera_frames(
    camera_frames: dict[str, list[Image.Image]],
    parquet_row_count: int,
    episode_name: str,
) -> dict[str, list[Image.Image]]:
    counts = {camera: len(frames) for camera, frames in camera_frames.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"camera frame count mismatch for {episode_name}: {counts}")

    video_frame_count = next(iter(counts.values()))
    if video_frame_count == parquet_row_count:
        return camera_frames
    if video_frame_count == parquet_row_count + 1:
        return {camera: frames[:-1] for camera, frames in camera_frames.items()}
    raise RuntimeError(
        f"expected equal camera frame counts = parquet_rows or parquet_rows + 1 for {episode_name}, "
        f"got video_frames={video_frame_count} parquet_rows={parquet_row_count}"
    )


def rerecognize_blocks_on_stage_switch(
    *,
    episode_name: str,
    frame_index: int,
    global_frame: Image.Image,
    task_state: TaskState,
    dino_client: DinoClient,
    recognition_padding: int,
    verbose_task_state: bool,
    artifact_dir: Path | None,
    config: dict[str, Any],
) -> TaskState:
    """Re-detect movable objects on a confirmed stage switch (opt-in).

    This does NOT advance progress: subtask scheduling is owned by the
    SubtaskManager. Here we refresh detections and re-plan the *current* target.
    """
    previous_target_summary = summarize_target(task_state)
    handler = get_task_handler(getattr(task_state, "task_name", None), config)
    refresh = getattr(handler, "refresh_movables", None)
    if not callable(refresh):
        return task_state
    task_state = refresh(
        image=global_frame,
        dino_client=dino_client,
        config=config,
        task_state=task_state,
        recognition_padding=recognition_padding,
    )
    new_blocks = task_state.blocks

    if verbose_task_state and artifact_dir is not None:
        save_recognition_artifact(
            artifact_dir=artifact_dir,
            episode_name=episode_name,
            frame_index=frame_index,
            image=global_frame,
            blocks=new_blocks,
        )
    print(
        f"[stage_switch_recognition][{episode_name}] frame={frame_index} "
        f"block_count={len(new_blocks)}",
        flush=True,
    )
    log_target_switch(
        episode_name=episode_name,
        frame_index=frame_index,
        previous_summary=previous_target_summary,
        new_summary=summarize_target(task_state),
    )
    return task_state


def render_global_frame(global_frame: Image.Image, task_state: TaskState) -> Image.Image:
    if is_task_complete(task_state):
        return global_frame.convert("RGB")
    from dataset_export.preprocessing.dataset_utils import render_pick_and_place_fill

    return render_pick_and_place_fill(
        image=global_frame,
        target_block=resolve_target_block(task_state),
        target_placement=resolve_target_placement(task_state),
    )


def build_batch_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_reasoning_config(args.config)
    if args.semantic_intent_injection:
        config.setdefault("task", {})["semantic_intent_injection"] = (
            args.semantic_intent_injection
        )
    if args.grounder:
        config.setdefault("grounding", {})["grounder"] = args.grounder
    if args.instant_stage_switch:
        config.setdefault("stage_aware", {})["stage_stability_frames"] = 1
    if args.stage_column:
        config.setdefault("stages", {})["column"] = args.stage_column
    return config


def resolve_output_roots(
    dst_root: Path,
    semantic_intent_mode: str,
    attention_dst_root: str | Path | None = None,
) -> tuple[Path | None, Path | None]:
    """Return enabled output destinations; both mode shares one dataset root."""
    explicit_attention_root = Path(attention_dst_root) if attention_dst_root else None
    if semantic_intent_mode == SEMANTIC_INTENT_HIGHLIGHTING:
        if explicit_attention_root is not None:
            raise ValueError("--attention-dst-root requires semantic_intent_injection: both")
        return dst_root, None
    if semantic_intent_mode == SEMANTIC_INTENT_ATTENTION:
        if explicit_attention_root is not None:
            raise ValueError("--attention-dst-root requires semantic_intent_injection: both")
        return None, dst_root
    if semantic_intent_mode == SEMANTIC_INTENT_BOTH:
        if explicit_attention_root is not None and explicit_attention_root.resolve() != dst_root.resolve():
            raise ValueError("both mode writes one dataset; --attention-dst-root must equal --dst-root")
        return dst_root, dst_root
    raise ValueError(f"unsupported semantic intent mode: {semantic_intent_mode!r}")


def evaluate_multicam_render_success(
    task_state: TaskState,
    rendered_frame_count: int,
    expected_frame_count: int,
    output_video_paths: dict[str, Path],
    skipped_segments: Sequence[str] | None = None,
) -> tuple[bool, str | None]:
    del task_state
    if rendered_frame_count != expected_frame_count:
        return (
            False,
            f"rendered_frame_count={rendered_frame_count} expected={expected_frame_count}",
        )
    for camera, output_path in output_video_paths.items():
        if not output_path.is_file():
            return False, f"missing output video for {camera}: {output_path}"
    if skipped_segments:
        preview = "; ".join(list(skipped_segments)[:4])
        more = "" if len(skipped_segments) <= 4 else f" (+{len(skipped_segments) - 4} more)"
        return False, f"ungrounded track segments ({len(skipped_segments)}): {preview}{more}"
    return True, None


def evaluate_task_completion(task_state: TaskState) -> tuple[bool, str | None]:
    observed = task_state.metadata.get("observed_subtask_count")
    if observed is not None and len(task_state.target_labels) < int(observed):
        return False, (
            f"undetected_subtasks target_count={len(task_state.target_labels)} "
            f"observed_subtask_count={observed}"
        )
    if is_task_complete(task_state):
        return True, None
    return (
        False,
        (
            f"task_incomplete progress_idx={task_state.progress_idx} "
            f"target_count={target_count(task_state)} "
            f"target_labels_len={len(task_state.target_labels)} "
            f"target_word_len={len(task_state.target_word)}"
        ),
    )


def count_completed_subtasks(
    *,
    stage_ids: Sequence[int],
    router: ManipulationPatternRouter,
    completion_stages: tuple[str, ...],
) -> int:
    """Count completed subtasks from a confirmed action pattern sequence."""
    default_stage = router.registry.default_stage_name
    default_stage_id = router.registry.default_stage_id
    completion_set = set(completion_stages)
    previous_stage: str | None = None
    visited_completion = False
    completed = 0

    for stage_id in stage_ids:
        payload = {"stage_id": stage_id}
        stage_output = StageClassifierOutput.from_payload(
            payload,
            default_stage=default_stage,
            default_stage_id=default_stage_id,
        )
        stage = router.decide(stage_output).confirmed_stage
        if previous_stage is None:
            previous_stage = stage
            if stage in completion_set:
                visited_completion = True
            continue
        if stage == previous_stage:
            continue
        if stage in completion_set:
            visited_completion = True
        elif visited_completion:
            completed += 1
            visited_completion = False
        previous_stage = stage

    if visited_completion:
        completed += 1
    return completed


def align_auto_detected_targets_to_subtask_count(task_state: TaskState, completed: int) -> None:
    """Offline render: cap auto-detected object queues to observed action count."""
    if completed <= 0:
        return
    if not str(task_state.metadata.get("subtask_source") or "").startswith("detected"):
        return
    # Keep the independent action count even when detection missed an object.
    # An exhausted, undersized queue must not count as successful processing.
    task_state.metadata["observed_subtask_count"] = completed
    task_state.metadata["detected_target_count"] = len(task_state.target_labels)
    if len(task_state.target_labels) <= completed:
        return

    task_state.target_labels = task_state.target_labels[:completed]
    task_state.target_categories = task_state.target_categories[:completed]
    task_state.target_placements = task_state.target_placements[:completed]
    task_state.target_block_id, task_state.target_placement_id = plan_next_target(task_state)


def evaluate_task_vlm_success(task_state: TaskState) -> bool:
    return bool(task_state.target_labels) and all(bool(label) for label in task_state.target_labels)


def evaluate_task_detection_success(task_state: TaskState, blocks) -> tuple[bool, list[str] | None]:
    if not task_state.target_labels:
        return False, None
    missing = [
        label
        for label in task_state.target_labels
        if not any((block.label or block.letter or "").lower() == label.lower() for block in blocks)
    ]
    observed = int(task_state.metadata.get("observed_subtask_count", 0))
    return not missing and len(task_state.target_labels) >= observed, missing or None


def build_stage_aware_success_flags(
    *,
    config: dict[str, Any],
    target_word: str | None,
    blocks,
    task_state: TaskState,
    rendered_frame_count: int,
    expected_frame_count: int,
    output_video_paths: dict[str, Path],
    skipped_segments: Sequence[str] | None = None,
) -> dict[str, Any]:
    render_success, render_failure_reason = evaluate_multicam_render_success(
        task_state=task_state,
        rendered_frame_count=rendered_frame_count,
        expected_frame_count=expected_frame_count,
        output_video_paths=output_video_paths,
        skipped_segments=skipped_segments,
    )
    vlm_success = evaluate_task_vlm_success(task_state)
    dino_success, missing_targets = evaluate_task_detection_success(task_state, blocks)
    task_complete, task_incomplete_reason = evaluate_task_completion(task_state)
    reset_failure = task_state.metadata.get("reset_failure")
    if reset_failure:
        # Soft reset misses still produce videos; keep quality flags honest.
        dino_success = False
    del config, target_word
    return {
        "success": bool(
            vlm_success
            and dino_success
            and render_success
            and task_complete
            and not reset_failure
        ),
        "vlm_success": vlm_success,
        "dino_success": dino_success,
        "render_success": render_success,
        "task_complete": task_complete,
        "dino_missing_letters": missing_targets,
        "render_failure_reason": render_failure_reason,
        "task_incomplete_reason": task_incomplete_reason,
        "error": str(reset_failure) if reset_failure else None,
    }


def merge_rerun_worker_summaries(
    existing_summary_path: Path,
    rerun_worker_summaries: list[dict[str, Any]],
    *,
    expected_rerun_episode_names: set[str] | None = None,
    output_path: Path,
    render_mode: str,
    config_path: Path,
) -> dict[str, Any]:
    if not existing_summary_path.is_file():
        raise FileNotFoundError(f"merge summary not found: {existing_summary_path}")
    existing = json.loads(existing_summary_path.read_text(encoding="utf-8"))
    previous_worker_summaries = existing.get("worker_summaries")
    if not isinstance(previous_worker_summaries, list):
        raise ValueError(f"summary has no worker_summaries array: {existing_summary_path}")

    reported_rerun_episode_names = {
        result.get("episode_name")
        for worker_summary in rerun_worker_summaries
        for result in worker_summary.get("results", [])
        if result.get("episode_name")
    }
    if not reported_rerun_episode_names:
        raise ValueError("rerun produced no episode results to merge")
    rerun_episode_names = expected_rerun_episode_names or reported_rerun_episode_names

    retained_worker_summaries: list[dict[str, Any]] = []
    for worker_summary in previous_worker_summaries:
        retained = dict(worker_summary)
        retained["results"] = [
            result
            for result in worker_summary.get("results", [])
            if result.get("episode_name") not in rerun_episode_names
        ]
        retained_worker_summaries.append(retained)

    merged_worker_summaries = retained_worker_summaries + rerun_worker_summaries
    retained_episode_names = {
        result.get("episode_name")
        for worker_summary in retained_worker_summaries
        for result in worker_summary.get("results", [])
        if result.get("episode_name")
    }
    expected_episode_jobs = max(
        int(existing.get("expected_episode_jobs") or 0),
        len(retained_episode_names | rerun_episode_names),
    )
    merged = merge_worker_summaries(
        merged_worker_summaries,
        output_path=output_path,
        expected_episode_jobs=expected_episode_jobs,
    )
    merged["render_mode"] = render_mode
    merged["config_path"] = str(config_path.resolve())
    merged["merged_from_summary"] = str(existing_summary_path)
    merged["merged_rerun_episode_names"] = sorted(rerun_episode_names)
    output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return merged


def process_stage_aware_episode(
    job: StageAwareEpisodeJob,
    worker_id: int,
    gpu_id: int,
    artifact_dir: Path,
    config: dict[str, Any],
    recognition_padding: int,
    re_recognize_on_stage_switch: bool,
    fps_mode: str,
    fixed_fps: float,
    state_column: str,
    global_render_mode: str,
    render_highlighting: bool,
    export_semantic_grounding: bool,
    video_tracker: Sam2VideoSegmentTracker | None,
    stage_predictor: OnlineStagePredictor | None,
) -> EpisodeResult:
    episode_start = time.time()
    if job.task_context_error:
        raise TaskContextError(job.task_context_error)
    dino_client = DinoClient(config=config) if requires_dino(config) else None

    stage_source = build_stage_source(config)

    router = ManipulationPatternRouter(config)
    subtask_manager = build_subtask_manager(config)
    subtask_manager.reset()
    render_config = MaskRenderConfig.from_config(config)
    render_enabled = bool(config["render"]["enabled"])
    grounding_cfg = config["grounding"]
    grounder = str(grounding_cfg["grounder"]).lower()
    qwen_cfg = grounding_cfg["qwen"]
    qwen_max_new_tokens = int(qwen_cfg["max_new_tokens"])
    qwen_board_pad = float(qwen_cfg["board_pad"])
    qwen_ground_phrase = None
    if grounder == "qwen":
        from perception.semantic_grounder.qwen import qwen_ground_phrase as _qwen_ground_phrase

        qwen_ground_phrase = _qwen_ground_phrase
    default_stage = router.registry.default_stage_name
    default_stage_id = router.registry.default_stage_id

    observation_states = load_episode_states(job.parquet_path, state_column=state_column)
    camera_frames = {
        camera: list(iter_video_frames(video_path))
        for camera, video_path in job.video_paths.items()
    }
    camera_frames = align_camera_frames(
        camera_frames,
        parquet_row_count=len(observation_states),
        episode_name=job.episode_name,
    )
    frame_count = len(observation_states)

    if isinstance(stage_source, AnnotationStageSource):
        stage_ids = stage_source.read_episode_stage_ids(job.parquet_path)
        stage_payloads = [stage_source.payload_for_stage_id(stage_id) for stage_id in stage_ids]
    else:
        low_dim = load_episode_low_dim(job.parquet_path, state_column)
        stage_predictor.reset()
        stage_payloads = [
            stage_predictor.predict(
                {
                    camera: camera_frames[camera][frame_index].convert("RGB")
                    for camera in ALL_CAMERAS
                },
                low_dim[frame_index],
            ).to_payload()
            for frame_index in range(frame_count)
        ]
        stage_ids = [int(payload["stage_id"]) for payload in stage_payloads]

    if len(stage_ids) != frame_count:
        raise RuntimeError(
            f"stage length mismatch for {job.episode_name}: "
            f"stages={len(stage_ids)} frames={frame_count}"
        )

    source_fps, _, _, _ = get_video_metadata(job.video_paths[GLOBAL_CAMERA])
    output_fps = choose_output_fps(source_fps=source_fps, fps_mode=fps_mode, fixed_fps=fixed_fps)
    model_input_resolution = resolve_model_input_resolution(config)
    output_resolution = resolve_output_resolution(config)

    global_frames = camera_frames[GLOBAL_CAMERA]
    task_handler = get_task_handler(config=config)
    task_state = task_handler.build_initial_state(
        image=global_frames[0],
        dino_client=dino_client,
        config=config,
        recognition_padding=recognition_padding,
        task_context=job.task_context,
    )
    completed_subtasks = count_completed_subtasks(
        stage_ids=stage_ids,
        router=router,
        completion_stages=subtask_manager.completion_stages,
    )
    align_auto_detected_targets_to_subtask_count(task_state, completed_subtasks)
    init_blocks = list(task_state.blocks)
    initial_task_state = json.loads(dump_task_state(task_state))
    print(
        f"[episode_init][{job.episode_name}] worker={worker_id} gpu={gpu_id} "
        f"target_word={task_state.target_word} {summarize_target(task_state)}",
        flush=True,
    )
    if job.verbose_task_state:
        print(
            f"[task_state][{job.episode_name}] frame=0 stage=episode_init\n"
            f"{dump_task_state(task_state)}",
            flush=True,
        )

    # ---- Pass 1: stage logic + segment plan (+ global pick/place render if used) ----
    # SAM2 video propagation needs each contiguous camera-active run as a unit, so
    # we first walk the episode recording one DINO box per segment, then propagate
    # with the video predictor in pass 2.
    #
    # global_render_mode:
    #   pick_place_fill - global keeps the red target-block / blue placement fill.
    #   block_mask      - global renders the active target with the same mask
    #                     style as wrists. Tasks may select a fixed bbox mask for
    #                     an empty placement area instead of SAM2 propagation.
    block_mask_global = global_render_mode == "block_mask"
    tracked_cameras: tuple[str, ...] = WRIST_CAMERAS + ((GLOBAL_CAMERA,) if block_mask_global else ())
    skip_stage_render = skipped_stages(config)

    global_rendered: list[Image.Image] = []
    segments: dict[str, list[WristTrackSegment]] = {camera: [] for camera in tracked_cameras}
    open_segment: dict[str, WristTrackSegment | None] = {camera: None for camera in tracked_cameras}
    stage_switch_count = 0
    frames_in_confirmed_stage = 0

    for frame_index in range(frame_count):
        images = {
            camera: camera_frames[camera][frame_index].convert("RGB")
            for camera in ALL_CAMERAS
        }
        stage_payload = stage_payloads[frame_index]
        stage_output = StageClassifierOutput.from_payload(
            stage_payload,
            default_stage=default_stage,
            default_stage_id=default_stage_id,
        )
        decision = router.decide(stage_output)

        if decision.stage_changed:
            frames_in_confirmed_stage = 0
            stage_switch_count += 1
            print(
                f"[stage_switch][{job.episode_name}] frame={frame_index} "
                f"-> {decision.confirmed_stage} (focus={decision.focus}) "
                f"{summarize_target(task_state)}",
                flush=True,
            )
            if re_recognize_on_stage_switch:
                task_state = rerecognize_blocks_on_stage_switch(
                    episode_name=job.episode_name,
                    frame_index=frame_index,
                    global_frame=images[GLOBAL_CAMERA],
                    task_state=task_state,
                    dino_client=dino_client,
                    recognition_padding=recognition_padding,
                    verbose_task_state=job.verbose_task_state,
                    artifact_dir=artifact_dir if job.verbose_task_state else None,
                    config=config,
                )

        # Subtask scheduling (memory): advance only when the action pattern leaves
        # the contact group for a free one, so the active subtask stays locked
        # through free_move -> pre_contact -> contact -> transport.
        if subtask_manager.observe(decision=decision, task_state=task_state):
            print(
                f"[target_progress][{job.episode_name}] frame={frame_index} "
                f"advanced_to_progress_idx={task_state.progress_idx} "
                f"subtask_label={current_subtask_label(task_state)}",
                flush=True,
            )

        prompt = resolve_target_phrase(
            decision=decision,
            images=images,
            task_state=task_state,
            task_instruction=None,
            config=config,
        )
        if not block_mask_global:
            global_rendered.append(render_global_frame(images[GLOBAL_CAMERA], task_state))

        # Open/extend/close a tracking segment for whichever tracked camera is the
        # active route this frame (only one camera is active per the routing table).
        # A segment also breaks when the grounding prompt changes mid-run (e.g. the
        # global camera stays active across transport -> next free_move while the
        # active subtask advances), so each segment tracks exactly one prompt.
        # Pick the active tracked camera. Global tracks the table object in
        # free_move and the destination in transport_contact.
        global_stages = (FREE_MOVE_STAGE, TRANSPORT_CONTACT_STAGE)
        active_camera = None
        defer_grounding = should_defer_free_move_grounding(
            config=config,
            stage_name=decision.confirmed_stage,
            frames_in_stage=frames_in_confirmed_stage,
        )
        if defer_grounding and frames_in_confirmed_stage == 0:
            print(
                f"[grounding_delay][{job.episode_name}] frame={frame_index} "
                f"stage={decision.confirmed_stage} "
                f"wait_frames={free_move_grounding_delay_frames(config)}",
                flush=True,
            )
        if (
            not defer_grounding
            and not is_task_complete(task_state)
            and decision.confirmed_stage not in skip_stage_render
        ):
            for camera in tracked_cameras:
                if camera not in decision.route.cameras:
                    continue
                if camera == GLOBAL_CAMERA and decision.confirmed_stage not in global_stages:
                    continue
                active_camera = camera
                break

        for camera in tracked_cameras:
            segment = open_segment[camera]
            if camera != active_camera:
                if segment is not None:
                    segments[camera].append(segment)
                    open_segment[camera] = None
                continue

            seg_prompt = segment_grounding_prompt(camera, prompt, task_state)
            if segment is not None and segment.prompt != (seg_prompt or ""):
                segments[camera].append(segment)
                segment = None
                open_segment[camera] = None

            if segment is None:
                if seg_prompt:
                    box = ground_track_segment_box(
                        camera=camera,
                        image=images[camera],
                        seg_prompt=seg_prompt,
                        task_state=task_state,
                        stage_name=decision.confirmed_stage,
                        grounder=grounder,
                        grounding_cfg=grounding_cfg,
                        dino_client=dino_client,
                        config=config,
                        qwen_ground_phrase=qwen_ground_phrase,
                        qwen_max_new_tokens=qwen_max_new_tokens,
                        qwen_board_pad=qwen_board_pad,
                        recognition_padding=recognition_padding,
                    )
                    mode_hook = getattr(task_handler, "resolve_segment_render_mode", None)
                    render_mode = (
                        str(
                            mode_hook(
                                task_state,
                                camera=camera,
                                stage_name=decision.confirmed_stage,
                                prompt=seg_prompt,
                            )
                        )
                        if callable(mode_hook)
                        else "sam2"
                    )
                    open_segment[camera] = WristTrackSegment(
                        camera=camera,
                        start=frame_index,
                        end=frame_index,
                        box=box,
                        prompt=seg_prompt,
                        render_mode=render_mode,
                    )
            else:
                segment.end = frame_index

        frames_in_confirmed_stage += 1

    for camera in tracked_cameras:
        if open_segment[camera] is not None:
            segments[camera].append(open_segment[camera])
            open_segment[camera] = None

    # Episodes frequently end during/right after the final placement, so the stage
    # never leaves the completion stage and the last subtask is not advanced inside
    # the loop. Flush it here so task completion reflects the placed final letter.
    if subtask_manager.finalize(task_state):
        print(
            f"[target_progress][{job.episode_name}] frame={frame_count - 1} "
            f"finalized_to_progress_idx={task_state.progress_idx}",
            flush=True,
        )

    # ---- Pass 2: SAM2 video-predictor propagation per segment ----
    # Qwen and the SAM2 video predictor together can exceed one GPU even though
    # grounding is finished before propagation. Peg-in-hole enables staged model
    # loading: release Qwen now, then load SAM2 for this episode.
    owns_video_tracker = video_tracker is None
    if owns_video_tracker:
        from perception.task_manager.qwen_runtime import reset_qwen_runtime

        reset_qwen_runtime()
        video_tracker = Sam2VideoSegmentTracker(config)

    tracked_masks: dict[str, dict[int, np.ndarray]] = {camera: {} for camera in tracked_cameras}
    skipped_segments: list[str] = []
    segment_diagnostics: list[dict[str, Any]] = []
    for camera in tracked_cameras:
        for segment in segments[camera]:
            seg_len = segment.end - segment.start + 1
            if segment.box is None:
                segment_diagnostics.append(summarize_segment_masks(segment, {}))
                skip_desc = (
                    f"{camera} frames[{segment.start},{segment.end}] "
                    f"prompt='{segment.prompt}'"
                )
                skipped_segments.append(skip_desc)
                print(
                    f"[track_segment][{job.episode_name}] {skip_desc} "
                    f"no grounding box, skipped",
                    flush=True,
                )
                continue
            if segment.render_mode == AREA_PLACEMENT_RENDER_MODE:
                fixed_mask = bbox_to_mask(
                    camera_frames[camera][segment.start].size,
                    segment.box,
                )
                local_masks = {local_idx: fixed_mask for local_idx in range(seg_len)}
            elif segment.render_mode == "static_shape":
                local_masks = {
                    local_idx: static_shape_mask(
                        camera_frames[camera][segment.start + local_idx].size,
                        segment.box,
                        segment.prompt,
                    )
                    for local_idx in range(seg_len)
                }
            else:
                seg_frames = [
                    camera_frames[camera][i]
                    for i in range(segment.start, segment.end + 1)
                ]
                assert video_tracker is not None
                local_masks = video_tracker.track_segment(seg_frames, segment.box)
            diagnostic = summarize_segment_masks(segment, local_masks)
            segment_diagnostics.append(diagnostic)
            if diagnostic["produced_frames"] != seg_len or diagnostic["nonempty_frames"] == 0:
                skipped_segments.append(
                    f"{camera} frames[{segment.start},{segment.end}] prompt='{segment.prompt}' "
                    f"mask coverage produced={diagnostic['produced_frames']}/{seg_len} "
                    f"nonempty={diagnostic['nonempty_frames']}/{seg_len}"
                )
            for local_idx, mask in local_masks.items():
                tracked_masks[camera][segment.start + local_idx] = mask
            print(
                f"[track_segment][{job.episode_name}] {camera} "
                f"frames[{segment.start},{segment.end}] len={seg_len} "
                f"prompt='{segment.prompt}' mode={segment.render_mode} "
                f"masked={len(local_masks)}",
                flush=True,
            )

    if export_semantic_grounding:
        patch_maps = masks_to_patch_attention_maps(
            tracked_masks=tracked_masks,
            image_sizes={camera: camera_frames[camera][0].size for camera in ALL_CAMERAS},
            frame_count=frame_count,
            model_input_resolution=model_input_resolution,
        )
        write_semantic_grounding_parquet(
            job.semantic_grounding_parquet_path or job.output_parquet_path,
            maps_by_camera=patch_maps,
            columns=semantic_grounding_columns(config),
            model_input_resolution=model_input_resolution,
        )
    if export_semantic_grounding and not render_highlighting:
        if job.attention_output_video_paths is None:
            raise RuntimeError("attention export has no output video paths")
        for camera, output_path in job.attention_output_video_paths.items():
            policy_frames = [
                resize_policy_image(frame, output_resolution)
                for frame in camera_frames[camera]
            ]
            write_rendered_video_stream(output_path, policy_frames, fps=output_fps)

    # ---- Pass 3: overlay masks on tracked frames, then write videos ----
    # Both mode combines these highlighted videos with the patch-map columns
    # already written to the same dataset's episode parquet.
    if render_highlighting:
        rendered_by_camera: dict[str, list[Image.Image]] = {}
        if block_mask_global:
            rendered_by_camera[GLOBAL_CAMERA] = overlay_masked_frames(
                camera_frames[GLOBAL_CAMERA],
                tracked_masks[GLOBAL_CAMERA],
                frame_count,
                render_enabled,
                render_config,
            )
        else:
            rendered_by_camera[GLOBAL_CAMERA] = global_rendered
        for camera in WRIST_CAMERAS:
            rendered_by_camera[camera] = overlay_masked_frames(
                camera_frames[camera],
                tracked_masks[camera],
                frame_count,
                render_enabled,
                render_config,
            )
        for camera, output_path in job.output_video_paths.items():
            policy_frames = [
                resize_policy_image(frame, output_resolution)
                for frame in rendered_by_camera[camera]
            ]
            write_rendered_video_stream(output_path, policy_frames, fps=output_fps)

    diagnostic_path = artifact_dir / f"{job.episode_name}_quality.json"
    diagnostic_path.write_text(json.dumps({
        "episode_name": job.episode_name,
        "frame_count": frame_count,
        "observed_subtask_count": completed_subtasks,
        "initial_task_state": initial_task_state,
        "final_task_state": json.loads(dump_task_state(task_state)),
        "segments": segment_diagnostics,
        "failed_segments": skipped_segments,
    }, indent=2), encoding="utf-8")
    success_flags = build_stage_aware_success_flags(
        config=config,
        target_word=task_state.target_word,
        blocks=init_blocks,
        task_state=task_state,
        rendered_frame_count=frame_count,
        expected_frame_count=frame_count,
        output_video_paths=job.output_video_paths,
        skipped_segments=skipped_segments,
    )
    result = EpisodeResult(
        episode_name=job.episode_name,
        worker_id=worker_id,
        gpu_id=gpu_id,
        frame_count=frame_count,
        stage_switch_count=stage_switch_count,
        target_word=task_state.target_word,
        task_instruction=(
            task_state.task_context.instruction if task_state.task_context is not None else None
        ),
        task_context_source=(
            task_state.task_context.source if task_state.task_context is not None else None
        ),
        elapsed_sec=time.time() - episode_start,
        quality_report_path=str(diagnostic_path.resolve()),
        **success_flags,
    )
    if owns_video_tracker:
        import gc
        import torch

        del video_tracker
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def run_worker(
    worker_id: int,
    gpu_id: int,
    jobs: list[StageAwareEpisodeJob],
    args: argparse.Namespace,
    config: dict[str, Any],
    result_queue,
    log_path: str,
    artifact_dir: str,
) -> None:
    configure_process_logging(Path(log_path))
    # Give each spawned process exactly one physical GPU before the first CUDA
    # call. Qwen uses device_map="auto", so torch.cuda.set_device alone would
    # still let one worker shard itself across every visible card.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"worker {worker_id} expected one visible CUDA device for physical "
            f"GPU {gpu_id}, got {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(0)
    print(
        f"[worker_device] worker={worker_id} physical_gpu={gpu_id} "
        "logical_cuda_device=0 visible_device_count=1",
        flush=True,
    )

    from perception.task_manager.qwen_runtime import initialize_qwen_runtime_from_config

    worker_start = time.time()
    worker_results: list[dict[str, Any]] = []
    summary_payload: dict[str, Any]

    try:
        staged_model_loading = bool(config["tracker"].get("staged_model_loading", False))
        stage_predictor = build_stage_predictor(config)
        video_tracker = None
        if not staged_model_loading:
            initialize_qwen_runtime_from_config(config)
            # Build the SAM2 video predictor once per worker and reuse it across
            # episodes/segments when both models fit together.
            video_tracker = Sam2VideoSegmentTracker(config)
        for job in jobs:
            try:
                if staged_model_loading:
                    initialize_qwen_runtime_from_config(config)
                result = process_stage_aware_episode(
                    job=job,
                    worker_id=worker_id,
                    gpu_id=gpu_id,
                    artifact_dir=Path(artifact_dir),
                    config=config,
                    recognition_padding=args.recognition_padding,
                    re_recognize_on_stage_switch=args.re_recognize_on_stage_switch,
                    fps_mode=args.fps_mode,
                    fixed_fps=args.fixed_fps,
                    state_column=args.state_column,
                    global_render_mode=args.global_render_mode,
                    render_highlighting=args.render_highlighting,
                    export_semantic_grounding=args.export_semantic_grounding,
                    video_tracker=video_tracker,
                    stage_predictor=stage_predictor,
                )
            except Exception as exc:  # noqa: BLE001
                result = EpisodeResult(
                    episode_name=job.episode_name,
                    success=False,
                    vlm_success=False,
                    dino_success=False,
                    render_success=False,
                    worker_id=worker_id,
                    gpu_id=gpu_id,
                    elapsed_sec=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            worker_results.append(asdict(result))

        summary_payload = {
            "worker_id": worker_id,
            "gpu_id": gpu_id,
            "success": True,
            "elapsed_sec": time.time() - worker_start,
            "results": worker_results,
        }
    except Exception as exc:  # noqa: BLE001
        summary_payload = {
            "worker_id": worker_id,
            "gpu_id": gpu_id,
            "success": False,
            "elapsed_sec": time.time() - worker_start,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "results": worker_results,
        }

    result_queue.put(summary_payload)


def main() -> int:
    args = parse_args()
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    summary_path = Path(args.summary_path) if args.summary_path else dst_root / "render_stage_aware_summary.json"
    merge_summary_path = Path(args.merge_summary) if args.merge_summary else None
    partial_summary_path = summary_path
    if merge_summary_path is not None and partial_summary_path.resolve() == merge_summary_path.resolve():
        partial_summary_path = summary_path.with_name(f"{summary_path.stem}.partial_rerun.json")
    log_path = build_log_path(Path(args.log_dir))
    artifact_dir = build_artifact_dir(log_path)
    configure_process_logging(log_path)
    parsed_gpu_ids = parse_physical_gpu_ids(args.gpu_ids)
    gpu_ids = expand_per_gpu_worker_assignments(parsed_gpu_ids, args.num_workers)
    config = build_batch_config(args)
    semantic_intent_mode = semantic_intent_injection(config)
    highlighting_root, attention_root = resolve_output_roots(
        dst_root,
        semantic_intent_mode,
        args.attention_dst_root,
    )
    model_input_resolution = resolve_model_input_resolution(config)
    output_resolution = resolve_output_resolution(config)
    if args.export_semantic_grounding and semantic_intent_mode not in {
        SEMANTIC_INTENT_ATTENTION,
        SEMANTIC_INTENT_BOTH,
    }:
        raise ValueError(
            "--export-semantic-grounding requires "
            "task.semantic_intent_injection: attention or both"
        )
    args.export_semantic_grounding = attention_root is not None
    args.render_highlighting = highlighting_root is not None
    if args.export_semantic_grounding:
        assert attention_root is not None
        if src_root.resolve() == attention_root.resolve():
            raise ValueError("attention bbox export requires distinct --src-root and --dst-root")
        if args.global_render_mode != "block_mask":
            raise ValueError("attention bbox export requires --global-render-mode=block_mask")
    task_context_provider = DatasetTaskContextProvider.from_config(
        config,
        src_root,
        path_override=Path(args.task_context_file) if args.task_context_file else None,
    )
    camera_keys = camera_video_keys(config)
    video_keys = tuple(camera_keys.values())
    requested_episode_names = parse_episode_names(args.episode_ids)
    if args.failed_from_summary:
        failed_names = failed_episode_names_from_summary(Path(args.failed_from_summary))
        if not failed_names and not requested_episode_names:
            print(f"no failed episodes in summary: {args.failed_from_summary}", flush=True)
            return 0
        requested_episode_names.update(failed_names)
    partial_rerender = bool(requested_episode_names)
    if partial_rerender:
        args.skip_copy = True
        if merge_summary_path is None and args.summary_path is None:
            summary_path = dst_root / "render_stage_aware_summary.partial_rerun.json"
            partial_summary_path = summary_path
    if merge_summary_path is not None and partial_summary_path.resolve() == merge_summary_path.resolve():
        partial_summary_path = summary_path.with_name(f"{summary_path.stem}.partial_rerun.json")

    print(f"run log: {log_path}", flush=True)
    print(f"artifact dir: {artifact_dir}", flush=True)
    print(f"config: {Path(args.config).resolve()}", flush=True)
    print(f"semantic_intent_injection={semantic_intent_mode}", flush=True)
    print(
        f"output_roots highlighting={highlighting_root} attention={attention_root}",
        flush=True,
    )
    if args.export_semantic_grounding:
        grid_shape = patch_grid_shape(model_input_resolution)
        print(
            f"semantic_grounding_export=true columns={semantic_grounding_columns(config)} "
            f"model_input_resolution={model_input_resolution} patch_size={VIT_PATCH_SIZE} "
            f"patch_grid={grid_shape} output_resolution={output_resolution} "
            f"videos=direct_resize_no_padding",
            flush=True,
        )
    if requested_episode_names:
        print(f"requested_episodes={sorted(requested_episode_names)}", flush=True)
    if merge_summary_path is not None:
        print(f"merge_summary={merge_summary_path}", flush=True)
        print(f"partial_summary={partial_summary_path}", flush=True)
    print(
        f"physical_gpus={parsed_gpu_ids} workers_per_gpu={args.num_workers} "
        f"total_worker_processes={len(gpu_ids)} assignment={gpu_ids}",
        flush=True,
    )

    if not src_root.is_dir():
        raise FileNotFoundError(f"input dataset root not found: {src_root}")
    validate_common_dataset_files(src_root)
    if not sorted(src_root.glob(f"videos/chunk-*/{camera_keys[GLOBAL_CAMERA]}")):
        raise FileNotFoundError(
            f"no global video directories found under {src_root} "
            f"for key {camera_keys[GLOBAL_CAMERA]!r}"
        )

    validate_gpu_environment(parsed_gpu_ids)
    print(f"camera_keys={camera_keys}", flush=True)
    print(
        f"task_context_source={task_context_provider.source} "
        f"path={task_context_provider.source_path}",
        flush=True,
    )
    if requires_dino(config):
        # Load GroundingDINO lazily after each worker selects its assigned CUDA
        # device. Loading it in the parent would silently consume GPU 0.
        print("dino: initialization deferred to assigned GPU workers", flush=True)
    else:
        print("dino: disabled (qwen-only)", flush=True)

    output_roots = tuple(dict.fromkeys(
        root for root in (highlighting_root, attention_root) if root is not None
    ))
    if not args.skip_copy:
        for output_root in output_roots:
            prepare_output_dataset(
                src_root=src_root,
                dst_root=output_root,
                video_keys=video_keys,
                overwrite=args.overwrite,
            )
    else:
        for output_root in output_roots:
            output_root.mkdir(parents=True, exist_ok=True)
    for output_root in output_roots:
        validate_common_dataset_files(output_root)

    jobs = collect_stage_aware_jobs(
        src_root=src_root,
        dst_root=highlighting_root or attention_root or dst_root,
        attention_dst_root=attention_root,
        camera_keys=camera_keys,
        episode_pattern=args.episode_pattern,
        limit=args.limit,
        episode_names=requested_episode_names or None,
        task_context_provider=task_context_provider,
    )
    if args.limit is not None and not args.skip_copy and not partial_rerender:
        selected_episode_indices = [job.episode_index for job in jobs]
        for output_root in output_roots:
            prune_to_contiguous_episode_subset(
                output_root,
                episode_indices=selected_episode_indices,
                video_keys=video_keys,
            )
        print(
            f"standalone_subset=true episode_indices={selected_episode_indices}",
            flush=True,
        )
    job_groups = split_jobs_for_workers(jobs, gpu_ids)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    for worker_id, (gpu_id, worker_jobs) in enumerate(zip(gpu_ids, job_groups)):
        process = ctx.Process(
            target=run_worker,
            args=(worker_id, gpu_id, worker_jobs, args, config, result_queue, str(log_path), str(artifact_dir)),
            daemon=False,
        )
        process.start()
        processes.append(process)

    exit_code = 0
    for process in processes:
        process.join()
        if process.exitcode != 0:
            exit_code = 1

    worker_summaries: list[dict[str, Any]] = []
    while True:
        try:
            worker_summaries.append(result_queue.get_nowait())
        except queue.Empty:
            break

    seen_worker_ids = {summary["worker_id"] for summary in worker_summaries}
    for worker_id, gpu_id in enumerate(gpu_ids):
        if worker_id not in seen_worker_ids:
            worker_summaries.append(
                {
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "success": False,
                    "elapsed_sec": 0.0,
                    "error": "worker exited without publishing a summary",
                    "results": [],
                }
            )
            exit_code = 1

    summary = merge_worker_summaries(
        worker_summaries,
        output_path=partial_summary_path,
        expected_episode_jobs=len(jobs),
    )
    render_mode = {
        SEMANTIC_INTENT_HIGHLIGHTING: "stage_aware_three_camera",
        SEMANTIC_INTENT_ATTENTION: "semantic_grounding_policy_resolution_rgb",
        SEMANTIC_INTENT_BOTH: "stage_aware_three_camera+semantic_grounding",
    }[semantic_intent_mode]
    summary["render_mode"] = render_mode
    summary["output_roots"] = {
        "highlighting": str(highlighting_root) if highlighting_root is not None else None,
        "attention": str(attention_root) if attention_root is not None else None,
    }
    summary["config_path"] = str(Path(args.config).resolve())
    summary["model_input_resolution"] = list(model_input_resolution)
    summary["video_output_resolution"] = list(output_resolution)
    summary["video_resize_mode"] = "direct_resize_no_padding"
    summary["vit_patch_size"] = VIT_PATCH_SIZE
    summary["patch_grid_shape"] = list(patch_grid_shape(model_input_resolution))
    summary["task_context_source"] = task_context_provider.source
    summary["task_context_path"] = (
        str(task_context_provider.source_path)
        if task_context_provider.source_path is not None
        else None
    )
    rendered_episode_indices = sorted(
        {
            int(str(result["episode_name"]).removeprefix("episode_"))
            for worker_summary in worker_summaries
            for result in worker_summary.get("results", [])
            if int(result.get("frame_count") or 0) > 0 and result.get("episode_name")
        }
    )
    metadata_update_error: str | None = None
    try:
        if rendered_episode_indices:
            for output_root in output_roots:
                update_episode_image_stats(
                    output_root,
                    video_keys=video_keys,
                    episode_indices=rendered_episode_indices,
                )
                update_info_video_metadata(output_root, video_keys)
            if attention_root is not None:
                update_info_semantic_grounding_features(
                    attention_root,
                    columns=semantic_grounding_columns(config),
                    model_input_resolution=model_input_resolution,
                )
        summary["metadata_updated_episode_indices"] = rendered_episode_indices
        summary["image_stats_updated"] = bool(rendered_episode_indices)
    except Exception as exc:  # noqa: BLE001
        metadata_update_error = f"{type(exc).__name__}: {exc}"
        summary["metadata_update_error"] = metadata_update_error
        summary["image_stats_updated"] = False
        exit_code = 1
    partial_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if merge_summary_path is not None:
        summary = merge_rerun_worker_summaries(
            merge_summary_path,
            worker_summaries,
            expected_rerun_episode_names={job.episode_name for job in jobs},
            output_path=merge_summary_path,
            render_mode=render_mode,
            config_path=Path(args.config),
        )
        summary["task_context_source"] = task_context_provider.source
        summary["task_context_path"] = (
            str(task_context_provider.source_path)
            if task_context_provider.source_path is not None
            else None
        )
        summary["model_input_resolution"] = list(model_input_resolution)
        summary["video_output_resolution"] = list(output_resolution)
        summary["video_resize_mode"] = "direct_resize_no_padding"
        summary["vit_patch_size"] = VIT_PATCH_SIZE
        summary["patch_grid_shape"] = list(patch_grid_shape(model_input_resolution))
        summary["output_roots"] = {
            "highlighting": str(highlighting_root) if highlighting_root is not None else None,
            "attention": str(attention_root) if attention_root is not None else None,
        }
        summary["metadata_updated_episode_indices"] = rendered_episode_indices
        summary["image_stats_updated"] = bool(rendered_episode_indices) and not metadata_update_error
        if metadata_update_error:
            summary["metadata_update_error"] = metadata_update_error
        merge_summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary_path = merge_summary_path

    if summary.get("episode_result_count_mismatch"):
        print(
            "WARNING: episode result rows != jobs — a worker may have been killed (e.g. OOM) before finishing.",
            f"expected={summary.get('expected_episode_jobs')} got={summary.get('total_episode_results')}",
            flush=True,
        )
    if summary.get("failed_episode_count", 0) > 0:
        print(
            f"NOTE: {summary['failed_episode_count']} episode(s) failed; see failed_episodes in summary JSON.",
            flush=True,
        )
    if summary["failed_worker_count"] > 0 or summary["failed_episode_count"] > 0:
        exit_code = 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
