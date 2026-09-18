"""Shared helpers for offline dataset rendering (video I/O, workers, logging)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from perception.semantic_grounder.dino import DinoClient
from common.vision.bbox import bbox_to_int
from intent.drawing import DEFAULT_BLOCK_FILL, DEFAULT_PLACEMENT_FILL
from common.task_types import TargetObject, PlacementCandidate, TaskState
from common.config_loader import PROJECT_ROOT

DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs"
DEFAULT_STATE_COLUMN = "state"
DEFAULT_RECOGNITION_PADDING = 4


@dataclass
class EpisodeResult:
    episode_name: str
    success: bool
    vlm_success: bool = False
    dino_success: bool = False
    render_success: bool = False
    task_complete: bool = False
    worker_id: int = 0
    gpu_id: int = 0
    frame_count: int = 0
    stage_switch_count: int = 0
    target_word: str | None = None
    task_instruction: str | None = None
    task_context_source: str | None = None
    elapsed_sec: float = 0.0
    error: str | None = None
    dino_missing_letters: list[str] | None = None
    render_failure_reason: str | None = None
    task_incomplete_reason: str | None = None
    quality_report_path: str | None = None


class TeeStream:
    def __init__(self, original_stream, log_path: Path) -> None:
        self.original_stream = original_stream
        self.log_handle = log_path.open("a", encoding="utf-8", buffering=1)

    def write(self, data: str) -> int:
        self.original_stream.write(data)
        self.log_handle.write(data)
        return len(data)

    def flush(self) -> None:
        self.original_stream.flush()
        self.log_handle.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.original_stream, "isatty", lambda: False)())


def parse_physical_gpu_ids(raw_value: str) -> list[int]:
    """Physical GPUs to use in this run; each device index appears at most once."""
    raw_ids = [int(item.strip()) for item in raw_value.split(",") if item.strip()]
    if not raw_ids:
        raise ValueError("expected at least one gpu id in --gpu-ids (e.g. 0 or 0,1)")
    seen: set[int] = set()
    ordered: list[int] = []
    for gid in raw_ids:
        if gid in seen:
            raise ValueError(
                f"duplicate gpu id {gid} in --gpu-ids; list each physical GPU once "
                "(use --num-workers for multiple processes on the same card)."
            )
        seen.add(gid)
        ordered.append(gid)
    return ordered


def expand_per_gpu_worker_assignments(physical_gpu_ids: list[int], workers_per_gpu: int) -> list[int]:
    """One entry per worker process: each physical GPU id repeated ``workers_per_gpu`` times."""
    if workers_per_gpu < 1:
        raise ValueError(f"--num-workers (per GPU) must be >= 1, got {workers_per_gpu}")
    assignments: list[int] = []
    for gid in physical_gpu_ids:
        assignments.extend([gid] * workers_per_gpu)
    return assignments


def build_log_path(log_dir: Path, *, prefix: str = "render_dataset") -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"{prefix}_{timestamp}.log"


def build_artifact_dir(log_path: Path) -> Path:
    artifact_dir = log_path.parent / f"{log_path.stem}_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def configure_process_logging(log_path: Path) -> None:
    sys.stdout = TeeStream(sys.stdout, log_path)
    sys.stderr = TeeStream(sys.stderr, log_path)


def get_default_font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def validate_gpu_environment(physical_gpu_ids: list[int]) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false; CUDA is required")
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("no CUDA devices visible to PyTorch")
    for gpu_id in physical_gpu_ids:
        if gpu_id < 0 or gpu_id >= device_count:
            raise RuntimeError(f"requested gpu id {gpu_id} is out of range for device_count={device_count}")


def health_check_dino(config: dict[str, Any] | None = None) -> dict[str, Any]:
    client = DinoClient(config=config)
    return client.health_check()


def ignore_rendered_stream_mp4s(video_keys: str | Sequence[str]):
    key_set = {video_keys} if isinstance(video_keys, str) else set(video_keys)

    def ignore(directory: str, names: list[str]) -> list[str]:
        path = Path(directory)
        if path.name in key_set:
            return [name for name in names if name.endswith(".mp4")]
        return []

    return ignore


def prepare_output_dataset(
    src_root: Path,
    dst_root: Path,
    video_keys: str | Sequence[str],
    overwrite: bool = False,
) -> None:
    if dst_root.exists():
        if overwrite:
            shutil.rmtree(dst_root)
        elif any(dst_root.iterdir()):
            raise RuntimeError(f"output dataset already exists and is not empty: {dst_root}")
        else:
            shutil.rmtree(dst_root)

    shutil.copytree(src_root, dst_root, ignore=ignore_rendered_stream_mp4s(video_keys))


def split_jobs_for_workers(jobs: list[Any], gpu_ids: list[int]) -> list[list[Any]]:
    job_groups: list[list[Any]] = [[] for _ in gpu_ids]
    for idx, job in enumerate(jobs):
        job_groups[idx % len(gpu_ids)].append(job)
    return job_groups


def load_episode_states(parquet_path: Path, state_column: str) -> list[Sequence[float]]:
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


def get_video_metadata(video_path: Path) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        probed = get_video_metadata_ffprobe(video_path)
        if probed is not None:
            return probed
        raise RuntimeError(f"failed to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    if width <= 0 or height <= 0:
        probed = get_video_metadata_ffprobe(video_path)
        if probed is not None:
            return probed
        raise RuntimeError(f"invalid video dimensions for {video_path}: {(width, height)}")
    return fps, frame_count, width, height


def get_video_metadata_ffprobe(video_path: Path) -> tuple[float, int, int, int] | None:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    payload = json.loads(completed.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        return None
    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    frame_count = int(stream.get("nb_frames") or 0)
    raw_fps = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1")
    try:
        fps = float(Fraction(raw_fps))
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    if width <= 0 or height <= 0:
        return None
    return fps, frame_count, width, height


def is_av1_video(video_path: Path) -> bool:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return completed.stdout.strip().lower() == "av1"


def iter_video_frames_ffmpeg(video_path: Path):
    _fps, _frame_count, width, height = get_video_metadata(video_path)
    frame_size = width * height * 3
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    try:
        while True:
            buffer = process.stdout.read(frame_size)
            if not buffer:
                break
            if len(buffer) != frame_size:
                raise RuntimeError(f"short ffmpeg frame read for {video_path}: {len(buffer)} != {frame_size}")
            array = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 3)).copy()
            yield Image.fromarray(array, mode="RGB")
    finally:
        if process.stdout is not None:
            process.stdout.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr is not None else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed to decode {video_path} with exit code {return_code}: {stderr.strip()}")


def iter_video_frames(video_path: Path):
    if is_av1_video(video_path):
        yield from iter_video_frames_ffmpeg(video_path)
        return

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        yield from iter_video_frames_ffmpeg(video_path)
        return

    frame_count = 0
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_count += 1
            yield Image.fromarray(frame_rgb)
    finally:
        capture.release()
    if frame_count == 0:
        yield from iter_video_frames_ffmpeg(video_path)


def choose_output_fps(source_fps: float, fps_mode: str, fixed_fps: float) -> float:
    if fps_mode == "fixed":
        return fixed_fps
    if source_fps > 0:
        return source_fps
    return fixed_fps


def write_rendered_video(output_path: Path, rendered_frames: list[Image.Image], fps: float) -> None:
    if not rendered_frames:
        raise ValueError("rendered_frames must not be empty")

    first_frame = rendered_frames[0]
    width, height = first_frame.size
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{output_path.stem}_", dir=str(output_path.parent)) as temp_dir:
        temp_dir_path = Path(temp_dir)
        for frame_index, frame in enumerate(rendered_frames):
            if frame.size != (width, height):
                raise RuntimeError("all rendered frames must have the same size")
            frame_path = temp_dir_path / f"frame_{frame_index:06d}.png"
            frame.save(frame_path)

        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            str(temp_dir_path / "frame_%06d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        completed = subprocess.run(
            ffmpeg_command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed for {output_path} with exit code {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )


def render_pick_and_place_fill(
    image: Image.Image,
    target_block: TargetObject | None,
    target_placement: PlacementCandidate | None,
    block_fill: tuple[int, int, int, int] = DEFAULT_BLOCK_FILL,
    placement_fill: tuple[int, int, int, int] = DEFAULT_PLACEMENT_FILL,
) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if target_block is not None:
        draw.rectangle(bbox_to_int(target_block.bbox_xyxy), fill=block_fill)
    if target_placement is not None:
        draw.rectangle(bbox_to_int(target_placement.bbox_xyxy), fill=placement_fill)

    return Image.alpha_composite(base, overlay).convert("RGB")


def dump_task_state(task_state: TaskState) -> str:
    return json.dumps(asdict(task_state), ensure_ascii=False, indent=2)


def log_target_switch(
    episode_name: str,
    frame_index: int,
    previous_summary: str,
    new_summary: str,
) -> None:
    if previous_summary == new_summary:
        return
    print(
        f"[target_switch][{episode_name}] frame={frame_index} "
        f"from ({previous_summary}) to ({new_summary})",
        flush=True,
    )


def parse_episode_number(episode_name: str) -> int:
    match = re.fullmatch(r"episode_(\d+)", episode_name)
    if match:
        return int(match.group(1))
    trailing_digits = re.search(r"(\d+)$", episode_name)
    if trailing_digits:
        return int(trailing_digits.group(1))
    raise ValueError(f"unable to parse episode number from {episode_name!r}")


def collect_episode_ids(
    results: list[dict[str, Any]],
    predicate,
) -> list[int]:
    return sorted(parse_episode_number(result["episode_name"]) for result in results if predicate(result))


def merge_worker_summaries(
    worker_summaries: list[dict[str, Any]],
    output_path: Path,
    expected_episode_jobs: int | None = None,
) -> dict[str, Any]:
    all_results: list[dict[str, Any]] = []
    failed_workers: list[dict[str, Any]] = []

    for worker_summary in worker_summaries:
        all_results.extend(worker_summary.get("results", []))
        if not worker_summary.get("success", False):
            failed_workers.append(
                {
                    "worker_id": worker_summary["worker_id"],
                    "gpu_id": worker_summary["gpu_id"],
                    "error": worker_summary.get("error"),
                }
            )

    failed_episodes = [result for result in all_results if not result.get("success", False)]
    summary = {
        "worker_count": len(worker_summaries),
        "success_episode_count": sum(1 for result in all_results if result.get("success", False)),
        "vlm_success_episode_count": sum(1 for result in all_results if result.get("vlm_success", False)),
        "dino_success_episode_count": sum(1 for result in all_results if result.get("dino_success", False)),
        "render_success_episode_count": sum(1 for result in all_results if result.get("render_success", False)),
        "failed_episode_count": len(failed_episodes),
        "failed_episode_ids": collect_episode_ids(all_results, lambda result: not result.get("success", False)),
        "failed_worker_count": len(failed_workers),
        "failed_workers": failed_workers,
        "failed_episodes": failed_episodes,
        "worker_summaries": worker_summaries,
    }
    summary["total_episode_results"] = len(all_results)
    if expected_episode_jobs is not None:
        summary["expected_episode_jobs"] = expected_episode_jobs
        if len(all_results) != expected_episode_jobs:
            summary["episode_result_count_mismatch"] = True
            summary["missing_episode_result_rows"] = max(0, expected_episode_jobs - len(all_results))
        else:
            summary["episode_result_count_mismatch"] = False
            summary["missing_episode_result_rows"] = 0
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def render_recognized_blocks(image: Image.Image, blocks: list[TargetObject]) -> Image.Image:
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    font = get_default_font()

    for block in blocks:
        x1, y1, x2, y2 = bbox_to_int(block.bbox_xyxy)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        label = f"id={block.id} {block.label or block.letter or '?'}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        tx1 = x1
        ty1 = max(0, y1 - text_h - 6)
        tx2 = tx1 + text_w + 6
        ty2 = ty1 + text_h + 4
        draw.rectangle([tx1, ty1, tx2, ty2], fill=(0, 0, 0))
        draw.text((tx1 + 3, ty1 + 2), label, fill=(255, 255, 255), font=font)

    return rendered


def save_recognition_artifact(
    artifact_dir: Path,
    episode_name: str,
    frame_index: int,
    image: Image.Image,
    blocks: list[TargetObject],
) -> None:
    episode_dir = artifact_dir / episode_name
    episode_dir.mkdir(parents=True, exist_ok=True)
    rendered = render_recognized_blocks(image, blocks)
    output_path = episode_dir / f"frame_{frame_index:06d}_recognized_blocks.png"
    rendered.save(output_path)
