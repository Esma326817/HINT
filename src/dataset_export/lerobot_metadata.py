"""Validate and refresh LeRobot v2.1 metadata after video rendering."""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from dataset_export.preprocessing.dataset_utils import iter_video_frames

REQUIRED_META_FILES = (
    "info.json",
    "episodes.jsonl",
    "episodes_stats.jsonl",
    "tasks.jsonl",
)


def validate_common_dataset_files(dataset_root: Path) -> None:
    missing = [
        str(dataset_root / "meta" / name)
        for name in REQUIRED_META_FILES
        if not (dataset_root / "meta" / name).is_file()
    ]
    if not sorted(dataset_root.glob("data/chunk-*/episode_*.parquet")):
        missing.append(str(dataset_root / "data/chunk-*/episode_*.parquet"))
    if missing:
        raise FileNotFoundError(f"incomplete LeRobot dataset; missing: {missing}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_write_text(path: Path, text: str) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def estimate_num_samples(
    dataset_len: int,
    min_num_samples: int = 100,
    max_num_samples: int = 10_000,
    power: float = 0.75,
) -> int:
    if dataset_len <= 0:
        return 0
    minimum = min(min_num_samples, dataset_len)
    return max(minimum, min(int(dataset_len**power), max_num_samples))


def sample_indices(data_len: int) -> list[int]:
    count = estimate_num_samples(data_len)
    if count <= 0:
        return []
    return np.round(np.linspace(0, data_len - 1, count)).astype(int).tolist()


def _video_frame_count(video_path: Path) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = (json.loads(completed.stdout or "{}").get("streams") or [])
    if not streams:
        raise RuntimeError(f"ffprobe returned no video stream: {video_path}")
    raw = streams[0].get("nb_read_frames") or streams[0].get("nb_frames")
    count = int(raw or 0)
    if count <= 0:
        raise RuntimeError(f"invalid video frame count for {video_path}: {raw!r}")
    return count


def compute_video_image_stats(video_path: Path) -> dict[str, Any]:
    """Compute the v2.1 per-episode image stats with bounded uniform sampling."""
    frame_count = _video_frame_count(video_path)
    wanted = set(sample_indices(frame_count))
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    sampled = 0
    for index, frame in enumerate(iter_video_frames(video_path)):
        if index not in wanted:
            continue
        array = np.asarray(frame.convert("RGB"), dtype=np.uint8)
        height, width = array.shape[:2]
        stride = int(max(width, height) / 150) if max(width, height) >= 300 else 1
        stride = max(1, stride)
        values = array[::stride, ::stride].astype(np.float64) / 255.0
        flat = values.reshape(-1, 3)
        frame_min = flat.min(axis=0)
        frame_max = flat.max(axis=0)
        minimum = frame_min if minimum is None else np.minimum(minimum, frame_min)
        maximum = frame_max if maximum is None else np.maximum(maximum, frame_max)
        total += flat.sum(axis=0)
        total_sq += np.square(flat).sum(axis=0)
        pixel_count += len(flat)
        sampled += 1
    if sampled != len(wanted) or minimum is None or maximum is None or pixel_count <= 0:
        raise RuntimeError(
            f"failed to sample expected frames from {video_path}: "
            f"expected={len(wanted)} sampled={sampled}"
        )
    mean = total / pixel_count
    variance = np.maximum(total_sq / pixel_count - np.square(mean), 0.0)

    def shaped(values: np.ndarray) -> list[list[list[float]]]:
        return values.reshape(3, 1, 1).tolist()

    return {
        "min": shaped(minimum),
        "max": shaped(maximum),
        "mean": shaped(mean),
        "std": shaped(np.sqrt(variance)),
        "count": [sampled],
    }


def _episode_video(
    dataset_root: Path,
    video_key: str,
    episode_index: int,
) -> Path:
    chunk = episode_index // 1000
    return (
        dataset_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / video_key
        / f"episode_{episode_index:06d}.mp4"
    )


def update_episode_image_stats(
    dataset_root: Path,
    *,
    video_keys: Sequence[str],
    episode_indices: Iterable[int],
) -> None:
    stats_path = dataset_root / "meta" / "episodes_stats.jsonl"
    rows = {int(row["episode_index"]): row for row in _read_jsonl(stats_path)}
    for episode_index in sorted(set(int(value) for value in episode_indices)):
        if episode_index not in rows:
            raise KeyError(f"episode {episode_index} missing from {stats_path}")
        episode_stats = rows[episode_index].setdefault("stats", {})
        for video_key in video_keys:
            video_path = _episode_video(dataset_root, video_key, episode_index)
            if not video_path.is_file():
                raise FileNotFoundError(f"rendered video missing: {video_path}")
            episode_stats[video_key] = compute_video_image_stats(video_path)
    payload = "".join(
        json.dumps(rows[index], ensure_ascii=False, separators=(",", ":")) + "\n"
        for index in sorted(rows)
    )
    _atomic_write_text(stats_path, payload)


def _probe_video(video_path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,avg_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(completed.stdout or "{}").get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe returned no stream for {video_path}")
    return dict(streams[0])


def update_info_video_metadata(dataset_root: Path, video_keys: Sequence[str]) -> None:
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features") or {}
    output_fps: float | int | None = None
    for video_key in video_keys:
        candidates = sorted(dataset_root.glob(f"videos/chunk-*/{video_key}/episode_*.mp4"))
        if not candidates:
            raise FileNotFoundError(f"no rendered videos found for {video_key}")
        probe = _probe_video(candidates[0])
        width = int(probe["width"])
        height = int(probe["height"])
        feature = features.get(video_key)
        if not isinstance(feature, dict):
            raise KeyError(f"video feature {video_key!r} missing from {info_path}")
        feature["shape"] = [height, width, 3]
        video_info = feature.setdefault("info", {})
        video_info["video.height"] = height
        video_info["video.width"] = width
        video_info["video.codec"] = str(probe.get("codec_name") or "h264")
        video_info["video.pix_fmt"] = str(probe.get("pix_fmt") or "yuv420p")
        raw_fps = str(probe.get("avg_frame_rate") or "0/1")
        fps = float(Fraction(raw_fps))
        output_fps = int(fps) if fps.is_integer() else fps
        video_info["video.fps"] = output_fps
        video_info["video.channels"] = 3
    if output_fps is not None:
        info["fps"] = output_fps
    _atomic_write_text(
        info_path,
        json.dumps(info, ensure_ascii=False, indent=4) + "\n",
    )


def prune_to_contiguous_episode_subset(
    dataset_root: Path,
    *,
    episode_indices: Iterable[int],
    video_keys: Sequence[str],
) -> None:
    """Make a copied LeRobot v2.1 dataset a standalone 0..N-1 subset.

    ``--limit`` selects the first N source episodes. Copying the source tree
    without pruning leaves 83 parquet/meta rows but only N rendered videos,
    which is not loadable as a standalone dataset. This helper removes the
    unselected copied data and synchronizes episode-level metadata and totals.
    """
    selected = sorted(set(int(index) for index in episode_indices))
    if selected != list(range(len(selected))):
        raise ValueError(
            "standalone LeRobot subset must use contiguous episode indices 0..N-1; "
            f"got {selected}"
        )

    selected_set = set(selected)
    data_root = dataset_root / "data"
    for parquet_path in data_root.glob("chunk-*/episode_*.parquet"):
        episode_index = int(parquet_path.stem.removeprefix("episode_"))
        if episode_index not in selected_set:
            parquet_path.unlink()
    # Annotation backups are not LeRobot data and needlessly retain the full
    # source dataset in a small test export.
    for backup_path in data_root.glob("chunk-*/episode_*.parquet.bak"):
        backup_path.unlink()

    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    episode_rows = [
        row for row in _read_jsonl(episodes_path)
        if int(row["episode_index"]) in selected_set
    ]
    if len(episode_rows) != len(selected):
        raise ValueError(
            f"episode metadata mismatch while pruning {dataset_root}: "
            f"selected={len(selected)} rows={len(episode_rows)}"
        )
    _atomic_write_text(
        episodes_path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in episode_rows
        ),
    )

    stats_path = dataset_root / "meta" / "episodes_stats.jsonl"
    stats_rows = [
        row for row in _read_jsonl(stats_path)
        if int(row["episode_index"]) in selected_set
    ]
    if len(stats_rows) != len(selected):
        raise ValueError(
            f"episode stats mismatch while pruning {dataset_root}: "
            f"selected={len(selected)} rows={len(stats_rows)}"
        )
    _atomic_write_text(
        stats_path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in stats_rows
        ),
    )

    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_episodes"] = len(selected)
    info["total_frames"] = sum(int(row["length"]) for row in episode_rows)
    info["total_videos"] = len(selected) * len(video_keys)
    chunk_size = int(info.get("chunks_size") or 1000)
    info["total_chunks"] = len({index // chunk_size for index in selected})
    info["splits"] = {"train": f"0:{len(selected)}"}
    _atomic_write_text(
        info_path,
        json.dumps(info, ensure_ascii=False, indent=4) + "\n",
    )
