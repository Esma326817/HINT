from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pattern.models.progress import PROGRESS_HEAD_BY_STAGE_ID

from .keys import LeRobotDataKeys
from .streaming import StreamingFrameStore

CAMERA_KEYS = LeRobotDataKeys().camera_keys
IMAGE_OFFSETS = (0,)
_LEGACY_IMAGE_OFFSETS = (-9, -6, -3, 0)
LOW_HISTORY = 12


def image_offsets_for_num_times(num_times: int) -> tuple[int, ...]:
    """Return the supported image sampling pattern for a checkpoint/config."""
    num_times = int(num_times)
    if num_times == 1:
        return IMAGE_OFFSETS
    if num_times == 4:
        return _LEGACY_IMAGE_OFFSETS
    raise ValueError(f"Unsupported image_encoder.num_times={num_times}; expected 1 or 4")


STAGE_TO_INDEX = {stage: stage - 1 for stage in range(1, 7)}
INDEX_TO_STAGE = {idx: stage for stage, idx in STAGE_TO_INDEX.items()}

STAGE_TO_PROGRESS_HEAD = {
    stage_id: head_index
    for stage_id, head_index in enumerate(PROGRESS_HEAD_BY_STAGE_ID, start=1)
}


def compute_stage_progress(stage_ids: np.ndarray) -> np.ndarray:
    """Linear progress [0, 1] within each contiguous stage segment in an episode."""

    progress = np.zeros(stage_ids.shape[0], dtype=np.float32)
    if stage_ids.shape[0] == 0:
        return progress
    start = 0
    for i in range(1, stage_ids.shape[0] + 1):
        if i == stage_ids.shape[0] or stage_ids[i] != stage_ids[start]:
            seg_len = i - start
            denom = max(seg_len - 1, 1)
            for j in range(seg_len):
                progress[start + j] = j / denom
            start = i
    return progress


@dataclass
class EpisodeData:
    episode_index: int
    parquet_path: Path
    low_dim: np.ndarray
    stage: np.ndarray
    progress_head: np.ndarray
    progress: np.ndarray
    weight: np.ndarray

    @property
    def length(self) -> int:
        return int(self.stage.shape[0])


@dataclass(frozen=True)
class LeRobotPaths:
    root: Path
    data_path: str
    video_path: str
    chunks_size: int
    fps: float

    @classmethod
    def from_root(cls, dataset_root: str | Path) -> "LeRobotPaths":
        root = Path(dataset_root)
        info_path = root / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        return cls(
            root=root,
            data_path=info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"),
            video_path=info.get(
                "video_path",
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            ),
            chunks_size=int(info.get("chunks_size", 1000)),
            fps=float(info.get("fps", 30)),
        )

    def episode_chunk(self, episode_index: int) -> int:
        return int(episode_index) // self.chunks_size

    def parquet_path(self, episode_index: int) -> Path:
        return self.root / self.data_path.format(
            episode_chunk=self.episode_chunk(episode_index),
            episode_index=episode_index,
        )

    def video_file_path(self, camera_key: str, episode_index: int) -> Path:
        return self.root / self.video_path.format(
            episode_chunk=self.episode_chunk(episode_index),
            video_key=camera_key,
            episode_index=episode_index,
        )


@dataclass(frozen=True)
class DatasetSource:
    """One task dataset participating in manipulation-pattern training."""

    task: str
    root: Path
    episodes: tuple[int, ...]


def build_dataset_source(
    *,
    task: str,
    root: str | Path,
    num_episodes: int | None = None,
) -> DatasetSource:
    """Discover one source and deterministically select its first episodes."""

    dataset_root = Path(root)
    available = discover_episodes(dataset_root)
    if num_episodes is None:
        selected = available
    else:
        requested = int(num_episodes)
        if requested < 2:
            raise ValueError(f"dataset source {task!r} num_episodes must be at least 2")
        if len(available) < requested:
            raise ValueError(
                f"dataset source {task!r} requested {requested} episodes, "
                f"but only {len(available)} were discovered"
            )
        selected = available[:requested]
    if len(selected) < 2:
        raise ValueError(f"dataset source {task!r} needs at least two episodes for train/val split")
    return DatasetSource(
        task=str(task),
        root=dataset_root,
        episodes=tuple(sorted(selected)),
    )


def discover_episodes(dataset_root: str | Path) -> list[int]:
    paths = LeRobotPaths.from_root(dataset_root)
    data_dir = paths.root / "data"
    episodes = []
    for path in sorted(data_dir.glob("chunk-*/episode_*.parquet")):
        episodes.append(int(path.stem.split("_")[-1]))
    return episodes


def split_episodes(
    episodes: list[int],
    val_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(sorted(episodes), dtype=np.int64)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val = sorted(int(x) for x in shuffled[:val_count])
    train = sorted(int(x) for x in shuffled[val_count:])
    return train, val


def _boundary_weights(stage_ids: np.ndarray, window: int, boundary_weight: float) -> np.ndarray:
    weights = np.ones(stage_ids.shape[0], dtype=np.float32)
    if stage_ids.shape[0] <= 1:
        return weights
    change_indices = np.flatnonzero(stage_ids[1:] != stage_ids[:-1]) + 1
    for idx in change_indices:
        lo = max(0, int(idx) - window)
        hi = min(stage_ids.shape[0], int(idx) + window + 1)
        weights[lo:hi] = boundary_weight
    return weights


def load_episode_data(
    dataset_root: str | Path,
    episode_index: int,
    boundary_window: int = 5,
    boundary_weight: float = 0.5,
    *,
    data_keys: LeRobotDataKeys | None = None,
) -> EpisodeData:
    paths = LeRobotPaths.from_root(dataset_root)
    parquet_path = paths.parquet_path(episode_index)
    keys = data_keys or LeRobotDataKeys()
    df = pd.read_parquet(parquet_path, columns=[keys.state_key, keys.effort_key, "stage_id_gt"])
    state = np.stack(df[keys.state_key].to_numpy()).astype(np.float32)
    effort = np.stack(df[keys.effort_key].to_numpy()).astype(np.float32)
    low_dim = np.concatenate([state, effort], axis=-1)

    stage_ids = df["stage_id_gt"].to_numpy().astype(np.int64)
    stage = np.asarray([STAGE_TO_INDEX[int(x)] for x in stage_ids], dtype=np.int64)
    progress_head = np.asarray(
        [STAGE_TO_PROGRESS_HEAD[int(x)] for x in stage_ids],
        dtype=np.int64,
    )
    progress = compute_stage_progress(stage_ids)
    weight = _boundary_weights(stage_ids, window=boundary_window, boundary_weight=boundary_weight)
    return EpisodeData(
        episode_index=episode_index,
        parquet_path=parquet_path,
        low_dim=low_dim,
        stage=stage,
        progress_head=progress_head,
        progress=progress,
        weight=weight,
    )


def compute_low_dim_stats(episodes: list[EpisodeData]) -> tuple[np.ndarray, np.ndarray]:
    all_low = np.concatenate([ep.low_dim for ep in episodes], axis=0)
    mean = all_low.mean(axis=0).astype(np.float32)
    std = all_low.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


class StageWindowDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        episode_indices: list[int],
        low_dim_mean: np.ndarray,
        low_dim_std: np.ndarray,
        image_size: tuple[int, int] = (120, 160),
        low_history: int = LOW_HISTORY,
        image_offsets: tuple[int, ...] = IMAGE_OFFSETS,
        boundary_window: int = 5,
        boundary_weight: float = 0.5,
        frame_store: StreamingFrameStore | None = None,
        episode_data: list[EpisodeData] | None = None,
        data_keys: LeRobotDataKeys | None = None,
        decode_chunk_size: int = 16,
    ):
        self.frame_store = frame_store or StreamingFrameStore(*image_size, decode_chunk_size)
        if (self.frame_store.height, self.frame_store.width) != tuple(image_size):
            raise ValueError("frame_store dimensions must match image_size")
        self.dataset_root = Path(dataset_root)
        self.paths = LeRobotPaths.from_root(dataset_root)
        self.episode_indices = list(episode_indices)
        self.data_keys = data_keys or LeRobotDataKeys()
        self.episodes = episode_data if episode_data is not None else [
            load_episode_data(
                self.dataset_root,
                idx,
                boundary_window=boundary_window,
                boundary_weight=boundary_weight,
                data_keys=self.data_keys,
            )
            for idx in self.episode_indices
        ]
        if [ep.episode_index for ep in self.episodes] != self.episode_indices:
            raise ValueError("episode_data must match episode_indices in the same order")
        self.low_dim_mean = low_dim_mean.astype(np.float32)
        self.low_dim_std = low_dim_std.astype(np.float32)
        self.image_size = image_size
        self.low_history = int(low_history)
        self.image_offsets = tuple(int(x) for x in image_offsets)
        self.video_paths = {
            index: tuple(str(self.paths.video_file_path(camera, index)) for camera in self.data_keys.camera_keys)
            for index in self.episode_indices
        }

        self.cumulative_lengths: list[int] = []
        total = 0
        for ep in self.episodes:
            total += ep.length
            self.cumulative_lengths.append(total)

    def __len__(self) -> int:
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def _locate(self, index: int) -> tuple[EpisodeData, int]:
        ep_idx = bisect.bisect_right(self.cumulative_lengths, index)
        start = 0 if ep_idx == 0 else self.cumulative_lengths[ep_idx - 1]
        return self.episodes[ep_idx], index - start

    def _window_indices(self, frame_idx: int, length: int) -> np.ndarray:
        start = frame_idx - self.low_history + 1
        indices = np.arange(start, frame_idx + 1, dtype=np.int64)
        return np.clip(indices, 0, length - 1)

    def _image_indices(self, frame_idx: int, length: int) -> tuple[int, ...]:
        return tuple(int(np.clip(frame_idx + offset, 0, length - 1)) for offset in self.image_offsets)

    def _load_images(self, episode_index: int, frame_indices: tuple[int, ...]) -> torch.Tensor:
        requests = [(path, frame_indices) for path in self.video_paths[episode_index]]
        return torch.stack(self.frame_store.read_many(requests), dim=1)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._sample(index)

    def _sample(self, index: int, images: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        ep, frame_idx = self._locate(index)
        low_indices = self._window_indices(frame_idx, ep.length)
        low_dim = (ep.low_dim[low_indices] - self.low_dim_mean) / self.low_dim_std
        image_indices = self._image_indices(frame_idx, ep.length)

        return {
            "low_dim": torch.from_numpy(low_dim).float(),
            "images": self._load_images(ep.episode_index, image_indices) if images is None else images,
            "stage": torch.tensor(ep.stage[frame_idx], dtype=torch.long),
            "progress_head": torch.tensor(ep.progress_head[frame_idx], dtype=torch.long),
            "progress": torch.tensor([ep.progress[frame_idx]], dtype=torch.float32),
            "weight": torch.tensor(ep.weight[frame_idx], dtype=torch.float32),
            "episode_index": torch.tensor(ep.episode_index, dtype=torch.long),
            "frame_index": torch.tensor(frame_idx, dtype=torch.long),
        }

    def __getitems__(self, indices: list[int]) -> list[dict[str, torch.Tensor]]:
        """Coalesce streaming I/O without changing the requested batch order."""
        requests = []
        for index in indices:
            episode, frame = self._locate(index)
            image_indices = self._image_indices(frame, episode.length)
            requests.extend((path, image_indices) for path in self.video_paths[episode.episode_index])
        frames = self.frame_store.read_many(requests)
        cameras = len(self.data_keys.camera_keys)
        return [self._sample(index, torch.stack(frames[pos * cameras:(pos + 1) * cameras], dim=1))
                for pos, index in enumerate(indices)]
