"""Build deterministic datasets and batched loaders with on-demand video streaming."""

from __future__ import annotations

from dataclasses import dataclass
import bisect
from collections import defaultdict

import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import ConcatDataset, DataLoader

from .dataset import (
    DatasetSource,
    LeRobotPaths,
    StageWindowDataset,
    compute_low_dim_stats,
    image_offsets_for_num_times,
    load_episode_data,
    split_episodes,
)
from .keys import resolve_data_keys
from .streaming import StreamingFrameStore


class BatchedConcatDataset(ConcatDataset):
    """Forward batched reads to task datasets while preserving global order."""

    def __getitems__(self, indices: list[int]) -> list[dict[str, torch.Tensor]]:
        groups = defaultdict(list)
        for position, index in enumerate(indices):
            if index < 0:
                index += len(self)
            if not 0 <= index < len(self):
                raise IndexError(index)
            dataset_index = bisect.bisect_right(self.cumulative_sizes, index)
            offset = self.cumulative_sizes[dataset_index - 1] if dataset_index else 0
            groups[dataset_index].append((position, index - offset))
        results: dict[int, dict[str, torch.Tensor]] = {}
        for dataset_index, group in groups.items():
            dataset = self.datasets[dataset_index]
            local_indices = [index for _, index in group]
            samples = dataset.__getitems__(local_indices)
            for (position, _), sample in zip(group, samples, strict=True):
                results[position] = sample
        return [results[position] for position in range(len(indices))]


@dataclass
class TrainingData:
    train: ConcatDataset
    val: ConcatDataset
    splits: dict[str, dict[str, list[int]]]
    low_mean: np.ndarray
    low_std: np.ndarray
    frame_store: StreamingFrameStore
    preparation: dict[str, float]


def build_training_data(
    sources: list[DatasetSource],
    dataset_cfg: EasyDict,
    *,
    num_image_times: int,
    camera_names: tuple[str, ...] | None = None,
) -> TrainingData:
    """Load metadata once and preserve legacy split/normalization ordering."""
    data_keys = resolve_data_keys(dataset_cfg, camera_names=camera_names)
    splits = {}
    episode_groups = []
    videos = []
    for source in sources:
        train, val = split_episodes(list(source.episodes), dataset_cfg.val_ratio, dataset_cfg.seed)
        splits[source.task] = {"train_episodes": train, "val_episodes": val}
        groups = []
        for indices in (train, val):
            groups.append([
                load_episode_data(
                    source.root, index, dataset_cfg.boundary_window, dataset_cfg.boundary_weight,
                    data_keys=data_keys,
                )
                for index in indices
            ])
        episode_groups.append(groups)
        paths = LeRobotPaths.from_root(source.root)
        videos.extend(
            str(paths.video_file_path(camera, index))
            for index in sorted(train + val) for camera in data_keys.camera_keys
        )
    mean, std = compute_low_dim_stats([episode for train, _ in episode_groups for episode in train])
    height, width = dataset_cfg.image_height, dataset_cfg.image_width
    store = StreamingFrameStore(
        height, width, int(getattr(dataset_cfg, "streaming_decode_chunk_size", 16))
    )
    preparation = store.prepare(videos)
    train_datasets, val_datasets = [], []
    for source, groups in zip(sources, episode_groups, strict=True):
        for episodes, destination in zip(groups, (train_datasets, val_datasets), strict=True):
            destination.append(StageWindowDataset(
                source.root, [ep.episode_index for ep in episodes], mean, std,
                image_size=(height, width), boundary_window=dataset_cfg.boundary_window,
                boundary_weight=dataset_cfg.boundary_weight,
                image_offsets=image_offsets_for_num_times(num_image_times),
                frame_store=store, episode_data=episodes, data_keys=data_keys,
            ))
    return TrainingData(
        BatchedConcatDataset(train_datasets), BatchedConcatDataset(val_datasets),
        splits, mean, std, store, preparation,
    )


def build_dataloaders(
    data: TrainingData, train_cfg: EasyDict, device: torch.device
) -> tuple[DataLoader, DataLoader]:
    """Use the original shuffle, drop-last and worker RNG semantics.

    Both the training CLI and the benchmark use this factory. No custom sampler,
    independent generator, sample filtering or out-of-order delivery is added.
    """
    loaders = []
    for dataset, training in ((data.train, True), (data.val, False)):
        workers = int(train_cfg.num_workers if training else getattr(
            train_cfg, "val_num_workers", train_cfg.num_workers
        ))
        prefetch = {"prefetch_factor": int(getattr(train_cfg, "prefetch_factor", 2))} if workers else {}
        loaders.append(DataLoader(
            dataset, batch_size=train_cfg.batch_size, shuffle=training,
            num_workers=workers, pin_memory=device.type == "cuda", drop_last=training,
            persistent_workers=workers > 0, **prefetch,
        ))
    return loaders[0], loaders[1]
