"""Compute Piper normalization statistics without decoding video frames.

Fast path for I/O: read ``state`` / ``actions`` from LeRobot parquet files.
Aggregation matches ``compute_norm_stats.py`` / the DataLoader path exactly:
same frame order, same ``batch_size`` chunks, and drop the final incomplete batch
(``num_batches = len(dataset) // batch_size``), so RunningStats (mean/std/q01/q99)
are bitwise-identical to the slow path for state/actions.
"""

from __future__ import annotations

import json
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tqdm
import tyro
from lerobot.common.constants import HF_LEROBOT_HOME

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


def _validate_config(config: _config.TrainConfig, data_config: _config.DataConfig) -> _config.PiperDataConfig:
    if not isinstance(config.data, _config.PiperDataConfig):
        raise ValueError("compute_norm_stats_simple.py only supports PiperDataConfig.")
    if data_config.rlds_data_dir is not None:
        raise ValueError("compute_norm_stats_simple.py does not support RLDS data.")
    return config.data


def _resolve_dataset_root(repo_id: str) -> pathlib.Path:
    local = _data_loader._local_lerobot_dataset_root(repo_id)
    if local is not None:
        return local
    home = pathlib.Path(os.environ.get("HF_LEROBOT_HOME") or os.environ.get("LEROBOT_HOME") or HF_LEROBOT_HOME)
    root = home / repo_id
    info = root / "meta" / "info.json"
    if not info.is_file():
        raise FileNotFoundError(
            f"Cannot find local LeRobot dataset for repo_id={repo_id!r}. "
            f"Tried {root}. Set HF_LEROBOT_HOME or pass an absolute path."
        )
    return root


def _episode_parquet_paths(root: pathlib.Path) -> list[pathlib.Path]:
    info = json.loads((root / "meta" / "info.json").read_text())
    data_path = info["data_path"]
    total_episodes = int(info["total_episodes"])
    chunks_size = int(info.get("chunks_size", 1000))
    paths: list[pathlib.Path] = []
    for ep_idx in range(total_episodes):
        chunk = ep_idx // chunks_size
        rel = data_path.format(episode_chunk=chunk, episode_index=ep_idx)
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"Missing episode parquet: {path}")
        paths.append(path)
    return paths


def _load_episode_arrays(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Load state/actions from one episode parquet as float32 arrays."""
    import pandas as pd

    df = pd.read_parquet(path, columns=["state", "actions"])
    state = np.stack(df["state"].to_numpy()).astype(np.float32, copy=False)
    actions = np.stack(df["actions"].to_numpy()).astype(np.float32, copy=False)
    if state.ndim != 2 or actions.ndim != 2:
        raise ValueError(f"Expected 2D state/actions in {path}, got {state.shape=} {actions.shape=}")
    if state.shape[0] != actions.shape[0]:
        raise ValueError(f"Length mismatch in {path}: state={state.shape[0]} actions={actions.shape[0]}")
    return state, actions


def _action_windows(actions: np.ndarray, horizon: int) -> np.ndarray:
    """Build (T, H, D) action windows with end-of-episode clamp (LeRobot-style)."""
    t, d = actions.shape
    if t == 0:
        return np.zeros((0, horizon, d), dtype=np.float32)
    frame_idx = np.arange(t, dtype=np.int64)[:, None]
    offsets = np.arange(horizon, dtype=np.int64)[None, :]
    gather = np.minimum(frame_idx + offsets, t - 1)
    return actions[gather]


def _apply_delta(state: np.ndarray, action_seq: np.ndarray) -> np.ndarray:
    """Match Piper ``DeltaActions(make_bool_mask(6, -1, 6, -1))``."""
    mask = np.asarray(transforms.make_bool_mask(6, -1, 6, -1), dtype=bool)
    dims = mask.shape[0]
    out = action_seq.copy()
    out[..., :dims] -= np.where(mask, state[..., :dims], 0.0)[:, None, :]
    return out


def _process_episode(
    path: pathlib.Path, action_horizon: int, extra_delta: bool
) -> tuple[np.ndarray, np.ndarray]:
    state, actions = _load_episode_arrays(path)
    action_seq = _action_windows(actions, action_horizon)
    if extra_delta:
        action_seq = _apply_delta(state, action_seq)
    return state, action_seq


def main(
    config_name: str,
    max_frames: int | None = None,
    num_workers: int = 16,
):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    piper_config = _validate_config(config, data_config)

    repo_ids = _data_loader._resolve_repo_ids(data_config)
    if not repo_ids:
        raise ValueError("Repo id(s) are not set.")

    action_horizon = int(config.model.action_horizon)
    extra_delta = bool(piper_config.extra_delta_transform)
    batch_size = int(config.batch_size)

    parquet_paths: list[pathlib.Path] = []
    for repo_id in repo_ids:
        root = _resolve_dataset_root(repo_id)
        print(f"Loading parquet from: {root}")
        parquet_paths.extend(_episode_parquet_paths(root))

    # Load episodes in LeRobot index order (ep0, ep1, ...), parallelize I/O only.
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        futures = [
            pool.submit(_process_episode, path, action_horizon, extra_delta) for path in parquet_paths
        ]
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        for fut in tqdm.tqdm(futures, total=len(futures), desc="Loading parquet"):
            state, action_seq = fut.result()
            if state.shape[0] == 0:
                continue
            states.append(state)
            actions.append(action_seq)

    if not states:
        raise RuntimeError("No frames found to compute norm stats.")

    state = np.concatenate(states, axis=0)
    action = np.concatenate(actions, axis=0)
    raw_total = int(state.shape[0])

    # Match TorchDataLoader + compute_norm_stats: only full batches, no shuffle.
    if max_frames is not None and max_frames < raw_total:
        state = state[: int(max_frames)]
        action = action[: int(max_frames)]
    capped_total = int(state.shape[0])

    num_batches = capped_total // batch_size
    if num_batches < 1:
        raise RuntimeError(
            f"Not enough frames for one batch (frames={capped_total}, batch_size={batch_size})."
        )
    usable = num_batches * batch_size
    state = state[:usable]
    action = action[:usable]
    print(
        f"Matching DataLoader path: {usable}/{capped_total} frames "
        f"({num_batches} batches of {batch_size}, dropped {capped_total - usable})"
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}
    for i in tqdm.trange(0, usable, batch_size, desc="Computing stats"):
        stats["state"].update(state[i : i + batch_size])
        stats["actions"].update(action[i : i + batch_size])

    norm_stats = {key: stats[key].get_statistics() for key in keys}
    asset_id = data_config.asset_id or data_config.repo_id or (
        data_config.repo_ids[0] if data_config.repo_ids else None
    )
    if asset_id is None:
        raise ValueError("Cannot determine output path: both repo_id and repo_ids are unset.")
    output_path = config.assets_dirs / asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
