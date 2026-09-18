"""Predict stage labels and write them to LeRobot parquet files.

Run with ``python -m pattern.annotation.predict_data --dataset-root DATASET --checkpoint CKPT``.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from pattern.config import load_pattern_config
from pattern.data import (
    LeRobotDataKeys,
    discover_episodes,
    resolve_data_keys,
    IMAGE_OFFSETS,
    LOW_HISTORY,
    INDEX_TO_STAGE,
    LeRobotPaths,
    image_offsets_for_num_times,
)
from pattern.data.streaming import StreamingFrameStore
from pattern.models import (
    ManipulationPatternRouterNet,
    build_manipulation_pattern_router_config,
)
from pattern.models.progress import progress_head_for_stage
from pattern.runtime.manipulation_pattern_router import ProgressGate
from pattern.runtime.postprocess import smooth_predictions


def parse_episode_list(value: str | None, dataset_root: str | Path) -> list[int]:
    if value is None:
        return discover_episodes(dataset_root)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def write_column(
    parquet_path: Path,
    column_name: str,
    values: Any,
    *,
    arrow_type: pa.DataType,
    backup_suffix: str,
) -> Path:
    """Back up a parquet once, then append or replace an annotation column."""
    backup_path = parquet_path.with_name(parquet_path.name + backup_suffix)
    if not backup_path.exists():
        shutil.copy2(parquet_path, backup_path)

    table = pq.read_table(parquet_path)
    array = pa.array(values, type=arrow_type)
    if len(array) != table.num_rows:
        raise ValueError(
            f"{parquet_path}: got {len(array)} values for {table.num_rows} rows"
        )
    if column_name in table.column_names:
        table = table.set_column(table.column_names.index(column_name), column_name, array)
    else:
        table = table.append_column(column_name, array)
    pq.write_table(table, parquet_path)
    return backup_path


class UnlabeledStageDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        episode_indices: list[int],
        low_dim_mean: np.ndarray,
        low_dim_std: np.ndarray,
        image_size: tuple[int, int] = (120, 160),
        low_history: int = LOW_HISTORY,
        image_offsets: tuple[int, ...] = IMAGE_OFFSETS,
        data_keys: LeRobotDataKeys | None = None,
        decode_chunk_size: int = 16,
    ):
        self.dataset_root = Path(dataset_root)
        self.paths = LeRobotPaths.from_root(dataset_root)
        self.episode_indices = list(episode_indices)
        self.data_keys = data_keys or LeRobotDataKeys()
        self.low_dim_mean = low_dim_mean.astype(np.float32)
        self.low_dim_std = low_dim_std.astype(np.float32)
        self.image_size = image_size
        self.frame_store = StreamingFrameStore(*image_size, decode_chunk_size)
        self.low_history = int(low_history)
        self.image_offsets = tuple(int(x) for x in image_offsets)

        self.episodes = []
        total = 0
        self.cumulative_lengths = []
        for episode_index in self.episode_indices:
            parquet_path = self.paths.parquet_path(episode_index)
            keys = self.data_keys
            df = pd.read_parquet(parquet_path, columns=[keys.state_key, keys.effort_key])
            state = np.stack(df[keys.state_key].to_numpy()).astype(np.float32)
            effort = np.stack(df[keys.effort_key].to_numpy()).astype(np.float32)
            low_dim = np.concatenate([state, effort], axis=-1)
            self.episodes.append(
                {
                    "episode_index": int(episode_index),
                    "parquet_path": parquet_path,
                    "low_dim": low_dim,
                    "length": int(low_dim.shape[0]),
                }
            )
            total += int(low_dim.shape[0])
            self.cumulative_lengths.append(total)

    def __len__(self) -> int:
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def _locate(self, index: int) -> tuple[dict, int]:
        ep_idx = int(np.searchsorted(self.cumulative_lengths, index, side="right"))
        start = 0 if ep_idx == 0 else self.cumulative_lengths[ep_idx - 1]
        return self.episodes[ep_idx], int(index - start)

    def _window_indices(self, frame_idx: int, length: int) -> np.ndarray:
        start = frame_idx - self.low_history + 1
        indices = np.arange(start, frame_idx + 1, dtype=np.int64)
        return np.clip(indices, 0, length - 1)

    def _image_indices(self, frame_idx: int, length: int) -> tuple[int, ...]:
        return tuple(int(np.clip(frame_idx + offset, 0, length - 1)) for offset in self.image_offsets)

    def _load_images(self, episode_index: int, frame_indices: tuple[int, ...]) -> torch.Tensor:
        requests = [
            (str(self.paths.video_file_path(key, episode_index)), frame_indices)
            for key in self.data_keys.camera_keys
        ]
        return torch.stack(self.frame_store.read_many(requests), dim=1)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ep, frame_idx = self._locate(index)
        low_indices = self._window_indices(frame_idx, ep["length"])
        low_dim = (ep["low_dim"][low_indices] - self.low_dim_mean) / self.low_dim_std
        image_indices = self._image_indices(frame_idx, ep["length"])
        return {
            "low_dim": torch.from_numpy(low_dim).float(),
            "images": self._load_images(ep["episode_index"], image_indices),
            "episode_index": torch.tensor(ep["episode_index"], dtype=torch.long),
            "frame_index": torch.tensor(frame_idx, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate an unlabeled LeRobot dataset with a manipulation-pattern checkpoint."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--data-config", help="Optional YAML with dataset key overrides; otherwise use checkpoint keys.")
    parser.add_argument("--checkpoint", default="outputs/manipulation_pattern_router/best.pt")
    parser.add_argument("--episodes", default=None, help="Comma-separated episode ids. Defaults to all episodes.")
    parser.add_argument("--output-column", default="stage_id_gt")
    parser.add_argument("--smooth-k", type=int, default=2)
    parser.add_argument(
        "--progress-gate-mode",
        choices=("auto", "both", "either", "off"),
        default="auto",
        help=(
            "Use predicted progress to accept stage transitions. "
            "'both' requires current stage near end and candidate near start; "
            "'auto' resolves to both (falls back to whichever end has progress)."
        ),
    )
    parser.add_argument(
        "--progress-start-threshold",
        type=float,
        default=0.3,
        help="Candidate stage progress must be <= this threshold for a gated transition.",
    )
    parser.add_argument(
        "--progress-end-threshold",
        type=float,
        default=0.65,
        help="Current stage progress must be >= this threshold for a gated transition.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Optional per-frame prediction CSV path. Not written by default.",
    )
    parser.add_argument("--backup-suffix", default=".bak")
    parser.add_argument("--dry-run", action="store_true", help="Run prediction without modifying parquet files.")
    return parser.parse_args()


def has_progress_head(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(key.startswith("progress_head") or key.startswith("progress_heads") for key in state_dict)


def row_progress_for_stage(row: dict, stage_id: int, num_heads: int) -> float | None:
    if "raw_progress" not in row:
        return None
    head_idx = progress_head_for_stage(stage_id, num_heads)
    if head_idx is not None and num_heads > 1:
        value = row.get(f"progress_head_{head_idx}")
        return float(value) if value is not None else None
    if int(row["raw_stage"]) == int(stage_id):
        return float(row["raw_progress"])
    return None


def progress_aware_smooth_predictions(
    rows: list[dict],
    *,
    k: int,
    num_progress_heads: int,
    mode: str,
    start_threshold: float,
    end_threshold: float,
) -> list[int]:
    if not rows:
        return []
    if k <= 1 and mode == "off":
        return [int(row["raw_stage"]) for row in rows]
    if mode == "auto":
        mode = "both"
    if mode == "off" or "raw_progress" not in rows[0]:
        return smooth_predictions(
            [int(row["raw_stage"]) for row in rows],
            stable_frames=k,
        )

    progress_gate = ProgressGate(
        mode=mode,
        start_threshold=start_threshold,
        end_threshold=end_threshold,
    )

    current_stage = int(rows[0]["raw_stage"])
    candidate_stage = None
    candidate_rows: list[dict] = []
    last_current_progress = row_progress_for_stage(rows[0], current_stage, num_progress_heads)
    smoothed = []

    for row in rows:
        raw_stage = int(row["raw_stage"])
        if raw_stage == current_stage:
            candidate_stage = None
            candidate_rows = []
            progress = row_progress_for_stage(row, current_stage, num_progress_heads)
            if progress is not None:
                last_current_progress = progress
            smoothed.append(current_stage)
            continue

        if raw_stage != candidate_stage:
            candidate_stage = raw_stage
            candidate_rows = [row]
        else:
            candidate_rows.append(row)

        if len(candidate_rows) >= max(1, k):
            current_values = [
                row_progress_for_stage(item, current_stage, num_progress_heads)
                for item in candidate_rows
            ]
            current_values = [value for value in current_values if value is not None]
            if last_current_progress is not None:
                current_values.append(last_current_progress)

            candidate_values = [
                row_progress_for_stage(item, candidate_stage, num_progress_heads)
                for item in candidate_rows
            ]
            candidate_values = [value for value in candidate_values if value is not None]

            current_progress = max(current_values) if current_values else None
            candidate_progress = min(candidate_values) if candidate_values else None
            if progress_gate.passes(
                current_progress,
                candidate_progress,
                num_progress_heads=num_progress_heads,
            ):
                current_stage = candidate_stage
                last_current_progress = row_progress_for_stage(row, current_stage, num_progress_heads)
                candidate_stage = None
                candidate_rows = []

        smoothed.append(current_stage)

    return smoothed


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_config = ckpt.get("config")
    if not isinstance(ckpt_config, dict):
        raise ValueError("Checkpoint missing 'config'. Retrain with the current train script.")
    ckpt_dataset = ckpt_config.get("dataset", {})
    image_height = int(args.image_height or ckpt_dataset.get("image_height", 120))
    image_width = int(args.image_width or ckpt_dataset.get("image_width", 160))
    low_mean = np.asarray(ckpt["low_dim_mean"], dtype=np.float32)
    low_std = np.asarray(ckpt["low_dim_std"], dtype=np.float32)
    model_cfg_dict = ckpt_config.get("model", {})
    if not isinstance(model_cfg_dict, dict) or not model_cfg_dict:
        raise ValueError("Checkpoint config missing 'model'.")
    image_encoder_cfg = model_cfg_dict.get("image_encoder", {}) or {}
    image_offsets = image_offsets_for_num_times(image_encoder_cfg.get("num_times", 1))

    data_config = load_pattern_config(args.data_config).get("dataset", {}) if args.data_config else {}
    data_keys = resolve_data_keys(
        ckpt_dataset, data_config, camera_names=image_encoder_cfg.get("camera_names")
    )

    episodes = parse_episode_list(args.episodes, args.dataset_root)
    paths = LeRobotPaths.from_root(args.dataset_root)
    dataset = UnlabeledStageDataset(
        args.dataset_root,
        episodes,
        low_mean,
        low_std,
        image_size=(image_height, image_width),
        image_offsets=image_offsets,
        data_keys=data_keys,
        decode_chunk_size=int(ckpt_dataset.get("streaming_decode_chunk_size", 16)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda") and torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device(args.device)
    ckpt_state = ckpt["model"]
    predict_progress = has_progress_head(ckpt_state)
    model_cfg_dict = dict(model_cfg_dict)
    model_cfg_dict["predict_progress"] = predict_progress
    num_progress_heads = int(model_cfg_dict.get("num_progress_heads", 1)) if predict_progress else 0
    model = ManipulationPatternRouterNet(
        build_manipulation_pattern_router_config(model_cfg_dict)
    ).to(device)
    model.load_state_dict(ckpt_state)
    model.eval()

    rows = []
    with torch.no_grad():
        progress = tqdm(loader, desc="annotate", dynamic_ncols=True)
        for batch in progress:
            low_dim = batch["low_dim"].to(device, non_blocking=True)
            images = batch["images"].to(device, non_blocking=True)
            out = model(low_dim, images)
            probs = out["pattern"].softmax(dim=-1).detach().cpu()
            pred_idx = probs.argmax(dim=-1)
            conf = probs.max(dim=-1).values
            progress_pred = (
                out["progress"].detach().cpu().squeeze(-1)
                if predict_progress and "progress" in out
                else None
            )
            progress_all = (
                out["progress_all"].detach().cpu()
                if predict_progress and "progress_all" in out
                else None
            )
            for i in range(pred_idx.shape[0]):
                row = {
                    "episode_index": int(batch["episode_index"][i].item()),
                    "frame_index": int(batch["frame_index"][i].item()),
                    "raw_stage": INDEX_TO_STAGE[int(pred_idx[i].item())],
                    "raw_conf": float(conf[i].item()),
                }
                if progress_pred is not None:
                    row["raw_progress"] = float(progress_pred[i].item())
                if progress_all is not None:
                    for head_idx in range(progress_all.shape[1]):
                        row[f"progress_head_{head_idx}"] = float(progress_all[i, head_idx].item())
                rows.append(row)

    rows.sort(key=lambda item: (item["episode_index"], item["frame_index"]))
    by_episode: dict[int, list[dict]] = {ep: [] for ep in episodes}
    for row in rows:
        by_episode.setdefault(row["episode_index"], []).append(row)

    for episode_index in episodes:
        ep_rows = by_episode.get(episode_index, [])
        smooth = progress_aware_smooth_predictions(
            ep_rows,
            k=args.smooth_k,
            num_progress_heads=num_progress_heads,
            mode=args.progress_gate_mode,
            start_threshold=args.progress_start_threshold,
            end_threshold=args.progress_end_threshold,
        )
        for row, smooth_stage in zip(ep_rows, smooth):
            row["smooth_stage"] = int(smooth_stage)

        values = smooth
        parquet_path = paths.parquet_path(episode_index)
        if not args.dry_run:
            write_column(
                parquet_path,
                args.output_column,
                np.asarray(values, dtype=np.int32),
                arrow_type=pa.int32(),
                backup_suffix=args.backup_suffix,
            )

    if args.csv_path is not None:
        csv_path = Path(args.csv_path)
        fieldnames = ["episode_index", "frame_index", "raw_stage", "smooth_stage", "raw_conf"]
        if predict_progress and rows and "raw_progress" in rows[0]:
            fieldnames.append("raw_progress")
        if predict_progress and rows:
            fieldnames.extend(
                f"progress_head_{idx}"
                for idx in range(num_progress_heads)
                if f"progress_head_{idx}" in rows[0]
            )
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)

    gate_mode = args.progress_gate_mode
    if gate_mode == "auto":
        gate_mode = "both"
    if not predict_progress:
        gate_mode = "off"
    mode = (
        "raw"
        if args.smooth_k <= 1 and gate_mode == "off"
        else f"K={args.smooth_k}, progress_gate={gate_mode}"
    )
    action = "would write" if args.dry_run else "wrote"
    print(f"{action} {mode} predictions to column `{args.output_column}` for {len(episodes)} episodes")
    print(f"checkpoint: {checkpoint_path}")
    if args.csv_path is not None:
        print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
