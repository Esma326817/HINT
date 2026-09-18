"""Evaluate a manipulation-pattern-router checkpoint by episode.

Run with ``python -m pattern.evaluation.evaluate --config configs/pattern/eval.yaml``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import DataLoader, Subset

from pattern.config import load_pattern_config
from pattern.data import (
    INDEX_TO_STAGE,
    StageWindowDataset,
    discover_episodes,
    image_offsets_for_num_times,
    resolve_data_keys,
    split_episodes,
)
from pattern.models import (
    ManipulationPatternRouterNet,
    build_manipulation_pattern_router_config,
)
from pattern.runtime.postprocess import smooth_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a manipulation-pattern router by episode.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pattern/eval.yaml",
        help="Path to the manipulation-pattern evaluation YAML config.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_episode_list(value: str | list[int] | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(x) for x in value]
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _load_ckpt_config(ckpt: dict) -> dict:
    config = ckpt.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint missing 'config'. Retrain with the current train script.")
    return config


def _load_ckpt_dataset(ckpt: dict) -> dict:
    dataset = _load_ckpt_config(ckpt).get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("Checkpoint config missing 'dataset'.")
    return dataset


def resolve_episodes(
    cfg: EasyDict,
    ckpt_dataset: dict,
    checkpoint_dir: Path,
    *,
    task: str | None = None,
    dataset_root: str | Path | None = None,
) -> list[int]:
    episodes = parse_episode_list(cfg.eval.episodes)
    if episodes is not None:
        return episodes

    split_path = checkpoint_dir / "split.json"
    if split_path.exists():
        split = load_json(split_path)
        if "val_episodes" in split:
            return [int(x) for x in split["val_episodes"]]
        task_split = (split.get("sources", {}) or {}).get(task or "")
        if task_split and "val_episodes" in task_split:
            return [int(x) for x in task_split["val_episodes"]]

    dataset_root = dataset_root or cfg.dataset.root or ckpt_dataset.get("root")
    val_ratio = float(ckpt_dataset.get("val_ratio", 0.1))
    seed = int(ckpt_dataset.get("seed", 0))
    _, val_eps = split_episodes(discover_episodes(dataset_root), val_ratio=val_ratio, seed=seed)
    return val_eps


def _resolve_from_ckpt_or_config(
    cfg_value: Any,
    ckpt_dataset: dict,
    ckpt_key: str,
    default: Any,
) -> Any:
    if cfg_value is not None:
        return cfg_value
    value = ckpt_dataset.get(ckpt_key)
    if value is not None:
        return value
    return default


def count_transitions(values: list[int]) -> int:
    if len(values) <= 1:
        return 0
    return sum(int(a != b) for a, b in zip(values[:-1], values[1:]))


def accuracy(pred: list[int], gt: list[int]) -> float:
    if not gt:
        return 0.0
    return float(np.mean(np.asarray(pred) == np.asarray(gt)))


def confusion_matrix(pred: list[int], gt: list[int], num_stages: int = 6) -> list[list[int]]:
    matrix = np.zeros((num_stages, num_stages), dtype=np.int64)
    for g, p in zip(gt, pred):
        matrix[g - 1, p - 1] += 1
    return matrix.tolist()


def mean_abs_error(pred: list[float], gt: list[float]) -> float:
    if not gt:
        return 0.0
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(gt))))


def mean_abs_error_masked(
    pred: list[float],
    gt: list[float],
    weights: list[float],
    *,
    boundary: bool,
) -> float:
    if not gt:
        return 0.0
    pred_arr = np.asarray(pred, dtype=np.float64)
    gt_arr = np.asarray(gt, dtype=np.float64)
    weight_arr = np.asarray(weights, dtype=np.float64)
    if boundary:
        mask = weight_arr < 1.0
    else:
        mask = weight_arr >= 1.0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(pred_arr[mask] - gt_arr[mask])))


def main() -> None:
    cli = parse_args()
    cfg = load_pattern_config(cli.config)

    checkpoint_path = Path(cfg.checkpoint)
    checkpoint_dir = checkpoint_path.parent
    output_dir = Path(cfg.eval.output_dir) if cfg.eval.output_dir else checkpoint_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_config = _load_ckpt_config(ckpt)
    ckpt_dataset = _load_ckpt_dataset(ckpt)
    model_cfg_dict = ckpt_config.get("model", {})
    if not isinstance(model_cfg_dict, dict) or not model_cfg_dict:
        raise ValueError("Checkpoint config missing 'model'.")
    image_encoder_cfg = model_cfg_dict.get("image_encoder", {}) or {}
    image_offsets = image_offsets_for_num_times(image_encoder_cfg.get("num_times", 1))

    task = getattr(cfg.dataset, "task", None)
    source_cfg: dict[str, Any] = {}
    sources = ckpt_dataset.get("sources", []) or []
    if sources:
        if not task:
            raise ValueError(
                "joint checkpoint requires dataset.task in configs/pattern/eval.yaml"
            )
        source_cfg = next(
            (
                source
                for source in sources
                if str(source.get("task", source.get("name", ""))) == str(task)
            ),
            {},
        )
        if not source_cfg:
            known = [source.get("task", source.get("name")) for source in sources]
            raise ValueError(f"unknown dataset.task {task!r}; expected one of {known}")

    dataset_root = cfg.dataset.root or source_cfg.get("root") or ckpt_dataset.get("root")
    if dataset_root is None:
        raise ValueError(
            "dataset root is required; set dataset.root in configs/pattern/eval.yaml "
            "or use a checkpoint saved with train config."
        )

    image_height = int(
        _resolve_from_ckpt_or_config(cfg.dataset.image_height, ckpt_dataset, "image_height", 120)
    )
    image_width = int(
        _resolve_from_ckpt_or_config(cfg.dataset.image_width, ckpt_dataset, "image_width", 160)
    )
    boundary_window = int(
        _resolve_from_ckpt_or_config(cfg.dataset.boundary_window, ckpt_dataset, "boundary_window", 5)
    )
    boundary_weight = float(
        _resolve_from_ckpt_or_config(cfg.dataset.boundary_weight, ckpt_dataset, "boundary_weight", 0.5)
    )
    episodes = resolve_episodes(
        cfg,
        ckpt_dataset,
        checkpoint_dir,
        task=str(task) if task else None,
        dataset_root=dataset_root,
    )

    if "low_dim_mean" in ckpt and "low_dim_std" in ckpt:
        low_mean = np.asarray(ckpt["low_dim_mean"], dtype=np.float32)
        low_std = np.asarray(ckpt["low_dim_std"], dtype=np.float32)
    else:
        stats = np.load(checkpoint_dir / "low_dim_stats.npz")
        low_mean = stats["mean"].astype(np.float32)
        low_std = stats["std"].astype(np.float32)

    dataset = StageWindowDataset(
        dataset_root,
        episodes,
        low_mean,
        low_std,
        image_size=(image_height, image_width),
        boundary_window=boundary_window,
        boundary_weight=boundary_weight,
        image_offsets=image_offsets,
        decode_chunk_size=int(_resolve_from_ckpt_or_config(
            getattr(cfg.dataset, "streaming_decode_chunk_size", None),
            ckpt_dataset, "streaming_decode_chunk_size", 16,
        )),
        data_keys=resolve_data_keys(
            ckpt_dataset, cfg.dataset, camera_names=image_encoder_cfg.get("camera_names")
        ),
    )
    if cfg.eval.limit_samples > 0 and cfg.eval.limit_samples < len(dataset):
        dataset = Subset(dataset, range(cfg.eval.limit_samples))

    loader = DataLoader(
        dataset,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.eval.num_workers,
        pin_memory=cfg.device.startswith("cuda") and torch.cuda.is_available(),
        persistent_workers=cfg.eval.num_workers > 0,
    )

    device = torch.device(cfg.device)
    ckpt_state = ckpt["model"]
    has_progress_head = any(
        key.startswith("progress_head") or key.startswith("progress_heads")
        for key in ckpt_state
    )
    model_cfg_dict = dict(model_cfg_dict)
    model_cfg_dict["predict_progress"] = has_progress_head
    model_cfg = build_manipulation_pattern_router_config(model_cfg_dict)
    model = ManipulationPatternRouterNet(model_cfg).to(device)
    model.load_state_dict(ckpt_state, strict=True)
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            low_dim = batch["low_dim"].to(device, non_blocking=True)
            images = batch["images"].to(device, non_blocking=True)
            out = model(low_dim, images)
            probs = out["pattern"].softmax(dim=-1).detach().cpu()
            pred_idx = probs.argmax(dim=-1)
            conf = probs.max(dim=-1).values
            progress_pred = (
                out["progress"].detach().cpu().squeeze(-1)
                if has_progress_head and "progress" in out
                else None
            )
            for i in range(pred_idx.shape[0]):
                row = {
                    "episode_index": int(batch["episode_index"][i].item()),
                    "frame_index": int(batch["frame_index"][i].item()),
                    "gt_stage": INDEX_TO_STAGE[int(batch["stage"][i].item())],
                    "raw_stage": INDEX_TO_STAGE[int(pred_idx[i].item())],
                    "raw_conf": float(conf[i].item()),
                    "gt_progress": float(batch["progress"][i].item()),
                    "sample_weight": float(batch["weight"][i].item()),
                }
                if progress_pred is not None:
                    row["raw_progress"] = float(progress_pred[i].item())
                rows.append(row)

    rows.sort(key=lambda item: (item["episode_index"], item["frame_index"]))

    per_episode = []
    all_gt, all_raw, all_smooth = [], [], []
    for ep in episodes:
        ep_rows = [row for row in rows if row["episode_index"] == ep]
        gt = [row["gt_stage"] for row in ep_rows]
        raw = [row["raw_stage"] for row in ep_rows]
        smooth = smooth_predictions(raw, stable_frames=cfg.eval.smooth_k)
        for row, smooth_stage in zip(ep_rows, smooth):
            row["smooth_stage"] = smooth_stage

        all_gt.extend(gt)
        all_raw.extend(raw)
        all_smooth.extend(smooth)
        per_episode.append(
            {
                "episode_index": ep,
                "frames": len(gt),
                "raw_acc": accuracy(raw, gt),
                "smooth_acc": accuracy(smooth, gt),
                "gt_transitions": count_transitions(gt),
                "raw_transitions": count_transitions(raw),
                "smooth_transitions": count_transitions(smooth),
            }
        )

    summary = {
        "checkpoint": str(checkpoint_path),
        "config": str(Path(cli.config).resolve()),
        "dataset_root": str(dataset_root),
        "episodes": episodes,
        "smooth_k": cfg.eval.smooth_k,
        "predict_progress": has_progress_head,
        "frames": len(all_gt),
        "raw_acc": accuracy(all_raw, all_gt),
        "smooth_acc": accuracy(all_smooth, all_gt),
        "gt_transitions": count_transitions(all_gt),
        "raw_transitions": count_transitions(all_raw),
        "smooth_transitions": count_transitions(all_smooth),
        "raw_confusion": confusion_matrix(all_raw, all_gt),
        "smooth_confusion": confusion_matrix(all_smooth, all_gt),
        "per_episode": per_episode,
    }
    if has_progress_head and rows and "raw_progress" in rows[0]:
        gt_progress = [row["gt_progress"] for row in rows]
        raw_progress = [row["raw_progress"] for row in rows]
        sample_weights = [row["sample_weight"] for row in rows]
        summary["progress_mae"] = mean_abs_error(raw_progress, gt_progress)
        summary["progress_mae_boundary"] = mean_abs_error_masked(
            raw_progress,
            gt_progress,
            sample_weights,
            boundary=True,
        )
        summary["progress_mae_stable"] = mean_abs_error_masked(
            raw_progress,
            gt_progress,
            sample_weights,
            boundary=False,
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "episode_index",
        "frame_index",
        "gt_stage",
        "raw_stage",
        "smooth_stage",
        "raw_conf",
        "gt_progress",
        "sample_weight",
    ]
    if has_progress_head and rows and "raw_progress" in rows[0]:
        fieldnames.append("raw_progress")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"frames={summary['frames']} "
        f"raw_acc={summary['raw_acc']:.4f} "
        f"smooth_acc={summary['smooth_acc']:.4f} "
        f"transitions gt/raw/smooth="
        f"{summary['gt_transitions']}/{summary['raw_transitions']}/{summary['smooth_transitions']}"
    )
    if "progress_mae" in summary:
        print(
            f"progress_mae={summary['progress_mae']:.4f} "
            f"stable={summary['progress_mae_stable']:.4f} "
            f"boundary={summary['progress_mae_boundary']:.4f}"
        )
    print(f"wrote {output_dir / 'summary.json'}")
    print(f"wrote {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
