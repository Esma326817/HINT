"""Randomly sample single frames from the dataset and test direct stage inference.

Unlike :mod:`stage.evaluation.evaluate`, which evaluates full episodes, this
draws N random (episode, frame) samples and runs one-shot inference per sample,
reproducing exactly the StageWindowDataset preprocessing used in training.

    python -m pattern.evaluation.random_frames --n 400 --split val
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from pattern.data import (
    INDEX_TO_STAGE,
    StageWindowDataset,
    discover_episodes,
    image_offsets_for_num_times,
    resolve_data_keys,
)
from pattern.models import (
    ManipulationPatternRouterNet,
    build_manipulation_pattern_router_config,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=400, help="number of random frames")
    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument(
        "--task",
        choices=["sorting", "spelling", "peg_in_hole"],
        help="Task source to evaluate when the checkpoint was jointly trained.",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    dcfg = cfg["dataset"]
    sources = dcfg.get("sources", []) or []
    if sources:
        if args.task is None:
            raise ValueError("--task is required for a jointly trained checkpoint")
        source = next(
            (
                item
                for item in sources
                if str(item.get("task", item.get("name", ""))) == args.task
            ),
            None,
        )
        if source is None:
            raise ValueError(f"checkpoint does not contain task source {args.task!r}")
        root = source["root"]
    else:
        root = dcfg["root"]
    H, W = int(dcfg.get("image_height", 120)), int(dcfg.get("image_width", 160))
    image_encoder_cfg = cfg.get("model", {}).get("image_encoder", {}) or {}
    image_offsets = image_offsets_for_num_times(image_encoder_cfg.get("num_times", 1))

    low_mean = np.asarray(ckpt["low_dim_mean"], dtype=np.float32)
    low_std = np.asarray(ckpt["low_dim_std"], dtype=np.float32)

    # Episode selection
    split_path = ckpt_path.parent / "split.json"
    split = json.loads(split_path.read_text()) if split_path.exists() else {}
    if sources:
        split = (split.get("sources", {}) or {}).get(args.task, {})
    if args.split == "val":
        episodes = split.get("val_episodes") or discover_episodes(root)
    elif args.split == "train":
        episodes = split.get("train_episodes") or discover_episodes(root)
    else:
        episodes = discover_episodes(root)
    episodes = [int(x) for x in episodes]

    print(f"split={args.split} episodes={len(episodes)} -> {episodes}")

    dataset = StageWindowDataset(
        root, episodes, low_mean, low_std,
        image_size=(H, W),
        boundary_window=int(dcfg.get("boundary_window", 5)),
        boundary_weight=float(dcfg.get("boundary_weight", 0.5)),
        image_offsets=image_offsets,
        decode_chunk_size=int(dcfg.get("streaming_decode_chunk_size", 16)),
        data_keys=resolve_data_keys(dcfg, camera_names=image_encoder_cfg.get("camera_names")),
    )
    total = len(dataset)
    n = min(args.n, total)
    sample_idx = random.sample(range(total), n)
    print(f"dataset frames={total}, sampling {n} random single frames")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ManipulationPatternRouterNet(
        build_manipulation_pattern_router_config(cfg["model"])
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    correct = 0
    conf_list = []
    per_stage_total = Counter()
    per_stage_correct = Counter()
    confusion = np.zeros((6, 6), dtype=np.int64)

    with torch.no_grad():
        for k, di in enumerate(sample_idx):
            item = dataset[di]
            low = item["low_dim"].unsqueeze(0).to(device)
            img = item["images"].unsqueeze(0).to(device)
            out = model(low, img)
            probs = out["pattern"].softmax(dim=-1)[0].cpu()
            pred = int(probs.argmax().item())
            conf = float(probs[pred].item())
            gt = int(item["stage"].item())
            gt_s, pred_s = INDEX_TO_STAGE[gt], INDEX_TO_STAGE[pred]

            confusion[gt, pred] += 1
            per_stage_total[gt_s] += 1
            if gt == pred:
                correct += 1
                per_stage_correct[gt_s] += 1
            conf_list.append(conf)
            if (k + 1) % 50 == 0:
                print(f"  {k+1}/{n} running acc={correct/(k+1):.3f}")

    acc = correct / n
    print("\n==================== RESULTS ====================")
    print(f"single-frame raw accuracy: {acc:.4f}  ({correct}/{n})")
    print(f"mean confidence: {np.mean(conf_list):.3f}")
    print("\nper-stage accuracy (gt stage_id -> acc, n):")
    for s in range(1, 7):
        t = per_stage_total[s]
        c = per_stage_correct[s]
        if t:
            print(f"  stage {s} ({INDEX_TO_STAGE[s-1]}): {c/t:.3f}  ({c}/{t})")
    print("\nconfusion matrix (rows=GT stage_id, cols=pred stage_id, 1..6):")
    header = "       " + " ".join(f"p{j+1:>4}" for j in range(6))
    print(header)
    for i in range(6):
        row = " ".join(f"{confusion[i,j]:>5}" for j in range(6))
        print(f"  g{i+1}  {row}")


if __name__ == "__main__":
    main()
