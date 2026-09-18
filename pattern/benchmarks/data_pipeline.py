"""Measure a complete training split without truncating samples or batches.

This command does not update checkpoints, W&B runs, or dataset files. Reports
include configuration, sample-order fingerprints and Linux process I/O counters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from pattern.config import load_pattern_config, to_plain_dict
from pattern.data.pipeline import build_dataloaders, build_training_data
from pattern.models import ManipulationPatternRouterNet, build_manipulation_pattern_router_config
from pattern.train import (resolve_dataset_sources, set_seed, resolve_training_device,
                           wrap_model_for_training, run_epoch, build_lr_scheduler,
                           get_lr_cfg, resolve_peak_lr)


def process_counters(pids: list[int]) -> dict[str, int]:
    totals = dict(minor_faults=0, major_faults=0, read_bytes=0)
    for pid in pids:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            totals["minor_faults"] += int(stat[7])
            totals["major_faults"] += int(stat[9])
            io = dict(line.split(": ") for line in Path(f"/proc/{pid}/io").read_text().splitlines())
            totals["read_bytes"] += int(io["read_bytes"])
        except (FileNotFoundError, PermissionError):
            continue
    return totals


def process_memory(pids: list[int]) -> dict[str, int]:
    """Sample proportional memory, counting shared pages only proportionally.

    This is a periodic observation, not a kernel-tracked absolute peak. It
    excludes the operating system's unmapped file cache and other jobs.
    """
    totals = dict(pss_bytes=0, private_bytes=0)
    for pid in pids:
        try:
            values = {}
            for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines()[1:]:
                key, value = line.split(":", 1)
                values[key] = int(value.split()[0]) * 1024
            totals["pss_bytes"] += values.get("Pss", 0)
            totals["private_bytes"] += values.get("Private_Clean", 0) + values.get("Private_Dirty", 0)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("loader", "train"), default="loader",
                        help="train also runs one complete training and validation epoch")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_pattern_config(args.config)
    set_seed(cfg.dataset.seed)
    device, gpu_ids = resolve_training_device(cfg)
    start = time.perf_counter()
    data = build_training_data(resolve_dataset_sources(cfg.dataset, args.config), cfg.dataset,
                               num_image_times=cfg.model.image_encoder.num_times,
                               camera_names=getattr(cfg.model.image_encoder, "camera_names", None),
                               )
    dataset = data.train
    loader, validation_loader = build_dataloaders(data, cfg.train, device)
    print(f"Full split: {len(dataset)} samples; {len(loader)} batches; preparation {time.perf_counter()-start:.2f}s", flush=True)
    digest = hashlib.sha256()
    waits = []
    start = last = time.perf_counter()
    iterator = iter(loader)
    # Worker PIDs are diagnostic only; training never depends on private APIs.
    pids = [os.getpid()] + [worker.pid for worker in getattr(iterator, "_workers", [])]
    print(f"Reader={data.frame_store.backend}; pids={pids}", flush=True)
    initial_io = process_counters(pids)
    memory_max = process_memory(pids)
    for step, batch in enumerate(iterator, 1):
        now = time.perf_counter()
        waits.append(now - last)
        for key in ("episode_index", "frame_index", "stage", "progress_head", "progress", "weight", "low_dim"):
            digest.update(batch[key].numpy().tobytes())
        if step == 1 or step % 100 == 0:
            memory = process_memory(pids)
            memory_max = {key: max(memory_max[key], value) for key, value in memory.items()}
            print(json.dumps(dict(step=step, elapsed_s=now-start, wait_s=waits[-1],
                                  **process_counters(pids), **memory)), flush=True)
        last = time.perf_counter()
    elapsed = time.perf_counter() - start
    final_io = process_counters(pids)
    report = dict(
        reader=data.frame_store.backend, config=to_plain_dict(cfg), splits=data.splits, preparation=data.preparation,
        torch_version=torch.__version__, python_version=platform.python_version(),
        samples=len(loader)*cfg.train.batch_size, batches=len(loader), elapsed_s=elapsed,
        samples_per_s=len(loader)*cfg.train.batch_size/elapsed,
        first_batch_s=waits[0], wait_p50_s=float(np.quantile(waits, .5)),
        wait_p95_s=float(np.quantile(waits, .95)), wait_max_s=max(waits),
        order_and_targets_sha256=digest.hexdigest(),
        io={key: final_io[key]-initial_io[key] for key in initial_io},
        max_sampled_loader_memory=memory_max,
    )
    del iterator, loader, validation_loader
    if args.mode == "train":
        # Recreate the exact training RNG lifecycle: data preparation consumes
        # no random draws; model initialization precedes the loader iterator.
        set_seed(cfg.dataset.seed)
        device, gpu_ids = resolve_training_device(cfg)
        model_cfg = build_manipulation_pattern_router_config(to_plain_dict(cfg.model))
        model = wrap_model_for_training(ManipulationPatternRouterNet(model_cfg).to(device), gpu_ids)
        optimizer = torch.optim.AdamW(model.parameters(), lr=resolve_peak_lr(get_lr_cfg(cfg)),
                                      weight_decay=cfg.train.weight_decay)
        train_loader, val_loader = build_dataloaders(data, cfg.train, device)
        scheduler = build_lr_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
        common = dict(device=device, progress_loss_weight=cfg.train.progress_loss_weight,
                      predict_progress=model_cfg.predict_progress, grad_clip_norm=cfg.train.grad_clip_norm,
                      label_smoothing=cfg.train.label_smoothing, epoch=1,
                      log_every=int(getattr(cfg.train, "log_every", 50)))
        report["train"] = run_epoch(model, train_loader, optimizer, scheduler=scheduler, **common)
        report["val"] = run_epoch(model, val_loader, None, **common)
        report["optimizer_steps"] = len(train_loader)
        report["validation_samples"] = len(data.val)
        report["validation_batches"] = len(val_loader)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("config", "splits")}), flush=True)


if __name__ == "__main__":
    main()
