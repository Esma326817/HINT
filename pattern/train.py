"""Train the manipulation-pattern router from a YAML configuration.

Run with ``python -m pattern.train --config configs/pattern/train_fruit.yaml``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pattern.config import load_pattern_config, to_plain_dict
from pattern.data import (
    DatasetSource,
    build_dataset_source,
    torchcodec_available,
)
from pattern.data.pipeline import build_dataloaders, build_training_data
from pattern.models import (
    ManipulationPatternRouterNet,
    build_manipulation_pattern_router_config,
)
from pattern.training_metrics import EpochMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the manipulation-pattern router.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pattern/train_fruit.yaml",
        help="Path to the manipulation-pattern training YAML config.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _path_from_config(value: str | Path, config_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_dir / path).resolve()


def resolve_dataset_sources(dataset_cfg: EasyDict, config_path: str | Path) -> list[DatasetSource]:
    """Resolve the multi-task ``dataset.sources`` schema and legacy single root."""

    config_dir = Path(config_path).resolve().parent
    raw_sources = getattr(dataset_cfg, "sources", None)
    if raw_sources is None:
        root = _path_from_config(dataset_cfg.root, config_dir)
        return [build_dataset_source(task="default", root=root)]
    if not raw_sources:
        raise ValueError("dataset.sources must contain at least one source")

    sources: list[DatasetSource] = []
    seen_tasks: set[str] = set()
    for raw in raw_sources:
        task = str(getattr(raw, "task", getattr(raw, "name", ""))).strip()
        if not task:
            raise ValueError("every dataset.sources entry requires a non-empty task")
        if task in seen_tasks:
            raise ValueError(f"duplicate dataset source task: {task!r}")
        seen_tasks.add(task)
        sources.append(
            build_dataset_source(
                task=task,
                root=_path_from_config(raw.root, config_dir),
                num_episodes=getattr(raw, "num_episodes", None),
            )
        )
    return sources


def weighted_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    loss = F.cross_entropy(
        logits,
        target,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = F.mse_loss(pred.squeeze(-1), target.squeeze(-1), reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


def progress_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.abs(pred.squeeze(-1) - target.squeeze(-1)).mean().item())


def progress_mae_masked(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float | None:
    if not bool(mask.any()):
        return None
    return float(torch.abs(pred.squeeze(-1)[mask] - target.squeeze(-1)[mask]).mean().item())


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> tuple[int, int]:
    pred = logits.argmax(dim=-1)
    return int((pred == target).sum().item()), int(target.numel())


class LRScheduler(Protocol):
    step_per_batch: bool

    def step(self) -> None: ...

    def state_dict(self) -> dict: ...


class WarmupCosineDecayScheduler:
    """Match optax.warmup_cosine_decay_schedule (see openpi CosineDecaySchedule)."""

    step_per_batch = True

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        peak_lr: float,
        decay_steps: int,
        decay_lr: float,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if decay_steps < 1:
            raise ValueError(f"decay_steps must be >= 1, got {decay_steps}")

        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.peak_lr = float(peak_lr)
        self.decay_steps = int(decay_steps)
        self.decay_lr = float(decay_lr)
        self.init_lr = self.peak_lr / (self.warmup_steps + 1)
        self.total_steps = self.warmup_steps + self.decay_steps
        self.step_count = 0
        self._apply_lr()

    def _lr_at_step(self, step: int) -> float:
        if step < self.warmup_steps:
            if self.warmup_steps == 0:
                return self.peak_lr
            progress = step / self.warmup_steps
            return self.init_lr + (self.peak_lr - self.init_lr) * progress
        if step < self.total_steps:
            progress = (step - self.warmup_steps) / self.decay_steps
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.decay_lr + (self.peak_lr - self.decay_lr) * cosine
        return self.decay_lr

    def _apply_lr(self) -> None:
        lr = self._lr_at_step(self.step_count)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self) -> None:
        self.step_count += 1
        self._apply_lr()

    def state_dict(self) -> dict:
        return {
            "step_count": self.step_count,
            "warmup_steps": self.warmup_steps,
            "peak_lr": self.peak_lr,
            "decay_steps": self.decay_steps,
            "decay_lr": self.decay_lr,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.step_count = int(state_dict["step_count"])
        self._apply_lr()


def scheduler_steps_per_batch(scheduler: LRScheduler | torch.optim.lr_scheduler.LRScheduler) -> bool:
    return bool(getattr(scheduler, "step_per_batch", False))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    progress_loss_weight: float,
    predict_progress: bool,
    grad_clip_norm: float,
    label_smoothing: float,
    epoch: int,
    scheduler: LRScheduler | torch.optim.lr_scheduler.LRScheduler | None = None,
    log_every: int = 50,
    profile_batches: int = 0,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    mode = "train" if train else "val"

    totals = EpochMetrics(device, predict_progress)
    epoch_start = time.perf_counter()
    last_batch_end = epoch_start
    data_wait_s = first_batch_wait_s = 0.0
    gpu_events = []
    progress_bar = tqdm(loader, desc=f"{mode} {epoch:03d}", dynamic_ncols=True, leave=False)
    for step, batch in enumerate(progress_bar, start=1):
        batch_ready = time.perf_counter()
        data_wait_s += batch_ready - last_batch_end
        if step == 1:
            first_batch_wait_s = batch_ready - epoch_start
            tqdm.write(f"[*] {mode} first batch ready in {first_batch_wait_s:.2f}s")
        events = None
        if device.type == "cuda" and step <= profile_batches:
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record(torch.cuda.current_stream(device))
        low_dim = batch["low_dim"].to(device, non_blocking=True)
        images = batch["images"].to(device, non_blocking=True)
        stage = batch["stage"].to(device, non_blocking=True)
        progress_head = batch["progress_head"].to(device, non_blocking=True)
        progress_target = batch["progress"].to(device, non_blocking=True)
        weight = batch["weight"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            out = model(low_dim, images, progress_head_index=progress_head)
            ce_label_smoothing = label_smoothing if train else 0.0
            pattern_loss = weighted_ce(out["pattern"], stage, weight, ce_label_smoothing)
            loss = pattern_loss
            progress_loss = torch.zeros((), device=device)
            if predict_progress and "progress" in out:
                progress_loss = weighted_mse(out["progress"], progress_target, weight)
                loss = loss + progress_loss_weight * progress_loss

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    unwrap_model(model).parameters(),
                    max_norm=grad_clip_norm,
                )
                optimizer.step()
                if scheduler is not None and scheduler_steps_per_batch(scheduler):
                    scheduler.step()

        totals.update(out, stage, progress_target, weight, loss, pattern_loss, progress_loss)
        if events is not None:
            events[1].record(torch.cuda.current_stream(device))
            gpu_events.append(events)
        if step == 1 or step % max(1, log_every) == 0:
            current = totals.compute()
            progress_bar.set_postfix(loss=f"{current['loss']:.4f}", pattern_acc=f"{current['pattern_acc']:.4f}")
        last_batch_end = time.perf_counter()

    metrics = totals.compute()
    metrics["epoch_s"] = time.perf_counter() - epoch_start
    metrics["samples_per_s"] = totals.count / max(metrics["epoch_s"], 1e-9)
    metrics["loader_wait_s"] = data_wait_s
    metrics["first_batch_wait_s"] = first_batch_wait_s
    if gpu_events:
        gpu_events[-1][1].synchronize()
        metrics["profile_gpu_step_ms"] = sum(a.elapsed_time(b) for a, b in gpu_events) / len(gpu_events)
    print(f"[*] {mode}: {metrics['samples_per_s']:.1f} samples/s, "
          f"epoch {metrics['epoch_s']:.1f}s, loader wait {data_wait_s:.1f}s", flush=True)
    return metrics


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _configured_gpu_ids(cfg: EasyDict) -> list[int] | None:
    raw = getattr(cfg, "gpu_ids", None)
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        ids = [int(i) for i in raw]
        return ids or None
    return [int(raw)]


def resolve_training_device(cfg: EasyDict) -> tuple[torch.device, list[int] | None]:
    """Return primary device and optional GPU id list for DataParallel.

    ``gpu_ids`` is read first, including a single-card list such as ``[3]``.
    If it is omitted, fall back to ``device`` (default ``cuda``).
    """
    requested = str(getattr(cfg, "device", "cuda"))
    if requested == "cpu" or (requested.startswith("cuda") and not torch.cuda.is_available()):
        if requested.startswith("cuda"):
            print("[!] CUDA unavailable, falling back to CPU", flush=True)
        return torch.device("cpu"), None

    gpu_ids = _configured_gpu_ids(cfg)
    multi_gpu = bool(getattr(cfg, "multi_gpu", False))
    if gpu_ids is None:
        if not multi_gpu:
            device = torch.device(requested)
            print(f"[*] training device: {device}", flush=True)
            return device, None
        gpu_ids = list(range(torch.cuda.device_count()))

    primary = gpu_ids[0]
    torch.cuda.set_device(primary)
    device = torch.device(f"cuda:{primary}")
    print(f"[*] training device: {device}", flush=True)
    if multi_gpu and len(gpu_ids) >= 2:
        return device, gpu_ids
    return device, None


def wrap_model_for_training(model: nn.Module, gpu_ids: list[int] | None) -> nn.Module:
    if gpu_ids is None:
        return model
    wrapped = nn.DataParallel(model, device_ids=gpu_ids)
    print(f"[*] DataParallel enabled on GPUs {gpu_ids}", flush=True)
    return wrapped


def get_lr_cfg(cfg: EasyDict) -> EasyDict:
    raw = cfg.train.lr
    if isinstance(raw, (dict, EasyDict)):
        return raw if isinstance(raw, EasyDict) else EasyDict(raw)
    return EasyDict(
        lr=raw,
        peak_lr=getattr(cfg.train, "peak_lr", raw),
        lr_scheduler=getattr(cfg.train, "lr_scheduler", "none"),
        warmup_epoch=getattr(cfg.train, "warmup_epoch", 0),
        decay_epoch=getattr(cfg.train, "decay_epoch", 0),
        decay_lr=getattr(cfg.train, "decay_lr", getattr(cfg.train, "lr_min", float(raw) * 0.1)),
    )


def resolve_peak_lr(lr_cfg: EasyDict) -> float:
    return float(getattr(lr_cfg, "peak_lr", lr_cfg.lr))


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: EasyDict,
    *,
    steps_per_epoch: int,
) -> LRScheduler | torch.optim.lr_scheduler.LRScheduler | None:
    lr_cfg = get_lr_cfg(cfg)
    scheduler_type = str(getattr(lr_cfg, "lr_scheduler", "none")).lower()
    if scheduler_type in {"none", "null", "constant", ""}:
        return None

    if scheduler_type in {"warmup_cosine", "cosine_decay", "warmup_cosine_decay"}:
        peak_lr = resolve_peak_lr(lr_cfg)
        decay_lr = float(getattr(lr_cfg, "decay_lr", peak_lr * 0.1))
        warmup_epoch = int(getattr(lr_cfg, "warmup_epoch", 0))
        decay_epoch_cfg = int(getattr(lr_cfg, "decay_epoch", 0))
        warmup_steps = warmup_epoch * steps_per_epoch
        if decay_epoch_cfg <= 0:
            decay_epochs = max(1, int(cfg.train.epochs) - warmup_epoch)
        else:
            decay_epochs = decay_epoch_cfg
        decay_steps = max(1, decay_epochs * steps_per_epoch)

        init_lr = peak_lr / (warmup_steps + 1) if warmup_steps > 0 else peak_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = init_lr

        return WarmupCosineDecayScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            peak_lr=peak_lr,
            decay_steps=decay_steps,
            decay_lr=decay_lr,
        )

    if scheduler_type == "cosine":
        eta_min = float(getattr(lr_cfg, "decay_lr", getattr(lr_cfg, "lr_min", 1e-6)))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.train.epochs),
            eta_min=eta_min,
        )

    raise ValueError(f"Unknown train.lr.lr_scheduler: {scheduler_type!r}")


def describe_lr_scheduler(
    scheduler: LRScheduler | torch.optim.lr_scheduler.LRScheduler,
    cfg: EasyDict,
    *,
    steps_per_epoch: int,
) -> str:
    lr_cfg = get_lr_cfg(cfg)
    if isinstance(scheduler, WarmupCosineDecayScheduler):
        warmup_epoch = int(getattr(lr_cfg, "warmup_epoch", 0))
        decay_epoch_cfg = int(getattr(lr_cfg, "decay_epoch", 0))
        decay_epochs = (
            max(1, int(cfg.train.epochs) - warmup_epoch)
            if decay_epoch_cfg <= 0
            else decay_epoch_cfg
        )
        return (
            "WarmupCosineDecayScheduler "
            f"(warmup_epoch={warmup_epoch}, "
            f"decay_epoch={decay_epochs}, "
            f"peak_lr={scheduler.peak_lr:g}, "
            f"decay_lr={scheduler.decay_lr:g}, "
            f"init_lr={scheduler.init_lr:g}, "
            f"steps_per_epoch={steps_per_epoch})"
        )
    if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        eta_min = float(getattr(lr_cfg, "decay_lr", getattr(lr_cfg, "lr_min", 1e-6)))
        return f"CosineAnnealingLR (T_max={cfg.train.epochs}, eta_min={eta_min:g})"
    return scheduler.__class__.__name__


def init_wandb(cfg: EasyDict, run_config: dict) -> Any | None:
    if not cfg.wandb.enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is not installed. Install it or set wandb.enabled: false in config.") from exc

    run_name = cfg.wandb.name
    if run_name is None:
        run_name = f"manipulation-pattern-router-{time.strftime('%Y%m%d-%H%M%S')}"
    return wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        config=run_config,
        mode="online",
        dir=str(Path(cfg.train.output_dir).resolve()),
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: EasyDict,
    low_mean: np.ndarray,
    low_std: np.ndarray,
    dataset_splits: dict[str, dict[str, list[int]]] | None = None,
    scheduler: LRScheduler | torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict:
    ckpt = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": to_plain_dict(cfg),
        "low_dim_mean": low_mean,
        "low_dim_std": low_std,
        "dataset_splits": dataset_splits or {},
        "manipulation_pattern_labels": {
            "1": "free-move",
            "2": "pre_contact-left",
            "3": "pre_contact-right",
            "4": "dexterous_contact-left",
            "5": "dexterous_contact-right",
            "6": "transport_contact",
        },
        "progress_head_labels": {
            "0": "free_move",
            "1": "pre_contact",
            "2": "transport_contact",
            "3": "dexterous_contact",
        },
    }
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    torch.save(ckpt, path)
    return ckpt


def main() -> None:
    cli = parse_args()
    cfg = load_pattern_config(cli.config)
    set_seed(cfg.dataset.seed)

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device, gpu_ids = resolve_training_device(cfg)

    sources = resolve_dataset_sources(cfg.dataset, cli.config)
    print(
        f"[*] video backend: {'torchcodec' if torchcodec_available() else 'opencv'}",
        flush=True,
    )
    data = build_training_data(
        sources, cfg.dataset, num_image_times=cfg.model.image_encoder.num_times,
        camera_names=getattr(cfg.model.image_encoder, "camera_names", None),
    )
    source_splits = data.splits
    low_mean, low_std = data.low_mean, data.low_std

    np.savez(output_dir / "low_dim_stats.npz", mean=low_mean, std=low_std)
    split_info = {
        "seed": int(cfg.dataset.seed),
        "val_ratio": float(cfg.dataset.val_ratio),
        "sources": source_splits,
    }
    (output_dir / "split.json").write_text(json.dumps(split_info, indent=2) + "\n", encoding="utf-8")

    train_dataset, val_dataset = data.train, data.val

    train_workers = int(cfg.train.num_workers)
    prefetch_factor = int(getattr(cfg.train, "prefetch_factor", 2))
    train_loader, val_loader = build_dataloaders(data, cfg.train, device)
    print(f"[*] train: {len(train_dataset)} samples, {len(train_loader)} steps/epoch; "
          f"workers={train_workers}, prefetch_factor={prefetch_factor}", flush=True)

    model_cfg = build_manipulation_pattern_router_config(to_plain_dict(cfg.model))
    predict_progress = bool(model_cfg.predict_progress)
    progress_loss_weight = float(getattr(cfg.train, "progress_loss_weight", 0.5))
    label_smoothing = float(getattr(cfg.train, "label_smoothing", 0.0))
    model = wrap_model_for_training(ManipulationPatternRouterNet(model_cfg).to(device), gpu_ids)
    lr_cfg = get_lr_cfg(cfg)
    optimizer = torch.optim.AdamW(
        unwrap_model(model).parameters(),
        lr=resolve_peak_lr(lr_cfg),
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = build_lr_scheduler(
        optimizer,
        cfg,
        steps_per_epoch=len(train_loader),
    )
    if scheduler is not None:
        print(
            f"[*] lr scheduler: {describe_lr_scheduler(scheduler, cfg, steps_per_epoch=len(train_loader))}",
            flush=True,
        )

    run_config = to_plain_dict(cfg) | {
        "dataset_splits": source_splits,
        "config_path": str(Path(cli.config).resolve()),
        "gpu_ids": gpu_ids,
        "per_gpu_batch_size": (
            cfg.train.batch_size // len(gpu_ids) if gpu_ids else cfg.train.batch_size
        ),
    }
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    wandb_run = init_wandb(cfg, run_config)

    best_val_acc = -1.0
    history = []
    try:
        for epoch in range(1, cfg.train.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                progress_loss_weight=progress_loss_weight,
                predict_progress=predict_progress,
                grad_clip_norm=cfg.train.grad_clip_norm,
                label_smoothing=label_smoothing,
                epoch=epoch,
                scheduler=scheduler,
                log_every=int(getattr(cfg.train, "log_every", 50)),
                profile_batches=int(getattr(cfg.train, "profile_batches", 0)) if epoch == 1 else 0,
            )
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    optimizer=None,
                    device=device,
                    progress_loss_weight=progress_loss_weight,
                    predict_progress=predict_progress,
                    grad_clip_norm=cfg.train.grad_clip_norm,
                    label_smoothing=label_smoothing,
                    epoch=epoch,
                    log_every=int(getattr(cfg.train, "log_every", 50)),
                )

            record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            history.append(record)
            (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
            print(
                f"epoch {epoch:03d} "
                f"train loss {train_metrics['loss']:.4f} acc {train_metrics['pattern_acc']:.4f} "
                f"val loss {val_metrics['loss']:.4f} acc {val_metrics['pattern_acc']:.4f} "
                f"progress_mae {val_metrics['progress_mae']:.4f}",
                flush=True,
            )

            wandb_log = {
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/pattern_loss": train_metrics["pattern_loss"],
                "train/progress_loss": train_metrics["progress_loss"],
                "train/pattern_acc": train_metrics["pattern_acc"],
                "train/progress_mae": train_metrics["progress_mae"],
                "train/progress_mae_boundary": train_metrics["progress_mae_boundary"],
                "train/progress_mae_stable": train_metrics["progress_mae_stable"],
                "val/loss": val_metrics["loss"],
                "val/pattern_loss": val_metrics["pattern_loss"],
                "val/progress_loss": val_metrics["progress_loss"],
                "val/pattern_acc": val_metrics["pattern_acc"],
                "val/progress_mae": val_metrics["progress_mae"],
                "val/progress_mae_boundary": val_metrics["progress_mae_boundary"],
                "val/progress_mae_stable": val_metrics["progress_mae_stable"],
                "lr": optimizer.param_groups[0]["lr"],
                "label_smoothing": label_smoothing,
            }
            if wandb_run is not None:
                for mode, values in (("train", train_metrics), ("val", val_metrics)):
                    for key in ("epoch_s", "samples_per_s", "loader_wait_s", "first_batch_wait_s", "profile_gpu_step_ms"):
                        if key in values:
                            wandb_log[f"{mode}/{key}"] = values[key]
                wandb_run.log(wandb_log, step=epoch)

            if scheduler is not None and not scheduler_steps_per_batch(scheduler):
                scheduler.step()

            ckpt = save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer,
                epoch,
                cfg,
                low_mean,
                low_std,
                dataset_splits=source_splits,
                scheduler=scheduler,
            )
            if val_metrics["pattern_acc"] > best_val_acc:
                best_val_acc = val_metrics["pattern_acc"]
                torch.save(ckpt, output_dir / "best.pt")
                if wandb_run is not None:
                    wandb_run.summary["best_val_pattern_acc"] = best_val_acc
                    wandb_run.summary["best_epoch"] = epoch

            if cfg.train.ckpt_every > 0 and epoch % cfg.train.ckpt_every == 0:
                ckpt_path = output_dir / f"epoch_{epoch:03d}.pt"
                torch.save(ckpt, ckpt_path)
                if wandb_run is not None:
                    wandb_run.save(str(ckpt_path))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
