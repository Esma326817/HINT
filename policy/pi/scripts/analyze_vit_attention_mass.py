"""Measure SigLIP target attention mass with and without fixed grounding bias.

This diagnostic is parameter-free: it loads an existing Pi checkpoint, runs the
same images twice with ViT alpha 0 and 1, and reconstructs the per-head Q/K
attention probabilities at the configured injection layers.
"""

import argparse
import json
import logging
import pathlib
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import siglip as _siglip
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

CAMERA_TO_DATASET_COLUMN = {
    "base_0_rgb": "global_semantic_grounding",
    "left_wrist_0_rgb": "left_wrist_semantic_grounding",
    "right_wrist_0_rgb": "right_wrist_semantic_grounding",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--samples-per-camera", type=int, default=4)
    parser.add_argument("--min-index-gap", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def _unwrap_lerobot_dataset(dataset):
    while hasattr(dataset, "_dataset"):
        dataset = dataset._dataset  # noqa: SLF001
    if not hasattr(dataset, "hf_dataset"):
        raise TypeError(f"Expected a LeRobotDataset, got {type(dataset).__name__}")
    return dataset


def _map_sum(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float32).sum())


def select_grounded_indices(raw_dataset, *, samples_per_camera: int, min_index_gap: int) -> dict[str, list[int]]:
    lerobot_dataset = _unwrap_lerobot_dataset(raw_dataset)
    columns = list(CAMERA_TO_DATASET_COLUMN.values())
    grounding_rows = lerobot_dataset.hf_dataset.select_columns(columns)
    selected = {camera: [] for camera in CAMERA_TO_DATASET_COLUMN}

    for index, row in enumerate(grounding_rows):
        for camera, column in CAMERA_TO_DATASET_COLUMN.items():
            camera_indices = selected[camera]
            if len(camera_indices) >= samples_per_camera or _map_sum(row[column]) <= 0:
                continue
            if not camera_indices or index - camera_indices[-1] >= min_index_gap:
                camera_indices.append(index)
        if all(len(indices) >= samples_per_camera for indices in selected.values()):
            break

    missing = {camera: len(indices) for camera, indices in selected.items() if len(indices) < samples_per_camera}
    if missing:
        raise ValueError(f"Could not find enough grounded samples: {missing}")
    return selected


def load_camera_batch(dataset, camera: str, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    samples = [dataset[index] for index in indices]
    images = np.stack([np.asarray(sample["image"][camera]) for sample in samples])
    weights = np.stack(
        [np.asarray(sample["semantic_grounding_map"][camera], dtype=np.float32).reshape(-1) for sample in samples]
    )
    return images, weights


def make_vit(*, injection_layers: tuple[int, ...], alpha: float) -> nn.Module:
    return _siglip.Module(
        num_classes=2048,
        variant="So400m/14",
        pool_type="none",
        scan=True,
        dtype_mm="bfloat16",
        attention_injection=True,
        attention_injection_layers=injection_layers,
        attention_alpha_mode="fixed",
        attention_alpha=alpha,
    )


def _layer_input(output: dict[str, Any], layer_index: int) -> jax.Array:
    if layer_index == 0:
        return output["with_posemb"].astype(jnp.bfloat16)
    return output["encoder"][f"block{layer_index - 1:02d}"]["+mlp"]


def _project_qk(
    x: jax.Array,
    *,
    layer_index: int,
    transformer_params: dict[str, Any],
) -> tuple[jax.Array, jax.Array]:
    block_params = transformer_params["encoderblock"]
    norm_params = {
        "scale": block_params["LayerNorm_0"]["scale"][layer_index],
        "bias": block_params["LayerNorm_0"]["bias"][layer_index],
    }
    normed = nn.LayerNorm(dtype="bfloat16").apply({"params": norm_params}, x)
    attention_params = block_params["MultiHeadDotProductAttention_0"]

    def project(name: str) -> jax.Array:
        params = attention_params[name]
        projected = jnp.einsum("bqd,dhn->bqhn", normed, params["kernel"][layer_index]) + params["bias"][layer_index]
        return projected.astype(jnp.float32)

    return project("query"), project("key")


def _attention_statistics(probs: jax.Array, weights: jax.Array) -> dict[str, jax.Array]:
    support = weights > 0
    support_mass = jnp.sum(probs * support[:, None, None, :], axis=-1)
    weighted_mass = jnp.sum(probs * weights[:, None, None, :], axis=-1)
    entropy = -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs), 0.0), axis=-1) / jnp.log(probs.shape[-1])
    return {
        "support_mass": jnp.mean(support_mass, axis=-1),
        "weighted_mass": jnp.mean(weighted_mass, axis=-1),
        "normalized_entropy": jnp.mean(entropy, axis=-1),
    }


def collect_statistics(
    vit: nn.Module,
    params: dict[str, Any],
    images: jax.Array,
    weights: jax.Array,
    *,
    injection_layers: tuple[int, ...],
    diagnostic_alpha: float,
) -> dict[str, jax.Array]:
    _, output = vit.apply({"params": params}, images, weights, train=False)
    transformer_params = params["Transformer"]
    stats: dict[str, list[jax.Array]] = {
        "raw_support_mass": [],
        "biased_support_mass": [],
        "raw_weighted_mass": [],
        "biased_weighted_mass": [],
        "raw_normalized_entropy": [],
        "biased_normalized_entropy": [],
    }

    for layer_number in injection_layers:
        layer_index = layer_number - 1
        query, key = _project_qk(
            _layer_input(output, layer_index),
            layer_index=layer_index,
            transformer_params=transformer_params,
        )
        logits = jnp.einsum("bqhd,bkhd->bhqk", query, key) * query.shape[-1] ** -0.5
        raw_probs = jax.nn.softmax(logits, axis=-1)
        biased_probs = jax.nn.softmax(
            logits + diagnostic_alpha * weights[:, None, None, :],
            axis=-1,
        )
        raw = _attention_statistics(raw_probs, weights)
        biased = _attention_statistics(biased_probs, weights)
        for metric in ("support_mass", "weighted_mass", "normalized_entropy"):
            stats[f"raw_{metric}"].append(raw[metric])
            stats[f"biased_{metric}"].append(biased[metric])

    return {name: jnp.stack(values) for name, values in stats.items()}


def _summarize(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "per_head_mean": values.mean(axis=0).tolist(),
    }


def summarize_camera(
    *,
    indices: list[int],
    weights: np.ndarray,
    alpha_zero_stats: dict[str, np.ndarray],
    alpha_one_stats: dict[str, np.ndarray],
    injection_layers: tuple[int, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "indices": indices,
        "coverage_sum": weights.sum(axis=-1).tolist(),
        "support_patch_count": (weights > 0).sum(axis=-1).tolist(),
        "max_patch_coverage": weights.max(axis=-1).tolist(),
        "layers": {},
    }
    for layer_offset, layer_number in enumerate(injection_layers):
        layer_result = {}
        for name, values in alpha_zero_stats.items():
            layer_result[f"alpha0_state_{name}"] = _summarize(values[layer_offset])
        for name, values in alpha_one_stats.items():
            layer_result[f"alpha1_state_{name}"] = _summarize(values[layer_offset])
        result["layers"][str(layer_number)] = layer_result
    return result


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_config = _config.get_config(args.config_name)
    injection_layers = tuple(train_config.model.vit_attention_injection_layers)
    if not injection_layers:
        raise ValueError(f"Config {args.config_name} has no ViT attention injection layers")

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    raw_dataset = _data_loader.create_torch_dataset(
        data_config,
        train_config.model.action_horizon,
        train_config.model,
    )
    dataset = _data_loader.transform_dataset(raw_dataset, data_config, skip_norm_stats=True)
    selected = select_grounded_indices(
        raw_dataset,
        samples_per_camera=args.samples_per_camera,
        min_index_gap=args.min_index_gap,
    )
    logging.info("Selected grounded indices: %s", selected)

    checkpoint_params = _model.restore_params(args.checkpoint_dir / "params", dtype=jnp.bfloat16)
    vision_params = checkpoint_params["PaliGemma"]["img"]
    del checkpoint_params

    vit_alpha_zero = make_vit(injection_layers=injection_layers, alpha=0.0)
    vit_alpha_one = make_vit(injection_layers=injection_layers, alpha=args.alpha)

    alpha_zero_fn = jax.jit(
        lambda images, weights: collect_statistics(
            vit_alpha_zero,
            vision_params,
            images,
            weights,
            injection_layers=injection_layers,
            diagnostic_alpha=args.alpha,
        )
    )
    alpha_one_fn = jax.jit(
        lambda images, weights: collect_statistics(
            vit_alpha_one,
            vision_params,
            images,
            weights,
            injection_layers=injection_layers,
            diagnostic_alpha=args.alpha,
        )
    )

    report = {
        "config_name": args.config_name,
        "checkpoint_dir": str(args.checkpoint_dir),
        "diagnostic_alpha": args.alpha,
        "injection_layers": list(injection_layers),
        "cameras": {},
    }
    for camera, indices in selected.items():
        images_np, weights_np = load_camera_batch(dataset, camera, indices)
        images = jnp.asarray(images_np)
        weights = jnp.asarray(weights_np)
        logging.info("Running %s with batch size %d", camera, len(indices))
        alpha_zero_stats = jax.device_get(alpha_zero_fn(images, weights))
        alpha_one_stats = jax.device_get(alpha_one_fn(images, weights))
        report["cameras"][camera] = summarize_camera(
            indices=indices,
            weights=weights_np,
            alpha_zero_stats=alpha_zero_stats,
            alpha_one_stats=alpha_one_stats,
            injection_layers=injection_layers,
        )

    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        logging.info("Wrote %s", args.output)
    else:
        print(rendered)


if __name__ == "__main__":
    main()
