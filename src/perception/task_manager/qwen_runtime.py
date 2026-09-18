from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
import os
import time

import torch
from transformers import AutoProcessor

from common.config_loader import PROJECT_ROOT

# Default base-model directory where downloaded checkpoints live. Drop new models
# here (e.g. Qwen3-VL) and point QWEN_MODEL_PATH at them to A/B against Qwen2.5.
BASE_MODELS_DIR = Path("/data/checkpoints/base_models")

# Candidate model directory names, in preference order. The first that exists is
# used when QWEN_MODEL_PATH is not set. Add new versions here as you download them.
# Qwen3-VL-8B is preferred: it reads specific target objects correctly (e.g. a wolf
# as "wolf"), where Qwen2.5-VL-7B over-generalized to "dog".
_MODEL_DIR_CANDIDATES = (
    "Qwen3-VL-8B-Instruct",
    "Qwen3-VL-7B-Instruct",
    "Qwen2.5-VL-7B-Instruct",
)


def _default_qwen_model_path() -> Path:
    env_value = os.getenv("QWEN_MODEL_PATH")
    if env_value:
        return Path(env_value)

    search_roots = (PROJECT_ROOT / "models", Path.home() / "models", BASE_MODELS_DIR)
    candidates = [root / name for root in search_roots for name in _MODEL_DIR_CANDIDATES]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_QWEN_MODEL_PATH = str(_default_qwen_model_path())

# Which transformers class loads the model. "auto" dispatches by the model's own
# config and works for both Qwen2.5-VL and Qwen3-VL; set QWEN_MODEL_CLASS to an
# explicit class name (e.g. "Qwen3VLForConditionalGeneration") to force one.
DEFAULT_QWEN_MODEL_CLASS = os.getenv("QWEN_MODEL_CLASS", "auto")


def _is_configured_path(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return text not in ("", "null", "none", "~")


def resolve_qwen_model_settings(config: dict[str, Any] | None = None) -> tuple[str, str]:
    """Resolve Qwen model path/class from env, then config ``vlm`` block, then defaults."""
    vlm_cfg = {}
    if isinstance(config, dict) and isinstance(config.get("vlm"), dict):
        vlm_cfg = config["vlm"]

    model_class = os.getenv("QWEN_MODEL_CLASS") or str(vlm_cfg.get("model_class") or DEFAULT_QWEN_MODEL_CLASS)

    env_path = os.getenv("QWEN_MODEL_PATH")
    if _is_configured_path(env_path):
        return str(env_path), model_class

    cfg_path = vlm_cfg.get("model_path")
    if _is_configured_path(cfg_path):
        return str(cfg_path), model_class

    return DEFAULT_QWEN_MODEL_PATH, model_class


@dataclass
class QwenRuntime:
    model: Any
    processor: AutoProcessor
    device: str
    model_path: str = ""
    model_class: str = ""


_runtime: QwenRuntime | None = None
_runtime_lock = Lock()


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def _resolve_model_class(class_name: str):
    """Return the transformers class used to load the VLM.

    "auto" uses AutoModelForImageTextToText (config-dispatched, supports both
    Qwen2.5-VL and Qwen3-VL). An explicit name is looked up on the transformers
    module. Both paths fall back to Qwen2_5_VLForConditionalGeneration if the
    requested class is unavailable in the installed transformers version.
    """
    import transformers

    if class_name and class_name.lower() != "auto":
        model_cls = getattr(transformers, class_name, None)
        if model_cls is None:
            raise ValueError(
                f"QWEN_MODEL_CLASS={class_name!r} not found in transformers "
                f"{transformers.__version__}; upgrade transformers or use 'auto'."
            )
        return model_cls

    auto_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if auto_cls is not None:
        return auto_cls
    # Older transformers without the generic auto class: fall back to Qwen2.5-VL.
    return transformers.Qwen2_5_VLForConditionalGeneration


def _speedup_vision_patch_embed(model) -> None:
    """Run the Qwen-VL vision patch-embed Conv3d in fp32.

    On torch 2.9+cu128 the bf16/fp16 Conv3d patch embedder hits a pathological
    cuDNN fallback (~9 s per image on an RTX 3090); the identical conv in fp32
    runs in <1 ms. The patch embed is a tiny, non-overlapping projection so fp32
    here costs nothing and does not change the model's compute precision
    elsewhere. Without this, every grounding/recognition VLM call is dominated by
    a multi-second vision encode.
    """
    import torch

    visual = getattr(model, "visual", None) or getattr(getattr(model, "model", None), "visual", None)
    patch_embed = getattr(visual, "patch_embed", None)
    proj = getattr(patch_embed, "proj", None)
    if proj is None:
        return
    orig_dtype = proj.weight.dtype
    if orig_dtype == torch.float32:
        return
    patch_embed.proj = proj.float()
    in_c = patch_embed.in_channels
    t = patch_embed.temporal_patch_size
    p = patch_embed.patch_size
    emb = patch_embed.embed_dim

    def _forward(hidden_states):
        hs = hidden_states.view(-1, in_c, t, p, p).float()
        return patch_embed.proj(hs).view(-1, emb).to(orig_dtype)

    patch_embed.forward = _forward
    print("[qwen_runtime] patched vision patch_embed to fp32 conv (avoids slow bf16 Conv3d)", flush=True)


def load_qwen_runtime(
    model_path: str = DEFAULT_QWEN_MODEL_PATH,
    model_class: str = DEFAULT_QWEN_MODEL_CLASS,
) -> QwenRuntime:
    start_time = time.time()
    print(
        f"[qwen_runtime] loading Qwen runtime from {model_path} (class={model_class})",
        flush=True,
    )
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Qwen model not found: {model_path}. "
            f"Set QWEN_MODEL_PATH to a local model dir under {BASE_MODELS_DIR}, "
            "or QWEN_MODEL_CLASS to force a loader class."
        )
    device = _resolve_device()
    dtype = _resolve_dtype(device)

    model_cls = _resolve_model_class(model_class)
    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    if device == "cuda":
        _speedup_vision_patch_embed(model)
    processor = AutoProcessor.from_pretrained(model_path)
    elapsed = time.time() - start_time
    print(
        f"[qwen_runtime] Qwen runtime ready on {device} in {elapsed:.2f}s "
        f"(class={model_cls.__name__})",
        flush=True,
    )
    return QwenRuntime(
        model=model,
        processor=processor,
        device=device,
        model_path=str(model_path),
        model_class=model_cls.__name__,
    )


def get_qwen_runtime(
    model_path: str = DEFAULT_QWEN_MODEL_PATH,
    model_class: str = DEFAULT_QWEN_MODEL_CLASS,
) -> QwenRuntime:
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = load_qwen_runtime(model_path=model_path, model_class=model_class)
    return _runtime


def initialize_qwen_runtime(
    model_path: str = DEFAULT_QWEN_MODEL_PATH,
    model_class: str = DEFAULT_QWEN_MODEL_CLASS,
) -> QwenRuntime:
    """Eagerly initialize the singleton runtime during application startup."""
    print("[qwen_runtime] initialize_qwen_runtime() called", flush=True)
    return get_qwen_runtime(model_path=model_path, model_class=model_class)


def initialize_qwen_runtime_from_config(config: dict[str, Any] | None = None) -> QwenRuntime:
    """Initialize Qwen from ``vlm.model_path`` / ``vlm.model_class`` in ReasoningAgent YAML."""
    model_path, model_class = resolve_qwen_model_settings(config)
    return initialize_qwen_runtime(model_path=model_path, model_class=model_class)


@contextlib.contextmanager
def qwen_generate_context(runtime: "QwenRuntime | None" = None):
    """Run a Qwen forward/generate with CUDA autocast DISABLED.

    The vision patch-embed Conv3d is patched to fp32 (see
    ``_speedup_vision_patch_embed``) to dodge a pathological multi-second bf16
    cuDNN fallback. But callers (e.g. the stage-aware tracker) may hold an outer
    ``torch.autocast("cuda", bf16)`` for SAM2; that autocast re-casts the conv
    inputs back to bf16 and reinstates the ~9 s/call slowdown. Qwen weights are
    already bf16, so disabling autocast here costs nothing for the LLM/attention
    (they run natively in bf16) while keeping the patched conv in fast fp32.

    Always pair every ``runtime.model.generate`` for the VLM with this context.
    """
    device = getattr(runtime, "device", None)
    if (device == "cuda" or device is None) and torch.cuda.is_available():
        with torch.autocast("cuda", enabled=False):
            yield
    else:
        yield


def is_model_ready() -> bool:
    return _runtime is not None


def reset_qwen_runtime() -> None:
    """Release the singleton and its CUDA weights before another large model loads."""
    import gc

    global _runtime
    with _runtime_lock:
        runtime = _runtime
        _runtime = None
    if runtime is not None:
        del runtime
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
