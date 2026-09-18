# Setup

From the repository root. You need an NVIDIA driver, a CUDA toolkit matching
the PyTorch build, a C++ compiler (GroundingDINO), and `ffmpeg` / `ffprobe` on
`PATH`.

## Install

One conda environment covers HINT perception and π₀.₅ (JAX). Do not mix both
models in the same Python process; run `api_server` and the policy server as
two processes.

```bash
conda create -n hint python=3.11 -y
conda activate hint
bash install.sh
```

`install.sh` installs PyTorch 2.9.1 + torchvision 0.24.1 (CUDA 12.8), HINT
`requirements.txt`, editable HINT packages from `src/` and `pattern/`, SAM2,
GroundingDINO, then JAX 0.5.3 / Flax and editable
`policy/pi`. It keeps HINT's `transformers` 4.57 and `torchcodec` 0.9.1;
π is installed with `--no-deps` so those pins are not overwritten. For another
CUDA version, change the wheel index in `install.sh` first.

Check both stacks:

```bash
python -c "import torch, groundingdino, sam2, transformers; print(torch.__version__, torch.cuda.is_available())"
python -c "import jax, openpi; print(jax.devices())"
```

π commands run from `policy/pi/` after `conda activate hint`. See
[π₀.₅ training and inference](docs/pi_policy.md).

Router training needs a compatible PyTorch stack, `pattern/requirements.txt`,
and the local package installation below. Optional: `pip install pytest` to run
`python -m pytest tests`.

In an existing environment, install the new source layout without changing its
dependencies:

```bash
python -m pip install --no-deps -e .
```

The editable installation resolves `pattern` from the repository root, the other
HINT packages from `src/`, and default configuration from `configs/`, including when launched from
another directory. Explicit relative `--config` paths still use the working
directory. Continue using `python -m ...`; setting `PYTHONPATH` is unnecessary.

## Checkpoints

Download weights, then paste **absolute paths** into the YAML you run under
`configs/`. HINT does not expand `$CKPT_ROOT`.

```bash
export CKPT_ROOT=/path/to/checkpoints/base_models
mkdir -p "$CKPT_ROOT"

wget -O "$CKPT_ROOT/groundingdino_swint_ogc.pth" \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

wget -O "$CKPT_ROOT/sam2.1_hiera_small.pt" \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

hf download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir "$CKPT_ROOT/Qwen3-VL-8B-Instruct"
```

| Weight | YAML | Env override |
| --- | --- | --- |
| GroundingDINO | `grounding.dino.checkpoint` | `GROUNDING_DINO_WEIGHTS` |
| SAM2.1 Hiera-S | `tracker.sam2_checkpoint` | — |
| SAM2 config | `tracker.sam2_model_cfg: configs/sam2.1/sam2.1_hiera_s.yaml` | from the SAM2 package, not `HINT/configs/` |
| Qwen3-VL-8B | `vlm.model_path` | `QWEN_MODEL_PATH` |
| Pattern router | `stage_source.predict.checkpoint` | — |

A different SAM2 family needs its matching `sam2_model_cfg`. Tracker backend
(`sam2`, `sam2_memory`, `bbox_mask`) is a separate field.

Commands are in [README.md](README.md).
