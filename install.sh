#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
PI_DIR="${ROOT_DIR}/policy/pi"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

# Install a matched stack before importing torch. TorchCodec 0.9.x uses torch 2.9.
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .

# SAM2: keep the torch build installed above.
python -m pip install --no-build-isolation --no-deps "git+https://github.com/facebookresearch/sam2.git"

# GroundingDINO: compile against the current torch/CUDA.
# PyTorch 2.9 needs scalar_type() in AT_DISPATCH_FLOATING_TYPES (PR 415).
GDINO_SRC="$(mktemp -d)"
git clone --depth 1 https://github.com/IDEA-Research/GroundingDINO.git "${GDINO_SRC}"
sed -i 's/AT_DISPATCH_FLOATING_TYPES(value.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/' \
  "${GDINO_SRC}/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu"
python -m pip install --no-build-isolation "${GDINO_SRC}"
rm -rf "${GDINO_SRC}"

# π₀.₅: JAX/Flax on the same env. --no-deps keeps HINT's torch 2.9 / transformers 4.57.
python -m pip install -r "${PI_DIR}/requirements.txt"
python -m pip install "git+https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
python -m pip install -e "${PI_DIR}/packages/openpi-client"
python -m pip install --no-deps -e "${PI_DIR}"
# lerobot may pull a newer huggingface-hub; restore the HINT pin for Qwen3-VL.
python -m pip install huggingface-hub==0.34.4 numpy==1.26.4

python - <<'PY'
from pathlib import Path
import sys
import torch

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

import groundingdino
from groundingdino.util.inference import load_model, predict
from qwen_vl_utils import process_vision_info
from sam2.build_sam import build_sam2_video_predictor
from transformers import AutoProcessor

print("sam2:", build_sam2_video_predictor)
print("groundingdino:", Path(groundingdino.__file__).resolve().parent)
print("qwen_vl_utils:", process_vision_info)
print("transformers:", AutoProcessor)

import jax
import openpi.training.config as _pi_config

print("jax:", jax.__version__)
print("jax devices:", jax.devices())
print("openpi:", _pi_config.get_config("pi05_piper_spell_HINT").name)
PY
