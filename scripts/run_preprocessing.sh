#!/usr/bin/env bash
# Stage-aware three-camera batch render for annotated LeRobot letter datasets.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_ROOT="${SRC_ROOT:-/dataset/robot/real_world/piper/lerobot/letter/piper_letter_v2_big_annotation_merged}"
DST_ROOT="${DST_ROOT:-/dataset/robot/real_world/piper/lerobot/letter/piper_letter_v2_big_annotation_merged_stage_rendered}"
CONFIG="${REASONING_AGENT_CONFIG:-${ROOT}/configs/data/reasoning_agent_letter.yaml}"
GPU_IDS="${GPU_IDS:-0,1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
# Use the active installed environment, or an explicit interpreter override.
if [[ -n "${REASONING_AGENT_PYTHON:-}" ]]; then
  PYTHON="${REASONING_AGENT_PYTHON}"
else
  PYTHON="python3"
fi

cd "${ROOT}"

"${PYTHON}" -m dataset_export.preprocessing.prepare_dataset \
  --src-root "${SRC_ROOT}" \
  --dst-root "${DST_ROOT}" \
  --config "${CONFIG}" \
  --gpu-ids "${GPU_IDS}" \
  --num-workers "${NUM_WORKERS}" \
  --state-column state \
  --overwrite \
  --global-render-mode block_mask \
  "$@"
