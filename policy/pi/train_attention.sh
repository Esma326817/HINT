export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=2,3
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_LEROBOT_HOME=/dataset/robot/real_world/piper/lerobot

# python scripts/compute_norm_stats.py --config-name pi05_piper_spell_HINT
python scripts/train.py pi05_piper_sort_HINT --exp-name=v1 --resume
