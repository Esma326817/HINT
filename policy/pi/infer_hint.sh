# CUDA_VISIBLE_DEVICES=1 python agent_policy/server_agent_piper_stage.py \
#     --port 8001 \
#     --config pi05_piper_sort_HINT \
#     --checkpoint-dir /mnt/hzx/data/ckpt/finetuned/pi05_piper_classify_HINT/30000 \
#     --hint-url http://localhost:8000 \
#     --agent-prompt "Sort fruits into the blue basket and vegetables into the pink basket"

CUDA_VISIBLE_DEVICES=1 python agent_policy/server_agent_piper_stage.py \
    --port 8001 \
    --config pi05_piper_spell_HINT \
    --checkpoint-dir /mnt/hzx/data/ckpt/finetuned/pi05_piper_letter_v2_HINT/32500 \
    --hint-url http://localhost:8000 \
    --agent-prompt "pick the letter"
