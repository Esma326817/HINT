# HINT π

**π₀.₅ policy training and inference for HINT on Piper dual-arm robots.**

Built on [Physical Intelligence's openpi](https://github.com/Physical-Intelligence/openpi). This tree adds semantic attention injection, visual highlighting support, and HINT HTTP integration. It keeps the `openpi` and `openpi_client` package names.

Install the shared conda env from the repository root: [SETUP.md](../SETUP.md). Then run the commands below from `policy/pi/`.

## Data and Configuration

Configurations in [src/openpi/training/config.py](../policy/pi/src/openpi/training/config.py) use the paper's task abbreviations: `sort`, `spell`, and `peg` (Table S1).

| Configuration | Task |
| --- | --- |
| `pi05_piper_sort_HINT` | Fruit–vegetable sorting |
| `pi05_piper_spell_HINT` | Word spelling |
| `pi05_piper_peg_HINT` | Peg-in-hole insertion |

`pi05_piper_sort_mixed_HINT` is an additional sorting configuration for mixed datasets.

The HINT configurations use the following paper-aligned injection settings in `Pi0Config`:

```python
vit_attention_injection=True,
vit_attention_injection_layers=[6, 12, 14, 15, 16, 17, 18, 20, 24],
vit_attention_alpha_mode="fixed",
vit_attention_alpha=1.0,
attention_injection=True,
attention_injection_layers=[15, 16, 17, 18],
attention_alpha_mode="fixed",
attention_alpha=1.0,
```

- **Visual injection:** `vit_attention_injection` enables grounding-guided attention in the **27-layer SigLIP** vision encoder. Following the paper's setup, we select layers **6, 12, 14–18, 20, and 24**.
- **Action-to-vision injection:** `attention_injection` guides action tokens toward image tokens. The **PaliGemma Gemma backbone has 18 transformer layers**, as does the action expert; we inject into the **final four layers (15–18)**. All layer numbers are **1-based**.
- **Injection strength:** both alpha values are **fixed at 1.0**. They remain constant during training and introduce no learnable injection gains.

Our Piper datasets are converted to LeRobot format with the following fields:

- `images.global`, `images.left_wrist`, `images.right_wrist`: three RGB views.
- `state`, `actions`: 14 values per frame, with six joints and one gripper value per arm.
- `task_index` and task metadata: language instructions, loaded through `prompt_from_task=True`.
- Per-camera grounding columns: `16 × 16` patch weights in `[0, 1]`, aligned with the `224 × 224` policy images.

Grounding column names are specified when exporting data from HINT. In `PiperDataConfig`, `semantic_grounding_keys` maps each policy camera to its exported dataset column:

```python
semantic_grounding_keys={
    "global": "global_semantic_grounding",
    "left_wrist": "left_wrist_semantic_grounding",
    "right_wrist": "right_wrist_semantic_grounding",
},
```

The values on the right must match `dataset.semantic_grounding_columns` in the HINT data YAML (for example, `configs/data/reasoning_agent_classify.yaml`). Update this mapping if you export under different names.

Field names and representations depend on your data conversion pipeline. For your own datasets, adjust the field mappings and data transforms in [PiperDataConfig](../policy/pi/src/openpi/training/config.py) and [piper_policy.py](../policy/pi/src/openpi/policies/piper_policy.py) to match your converted data.

Prepare highlighted training images and grounding maps consistently with the intended inference mode. The HINT annotation pipeline is at the repository root (`python -m dataset_export...` and `python -m inference.api_server`). By default, joint actions are converted to deltas for training and restored to absolute values at inference; gripper actions remain absolute.

## Training

After updating the selected configuration, compute normalization statistics and train:

```bash
python scripts/compute_norm_stats.py --config-name pi05_piper_spell_HINT

CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py pi05_piper_spell_HINT \
    --exp-name=hint_spell \
    --checkpoint-base-dir=./checkpoints \
    --no-wandb-enabled
```

Statistics are saved to `assets/<config>/<asset_id>/norm_stats.json` and included in training checkpoints. Checkpoints are written to `checkpoints/<config>/<exp_name>/<step>/`, every 5,000 steps in the provided HINT configurations. Add `--resume` to continue the same experiment. To enable Weights & Biases, configure your account and remove `--no-wandb-enabled`.

## Inference

Start HINT's `api_server` first (`/health`, `/reset`, `/step`). It must return camera renders and semantic grounding in the format expected by [VLMClient](../policy/pi/agent_policy/server_agent_piper_stage.py), including matching `frame_id` values for attention maps.

Then launch the policy server using an existing checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 \
python agent_policy/server_agent_piper_stage.py \
    --host 127.0.0.1 \
    --port 8001 \
    --config pi05_piper_spell_HINT \
    --checkpoint-dir ./checkpoints/pi05_piper_spell_HINT/hint_spell/5000 \
    --hint-url http://127.0.0.1:8000 \
    --agent-prompt "pick the letter"
```

The checkpoint directory must contain `params/` and `assets/`; use the same configuration and `asset_id` as training. For remote clients, set `--host` to an address reachable from the robot.

Connect with [WebsocketClientPolicy](../policy/pi/packages/openpi-client/src/openpi_client/websocket_client_policy.py) and call `infer(observation)` with `observation/global`, `observation/left_wrist`, `observation/right_wrist` (HWC RGB `uint8`), `observation/state` (14 values), and `prompt`. The server obtains semantic inputs from HINT and returns `actions` with 14 values per step. Action smoothing is enabled by default.

[train_attention.sh](../policy/pi/train_attention.sh) and [infer_hint.sh](../policy/pi/infer_hint.sh) contain experiment launch examples. Update their local paths, GPU IDs, and service settings before use.
