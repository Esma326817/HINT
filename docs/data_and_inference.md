# Data processing and inference configuration

Use `configs/data/` for offline LeRobot preparation and
[`configs/inference/`](../configs/inference) for the online HINT service. Both use task
definitions from [`configs/tasks/`](task_configuration.md). Run the commands below
from the repository root after [installation](../SETUP.md).

## Reference configs

| Task | Offline config | Online config |
| --- | --- | --- |
| Fruit/vegetable sorting | [reasoning_agent_classify.yaml](../configs/data/reasoning_agent_classify.yaml) | [reasoning_agent_classify_predict.yaml](../configs/inference/reasoning_agent_classify_predict.yaml) |
| Letter spelling | [reasoning_agent_letter.yaml](../configs/data/reasoning_agent_letter.yaml) | [reasoning_agent_letter_predict.yaml](../configs/inference/reasoning_agent_letter_predict.yaml) |
| Peg insertion | [reasoning_agent_peg_in_hole.yaml](../configs/data/reasoning_agent_peg_in_hole.yaml) | [reasoning_agent_peg_in_hole_predict.yaml](../configs/inference/reasoning_agent_peg_in_hole_predict.yaml) |

These YAML files are examples to copy and override, not configs to run as-is.
Each directory has its own `_base.yaml`. `extends: _base` inherits that file;
nested maps merge, lists replace. Set absolute paths for dataset roots and
model weights; the example paths are machine-specific. YAML values do not
expand environment variables such as `$HOME`.

## Configure the models

| Setting | Purpose |
| --- | --- |
| `vlm.model_path` | Local Qwen-VL model directory. Set it explicitly; a task override of `null` uses model-path discovery. `QWEN_MODEL_PATH` takes precedence. |
| `grounding.reset_detector` | Scene detection at reset: `dino` or `qwen`. Sorting and spelling use `dino`; peg insertion uses `qwen`. |
| `grounding.dino.checkpoint` | GroundingDINO `.pth` weights. |
| `grounding.grounder` | Target localization after reset; the supplied configs use `qwen`. This is separate from the reset detector. |
| `tracker.sam2_checkpoint` | SAM2 weights. |
| `tracker.sam2_model_cfg` | Corresponding SAM2 architecture, e.g. `configs/sam2.1/sam2.1_hiera_s.yaml` for Hiera-S. This path is inside the SAM2 package. |
| `stage_source.predict.checkpoint` | Trained pattern router `.pt`; needed for online `mode: predict`. See [router training](pattern_training.md). |

The supplied tasks can override shared model settings, so check the merged
config rather than editing only `_base.yaml`. Checkpoint downloads are described
in [SETUP.md](../SETUP.md#checkpoints).

## Offline data processing

Input datasets contain `data/chunk-*/episode_*.parquet`, all three camera videos
under `videos/chunk-*/images.{global,left_wrist,right_wrist}/`, and LeRobot
`meta/` files. The annotation workflow reads `state` and `stage_id_gt`; camera
frames must align with parquet rows.

Create `configs/data/my_classify.yaml`:

```yaml
extends: reasoning_agent_classify
vlm:
  model_path: /absolute/path/to/Qwen3-VL-8B-Instruct
grounding:
  reset_detector: dino
  dino:
    checkpoint: /absolute/path/to/groundingdino_swint_ogc.pth
tracker:
  sam2_checkpoint: /absolute/path/to/sam2.1_hiera_small.pt
  sam2_model_cfg: configs/sam2.1/sam2.1_hiera_s.yaml
```

Start with one episode:

```bash
python -m dataset_export.preprocessing.prepare_dataset \
  --config configs/data/my_classify.yaml \
  --src-root /absolute/path/to/raw_dataset \
  --dst-root outputs/classify_preview \
  --semantic-intent-injection highlighting \
  --gpu-ids 0 --num-workers 1 --limit 1
```

`--gpu-ids` selects physical GPU IDs for the worker processes. Remove `--limit`
after inspecting the preview; use `--episode-ids 0,2,4` for selected episodes.
The command-line injection mode overrides `task.semantic_intent_injection`.

| Mode | Output |
| --- | --- |
| `highlighting` | Highlighted camera videos under `--dst-root`. |
| `attention` | Dataset with per-camera spatial guidance columns under `--dst-root`. |
| `both` | One dataset under `--dst-root`: highlighted camera videos plus per-camera spatial guidance columns in each episode parquet. |

Output column names are configured in `dataset.semantic_grounding_columns` in
`configs/data/*.yaml` (shared defaults in `_base.yaml`):

```yaml
dataset:
  semantic_grounding_columns:
    global: global_semantic_grounding
    left_wrist: left_wrist_semantic_grounding
    right_wrist: right_wrist_semantic_grounding
```

Use these same values in the policy's `semantic_grounding_keys`. The exporter
also registers the configured columns in `meta/info.json`. Both mode no longer
creates a separate `_attention` dataset; omit `--attention-dst-root`.

Use `render_stage_aware_summary.json` in the output directory to find failed
episodes and reasons. A successful export still needs a visual check that the
intended object and destination are highlighted.

**Offline masks:** the batch exporter uses SAM2 video propagation for normal
segments, even when the offline config says `tracker.backend: bbox_mask`.
Task-level `render: area` uses a rectangle instead. To segment a basket, leave
that setting unset and locate the destination on the current frame; see
[task highlighting](task_configuration.md#highlighting).

### Episode instructions for peg insertion

Peg insertion is a *fixed* two-step task whose selected block color and peg
shape change every episode. Sorting and spelling recover that identity at reset
from the scene, so they default to `dataset.task_context.source: none`. Peg
needs a per-episode sidecar.

The peg config requires `meta/episode_prompts.jsonl` inside the source dataset.
Each row provides the episode instruction and the selected object attributes:

```json
{"episode_index": 0, "instruction": "place the green rectangular block into the black rectangular slot, then insert the white circular peg into the circular hole in the green rectangular block", "selected_block_color": "green", "selected_peg_shape": "circular"}
```

`dataset.task_context.source: jsonl` reads this file; `fields` lists the additional
attributes to pass to the task. To use a file elsewhere, pass
`--task-context-file /absolute/path/to/episode_prompts.jsonl`. Relative sidecar
paths resolve against the source dataset. `source: lerobot` reads instructions
from `meta/episodes.jsonl`.

If another task needs the same kind of per-episode prompt sidecar, follow the
peg-in-hole annotator: it keeps the finish-frame / empty-hole visual pipeline,
but reads the output path, instruction field, camera key, stage column, and
color/shape vocabulary from the data config plus
[`configs/tasks/peg_in_hole.yaml`](../configs/tasks/peg_in_hole.yaml).

```bash
python -m dataset_export.tools.annotate_peg_in_hole_prompts \
  --dataset-root /absolute/path/to/lerobot_dataset \
  --config configs/data/reasoning_agent_peg_in_hole.yaml
```

The command writes `<dataset-root>/meta/episode_prompts.jsonl` by default
(`dataset.task_context.path`), plus labeled finish frames under `check_images/`
for review. Set `dataset.task_context.write_to_lerobot: true` (or pass
`--write-to-lerobot`) to copy those instructions into `tasks.jsonl` /
`episodes.jsonl` and parquet `task_index` once the sidecar covers every
episode. The same copy can still be run later with
`python -m dataset_export.tools.annotate_prompt_to_lerobot`.

## Online inference

Create `configs/inference/my_classify.yaml`. Set the same Qwen, DINO, and SAM2
paths as above, plus the trained router checkpoint:

```yaml
extends: reasoning_agent_classify_predict
output:
  root: outputs
  log_dir: outputs/logs
vlm:
  model_path: /absolute/path/to/Qwen3-VL-8B-Instruct
grounding:
  reset_detector: dino
  dino:
    checkpoint: /absolute/path/to/groundingdino_swint_ogc.pth
tracker:
  backend: sam2_memory
  sam2_checkpoint: /absolute/path/to/sam2.1_hiera_small.pt
  sam2_model_cfg: configs/sam2.1/sam2.1_hiera_s.yaml
stage_source:
  mode: predict
  predict:
    checkpoint: /absolute/path/to/pattern/best.pt
deploy:
  infer_mode: sync
```

```bash
CUDA_VISIBLE_DEVICES=0 \
REASONING_AGENT_CONFIG=configs/inference/my_classify.yaml \
python -m inference.api_server
```

The service listens on port 8000 and loads models at startup. Open
`http://127.0.0.1:8000/docs` for the current request and response schemas.
`GET /health` reports readiness, `POST /reset` initializes an episode from its
global image and robot state, and `POST /step` processes subsequent frames.
One service process maintains one active episode.

| Setting | Effect |
| --- | --- |
| `task.semantic_intent_injection` | `highlighting`, `attention`, or `both`. |
| `task.model_input_resolution`, `output_resolution` | Policy input and returned image sizes; the supplied π₀.₅ setup uses `[224, 224]`. |
| `tracker.backend` | `sam2_memory` tracks masks with history; `sam2` segments each frame from a box; `bbox_mask` produces rectangular masks. |
| `grounding.ground_every_step` | Re-run localization on each active step; increases model calls. |
| `stage_aware.stage_stability_frames` | Number of consistent stage predictions before switching. |
| `stage_aware.free_move_grounding_delay_frames` | Delay initial approach grounding while the arm clears the view. |
| `stage_aware.skip_stages` | Stages excluded from highlighting and attention guidance. |
| `deploy.infer_mode` | Start with `sync`. Background tracking additionally needs a camera service at `deploy.img_url`. |

The Piper client sends `robot_state` as a JSON array of 14 values. Supply effort
or explicit low-dimensional inputs for a router trained with state and effort.
For the supplied router, `low_dim_window` is a 12×28 JSON array ordered oldest to
newest. Build it at the training sampling rate; the server's fallback history
advances once per request, which can be slower.

Images in responses are base64-encoded (JPEG by default). Attention modes also
return per-camera `semantic_grounding`. The action policy consumes these outputs;
see [π₀.₅ serving](pi_policy.md) for the integrated client.
