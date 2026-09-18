<div align="center">
  <h1>HINT: Human-Intent Inception for Long-Horizon Robot Manipulation</h1>

  <p>
    <a href="https://github.com/zming-Mei">Mingyu Mei</a><sup>1</sup>,
    <a href="https://github.com/xuhaojie026">Haojie Xu</a><sup>1</sup>,
    <a href="https://github.com/orgs/ZJU-RVIL/people/kirchhoff114514">Shihao Jin</a><sup>1</sup>,
    <a href="https://github.com/SelfGala">Zibo Dai</a><sup>1</sup>,
    <a href="https://github.com/Clement1na">Qihao Cheng</a><sup>1</sup>,
    <a href="mailto:3240103193@zju.edu.cn">Zhengrui Lv</a><sup>1</sup>,
    <a href="https://tonyfang.net/">Hongjie Fang</a><sup>2</sup>,
    <a href="https://www.linkedin.com/in/shiruntang/">Shirun Tang</a><sup>3</sup>,
    <a href="mailto:icomputing@126.com">Guang Chen</a><sup>1,4</sup>,
    <a href="mailto:zhaoxinyue@zju.edu.cn">Xinyue Zhao</a><sup>1</sup>,
    <a href="https://person.zju.edu.cn/en/shenhl">Huiliang Shen</a><sup>1</sup>,
    <a href="https://person.zju.edu.cn/en/zaixinghe">Zaixing He</a><sup>1,†</sup>
  </p>

  <p>
    <sup>1</sup>Zhejiang University &ensp;
    <sup>2</sup>Shanghai Jiao Tong University &ensp;
    <sup>3</sup>Noematrix &ensp;
    <sup>4</sup>EndlessAI
  </p>
  <p><sup>†</sup>Corresponding author.</p>

  <p>
    <a href="https://arxiv.org/abs/2609.02653">
      <img src="https://img.shields.io/badge/Paper-arXiv%3A2609.02653-b31b1b.svg" alt="Paper">
    </a>
    <a href="https://robot-hint.github.io/">
      <img src="https://img.shields.io/badge/Project-Website-blue.svg" alt="Project Website">
    </a>
  </p>
</div>




HINT is an agentic framework for long-horizon manipulation. A pattern router
decides *which* camera to trust and *when* to update semantics. At those
transitions a task manager and grounder resolve the active subtask and target;
tracking holds that commitment between updates. The tracked intent is injected
into the action policy with visual highlighting and an attention prior, without
adding trainable parameters to the foundation model.



## 📁 1. Repository Structure

```text
HINT/
├── src/                   # perception, task spec, intent injection, FastAPI service
│   ├── task/              # task YAML, subtask progress, hooks
│   ├── perception/        # task manager, grounding, tracking
│   ├── intent/            # visual highlighting and attention prior
│   ├── runtime/           # frame processing and session state
│   ├── inference/         # FastAPI /reset, /step, /health and replay client
│   ├── dataset_export/    # offline LeRobot highlighting / attention export
│   └── common/            # shared types, config loader, utilities
├── pattern/               # pattern router: train, eval, annotation, online predictor
├── policy/pi/             # π₀.₅ training and serving (OpenPI)
├── configs/               # data / inference / pattern / task YAML
│   ├── data/              # offline dataset prep
│   ├── inference/         # online HINT service
│   ├── pattern/           # router train / eval
│   └── tasks/             # task specs (selected by task.name)
├── docs/                  # usage guides
├── scripts/               # launchers
├── SETUP.md               # environment and checkpoints
└── install.sh             # one-shot install
```

---

## ⚙️ 2. Environment Setup

Install and checkpoint download are in [SETUP.md](SETUP.md).

```bash
conda create -n hint python=3.11 -y
conda activate hint
bash install.sh
```

One conda env covers HINT (PyTorch) and π₀.₅ (JAX). Do not load both models in
the same Python process. For another CUDA version, change the wheel index in
`install.sh` first. In an existing env: `python -m pip install --no-deps -e .`.

Download GroundingDINO, SAM2, and Qwen3-VL as in
[SETUP.md § Checkpoints](SETUP.md#checkpoints). Paste **absolute paths** into
the YAML you run; HINT does not expand `$HOME` or `$CKPT_ROOT`.

---

## 🧩 3. Configuration

Copy an example YAML and override paths. Nested maps merge; lists replace.
Details live in the docs — do not edit the examples in place.


| What                 | YAML                 | Guide                                                                 |
| -------------------- | -------------------- | --------------------------------------------------------------------- |
| Task spec            | `configs/tasks/`     | [Add a task](docs/task_configuration.md)                              |
| Offline dataset prep | `configs/data/`      | [Data processing](docs/data_and_inference.md#offline-data-processing) |
| Online HINT service  | `configs/inference/` | [Online inference](docs/data_and_inference.md#online-inference)       |
| Pattern router       | `configs/pattern/`   | [Router training](docs/pattern_training.md)                           |
| π₀.₅                 | `policy/pi` configs  | [π₀.₅ training and serving](docs/pi_policy.md)                        |


Dataset files inherit `configs/data/_base.yaml`; online files inherit
`configs/inference/_base.yaml`. Task YAML is selected by `task.name`.
Reference configs per task are listed in
[data_and_inference.md](docs/data_and_inference.md#reference-configs).

---

## 🚀 4. Training

### 4.1 Training data

**Pattern router.** LeRobot episodes with aligned videos and parquet
`stage_id_gt` (and `state` / `effort`). Annotate stages, then train. See
[pattern training — prepare the data](docs/pattern_training.md#prepare-the-data).

**π₀.₅.** Export highlighted videos and/or attention maps from the annotated
dataset, then train the policy on that LeRobot output. Start with one episode
to check highlighting before a full export:

```bash
python -m dataset_export.preprocessing.prepare_dataset \
  --config configs/data/reasoning_agent_letter.yaml \
  --src-root /path/to/annotated_dataset \
  --dst-root /path/to/rendered_dataset \
  --gpu-ids 0 --num-workers 1 --limit 1
```

Modes, columns, and peg-in-hole prompts:
[offline data processing](docs/data_and_inference.md#offline-data-processing).
Policy field mapping (`semantic_grounding_keys`): [π₀.₅ data](docs/pi_policy.md).

### 4.2 Training

**Pattern router** (from the repository root):

```bash
python -m pattern.train --config configs/pattern/train_manipulation_pattern_joint.yaml
```

**π₀.₅** (from `policy/pi/`):

```bash
cd policy/pi
python scripts/compute_norm_stats.py --config-name pi05_piper_spell_HINT
python scripts/train.py pi05_piper_spell_HINT --exp-name=hint_spell --no-wandb-enabled
```

Config names, injection settings, and checkpoint layout:
[π₀.₅ training](docs/pi_policy.md#training).

---

## 🤖 5. Inference

HINT and the action policy are two processes. They talk over HTTP (HINT is a
FastAPI service). Default ports: HINT **8000**, policy **8001**.
`--hint-url` must match the HINT URL. Start HINT first, then the policy.
If you change either port, update `--hint-url` and any replay `--server`.

**HINT service** (`/reset`, `/step`, `/health`), from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 \
REASONING_AGENT_CONFIG=configs/inference/reasoning_agent_letter_predict.yaml \
python -m inference.api_server
```

**π₀.₅** (from `policy/pi/`):

```bash
cd policy/pi
CUDA_VISIBLE_DEVICES=0 python agent_policy/server_agent_piper_stage.py \
    --host 127.0.0.1 --port 8001 \
    --config pi05_piper_spell_HINT \
    --checkpoint-dir ./checkpoints/pi05_piper_spell_HINT/hint_spell/5000 \
    --hint-url http://127.0.0.1:8000 \
    --agent-prompt "pick the letter"
```

Request schemas and serving details:
[online inference](docs/data_and_inference.md#online-inference) and
[π₀.₅ inference](docs/pi_policy.md#inference).

---

## 🙏 Acknowledgement

We thank the following projects for their open-source contributions:
[OpenPI (π₀ / π₀.₅)](https://github.com/Physical-Intelligence/openpi),
[SAM 2](https://github.com/facebookresearch/sam2),
[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL),
and [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO).

## 📄 License

This repository is released under the [Apache License 2.0](LICENSE).
Third-party components keep their original licenses (OpenPI, SAM 2, Grounding DINO, Qwen).

---

## 📚 Citation

```bibtex
@misc{mei2026hinthumanintentinceptionlonghorizon,
  title={HINT: Human-Intent Inception for Long-Horizon Robot Manipulation},
  author={Mingyu Mei and Haojie Xu and Shihao Jin and Zibo Dai and Qihao Cheng and Zhengrui Lv and Hongjie Fang and Shirun Tang and Guang Chen and Xinyue Zhao and Huiliang Shen and Zaixing He},
  year={2026},
  eprint={2609.02653},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2609.02653},
}
```

