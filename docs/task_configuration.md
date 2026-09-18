# Add a task

A task YAML describes the scene, the pick/place sequence, and recognition prompts.
Model checkpoints and runtime options belong in
[data and inference configs](data_and_inference.md).

Run from the repository root after [installation](../SETUP.md).

## How to configure

Copy [`configs/tasks/_template.yaml`](../configs/tasks/_template.yaml) to
`configs/tasks/my_task.yaml`. Fill it in this order:

1. `name` (must match the filename) and `instruction`.
2. `scene`: unique `key`, `role` (`movable` / `placement` / `landmark`), and a
   concrete `ground_prompt`.
3. `context`: how the episode plan is built (`detected`, `word_chars`, or `fixed`).
4. `recognition` / `classify`: keep category names aligned with `context.place_by`.

The filename and `name` should agree. Files beginning with `_` are not discovered.
`extends: fruit_vegetable` inherits that YAML; nested maps merge, lists such as
`scene` replace the inherited list.

Use the supplied tasks as references. Field-level options (prompts, parsers,
`optional`, `boxes: all`, `dino_prompt`, `place_by`, `steps`) are documented
there:

| Pattern | Example | `context.source` |
| --- | --- | --- |
| Detected objects → destinations | [fruit_vegetable.yaml](../configs/tasks/fruit_vegetable.yaml) | `detected` |
| Word / character sequence | [letter.yaml](../configs/tasks/letter.yaml) | `word_chars` |
| Fixed steps, episode-specific objects | [peg_in_hole.yaml](../configs/tasks/peg_in_hole.yaml) | `fixed` |

[fruit_vegetable_bowl.yaml](../configs/tasks/fruit_vegetable_bowl.yaml) is a
variant of sorting: keep a separate file when grounding differs (colored baskets
vs identical bowls). Do not fold those scenes into one task.

`classify.*.labels` match instruction text after `normalize_words`, so
`left_bowl` equals `left bowl`. Aliases are only for true synonyms. Placement
labels come from placement scene keys or `scene[].as`.

## Highlighting

Omit `scene[].render` for a SAM2 object mask (baskets). `render: area` fills a
rectangle; use it for a geometric slot, not a container that needs a mask.
`tracking.known_bbox_seeds` reuses reset boxes only for objects that stay put.

## Validate and run

Set `task.name` in a data or inference YAML. For a detected-object task:

```yaml
extends: reasoning_agent_classify
task:
  name: my_task
```

Run [offline export](data_and_inference.md#offline-data-processing) on a few
episodes (`--limit 1` or `--episode-ids`). Inspect the highlighted target and
destination: if that preview looks right, the task spec, grounding, and
highlighting path are configured correctly. Process the rest of the dataset
only after that check. Restart a running service after adding or editing task
files.

Add `src/task/task_hooks/<name>.py` with a `HOOKS` dict and `hooks: <name>` only
when YAML cannot express the behavior. See
[letter hooks](../src/task/task_hooks/letter.py) and
[task/hooks.py](../src/task/hooks.py).
