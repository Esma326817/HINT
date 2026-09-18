"""First-pass prompt annotation for the peg-in-hole color/shape dataset.

This script writes one JSONL sidecar row per episode, containing the block
color, the selected white peg shape, and a stable instruction string. It also
saves a labeled finish-frame image under ``check_images/`` for human review.
The second render pass can read this sidecar instead of asking the VLM to
understand the task again.

If ``dataset.task_context.write_to_lerobot`` is true, and the sidecar covers
every episode, the same instructions are copied into ``tasks.jsonl``,
``episodes.jsonl``, and parquet ``task_index``.

I/O paths and the color/shape vocabulary come from the data config and
``configs/tasks/<task>.yaml``. The finish-frame / empty-hole visual pipeline
stays peg-specific.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.config_loader import PROJECT_ROOT, load_reasoning_config
from task.base import normalize_words
from task.spec import ClassifySpec, load_task_spec

DEFAULT_DATASET_ROOT = Path(
    "/dataset/robot/real_world/piper/lerobot/peg_in_hole/peg_in_hole_v4_5_merged"
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data" / "reasoning_agent_peg_in_hole.yaml"
DEFAULT_PEG_COLOR = "white"
PLACED_BLOCK_BOX_LIMITS = {
    "max_area_ratio": 0.14,
    "max_width_ratio": 0.40,
    "max_height_ratio": 0.60,
}


@dataclass(frozen=True)
class PegPromptSettings:
    """Config-driven I/O and label sets for peg prompt annotation."""

    task_name: str
    color_spec: ClassifySpec
    shape_spec: ClassifySpec
    plan_fields: tuple[str, ...]
    instruction_field: str
    output_fields: tuple[str, ...]
    output_relpath: str
    write_to_lerobot: bool
    stage_column: str
    global_camera: str
    default_peg_color: str = DEFAULT_PEG_COLOR

    @property
    def colors(self) -> tuple[str, ...]:
        return self.color_spec.labels

    @property
    def shapes(self) -> tuple[str, ...]:
        return self.shape_spec.labels

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> PegPromptSettings:
        task_name = str(config["task"]["name"])
        spec = load_task_spec(task_name)
        task_context = config["dataset"]["task_context"]
        cameras = config.get("cameras") or {}
        stages = config.get("stages") or {}
        output_fields = tuple(str(item) for item in task_context.get("fields") or ())
        return cls(
            task_name=task_name,
            color_spec=spec.classify_spec("block_color"),
            shape_spec=spec.classify_spec("peg_shape"),
            plan_fields=tuple(spec.context_fields),
            instruction_field=str(task_context.get("instruction_field") or "instruction"),
            output_fields=output_fields,
            output_relpath=str(task_context.get("path") or "meta/episode_prompts.jsonl"),
            write_to_lerobot=bool(task_context.get("write_to_lerobot", False)),
            stage_column=str(stages.get("column") or "stage_id_gt"),
            global_camera=str(cameras.get("global") or "images.global"),
        )


def block_color_prompt(colors: Sequence[str]) -> str:
    listed = ", ".join(colors)
    choices = "|".join(colors)
    return f"""You are looking at the final frame of a robot peg-in-hole task.

In the center of the image there is a black rectangular slot/fixture. A colored
rectangular block has been placed inside that center black slot.

What color is the colored rectangular block in the center black slot?

Rules:
- Answer only one of: {listed}.
- Ignore the unused colored blocks on the left side of the image.
- Ignore the white pegs on the right side of the image.
- Ignore the robot gripper, black slot, metal table, and table holes.
- Output exactly one JSON object: {{"selected_block_color": "<{choices}>"}}
"""


def display_shape(shape: str) -> str:
    return "L-shaped" if shape == "l shaped" else shape


def final_empty_holes_prompt(block_color: str, shapes: Sequence[str]) -> str:
    listed = ", ".join(shapes)
    named = [display_shape(shape) for shape in shapes]
    if not named:
        shape_names = "shaped holes"
    elif len(named) == 1:
        shape_names = f"one {named[0]} hole"
    elif len(named) == 2:
        shape_names = f"one {named[0]} hole and one {named[1]} hole"
    else:
        shape_names = (
            "one "
            + " hole, one ".join(named[:-1])
            + f" hole, and one {named[-1]} hole"
        )
    count = len(shapes)
    empty_count = max(count - 1, 1)
    return f"""You are annotating a colored rectangular block after a peg insertion.

The image is a crop from the final global-camera frame. It contains the selected
{block_color} rectangular block placed in the center black slot. A white peg is
inserted into one of the {count} holes.

The block has exactly {count} hole shapes: {shape_names}. The inserted white peg
covers one of those holes. The other {empty_count} holes are still visible and empty.

Directly identify the shapes of the two visible empty holes on the selected
{block_color} block. Do not classify the inserted white peg itself.

Rules:
- Use only these shape names: {listed}.
- Output exactly two empty-hole shapes.
- Ignore the metal table holes, black slot, robot gripper, right-side peg holder,
  and objects outside the selected {block_color} block.
- Output exactly one JSON object:
{{
  "empty_hole_shapes": ["shape1", "shape2"]
}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate peg-in-hole episode prompts from start/finish global frames.",
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--out",
        default=None,
        help="Default: <dataset-root>/<dataset.task_context.path> from the config.",
    )
    parser.add_argument(
        "--stage-column",
        default=None,
        help="Default: stages.column from the config (stage_id_gt).",
    )
    parser.add_argument("--episodes", default=None, help="Comma/space separated episode ids, e.g. 0,3,12")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-keyframes", type=int, default=12, help="Compatibility option; ignored by this two-frame annotator")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-sheets", action="store_true", help="Save contact sheets under meta/peg_prompt_sheets")
    parser.add_argument(
        "--check-images-dir",
        default=None,
        help="Labeled finish frames. Default: <dataset-root>/check_images. Pass empty to skip.",
    )
    parser.add_argument("--print-vlm-outputs", action="store_true", help="Print raw VLM outputs and postprocessed fields")
    parser.add_argument(
        "--write-to-lerobot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Copy sidecar instructions into LeRobot tasks/episodes/parquet. "
            "Default: dataset.task_context.write_to_lerobot."
        ),
    )
    return parser.parse_args()


def normalize_episode_id(token: str) -> int:
    token = token.strip()
    if token.startswith("episode_"):
        token = token[len("episode_") :]
    if token.endswith(".parquet") or token.endswith(".mp4"):
        token = Path(token).stem.replace("episode_", "")
    return int(token)


def selected_episodes(dataset_root: Path, raw: str | None, limit: int | None) -> list[int]:
    if raw:
        episodes = [normalize_episode_id(token) for token in raw.replace(",", " ").split() if token.strip()]
    else:
        episodes_path = dataset_root / "meta" / "episodes.jsonl"
        episodes = []
        with episodes_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                episodes.append(int(json.loads(line)["episode_index"]))
    episodes = sorted(dict.fromkeys(episodes))
    return episodes[:limit] if limit is not None else episodes


def read_existing(output_path: Path) -> set[int]:
    if not output_path.is_file():
        return set()
    seen: set[int] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                seen.add(int(json.loads(line)["episode_index"]))
            except Exception:
                continue
    return seen


def parquet_path(dataset_root: Path, episode_index: int) -> Path:
    return dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"


def video_path(dataset_root: Path, camera_key: str, episode_index: int) -> Path:
    return dataset_root / "videos" / "chunk-000" / camera_key / f"episode_{episode_index:06d}.mp4"


def read_stage_ids(path: Path, stage_column: str) -> list[int]:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas/pyarrow are required to read LeRobot parquet files") from exc

    frame = pd.read_parquet(path, columns=[stage_column])
    if stage_column not in frame.columns:
        available = ", ".join(str(column) for column in pd.read_parquet(path).columns)
        raise ValueError(f"missing stage column {stage_column!r} in {path}; available: {available}")
    return [int(value) for value in frame[stage_column].tolist()]


def stage_segments(stage_ids: list[int]) -> list[tuple[int, int, int]]:
    if not stage_ids:
        return []
    segments: list[tuple[int, int, int]] = []
    start = 0
    current = int(stage_ids[0])
    for idx, value in enumerate(stage_ids[1:], start=1):
        value = int(value)
        if value == current:
            continue
        segments.append((current, start, idx - 1))
        current = value
        start = idx
    segments.append((current, start, len(stage_ids) - 1))
    return segments


def choose_stage_transition_frame(stage_ids: list[int], from_stage: int = 3, to_stage: int = 6) -> int:
    """Return the last frame before a from_stage -> to_stage transition."""
    if not stage_ids:
        return 0

    transitions = [
        idx - 1
        for idx, value in enumerate(stage_ids[1:], start=1)
        if int(stage_ids[idx - 1]) == from_stage and int(value) == to_stage
    ]
    if transitions:
        return transitions[-1]

    segments = stage_segments(stage_ids)
    fallback = [end for stage, _start, end in segments if int(stage) == from_stage]
    if fallback:
        return fallback[-1]
    return len(stage_ids) - 1


def choose_keyframes(
    stage_ids: list[int],
    max_keyframes: int,
    camera: str = "images.global",
) -> list[dict[str, int | str]]:
    """Use only unobstructed global views for task-level prompt annotation."""
    del max_keyframes
    if not stage_ids:
        return [{"stage_id": 1, "frame_index": 0, "camera": camera, "role": "start"}]
    last_frame = len(stage_ids) - 1
    keyframes: list[dict[str, int | str]] = [
        {
            "stage_id": int(stage_ids[0]),
            "frame_index": 0,
            "camera": camera,
            "role": "start",
        }
    ]
    if last_frame > 0:
        keyframes.append(
            {
                "stage_id": int(stage_ids[-1]),
                "frame_index": last_frame,
                "camera": camera,
                "role": "finish",
            }
        )
    return keyframes


def extract_frame_png(video: Path, frame_index: int):
    from PIL import Image

    vf = f"select=eq(n\\,{int(frame_index)})"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        vf,
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    completed = subprocess.run(command, check=True, capture_output=True)
    if not completed.stdout:
        raise RuntimeError(f"ffmpeg produced no frame for {video} frame={frame_index}")
    return Image.open(io.BytesIO(completed.stdout)).convert("RGB")


def make_image_sheet(cells: list[tuple[str, Any]]):
    from PIL import Image, ImageDraw, ImageFont

    if not cells:
        raise ValueError("cannot build an empty contact sheet")

    thumb_w, thumb_h = 320, 240
    label_h = 22
    cols = min(4, max(1, len(cells)))
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for idx, (label, image) in enumerate(cells):
        row, col = divmod(idx, cols)
        x = col * thumb_w
        y = row * (thumb_h + label_h)
        sheet.paste(image.resize((thumb_w, thumb_h)), (x, y + label_h))
        draw.text((x + 4, y + 4), label, fill=(0, 0, 0), font=font)
    return sheet


def make_contact_sheet(dataset_root: Path, episode_index: int, keyframes: list[dict[str, Any]]):
    cells = []
    for item in keyframes:
        camera = str(item["camera"])
        frame_index = int(item["frame_index"])
        stage_id = int(item["stage_id"])
        image = extract_frame_png(video_path(dataset_root, camera, episode_index), frame_index)
        role = str(item.get("role", "keyframe"))
        cells.append((f"ep{episode_index:06d} {role} f{frame_index} s{stage_id} {camera}", image))
    return make_image_sheet(cells)


def coerce_label(spec: ClassifySpec, value: Any) -> str:
    """Map raw VLM text onto one classify label from the task YAML."""
    normalized = normalize_words(value)
    if normalized in spec.labels:
        return normalized
    mapped = spec.normalize(value)
    if mapped != spec.unknown_label:
        return mapped
    return spec.find_in_text(value)


def force_label(spec: ClassifySpec, value: Any, fallback: Any | None = None) -> str:
    label = coerce_label(spec, value)
    if label != spec.unknown_label:
        return label
    if fallback is not None:
        fallback_label = coerce_label(spec, fallback)
        if fallback_label != spec.unknown_label:
            return fallback_label
    if spec.labels:
        return spec.labels[0]
    return spec.unknown_label


def extract_labels(spec: ClassifySpec, value: Any) -> list[str]:
    if isinstance(value, list):
        labels: list[str] = []
        for item in value:
            label = coerce_label(spec, item)
            if label != spec.unknown_label and label not in labels:
                labels.append(label)
        return labels

    text = normalize_words(value)
    found: list[str] = []
    alias_items = sorted(spec.aliases.items(), key=lambda item: len(item[0]), reverse=True)
    for alias, label in alias_items:
        if label in found:
            continue
        if re.search(rf"\b{re.escape(alias)}\b", text):
            found.append(label)
    for label in spec.labels:
        if label not in found and re.search(rf"\b{re.escape(label)}\b", text):
            found.append(label)
    return found


def force_shape_permutation(shapes: Sequence[str], allowed: Sequence[str]) -> list[str]:
    """Return a permutation that uses each allowed shape once."""
    allowed_list = list(allowed)
    result: list[str | None] = []
    used: set[str] = set()
    for shape in list(shapes)[: len(allowed_list)]:
        if shape in allowed_list and shape not in used:
            result.append(shape)
            used.add(shape)
        else:
            result.append(None)

    missing = [shape for shape in allowed_list if shape not in used]
    for idx, shape in enumerate(result):
        if shape is None:
            result[idx] = missing.pop(0)
    while len(result) < len(allowed_list):
        result.append(missing.pop(0))
    return [str(shape) for shape in result[: len(allowed_list)]]


def infer_removed_shape(
    payload: dict[str, Any],
    raw_text: str,
    spec: ClassifySpec,
) -> tuple[str, list[str]]:
    selected = coerce_label(
        spec,
        payload.get("selected_peg_shape")
        or payload.get("removed_peg_shape")
        or payload.get("missing_peg_shape"),
    )
    if selected != spec.unknown_label:
        return selected, []

    remaining = extract_labels(
        spec,
        payload.get("empty_hole_shapes")
        or payload.get("visible_empty_hole_shapes")
        or payload.get("remaining_hole_shapes")
        or payload.get("remaining_peg_shapes")
        or raw_text,
    )
    unique_remaining: list[str] = []
    for shape in remaining:
        if shape in spec.labels and shape not in unique_remaining:
            unique_remaining.append(shape)

    missing = [shape for shape in spec.labels if shape not in unique_remaining]
    expected_empty = max(len(spec.labels) - 1, 0)
    if len(unique_remaining) == expected_empty and len(missing) == 1:
        return missing[0], unique_remaining
    return force_label(spec, selected), unique_remaining


def enforce_middle_hole(holes: list[str], middle_hole: str, spec: ClassifySpec) -> list[str]:
    """Keep a legal left/middle/right permutation and apply the color prior."""
    middle_hole = force_label(spec, middle_hole)
    normalized = force_shape_permutation(holes, spec.labels)
    result: list[str | None] = [None, middle_hole, None]
    used = {middle_hole}

    for idx in (0, 2):
        shape = normalized[idx]
        if shape in spec.labels and shape not in used:
            result[idx] = shape
            used.add(shape)

    missing = [shape for shape in spec.labels if shape not in used]
    for idx in (0, 2):
        if result[idx] is None:
            result[idx] = missing.pop(0)
    return [str(shape) for shape in result]


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"VLM output did not contain a JSON object: {text!r}")
    return json.loads(text[start : end + 1])


def parse_json_or_empty(text: str) -> dict[str, Any]:
    try:
        return extract_json_object(text)
    except Exception:
        return {}


def locate_placed_block_bbox(image: Any, block_color: str) -> tuple[int, int, int, int] | None:
    """Find the seated block with the same Qwen box prompts as data processing."""
    from perception.semantic_grounder.qwen_crop import normalize_compact_box
    from task.base import qwen_box

    parent_box = None
    if block_color not in ("", "unknown"):
        parent_box = qwen_box(
            image,
            (
                f"the {block_color} rectangular block containing three dark "
                "shaped holes, seated in a black destination base"
            ),
        )
    if parent_box is None:
        parent_box = qwen_box(
            image,
            (
                "the one colored rectangular block seated inside the black "
                "rectangular destination base, containing three dark shaped holes; "
                "exclude all unused colored blocks and the black peg holder"
            ),
        )
    parent_box = normalize_compact_box(parent_box, image.size, **PLACED_BLOCK_BOX_LIMITS)
    if parent_box is None:
        return None
    x0, y0, x1, y1 = parent_box
    return int(x0), int(y0), int(x1), int(y1)


def expand_bbox(bbox: tuple[int, int, int, int], image_size: tuple[int, int], padding: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    width, height = image_size
    return max(0, x0 - padding), max(0, y0 - padding), min(width, x1 + padding), min(height, y1 + padding)


def crop_or_full(image: Any, bbox: tuple[int, int, int, int] | None, padding: int = 18):
    if bbox is None:
        return image.copy(), None
    expanded = expand_bbox(bbox, image.size, padding)
    return image.crop(expanded), expanded


def center_crop(image: Any):
    width, height = image.size
    bbox = (width // 4, height // 5, (width * 3) // 4, (height * 4) // 5)
    return image.crop(bbox), bbox


def enlarge_for_vlm(image: Any, min_width: int = 640):
    width, height = image.size
    if width >= min_width:
        return image
    scale = max(1, int(round(min_width / max(1, width))))
    return image.resize((width * scale, height * scale))


def instruction_from_parts(
    block_color: str,
    peg_shape: str,
    *,
    peg_color: str = DEFAULT_PEG_COLOR,
    block_position: str = "unknown",
) -> str:
    from task.task_hooks.peg_in_hole import _plan_context

    return _plan_context(
        {
            "block_color": block_color,
            "block_position": block_position,
            "peg_color": peg_color,
            "peg_shape": peg_shape,
        }
    )["instruction"]


def _check_image_font(size: int = 28):
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def save_check_image(
    image: Any,
    *,
    block_color: str,
    peg_shape: str,
    output_path: Path,
) -> None:
    """Save the finish frame with color/shape labels for human review."""
    from PIL import ImageDraw

    labeled = image.copy().convert("RGB")
    draw = ImageDraw.Draw(labeled)
    font = _check_image_font()
    lines = [f"color: {block_color}", f"shape: {peg_shape}"]

    margin = 12
    line_gap = 6
    text_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    text_w = max(box[2] - box[0] for box in text_boxes)
    text_h = sum(box[3] - box[1] for box in text_boxes) + line_gap * (len(lines) - 1)
    pad = 8
    box = (
        margin - pad,
        margin - pad,
        margin + text_w + pad,
        margin + text_h + pad,
    )
    draw.rectangle(box, fill=(0, 0, 0))
    y = margin
    for line, text_box in zip(lines, text_boxes):
        draw.text((margin, y), line, fill=(255, 255, 255), font=font)
        y += (text_box[3] - text_box[1]) + line_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.save(output_path, quality=95)


def print_vlm_debug(row: dict[str, Any]) -> None:
    print(f"[debug] {row['episode_name']}", flush=True)
    print(f"  selected_block_color: {row['selected_block_color']}", flush=True)
    print(f"  selected_peg_shape: {row['selected_peg_shape']}", flush=True)
    print(f"  instruction: {row['instruction']}", flush=True)

    crops = row.get("crops") or {}
    print(f"  finish_block_bbox_xyxy: {crops.get('finish_block_bbox_xyxy')}", flush=True)
    print(f"  empty_hole_shapes: {crops.get('empty_hole_shapes')}", flush=True)

    raw_outputs = row.get("raw_vlm_outputs") or {}
    print("  raw_vlm_outputs:", flush=True)
    for key, value in raw_outputs.items():
        print(f"    {key}: {value!r}", flush=True)


def load_lerobot_ground_truth(dataset_root: Path) -> dict[int, dict[str, str]]:
    """Parse color/shape/instruction labels from ``meta/episodes.jsonl`` tasks."""
    from task.task_hooks.peg_in_hole import _parse_plan

    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        return {}
    truth: dict[int, dict[str, str]] = {}
    with episodes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            tasks = row.get("tasks") or []
            if not tasks:
                continue
            instruction = str(tasks[0])
            plan = _parse_plan(instruction)
            truth[int(row["episode_index"])] = {
                "instruction": instruction,
                "selected_block_color": plan["block_color"],
                "selected_peg_shape": plan["peg_shape"],
                "selected_peg_color": plan["peg_color"],
            }
    return truth


def summarize_accuracy(
    rows: Sequence[Mapping[str, Any]],
    truth: Mapping[int, Mapping[str, str]],
) -> dict[str, Any]:
    keys = ("selected_block_color", "selected_peg_shape", "instruction")
    scored = [row for row in rows if int(row["episode_index"]) in truth]
    totals = {key: 0 for key in keys}
    totals["color_and_shape"] = 0
    mismatches: list[dict[str, Any]] = []
    for row in scored:
        episode_index = int(row["episode_index"])
        expected = truth[episode_index]
        color_ok = str(row.get("selected_block_color")) == expected["selected_block_color"]
        shape_ok = str(row.get("selected_peg_shape")) == expected["selected_peg_shape"]
        instruction_ok = str(row.get("instruction")) == expected["instruction"]
        totals["selected_block_color"] += int(color_ok)
        totals["selected_peg_shape"] += int(shape_ok)
        totals["instruction"] += int(instruction_ok)
        totals["color_and_shape"] += int(color_ok and shape_ok)
        if not (color_ok and shape_ok):
            mismatches.append(
                {
                    "episode_index": episode_index,
                    "pred_color": row.get("selected_block_color"),
                    "gt_color": expected["selected_block_color"],
                    "pred_shape": row.get("selected_peg_shape"),
                    "gt_shape": expected["selected_peg_shape"],
                }
            )
    n = len(scored)
    return {"n": n, "correct": totals, "mismatches": mismatches}


def print_accuracy(summary: Mapping[str, Any]) -> None:
    n = int(summary["n"])
    if n <= 0:
        print("[eval] no LeRobot episode tasks to compare against", flush=True)
        return
    correct = summary["correct"]
    print(f"[eval] compared {n} episodes against meta/episodes.jsonl", flush=True)
    for key in ("selected_block_color", "selected_peg_shape", "color_and_shape", "instruction"):
        value = int(correct[key])
        print(f"[eval] {key}: {value}/{n} ({100.0 * value / n:.1f}%)", flush=True)
    mismatches = list(summary["mismatches"])[:12]
    for item in mismatches:
        print(
            f"[eval] mismatch episode_{item['episode_index']:06d}: "
            f"color {item['pred_color']} vs {item['gt_color']}, "
            f"shape {item['pred_shape']} vs {item['gt_shape']}",
            flush=True,
        )


def annotate_episode(
    *,
    dataset_root: Path,
    episode_index: int,
    settings: PegPromptSettings,
    stage_column: str,
    max_keyframes: int,
    max_new_tokens: int,
    save_sheet_dir: Path | None,
    check_images_dir: Path | None,
) -> dict[str, Any]:
    stage_ids = read_stage_ids(parquet_path(dataset_root, episode_index), stage_column)
    keyframes = choose_keyframes(
        stage_ids, max_keyframes=max_keyframes, camera=settings.global_camera
    )
    start_frame = int(keyframes[0]["frame_index"])
    finish_frame = int(keyframes[-1]["frame_index"])
    start_image = extract_frame_png(
        video_path(dataset_root, settings.global_camera, episode_index), start_frame
    )
    finish_image = extract_frame_png(
        video_path(dataset_root, settings.global_camera, episode_index), finish_frame
    )

    from perception.task_manager.qwen_prompt import generate_text
    from perception.task_manager.qwen_runtime import get_qwen_runtime

    runtime = get_qwen_runtime()

    raw_block_color = generate_text(
        finish_image,
        block_color_prompt(settings.colors),
        runtime,
        max_new_tokens=max_new_tokens,
    )
    block_payload = parse_json_or_empty(raw_block_color)
    block_color = coerce_label(
        settings.color_spec, block_payload.get("selected_block_color") or raw_block_color
    )

    finish_block_bbox = locate_placed_block_bbox(finish_image, block_color)
    if finish_block_bbox is None:
        finish_block_crop, finish_block_crop_bbox = center_crop(finish_image)
    else:
        finish_block_crop, finish_block_crop_bbox = crop_or_full(
            finish_image, finish_block_bbox, padding=28
        )

    raw_final_empty_holes = generate_text(
        enlarge_for_vlm(finish_block_crop),
        final_empty_holes_prompt(block_color, settings.shapes),
        runtime,
        max_new_tokens=max_new_tokens,
    )
    final_empty_holes_payload = parse_json_or_empty(raw_final_empty_holes)
    peg_shape, empty_hole_shapes = infer_removed_shape(
        final_empty_holes_payload, raw_final_empty_holes, settings.shape_spec
    )
    peg_color = settings.default_peg_color
    instruction = instruction_from_parts(block_color, peg_shape, peg_color=peg_color)

    debug_sheet = make_image_sheet(
        [
            (f"ep{episode_index:06d} start global", start_image),
            (f"ep{episode_index:06d} finish global", finish_image),
            (f"finish {block_color} block+peg crop", finish_block_crop),
        ]
    )
    if save_sheet_dir is not None:
        save_sheet_dir.mkdir(parents=True, exist_ok=True)
        debug_sheet.save(save_sheet_dir / f"episode_{episode_index:06d}.jpg", quality=95)

    if check_images_dir is not None:
        save_check_image(
            finish_image,
            block_color=block_color,
            peg_shape=peg_shape,
            output_path=check_images_dir / f"episode_{episode_index:06d}.jpg",
        )

    raw_outputs = {
        "block_color": raw_block_color,
        "final_empty_holes": raw_final_empty_holes,
    }
    field_values = {
        "selected_block_color": block_color,
        "selected_peg_shape": peg_shape,
        "selected_peg_color": peg_color,
        "selected_block_position": "unknown",
    }
    row: dict[str, Any] = {
        "episode_index": episode_index,
        "episode_name": f"episode_{episode_index:06d}",
        settings.instruction_field: instruction,
        "raw_vlm_output": json.dumps(raw_outputs, ensure_ascii=False),
        "raw_vlm_outputs": raw_outputs,
        "keyframes": keyframes,
        "crops": {
            "finish_block_bbox_xyxy": finish_block_crop_bbox,
            "vlm_block_color": block_color,
            "empty_hole_shapes": empty_hole_shapes,
        },
    }
    for field_name in (*settings.output_fields, *settings.plan_fields):
        if field_name in field_values:
            row[field_name] = field_values[field_name]
    return row


def maybe_write_to_lerobot(
    dataset_root: Path,
    annotations_path: Path,
    *,
    instruction_field: str,
    enabled: bool,
) -> bool:
    """Copy sidecar instructions into LeRobot metadata when coverage is complete."""
    if not enabled:
        return False

    from dataset_export.tools.annotate_prompt_to_lerobot import PromptToLeRobotWriter

    writer = PromptToLeRobotWriter(
        dataset_root,
        annotations_path,
        instruction_field=instruction_field,
    )
    plan = writer.try_write()
    if plan is None:
        print(
            "[skip] write_to_lerobot: episode_prompts.jsonl must cover the dataset exactly",
            flush=True,
        )
        return False
    print(writer.format_result(plan, mode="write_to_lerobot"), flush=True)
    return True


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    config = load_reasoning_config(args.config)
    settings = PegPromptSettings.from_config(config)
    write_to_lerobot = (
        settings.write_to_lerobot
        if args.write_to_lerobot is None
        else args.write_to_lerobot
    )
    output_path = (
        Path(args.out) if args.out else dataset_root / settings.output_relpath
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage_column = args.stage_column or settings.stage_column

    from perception.task_manager.qwen_runtime import initialize_qwen_runtime_from_config

    initialize_qwen_runtime_from_config(config)

    done = set() if args.overwrite else read_existing(output_path)
    episodes = selected_episodes(dataset_root, args.episodes, args.limit)
    save_sheet_dir = dataset_root / "meta" / "peg_prompt_sheets" if args.save_sheets else None
    if args.check_images_dir is None:
        check_images_dir = dataset_root / "check_images"
    elif str(args.check_images_dir).strip() == "":
        check_images_dir = None
    else:
        check_images_dir = Path(args.check_images_dir)
    truth = load_lerobot_ground_truth(dataset_root)
    annotated_rows: list[dict[str, Any]] = []

    mode = "w" if args.overwrite else "a"
    with output_path.open(mode, encoding="utf-8") as handle:
        for episode_index in episodes:
            if episode_index in done:
                print(f"[skip] episode_{episode_index:06d} already annotated", flush=True)
                continue
            print(f"[annotate] episode_{episode_index:06d}", flush=True)
            row = annotate_episode(
                dataset_root=dataset_root,
                episode_index=episode_index,
                settings=settings,
                stage_column=stage_column,
                max_keyframes=args.max_keyframes,
                max_new_tokens=args.max_new_tokens,
                save_sheet_dir=save_sheet_dir,
                check_images_dir=check_images_dir,
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            annotated_rows.append(row)
            if args.print_vlm_outputs:
                print_vlm_debug(row)
            print(f"[ok] episode_{episode_index:06d}: {row.get(settings.instruction_field)}", flush=True)

    if annotated_rows:
        print_accuracy(summarize_accuracy(annotated_rows, truth))

    maybe_write_to_lerobot(
        dataset_root,
        output_path,
        instruction_field=settings.instruction_field,
        enabled=write_to_lerobot,
    )


if __name__ == "__main__":
    main()
