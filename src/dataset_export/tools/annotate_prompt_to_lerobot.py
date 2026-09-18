"""Write annotated episode instructions into a trainable LeRobot dataset.

The instruction produced by ``annotate_peg_in_hole_prompts.py`` is written to
``tasks.jsonl`` and ``episodes.jsonl``.  Its task index is also written to every
action row in the corresponding episode parquet file.

Call from other code::

    writer = PromptToLeRobotWriter(dataset_root, annotations_path)
    plan = writer.write()

Or from the command line::

    python -m dataset_export.tools.annotate_prompt_to_lerobot \
      --dataset-root /dataset/robot/real_world/piper/lerobot/peg_in_hole/peg_in_hole_v4_prompt
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = Path(
    "/dataset/robot/real_world/piper/lerobot/peg_in_hole/peg_in_hole_v4_prompt"
)
DEFAULT_ANNOTATIONS_RELPATH = Path("meta") / "episode_prompts.jsonl"


class ConversionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConversionPlan:
    episode_rows: list[dict[str, Any]]
    task_rows: list[dict[str, Any]]
    task_index_by_episode: dict[int, int]
    parquet_by_episode: dict[int, Path]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(text, encoding="utf-8")


class PromptToLeRobotWriter:
    """Copy sidecar episode instructions into LeRobot metadata and parquet."""

    def __init__(
        self,
        dataset_root: str | Path,
        annotations_path: str | Path | None = None,
        *,
        instruction_field: str = "instruction",
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.annotations_path = (
            Path(annotations_path)
            if annotations_path is not None
            else self.dataset_root / DEFAULT_ANNOTATIONS_RELPATH
        )
        self.instruction_field = instruction_field

    def build_plan(self) -> ConversionPlan:
        episodes_path = self.dataset_root / "meta" / "episodes.jsonl"
        episodes = {
            int(row["episode_index"]): row for row in _read_jsonl(episodes_path)
        }
        annotations = {
            int(row["episode_index"]): str(row[self.instruction_field]).strip()
            for row in _read_jsonl(self.annotations_path)
        }
        if episodes.keys() != annotations.keys():
            raise ConversionError(
                "episode_prompts.jsonl must cover the dataset exactly"
            )

        task_index_by_instruction: dict[str, int] = {}
        task_index_by_episode: dict[int, int] = {}
        episode_rows: list[dict[str, Any]] = []

        for episode_index in sorted(episodes):
            instruction = annotations[episode_index]
            task_index = task_index_by_instruction.setdefault(
                instruction, len(task_index_by_instruction)
            )
            task_index_by_episode[episode_index] = task_index
            episode_row = dict(episodes[episode_index])
            episode_row["tasks"] = [instruction]
            episode_rows.append(episode_row)

        task_rows = [
            {"task_index": task_index, "task": instruction}
            for instruction, task_index in task_index_by_instruction.items()
        ]
        parquet_by_episode = {
            int(path.stem.removeprefix("episode_")): path
            for path in self.dataset_root.glob("data/chunk-*/episode_*.parquet")
        }
        return ConversionPlan(
            episode_rows,
            task_rows,
            task_index_by_episode,
            parquet_by_episode,
        )

    def apply_plan(self, plan: ConversionPlan) -> None:
        for episode_index, task_index in plan.task_index_by_episode.items():
            self._update_episode_parquet(
                plan.parquet_by_episode[episode_index], task_index
            )

        meta = self.dataset_root / "meta"
        _write_jsonl(meta / "episodes.jsonl", plan.episode_rows)
        _write_jsonl(meta / "tasks.jsonl", plan.task_rows)

        info_path = meta / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["total_tasks"] = len(plan.task_rows)
        info_path.write_text(
            json.dumps(info, ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
        )

    def write(self, *, dry_run: bool = False) -> ConversionPlan:
        """Build the conversion plan and optionally apply it.

        Raises:
            ConversionError: sidecar does not cover every episode.
        """
        plan = self.build_plan()
        if not dry_run:
            self.apply_plan(plan)
        return plan

    def try_write(self, *, dry_run: bool = False) -> ConversionPlan | None:
        """Like :meth:`write`, but return ``None`` when coverage is incomplete."""
        try:
            return self.write(dry_run=dry_run)
        except ConversionError:
            return None

    def format_result(self, plan: ConversionPlan, *, mode: str) -> str:
        header = (
            f"[ok] {mode}: episodes={len(plan.episode_rows)}, "
            f"tasks={len(plan.task_rows)}"
        )
        lines = [header]
        lines.extend(
            f"  {row['task_index']}: {row['task']}" for row in plan.task_rows
        )
        return "\n".join(lines)

    @staticmethod
    def _update_episode_parquet(path: Path, task_index: int) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        column_index = table.schema.get_field_index("task_index")
        field = table.schema.field(column_index)
        task_indices = pa.array([task_index] * table.num_rows, type=field.type)
        table = table.set_column(column_index, field, task_indices)

        temp_path = path.with_suffix(".parquet.tmp")
        pq.write_table(table, temp_path, compression="snappy")
        temp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument(
        "--annotations",
        default=None,
        help=f"Default: <dataset-root>/{DEFAULT_ANNOTATIONS_RELPATH.as_posix()}",
    )
    parser.add_argument("--instruction-field", default="instruction")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    writer = PromptToLeRobotWriter(
        args.dataset_root,
        args.annotations,
        instruction_field=args.instruction_field,
    )
    plan = writer.write(dry_run=args.dry_run)
    mode = "dry-run" if args.dry_run else "updated"
    print(writer.format_result(plan, mode=mode))


if __name__ == "__main__":
    main()
