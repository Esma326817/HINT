"""Subtask memory/scheduler for long-horizon manipulation.

A long-horizon task is recognized once at reset (the target word plus the letter
blocks on the table, the toys to sort, the episode plan) and split into subtasks.
A subtask walks the four action patterns, which split by what the hand holds:

    free_move -> pre_contact -> dexterous_contact -> transport_contact -> free_move
    |------------ free: empty hand -----|------- contact: holding it -------|

This manager is the *memory* deciding which subtask is active as the robot moves
through those patterns, so every VLM/grounding call inside one subtask is
scheduled to the same target. The active subtask is *locked* for the whole
sequence: the phrase produced in ``pre_contact`` is the one chosen in
``free_move``, and leaving the contact patterns for a free one starts the next
subtask.

The cursor lives on ``TaskState.progress_idx`` (the single source of truth that
the grounding prompt already reads), so this manager only schedules transitions;
it does not duplicate task state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pattern.runtime.types import FREE_MOVE_STAGE, StageName
from task.base import ACTION_PATTERNS

if TYPE_CHECKING:
    from common.task_types import TaskState
    from pattern.runtime.types import StageDecision

_logger = logging.getLogger(__name__)

# Visiting a contact pattern means the subtask has truly engaged the object, so
# the next free pattern should advance to the next subtask.
DEFAULT_COMPLETION_STAGES: tuple[StageName, ...] = ACTION_PATTERNS["contact"]


@dataclass(frozen=True)
class Subtask:
    """One unit of a long-horizon task (one letter, one toy, one insertion)."""

    index: int
    label: str
    category: str | None = None
    placement: str | None = None

    @property
    def description(self) -> str:
        suffix = f" -> {self.placement}" if self.placement else ""
        return f"subtask[{self.index}] {self.label.lower()}{suffix}"


class SubtaskManager:
    """Schedule subtasks across action patterns over a single TaskState."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        advance_on_return_to: StageName = FREE_MOVE_STAGE,
        completion_stages: tuple[StageName, ...] = DEFAULT_COMPLETION_STAGES,
        on_advance: Any | None = None,
    ) -> None:
        self.enabled = enabled
        self.advance_on_return_to = advance_on_return_to
        self.completion_stages = tuple(completion_stages)
        self.on_advance = on_advance
        self.reset()

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        on_advance: Any | None = None,
    ) -> "SubtaskManager":
        cfg = config.get("subtask", {}) or {}
        completion = cfg.get("completion_stages")
        if isinstance(completion, str):
            completion = (completion,)
        elif isinstance(completion, (list, tuple)) and completion:
            completion = tuple(str(item) for item in completion)
        else:
            completion = DEFAULT_COMPLETION_STAGES
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            advance_on_return_to=str(cfg.get("advance_on_return_to") or FREE_MOVE_STAGE),
            completion_stages=completion,
            on_advance=on_advance,
        )

    def reset(self) -> None:
        self._prev_stage: StageName | None = None
        self._visited_completion = False
        self._planned = False

    # --- subtask views over the task memory -------------------------------

    @staticmethod
    def subtasks(task_state: TaskState) -> list[Subtask]:
        return [
            Subtask(
                index=i,
                label=label,
                category=task_state.target_categories[i] if i < len(task_state.target_categories) else None,
                placement=task_state.target_placements[i] if i < len(task_state.target_placements) else None,
            )
            for i, label in enumerate(task_state.target_labels)
        ]

    @staticmethod
    def current_subtask(task_state: TaskState | None) -> Subtask | None:
        if task_state is None:
            return None
        idx = task_state.progress_idx
        if not (0 <= idx < len(task_state.target_labels)):
            return None
        return Subtask(
            index=idx,
            label=task_state.target_labels[idx],
            category=task_state.target_categories[idx] if idx < len(task_state.target_categories) else None,
            placement=task_state.target_placements[idx] if idx < len(task_state.target_placements) else None,
        )

    # --- scheduling --------------------------------------------------------

    def observe(self, *, decision: StageDecision, task_state: TaskState | None) -> bool:
        """Update the active subtask from the confirmed pattern. Returns True if advanced."""
        if not self.enabled or task_state is None or not task_state.target_labels:
            return False

        # Ensure the first subtask has a planned target/placement before any move.
        if not self._planned:
            if task_state.target_block_id is None and task_state.target_placement_id is None:
                self._plan(task_state)
            self._planned = True

        stage = decision.confirmed_stage

        if self._prev_stage is None:
            self._prev_stage = stage
            if stage in self.completion_stages:
                self._visited_completion = True
            return False

        if stage == self._prev_stage:
            return False

        # Advance when the subtask *leaves* a contact pattern for a free one. The
        # rule is stated as a return to free_move, but GT annotations do not always
        # label a free_move gap between consecutive subtasks: a place
        # (transport_contact) is frequently followed directly by the next subtask's
        # pre_contact. Triggering on the exit from contact counts every placement,
        # including the final one when episodes abut subtasks back to back.
        advanced = False
        if stage in self.completion_stages:
            self._visited_completion = True
        elif self._visited_completion:
            advanced = self._advance(task_state)
            self._visited_completion = False
        self._prev_stage = stage
        return advanced

    def finalize(self, task_state: TaskState | None) -> bool:
        """Flush a trailing placement at episode end. Returns True if advanced.

        When a recording stops while (or right after) the last object is placed,
        the pattern never leaves the contact group, so ``observe`` never sees the
        exit that would advance the final subtask. Call this once after the last
        frame so the last placement still counts toward task completion.
        """
        if not self.enabled or task_state is None or not task_state.target_labels:
            return False
        if not self._visited_completion:
            return False
        advanced = self._advance(task_state)
        self._visited_completion = False
        return advanced

    def _advance(self, task_state: "TaskState") -> bool:
        from task.operations import is_task_complete

        if is_task_complete(task_state):
            return False
        before = self.current_subtask(task_state)
        if self.on_advance is not None:
            self.on_advance(task_state)
        block_id = task_state.target_block_id
        if block_id is not None:
            if block_id not in task_state.picked_block_ids:
                task_state.picked_block_ids.append(block_id)
            if block_id not in task_state.placed_block_ids:
                task_state.placed_block_ids.append(block_id)
        task_state.progress_idx += 1
        self._plan(task_state)
        after = self.current_subtask(task_state)
        _logger.info(
            "[subtask] complete: %s -> %s",
            before.description if before else "none",
            after.description if after else "task complete",
        )
        return True

    @staticmethod
    def _plan(task_state: "TaskState") -> None:
        from task.operations import plan_next_target

        task_state.target_block_id, task_state.target_placement_id = plan_next_target(task_state)
