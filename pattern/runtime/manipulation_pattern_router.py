"""Manipulation-pattern stabilization and routing via the unified pattern table."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pattern.runtime.registry import StageRegistry
from pattern.runtime.types import StageClassifierOutput, StageDecision

_logger = logging.getLogger(__name__)

def _parse_gated_stages(raw: Any) -> tuple[str, ...]:
    """Stage names the gate is limited to; ``all`` (or empty) means every switch."""
    if isinstance(raw, str):
        raw = () if raw.strip().lower() == "all" else (raw,)
    return tuple(str(item).strip() for item in (raw or ()) if str(item).strip())


def _parse_gated_transitions(raw: Any) -> tuple[tuple[str, str], ...]:
    """Directed ``(from_stage, to_stage)`` name pairs the gate is limited to."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        raise ValueError(
            "stage_aware.progress_gate.transitions must be a list of "
            "[from_stage, to_stage] pairs"
        )
    transitions: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, str):
            parts = [part.strip() for part in item.split("->") if part.strip()]
        else:
            parts = [str(part).strip() for part in item]
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                "stage_aware.progress_gate.transitions entries must be "
                f"[from_stage, to_stage] pairs; got {item!r}"
            )
        transitions.append((parts[0], parts[1]))
    return tuple(transitions)


@dataclass(frozen=True)
class ProgressGate:
    """Accept a stage switch only when predicted progress agrees with it.

    Mirrors the offline gate in :mod:`pattern.annotation.predict_data`: the
    stage being left must sit near its end and the stage being entered near its
    start. ``auto`` resolves to ``both`` (when one side lacks progress, ``both``
    falls back to whichever end is available).
    """

    mode: str = "auto"
    start_threshold: float = 0.3
    end_threshold: float = 0.7
    override_after_frames: int = 5
    """Release a candidate the gate held for this many consecutive frames; 0 disables."""
    stages: tuple[str, ...] = ()
    """Gate only switches with one of these stages on either end; empty gates all of them."""
    transitions: tuple[tuple[str, str], ...] = ()
    """If set, gate only these directed ``from -> to`` switches (overrides ``stages``)."""
    use_completion_transitions: bool = True
    """When no ``stages``/``transitions`` are set, gate ``advance_on_return_to -> completion_stages``."""

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ProgressGate":
        raw = (config.get("stage_aware", {}) or {}).get("progress_gate")
        if raw is None or raw is True:
            return cls()
        if raw is False:
            return cls(mode="off", use_completion_transitions=False)
        if not bool(raw.get("enabled", True)):
            return cls(mode="off", use_completion_transitions=False)
        # PyYAML (1.1) turns unquoted ``off``/``on``/``yes``/``no`` into bools.
        # ``False or "auto"`` would silently revive the gate as ``auto``.
        raw_mode = raw.get("mode", "auto")
        if isinstance(raw_mode, bool):
            if raw_mode:
                raise ValueError(
                    "unsupported stage_aware.progress_gate.mode=true/on; "
                    "expected auto, both, either, or off"
                )
            mode = "off"
        else:
            mode = str(raw_mode or "auto").lower()
        if mode not in {"auto", "both", "either", "off"}:
            raise ValueError(
                f"unsupported stage_aware.progress_gate.mode={mode!r}; "
                "expected auto, both, either, or off"
            )
        has_transitions = "transitions" in raw and raw.get("transitions") is not None
        has_stages = "stages" in raw
        if has_transitions:
            transitions = _parse_gated_transitions(raw.get("transitions"))
            stages: tuple[str, ...] = ()
            use_completion = False
        elif has_stages:
            transitions = ()
            stages = _parse_gated_stages(raw.get("stages"))
            use_completion = False
        else:
            transitions = ()
            stages = ()
            use_completion = True
        return cls(
            mode=mode,
            start_threshold=float(raw.get("start_threshold", 0.3)),
            end_threshold=float(raw.get("end_threshold", 0.7)),
            override_after_frames=max(0, int(raw.get("override_after_frames", 5))),
            stages=stages,
            transitions=transitions,
            use_completion_transitions=use_completion,
        )

    def resolve_mode(self, num_progress_heads: int) -> str:
        del num_progress_heads
        if self.mode == "auto":
            return "both"
        return self.mode

    def passes(
        self,
        current_progress: float | None,
        candidate_progress: float | None,
        *,
        num_progress_heads: int,
    ) -> bool:
        mode = self.resolve_mode(num_progress_heads)
        if mode == "off":
            return True
        current_ready = current_progress is not None and current_progress >= self.end_threshold
        candidate_ready = candidate_progress is not None and candidate_progress <= self.start_threshold
        if mode == "both":
            if current_progress is None or candidate_progress is None:
                return current_ready or candidate_ready
            return current_ready and candidate_ready
        if mode == "either":
            return current_ready or candidate_ready
        raise ValueError(f"unsupported progress gate mode: {mode}")


class StageStabilizer:
    """Debounce stage switches by requiring the same stage_id for N frames.

    With a progress-predicting checkpoint the debounced switch must also clear
    ``ProgressGate``; until it does, the candidate stays pending and keeps
    accumulating progress observations. Frames without progress bypass the gate,
    and so does a candidate the gate has held for ``override_after_frames``.
    """

    def __init__(
        self,
        *,
        default_stage_id: int,
        stable_frames: int = 3,
        progress_gate: ProgressGate | None = None,
        gated_stage_ids: frozenset[int] | None = None,
        gated_transitions: frozenset[tuple[int, int]] | None = None,
    ) -> None:
        self.default_stage_id = int(default_stage_id)
        self.stable_frames = max(1, int(stable_frames))
        self.progress_gate = progress_gate or ProgressGate()
        self.gated_stage_ids = gated_stage_ids
        self.gated_transitions = gated_transitions
        self.confirmed_stage_id = self.default_stage_id
        self._candidate_stage_id: int | None = None
        self._candidate_count = 0
        self._confirmed_progress: float | None = None
        self._pending_confirmed_progress: float | None = None
        self._pending_candidate_progress: float | None = None

    def reset(self) -> None:
        self.confirmed_stage_id = self.default_stage_id
        self._confirmed_progress = None
        self._clear_candidate()

    def _clear_candidate(self) -> None:
        self._candidate_stage_id = None
        self._candidate_count = 0
        self._pending_confirmed_progress = None
        self._pending_candidate_progress = None

    def _observe_progress(self, output: StageClassifierOutput, candidate_stage_id: int) -> None:
        """Track how far the confirmed stage got and how early the candidate is."""
        candidate = output.progress_for_stage(candidate_stage_id)
        if candidate is not None:
            self._pending_candidate_progress = (
                candidate
                if self._pending_candidate_progress is None
                else min(self._pending_candidate_progress, candidate)
            )
        confirmed = output.progress_for_stage(self.confirmed_stage_id)
        if confirmed is not None:
            self._pending_confirmed_progress = (
                confirmed
                if self._pending_confirmed_progress is None
                else max(self._pending_confirmed_progress, confirmed)
            )

    def _gate_applies(self, candidate_stage_id: int) -> bool:
        """Whether this particular switch is one the gate is scoped to.

        Directed ``gated_transitions`` win when set (exact ``from -> to`` edges).
        Otherwise ``gated_stage_ids`` keeps the undirected free_move guard:
        bouncing out of and back into free_move makes the subtask manager count
        a second placement and skip an object, while contact-side switches move
        freely.
        """
        if self.gated_transitions is not None:
            return (self.confirmed_stage_id, int(candidate_stage_id)) in self.gated_transitions
        if self.gated_stage_ids is None:
            return True
        return (
            self.confirmed_stage_id in self.gated_stage_ids
            or int(candidate_stage_id) in self.gated_stage_ids
        )

    def _gate_allows(self, output: StageClassifierOutput | None) -> bool:
        if output is None or not output.has_progress:
            return True
        observed = [
            value
            for value in (self._pending_confirmed_progress, self._confirmed_progress)
            if value is not None
        ]
        return self.progress_gate.passes(
            max(observed) if observed else None,
            self._pending_candidate_progress,
            num_progress_heads=output.num_progress_heads,
        )

    def _gate_override_due(self) -> bool:
        """Whether the gate has held one steady candidate long enough to give up.

        A classifier oscillating around a stage boundary can keep the progress
        conditions unmet indefinitely. Once the same candidate has survived
        debounce for this many consecutive frames it is no longer noise, so the
        stage advances rather than staying stuck in the one it already left.
        """
        limit = self.progress_gate.override_after_frames
        return limit > 0 and self._candidate_count >= limit

    def update(
        self,
        stage_id: int,
        output: StageClassifierOutput | None = None,
    ) -> tuple[int, bool]:
        stage_id = int(stage_id)
        if stage_id == self.confirmed_stage_id:
            self._clear_candidate()
            if output is not None:
                progress = output.progress_for_stage(self.confirmed_stage_id)
                if progress is not None:
                    self._confirmed_progress = progress
            return self.confirmed_stage_id, False

        if stage_id == self._candidate_stage_id:
            self._candidate_count += 1
        else:
            self._candidate_stage_id = stage_id
            self._candidate_count = 1
            self._pending_confirmed_progress = None
            self._pending_candidate_progress = None

        if output is not None:
            self._observe_progress(output, stage_id)

        if self._candidate_count < self.stable_frames:
            return self.confirmed_stage_id, False
        if self._gate_applies(stage_id) and not self._gate_allows(output):
            if not self._gate_override_due():
                return self.confirmed_stage_id, False
            _logger.info(
                "progress gate overridden: stage %d -> %d after %d steady frames "
                "(current=%s candidate=%s)",
                self.confirmed_stage_id,
                stage_id,
                self._candidate_count,
                self._pending_confirmed_progress,
                self._pending_candidate_progress,
            )

        previous_stage_id = self.confirmed_stage_id
        self.confirmed_stage_id = stage_id
        self._confirmed_progress = (
            output.progress_for_stage(stage_id) if output is not None else None
        )
        self._clear_candidate()
        return self.confirmed_stage_id, self.confirmed_stage_id != previous_stage_id


class ManipulationPatternRouter:
    def __init__(self, config: dict[str, Any]) -> None:
        self.registry = StageRegistry.from_config(config)
        stable_frames = int(config["stage_aware"]["stage_stability_frames"])
        gate = ProgressGate.from_config(config)
        gated_stage_ids: frozenset[int] | None = None
        gated_transitions: frozenset[tuple[int, int]] | None = None
        if gate.transitions:
            gated_transitions = self.registry.ids_for_transitions(gate.transitions)
        elif gate.use_completion_transitions:
            from task.subtask import SubtaskManager

            subtask = SubtaskManager.from_config(config)
            pairs = [
                (subtask.advance_on_return_to, stage)
                for stage in subtask.completion_stages
            ]
            gated_transitions = self.registry.ids_for_transitions(pairs)
        elif gate.stages:
            gated_stage_ids = self.registry.ids_for_stages(gate.stages)
        self.stabilizer = StageStabilizer(
            default_stage_id=self.registry.default_stage_id,
            stable_frames=stable_frames,
            progress_gate=gate,
            gated_stage_ids=gated_stage_ids,
            gated_transitions=gated_transitions,
        )

    def reset(self) -> None:
        self.stabilizer.reset()

    def _resolve_stage_id(self, output: StageClassifierOutput) -> int:
        if output.stage_id is not None:
            return int(output.stage_id)
        raw_stage_id = output.raw.get("stage_id")
        if raw_stage_id is not None:
            return int(raw_stage_id)
        return self.registry.resolve_id(output.stage, output.focus)

    def decide(self, output: StageClassifierOutput) -> StageDecision:
        raw_stage_id = self._resolve_stage_id(output)
        confirmed_stage_id, changed = self.stabilizer.update(raw_stage_id, output)
        spec = self.registry.get(confirmed_stage_id)
        raw_spec = self.registry.get(raw_stage_id)
        return StageDecision(
            stage_id=confirmed_stage_id,
            raw_stage_id=raw_stage_id,
            raw_stage=raw_spec.name,
            confirmed_stage=spec.name,
            stage_changed=changed,
            route=spec.to_route(),
            phase=output.phase,
            focus=spec.focus,
            confidence=output.confidence,
            # Progress of the *confirmed* stage (gate may still be holding a raw switch).
            progress=output.progress_for_stage(confirmed_stage_id),
        )
