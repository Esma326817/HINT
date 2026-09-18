"""Core model latency profiling for ``/step``, bucketed by stage.

Only expensive inference/render sections are instrumented: stage prediction,
SAM tracking/initialization, visual grounding models, and mask rendering. The
samples are accumulated per stage and split by whether a frame attempted
grounding or only tracked. Stats are flushed to a JSON report for inspection.
"""

from __future__ import annotations

import atexit
import functools
import inspect
import json
import logging
import os
import statistics
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT = "benchmarks/step_module_latency.json"
_DEFAULT_FLUSH_EVERY = 20
_DEFAULT_SAMPLE_WINDOW = 1000
_OVERALL_KEY = "__overall__"


def _env_flag(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _summary(samples: deque[float]) -> dict[str, Any]:
    values = list(samples)
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {
        "count": len(values),
        "mean_ms": round(statistics.mean(values), 2),
        "median_ms": round(statistics.median(values), 2),
        "p95_ms": round(p95, 2),
        "min_ms": round(ordered[0], 2),
        "max_ms": round(ordered[-1], 2),
    }


class _NullSection:
    def __enter__(self) -> "_NullSection":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


_NULL_SECTION = _NullSection()


class _Section:
    __slots__ = ("_profiler", "_record", "_name", "_cuda", "_started")

    def __init__(self, profiler: "StepProfiler", record: "_FrameRecord", name: str, cuda: bool) -> None:
        self._profiler = profiler
        self._record = record
        self._name = name
        self._cuda = cuda and profiler.cuda_sync
        self._started = 0.0

    def __enter__(self) -> "_Section":
        if self._cuda:
            self._profiler.sync_cuda()
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self._cuda:
            self._profiler.sync_cuda()
        elapsed = (time.perf_counter() - self._started) * 1000.0
        record = self._record
        record.timings[self._name] = record.timings.get(self._name, 0.0) + elapsed
        return False


class _FrameRecord:
    __slots__ = (
        "timings",
        "stage",
        "stage_id",
        "grounded_cameras",
        "grounding_sources",
    )

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self.stage: str | None = None
        self.stage_id: int | None = None
        self.grounded_cameras: list[str] = []
        self.grounding_sources: list[str] = []


class _FrameContext:
    __slots__ = ("_profiler",)

    def __init__(self, profiler: "StepProfiler") -> None:
        self._profiler = profiler

    def __enter__(self) -> "_FrameContext":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self._profiler._end_frame()
        return False


class StepProfiler:
    """Accumulate per-module ``/step`` timings, bucketed by stage."""

    def __init__(self) -> None:
        self.enabled = False
        self.cuda_sync = True
        self.output_path: Path | None = None
        self.flush_every = _DEFAULT_FLUSH_EVERY
        self.sample_window = _DEFAULT_SAMPLE_WINDOW
        self._lock = threading.Lock()
        self._local = threading.local()
        self._configured = False
        self._buckets: dict[str, dict[str, Any]] = {}
        self._frames_total = 0
        self._frames_reasoning = 0
        self._frames_since_flush = 0
        self._started_at: str | None = None
        self._config_path: str | None = None
        self._torch: Any = None

    # ── configuration ──────────────────────────────────────────────────────
    def configure(self, config: dict[str, Any], *, config_path: str | Path | None = None) -> None:
        if self._configured:
            return
        profiling_cfg = config.get("profiling") or {}
        enabled = _env_flag("REASONING_PROFILE")
        if enabled is None:
            # Profiling synchronizes CUDA around selected sections and therefore
            # intentionally changes throughput.  Keep normal service runs clean;
            # opt in through YAML or REASONING_PROFILE=1.
            enabled = bool(profiling_cfg.get("enabled", False))
        self.enabled = enabled
        self._configured = True
        if not self.enabled:
            return

        self.cuda_sync = bool(profiling_cfg.get("cuda_sync", True))
        self.flush_every = max(1, int(profiling_cfg.get("flush_every", _DEFAULT_FLUSH_EVERY)))
        self.sample_window = max(1, int(profiling_cfg.get("sample_window", _DEFAULT_SAMPLE_WINDOW)))
        self.output_path = self._resolve_output(config, profiling_cfg)
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self._config_path = str(config_path) if config_path is not None else None
        atexit.register(self.flush)
        _logger.info("[profile] step module timings → %s", self.output_path)

    def profile_frame(self, function: Any) -> Any:
        """Decorate one request as a profiling frame while preserving its API signature."""

        @functools.wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with self.frame():
                return function(*args, **kwargs)

        # Resolve postponed annotations in the endpoint's own module. FastAPI
        # otherwise evaluates strings such as UploadFile in this wrapper's globals.
        wrapped.__signature__ = inspect.signature(function, eval_str=True)
        return wrapped

    @staticmethod
    def _resolve_output(config: dict[str, Any], profiling_cfg: dict[str, Any]) -> Path:
        configured = os.getenv("REASONING_PROFILE_OUTPUT") or profiling_cfg.get("output") or _DEFAULT_OUTPUT
        path = Path(str(configured))
        if path.is_absolute():
            return path
        from common.project_config import resolve_output_root

        return resolve_output_root(config) / path

    # ── recording ──────────────────────────────────────────────────────────
    def frame(self) -> Any:
        """Open a profiling frame for the current thread (no-op when disabled)."""
        if not self.enabled:
            return _NULL_SECTION
        self._local.record = _FrameRecord()
        return _FrameContext(self)

    def section(self, name: str, *, cuda: bool = False) -> Any:
        record = getattr(self._local, "record", None)
        if record is None:
            return _NULL_SECTION
        return _Section(self, record, name, cuda)

    def set_stage(self, stage: str | None, stage_id: int | None) -> None:
        record = getattr(self._local, "record", None)
        if record is None:
            return
        record.stage = stage
        record.stage_id = stage_id

    def mark_grounded(self, camera: str, source: str | None = None) -> None:
        """Flag a grounding attempt and optionally record its selected source."""
        record = getattr(self._local, "record", None)
        if record is None:
            return
        record.grounded_cameras.append(camera)
        if source:
            record.grounding_sources.append(str(source))

    def sync_cuda(self) -> None:
        torch = self._torch
        if torch is None:
            import torch as torch_module

            torch = self._torch = torch_module
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # ── aggregation ────────────────────────────────────────────────────────
    def _end_frame(self) -> None:
        record: _FrameRecord | None = getattr(self._local, "record", None)
        self._local.record = None
        if record is None:
            return
        reasoning = bool(record.grounded_cameras)
        stage_key = record.stage or "unknown"
        if record.stage_id is not None:
            stage_key = f"{stage_key}#{record.stage_id}"

        should_flush = False
        with self._lock:
            self._frames_total += 1
            if reasoning:
                self._frames_reasoning += 1
            for key in (stage_key, _OVERALL_KEY):
                self._add(key, reasoning, record)
            self._frames_since_flush += 1
            if self._frames_since_flush >= self.flush_every:
                self._frames_since_flush = 0
                should_flush = True
        if should_flush:
            self.flush()

    def _add(self, bucket_key: str, reasoning: bool, record: _FrameRecord) -> None:
        bucket = self._buckets.setdefault(bucket_key, {})
        group = bucket.setdefault(
            "reasoning" if reasoning else "non_reasoning",
            {
                "frames": 0,
                "modules": {},
                "grounded_cameras": {},
                "grounding_sources": {},
            },
        )
        group["frames"] += 1
        modules = group["modules"]
        for name, value in record.timings.items():
            samples = modules.get(name)
            if samples is None:
                samples = modules[name] = deque(maxlen=self.sample_window)
            samples.append(value)
        for camera in record.grounded_cameras:
            group["grounded_cameras"][camera] = group["grounded_cameras"].get(camera, 0) + 1
        for source in record.grounding_sources:
            group["grounding_sources"][source] = group["grounding_sources"].get(source, 0) + 1

    # ── reporting ──────────────────────────────────────────────────────────
    def _build_report(self) -> dict[str, Any]:
        stages = {
            key: {
                mode: {
                    "frames": group["frames"],
                    "grounded_cameras": dict(group["grounded_cameras"]),
                    "grounding_sources": dict(group["grounding_sources"]),
                    "modules": {
                        name: _summary(samples) for name, samples in sorted(group["modules"].items())
                    },
                }
                for mode, group in sorted(bucket.items())
            }
            for key, bucket in sorted(self._buckets.items())
            if key != _OVERALL_KEY
        }
        overall = self._buckets.get(_OVERALL_KEY, {})
        return {
            "enabled": self.enabled,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "started_at": self._started_at,
            "config_path": self._config_path,
            "cuda_sync": self.cuda_sync,
            "sample_window": self.sample_window,
            "module_note": (
                "Only core compute is timed: stage model, SAM update/init, grounding "
                "models, and render. The group name 'reasoning' means a grounding-selection "
                "attempt; consult grounding_sources to distinguish model calls from known "
                "bbox reuse."
            ),
            "frames": {
                "total": self._frames_total,
                "reasoning": self._frames_reasoning,
                "non_reasoning": self._frames_total - self._frames_reasoning,
            },
            "overall": {
                mode: {
                    "frames": group["frames"],
                    "grounded_cameras": dict(group["grounded_cameras"]),
                    "grounding_sources": dict(group["grounding_sources"]),
                    "modules": {
                        name: _summary(samples) for name, samples in sorted(group["modules"].items())
                    },
                }
                for mode, group in sorted(overall.items())
            },
            "stages": stages,
        }

    def snapshot(self) -> dict[str, Any]:
        """Return the current JSON-serializable report without writing it."""
        if not self.enabled:
            return {"enabled": False, "frames": {"total": 0}}
        with self._lock:
            return self._build_report()

    def flush(self) -> None:
        if not self.enabled or self.output_path is None:
            return
        with self._lock:
            if self._frames_total == 0:
                return
            report = self._build_report()
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
            temporary.replace(self.output_path)
        except OSError:
            _logger.exception("failed to write step profile to %s", self.output_path)


PROFILER = StepProfiler()
