"""Persist rendered frames from online inference to image outputs."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from common.project_config import resolve_render_image_dir

_logger = logging.getLogger(__name__)
_session_lock = threading.Lock()
_session_dir: Path | None = None

_STITCH_ORDER = ("left_wrist", "global", "right_wrist")
_STAGE_COLOR = (255, 0, 0)

_stage_font: ImageFont.ImageFont | None = None
_writer: _AsyncRenderWriter | None = None


def begin_render_session() -> Path:
    """Start a new session directory under ``output.render_image`` (on ``/reset``)."""
    flush_render_writes()
    global _session_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = resolve_render_image_dir() / f"session_{timestamp}"
    session.mkdir(parents=True, exist_ok=True)
    with _session_lock:
        _session_dir = session
    _logger.info("render image output session: %s", session)
    return session


def get_render_output_dir() -> Path:
    """Return the active session dir, or ``<render_image>/live`` before the first reset."""
    global _session_dir
    with _session_lock:
        if _session_dir is None:
            _session_dir = resolve_render_image_dir() / "live"
            _session_dir.mkdir(parents=True, exist_ok=True)
        return _session_dir


def flush_render_writes(timeout: float = 30.0) -> None:
    """Block until queued render writes finish (e.g. before ``/reset``)."""
    if _writer is not None:
        _writer.flush(timeout=timeout)


def _write_image_immediate(image: Image.Image, path: Path) -> None:
    """Flush PNG to disk immediately so file watchers / viewers see it right away."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    payload = buffer.getvalue()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def _stage_label_font() -> ImageFont.ImageFont:
    global _stage_font
    if _stage_font is not None:
        return _stage_font
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if Path(path).is_file():
            _stage_font = ImageFont.truetype(path, 28)
            return _stage_font
    _stage_font = ImageFont.load_default()
    return _stage_font


def _resize_to_height(image: Image.Image, height: int) -> Image.Image:
    if image.height == height:
        return image
    width = max(1, int(round(image.width * height / image.height)))
    return image.resize((width, height), Image.Resampling.BILINEAR)


def _draw_stage_label(image: Image.Image, stage: str) -> Image.Image:
    labeled = image.copy()
    draw = ImageDraw.Draw(labeled)
    font = _stage_label_font()
    label = str(stage)
    margin = 10
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    pad = 4
    box = (margin - pad, margin - pad, margin + text_w + pad, margin + text_h + pad)
    draw.rectangle(box, fill=(0, 0, 0))
    draw.text((margin, margin), label, fill=_STAGE_COLOR, font=font)
    return labeled


def _build_combined_panel(cameras: dict[str, Image.Image], stage: str | None) -> Image.Image | None:
    panels = [
        cameras[key].convert("RGB")
        for key in _STITCH_ORDER
        if key in cameras
    ]
    if not panels:
        return None

    target_height = max(panel.height for panel in panels)
    panels = [_resize_to_height(panel, target_height) for panel in panels]
    combined = Image.new("RGB", (sum(panel.width for panel in panels), target_height))
    offset_x = 0
    for panel in panels:
        combined.paste(panel, (offset_x, 0))
        offset_x += panel.width

    if stage:
        combined = _draw_stage_label(combined, stage)
    return combined


def _persist_render_job(
    *,
    out_dir: Path,
    cameras: dict[str, Image.Image],
    frame_idx: int,
    stage: str | None,
    prompt: str | None,
) -> None:
    latest_dir = out_dir / "latest"
    try:
        combined = _build_combined_panel(cameras, stage)
        if combined is not None:
            _write_image_immediate(combined, out_dir / f"combined_{frame_idx:06d}.png")
            _write_image_immediate(combined, latest_dir / "combined.png")

        meta_lines = [
            f"frame_idx={frame_idx}",
            f"updated_at={datetime.now().isoformat(timespec='milliseconds')}",
        ]
        if stage is not None:
            meta_lines.append(f"stage={stage}")
        if prompt is not None:
            meta_lines.append(f"prompt={prompt!r}")
        meta_path = latest_dir / "meta.txt"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
    except Exception:
        _logger.exception("failed to save render image output to %s", out_dir)


@dataclass(frozen=True)
class _RenderSaveJob:
    out_dir: Path
    cameras: dict[str, Image.Image]
    frame_idx: int
    stage: str | None
    prompt: str | None


class _AsyncRenderWriter:
    def __init__(self) -> None:
        self._queue: queue.Queue[_RenderSaveJob] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="render-save", daemon=True)
        self._thread.start()

    def enqueue(self, job: _RenderSaveJob) -> None:
        self._queue.put(job)

    def flush(self, timeout: float = 30.0) -> None:
        if not self._thread.is_alive():
            return
        deadline = time.time() + timeout
        while True:
            if self._queue.unfinished_tasks == 0:
                return
            if time.time() > deadline:
                _logger.warning(
                    "render save flush timed out with %d pending jobs",
                    self._queue.unfinished_tasks,
                )
                return
            time.sleep(0.01)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                _persist_render_job(
                    out_dir=job.out_dir,
                    cameras=job.cameras,
                    frame_idx=job.frame_idx,
                    stage=job.stage,
                    prompt=job.prompt,
                )
            except Exception:
                _logger.exception("async render save worker failed")
            finally:
                self._queue.task_done()


def _get_writer() -> _AsyncRenderWriter:
    global _writer
    if _writer is None:
        _writer = _AsyncRenderWriter()
    return _writer


def save_render_images(
    rendered: dict[str, Image.Image] | Image.Image,
    frame_idx: int,
    *,
    stage: str | None = None,
    prompt: str | None = None,
) -> None:
    """Queue a left|global|right combined frame for background disk write.

    PNG encode + fsync run on a daemon thread so ``/step`` is not blocked.
    At ~0.3 s/step the writer easily keeps up; the queue is unbounded.

    Per-frame archive: ``combined_{frame_idx:06d}.png``
    Live preview:       ``latest/combined.png``
    """
    if isinstance(rendered, Image.Image):
        camera_map = {"global": rendered}
    else:
        camera_map = rendered

    cameras_copy = {
        key: camera_map[key].copy()
        for key in _STITCH_ORDER
        if key in camera_map
    }
    if not cameras_copy:
        return

    _get_writer().enqueue(
        _RenderSaveJob(
            out_dir=get_render_output_dir(),
            cameras=cameras_copy,
            frame_idx=frame_idx,
            stage=stage,
            prompt=prompt,
        )
    )
