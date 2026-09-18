#!/usr/bin/env python3
"""Annotate or review LeRobot ``stage_id_gt`` labels.

Usage:
  python pattern_tool.py                 # annotate (default; also loads existing parquet labels)
  python pattern_tool.py --mode review
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PyQt5.QtCore import QPoint, QRect, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSlider,
    QSplitter,
    QToolTip,
    QVBoxLayout,
    QWidget,
)


VIDEO_EXTS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
    ".flv",
    ".wmv",
    ".mpg",
    ".mpeg",
    ".3gp",
    ".ts",
    ".m2ts",
    ".mts",
    ".ogv",
}

EPISODE_RE = re.compile(r"episode_(\d+)")
STAGE_COLUMN = "stage_id_gt"

STAGE_LABELS = {
    1: "free-move",
    2: "pre_contact-left",
    3: "pre_contact-right",
    4: "dexterous_contact-left",
    5: "dexterous_contact-right",
    6: "transport_contact",
}

STAGE_BADGE_COLORS = {
    1: ("#2e7d32", "#ffffff"),
    2: ("#ef6c00", "#ffffff"),
    3: ("#2563eb", "#ffffff"),
    4: ("#c62828", "#ffffff"),
    5: ("#7e22ce", "#ffffff"),
    6: ("#047857", "#ffffff"),
}

CAMERA_ROLES = ("global", "left_wrist", "right_wrist")
CAMERA_VIDEO_DIRS = {
    "global": "images.global",
    "left_wrist": "images.left_wrist",
    "right_wrist": "images.right_wrist",
}
CAMERA_TITLES = {
    "global": "Global",
    "left_wrist": "Left wrist",
    "right_wrist": "Right wrist",
}

PLAY_TEXT = "Play"
PAUSE_TEXT = "Pause"


@dataclass
class StageEvent:
    frame_idx: int
    stage: int


def parse_fps(value: str) -> float:
    if not value or value == "0/0":
        return 25.0
    if "/" in value:
        num_s, den_s = value.split("/", 1)
        den = float(den_s)
        return float(num_s) / den if den else 25.0
    return float(value)


def format_time(seconds: float) -> str:
    seconds_i = max(0, int(seconds))
    return f"{seconds_i // 60:02d}:{seconds_i % 60:02d}"


def fit_pixmap_to_label(qimg: QImage, label: QLabel) -> QPixmap:
    return QPixmap.fromImage(qimg).scaled(
        max(1, label.width()),
        max(1, label.height()),
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation,
    )


def stage_events_from_ids(stage_ids: np.ndarray) -> list[StageEvent]:
    stage_ids = np.asarray(stage_ids, dtype=np.int32).ravel()
    if stage_ids.size == 0:
        return []
    events = [StageEvent(frame_idx=0, stage=int(stage_ids[0]))]
    prev = int(stage_ids[0])
    for idx, value in enumerate(stage_ids[1:], start=1):
        stage = int(value)
        if stage != prev:
            events.append(StageEvent(frame_idx=idx, stage=stage))
            prev = stage
    return events


def canonical_stage_events(history: list[StageEvent]) -> list[StageEvent]:
    dedup: dict[int, StageEvent] = {}
    for event in history:
        dedup[event.frame_idx] = event
    return sorted(dedup.values(), key=lambda item: item.frame_idx)


def resolve_dataset_root(folder: Path) -> Path:
    for candidate in [folder, *folder.parents]:
        if (candidate / "data").is_dir():
            return candidate
    return folder


def find_videos_recursively(folder: Path) -> list[Path]:
    videos: list[Path] = []
    for root, _dirs, files in os.walk(folder, followlinks=True):
        root_path = Path(root)
        for name in files:
            path = root_path / name
            if path.suffix.lower() in VIDEO_EXTS:
                videos.append(path)
    videos = sorted(videos, key=lambda p: str(p).lower())
    global_videos = [
        path for path in videos if path.parent.name == CAMERA_VIDEO_DIRS["global"]
    ]
    return global_videos if global_videos else videos


def find_camera_videos_for_episode(video_path: Path, dataset_root: Path | None) -> dict[str, Path]:
    match = EPISODE_RE.search(video_path.name)
    if match is None:
        return {"global": video_path}

    episode_name = f"episode_{int(match.group(1)):06d}{video_path.suffix.lower()}"
    roots: list[Path] = []
    if dataset_root is not None:
        roots.extend([dataset_root / "videos", dataset_root])
    roots.append(video_path.parents[0])

    out: dict[str, Path] = {}
    for role, dirname in CAMERA_VIDEO_DIRS.items():
        for root in roots:
            if not root.exists():
                continue
            matches = sorted(root.rglob(f"{dirname}/{episode_name}"))
            if matches:
                out[role] = matches[0]
                break
    if "global" not in out:
        out["global"] = video_path
    return out


def find_parquet_for_video(video_path: Path, dataset_root: Path | None) -> Path | None:
    match = EPISODE_RE.search(video_path.name)
    if match is None:
        return None

    episode_name = f"episode_{int(match.group(1)):06d}.parquet"
    search_roots = []
    if dataset_root is not None:
        search_roots.append(dataset_root / "data")
        search_roots.append(dataset_root)
    search_roots.append(video_path.parents[0])

    for root in search_roots:
        if not root.exists():
            continue
        matches = sorted(root.rglob(episode_name))
        if matches:
            return matches[0]
    return None


def row_positions_in_video(
    row_count: int,
    table: pa.Table,
    total_frames: int,
    last_video_frame: int,
) -> np.ndarray:
    if row_count == total_frames:
        return np.arange(row_count, dtype=np.float64)
    if "frame_index" in table.column_names:
        frame_index = np.asarray(table.column("frame_index").to_numpy(), dtype=np.float64)
        if frame_index.size == row_count:
            min_frame = float(np.min(frame_index))
            max_frame = float(np.max(frame_index))
            if max_frame > min_frame:
                return (frame_index - min_frame) / (max_frame - min_frame) * last_video_frame
    return np.linspace(0, last_video_frame, row_count, dtype=np.float64)


def compute_stage_ids(
    events: list[StageEvent],
    row_count: int,
    table: pa.Table,
    total_frames: int,
) -> np.ndarray:
    if not events:
        raise ValueError("No stage events on the current video")
    last_video_frame = max(1, total_frames - 1)
    row_video_frames = row_positions_in_video(row_count, table, total_frames, last_video_frame)
    event_frames = np.asarray([event.frame_idx for event in events], dtype=np.float64)
    event_stages = np.asarray([event.stage for event in events], dtype=np.int32)
    stage_ids = np.zeros(row_count, dtype=np.int32)
    for i, frame_pos in enumerate(row_video_frames):
        event_idx = int(np.searchsorted(event_frames, frame_pos, side="right")) - 1
        event_idx = max(0, min(event_idx, len(event_stages) - 1))
        stage_ids[i] = event_stages[event_idx]
    return stage_ids


def write_stage_id_to_parquet(
    parquet_path: Path,
    events: list[StageEvent],
    total_frames: int,
) -> int:
    table = pq.read_table(parquet_path)
    row_count = table.num_rows
    if row_count <= 0:
        raise ValueError("parquet is empty")

    stage_ids = compute_stage_ids(events, row_count, table, total_frames)
    backup_path = parquet_path.with_suffix(parquet_path.suffix + ".bak")
    if not backup_path.exists():
        shutil.copy2(parquet_path, backup_path)

    stage_id_array = pa.array(stage_ids.astype(np.int32))
    if STAGE_COLUMN in table.column_names:
        col_idx = table.column_names.index(STAGE_COLUMN)
        table = table.set_column(col_idx, STAGE_COLUMN, stage_id_array)
    else:
        table = table.append_column(STAGE_COLUMN, stage_id_array)
    pq.write_table(table, parquet_path)
    return row_count


def load_stage_events_from_parquet(
    parquet_path: Path | None,
    total_frames: int,
) -> list[StageEvent]:
    if parquet_path is None or not parquet_path.exists():
        return []
    try:
        table = pq.read_table(parquet_path)
    except Exception:
        return []
    if STAGE_COLUMN not in table.column_names:
        return []

    stage_ids = np.asarray(table[STAGE_COLUMN].to_numpy(), dtype=np.int32)
    row_events = stage_events_from_ids(stage_ids)
    row_count = stage_ids.size
    if row_count == 0:
        return []
    if row_count == total_frames:
        return row_events

    last_video_frame = max(1, total_frames - 1)
    row_video_frames = row_positions_in_video(row_count, table, total_frames, last_video_frame)
    mapped: list[StageEvent] = []
    for event in row_events:
        frame_idx = int(round(float(row_video_frames[event.frame_idx])))
        frame_idx = max(0, min(frame_idx, max(0, total_frames - 1)))
        mapped.append(StageEvent(frame_idx=frame_idx, stage=event.stage))
    return mapped


class GlobalVideoWidget(QWidget):
    """Global camera view with a stage overlay in the top-left corner."""

    def __init__(self, placeholder: str = "Select a video", parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.video_label = QLabel(placeholder, self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setObjectName("videoCanvas")
        self.stage_overlay = QLabel("-", self)
        self.stage_overlay.setObjectName("stageOverlay")
        self.stage_overlay.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.stage_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._position_overlay()

    def _position_overlay(self):
        self.stage_overlay.adjustSize()
        self.stage_overlay.move(16, 14)
        self.stage_overlay.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.video_label.setGeometry(0, 0, self.width(), self.height())
        self._position_overlay()


class CameraPanelWidget(QWidget):
    layout_updated = pyqtSignal()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.layout_updated.emit()


class FFmpegVideoReader:
    """Decode video frames through system ffmpeg."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.width = 0
        self.height = 0
        self.fps = 25.0
        self.duration = 0.0
        self.total_frames = 0
        self.current_frame = 0
        self.process: subprocess.Popen | None = None
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise RuntimeError("ffmpeg/ffprobe not found; install system ffmpeg")
        self._probe()

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * 3

    def _probe(self):
        cmd = [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(self.path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "ffprobe failed to read video metadata")

        meta = json.loads(proc.stdout)
        video_stream = next(
            (s for s in meta.get("streams", []) if s.get("codec_type") == "video"),
            None,
        )
        if video_stream is None:
            raise RuntimeError("No video stream in file")

        self.width = int(video_stream.get("width") or 0)
        self.height = int(video_stream.get("height") or 0)
        self.fps = parse_fps(
            video_stream.get("avg_frame_rate")
            or video_stream.get("r_frame_rate")
            or "25/1"
        )
        duration_s = (
            video_stream.get("duration")
            or meta.get("format", {}).get("duration")
            or 0
        )
        self.duration = float(duration_s or 0.0)
        nb_frames = video_stream.get("nb_frames")
        self.total_frames = int(nb_frames) if nb_frames and str(nb_frames).isdigit() else 0
        if self.total_frames <= 0 and self.duration > 0:
            self.total_frames = max(1, int(round(self.duration * self.fps)))
        if self.width <= 0 or self.height <= 0 or self.total_frames <= 0:
            raise RuntimeError("Invalid video width, height, frame count, or duration")

    def start(self, start_frame: int = 0):
        self.close()
        self.current_frame = max(0, min(start_frame, max(0, self.total_frames - 1)))
        start_time = self.current_frame / max(self.fps, 1.0)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_time:.6f}", "-i", str(self.path),
            "-an", "-sn", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.frame_bytes * 2,
        )

    def read_frame(self) -> tuple[bool, int, bytes | None]:
        if self.process is None or self.process.stdout is None:
            return False, self.current_frame, None
        data = self.process.stdout.read(self.frame_bytes)
        if len(data) != self.frame_bytes:
            return False, self.current_frame, None
        frame_idx = self.current_frame
        self.current_frame += 1
        return True, frame_idx, data

    def close(self):
        if self.process is None:
            return
        proc = self.process
        self.process = None
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=0.5)


class AnnotationSlider(QSlider):
    marker_clicked = pyqtSignal(int)

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.markers: list[int] = []
        self.marker_stages: list[int] = []
        self.selected_marker: int | None = None
        self._marker_rects: dict[int, QRect] = {}
        self.dark_theme = False
        self.fps = 25.0
        self._jump_dragging = False
        self.setMouseTracking(True)
        self.setMinimumHeight(38)

    def set_dark_theme(self, enabled: bool):
        self.dark_theme = bool(enabled)
        self.update()

    def set_fps(self, fps: float):
        self.fps = max(float(fps), 1.0)

    def set_markers(
        self,
        markers: list[int],
        marker_stages: list[int],
        selected_marker: int | None = None,
    ):
        self.markers = list(markers)
        self.marker_stages = list(marker_stages)
        self.selected_marker = selected_marker
        self.update()

    def _x_for_value(self, value: int) -> int:
        max_value = max(1, self.maximum() - self.minimum())
        ratio = (value - self.minimum()) / max_value
        usable = max(1, self.width() - 16)
        return int(8 + ratio * usable)

    def _value_for_x(self, x: int) -> int:
        usable = max(1, self.width() - 16)
        ratio = max(0.0, min(1.0, (x - 8) / usable))
        return int(round(self.minimum() + ratio * (self.maximum() - self.minimum())))

    def _tooltip_for_value(self, value: int) -> str:
        seconds = max(0, int(value / self.fps))
        return f"{seconds // 60:02d}:{seconds % 60:02d} ({seconds}s)"

    def _show_time_tooltip(self, value: int, pos: QPoint):
        self.setToolTip(self._tooltip_for_value(value))
        QToolTip.showText(self.mapToGlobal(pos), self.toolTip(), self)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.markers:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        font = QFont()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)

        if self.dark_theme:
            normal_fill, normal_border, normal_text = "#2b2c2f", "#8ab4f8", "#e8eaed"
            stage_palette = {
                1: ("#22543d", "#68d391", "#d9fbe8"),
                2: ("#6b3f12", "#f6ad55", "#fff1dc"),
                3: ("#1e3a8a", "#7aa2ff", "#e6f0ff"),
                4: ("#7f1d1d", "#f87171", "#fee2e2"),
                5: ("#4c1d95", "#c084fc", "#f3e8ff"),
                6: ("#064e3b", "#34d399", "#d1fae5"),
            }
        else:
            normal_fill, normal_border, normal_text = "#ffffff", "#1a73e8", "#1a73e8"
            stage_palette = {
                1: ("#d6f5d6", "#2e7d32", "#1b5e20"),
                2: ("#ffe6bf", "#ef6c00", "#8d4e00"),
                3: ("#dbeafe", "#2563eb", "#1e3a8a"),
                4: ("#fde2e2", "#c62828", "#7f1d1d"),
                5: ("#f3e8ff", "#7e22ce", "#4c1d95"),
                6: ("#dcfce7", "#047857", "#064e3b"),
            }

        self._marker_rects.clear()
        for idx, frame_idx in enumerate(self.markers):
            stage = self.marker_stages[idx] if idx < len(self.marker_stages) else 0
            label = str(stage if stage else idx + 1)
            x = self._x_for_value(frame_idx)
            selected = idx == self.selected_marker
            width = max(18, 10 + len(label) * 7)
            rect = QRect(x - width // 2, 1, width, 16)
            self._marker_rects[idx] = rect
            fill_color, border_color, text_color = stage_palette.get(
                stage, (normal_fill, normal_border, normal_text)
            )
            painter.setPen(QPen(QColor(border_color), 1))
            painter.drawLine(x, rect.bottom(), x, self.height() - 12)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 28 if not self.dark_theme else 70))
            painter.drawRoundedRect(rect.translated(0, 1), 8, 8)
            painter.setBrush(QColor(border_color if selected else fill_color))
            painter.setPen(QPen(QColor(border_color), 2 if selected else 1))
            painter.drawRoundedRect(rect, 8, 8)
            painter.setPen(QPen(QColor("#ffffff" if selected else text_color)))
            painter.drawText(rect, Qt.AlignCenter, label)

    def mousePressEvent(self, event):
        for marker_idx, rect in self._marker_rects.items():
            if rect.adjusted(-4, -4, 4, 4).contains(event.pos()):
                self.marker_clicked.emit(marker_idx)
                return
        if event.button() == Qt.LeftButton:
            self._jump_dragging = True
            self.sliderPressed.emit()
            value = self._value_for_x(event.pos().x())
            self.setValue(value)
            self.sliderMoved.emit(value)
            self._show_time_tooltip(value, event.pos())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        value = self._value_for_x(event.pos().x())
        self._show_time_tooltip(value, event.pos())
        if self._jump_dragging:
            self.setValue(value)
            self.sliderMoved.emit(value)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._jump_dragging and event.button() == Qt.LeftButton:
            self._jump_dragging = False
            value = self._value_for_x(event.pos().x())
            self.setValue(value)
            self.sliderReleased.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


def theme_colors(dark: bool) -> dict[str, str]:
    if dark:
        return {
            "window_bg": "#202124", "text": "#e8eaed", "input_bg": "#2b2c2f",
            "border": "#4b4d52", "selection": "#5f8cff", "button_bg": "#34363a",
            "button_hover": "#41444a", "button_pressed": "#2b2d31",
            "button_border_hover": "#70757d",
            "stage1_bg": "#22543d", "stage1_hover": "#2f6a4f", "stage1_text": "#d9fbe8",
            "stage2_bg": "#6b3f12", "stage2_hover": "#875215", "stage2_text": "#fff1dc",
            "stage3_bg": "#1e3a8a", "stage3_hover": "#2b4db3", "stage3_text": "#e6f0ff",
            "stage4_bg": "#7f1d1d", "stage4_hover": "#991b1b", "stage4_text": "#fee2e2",
            "stage5_bg": "#4c1d95", "stage5_hover": "#5b21b6", "stage5_text": "#f3e8ff",
            "stage6_bg": "#064e3b", "stage6_hover": "#065f46", "stage6_text": "#d1fae5",
            "undo_bg": "#4a1d24", "undo_hover": "#61202c", "undo_text": "#ffd7de",
            "clear_bg": "#4a2e12", "clear_hover": "#624016", "clear_text": "#ffe7c2",
            "menu_bg": "#2b2c2f", "list_bg": "#17181b", "list_hover": "#2f3338",
            "slider_bg": "#4a4d52", "label_muted": "#bdc1c6",
            "canvas_bg": "#111111", "canvas_text": "#aaaaaa",
            "theme_bg": "#34363a", "theme_hover": "#41444a",
            "scroll_groove": "#2b2c2f", "scroll_handle": "#5f6368",
            "scroll_handle_hover": "#7a8088", "scroll_handle_pressed": "#1a73e8",
        }
    return {
        "window_bg": "#f6f8fb", "text": "#202124", "input_bg": "#ffffff",
        "border": "#d0d7de", "selection": "#d2e3fc", "button_bg": "#ffffff",
        "button_hover": "#f3f4f6", "button_pressed": "#eaeef2",
        "button_border_hover": "#afb8c1",
        "stage1_bg": "#e8f5e9", "stage1_hover": "#d9f2db", "stage1_text": "#1b5e20",
        "stage2_bg": "#fff2df", "stage2_hover": "#ffe9c2", "stage2_text": "#8d4e00",
        "stage3_bg": "#dbeafe", "stage3_hover": "#cfe3fd", "stage3_text": "#1e3a8a",
        "stage4_bg": "#fde2e2", "stage4_hover": "#fecaca", "stage4_text": "#7f1d1d",
        "stage5_bg": "#f3e8ff", "stage5_hover": "#ead6ff", "stage5_text": "#4c1d95",
        "stage6_bg": "#dcfce7", "stage6_hover": "#bbf7d0", "stage6_text": "#064e3b",
        "undo_bg": "#fce7eb", "undo_hover": "#f9d9df", "undo_text": "#8a2434",
        "clear_bg": "#fff1da", "clear_hover": "#ffe5b8", "clear_text": "#8a5a11",
        "menu_bg": "#ffffff", "list_bg": "#ffffff", "list_hover": "#f3f4f6",
        "slider_bg": "#d0d7de", "label_muted": "#57606a",
        "canvas_bg": "#eef1f5", "canvas_text": "#6e7781",
        "theme_bg": "#fff7d6", "theme_hover": "#ffefad",
        "scroll_groove": "#e8eef3", "scroll_handle": "#c9d1d9",
        "scroll_handle_hover": "#afb8c1", "scroll_handle_pressed": "#1a73e8",
    }


STYLESHEET = """
    QMainWindow, QWidget { background-color: @window_bg@; color: @text@; font-size: 14px; }
    QSplitter::handle:horizontal { background-color: @border@; width: 6px; margin: 2px 0; border-radius: 3px; }
    QSplitter::handle:horizontal:hover { background-color: @selection@; }
    QWidget#bottomPanel QPushButton { font-size: 12px; padding: 5px 10px; }
    QWidget#bottomPanel QPushButton#stage1Button,
    QWidget#bottomPanel QPushButton#stage2Button,
    QWidget#bottomPanel QPushButton#stage3Button,
    QWidget#bottomPanel QPushButton#stage4Button,
    QWidget#bottomPanel QPushButton#stage5Button,
    QWidget#bottomPanel QPushButton#stage6Button { font-size: 11px; padding: 4px 6px; font-weight: 500; }
    QLabel#statusLabel { color: @label_muted@; font-size: 11px; }
    QLineEdit { background-color: @input_bg@; border: 1px solid @border@; border-radius: 8px; padding: 7px 10px; color: @text@; selection-background-color: @selection@; }
    QPushButton { background-color: @button_bg@; border: 1px solid @border@; border-radius: 8px; padding: 7px 14px; color: @text@; }
    QPushButton:hover { background-color: @button_hover@; border-color: @button_border_hover@; }
    QPushButton:pressed { background-color: @button_pressed@; }
    QPushButton#primaryButton { background-color: #1a73e8; border-color: #1a73e8; color: #ffffff; font-weight: 600; }
    QPushButton#primaryButton:hover { background-color: #2b7de9; }
    QPushButton#stage1Button { background-color: @stage1_bg@; color: @stage1_text@; border-color: @stage1_text@; }
    QPushButton#stage1Button:hover { background-color: @stage1_hover@; }
    QPushButton#stage2Button { background-color: @stage2_bg@; color: @stage2_text@; border-color: @stage2_text@; }
    QPushButton#stage2Button:hover { background-color: @stage2_hover@; }
    QPushButton#stage3Button { background-color: @stage3_bg@; color: @stage3_text@; border-color: @stage3_text@; }
    QPushButton#stage3Button:hover { background-color: @stage3_hover@; }
    QPushButton#stage4Button { background-color: @stage4_bg@; color: @stage4_text@; border-color: @stage4_text@; }
    QPushButton#stage4Button:hover { background-color: @stage4_hover@; }
    QPushButton#stage5Button { background-color: @stage5_bg@; color: @stage5_text@; border-color: @stage5_text@; }
    QPushButton#stage5Button:hover { background-color: @stage5_hover@; }
    QPushButton#stage6Button { background-color: @stage6_bg@; color: @stage6_text@; border-color: @stage6_text@; }
    QPushButton#stage6Button:hover { background-color: @stage6_hover@; }
    QPushButton#undoButton { background-color: @undo_bg@; color: @undo_text@; border-color: @undo_text@; font-weight: 600; }
    QPushButton#undoButton:hover { background-color: @undo_hover@; }
    QPushButton#clearButton { background-color: @clear_bg@; color: @clear_text@; border-color: @clear_text@; font-weight: 600; }
    QPushButton#clearButton:hover { background-color: @clear_hover@; }
    QPushButton#speedButton { min-width: 0; max-width: 72px; padding: 4px 0px; border-radius: 10px; font-weight: 500; }
    QPushButton#themeButton { background-color: @theme_bg@; min-width: 34px; max-width: 34px; min-height: 34px; max-height: 34px; padding: 0px; border-radius: 17px; font-size: 18px; border: 1px solid @border@; }
    QPushButton#themeButton:hover { background-color: @theme_hover@; }
    QMenu { background-color: @menu_bg@; color: @text@; border: 1px solid @border@; border-radius: 8px; padding: 4px; }
    QMenu::item { padding: 5px 18px; border-radius: 5px; }
    QMenu::item:selected { background-color: #1a73e8; color: #ffffff; }
    QListWidget { background-color: @list_bg@; border: 1px solid @border@; border-radius: 8px; padding: 4px; color: @text@; }
    QListWidget::item { padding: 6px 8px; border-radius: 5px; }
    QListWidget::item:hover { background-color: @list_hover@; }
    QListWidget::item:selected { background-color: #1a73e8; color: white; }
    QScrollBar:vertical { background: transparent; width: 10px; margin: 4px 2px 4px 0px; }
    QScrollBar::handle:vertical { background: @scroll_handle@; min-height: 28px; border-radius: 5px; margin: 2px 0px; }
    QScrollBar::handle:vertical:hover { background: @scroll_handle_hover@; }
    QScrollBar::handle:vertical:pressed { background: @scroll_handle_pressed@; }
    QScrollBar::groove:vertical { background: @scroll_groove@; border-radius: 5px; width: 10px; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; width: 0px; border: none; background: none; }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
    QScrollBar:horizontal { background: transparent; height: 10px; margin: 0px 4px 2px 4px; }
    QScrollBar::handle:horizontal { background: @scroll_handle@; min-width: 28px; border-radius: 5px; margin: 0px 2px; }
    QScrollBar::handle:horizontal:hover { background: @scroll_handle_hover@; }
    QScrollBar::handle:horizontal:pressed { background: @scroll_handle_pressed@; }
    QScrollBar::groove:horizontal { background: @scroll_groove@; border-radius: 5px; height: 10px; }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { height: 0px; width: 0px; border: none; background: none; }
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: none; }
    QSlider::groove:horizontal { height: 4px; background: @slider_bg@; border-radius: 2px; }
    QSlider::sub-page:horizontal { background: #1a73e8; border-radius: 2px; }
    QSlider::handle:horizontal { width: 10px; height: 10px; margin: -4px 0; border-radius: 5px; background: #1a73e8; border: none; }
    QSlider::handle:horizontal:hover { width: 12px; height: 12px; margin: -5px 0; border-radius: 6px; background: #1557b0; }
    QLabel#controlLabel { color: @label_muted@; }
    QLabel#stageBadge { color: #ffffff; background-color: #2563eb; border-radius: 10px; font-size: 12px; font-weight: 600; padding: 4px 10px; min-width: 88px; max-width: 180px; }
    QLabel#videoCanvas { background-color: @canvas_bg@; color: @canvas_text@; border: 1px solid @border@; border-radius: 10px; }
"""


class StageToolWindow(QMainWindow):
    """Shared UI for stage annotation and review."""

    WINDOW_TITLE = "Stage tool"
    HINT_TEXT = "Keys 1-6 set stages. Delete removes the selected marker. 8 writes parquet."
    UNSAVED_TITLE = "Unsaved changes"
    UNSAVED_TEXT = "This video has unsaved stage changes. Save before switching?"
    SEEK_STEP_SEC = 1.5
    REQUIRE_INITIAL_STAGE = False
    ENABLE_UNDO_CLEAR = False
    ENABLE_CTRL_S = False
    LOAD_STAGES_FROM_PARQUET = True
    CONFIRM_ON_FOLDER_SCAN = False
    CACHE_ON_SET_STAGE = False
    CACHE_ON_VIDEO_SWITCH = False
    AUTO_PLAY_ON_FIRST_STAGE = False
    AUTO_PLAY_ON_RESTORE = False
    OVERLAY_FONT_SIZE = 15
    OVERLAY_PADDING = "6px 10px"
    OVERLAY_RADIUS = 7

    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(1200, 760)
        self.video_paths: list[Path] = []
        self.dataset_root: Path | None = None
        self.current_index = -1
        self.reader: FFmpegVideoReader | None = None
        self.video_readers: dict[str, FFmpegVideoReader] = {}
        self.current_camera_paths: dict[str, Path] = {}
        self.current_video_path: Path | None = None
        self.current_parquet_path: Path | None = None
        self.total_frames = 0
        self.fps = 25.0
        self.playback_speed = 1.0
        self.is_playing = False
        self._frames_read_current_video = 0
        self._slider_dragging = False
        self._was_playing_before_drag = False
        self._last_qimages: dict[str, QImage] = {}
        self._camera_aspect: dict[str, float] = {role: 16 / 9 for role in CAMERA_ROLES}
        self.dark_theme = False
        self.stage_history: list[StageEvent] = []
        self.current_stage: int | None = None
        self.annotation_cache: dict[str, list[StageEvent]] = {}
        self.unsaved_changes = False
        self.awaiting_initial_stage = self.REQUIRE_INITIAL_STAGE
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._read_next_frame)
        self._build_ui()
        self._apply_styles()
        self._install_shortcuts()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout()
        central.setLayout(root)

        top_bar = QHBoxLayout()
        self.theme_btn = QPushButton("🌞")
        self.theme_btn.setObjectName("themeButton")
        self.theme_btn.setFixedSize(34, 34)
        self.theme_btn.clicked.connect(self._toggle_theme)
        top_bar.addWidget(self.theme_btn)
        top_bar.addWidget(QLabel("Folder:"))
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Enter or choose a folder that contains videos")
        self.folder_edit.returnPressed.connect(self._scan_folder)
        top_bar.addWidget(self.folder_edit, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._choose_folder)
        top_bar.addWidget(browse_btn)
        search_btn = QPushButton("Scan")
        search_btn.clicked.connect(self._scan_folder)
        top_bar.addWidget(search_btn)
        root.addLayout(top_bar)

        main = QSplitter(Qt.Horizontal)
        main.setChildrenCollapsible(False)
        main.setHandleWidth(6)
        root.addWidget(main, 1)

        left_panel = QWidget()
        left = QVBoxLayout(left_panel)
        left.setContentsMargins(0, 0, 0, 0)
        left.addWidget(QLabel("Videos:"))
        self.file_list = QListWidget()
        self.file_list.setMinimumWidth(160)
        self.file_list.currentRowChanged.connect(self._on_video_selected)
        left.addWidget(self.file_list, 1)
        main.addWidget(left_panel)

        right_panel = QWidget()
        right = QVBoxLayout(right_panel)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(4)
        self.camera_panel = CameraPanelWidget()
        self.camera_layout = QVBoxLayout(self.camera_panel)
        self.camera_layout.setContentsMargins(0, 0, 0, 0)
        self.camera_layout.setSpacing(4)
        self.global_view = GlobalVideoWidget()
        self.global_view.setMinimumHeight(90)
        self.video_label = self.global_view.video_label
        self.stage_overlay = self.global_view.stage_overlay
        self.camera_labels: dict[str, QLabel] = {"global": self.video_label}
        self.camera_layout.addWidget(self.global_view)
        self.wrist_panel = QWidget()
        wrist_row = QHBoxLayout(self.wrist_panel)
        wrist_row.setSpacing(4)
        wrist_row.setContentsMargins(0, 0, 0, 0)
        for role in ("left_wrist", "right_wrist"):
            label = QLabel(CAMERA_TITLES[role])
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumHeight(72)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label.setObjectName("videoCanvas")
            self.camera_labels[role] = label
            wrist_row.addWidget(label, 1)
        self.camera_layout.addWidget(self.wrist_panel)
        self.camera_layout.addStretch(1)
        self.camera_panel.layout_updated.connect(self._on_camera_panel_resized)
        right.addWidget(self.camera_panel, 1)
        main.addWidget(right_panel)
        main.setStretchFactor(0, 0)
        main.setStretchFactor(1, 1)
        main.setSizes([240, 960])
        self.main_splitter = main
        main.splitterMoved.connect(self._on_camera_panel_resized)

        bottom_panel = QWidget()
        bottom_panel.setObjectName("bottomPanel")
        bottom = QVBoxLayout(bottom_panel)
        bottom.setContentsMargins(0, 6, 0, 0)
        bottom.setSpacing(4)

        stage_row = QHBoxLayout()
        stage_row.setSpacing(6)
        self.stage_badge = QLabel("Stage: -")
        self.stage_badge.setObjectName("stageBadge")
        self.stage_badge.setAlignment(Qt.AlignCenter)
        stage_row.addWidget(self.stage_badge)
        self.stage_buttons = {}
        for stage, label in STAGE_LABELS.items():
            btn = QPushButton(f"{stage}. {label}")
            btn.setObjectName(f"stage{stage}Button")
            btn.clicked.connect(lambda _checked=False, s=stage: self._set_stage(s))
            self.stage_buttons[stage] = btn
            stage_row.addWidget(btn, 1)
        bottom.addLayout(stage_row)

        control = QHBoxLayout()
        control.setSpacing(8)
        self.play_btn = QPushButton(PLAY_TEXT)
        self.play_btn.setObjectName("primaryButton")
        self.play_btn.clicked.connect(self._toggle_play)
        control.addWidget(self.play_btn)
        self.skip_btn = QPushButton("Skip")
        self.skip_btn.clicked.connect(self._next_video)
        control.addWidget(self.skip_btn)
        self.progress = AnnotationSlider(Qt.Horizontal)
        self.progress.setRange(0, 0)
        self.progress.sliderPressed.connect(self._on_slider_pressed)
        self.progress.sliderMoved.connect(self._on_slider_moved)
        self.progress.sliderReleased.connect(self._on_slider_released)
        self.progress.marker_clicked.connect(self._on_marker_clicked)
        control.addWidget(self.progress, 1)
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setMinimumWidth(96)
        self.time_label.setAlignment(Qt.AlignCenter)
        control.addWidget(self.time_label)
        speed_label = QLabel("Speed")
        speed_label.setObjectName("controlLabel")
        control.addWidget(speed_label)
        self.speed_button = QPushButton("1x")
        self.speed_button.setObjectName("speedButton")
        self.speed_button.setFixedWidth(56)
        self.speed_menu = QMenu(self)
        for speed in (0.5, 1.0, 1.5, 2.0, 3.0):
            action = self.speed_menu.addAction(f"{speed:g}x")
            action.triggered.connect(lambda _checked=False, s=speed: self._set_speed(s))
        self.speed_button.clicked.connect(self._show_speed_menu)
        control.addWidget(self.speed_button)
        bottom.addLayout(control)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        if self.ENABLE_UNDO_CLEAR:
            self.undo_btn = QPushButton("Undo (7)")
            self.undo_btn.setObjectName("undoButton")
            self.undo_btn.clicked.connect(self._undo_last_annotation)
            action_row.addWidget(self.undo_btn)
        self.delete_marker_btn = QPushButton("Delete selected (Del)")
        self.delete_marker_btn.setObjectName("clearButton")
        self.delete_marker_btn.clicked.connect(self._delete_selected_marker)
        action_row.addWidget(self.delete_marker_btn)
        self.finish_annotation_btn = QPushButton("Write parquet (8)")
        self.finish_annotation_btn.setObjectName("primaryButton")
        self.finish_annotation_btn.clicked.connect(lambda: self._finish_annotation())
        action_row.addWidget(self.finish_annotation_btn)
        if self.ENABLE_UNDO_CLEAR:
            self.reset_marker_btn = QPushButton("Clear (9)")
            self.reset_marker_btn.setObjectName("clearButton")
            self.reset_marker_btn.clicked.connect(self._clear_annotations)
            action_row.addWidget(self.reset_marker_btn)
        self.status_label = QLabel(self.HINT_TEXT)
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        action_row.addWidget(self.status_label, 1)
        bottom.addLayout(action_row)
        root.addWidget(bottom_panel)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_camera_layout_heights()
        self._refresh_video_frames_display()

    def _on_camera_panel_resized(self):
        self._update_camera_layout_heights()
        self._refresh_video_frames_display()

    def _update_camera_layout_heights(self):
        panel = getattr(self, "camera_panel", None)
        if panel is None:
            return
        width = max(1, panel.width())
        height = max(1, panel.height())
        spacing = self.camera_layout.spacing()
        global_ratio = max(1e-3, self._camera_aspect.get("global", 16 / 9))
        wrist_ratio = max(
            1e-3,
            self._camera_aspect.get("left_wrist", 16 / 9),
            self._camera_aspect.get("right_wrist", 16 / 9),
        )
        global_h = width / global_ratio
        wrist_h = (width / 2) / wrist_ratio
        if global_h + wrist_h + spacing > height:
            scale = max(0.05, (height - spacing) / (global_h + wrist_h))
            global_h *= scale
            wrist_h *= scale
        self.global_view.setFixedHeight(max(72, int(global_h)))
        self.wrist_panel.setFixedHeight(max(72, int(wrist_h)))

    def _refresh_video_frames_display(self):
        for role, qimg in self._last_qimages.items():
            self._show_qimage(role, qimg)

    def _apply_styles(self):
        if hasattr(self, "progress"):
            self.progress.set_dark_theme(self.dark_theme)
        colors = theme_colors(self.dark_theme)
        self.theme_btn.setText("🌙" if self.dark_theme else "🌞")
        style = STYLESHEET
        for key, value in colors.items():
            style = style.replace(f"@{key}@", value)
        self.setStyleSheet(style)

    def _toggle_theme(self):
        self.dark_theme = not self.dark_theme
        self._apply_styles()

    def _install_shortcuts(self):
        bindings = [
            (Qt.Key_Up, self._previous_video),
            (Qt.Key_Down, self._next_video),
            (Qt.Key_Left, self._seek_backward),
            (Qt.Key_Right, self._seek_forward),
            (Qt.Key_Space, self._toggle_play),
            (Qt.Key_1, lambda: self._set_stage(1)),
            (Qt.Key_2, lambda: self._set_stage(2)),
            (Qt.Key_3, lambda: self._set_stage(3)),
            (Qt.Key_4, lambda: self._set_stage(4)),
            (Qt.Key_5, lambda: self._set_stage(5)),
            (Qt.Key_6, lambda: self._set_stage(6)),
            (Qt.Key_8, self._finish_annotation),
            (Qt.Key_Delete, self._delete_selected_marker),
            (Qt.Key_Backspace, self._delete_selected_marker),
        ]
        if self.ENABLE_UNDO_CLEAR:
            bindings.extend([
                (Qt.Key_7, self._undo_last_annotation),
                (Qt.Key_9, self._clear_annotations),
            ])
        for key, slot in bindings:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)
        if self.ENABLE_CTRL_S:
            save_sc = QShortcut(QKeySequence("Ctrl+S"), self)
            save_sc.setContext(Qt.WidgetWithChildrenShortcut)
            save_sc.activated.connect(self._finish_annotation)

    def _choose_folder(self):
        start_dir = self.folder_edit.text().strip() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select video folder", start_dir)
        if folder:
            self.folder_edit.setText(folder)
            self._scan_folder()

    def _scan_folder(self):
        folder = Path(os.path.expanduser(self.folder_edit.text().strip()))
        if not folder.exists() or not folder.is_dir():
            QMessageBox.warning(self, "Error", f"Folder does not exist:\n{folder}")
            return
        if self.CONFIRM_ON_FOLDER_SCAN and not self._confirm_discard_unsaved_changes():
            return
        self._close_video()
        self.dataset_root = resolve_dataset_root(folder)
        self.current_index = -1
        self.video_paths = find_videos_recursively(folder)
        self.file_list.clear()
        for path in self.video_paths:
            rel_path = path.relative_to(folder)
            item = QListWidgetItem(f"{rel_path}    [{path.suffix.lower()}]")
            item.setToolTip(str(path))
            self.file_list.addItem(item)
        if not self.video_paths:
            self.video_label.setText("No video files found")
            self.status_label.setText(f"No videos in {folder}")
            return
        self.status_label.setText(f"Found {len(self.video_paths)} videos")
        self.file_list.setCurrentRow(0)

    def _on_video_selected(self, row: int):
        if row < 0 or row >= len(self.video_paths) or row == self.current_index:
            return
        if not self._confirm_discard_unsaved_changes():
            self.file_list.blockSignals(True)
            self.file_list.setCurrentRow(self.current_index)
            self.file_list.blockSignals(False)
            return
        if self.CACHE_ON_VIDEO_SWITCH and self.current_video_path is not None:
            self._cache_current_annotations()
        self._open_video(row)

    def _open_video(self, index: int):
        self._close_video()
        self.current_index = index
        path = self.video_paths[index]
        self.current_video_path = path
        self.current_camera_paths = find_camera_videos_for_episode(path, self.dataset_root)
        self.current_parquet_path = find_parquet_for_video(path, self.dataset_root)
        try:
            self.video_readers = {
                role: FFmpegVideoReader(video_path)
                for role, video_path in self.current_camera_paths.items()
            }
            if "global" not in self.video_readers:
                self.video_readers["global"] = FFmpegVideoReader(path)
                self.current_camera_paths["global"] = path
            self.reader = self.video_readers["global"]
            for reader in self.video_readers.values():
                reader.start(0)
        except Exception as exc:
            self._close_camera_readers()
            self.reader = None
            self._show_video_message(
                f"Cannot open video: {path.name}\n\n{exc}\n\n"
                "Confirm that system ffmpeg supports this codec."
            )
            self.status_label.setText(f"Cannot open: {path.name}")
            return

        self.total_frames = self.reader.total_frames
        self.fps = self.reader.fps
        self.progress.set_fps(self.fps)
        self._frames_read_current_video = 0
        self.progress.setRange(0, max(0, self.total_frames - 1))
        self._restore_annotations(path)
        self._seek_to_frame(0)
        self._after_open_video(path)

    def _restore_annotations(self, path: Path):
        key = self._annotation_cache_key(path)
        if key is not None and key in self.annotation_cache:
            self.stage_history = list(self.annotation_cache[key])
        elif self.LOAD_STAGES_FROM_PARQUET:
            self.stage_history = load_stage_events_from_parquet(
                self.current_parquet_path, self.total_frames
            )
        else:
            self.stage_history = []
        self.current_stage = self.stage_history[-1].stage if self.stage_history else None
        self.awaiting_initial_stage = self.REQUIRE_INITIAL_STAGE and not bool(self.stage_history)
        self.unsaved_changes = False
        self._refresh_annotation_slider()

    def _after_open_video(self, path: Path):
        stage_count = len(self._canonical_stage_events())
        parquet_s = self.current_parquet_path.name if self.current_parquet_path else "no matching parquet"
        if self.AUTO_PLAY_ON_RESTORE and stage_count > 0:
            self.awaiting_initial_stage = False
            self._set_playing(True)
            self._read_next_frame()
            self.status_label.setText(
                f"Playing: {path.name} | {parquet_s} | restored {stage_count} stage events"
            )
        elif self.REQUIRE_INITIAL_STAGE and stage_count == 0:
            self.status_label.setText(
                f"Opened: {path.name} | {parquet_s} | press 1-6 to set the initial stage"
            )
        elif stage_count > 0:
            self.status_label.setText(
                f"Playing: {path.name} | {parquet_s} | loaded {stage_count} stage events"
            )
        else:
            self.status_label.setText(
                f"Playing: {path.name} | {parquet_s} | no stages yet; press 1-6 to add"
            )

    def _close_video(self):
        self.timer.stop()
        self.is_playing = False
        if hasattr(self, "play_btn"):
            self.play_btn.setText(PLAY_TEXT)
        self._close_camera_readers()
        self.reader = None
        self.total_frames = 0
        self._frames_read_current_video = 0
        self._last_qimages = {}
        if hasattr(self, "progress"):
            self.progress.setRange(0, 0)
            self.progress.blockSignals(True)
            self.progress.setValue(0)
            self.progress.blockSignals(False)
            self._refresh_annotation_slider()
        self.current_video_path = None
        self.current_camera_paths = {}
        self.current_parquet_path = None
        if hasattr(self, "time_label"):
            self.time_label.setText("00:00 / 00:00")
        if hasattr(self, "stage_badge"):
            self._set_stage_badge_style(None)
            self.stage_badge.setText("Stage: -")
            self.stage_overlay.setText("-")
        self.stage_history = []
        self.current_stage = None
        self.awaiting_initial_stage = self.REQUIRE_INITIAL_STAGE
        self.unsaved_changes = False

    def _close_camera_readers(self):
        for reader in self.video_readers.values():
            reader.close()
        self.video_readers = {}

    def _annotation_cache_key(self, path: Path | None = None) -> str | None:
        path = path or self.current_video_path
        return str(path) if path is not None else None

    def _canonical_stage_events(self, history: list[StageEvent] | None = None) -> list[StageEvent]:
        return canonical_stage_events(self.stage_history if history is None else history)

    def _cache_current_annotations(self):
        key = self._annotation_cache_key()
        if key is not None:
            self.annotation_cache[key] = list(self.stage_history)

    def _refresh_annotation_slider(self):
        events = self._canonical_stage_events()
        current_frame = int(self.progress.value()) if hasattr(self, "progress") else 0
        selected_marker = 0 if events else None
        current_stage = events[0].stage if events else None
        for idx, event in enumerate(events):
            if event.frame_idx <= current_frame:
                selected_marker = idx
                current_stage = event.stage
            else:
                break
        self.current_stage = current_stage
        self.progress.set_markers(
            [event.frame_idx for event in events],
            [event.stage for event in events],
            selected_marker,
        )
        self._update_stage_badge(current_stage)

    def _update_stage_badge(self, stage: int | None = None):
        if stage is None:
            stage = self._current_stage_at_frame(int(self.progress.value()))
        if stage is None:
            self._set_stage_badge_style(None)
            self.stage_badge.setText("Stage: -")
            self.stage_overlay.setText("-")
            return
        self._set_stage_badge_style(stage)
        label = STAGE_LABELS.get(stage, "unknown")
        self.stage_badge.setText(f"Stage {stage}: {label}")
        self.stage_overlay.setText(f"Stage {stage}\n{label}")

    def _set_stage_badge_style(self, stage: int | None):
        bg, fg = STAGE_BADGE_COLORS.get(stage, ("#2563eb", "#ffffff"))
        self.stage_badge.setStyleSheet(
            f"QLabel#stageBadge {{ color: {fg}; background-color: {bg}; border-radius: 10px; "
            f"font-size: 12px; font-weight: 600; padding: 4px 10px; min-width: 88px; max-width: 180px; }}"
        )
        self.stage_overlay.setStyleSheet(
            f"QLabel#stageOverlay {{ color: {fg}; background-color: {bg}; "
            f"border-radius: {self.OVERLAY_RADIUS}px; font-size: {self.OVERLAY_FONT_SIZE}px; "
            f"font-weight: 800; line-height: 120%; padding: {self.OVERLAY_PADDING}; }}"
        )
        self.global_view._position_overlay()

    def _start_timer(self):
        interval_ms = max(1, int(1000 / (max(self.fps, 1.0) * max(self.playback_speed, 0.1))))
        self.timer.start(interval_ms)

    def _set_playing(self, playing: bool):
        self.is_playing = playing
        self.play_btn.setText(PAUSE_TEXT if playing else PLAY_TEXT)
        if playing:
            self._start_timer()
        else:
            self.timer.stop()

    def _show_speed_menu(self):
        self.speed_menu.popup(self.speed_button.mapToGlobal(self.speed_button.rect().bottomLeft()))

    def _set_speed(self, speed: float):
        self.playback_speed = float(speed)
        self.speed_button.setText(f"{self.playback_speed:g}x")
        if self.is_playing and self.reader is not None:
            self._start_timer()
        self.status_label.setText(f"Playback speed set to {self.playback_speed:g}x")

    def _toggle_play(self):
        if self.reader is None:
            return
        if self.REQUIRE_INITIAL_STAGE and self.awaiting_initial_stage and not self._canonical_stage_events():
            self.status_label.setText("Press 1-6 to set the initial stage before playing")
            return
        self._set_playing(not self.is_playing)

    def _read_next_frame(self):
        if self.reader is None or self._slider_dragging:
            return
        ok, frame_idx, frame_rgb = self.reader.read_frame()
        if not ok:
            self._handle_read_failure()
            return
        self._frames_read_current_video += 1
        self._show_rgb_frames(self._collect_role_frames(frame_rgb))
        self.progress.blockSignals(True)
        self.progress.setValue(max(0, frame_idx))
        self.progress.blockSignals(False)
        self._update_time_label(max(0, frame_idx))
        self._refresh_annotation_slider()

    def _collect_role_frames(self, global_rgb: bytes | None) -> dict[str, bytes]:
        frames: dict[str, bytes] = {}
        if global_rgb is not None:
            frames["global"] = global_rgb
        for role in ("left_wrist", "right_wrist"):
            reader = self.video_readers.get(role)
            if reader is None:
                continue
            ok_role, _idx, role_rgb = reader.read_frame()
            if ok_role and role_rgb is not None:
                frames[role] = role_rgb
        return frames

    def _show_rgb_frames(self, frames: dict[str, bytes]):
        for role, frame_rgb in frames.items():
            reader = self.video_readers.get(role)
            label = self.camera_labels.get(role)
            if reader is None or label is None:
                continue
            qimg = QImage(
                frame_rgb, reader.width, reader.height, reader.width * 3, QImage.Format_RGB888
            ).copy()
            self._last_qimages[role] = qimg
            self._show_qimage(role, qimg)
        for role in CAMERA_ROLES:
            if role not in self.video_readers and role in self.camera_labels:
                self.camera_labels[role].setText(f"{CAMERA_TITLES[role]} missing")

    def _show_qimage(self, role: str, qimg: QImage):
        label = self.camera_labels.get(role)
        if label is None:
            return
        if qimg.width() > 0 and qimg.height() > 0:
            new_ratio = qimg.width() / qimg.height()
            old_ratio = self._camera_aspect.get(role, 0.0)
            if abs(old_ratio - new_ratio) > 1e-4:
                self._camera_aspect[role] = new_ratio
                self._update_camera_layout_heights()
        label.setPixmap(fit_pixmap_to_label(qimg, label))

    def _show_video_message(self, text: str):
        self._last_qimages = {}
        for role, label in self.camera_labels.items():
            label.setPixmap(QPixmap())
            label.setText(text if role == "global" else CAMERA_TITLES[role])

    def _stop_playback(self):
        self._set_playing(False)

    def _handle_read_failure(self):
        pos = self.reader.current_frame if self.reader is not None else 0
        reached_end = self.total_frames > 0 and pos >= self.total_frames - 1
        self._stop_playback()
        if reached_end and self._frames_read_current_video > 0:
            self.status_label.setText("Playback finished")
            return
        path = self.video_paths[self.current_index] if 0 <= self.current_index < len(self.video_paths) else None
        name = path.name if path is not None else "current video"
        self._show_video_message(
            f"Cannot decode video: {name}\n\n"
            "Possible causes:\n"
            "1. System ffmpeg does not support this codec (e.g. missing AV1)\n"
            "2. The file is corrupt or missing headers\n\n"
            "Press Down or Skip to open the next video.\n"
            "To play this file, transcode to H.264:\n"
            "ffmpeg -i input -c:v libx264 -pix_fmt yuv420p output.mp4"
        )
        self.status_label.setText(f"Decode failed: {name}")

    def _on_slider_pressed(self):
        self._was_playing_before_drag = self.is_playing
        self._slider_dragging = True
        self.timer.stop()

    def _on_slider_moved(self, frame_idx: int):
        self._update_time_label(frame_idx)
        self._refresh_annotation_slider()

    def _on_slider_released(self):
        try:
            if self.reader is None:
                return
            self._seek_to_frame(self.progress.value())
        finally:
            self._slider_dragging = False
            self._set_playing(self._was_playing_before_drag)

    def _seek_to_frame(self, frame_idx: int):
        if self.reader is None:
            return
        frame_idx = max(0, min(frame_idx, max(0, self.total_frames - 1)))
        for reader in self.video_readers.values():
            reader.start(frame_idx)
        ok, shown_idx, frame_rgb = self.reader.read_frame()
        if not ok and frame_idx > 0:
            for reader in self.video_readers.values():
                reader.start(frame_idx - 1)
            ok, shown_idx, frame_rgb = self.reader.read_frame()
        if not ok:
            self.status_label.setText("Seek failed; try another position or video")
            return
        self._show_rgb_frames(self._collect_role_frames(frame_rgb))
        self.progress.blockSignals(True)
        self.progress.setValue(shown_idx)
        self.progress.blockSignals(False)
        self._update_time_label(shown_idx)
        self._refresh_annotation_slider()

    def _current_stage_at_frame(self, frame_idx: int) -> int | None:
        events = self._canonical_stage_events()
        if not events:
            return None
        stage = events[0].stage
        for event in events:
            if event.frame_idx <= frame_idx:
                stage = event.stage
            else:
                break
        return stage

    def _set_stage(self, stage: int):
        if isinstance(QApplication.focusWidget(), QLineEdit):
            return
        if stage not in STAGE_LABELS:
            self.status_label.setText(f"Invalid stage: {stage}")
            return
        if self.reader is None or self.total_frames <= 1:
            return
        frame_idx = max(0, min(int(self.progress.value()), max(0, self.total_frames - 1)))
        current_stage = self._current_stage_at_frame(frame_idx)
        if self.stage_history and current_stage == stage:
            self.status_label.setText(
                f"Already stage {stage} ({STAGE_LABELS[stage]}); no change"
            )
            return
        self.stage_history.append(StageEvent(frame_idx=frame_idx, stage=stage))
        self.current_stage = stage
        self.awaiting_initial_stage = False
        self.unsaved_changes = True
        if self.CACHE_ON_SET_STAGE:
            self._cache_current_annotations()
        self._refresh_annotation_slider()
        if self.AUTO_PLAY_ON_FIRST_STAGE and len(self._canonical_stage_events()) == 1 and not self.is_playing:
            self._set_playing(True)
        self.status_label.setText(
            f"Set stage {stage} ({STAGE_LABELS[stage]}): {format_time(frame_idx / max(self.fps, 1.0))}"
        )

    def _undo_last_annotation(self):
        if isinstance(QApplication.focusWidget(), QLineEdit):
            return
        if not self.stage_history:
            self.status_label.setText("Nothing to undo")
            return
        removed = self.stage_history.pop()
        events = self._canonical_stage_events()
        self.current_stage = self._current_stage_at_frame(int(self.progress.value()))
        self.awaiting_initial_stage = self.REQUIRE_INITIAL_STAGE and not bool(events)
        self.unsaved_changes = True
        self._refresh_annotation_slider()
        if not events:
            self._stop_playback()
            self.status_label.setText("Undid the last stage event; set an initial stage again")
        else:
            self.status_label.setText(
                f"Undid stage {removed.stage} ({STAGE_LABELS.get(removed.stage, 'unknown')}): "
                f"{format_time(removed.frame_idx / max(self.fps, 1.0))}"
            )

    def _clear_annotations(self):
        if isinstance(QApplication.focusWidget(), QLineEdit) or self.reader is None:
            return
        removed_count = len(self._canonical_stage_events())
        self.stage_history = []
        self.current_stage = None
        self.awaiting_initial_stage = self.REQUIRE_INITIAL_STAGE
        self.unsaved_changes = False
        self._cache_current_annotations()
        self._refresh_annotation_slider()
        self._seek_to_frame(0)
        self._stop_playback()
        if removed_count == 0:
            self.status_label.setText("No stage events; you can start over")
        else:
            self.status_label.setText(f"Cleared {removed_count} stage events; you can start over")

    def _delete_selected_marker(self):
        if isinstance(QApplication.focusWidget(), QLineEdit) or self.reader is None:
            return
        events = self._canonical_stage_events()
        selected_idx = self.progress.selected_marker
        if selected_idx is None or selected_idx < 0 or selected_idx >= len(events):
            self.status_label.setText(
                "Click a timeline marker, or play to that stage, then press Delete"
            )
            return
        event = events[selected_idx]
        if event.frame_idx == 0:
            self.status_label.setText(
                "Cannot delete the stage at frame 0; press 1-6 there to change it"
            )
            return
        self.stage_history = [item for item in self.stage_history if item.frame_idx != event.frame_idx]
        events = self._canonical_stage_events()
        self.current_stage = self._current_stage_at_frame(int(self.progress.value()))
        self.awaiting_initial_stage = self.REQUIRE_INITIAL_STAGE and not bool(events)
        self.unsaved_changes = True
        self._cache_current_annotations()
        self._refresh_annotation_slider()
        if not events:
            self._stop_playback()
            self.status_label.setText("Deleted all stage events; set an initial stage again")
        else:
            self.status_label.setText(
                f"Deleted stage {event.stage} ({STAGE_LABELS.get(event.stage, 'unknown')}) "
                f"@ {format_time(event.frame_idx / max(self.fps, 1.0))}"
            )

    def _on_marker_clicked(self, marker_idx: int):
        events = self._canonical_stage_events()
        if marker_idx < 0 or marker_idx >= len(events):
            return
        event = events[marker_idx]
        was_playing = self.is_playing
        self.timer.stop()
        self._seek_to_frame(event.frame_idx)
        self._set_playing(was_playing)
        self.progress.set_markers(
            [item.frame_idx for item in events],
            [item.stage for item in events],
            marker_idx,
        )
        self._update_stage_badge()
        self.status_label.setText(
            f"Selected stage {event.stage} ({STAGE_LABELS.get(event.stage, 'unknown')}) "
            f"@ {format_time(event.frame_idx / max(self.fps, 1.0))}; press Delete to remove"
        )

    def _finish_annotation(self, silent: bool = False) -> bool:
        if isinstance(QApplication.focusWidget(), QLineEdit):
            return False
        if self.current_parquet_path is None:
            if not silent:
                QMessageBox.warning(self, "Cannot write", "No parquet file matched this video.")
            return False
        if self.total_frames <= 1:
            if not silent:
                QMessageBox.warning(self, "Cannot write", "Invalid frame count; cannot compute stage_id_gt.")
            return False
        events = self._canonical_stage_events()
        if not events:
            if not silent:
                QMessageBox.warning(self, "Cannot write", "No stage events to save.")
            return False
        try:
            count = write_stage_id_to_parquet(self.current_parquet_path, events, self.total_frames)
        except Exception as exc:
            if not silent:
                QMessageBox.warning(self, "Write failed", f"{self.current_parquet_path}\n\n{exc}")
            return False
        self.unsaved_changes = False
        self._cache_current_annotations()
        if not silent:
            QMessageBox.information(
                self,
                "Write complete",
                f"Wrote {STAGE_COLUMN}\n\nparquet: {self.current_parquet_path}\n"
                f"rows: {count}\nstages: {len(events)}",
            )
        self.status_label.setText(f"Wrote {self.current_parquet_path.name} | {len(events)} stages")
        return True

    def _confirm_discard_unsaved_changes(self) -> bool:
        if not self.unsaved_changes:
            return True
        choice = QMessageBox.question(
            self,
            self.UNSAVED_TITLE,
            self.UNSAVED_TEXT,
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if choice == QMessageBox.Cancel:
            return False
        if choice == QMessageBox.Save:
            return self._finish_annotation(silent=True)
        self.unsaved_changes = False
        return True

    def _seek_relative_seconds(self, seconds: float):
        if self.reader is None:
            return
        was_playing = self.is_playing
        self.timer.stop()
        offset_frames = int(round(seconds * max(self.fps, 1.0)))
        self._seek_to_frame(self.progress.value() + offset_frames)
        self._set_playing(was_playing)

    def _seek_backward(self):
        self._seek_relative_seconds(-self.SEEK_STEP_SEC)

    def _seek_forward(self):
        self._seek_relative_seconds(self.SEEK_STEP_SEC)

    def _update_time_label(self, frame_idx: int):
        cur_s = frame_idx / max(self.fps, 1.0)
        total_s = self.total_frames / max(self.fps, 1.0) if self.total_frames else 0.0
        self.time_label.setText(f"{format_time(cur_s)} / {format_time(total_s)}")

    def _previous_video(self):
        if self.video_paths:
            self.file_list.setCurrentRow(max(0, self.current_index - 1))

    def _next_video(self):
        if self.video_paths:
            self.file_list.setCurrentRow(min(len(self.video_paths) - 1, self.current_index + 1))

    def closeEvent(self, event):
        if not self._confirm_discard_unsaved_changes():
            event.ignore()
            return
        self._close_video()
        event.accept()


class StageAnnotateUI(StageToolWindow):
    WINDOW_TITLE = "Stage annotation"
    HINT_TEXT = (
        "1-6 set stages; click or play to a marker then Delete; "
        "7 undo, 8 write parquet, 9 clear."
    )
    UNSAVED_TEXT = "This video has unsaved stage annotations. Save before switching?"
    SEEK_STEP_SEC = 1.5
    REQUIRE_INITIAL_STAGE = True
    ENABLE_UNDO_CLEAR = True
    CONFIRM_ON_FOLDER_SCAN = True
    AUTO_PLAY_ON_FIRST_STAGE = True
    AUTO_PLAY_ON_RESTORE = True
    OVERLAY_FONT_SIZE = 30
    OVERLAY_PADDING = "12px 20px"
    OVERLAY_RADIUS = 14


class StageReviewUI(StageToolWindow):
    WINDOW_TITLE = "Stage review"
    HINT_TEXT = (
        "Review mode: 1-6 set stages; click or play to a marker then Delete; "
        "8 or Ctrl+S writes parquet."
    )
    UNSAVED_TEXT = "This video has unsaved stage edits. Save before switching?"
    SEEK_STEP_SEC = 3.0
    ENABLE_CTRL_S = True
    LOAD_STAGES_FROM_PARQUET = True
    CACHE_ON_SET_STAGE = True
    CACHE_ON_VIDEO_SWITCH = True


MODE_WINDOWS = {
    "annotate": StageAnnotateUI,
    "review": StageReviewUI,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate or review LeRobot stage_id_gt labels.")
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_WINDOWS),
        default="annotate",
        help="annotate: label with undo/clear (also loads existing parquet labels). review: load existing parquet labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    ui = MODE_WINDOWS[args.mode]()
    ui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()