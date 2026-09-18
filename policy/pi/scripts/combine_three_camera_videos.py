#!/usr/bin/env python3
"""Horizontally stitch LeRobot triple-camera episode videos (left | global | right).

The global (middle) pane is annotated with the current manipulation pattern,
read from each episode parquet's ``stage_id_gt`` column.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

CAMERA_KEYS = ("images.left_wrist", "images.global", "images.right_wrist")
GLOBAL_INPUT_INDEX = CAMERA_KEYS.index("images.global")
DEFAULT_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
STAGE_COLUMN = "stage_id_gt"

# Display names match the paper figure; stage_id mapping matches stage_predict/README.md.
STAGE_ID_TO_PATTERN = {
    1: "Free move",
    2: "Pre-contact",
    3: "Pre-contact",
    4: "Dexterous contact",
    5: "Dexterous contact",
    6: "Transport contact",
}

# Pattern-name RGB sampled from the paper figure (prefix stays white).
PATTERN_RGB = {
    "Free move": (139, 177, 111),
    "Pre-contact": (238, 191, 31),
    "Transport contact": (100, 152, 193),
    "Dexterous contact": (168, 120, 196),
}


@dataclass(frozen=True)
class CombineJob:
    chunk_dir: Path
    input_paths: list[Path]
    output_path: Path
    parquet_path: Path
    fps: float
    episode_index: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine three camera MP4s per episode into one left-to-right video "
            "(left wrist | global | right wrist). Overlay 'Manipulation pattern: ...' "
            "on the global pane from episode parquet stage_id_gt. "
            "Output: <chunk-dir>/images.combined/episode_XXXXXX.mp4"
        )
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help=(
            "LeRobot dataset root, or a videos/chunk-XXX directory. "
            "Example: /dataset/.../peg_in_hole_color_v2_stage_rendered"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Comma-separated episode indices to combine, e.g. 32,65. Default: all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing images.combined episode videos.",
    )
    parser.add_argument(
        "--stage-column",
        default=STAGE_COLUMN,
        help=f"Parquet column with per-frame stage ids (default: {STAGE_COLUMN}).",
    )
    parser.add_argument(
        "--fontfile",
        type=Path,
        default=DEFAULT_FONT,
        help="TTF used for the global-pane overlay.",
    )
    return parser


def parse_episode_filter(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    values = [part.strip() for part in raw.replace(" ", ",").split(",") if part.strip()]
    return {int(part) for part in values}


def episode_index_from_name(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def ensure_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg not found in PATH. Install ffmpeg first.")
    return ffmpeg


def resolve_dataset_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if (root / "meta" / "info.json").is_file():
        return root
    if root.name.startswith("chunk-") and root.parent.name == "videos":
        candidate = root.parent.parent
        if (candidate / "meta" / "info.json").is_file():
            return candidate
    for parent in root.parents:
        if (parent / "meta" / "info.json").is_file():
            return parent
    raise FileNotFoundError(f"could not find LeRobot meta/info.json above {root}")


def load_dataset_info(dataset_root: Path) -> tuple[str, int, float]:
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    data_path = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    chunks_size = int(info.get("chunks_size", 1000))
    fps = float(info.get("fps", 30))
    return data_path, chunks_size, fps


def parquet_path_for_episode(dataset_root: Path, data_path: str, chunks_size: int, episode_index: int) -> Path:
    return dataset_root / data_path.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )


def resolve_chunk_dirs(dataset_root: Path) -> list[Path]:
    root = dataset_root.expanduser().resolve()
    if (root / "images.global").is_dir():
        return [root]

    videos_dir = root / "videos"
    if videos_dir.is_dir():
        chunks = sorted(p for p in videos_dir.iterdir() if p.is_dir() and p.name.startswith("chunk-"))
        if chunks:
            return chunks

    raise FileNotFoundError(
        f"could not find LeRobot video chunks under {root} "
        f"(expected videos/chunk-*/images.global or images.global directly)"
    )


def stage_segments(stage_ids: list[int], fps: float) -> list[tuple[float, float, str]]:
    if not stage_ids:
        return []
    segments: list[tuple[float, float, str]] = []
    start = 0
    for i in range(1, len(stage_ids) + 1):
        if i == len(stage_ids) or stage_ids[i] != stage_ids[start]:
            stage_id = int(stage_ids[start])
            label = STAGE_ID_TO_PATTERN.get(stage_id, f"Stage {stage_id}")
            segments.append((start / fps, i / fps, label))
            start = i
    return segments


def ass_timestamp(seconds: float) -> str:
    centiseconds = int(round(max(0.0, seconds) * 100.0))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, cs = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def ass_color_bgr(rgb: tuple[int, int, int]) -> str:
    red, green, blue = rgb
    return f"&H{blue:02X}{green:02X}{red:02X}&"


def build_ass_document(segments: list[tuple[float, float, str]], play_res: tuple[int, int] = (640, 480)) -> str:
    width, height = play_res
    events: list[str] = []
    for start_s, end_s, label in segments:
        rgb = PATTERN_RGB.get(label, (255, 255, 255))
        text = (
            r"{\c&HFFFFFF&}Manipulation pattern: "
            rf"{{\c{ass_color_bgr(rgb)}}}{label}"
        )
        events.append(
            f"Dialogue: 0,{ass_timestamp(start_s)},{ass_timestamp(end_s)},Default,,0,0,0,,{text}"
        )
    body = "\n".join(events)
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,28,&H00FFFFFF,&H000000FF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,3,8,0,8,10,10,16,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
{body}
"""


def escape_filter_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def build_filter_complex(ass_path: Path, fontfile: Path) -> str:
    ass = escape_filter_path(ass_path)
    fontsdir = escape_filter_path(fontfile.parent)
    return (
        f"[{GLOBAL_INPUT_INDEX}:v]subtitles='{ass}':fontsdir='{fontsdir}'[global];"
        f"[0:v][global][2:v]hstack=inputs=3"
    )


def load_stage_ids(parquet_path: Path, column: str) -> list[int]:
    if not parquet_path.is_file():
        raise FileNotFoundError(f"episode parquet does not exist: {parquet_path}")
    frame = pd.read_parquet(parquet_path)
    if column not in frame.columns:
        raise KeyError(f"{parquet_path} has no column {column!r}; columns={list(frame.columns)}")
    return [int(value) for value in frame[column].tolist()]


def combine_episode(
    *,
    ffmpeg: str,
    job: CombineJob,
    stage_column: str,
    fontfile: Path,
    overwrite: bool = False,
) -> bool:
    missing = [path for path in job.input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing input videos: {missing}")
    if not fontfile.is_file():
        raise FileNotFoundError(f"overlay font not found: {fontfile}")

    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    if job.output_path.exists() and not overwrite:
        return False

    stage_ids = load_stage_ids(job.parquet_path, stage_column)
    segments = stage_segments(stage_ids, job.fps)
    if not segments:
        raise ValueError(f"no stage segments to overlay for {job.output_path.name}")

    cmd = [ffmpeg, "-loglevel", "error", "-y" if overwrite else "-n"]
    for path in job.input_paths:
        cmd.extend(["-i", str(path)])

    with tempfile.NamedTemporaryFile("w", suffix=".ass", delete=False, encoding="utf-8") as ass_file:
        ass_path = Path(ass_file.name)
        ass_file.write(build_ass_document(segments))

    try:
        cmd.extend(
            [
                "-filter_complex",
                build_filter_complex(ass_path, fontfile),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(job.output_path),
            ]
        )
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip() or f"exit {result.returncode}"
            raise RuntimeError(f"ffmpeg failed for {job.output_path.name}: {detail}")
    finally:
        ass_path.unlink(missing_ok=True)
    return True


def iter_episode_jobs(
    dataset_root: Path,
    chunk_dirs: list[Path],
    episode_filter: set[int] | None,
) -> list[CombineJob]:
    data_path, chunks_size, fps = load_dataset_info(dataset_root)
    jobs: list[CombineJob] = []
    for chunk_dir in chunk_dirs:
        camera_dirs = [chunk_dir / key for key in CAMERA_KEYS]
        for camera_dir in camera_dirs:
            if not camera_dir.is_dir():
                raise FileNotFoundError(f"camera directory does not exist: {camera_dir}")

        output_dir = chunk_dir / "images.combined"
        global_dir = chunk_dir / "images.global"
        for episode_path in sorted(global_dir.glob("episode_*.mp4"), key=lambda p: p.name):
            episode_index = episode_index_from_name(episode_path)
            if episode_filter is not None and episode_index not in episode_filter:
                continue
            input_paths = [camera_dir / episode_path.name for camera_dir in camera_dirs]
            jobs.append(
                CombineJob(
                    chunk_dir=chunk_dir,
                    input_paths=input_paths,
                    output_path=output_dir / episode_path.name,
                    parquet_path=parquet_path_for_episode(
                        dataset_root, data_path, chunks_size, episode_index
                    ),
                    fps=fps,
                    episode_index=episode_index,
                )
            )
    return jobs


def main() -> int:
    args = build_parser().parse_args()
    ffmpeg = ensure_ffmpeg()
    requested = args.dataset_root.expanduser().resolve()
    dataset_root = resolve_dataset_root(requested)
    jobs = iter_episode_jobs(
        dataset_root,
        resolve_chunk_dirs(requested),
        parse_episode_filter(args.episodes),
    )
    if not jobs:
        raise FileNotFoundError(f"no matching episodes under {requested}")

    combined = 0
    skipped = 0
    with tqdm(total=len(jobs), desc="Combining videos", unit="ep") as progress:
        for job in jobs:
            if combine_episode(
                ffmpeg=ffmpeg,
                job=job,
                stage_column=args.stage_column,
                fontfile=args.fontfile,
                overwrite=args.overwrite,
            ):
                combined += 1
            else:
                skipped += 1
            progress.set_postfix(combined=combined, skipped=skipped, refresh=False)
            progress.update(1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
