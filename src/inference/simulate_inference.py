"""Simulated inference: replay a recorded LeRobot episode against the running
``inference.api_server`` so the full online path (stage prediction → routing →
grounding → render) can be exercised without a robot.

Randomly samples an episode (or use ``--episode``), streams its 3-camera frames
plus ``state``/``effort`` to ``POST /step`` in order, and compares the server's
predicted stage to the dataset's ``stage_id_gt``.

Run the server in predict mode first (see README), then::

    python -m inference.simulate_inference \
        --dataset-root /dataset/.../piper_letter_v2_big_annotation_merged \
        --server http://localhost:8000 \
        --out-dir outputs/sim
"""

from __future__ import annotations

import argparse
import base64
import json
import random
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import pandas as pd
import requests

from pattern.data import LeRobotPaths, discover_episodes

_CAMERA_FIELDS = {
    "images.global": "global_image",
    "images.left_wrist": "left_wrist_image",
    "images.right_wrist": "right_wrist_image",
}


class _VideoReader(Protocol):
    def read_jpeg(self) -> bytes | None: ...

    def release(self) -> None: ...


def _encode_bgr_jpeg(bgr: np.ndarray) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes() if ok else None


class _Cv2VideoReader:
    def __init__(self, path: Path, first_jpeg: bytes, cap: cv2.VideoCapture) -> None:
        self.path = path
        self._first_jpeg = first_jpeg
        self._cap = cap

    @classmethod
    def open(cls, path: Path) -> "_Cv2VideoReader | None":
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            return None
        ok, bgr = cap.read()
        if not ok:
            cap.release()
            return None
        first_jpeg = _encode_bgr_jpeg(bgr)
        if first_jpeg is None:
            cap.release()
            return None
        return cls(path, first_jpeg, cap)

    def read_jpeg(self) -> bytes | None:
        if self._first_jpeg is not None:
            payload = self._first_jpeg
            self._first_jpeg = None
            return payload
        ok, bgr = self._cap.read()
        if not ok:
            return None
        return _encode_bgr_jpeg(bgr)

    def release(self) -> None:
        self._cap.release()


class _PyAvVideoReader:
    def __init__(self, path: Path) -> None:
        import av

        self.path = path
        self._container = av.open(str(path))
        self._frames = self._container.decode(video=0)

    def read_jpeg(self) -> bytes | None:
        try:
            frame = next(self._frames)
        except StopIteration:
            return None
        rgb = frame.to_ndarray(format="rgb24")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return _encode_bgr_jpeg(bgr)

    def release(self) -> None:
        self._container.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a recorded episode against the inference server.")
    parser.add_argument(
        "--dataset-root",
        default="/dataset/robot/real_world/piper/lerobot/letter/piper_letter_v2_big_annotation_merged",
    )
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--episode", type=int, default=None, help="Episode id; random if omitted.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = whole episode.")
    parser.add_argument("--out-dir", default="outputs/sim")
    parser.add_argument("--task-instruction", default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-reset", action="store_true", help="Skip POST /reset before streaming.")
    parser.add_argument("--use-gt-stage", action="store_true", help="Send parquet stage_id_gt to /step as stage_output.")
    return parser.parse_args()


def _open_videos(paths: LeRobotPaths, episode: int) -> dict[str, _VideoReader]:
    readers: dict[str, _VideoReader] = {}
    for camera_key in _CAMERA_FIELDS:
        video_path = paths.video_file_path(camera_key, episode)
        reader = _Cv2VideoReader.open(video_path)
        if reader is None:
            try:
                reader = _PyAvVideoReader(video_path)
                first_jpeg = reader.read_jpeg()
                if first_jpeg is None:
                    reader.release()
                    raise RuntimeError(f"failed to read first frame from {video_path}")
                reader = _BufferedVideoReader(reader, first_jpeg)
                print(f"[*] using PyAV decoder for {video_path}", flush=True)
            except ImportError as exc:
                raise RuntimeError(
                    f"OpenCV failed to decode {video_path} and PyAV is not installed"
                ) from exc
        readers[camera_key] = reader
    return readers


class _BufferedVideoReader:
    def __init__(self, reader: _VideoReader, first_jpeg: bytes) -> None:
        self._reader = reader
        self._first_jpeg: bytes | None = first_jpeg

    def read_jpeg(self) -> bytes | None:
        if self._first_jpeg is not None:
            payload = self._first_jpeg
            self._first_jpeg = None
            return payload
        return self._reader.read_jpeg()

    def release(self) -> None:
        self._reader.release()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    paths = LeRobotPaths.from_root(args.dataset_root)
    episodes = discover_episodes(args.dataset_root)
    if not episodes:
        raise SystemExit(f"no episodes under {args.dataset_root}")
    episode = args.episode if args.episode is not None else random.choice(episodes)
    print(f"[*] episode {episode} of {len(episodes)} (server={args.server})", flush=True)

    df = pd.read_parquet(paths.parquet_path(episode), columns=["state", "effort", "stage_id_gt"])
    states = np.stack(df["state"].to_numpy()).astype(np.float32)
    efforts = np.stack(df["effort"].to_numpy()).astype(np.float32)
    gt_stage = df["stage_id_gt"].to_numpy().astype(int)
    num_frames = len(df)
    if args.max_frames > 0:
        num_frames = min(num_frames, args.max_frames)

    out_dir = Path(args.out_dir) / f"episode_{episode:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    readers = _open_videos(paths, episode)
    try:
        rows = []
        for frame_idx in range(num_frames):
            frame_bytes = {key: reader.read_jpeg() for key, reader in readers.items()}
            if any(payload is None for payload in frame_bytes.values()):
                print(f"[!] video ran out at frame {frame_idx}", flush=True)
                break

            files = {
                _CAMERA_FIELDS[key]: (f"{key}.jpg", payload, "image/jpeg")
                for key, payload in frame_bytes.items()
            }
            data = {
                "robot_state": json.dumps(states[frame_idx].tolist()),
                "effort": json.dumps(efforts[frame_idx].tolist()),
            }
            if args.task_instruction:
                data["task_instruction"] = args.task_instruction
            if args.use_gt_stage:
                data["stage_output"] = json.dumps({"stage_id": int(gt_stage[frame_idx])})

            endpoint = "/reset" if (frame_idx == 0 and not args.no_reset) else None
            if endpoint == "/reset":
                # /reset takes a single global image; warm the session first.
                session.post(
                    f"{args.server}/reset",
                    files={"image": files["global_image"]},
                    data={
                        "robot_state": data["robot_state"],
                        **(
                            {"task_instruction": args.task_instruction}
                            if args.task_instruction
                            else {}
                        ),
                    },
                    timeout=args.timeout,
                ).raise_for_status()

            resp = session.post(f"{args.server}/step", files=files, data=data, timeout=args.timeout)
            resp.raise_for_status()
            result = resp.json()

            pred_stage = result.get("stage")
            rows.append({
                "frame_index": frame_idx,
                "gt_stage_id": int(gt_stage[frame_idx]),
                "pred_stage": pred_stage,
                "prompt": result.get("prompt"),
            })
            rendered_b64 = (result.get("rendered_images") or {}).get("global") or result.get("rendered_image")
            if rendered_b64:
                (out_dir / f"render_{frame_idx:06d}.png").write_bytes(base64.b64decode(rendered_b64))

            if frame_idx % 20 == 0:
                print(f"  frame {frame_idx:4d}  gt={int(gt_stage[frame_idx])}  pred={pred_stage}", flush=True)
    finally:
        for reader in readers.values():
            reader.release()

    (out_dir / "predictions.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"[*] {len(rows)} frames streamed; outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
