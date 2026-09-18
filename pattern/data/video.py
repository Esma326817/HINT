from __future__ import annotations

import ctypes
from pathlib import Path
import site

import torch
import torch.nn.functional as F

_TORCHCODEC_AVAILABLE: bool | None = None
_TORCHCODEC_DEPS_PRELOADED = False


def _torchcodec_runtime_library_dirs() -> list[Path]:
    dirs: list[Path] = []
    for site_dir in site.getsitepackages():
        root = Path(site_dir)
        for candidate in (
            root / "av.libs",
            root / "nvidia" / "npp" / "lib",
        ):
            if candidate.is_dir():
                dirs.append(candidate)
    return dirs


def _preload_torchcodec_runtime_deps() -> None:
    global _TORCHCODEC_DEPS_PRELOADED
    if _TORCHCODEC_DEPS_PRELOADED:
        return

    sonames = (
        "libavutil.so.60",
        "libavcodec.so.62",
        "libavformat.so.62",
        "libavdevice.so.62",
        "libavfilter.so.11",
        "libswresample.so.6",
        "libswscale.so.9",
        "libnppicc.so.12",
    )
    for lib_dir in _torchcodec_runtime_library_dirs():
        for soname in sonames:
            path = lib_dir / soname
            if path.is_file():
                try:
                    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
    _TORCHCODEC_DEPS_PRELOADED = True


def torchcodec_available() -> bool:
    global _TORCHCODEC_AVAILABLE
    if _TORCHCODEC_AVAILABLE is None:
        try:
            _preload_torchcodec_runtime_deps()
            from torchcodec.decoders import VideoDecoder  # noqa: F401

            _TORCHCODEC_AVAILABLE = True
        except (ImportError, OSError, RuntimeError):
            _TORCHCODEC_AVAILABLE = False
    return _TORCHCODEC_AVAILABLE


def _match_frames_to_timestamps(
    loaded_frames: list[torch.Tensor],
    loaded_ts: list[float],
    timestamps: list[float],
    tolerance_s: float,
    video_path: Path | str,
) -> torch.Tensor:
    query_ts = torch.tensor(timestamps, dtype=torch.float32)
    loaded_ts_tensor = torch.tensor(loaded_ts, dtype=torch.float32)
    dist = torch.cdist(query_ts[:, None], loaded_ts_tensor[:, None], p=1)
    min_dist, argmin = dist.min(dim=1)

    is_within_tol = min_dist < tolerance_s
    if not bool(is_within_tol.all()):
        raise RuntimeError(
            "One or more decoded video frames exceeded timestamp tolerance. "
            f"max_error={float(min_dist.max()):.6f}, tolerance_s={tolerance_s:.6f}, "
            f"video={video_path}"
        )

    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin])
    return closest_frames.float() / 255.0


def _decode_with_torchcodec(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
) -> torch.Tensor:
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(str(video_path), device="cpu", seek_mode="approximate")
    average_fps = float(decoder.metadata.average_fps)
    frame_indices = [round(ts * average_fps) for ts in timestamps]
    frames_batch = decoder.get_frames_at(indices=frame_indices)

    loaded_frames = []
    loaded_ts = []
    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=False):
        loaded_frames.append(frame)
        loaded_ts.append(float(pts.item()))

    return _match_frames_to_timestamps(loaded_frames, loaded_ts, timestamps, tolerance_s, video_path)


def _decode_with_cv2(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
) -> torch.Tensor:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video with OpenCV: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0

    loaded_frames: list[torch.Tensor] = []
    loaded_ts: list[float] = []
    for ts in timestamps:
        frame_idx = max(0, round(ts * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(
                f"Failed to read frame {frame_idx} at ts={ts:.6f} from video={video_path}"
            )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        loaded_frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
        loaded_ts.append(frame_idx / fps)
    cap.release()

    return _match_frames_to_timestamps(loaded_frames, loaded_ts, timestamps, tolerance_s, video_path)


def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
) -> torch.Tensor:
    """Decode a few frames at timestamps. Returns float32 [T, C, H, W] in [0, 1]."""

    if torchcodec_available():
        return _decode_with_torchcodec(video_path, timestamps, tolerance_s)
    return _decode_with_cv2(video_path, timestamps, tolerance_s)


def resize_and_normalize(
    frames: torch.Tensor,
    height: int,
    width: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Resize ``[T,C,H,W]`` RGB frames and normalize them to ``[-1, 1]``."""
    if frames.ndim != 4:
        raise RuntimeError(f"Expected decoded frames [T,C,H,W], got {tuple(frames.shape)}")
    frames = F.interpolate(frames, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
    return (frames * 2.0 - 1.0).to(dtype)


def _decode_full_video_torchcodec(
    video_path: Path | str,
    height: int,
    width: int,
    chunk_size: int,
) -> torch.Tensor:
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(str(video_path), device="cpu", seek_mode="approximate")
    num_frames = int(decoder.metadata.num_frames)
    if num_frames <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")

    chunks: list[torch.Tensor] = []
    for start in range(0, num_frames, chunk_size):
        end = min(num_frames, start + chunk_size)
        indices = list(range(start, end))
        batch = decoder.get_frames_at(indices=indices)
        chunks.append(
            resize_and_normalize(
                batch.data.float() / 255.0,
                height,
                width,
                dtype=torch.float16,
            )
        )
    return torch.cat(chunks, dim=0)


def _decode_full_video_cv2(video_path: Path | str, height: int, width: int) -> torch.Tensor:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video with OpenCV: {video_path}")

    frames: list[torch.Tensor] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        frames.append(
            resize_and_normalize(frame, height, width, dtype=torch.float16)
        )
    cap.release()

    if not frames:
        raise RuntimeError(f"Video has no frames: {video_path}")
    return torch.cat(frames, dim=0)


def decode_video_frames_batch(
    video_path: Path | str,
    height: int,
    width: int,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Decode an entire video, resize once, return float16 [N, C, H, W] in [-1, 1]."""

    return decode_video_frames_batch_raw(video_path, height, width, chunk_size)


def decode_video_frames_batch_raw(
    video_path: Path | str,
    height: int,
    width: int,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Decode an entire video into memory for offline analysis and reference checks."""

    if torchcodec_available():
        return _decode_full_video_torchcodec(video_path, height, width, chunk_size)
    return _decode_full_video_cv2(video_path, height, width)
