"""On-demand MP4 frames without a disk cache or corpus-sized RAM allocation.

Like the pinned LeRobot reader in pi_bright, each request owns a CPU decoder
using approximate seeking. Requests from the same batch are coalesced by video;
only the requested, resized frames survive decoding. Nothing is cached across
batches, and no decoder is inherited by a DataLoader worker.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time
from typing import Sequence

import torch

from .video import resize_and_normalize, torchcodec_available


class StreamingFrameStore:
    """Decode requested frames, with FP16 rounding before returning FP32 inputs."""

    backend = "streaming"

    def __init__(self, height: int, width: int, decode_chunk_size: int = 16) -> None:
        if min(height, width, decode_chunk_size) <= 0:
            raise ValueError("Image dimensions and streaming decode chunk size must be positive")
        self.height = int(height)
        self.width = int(width)
        self.decode_chunk_size = int(decode_chunk_size)

    def prepare(self, video_paths: Sequence[str]) -> dict[str, float]:
        # Check the complete manifest, without opening decoders or touching .pt.
        start = time.perf_counter()
        paths = set(map(str, video_paths))
        for path in sorted(paths):
            if not Path(path).is_file():
                raise FileNotFoundError(f"Video missing: {path}")
        print(f"[*] streaming {len(paths)} MP4 videos; no .pt cache or video preload", flush=True)
        return {"prepare_s": time.perf_counter() - start, "storage_bytes": 0.0}

    def _resize(self, frames: torch.Tensor) -> torch.Tensor:
        # Preserve the trained model's normalized input precision.
        return resize_and_normalize(
            frames.float() / 255.0, self.height, self.width, dtype=torch.float16
        ).float()

    def _decode(self, video_path: str, indices: list[int]) -> torch.Tensor:
        if torchcodec_available():
            from torchcodec.decoders import VideoDecoder

            decoder = VideoDecoder(
                video_path, device="cpu", seek_mode="approximate", num_ffmpeg_threads=1
            )
            count = int(decoder.metadata.num_frames)
            if count <= 0:
                raise ValueError(f"Video has no frames: {video_path}")
            clipped = [max(0, min(index, count - 1)) for index in indices]
            chunks = []
            for start in range(0, len(clipped), self.decode_chunk_size):
                frames = decoder.get_frames_at(indices=clipped[start:start + self.decode_chunk_size])
                chunks.append(self._resize(frames.data))
            return torch.cat(chunks)

        # OpenCV fallback with the same RGB conversion and normalization.
        import cv2

        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if count <= 0:
                raise ValueError(f"Video has no frames: {video_path}")
            frames = []
            for index in indices:
                clipped = max(0, min(index, count - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, clipped)
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f"Failed to read frame {clipped}: {video_path}")
                rgb = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                frames.append(self._resize(rgb.unsqueeze(0)))
            return torch.cat(frames)
        finally:
            cap.release()

    def read(self, video_path: str, indices: tuple[int, ...]) -> torch.Tensor:
        return self.read_many([(video_path, indices)])[0]

    def read_many(self, requests: Sequence[tuple[str, tuple[int, ...]]]) -> list[torch.Tensor]:
        """Deduplicate frames within each video, then restore request/frame order.

        This changes only I/O scheduling inside a batch, never the sampler, its
        random state, sample weighting, or the order presented to the model.
        """
        groups: dict[str, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
        for position, (path, indices) in enumerate(requests):
            if not indices:
                raise ValueError("Frame indices must not be empty")
            groups[str(path)].append((position, tuple(map(int, indices))))
        results: dict[int, torch.Tensor] = {}
        for path, group in groups.items():
            unique = sorted({index for _, indices in group for index in indices})
            frames = self._decode(path, unique)
            positions = {index: position for position, index in enumerate(unique)}
            for position, indices in group:
                results[position] = frames[[positions[index] for index in indices]]
        return [results[position] for position in range(len(requests))]
