"""SAM2 runtime configuration.

Keep SAM2 model/checkpoint choices separate from GroundingDINO. Environment
installation and compatible PyTorch/CUDA versions are documented in SETUP.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SAM2_MODEL_CFG = os.getenv("SAM2_MODEL_CFG", "configs/sam2.1/sam2.1_hiera_s.yaml")


def _default_sam2_checkpoint() -> Path:
    env_value = os.getenv("SAM2_CHECKPOINT")
    if env_value:
        return Path(env_value)

    perception_root = Path(__file__).resolve().parents[1]
    candidates = (
        perception_root / "models" / "sam2.1_hiera_small.pt",
        Path.home() / "models" / "sam2.1_hiera_small.pt",
        Path("/data/checkpoints/base_models/sam2.1_hiera_small.pt"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_SAM2_CHECKPOINT = _default_sam2_checkpoint()


@dataclass(frozen=True)
class SAM2Config:
    """Configuration for SAM2 video propagation."""

    model_cfg: str = DEFAULT_SAM2_MODEL_CFG
    checkpoint_path: Path = DEFAULT_SAM2_CHECKPOINT
    device: str = "cuda"
    vos_optimized: bool = False
    apply_postprocessing: bool = True

    def validate(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "SAM2 checkpoint not found: "
                f"{self.checkpoint_path}. Set SAM2_CHECKPOINT to your local sam2.1_hiera_small.pt path."
            )
