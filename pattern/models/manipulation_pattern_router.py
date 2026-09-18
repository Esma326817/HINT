from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .fusion import GatedFusion
from .progress import PROGRESS_HEAD_BY_STAGE_ID
from .image_encoder import ImageEncoder, ImageEncoderConfig, build_image_encoder_config
from .lowdim_encoder import LowdimEncoder, LowdimEncoderConfig


@dataclass(frozen=True)
class ManipulationPatternRouterConfig:
    image_encoder: ImageEncoderConfig = ImageEncoderConfig()
    lowdim_encoder: LowdimEncoderConfig = LowdimEncoderConfig()
    fusion_hidden_dim: int = 128
    num_patterns: int = 6
    predict_progress: bool = True
    num_progress_heads: int = 1


def build_manipulation_pattern_router_config(
    model_cfg: dict[str, Any],
) -> ManipulationPatternRouterConfig:
    """Build nested config from YAML ``model`` section."""
    image_cfg = model_cfg["image_encoder"]
    lowdim_cfg = model_cfg["lowdim_encoder"]

    num_patterns = model_cfg.get("num_patterns", model_cfg.get("num_stages"))
    if num_patterns is None:
        raise KeyError("model.num_patterns is required")

    return ManipulationPatternRouterConfig(
        image_encoder=build_image_encoder_config(image_cfg),
        lowdim_encoder=LowdimEncoderConfig(**lowdim_cfg),
        fusion_hidden_dim=int(model_cfg["fusion_hidden_dim"]),
        num_patterns=int(num_patterns),
        predict_progress=bool(model_cfg.get("predict_progress", True)),
        num_progress_heads=int(model_cfg.get("num_progress_heads", 1)),
    )


def _make_progress_head(dim_hidden: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_hidden, 1),
        nn.Sigmoid(),
    )


def _pattern_to_progress_head(pattern_idx: torch.Tensor, num_heads: int) -> torch.Tensor:
    if num_heads <= 1:
        return torch.zeros_like(pattern_idx)
    if num_heads == 4:
        # stage logits indices 0..5 -> free/pre/transport/dexterous heads.
        mapping = torch.tensor(PROGRESS_HEAD_BY_STAGE_ID, device=pattern_idx.device)
        return mapping[pattern_idx]
    if num_heads == 6:
        return pattern_idx
    raise ValueError(f"Unsupported num_progress_heads={num_heads}; expected 1, 4, or 6")


class ManipulationPatternRouterNet(nn.Module):
    """Manipulation-pattern classifier with visual, proprioceptive, and progress heads.

    Expected inputs:
      low_dim: [B, T, state_dim + effort_dim]
      images: [B, T, C, 3, H, W]
    """

    def __init__(self, cfg: ManipulationPatternRouterConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ManipulationPatternRouterConfig()
        self.image_encoder = ImageEncoder(self.cfg.image_encoder)
        self.lowdim_encoder = LowdimEncoder(self.cfg.lowdim_encoder)

        fusion_inputs = [self.image_encoder.output_dim] * self.image_encoder.num_outputs
        if self.lowdim_encoder.separate:
            fusion_inputs.extend(
                [self.lowdim_encoder.output_dim, self.lowdim_encoder.output_dim]
            )
        else:
            fusion_inputs.append(self.lowdim_encoder.output_dim)

        self.fusion = GatedFusion(
            dim_inputs=fusion_inputs,
            dim_hidden=self.cfg.fusion_hidden_dim,
        )
        self.pattern_head = nn.Linear(self.cfg.fusion_hidden_dim, self.cfg.num_patterns)
        if self.cfg.predict_progress:
            if self.cfg.num_progress_heads == 1:
                self.progress_head = _make_progress_head(self.cfg.fusion_hidden_dim)
                self.progress_heads = None
            else:
                self.progress_head = None
                self.progress_heads = nn.ModuleList(
                    [
                        _make_progress_head(self.cfg.fusion_hidden_dim)
                        for _ in range(self.cfg.num_progress_heads)
                    ]
                )
        else:
            self.progress_head = None
            self.progress_heads = None

    def _collect_fusion_inputs(
        self,
        low_dim: torch.Tensor,
        images: torch.Tensor,
    ) -> list[torch.Tensor]:
        # f_global, f_left, f_right each go into gated fusion separately.
        visual_features = self.image_encoder(images)
        low_features = self.lowdim_encoder(low_dim)
        if self.lowdim_encoder.separate:
            return [*visual_features, *low_features]
        return [*visual_features, low_features]

    def forward(
        self,
        low_dim: torch.Tensor,
        images: torch.Tensor,
        progress_head_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        fused = self.fusion(self._collect_fusion_inputs(low_dim, images))
        pattern_logits = self.pattern_head(fused)
        out: dict[str, torch.Tensor] = {"pattern": pattern_logits}
        if self.progress_head is not None:
            out["progress"] = self.progress_head(fused)
        elif self.progress_heads is not None:
            progress_all = torch.cat([head(fused) for head in self.progress_heads], dim=-1)
            if progress_head_index is None:
                predicted_pattern = pattern_logits.argmax(dim=-1)
                progress_head_index = _pattern_to_progress_head(
                    predicted_pattern,
                    self.cfg.num_progress_heads,
                )
            progress_head_index = progress_head_index.to(
                device=progress_all.device,
                dtype=torch.long,
            )
            out["progress_all"] = progress_all
            out["progress"] = progress_all.gather(1, progress_head_index.view(-1, 1))
        return out

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        """Load current or legacy checkpoints, including the former stage-head name."""
        filtered = state_dict.copy()
        if hasattr(state_dict, "_metadata"):
            filtered._metadata = state_dict._metadata  # type: ignore[attr-defined]
        for key in tuple(filtered):
            if key.startswith(("phase_head.", "focus_head.")):
                del filtered[key]
            elif key.startswith("stage_head."):
                filtered[f"pattern_head.{key.removeprefix('stage_head.')}"] = filtered.pop(key)
        return super().load_state_dict(filtered, strict=strict, assign=assign)
