from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .resnet_backbone import CameraStem, SharedResNet18Backbone, resolve_finetune_layers

# Internal camera names; dataset.camera_keys maps them to LeRobot video fields.
CAMERA_NAMES = ("global", "left_wrist", "right_wrist")


@dataclass(frozen=True)
class ImageEncoderConfig:
    num_times: int = 1
    num_cameras: int = 3
    camera_names: tuple[str, ...] = CAMERA_NAMES
    feature_dim: int = 64
    hidden_dim: int = 128
    stem_hidden_channels: int = 32
    pretrained: bool = True
    # Layers to finetune on shared ResNet18. Recommended: layer3 + layer4.
    finetune_layers: tuple[str, ...] = ("layer3", "layer4")


def build_image_encoder_config(cfg: dict[str, Any]) -> ImageEncoderConfig:
    if not isinstance(cfg, dict):
        raise KeyError("image_encoder must be a mapping")

    raw = dict(cfg)
    raw.pop("freeze_backbone", None)
    raw.pop("finetune_preset", None)
    raw.pop("stem_channels", None)
    raw.pop("vision_fusion_dim", None)

    # YAML may use either ``pretrained_resnet`` or ``pretrained``.
    if "pretrained" not in raw and "pretrained_resnet" in raw:
        raw["pretrained"] = raw.pop("pretrained_resnet")
    else:
        raw.pop("pretrained_resnet", None)

    finetune_layers = resolve_finetune_layers(raw)
    camera_names = raw.pop("camera_names", CAMERA_NAMES)
    if isinstance(camera_names, list):
        camera_names = tuple(camera_names)

    # Older checkpoints store the disabled image-GRU options in their config.
    # Keep those checkpoints loadable without retaining an image-GRU module.
    legacy_gru = raw.pop("gru", {}) or {}
    legacy_enabled = raw.pop("use_temporal_gru", False)
    if legacy_gru.get("use_temporal_gru", False) or legacy_enabled:
        raise ValueError("Image temporal GRU is no longer supported; use a checkpoint trained without it.")
    for key in ("temporal_hidden_dim", "temporal_num_layers", "temporal_dropout"):
        raw.pop(key, None)

    return ImageEncoderConfig(
        **raw,
        camera_names=tuple(camera_names),
        finetune_layers=finetune_layers,
    )


class CameraHistoryHead(nn.Module):
    """Per-camera frame projection, history concatenation, and output projection."""

    def __init__(self, cfg: ImageEncoderConfig) -> None:
        super().__init__()
        frame_dim = SharedResNet18Backbone.output_dim
        self.frame_proj = nn.Sequential(
            nn.Linear(frame_dim, cfg.feature_dim),
            nn.LayerNorm(cfg.feature_dim),
            nn.SiLU(inplace=True),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(cfg.num_times * cfg.feature_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        frame_features = self.frame_proj(frame_features)
        return self.output_proj(frame_features.flatten(start_dim=1))


class ImageEncoder(nn.Module):
    """Multi-camera encoder; each camera is encoded independently.

    Pipeline per camera (e.g. global / left_wrist / right_wrist):
        images -> camera stem -> shared ResNet18 -> frame projection -> history concatenation -> f_cam

    The three camera features are returned separately for gated fusion
    in ``ManipulationPatternRouterNet``.

    Input: [B, T, C, 3, H, W]
    Output: [f_global, f_left, f_right], each [B, hidden_dim]
    """

    def __init__(self, cfg: ImageEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if len(cfg.camera_names) != cfg.num_cameras:
            raise ValueError(
                f"camera_names length ({len(cfg.camera_names)}) must match "
                f"num_cameras ({cfg.num_cameras})"
            )

        self.stems = nn.ModuleDict(
            {
                name: CameraStem(hidden_channels=cfg.stem_hidden_channels)
                for name in cfg.camera_names
            }
        )
        self.backbone = SharedResNet18Backbone(
            pretrained=cfg.pretrained,
            finetune_layers=cfg.finetune_layers,
        )
        self.camera_heads = nn.ModuleList(
            [CameraHistoryHead(cfg) for _ in range(cfg.num_cameras)]
        )

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        bsz, num_times, num_cameras, channels, height, width = images.shape
        if num_times != self.cfg.num_times or num_cameras != self.cfg.num_cameras or channels != 3:
            raise ValueError(
                f"Expected images [B,{self.cfg.num_times},{self.cfg.num_cameras},3,H,W], "
                f"got {tuple(images.shape)}"
            )

        camera_features: list[torch.Tensor] = []
        for camera_idx, camera_name in enumerate(self.cfg.camera_names):
            cam_images = images[:, :, camera_idx].reshape(
                bsz * num_times, channels, height, width
            )
            cam_images = self.stems[camera_name](cam_images)
            frame_features = self.backbone(cam_images).reshape(bsz, num_times, -1)
            camera_features.append(self.camera_heads[camera_idx](frame_features))

        return camera_features

    @property
    def output_dim(self) -> int:
        return self.cfg.hidden_dim

    @property
    def num_outputs(self) -> int:
        return self.cfg.num_cameras
