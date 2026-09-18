from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

_VALID_LAYER_IDS = frozenset({1, 2, 3, 4})


def resolve_finetune_layers(cfg: dict[str, Any]) -> tuple[str, ...]:
    """Read ``finetune_layers: [3, 4]`` from YAML. Mutates *cfg* in-place."""
    layer_ids = [int(i) for i in cfg.pop("finetune_layers", (3, 4))]
    invalid = [i for i in layer_ids if i not in _VALID_LAYER_IDS]
    if invalid:
        raise ValueError(f"finetune_layers must be in 1-4, got {layer_ids}")
    return tuple(f"layer{i}" for i in layer_ids)


class CameraStem(nn.Module):
    """Per-camera lightweight adapter before the shared ResNet18 trunk."""

    def __init__(self, in_channels: int = 3, hidden_channels: int = 32) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images + self.adapter(images)


class SharedResNet18Backbone(nn.Module):
    """Shared ResNet18 trunk with selective layer finetuning."""

    output_dim = 512
    _LAYER_NAMES = ("layer1", "layer2", "layer3", "layer4")

    def __init__(
        self,
        pretrained: bool = True,
        finetune_layers: Sequence[str] = ("layer3", "layer4"),
    ) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:
            raise ImportError(
                "SharedResNet18Backbone requires torchvision."
            ) from exc

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)

        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.pool = model.avgpool

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

        finetune = set(finetune_layers)
        unknown = finetune - set(self._LAYER_NAMES)
        if unknown:
            raise ValueError(
                f"Unknown finetune_layers {sorted(unknown)}. "
                f"Valid layers: {self._LAYER_NAMES}"
            )

        trainable_modules = [
            self.stem,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.pool,
        ]
        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = False

        for layer_name in finetune:
            for param in getattr(self, layer_name).parameters():
                param.requires_grad = True

        self._finetune_layers = finetune

    def train(self, mode: bool = True) -> "SharedResNet18Backbone":
        super().train(mode)
        if not self._finetune_layers:
            self.stem.eval()
            self.layer1.eval()
            self.layer2.eval()
            self.layer3.eval()
            self.layer4.eval()
            self.pool.eval()
            return self

        frozen = [
            self.stem,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.pool,
        ]
        for module in frozen:
            if not any(param.requires_grad for param in module.parameters()):
                module.eval()
        return self

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        images = (images + 1.0) * 0.5
        return (images - self.mean) / self.std

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self._normalize(images)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return torch.flatten(x, start_dim=1)
