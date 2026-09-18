from __future__ import annotations

import torch
from torch import nn


class GatedFusion(nn.Module):
    """Fuse projected feature sources with learned softmax gates."""

    def __init__(
        self,
        dim_inputs: list[int],
        dim_hidden: int,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim_hidden),
                    nn.LayerNorm(dim_hidden),
                    nn.GELU(),
                )
                for dim in dim_inputs
            ]
        )
        self.num_sources = len(dim_inputs)
        self.gate_net = nn.Sequential(
            nn.Linear(dim_hidden * self.num_sources, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, self.num_sources),
        )
        self.norm = nn.LayerNorm(dim_hidden)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Fuse a list of ``[batch, feature_dim]`` tensors."""
        projected = [
            projection(value)
            for projection, value in zip(self.projections, inputs, strict=True)
        ]
        stacked = torch.stack(projected, dim=1)
        gates = self.gate_net(torch.cat(projected, dim=-1)).softmax(dim=1)
        return self.norm(torch.sum(gates.unsqueeze(-1) * stacked, dim=1))
