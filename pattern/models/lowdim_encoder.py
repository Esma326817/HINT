from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LowdimEncoderConfig:
    state_dim: int = 14
    effort_dim: int = 14
    separate: bool = True
    embed_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0


class _GRUStreamEncoder(nn.Module):
    def __init__(self, dim_input: int, cfg: LowdimEncoderConfig) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(dim_input, cfg.embed_dim),
            nn.LayerNorm(cfg.embed_dim),
            nn.SiLU(inplace=True),
        )
        self.gru = nn.GRU(
            input_size=cfg.embed_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embed(x)
        _, hidden = self.gru(embedded)
        return hidden[-1]


class LowdimEncoder(nn.Module):
    """Encode proprioceptive sequences from state and effort.

    Input: [B, T, state_dim + effort_dim]
    Output:
      - combined mode: [B, hidden_dim]
      - separate mode: list[[B, hidden_dim], [B, hidden_dim]] (state, effort)
    """

    def __init__(self, cfg: LowdimEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dim_total = cfg.state_dim + cfg.effort_dim

        if cfg.separate:
            self.enc_state = _GRUStreamEncoder(cfg.state_dim, cfg)
            self.enc_effort = _GRUStreamEncoder(cfg.effort_dim, cfg)
            self.enc_combined = None
        else:
            self.enc_state = None
            self.enc_effort = None
            self.enc_combined = _GRUStreamEncoder(dim_total, cfg)

    def _split_inputs(self, low_dim: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if low_dim.shape[-1] != self.cfg.state_dim + self.cfg.effort_dim:
            raise ValueError(
                f"Expected low_dim [..., {self.cfg.state_dim + self.cfg.effort_dim}], "
                f"got {tuple(low_dim.shape)}"
            )
        state = low_dim[..., : self.cfg.state_dim]
        effort = low_dim[..., self.cfg.state_dim :]
        return state, effort

    def forward(self, low_dim: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        if self.cfg.separate:
            state, effort = self._split_inputs(low_dim)
            return [self.enc_state(state), self.enc_effort(effort)]
        return self.enc_combined(low_dim)

    @property
    def output_dim(self) -> int:
        return self.cfg.hidden_dim

    @property
    def separate(self) -> bool:
        return self.cfg.separate
