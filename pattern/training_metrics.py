"""Epoch reductions that keep batch statistics on the training device."""

from collections.abc import Mapping

import torch


class EpochMetrics:
    """Preserve legacy sample-weighted batch losses and per-group MAE metrics.

    Scalar reductions stay on-device until ``compute`` is requested. Neither
    update nor compute changes the model graph, loss, or optimizer state.
    """

    def __init__(self, device: torch.device, predict_progress: bool) -> None:
        self.sums = torch.zeros(9, device=device, dtype=torch.float64)
        self.count = 0
        self.predict_progress = predict_progress

    @torch.no_grad()
    def update(
        self,
        out: Mapping[str, torch.Tensor],
        stage: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        loss: torch.Tensor,
        pattern_loss: torch.Tensor,
        progress_loss: torch.Tensor,
    ) -> None:
        count = stage.numel()
        zero = loss.new_zeros(())
        values = [
            loss.detach() * count,
            pattern_loss.detach() * count,
            progress_loss.detach() * count,
            (out["pattern"].argmax(-1) == stage).sum(),
        ]
        if self.predict_progress and "progress" in out:
            error = (out["progress"].detach().squeeze(-1) - target.squeeze(-1)).abs()
            boundary = weight < 1.0
            stable = weight >= 1.0
            # Excluded NaNs must not contaminate an empty/other group.
            values.extend([
                error.sum(),
                torch.where(boundary, error, 0).sum(),
                boundary.sum(),
                torch.where(stable, error, 0).sum(),
                stable.sum(),
            ])
        else:
            values.extend([zero] * 5)
        self.sums += torch.stack(values).to(torch.float64)
        self.count += count

    def compute(self) -> dict[str, float]:
        loss, pattern, progress, correct, error, boundary, nb, stable, ns = self.sums.tolist()
        count = max(1, self.count)
        return {
            "loss": loss / count,
            "pattern_loss": pattern / count,
            "progress_loss": progress / count,
            "pattern_acc": correct / count,
            "progress_mae": error / count,
            "progress_mae_boundary": boundary / max(1, nb),
            "progress_mae_stable": stable / max(1, ns),
        }
