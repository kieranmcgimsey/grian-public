"""Day-ahead quantile network for NEM price forecasting.

A PyTorch model that predicts all delivery periods at once as sorted
quantiles, trained with pinball loss. Runs on CPU or Apple Silicon (MPS).
"""

import torch
import torch.nn as nn


class DayAheadQuantileNet(nn.Module):
    """Neural network producing sorted quantile forecasts for a full day.

    Architecture: MLP that takes the feature vector and outputs
    ``(horizon * n_quantiles)`` values, reshaped to
    ``(horizon, n_quantiles)`` with a soft-sort to enforce monotonicity.
    """

    def __init__(
        self,
        n_features: int,
        horizon: int = 48,
        n_quantiles: int = 7,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
    ):
        """Initialise the quantile network.

        Args:
            n_features: Number of input features.
            horizon: Number of delivery periods to forecast.
            n_quantiles: Number of quantile levels.
            hidden_dims: Widths of hidden layers.
        """
        super().__init__()
        self.horizon = horizon
        self.n_quantiles = n_quantiles

        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.1)])
            in_dim = h
        layers.append(nn.Linear(in_dim, horizon * n_quantiles))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing sorted quantile forecasts.

        Args:
            x: Input tensor of shape ``(batch, n_features)``.

        Returns:
            Tensor of shape ``(batch, horizon, n_quantiles)``, sorted
            along the quantile dimension.
        """
        raw = self.net(x).view(-1, self.horizon, self.n_quantiles)
        return torch.sort(raw, dim=-1).values


def pinball_loss_fn(
    pred: torch.Tensor,
    actual: torch.Tensor,
    quantiles: torch.Tensor,
) -> torch.Tensor:
    """Compute mean pinball loss across all quantiles and periods.

    Args:
        pred: Predicted quantiles, shape ``(batch, horizon, n_quantiles)``.
        actual: Actual prices, shape ``(batch, horizon, 1)`` or broadcastable.
        quantiles: Quantile levels, shape ``(n_quantiles,)``.

    Returns:
        Scalar loss tensor.
    """
    diff = actual - pred
    loss = torch.where(diff >= 0, quantiles * diff, (quantiles - 1) * diff)
    return loss.mean()
