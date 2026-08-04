"""Scoring functions for point and probabilistic price forecasts.

Every model reports skill against a naive baseline and against the
operator's own forecast (AEMO pre-dispatch), using these shared metrics.
"""

import numpy as np
from numpy.typing import ArrayLike


def mae(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Mean absolute error.

    Args:
        actual: Observed values.
        forecast: Predicted values.

    Returns:
        MAE in the same units as the inputs.
    """
    actual, forecast = np.asarray(actual), np.asarray(forecast)
    return float(np.mean(np.abs(actual - forecast)))


def relative_mae(
    actual: ArrayLike,
    forecast: ArrayLike,
    baseline: ArrayLike,
) -> float:
    """Relative MAE: model MAE divided by baseline MAE.

    A value below 1 means the model beats the baseline.

    Args:
        actual: Observed values.
        forecast: Model predictions.
        baseline: Baseline predictions.

    Returns:
        Ratio of model MAE to baseline MAE.
    """
    return mae(actual, forecast) / mae(actual, baseline)


def pinball_loss(
    actual: ArrayLike,
    forecast: ArrayLike,
    quantile: float,
) -> float:
    """Pinball (quantile) loss for a single quantile level.

    Args:
        actual: Observed values.
        forecast: Quantile predictions.
        quantile: Quantile level in (0, 1).

    Returns:
        Mean pinball loss.
    """
    actual, forecast = np.asarray(actual), np.asarray(forecast)
    diff = actual - forecast
    return float(np.mean(np.where(diff >= 0, quantile * diff, (quantile - 1) * diff)))


def crps(
    actual: ArrayLike,
    quantile_forecasts: ArrayLike,
    quantiles: ArrayLike,
) -> float:
    """Continuous ranked probability score estimated from quantile forecasts.

    Approximates the CRPS by numerical integration of the pinball loss
    across quantile levels.

    Args:
        actual: Observed values, shape ``(n,)``.
        quantile_forecasts: Predicted quantiles, shape ``(n, q)``.
        quantiles: Quantile levels, shape ``(q,)``.

    Returns:
        Mean CRPS across observations.
    """
    actual = np.asarray(actual)
    quantile_forecasts = np.asarray(quantile_forecasts)
    quantiles = np.asarray(quantiles)
    losses = [
        pinball_loss(actual, quantile_forecasts[:, i], quantiles[i])
        for i in range(len(quantiles))
    ]
    return float(np.mean(losses))
