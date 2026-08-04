"""Tests for grian.metrics scoring functions."""

import numpy as np

from grian.metrics import crps, mae, pinball_loss, relative_mae


def test_mae_perfect():
    """MAE of a perfect forecast is zero."""
    assert mae([1, 2, 3], [1, 2, 3]) == 0.0


def test_mae_known():
    """MAE matches hand-computed value."""
    assert mae([1, 2, 3], [2, 2, 2]) == 2 / 3


def test_relative_mae_below_one_is_better():
    """A model closer to actual than the baseline scores below 1."""
    actual = [10, 20, 30]
    good = [11, 19, 31]
    bad = [15, 25, 35]
    assert relative_mae(actual, good, bad) < 1.0


def test_pinball_loss_median():
    """Pinball at q=0.5 is half the MAE."""
    actual = np.array([1.0, 2.0, 3.0])
    forecast = np.array([1.5, 2.5, 3.5])
    expected = 0.5 * mae(actual, forecast)
    assert abs(pinball_loss(actual, forecast, 0.5) - expected) < 1e-10


def test_crps_perfect():
    """CRPS of a degenerate distribution at the true value is zero."""
    actual = np.array([5.0, 10.0])
    quantiles = np.array([0.1, 0.5, 0.9])
    qf = np.column_stack([actual] * 3)
    assert crps(actual, qf, quantiles) < 1e-10
