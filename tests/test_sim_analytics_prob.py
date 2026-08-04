"""Tests for probabilistic forecast-quality metrics (CRPS, coverage)."""

import numpy as np
import pytest

from grian.sim.analytics import crps_from_quantiles, pinball_loss, quantile_coverage


def test_pinball_known_value():
    """Pinball loss matches the hand-computed value for an over-forecast."""
    # actual 10, predicted 0.9-quantile 8 → e=2, loss = max(.9*2, -.1*2) = 1.8
    assert pinball_loss(np.array([10.0]), np.array([8.0]), 0.9) == pytest.approx(1.8)


def test_pinball_zero_when_exact():
    """A perfect quantile prediction has zero pinball loss."""
    a = np.array([1.0, 5.0, 9.0])
    assert pinball_loss(a, a, 0.5) == pytest.approx(0.0)


def test_crps_zero_for_perfect_fan():
    """A fan that nails the actual at every quantile has ~zero CRPS."""
    a = np.array([10.0, 20.0, 30.0])
    fan = {0.1: a, 0.5: a, 0.9: a}
    assert crps_from_quantiles(a, fan) == pytest.approx(0.0)


def test_crps_positive_and_ordered():
    """A wider (worse) fan scores a higher CRPS than a tighter one."""
    a = np.array([10.0, 20.0, 30.0])
    tight = {0.1: a - 1, 0.5: a, 0.9: a + 1}
    wide = {0.1: a - 10, 0.5: a, 0.9: a + 10}
    assert crps_from_quantiles(a, wide) > crps_from_quantiles(a, tight) > 0


def test_coverage_perfectly_calibrated():
    """Coverage recovers the nominal levels on a calibrated Gaussian fan."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 20000)
    fan = {0.1: np.full_like(a, -1.2816),
           0.5: np.zeros_like(a),
           0.9: np.full_like(a, 1.2816)}
    cov = quantile_coverage(a, fan)
    assert cov[0.1] == pytest.approx(0.1, abs=0.02)
    assert cov[0.5] == pytest.approx(0.5, abs=0.02)
    assert cov[0.9] == pytest.approx(0.9, abs=0.02)
