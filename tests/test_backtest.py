"""Tests for grian.backtest rolling-origin backtesting."""

import numpy as np
import pandas as pd

from grian.backtest import rolling_origin


def test_rolling_origin_importable():
    """The rolling_origin function is importable."""
    assert callable(rolling_origin)


def test_rolling_origin_basic():
    """Rolling origin produces correct number of windows on simple data."""
    idx = pd.date_range("2020-01-01", periods=200, freq="30min")
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {"price": rng.normal(50, 10, 200)}, index=idx,
    )

    def dummy_model(train, horizon):
        return np.zeros(horizon)

    results = rolling_origin(
        data=data,
        model_fn=dummy_model,
        train_start="2020-01-01",
        test_start="2020-01-03",
        test_end="2020-01-04",
        horizon=4,
        step=4,
        embargo=4,
    )
    assert len(results) > 0
    assert "origin" in results[0]
    assert "actual" in results[0]
    assert "forecast" in results[0]
