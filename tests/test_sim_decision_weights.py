"""Tests for decision-focused training weights (models._decision_weights).

Capture is earned in price spikes, but accuracy loss is dominated by the flat
95% of intervals. These weights up-weight high-price intervals so the fit
chases spike timing. The tests pin the weighting *mechanism* (crisp, not model-
dependent) plus the plumbing that carries it into a LightGBM fit.
"""

import numpy as np
import pandas as pd

from grian import models


def test_magnitude_weights_rise_with_price():
    """Higher dollar price → larger weight, and the mean is normalised to 1."""
    dollars = np.array([50.0, 100.0, 300.0, 5000.0, 20000.0])
    y = np.arcsinh(dollars)  # target lives in asinh space
    w = models._decision_weights(y, "asinh", models.SampleWeighting(scheme="magnitude"))

    assert np.all(np.diff(w) > 0)              # monotone in price
    assert np.isclose(w.mean(), 1.0)           # effective LR unchanged
    assert np.all(w > 0)


def test_strength_widens_the_spread():
    """A larger strength pulls more weight onto the high-price tail."""
    y = np.arcsinh(np.array([50.0, 100.0, 300.0, 5000.0, 20000.0]))
    weak = models._decision_weights(y, "asinh", models.SampleWeighting(strength=0.5))
    strong = models._decision_weights(y, "asinh", models.SampleWeighting(strength=4.0))
    # Top interval carries a larger share of the (mean-1) weight budget.
    assert strong[-1] > weak[-1]


def test_quantile_scheme_boosts_only_the_tail():
    """The quantile scheme flat-boosts prices above the percentile, else 1×."""
    dollars = np.concatenate([np.full(90, 50.0), np.full(10, 8000.0)])
    y = np.arcsinh(dollars)
    w = models._decision_weights(
        y, "asinh",
        models.SampleWeighting(scheme="quantile", q=0.9, strength=3.0))
    # Two distinct levels; the tail weight is the larger one.
    assert len(np.unique(np.round(w, 6))) == 2
    assert w[-1] > w[0]


def test_uniform_prices_give_uniform_weights():
    """No price dispersion → no reweighting (all weights equal to 1)."""
    y = np.arcsinh(np.full(20, 100.0))
    w = models._decision_weights(y, "asinh", models.SampleWeighting(scheme="magnitude"))
    assert np.allclose(w, 1.0)


def _spiky_data(days=30, ppd=48):
    """30-min prices with a daily shape and occasional evening spikes."""
    idx = pd.date_range("2024-01-01", periods=days * ppd, freq="30min")
    t = np.arange(len(idx))
    rng = np.random.default_rng(0)
    price = 60 + 40 * np.sin(2 * np.pi * (t % ppd) / ppd - np.pi / 2)
    price = price + rng.normal(0, 5, len(idx))
    spike = (rng.random(len(idx)) < 0.02) & ((t % ppd) > ppd * 0.7)
    price = np.where(spike, price + rng.uniform(2000, 9000, len(idx)), price)
    demand = 1200 + 250 * np.sin(2 * np.pi * (t % ppd) / ppd)
    return pd.DataFrame({"price": np.arcsinh(price), "demand": demand}, index=idx)


def test_dfl_config_fits_and_predicts():
    """lightgbm_rich_dfl fits with sample weights and forecasts finite prices."""
    data = _spiky_data()
    cfg = {
        "model": "lightgbm_rich_dfl", "resolution": "30min", "horizon": 48,
        "target_col": "price", "transform": "asinh",
        "model_params": {
            "step_stride": 24, "n_estimators": 40, "include_scarcity": True,
            "sample_weighting": {"scheme": "magnitude", "strength": 1.0},
        },
    }
    spec = models.get_model("lightgbm_rich_dfl")
    state = spec["fit"](data, "price", cfg)
    fc = spec["predict"](state, data, 48)
    assert len(fc) == 48
    assert np.isfinite(fc.values).all()
