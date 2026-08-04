"""Tests for split-conformal calibration of the LightGBM quantile fan.

Calibration widens under-covering quantiles (here, the upper ones the raw fan
uses to under-predict spikes) so probabilistic dispatch sees spike risk. These
tests pin the two invariants that matter: it runs end-to-end and *widens the
upper tail*, and the ``_apply_conformal`` helper shifts and re-sorts correctly.
"""

import numpy as np
import pandas as pd

from grian.sim import models


def _spiky(days=120, ppd=48):
    """30-min prices (asinh) with a daily shape and frequent evening spikes."""
    idx = pd.date_range("2024-01-01", periods=days * ppd, freq="30min")
    t = np.arange(len(idx))
    rng = np.random.default_rng(1)
    price = 60 + 40 * np.sin(2 * np.pi * (t % ppd) / ppd - np.pi / 2)
    price = price + rng.normal(0, 5, len(idx))
    spike = rng.random(len(idx)) < 0.03
    price[spike] += rng.uniform(500, 8000, spike.sum())
    price = np.clip(price, 0, None)
    demand = 1200 + 250 * np.sin(2 * np.pi * (t % ppd) / ppd)
    return pd.DataFrame({"price": np.arcsinh(price), "demand": demand}, index=idx)


def _cfg(calibrate):
    mp = {"step_stride": 12, "n_estimators": 40,
          "quantiles": [0.05, 0.5, 0.9, 0.98]}
    if calibrate:
        mp = {**mp, "calibrate": True, "cal_days": 20}
    return {"model": "lightgbm_qmean", "resolution": "30min", "horizon": 48,
            "target_col": "price", "transform": "asinh", "model_params": mp}


def test_calibration_widens_the_upper_tail():
    """The calibrated fan's upper quantile sits above the raw fan's."""
    df = _spiky()
    spec = models.get_model("lightgbm_qmean")
    raw = spec["predict_fan"](spec["fit"](df, "price", _cfg(False)), df, 48)
    cal_state = spec["fit"](df, "price", _cfg(True))
    cal = spec["predict_fan"](cal_state, df, 48)

    assert cal_state["conformal_adjustments"]                 # non-empty
    assert np.mean(cal[0.98]) > np.mean(raw[0.98]) * 1.1      # tail widened up
    # Fan stays monotone across quantiles at every step.
    qs = sorted(cal)
    assert all((np.diff([cal[q][s] for q in qs]) >= -1e-6).all() for s in range(48))


def test_uncalibrated_has_no_adjustments():
    """Without the flag the fan is unchanged (adjustments are None)."""
    df = _spiky()
    spec = models.get_model("lightgbm_qmean")
    state = spec["fit"](df, "price", _cfg(False))
    assert state["conformal_adjustments"] is None


def test_apply_conformal_shifts_and_sorts():
    """Lower quantiles shift down, upper up, and the column is re-sorted."""
    qs = [0.05, 0.5, 0.9, 0.98]
    fan = {q: np.array([100.0]) for q in qs}
    adj = {0: [10.0, 0.0, 20.0, 50.0]}          # widen q0.05 down, q0.9/q0.98 up
    out = models._apply_conformal(fan, adj, qs, horizon=1)
    assert out[0.05][0] == 90.0                 # 100 - 10
    assert out[0.98][0] == 150.0                # 100 + 50
    col = [out[q][0] for q in qs]
    assert col == sorted(col)                   # monotone


def test_apply_conformal_noop_when_empty():
    """No adjustments → the fan is returned untouched."""
    qs = [0.05, 0.5, 0.9, 0.98]
    fan = {q: np.array([1.0, 2.0]) for q in qs}
    out = models._apply_conformal(fan, {}, qs, horizon=2)
    assert all(np.allclose(out[q], fan[q]) for q in qs)
