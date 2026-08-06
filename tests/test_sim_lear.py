"""Tests for the LEAR family (regularised linear on rich features)."""

import numpy as np
import pandas as pd

from grian import models


def _make_data(days=30, ppd=48):
    """30-min synthetic prices with a daily shape + demand column."""
    idx = pd.date_range("2024-01-01", periods=days * ppd, freq="30min")
    t = np.arange(len(idx))
    price = 60 + 40 * np.sin(2 * np.pi * (t % ppd) / ppd - np.pi / 2)
    price = price + np.random.default_rng(0).normal(0, 5, len(idx))
    demand = 1200 + 250 * np.sin(2 * np.pi * (t % ppd) / ppd)
    return pd.DataFrame({"price": price, "demand": demand}, index=idx)


def _cfg(model, **params):
    return {
        "model": model, "resolution": "30min", "horizon": 48,
        "target_col": "price",
        # Few steps → fast CV fits in the test.
        "model_params": {"step_stride": 24, **params},
    }


def test_lear_fits_and_predicts():
    """LEAR produces a finite day-ahead forecast of the right length."""
    data = _make_data()
    spec = models.get_model("lear")
    state = spec.fit(data, "price", _cfg("lear"))
    assert state["estimator"] == "lasso"        # LEAR = Lasso
    fc = spec.predict(state, data, 48)
    assert len(fc) == 48
    assert np.isfinite(fc.values).all()


def test_estimator_inferred_from_name():
    """ridge / elasticnet registry names select their estimator."""
    data = _make_data()
    for name, kind in [("ridge", "ridge"), ("elasticnet", "elasticnet")]:
        state = models.get_model(name).fit(data, "price", _cfg(name))
        assert state["estimator"] == kind


def test_lear_beats_flat_forecast():
    """LEAR tracks the daily shape better than a flat mean forecast."""
    data = _make_data()
    spec = models.get_model("lear")
    train, actual = data.iloc[:-48], data["price"].iloc[-48:]
    state = spec.fit(train, "price", _cfg("lear"))
    fc = spec.predict(state, train, 48)
    lear_mae = np.mean(np.abs(fc.values - actual.values))
    flat_mae = np.mean(np.abs(train["price"].mean() - actual.values))
    assert lear_mae < flat_mae


def test_lear_save_load_roundtrip(tmp_path):
    """Persisted LEAR reloads and reproduces its forecast."""
    data = _make_data()
    spec = models.get_model("lear")
    state = spec.fit(data, "price", _cfg("lear"))
    fc1 = spec.predict(state, data, 48)
    spec.save(state, tmp_path)
    reloaded = spec.load(tmp_path)
    fc2 = spec.predict(reloaded, data, 48)
    np.testing.assert_allclose(fc1.values, fc2.values, rtol=1e-9)


def _qcfg(**params):
    return {
        "model": "lear_qmean", "resolution": "30min", "horizon": 48,
        "target_col": "price", "transform": "asinh",
        "model_params": {"step_stride": 24,
                         "quantiles": [0.05, 0.5, 0.9, 0.98], **params},
    }


def test_lear_qmean_emits_monotone_fan():
    """Quantile-LEAR returns a dollar-space fan with quantiles ordered."""
    data = _make_data()
    spec = models.get_model("lear_qmean")
    # Fit expects asinh-space target (runner transforms upstream).
    tdata = data.copy()
    tdata["price"] = np.arcsinh(tdata["price"].values)
    state = spec.fit(tdata, "price", _qcfg())
    fan = spec.predict_fan(state, tdata, 48)
    assert set(fan) == {0.05, 0.5, 0.9, 0.98}
    stacked = np.vstack([fan[q] for q in sorted(fan)])
    # Non-crossing: each higher quantile ≥ the one below, at every step.
    assert (np.diff(stacked, axis=0) >= -1e-6).all()
    # Inverted to dollars → back in a plausible price range, not asinh units.
    assert np.nanmax(stacked) > 20


def test_lear_qmean_is_registered_as_fan_model():
    """The probabilistic LEAR exposes predict_fan for scenario dispatch."""
    assert models.get_model("lear_qmean").predict_fan is not None
    assert models.get_model("lear_qmean_weather").predict_fan is not None


def test_lear_qmean_fit_is_deprecated():
    """The sklearn quantile LEAR warns and points at the torch replacement."""
    import warnings

    data = _make_data()
    tdata = data.copy()
    tdata["price"] = np.arcsinh(tdata["price"].values)
    spec = models.get_model("lear_qmean")
    # simplefilter("always") defeats Python's once-per-location dedup, which
    # another test in this module may already have tripped.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec.fit(tdata, "price", _qcfg())
    assert any(issubclass(w.category, DeprecationWarning)
               and "lear_qmean_torch" in str(w.message) for w in caught)


def test_calendar_encodings_fit_and_predict():
    """LEAR fits and forecasts under one-hot, Fourier, and ordinal calendars."""
    data = _make_data()
    for enc in ("onehot", "fourier", "ordinal"):
        state = models.get_model("lear").fit(
            data, "price", _cfg("lear", calendar_encoding=enc))
        assert state["calendar_encoding"] == enc
        fc = models.get_model("lear").predict(state, data, 48)
        assert len(fc) == 48 and np.isfinite(fc.values).all()


def test_onehot_is_the_linear_default():
    """Linear models default to one-hot calendar (not ordinal)."""
    data = _make_data()
    state = models.get_model("lear").fit(data, "price", _cfg("lear"))
    assert state["calendar_encoding"] == "onehot"


def test_fourier_calendar_shape():
    """The Fourier encoder emits 2·Σharmonics features for [hour,dow,month]."""
    from grian.models import _fourier_calendar
    arr = np.array([[13.5, 2, 6], [0.0, 6, 12]])  # hour, dow, month
    feats = _fourier_calendar(arr)
    # harmonics 3+1+2 = 6 → 12 sin/cos columns.
    assert feats.shape == (2, 12)
    assert np.isfinite(feats).all()


def test_autoregression_baseline_encodings():
    """The AR baseline defaults to one-hot and supports Fourier/ordinal."""
    data = _make_data()
    spec = models.get_model("autoregression")
    for enc, name in [("onehot", "autoregression"),
                      ("fourier", "autoregression_fourier"),
                      ("ordinal", "autoregression_ordinal")]:
        cfg = {"model": name, "resolution": "30min", "horizon": 48,
               "target_col": "price",
               "model_params": {"lags": [48, 96, 336],
                                "calendar_encoding": enc}}
        state = models.get_model(name).fit(data, "price", cfg)
        assert state["calendar_encoding"] == enc
        fc = models.get_model(name).predict(state, data, 48)
        assert len(fc) == 48 and np.isfinite(fc.values).all()

    # Default (no calendar_encoding given) must be one-hot.
    default = spec.fit(data, "price",
                          {"model": "autoregression", "resolution": "30min",
                           "horizon": 48, "target_col": "price",
                           "model_params": {"lags": [48, 96, 336]}})
    assert default["calendar_encoding"] == "onehot"
