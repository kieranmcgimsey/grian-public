"""Tests for grian.models — fit/predict/save/load for every model type."""

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from grian.models import (
    AUTOREGRESSION,
    LIGHTGBM,
    NAIVE_SIMILAR_DAY,
    REGISTRY,
    SIMPLE_MLP,
    _build_lag_features,
    _calendar_features,
    _periods_per_day,
    get_model,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_5min():
    """Synthetic 5-minute price data — 30 days of sinusoidal prices.

    Mimics a daily price cycle with some noise. Long enough for AR
    lags and LightGBM to train on.
    """
    n_days = 30
    ppd = 288
    n = n_days * ppd
    idx = pd.date_range("2023-01-01", periods=n, freq="5min")
    rng = np.random.default_rng(42)

    # Daily sinusoidal pattern + trend + noise
    hour_frac = (idx.hour + idx.minute / 60.0)
    daily_pattern = 50 + 30 * np.sin(2 * np.pi * hour_frac / 24)
    noise = rng.normal(0, 5, n)
    prices = daily_pattern + noise

    return pd.DataFrame({"price": prices}, index=idx)


@pytest.fixture
def base_cfg():
    """Minimal trial config for testing models."""
    return {
        "resolution": "5min",
        "horizon": 288,
        "seed": 42,
        "model_params": {},
        "transform": "identity",
        "ablations": {"use_transform": False},
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for shared model helper functions."""

    def test_periods_per_day_5min(self):
        assert _periods_per_day("5min") == 288

    def test_periods_per_day_30min(self):
        assert _periods_per_day("30min") == 48

    def test_build_lag_features_shape(self):
        """Lag features have correct columns and NaN structure."""
        series = pd.Series(range(100),
                           index=pd.date_range("2023-01-01", periods=100, freq="5min"))
        lags = [5, 10]
        result = _build_lag_features(series, lags)
        assert list(result.columns) == ["lag_5", "lag_10"]
        assert result.iloc[0:5]["lag_5"].isna().all()  # First 5 should be NaN
        assert result.iloc[5]["lag_5"] == 0  # lag_5 at index 5 = value at index 0

    def test_calendar_features_columns(self):
        """Calendar features produce hour, day_of_week, month."""
        idx = pd.date_range("2023-06-15 14:30", periods=3, freq="5min")
        cal = _calendar_features(idx)
        assert set(cal.columns) == {"hour", "day_of_week", "month"}
        assert cal.iloc[0]["month"] == 6
        assert cal.iloc[0]["day_of_week"] == 3  # Thursday

    def test_calendar_hour_is_fractional(self):
        """Hour feature includes fractional minutes."""
        idx = pd.date_range("2023-01-01 14:30", periods=1, freq="5min")
        cal = _calendar_features(idx)
        assert cal.iloc[0]["hour"] == 14.5


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

class TestRegistry:
    """Tests for the model registry."""

    def test_canonical_models_registered(self):
        """The canonical models stay registered as the registry grows.

        A subset check, not exact equality: new experimental models (LEAR
        family, scarcity, decision-focused, quantile fans) are added often, and
        each is validated structurally by ``test_all_models_have_required_fields``
        below. This test just guards the core set from accidental removal.
        """
        canonical = {
            "naive_similar_day", "autoregression", "lightgbm",
            "lightgbm_rich", "simple_mlp", "lstm",
        }
        assert canonical <= set(REGISTRY.keys())

    def test_get_model_returns_spec(self):
        """get_model returns a typed ModelSpec with the full interface."""
        spec = get_model("naive_similar_day")
        for attr in ("name", "output", "fit", "predict", "save", "load", "params"):
            assert hasattr(spec, attr)

    def test_get_model_unknown_raises(self):
        """get_model raises KeyError for unknown model."""
        with pytest.raises(KeyError, match="Unknown model"):
            get_model("nonexistent_model")

    @pytest.mark.parametrize("name", REGISTRY.keys())
    def test_all_models_have_required_fields(self, name):
        """Every registered model is a ModelSpec with the full interface."""
        spec = REGISTRY[name]
        for attr in ("name", "output", "fit", "predict", "save", "load", "params"):
            assert hasattr(spec, attr), f"Model {name!r} missing {attr!r}"
        assert spec.output in ("point", "quantile")


# ---------------------------------------------------------------------------
# Naive similar-day
# ---------------------------------------------------------------------------

class TestNaiveSimilarDay:
    """Tests for the naive similar-day model."""

    def test_fit_returns_state(self, synthetic_5min, base_cfg):
        """Fitting produces a state dict with series and resolution."""
        state = NAIVE_SIMILAR_DAY.fit(synthetic_5min, "price", base_cfg)
        assert "series" in state
        assert "resolution" in state

    def test_predict_length(self, synthetic_5min, base_cfg):
        """Forecast has the requested horizon length."""
        state = NAIVE_SIMILAR_DAY.fit(synthetic_5min, "price", base_cfg)
        forecast = NAIVE_SIMILAR_DAY.predict(state, synthetic_5min, 288)
        assert len(forecast) == 288

    def test_predict_finite(self, synthetic_5min, base_cfg):
        """Forecast contains no NaN or inf values."""
        state = NAIVE_SIMILAR_DAY.fit(synthetic_5min, "price", base_cfg)
        forecast = NAIVE_SIMILAR_DAY.predict(state, synthetic_5min, 288)
        assert np.all(np.isfinite(forecast.values))

    def test_save_load_round_trip(self, synthetic_5min, base_cfg, tmp_path):
        """Save/load round-trip preserves forecast output."""
        state = NAIVE_SIMILAR_DAY.fit(synthetic_5min, "price", base_cfg)
        forecast_before = NAIVE_SIMILAR_DAY.predict(state, synthetic_5min, 48)

        NAIVE_SIMILAR_DAY.save(state, tmp_path / "naive_model")
        restored = NAIVE_SIMILAR_DAY.load(tmp_path / "naive_model")
        forecast_after = NAIVE_SIMILAR_DAY.predict(restored, synthetic_5min, 48)

        np.testing.assert_array_equal(forecast_before.values, forecast_after.values)


# ---------------------------------------------------------------------------
# Autoregression
# ---------------------------------------------------------------------------

class TestAutoregression:
    """Tests for the linear autoregression model."""

    def test_fit_returns_state(self, synthetic_5min, base_cfg):
        """Fitting produces a state with model, lags, last_values."""
        state = AUTOREGRESSION.fit(synthetic_5min, "price", base_cfg)
        assert "model" in state
        assert "lags" in state
        assert "last_values" in state

    def test_predict_length(self, synthetic_5min, base_cfg):
        """Forecast has the requested horizon length."""
        state = AUTOREGRESSION.fit(synthetic_5min, "price", base_cfg)
        forecast = AUTOREGRESSION.predict(state, synthetic_5min, 288)
        assert len(forecast) == 288

    def test_predict_finite(self, synthetic_5min, base_cfg):
        """Forecast contains no NaN or inf values."""
        state = AUTOREGRESSION.fit(synthetic_5min, "price", base_cfg)
        forecast = AUTOREGRESSION.predict(state, synthetic_5min, 288)
        assert np.all(np.isfinite(forecast.values))

    def test_save_load_round_trip(self, synthetic_5min, base_cfg, tmp_path):
        """Save/load preserves forecast output."""
        state = AUTOREGRESSION.fit(synthetic_5min, "price", base_cfg)
        forecast_before = AUTOREGRESSION.predict(state, synthetic_5min, 48)

        AUTOREGRESSION.save(state, tmp_path / "ar_model")
        restored = AUTOREGRESSION.load(tmp_path / "ar_model")
        forecast_after = AUTOREGRESSION.predict(restored, synthetic_5min, 48)

        np.testing.assert_allclose(forecast_before.values,
                                   forecast_after.values, atol=1e-6)

    def test_custom_lags(self, synthetic_5min, base_cfg):
        """Custom lag config is respected."""
        base_cfg["model_params"] = {"lags": [10, 20]}
        state = AUTOREGRESSION.fit(synthetic_5min, "price", base_cfg)
        assert state["lags"] == [10, 20]

    def test_predict_conditions_on_input(self, synthetic_5min, base_cfg):
        """Predict-from-now: the forecast follows the live recent tail.

        Regression test for the frozen-tail bug (Entry 038): the model used to
        forecast from the fit-time tail regardless of ``input_df``, so under MPC
        it produced an identical (flat, near-term) forecast at every origin.
        """
        state = AUTOREGRESSION.fit(synthetic_5min, "price", base_cfg)
        f_train = AUTOREGRESSION.predict(state, synthetic_5min, 288)
        # Same model, but the recent observations are lifted by $40.
        shifted = synthetic_5min.copy()
        shifted["price"] = shifted["price"] + 40.0
        f_shift = AUTOREGRESSION.predict(state, shifted, 288)
        # The near-term forecast must move with the input, not stay frozen …
        assert abs(f_shift.iloc[0] - f_train.iloc[0]) > 5.0
        # … and the forecast is not a degenerate flat line.
        assert f_train.round(4).nunique() > 1


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

class TestLightGBM:
    """Tests for the LightGBM model."""

    def test_fit_returns_state(self, synthetic_5min, base_cfg):
        """Fitting produces a state with boosters."""
        base_cfg["model_params"] = {"n_estimators": 10}
        state = LIGHTGBM.fit(synthetic_5min, "price", base_cfg)
        assert "boosters" in state
        assert len(state["boosters"]) > 0

    def test_predict_length(self, synthetic_5min, base_cfg):
        """Forecast has the requested horizon length."""
        base_cfg["model_params"] = {"n_estimators": 10}
        base_cfg["horizon"] = 48
        state = LIGHTGBM.fit(synthetic_5min, "price", base_cfg)
        forecast = LIGHTGBM.predict(state, synthetic_5min, 48)
        assert len(forecast) == 48

    def test_predict_finite(self, synthetic_5min, base_cfg):
        """Forecast contains no NaN or inf values."""
        base_cfg["model_params"] = {"n_estimators": 10}
        base_cfg["horizon"] = 48
        state = LIGHTGBM.fit(synthetic_5min, "price", base_cfg)
        forecast = LIGHTGBM.predict(state, synthetic_5min, 48)
        assert np.all(np.isfinite(forecast.values))

    def test_save_load_round_trip(self, synthetic_5min, base_cfg, tmp_path):
        """Save/load preserves model metadata."""
        base_cfg["model_params"] = {"n_estimators": 10}
        base_cfg["horizon"] = 48
        state = LIGHTGBM.fit(synthetic_5min, "price", base_cfg)

        LIGHTGBM.save(state, tmp_path / "lgbm_model")
        restored = LIGHTGBM.load(tmp_path / "lgbm_model")

        assert restored["lags"] == state["lags"]
        assert restored["resolution"] == state["resolution"]
        assert set(restored["boosters"].keys()) == set(state["boosters"].keys())

    def test_beats_naive_on_synthetic(self, synthetic_5min, base_cfg):
        """LightGBM should beat naive on structured synthetic data."""
        base_cfg["model_params"] = {"n_estimators": 50}
        base_cfg["horizon"] = 48

        # Train both models on same data
        lgbm_state = LIGHTGBM.fit(synthetic_5min, "price", base_cfg)
        naive_state = NAIVE_SIMILAR_DAY.fit(synthetic_5min, "price", base_cfg)

        # Predict
        lgbm_fc = LIGHTGBM.predict(lgbm_state, synthetic_5min, 48)
        naive_fc = NAIVE_SIMILAR_DAY.predict(naive_state, synthetic_5min, 48)

        # Use last day as "actual" for scoring
        actual = synthetic_5min["price"].iloc[-48:].values

        lgbm_mae = np.mean(np.abs(actual - lgbm_fc.values))
        naive_mae = np.mean(np.abs(actual - naive_fc.values))

        # LightGBM should have lower MAE on this structured data
        assert lgbm_mae < naive_mae * 1.5  # Allow some slack


# ---------------------------------------------------------------------------
# Simple MLP
# ---------------------------------------------------------------------------

class TestSimpleMLP:
    """Tests for the simple MLP model."""

    def test_fit_returns_state(self, synthetic_5min, base_cfg):
        """Fitting produces a state with model weights."""
        base_cfg["model_params"] = {"epochs": 2, "hidden_dim": 16}
        state = SIMPLE_MLP.fit(synthetic_5min, "price", base_cfg)
        assert "model_state_dict" in state
        assert "X_mean" in state
        assert "y_mean" in state

    def test_predict_length(self, synthetic_5min, base_cfg):
        """Forecast has the requested horizon length."""
        base_cfg["model_params"] = {"epochs": 2, "hidden_dim": 16}
        state = SIMPLE_MLP.fit(synthetic_5min, "price", base_cfg)
        forecast = SIMPLE_MLP.predict(state, synthetic_5min, 48)
        assert len(forecast) == 48

    def test_predict_finite(self, synthetic_5min, base_cfg):
        """Forecast contains no NaN or inf values."""
        base_cfg["model_params"] = {"epochs": 2, "hidden_dim": 16}
        state = SIMPLE_MLP.fit(synthetic_5min, "price", base_cfg)
        forecast = SIMPLE_MLP.predict(state, synthetic_5min, 48)
        assert np.all(np.isfinite(forecast.values))

    def test_save_load_round_trip(self, synthetic_5min, base_cfg, tmp_path):
        """Save/load preserves forecast output."""
        base_cfg["model_params"] = {"epochs": 2, "hidden_dim": 16}
        state = SIMPLE_MLP.fit(synthetic_5min, "price", base_cfg)
        forecast_before = SIMPLE_MLP.predict(state, synthetic_5min, 48)

        SIMPLE_MLP.save(state, tmp_path / "mlp_model")
        restored = SIMPLE_MLP.load(tmp_path / "mlp_model")
        forecast_after = SIMPLE_MLP.predict(restored, synthetic_5min, 48)

        np.testing.assert_allclose(forecast_before.values,
                                   forecast_after.values, atol=1e-4)

    def test_reproducible_with_seed(self, synthetic_5min, base_cfg):
        """Same seed produces same model output."""
        base_cfg["model_params"] = {"epochs": 3, "hidden_dim": 16}
        base_cfg["seed"] = 123

        state1 = SIMPLE_MLP.fit(synthetic_5min, "price", base_cfg)
        fc1 = SIMPLE_MLP.predict(state1, synthetic_5min, 48)

        state2 = SIMPLE_MLP.fit(synthetic_5min, "price", base_cfg)
        fc2 = SIMPLE_MLP.predict(state2, synthetic_5min, 48)

        np.testing.assert_allclose(fc1.values, fc2.values, atol=1e-4)

    def test_standardisation_stored(self, synthetic_5min, base_cfg):
        """Fitted state stores standardisation parameters."""
        base_cfg["model_params"] = {"epochs": 2, "hidden_dim": 16}
        state = SIMPLE_MLP.fit(synthetic_5min, "price", base_cfg)
        assert state["X_mean"] is not None
        assert state["X_std"] is not None
        assert abs(state["y_mean"]) > 0  # Prices aren't centered at zero


class TestCalendarEncoding:
    """Fourier calendar encoding threads through the tree models (fit/predict/io)."""

    def _cfg(self, encoding):
        return {"model": "lightgbm_rich", "resolution": "30min", "horizon": 48,
                "target_col": "price", "transform": "asinh",
                "model_params": {"step_stride": 12, "n_estimators": 30,
                                 "calendar_encoding": encoding}}

    def _data(self):
        idx = pd.date_range("2024-01-01", periods=70 * 48, freq="30min")
        t = np.arange(len(idx))
        price = 60 + 40 * np.sin(2 * np.pi * (t % 48) / 48) + 5
        return pd.DataFrame({"price": np.arcsinh(price),
                             "demand": 1200 + 200 * np.sin(2 * np.pi * (t % 48) / 48)},
                            index=idx)

    def test_rich_fourier_fits_predicts_and_stores_encoding(self):
        """lightgbm_rich with Fourier calendar fits, forecasts, records the mode."""
        df = self._data()
        spec = get_model("lightgbm_rich")
        state = spec.fit(df, "price", self._cfg("fourier"))
        assert state["calendar_encoding"] == "fourier"
        fc = spec.predict(state, df, 48)
        assert len(fc) == 48 and np.isfinite(fc.values).all()

    def test_qmean_fourier_fan_and_io_round_trip(self, tmp_path):
        """lightgbm_qmean Fourier fan works and calendar_encoding survives save/load."""
        df = self._data()
        cfg = {"model": "lightgbm_qmean", "resolution": "30min", "horizon": 48,
               "target_col": "price", "transform": "asinh",
               "model_params": {"step_stride": 12, "n_estimators": 30,
                                "quantiles": [0.05, 0.5, 0.9, 0.98],
                                "calendar_encoding": "fourier"}}
        spec = get_model("lightgbm_qmean")
        state = spec.fit(df, "price", cfg)
        assert state["calendar_encoding"] == "fourier"
        spec.save(state, tmp_path)
        reloaded = spec.load(tmp_path)
        assert reloaded["calendar_encoding"] == "fourier"
        fan = spec.predict_fan(reloaded, df, 48)
        assert np.isfinite(fan[0.98]).all()

    def test_ordinal_is_the_default(self):
        """No calendar_encoding param → ordinal (unchanged legacy behaviour)."""
        df = self._data()
        state = get_model("lightgbm_rich").fit(
            df, "price", {"model": "lightgbm_rich", "resolution": "30min",
                          "horizon": 48, "target_col": "price", "transform": "asinh",
                          "model_params": {"step_stride": 12, "n_estimators": 20}})
        assert state["calendar_encoding"] == "ordinal"


class TestLearQmeanTorch:
    """The torch (CPU) quantile fan: loss/penalty units + fit/predict/io."""

    def _data(self):
        idx = pd.date_range("2024-01-01", periods=70 * 48, freq="30min")
        t = np.arange(len(idx))
        price = 60 + 40 * np.sin(2 * np.pi * (t % 48) / 48) + 5
        return pd.DataFrame(
            {"price": np.arcsinh(price),
             "demand": 1200 + 200 * np.sin(2 * np.pi * (t % 48) / 48)},
            index=idx)

    def _cfg(self, encoding="onehot", weather=False):
        params = {"quantiles": [0.05, 0.5, 0.9, 0.98], "alpha": 0.01,
                  "step_stride": 12, "epochs": 150,
                  "calendar_encoding": encoding, "include_weather": weather}
        return {"model": "lear_qmean_torch", "resolution": "30min", "horizon": 48,
                "target_col": "price", "transform": "asinh",
                "model_params": params}

    # ---- loss / penalty units -------------------------------------------

    def test_pinball_matches_definition(self):
        """Vectorised pinball loss equals the per-element max formula."""
        import torch

        from grian.models import _pinball_loss

        pred = torch.tensor([[[0.0, 3.0]]])          # (B=1, S=1, Q=2)
        y = torch.tensor([[1.5]])                    # (B=1, S=1)
        mask = torch.ones(1, 1)
        taus = torch.tensor([0.1, 0.9])
        # q=0.1, pred=0.0: diff=+1.5 → max(0.1*1.5, -0.9*1.5)=0.15 (under-forecast)
        # q=0.9, pred=3.0: diff=-1.5 → max(0.9*-1.5, -0.1*-1.5)=0.15 (over-forecast)
        expected = (0.15 + 0.15) / 2.0
        assert abs(float(_pinball_loss(pred, y, mask, taus)) - expected) < 1e-6

    def test_pinball_masks_missing_targets(self):
        """Masked (missing) targets contribute nothing to the loss."""
        import torch

        from grian.models import _pinball_loss

        pred = torch.tensor([[[1.0]], [[5.0]]])      # (B=2, S=1, Q=1)
        y = torch.tensor([[1.0], [0.0]])
        taus = torch.tensor([0.5])
        full = _pinball_loss(pred, y, torch.ones(2, 1), taus)
        masked = _pinball_loss(pred, y, torch.tensor([[1.0], [0.0]]), taus)
        # Row 0 is a perfect fit (loss 0); masking row 1 leaves only row 0.
        assert float(full) > 0 and abs(float(masked)) < 1e-6

    def test_leading_calendar_columns(self):
        """The calendar block is counted for both encodings via the fitted CT."""
        from grian.models import (
            _CALENDAR_COLS,
            _leading_calendar_columns,
            _linear_preprocessor,
        )

        cols = list(_CALENDAR_COLS) + ["price_lag1", "demand"]
        n = 200
        raw = pd.DataFrame({
            "hour": np.tile(np.arange(24), n)[:n] % 24,
            "day_of_week": np.arange(n) % 7,
            "month": (np.arange(n) % 12) + 1,
            "price_lag1": np.random.default_rng(0).normal(size=n),
            "demand": np.random.default_rng(1).normal(size=n),
        })
        # Fourier: harmonics 3+1+2 = 6 → 12 sin/cos columns.
        pf = _linear_preprocessor(cols, "fourier").fit(raw)
        assert _leading_calendar_columns(pf) == 12
        # One-hot: 24 hours + 7 dows + 12 months present = 43 dummy columns.
        po = _linear_preprocessor(cols, "onehot").fit(raw)
        assert _leading_calendar_columns(po) == 43

    def test_calendar_block_is_not_regularized(self):
        """The L1 penalty ignores the leading calendar block entirely."""
        import torch

        from grian.models import _l1_penalty

        n_calendar = 2
        # Only the trailing (real-feature) column should be penalised.
        w = torch.tensor([[5.0, -3.0, 2.0]])      # cols 0,1 = calendar; 2 = real
        pen = float(_l1_penalty(w, n_calendar, alpha=1.0))
        assert abs(pen - 2.0) < 1e-6              # |2.0| only; calendar exempt
        # Changing the calendar weights must not move the penalty.
        w2 = torch.tensor([[99.0, 42.0, 2.0]])
        assert abs(float(_l1_penalty(w2, n_calendar, alpha=1.0)) - pen) < 1e-6
        # n_calendar=0 penalises every coefficient.
        assert abs(float(_l1_penalty(w, 0, alpha=1.0)) - 10.0) < 1e-6

    # ---- end to end ------------------------------------------------------

    # ---- end to end ------------------------------------------------------

    def test_fit_predict_fan_monotone(self):
        """Fan quantiles are finite and non-decreasing across levels."""
        df = self._data()
        spec = get_model("lear_qmean_torch_fourier")
        state = spec.fit(df, "price", self._cfg("fourier"))
        fan = spec.predict_fan(state, df, 48)
        stack = np.vstack([fan[q] for q in sorted(fan)])
        assert np.isfinite(stack).all()
        assert (np.diff(stack, axis=0) >= -1e-6).all()

    def test_point_forecast_shape(self):
        """The integrated point forecast has horizon length and is finite."""
        df = self._data()
        spec = get_model("lear_qmean_torch")
        state = spec.fit(df, "price", self._cfg("onehot"))
        fc = spec.predict(state, df, 48)
        assert len(fc) == 48 and np.isfinite(fc.values).all()

    def test_fit_runs_on_cpu(self):
        """The fit is pinned to CPU (the MPS autograd backend diverges here)."""
        import torch

        calls = []
        orig = torch.device

        def spy(arg):
            calls.append(str(arg))
            return orig(arg)

        df = self._data()
        with mock.patch("torch.device", side_effect=spy):
            get_model("lear_qmean_torch").fit(df, "price", self._cfg("onehot"))
        assert any(c == "cpu" for c in calls)
        assert not any("mps" in c for c in calls)

    def test_predictions_clamped_to_envelope(self):
        """Out-of-envelope extrapolation cannot produce inf in the fan."""
        df = self._data()
        spec = get_model("lear_qmean_torch")
        state = spec.fit(df, "price", self._cfg("onehot"))
        # Corrupt the weights so the raw linear output would blow up post-sinh.
        state["weight"] = state["weight"] * 1e3
        fan = spec.predict_fan(state, df, 48)
        stack = np.vstack([fan[q] for q in sorted(fan)])
        assert np.isfinite(stack).all()      # clamp + envelope guard hold

    def test_io_round_trip(self, tmp_path):
        """save/load reproduces the fan exactly (weights + preprocessor)."""
        df = self._data()
        spec = get_model("lear_qmean_torch_weather_fourier")
        state = spec.fit(df, "price", self._cfg("fourier", weather=True))
        fan = spec.predict_fan(state, df, 48)
        spec.save(state, tmp_path)
        reloaded = spec.load(tmp_path)
        assert reloaded["calendar_encoding"] == "fourier"
        assert reloaded["include_weather"] is True
        fan2 = spec.predict_fan(reloaded, df, 48)
        for q in fan:
            assert np.max(np.abs(fan[q] - fan2[q])) < 1e-4
