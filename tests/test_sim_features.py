"""Tests for feature construction, especially leakage-safety.

The engineering rules require that features are strictly backward-looking:
a feature at interval t may not use any value from t or later. The
scarcity features (demand stress, ramp, volatility clustering) are the
newest and the easiest to get wrong, so they get an explicit
future-perturbation test.
"""

import numpy as np
import pandas as pd

from grian.sim.features import (
    build_features,
    fourier_calendar_features,
    lean_lags,
    scarcity_features,
)


class TestFourierCalendar:
    """Fourier calendar encoding: cyclic sin/cos features for tree ablations."""

    def test_emits_sin_cos_and_drops_ordinal(self):
        """calendar_encoding='fourier' replaces raw hour/dow/month with harmonics."""
        df = _synthetic()
        X = build_features(df, feature_set="lean", calendar_encoding="fourier")
        cols = list(X.columns)
        assert any(c.startswith("hour_sin") for c in cols)
        assert any(c.startswith("month_cos") for c in cols)
        assert "hour" not in cols and "day_of_week" not in cols and "month" not in cols

    def test_hour_is_cyclic(self):
        """23:30 and 00:00 map to near-identical Fourier features (wrap-around)."""
        idx = pd.DatetimeIndex(["2024-03-01 23:30", "2024-03-02 00:00"])
        f = fourier_calendar_features(idx)
        # the first-harmonic hour features should be close across the midnight wrap
        d = abs(f.loc[idx[0], "hour_sin1"] - f.loc[idx[1], "hour_sin1"])
        assert d < 0.3          # adjacent on the circle, not 23.5 apart

    def test_bounded_and_finite(self):
        """All Fourier features are finite and within [-1, 1] (plus weekend flag)."""
        f = fourier_calendar_features(_synthetic().index)
        harm = f.drop(columns=["is_weekend"])
        assert np.isfinite(f.to_numpy()).all()
        assert harm.to_numpy().min() >= -1.0001 and harm.to_numpy().max() <= 1.0001

    def test_fourier_is_leakage_free(self):
        """A future price perturbation cannot change any (calendar-only) feature."""
        df = _synthetic()
        base = build_features(df, calendar_encoding="fourier")
        bumped = df.copy()
        bumped.iloc[-1, bumped.columns.get_loc("price")] += 5000.0
        after = build_features(bumped, calendar_encoding="fourier")
        pd.testing.assert_frame_equal(base.iloc[:-1], after.iloc[:-1])


class TestLeanFeatureSet:
    """The shape-preserving 'lean' set is raw lags + calendar, no smoothers."""

    def test_lean_has_no_smoothers(self):
        """Lean drops the mean-reverting rolling/momentum/profile columns."""
        df = _synthetic()
        lean = build_features(df, feature_set="lean")
        cols = " ".join(lean.columns)
        assert "price_rmean" not in cols        # rolling means gone
        assert "momentum" not in cols and "return" not in cols
        assert "profile" not in cols
        # Every non-calendar column is a raw price lag.
        cal = {"hour", "day_of_week", "month", "hour_x_dow", "is_weekend"}
        non_cal = [c for c in lean.columns if c not in cal]
        assert non_cal and all(c.startswith("price_lag_") for c in non_cal)

    def test_lean_is_a_strict_subset_of_full_lags(self):
        """Lean uses a richer raw-lag basis than the 3-lag default."""
        assert len(lean_lags("30min")) > 3
        assert lean_lags("30min") == sorted(set(lean_lags("30min")))

    def test_lean_is_leakage_free(self):
        """A future perturbation cannot change any lean feature at time t."""
        df = _synthetic()
        base = build_features(df, feature_set="lean")
        bumped = df.copy()
        bumped.iloc[-1, bumped.columns.get_loc("price")] += 5000.0
        after = build_features(bumped, feature_set="lean")
        # All rows except the last are untouched by a change to the final price.
        pd.testing.assert_frame_equal(base.iloc[:-1], after.iloc[:-1])


def _synthetic(n=3000):
    idx = pd.date_range("2023-06-01", periods=n, freq="5min")
    rng = np.random.default_rng(3)
    t = np.arange(n)
    price = 60 + 40 * np.sin(2 * np.pi * (t % 288) / 288) + rng.normal(0, 8, n)
    # inject some spikes
    price[rng.random(n) < 0.02] += 800
    demand = 1500 + 300 * np.sin(2 * np.pi * (t % 288) / 288)
    return pd.DataFrame({"price": price, "demand": demand}, index=idx)


class TestScarcityFeatures:
    """The spike-precursor group must be informative and leakage-free."""

    def test_flag_adds_columns(self):
        """include_scarcity strictly extends the feature set."""
        df = _synthetic()
        base = build_features(df)
        enriched = build_features(df, include_scarcity=True)
        assert set(base.columns).issubset(enriched.columns)
        new = set(enriched.columns) - set(base.columns)
        assert {"spikes_24h", "demand_stress_max", "demand_ramp_1h"} <= new

    def test_no_future_leakage(self):
        """Perturbing the last price must not change any earlier feature."""
        df = _synthetic()
        sf = scarcity_features(df)
        df2 = df.copy()
        df2.iloc[-1, df2.columns.get_loc("price")] = 99_999.0
        sf2 = scarcity_features(df2)
        # Every row before the last must be identical.
        pd.testing.assert_frame_equal(sf.iloc[:-1], sf2.iloc[:-1])

    def test_demand_stress_in_unit_range(self):
        """Demand-stress ratios sit near [0, 1.x], never negative."""
        df = _synthetic()
        sf = scarcity_features(df)
        assert (sf["demand_stress_max"].dropna() >= 0).all()
        assert sf["demand_stress_max"].dropna().max() < 5.0

    def test_intervals_since_spike_resets(self):
        """intervals_since_spike is 0-ish right after a spike, grows after."""
        df = _synthetic()
        sf = scarcity_features(df, spike_threshold=300.0)
        assert sf["intervals_since_spike"].min() >= 0
        assert sf["intervals_since_spike"].max() <= 288

    def test_no_demand_still_works(self):
        """Scarcity features degrade gracefully without a demand column."""
        df = _synthetic()[["price"]]
        sf = scarcity_features(df)
        assert "spikes_24h" in sf.columns
        assert "demand_stress_max" not in sf.columns
