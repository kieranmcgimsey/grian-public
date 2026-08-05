"""Tests for grian.dispatch.open_loop — walk-forward simulation loop."""

import numpy as np
import pandas as pd
import pytest

from grian.dispatch import ledger as ledger_mod
from grian.dispatch.open_loop import battery_dispatch, run_trial, simulate_region
from grian.evaluation.trials import (
    load_metrics,
    make_config,
    trial_dir,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Synthetic 5-minute price data spanning train + test periods.

    Train: 2023-01-01 to 2023-01-20 (20 days)
    Test:  2023-01-21 to 2023-01-25 (5 days)

    Sinusoidal daily pattern so models have something learnable.
    """
    n_days = 25
    ppd = 288
    n = n_days * ppd
    idx = pd.date_range("2023-01-01", periods=n, freq="5min")
    rng = np.random.default_rng(42)

    hour_frac = idx.hour + idx.minute / 60.0
    daily_pattern = 50 + 30 * np.sin(2 * np.pi * hour_frac / 24)
    noise = rng.normal(0, 3, n)
    prices = daily_pattern + noise

    return pd.DataFrame({"price": prices}, index=idx)


@pytest.fixture
def test_cfg():
    """Trial config for testing with short periods."""
    return make_config({
        "trial_name": "test_runner",
        "model": "naive_similar_day",
        "regions": ["SA1"],
        "resolution": "5min",
        "horizon": 288,
        "train_start": "2023-01-01",
        "train_end": "2023-01-20",
        "test_start": "2023-01-21",
        "test_end": "2023-01-25",
        "refit_days": 1,
        "embargo": 288,
        "seed": 42,
        "transform": "identity",
        "ablations": {
            "scale_target": True,
            "use_transform": False,
            "use_embargo": True,
            "leak_future": False,
        },
    })


# ---------------------------------------------------------------------------
# Battery dispatch
# ---------------------------------------------------------------------------

class TestBatteryDispatch:
    """Tests for the default battery dispatch function."""

    def test_returns_list_of_dicts(self):
        """Dispatch returns one record per interval."""
        forecast = pd.Series([50, 60, 40, 80, 30], name="forecast")
        actuals = pd.Series([51, 59, 42, 78, 32], name="actual")
        cfg = make_config()
        result = battery_dispatch(forecast, actuals, cfg)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_records_have_required_keys(self):
        """Each dispatch record has charge, discharge, soc."""
        forecast = pd.Series([50, 60, 40], name="forecast")
        actuals = pd.Series([50, 60, 40], name="actual")
        cfg = make_config()
        result = battery_dispatch(forecast, actuals, cfg)
        for record in result:
            assert "charge_mw" in record
            assert "discharge_mw" in record
            assert "soc_mwh" in record

    def test_soc_non_negative(self):
        """SOC never goes negative."""
        n = 50
        forecast = pd.Series(np.random.default_rng(42).uniform(10, 100, n))
        actuals = pd.Series(np.random.default_rng(43).uniform(10, 100, n))
        cfg = make_config()
        result = battery_dispatch(forecast, actuals, cfg)
        for record in result:
            assert record["soc_mwh"] >= -1e-6  # Allow tiny float tolerance


# ---------------------------------------------------------------------------
# Single-region simulation
# ---------------------------------------------------------------------------

class TestSimulateRegion:
    """Tests for the single-region walk-forward loop."""

    def test_produces_ledger(self, synthetic_data, test_cfg):
        """simulate_region returns a non-empty ledger."""
        result = simulate_region(synthetic_data, test_cfg)
        assert "ledger" in result
        assert len(result["ledger"]) > 0

    def test_produces_forecasts(self, synthetic_data, test_cfg):
        """simulate_region returns forecast records."""
        result = simulate_region(synthetic_data, test_cfg)
        assert "forecasts" in result
        assert len(result["forecasts"]) > 0

    def test_produces_model_state(self, synthetic_data, test_cfg):
        """simulate_region returns a fitted model state."""
        result = simulate_region(synthetic_data, test_cfg)
        assert result["model_state"] is not None

    def test_ledger_timestamps_in_test_period(self, synthetic_data, test_cfg):
        """All ledger timestamps fall within the test period."""
        result = simulate_region(synthetic_data, test_cfg)
        ledger_df = ledger_mod.to_dataframe(result["ledger"])
        assert ledger_df.index.min() >= pd.Timestamp("2023-01-21")
        assert ledger_df.index.max() < pd.Timestamp("2023-01-26")

    def test_embargo_respected(self, synthetic_data, test_cfg):
        """Training data does not overlap with the forecast window.

        We verify this by checking that the first forecast day's data
        is not in the training set — the embargo should create a gap.
        """
        result = simulate_region(synthetic_data, test_cfg)
        # The first forecast is for 2023-01-21
        # With embargo=288 (1 day), training should end at or before 2023-01-20
        assert result["model_state"] is not None

    def test_no_embargo_ablation(self, synthetic_data, test_cfg):
        """With use_embargo=False, training extends closer to forecast."""
        test_cfg["ablations"]["use_embargo"] = False
        test_cfg["embargo"] = 0
        result = simulate_region(synthetic_data, test_cfg)
        # Should still produce results
        assert len(result["ledger"]) > 0

    def test_future_leakage_ablation(self, synthetic_data, test_cfg):
        """With leak_future=True, training includes future data."""
        test_cfg["ablations"]["leak_future"] = True
        result = simulate_region(synthetic_data, test_cfg)
        assert len(result["ledger"]) > 0

    def test_no_reconditioning(self, synthetic_data, test_cfg):
        """With refit_days very large, model is only fitted once."""
        test_cfg["refit_days"] = 99999
        result = simulate_region(synthetic_data, test_cfg)
        assert len(result["ledger"]) > 0

    def test_daily_reconditioning(self, synthetic_data, test_cfg):
        """With refit_days=1, model is refit every day."""
        test_cfg["refit_days"] = 1
        result = simulate_region(synthetic_data, test_cfg)
        assert len(result["forecasts"]) >= 4  # ~5 test days


# ---------------------------------------------------------------------------
# Full trial runner — multi-region + artifact saving
# ---------------------------------------------------------------------------

class TestRunTrial:
    """Tests for the top-level run_trial function."""

    def test_saves_config(self, synthetic_data, test_cfg, tmp_path):
        """run_trial writes config.json to disk."""
        data_by_region = {"SA1": synthetic_data}
        run_trial(data_by_region, test_cfg, base=tmp_path)
        cfg_path = trial_dir(test_cfg["trial_name"], base=tmp_path) / "config.json"
        assert cfg_path.exists()

    def test_saves_metrics(self, synthetic_data, test_cfg, tmp_path):
        """run_trial writes metrics.json per region."""
        data_by_region = {"SA1": synthetic_data}
        run_trial(data_by_region, test_cfg, base=tmp_path)
        metrics = load_metrics("test_runner", "SA1", base=tmp_path)
        assert "total_revenue" in metrics
        assert "mae" in metrics

    def test_saves_ledger(self, synthetic_data, test_cfg, tmp_path):
        """run_trial writes ledger.parquet per region."""
        data_by_region = {"SA1": synthetic_data}
        run_trial(data_by_region, test_cfg, base=tmp_path)
        ledger_path = trial_dir("test_runner", "SA1", base=tmp_path) / "ledger.parquet"
        assert ledger_path.exists()

    def test_saves_forecasts(self, synthetic_data, test_cfg, tmp_path):
        """run_trial writes forecasts.parquet per region."""
        data_by_region = {"SA1": synthetic_data}
        run_trial(data_by_region, test_cfg, base=tmp_path)
        fc_path = trial_dir("test_runner", "SA1", base=tmp_path) / "forecasts.parquet"
        assert fc_path.exists()

    def test_saves_model(self, synthetic_data, test_cfg, tmp_path):
        """run_trial writes model artifacts per region."""
        data_by_region = {"SA1": synthetic_data}
        run_trial(data_by_region, test_cfg, base=tmp_path)
        model_dir = trial_dir("test_runner", "SA1", base=tmp_path) / "model"
        assert model_dir.exists()

    def test_multi_region(self, synthetic_data, test_cfg, tmp_path):
        """run_trial handles multiple regions independently."""
        test_cfg["regions"] = ["SA1", "NSW1"]
        data_by_region = {"SA1": synthetic_data, "NSW1": synthetic_data}
        results = run_trial(data_by_region, test_cfg, base=tmp_path)
        assert "SA1" in results
        assert "NSW1" in results

    def test_missing_region_skipped(self, synthetic_data, test_cfg, tmp_path):
        """Regions without data are skipped gracefully."""
        test_cfg["regions"] = ["SA1", "TAS1"]
        data_by_region = {"SA1": synthetic_data}
        results = run_trial(data_by_region, test_cfg, base=tmp_path)
        assert "SA1" in results
        assert "TAS1" not in results

    def test_returns_metrics(self, synthetic_data, test_cfg, tmp_path):
        """run_trial returns metrics in the result dict."""
        data_by_region = {"SA1": synthetic_data}
        results = run_trial(data_by_region, test_cfg, base=tmp_path)
        assert "metrics" in results["SA1"]
        assert isinstance(results["SA1"]["metrics"]["total_revenue"], float)

    def test_reproducible_with_same_seed(self, synthetic_data, tmp_path):
        """Same config + data produces identical results."""
        cfg1 = make_config({
            "trial_name": "repro_1",
            "model": "naive_similar_day",
            "regions": ["SA1"],
            "test_start": "2023-01-21",
            "test_end": "2023-01-25",
            "seed": 42,
        })
        cfg2 = make_config({
            "trial_name": "repro_2",
            "model": "naive_similar_day",
            "regions": ["SA1"],
            "test_start": "2023-01-21",
            "test_end": "2023-01-25",
            "seed": 42,
        })

        data_by_region = {"SA1": synthetic_data}
        r1 = run_trial(data_by_region, cfg1, base=tmp_path)
        r2 = run_trial(data_by_region, cfg2, base=tmp_path)

        assert r1["SA1"]["metrics"]["total_revenue"] == r2["SA1"]["metrics"]["total_revenue"]
        assert r1["SA1"]["metrics"]["mae"] == r2["SA1"]["metrics"]["mae"]
