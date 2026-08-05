"""Tests for grian.dashboard bundle helpers.

Covers the data-assembly functions the dashboard depends on: the name-derived
stub config (so config-less run_common_eval trials still load), trial-dir
enumeration, the day-ahead-fan full-trajectory filter, and the quantile-fan
loader that reads the saved testbed fans.
"""


import numpy as np
import pandas as pd

from grian import dashboard as dash

# ---------------------------------------------------------------------------
# _stub_config — recover a trial's settings from its name
# ---------------------------------------------------------------------------

class TestStubConfig:
    def test_parses_model_features_from_name(self):
        cfg = dash._stub_config("lear_weather_fourier__mpc30")
        assert cfg["model"] == "lear_weather_fourier"
        assert cfg["model_params"]["include_weather"] is True
        assert cfg["model_params"]["calendar_encoding"] == "fourier"
        # mpc30 is a real MPC → an mpc dict is present
        assert cfg["mpc"] and cfg["mpc"]["reforecast_every"] == 1

    def test_ordinal_and_plain(self):
        assert dash._stub_config("autoregression_ordinal__openloop"
                                 )["model_params"]["calendar_encoding"] == "ordinal"
        plain = dash._stub_config("autoregression__openloop")
        assert plain["model_params"] == {}          # no weather / special encoding
        assert plain["mpc"] is None                 # open-loop → no MPC block

    def test_spike_gate_executor(self):
        mpc = dash._stub_config("lear__mpc_spike")["mpc"]
        assert mpc["observe_gate"] == 3000.0 and mpc["reforecast_every"] == 48


# ---------------------------------------------------------------------------
# _all_trial_names — every trial dir with results, config.json or not
# ---------------------------------------------------------------------------

class TestAllTrialNames:
    def test_lists_dirs_with_region_metrics(self, tmp_path):
        for name in ("model_a__openloop", "model_b__mpc30"):
            d = tmp_path / name / "SA1"
            d.mkdir(parents=True)
            (d / "metrics.json").write_text("{}")
        # a dir with no metrics.json is ignored …
        (tmp_path / "empty__x" / "SA1").mkdir(parents=True)
        # … and the _oracle helper dir is excluded
        (tmp_path / "_oracle" / "SA1").mkdir(parents=True)
        (tmp_path / "_oracle" / "SA1" / "metrics.json").write_text("{}")
        assert dash._all_trial_names(tmp_path) == [
            "model_a__openloop", "model_b__mpc30"]

    def test_missing_dir(self, tmp_path):
        assert dash._all_trial_names(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# _fans — keep only full committed day-ahead trajectories
# ---------------------------------------------------------------------------

def _fan_frame(origins_steps):
    """Build a long-format forecasts frame: {origin: n_steps}."""
    rows = []
    for origin, n in origins_steps.items():
        for step in range(n):
            rows.append({"origin": pd.Timestamp(origin), "step": step,
                         "forecast": 50.0 + step, "actual": 55.0 + step})
    return pd.DataFrame(rows)


class TestFans:
    def test_drops_degenerate_one_step_origins(self):
        # One full 48-step trajectory (open-loop) and one 1-step origin (MPC).
        df = _fan_frame({"2026-06-01": 48, "2026-06-02": 1})
        fans = dash._fans(df, pd.Timestamp("2026-05-01"))
        assert len(fans) == 1 and fans[0]["origin"] == "2026-06-01"
        assert len(fans[0]["forecast"]) == 48

    def test_all_one_step_returns_empty(self):
        # Every-interval MPC saves only near-term steps → no day-ahead plan.
        df = _fan_frame({"2026-06-01": 1, "2026-06-02": 1})
        assert dash._fans(df, pd.Timestamp("2026-05-01")) == []

    def test_respects_cutoff(self):
        df = _fan_frame({"2026-06-01": 48, "2026-06-10": 48})
        fans = dash._fans(df, pd.Timestamp("2026-06-05"))
        assert [f["origin"] for f in fans] == ["2026-06-10"]


# ---------------------------------------------------------------------------
# _quantile_fans — load the saved testbed fan for a model
# ---------------------------------------------------------------------------

class TestQuantileFans:
    def _write_fan(self, path, model, ppd=48, n_days=3):
        rows = []
        start = pd.Timestamp("2026-06-01")
        for d in range(n_days):
            origin = (start + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
            for q in (0.05, 0.5, 0.9, 0.98):
                for step in range(ppd):
                    rows.append((origin, q, step, 100.0 * q + step))
        df = pd.DataFrame(rows, columns=["origin", "quantile", "step", "price"])
        (path).mkdir(parents=True, exist_ok=True)
        df.to_parquet(path / f"{model}__full__30min.parquet")

    def _sim_price(self):
        idx = pd.date_range("2026-06-01", periods=48 * 5, freq="30min")
        return pd.Series(np.arange(len(idx), dtype=float), index=idx)

    def test_loads_fan_with_quantile_keys(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, "_TESTBED_FANS", tmp_path)
        self._write_fan(tmp_path, "lightgbm_qmean", n_days=3)
        fans = dash._quantile_fans("lightgbm_qmean", self._sim_price(), n_days=2)
        assert fans and len(fans) == 2                 # last 2 daily origins
        row = fans[0]
        assert {"q05", "q50", "q90", "q98"} <= set(row) and "actual" in row
        assert len(row["actual"]) == 48                # full day-ahead horizon
        assert len(row["q98"]) == 48

    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, "_TESTBED_FANS", tmp_path)
        assert dash._quantile_fans("nope", self._sim_price()) is None

    def test_actual_padded_at_data_end(self, tmp_path, monkeypatch):
        # sim price ends only 5 steps into the last origin's horizon → the fan
        # is still full length, with the actual padded (None) beyond the data.
        monkeypatch.setattr(dash, "_TESTBED_FANS", tmp_path)
        self._write_fan(tmp_path, "m", n_days=1)
        idx = pd.date_range("2026-06-01", periods=5, freq="30min")
        short = pd.Series(np.arange(5, dtype=float), index=idx)
        fans = dash._quantile_fans("m", short, n_days=1)
        assert len(fans[0]["q50"]) == 48               # full fan …
        assert len(fans[0]["actual"]) == 48            # … actual padded to match
        assert fans[0]["actual"][10] is None           # beyond the data end
