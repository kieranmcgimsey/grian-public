"""Tests for the MPC executor and predict-from-now (W1.1, W1.2).

The load-bearing properties: models must forecast from the end of the
data they are handed (not from fit time), the MPC loop must respect
physics and daily cycle budgets, and with a perfect forecaster MPC
revenue must approach the oracle.
"""

import numpy as np
import pandas as pd
import pytest

from grian import models
from grian.dispatch import oracle
from grian.dispatch.mpc import _day_budgets, simulate_region_mpc

RNG = np.random.default_rng(11)


def _make_data(days=14, ppd=288):
    """Sinusoidal daily price shape plus noise, with a demand column."""
    idx = pd.date_range("2023-06-01", periods=days * ppd, freq="5min")
    t = np.arange(len(idx))
    price = 60 + 50 * np.sin(2 * np.pi * (t % ppd) / ppd - np.pi / 2)
    price = price + RNG.normal(0, 5, len(idx))
    demand = 1500 + 300 * np.sin(2 * np.pi * (t % ppd) / ppd)
    return pd.DataFrame({"price": price, "demand": demand}, index=idx)


def _base_cfg(data, model="naive_similar_day", **over):
    cfg = {
        "trial_name": "mpc_test",
        "model": model,
        "regions": ["SA1"],
        "resolution": "5min",
        "target_col": "price",
        "horizon": 288,
        "transform": "identity",
        "refit_days": 7,
        "seed": 42,
        "test_start": str(data.index[-3 * 288].date()),
        "test_end": str(data.index[-1].date()),
        "dispatch": {
            "power_mw": 100.0, "duration_hours": 2.0,
            "efficiency": 0.85, "max_cycles": 2,
        },
        "mpc": {"resolve_every": 6, "reforecast_every": 72},
        "model_params": {},
    }
    cfg.update(over)
    return cfg


class TestPredictFromNow:
    """W1.1 — predict() must key off the end of input_df."""

    def test_naive_uses_input_df(self):
        """The naive forecast follows the supplied data, not fit state."""
        data = _make_data()
        cfg = _base_cfg(data)
        spec = models.get_model("naive_similar_day")
        # Fit on the first half only.
        state = spec["fit"](data.iloc[: 7 * 288], "price", cfg)
        # Predict handing over everything up to the end.
        fc = spec["predict"](state, data, 288)
        expected = data["price"].iloc[-7 * 288: -6 * 288].values
        np.testing.assert_allclose(fc.values, expected)

    def test_naive_falls_back_without_input(self):
        """No input_df → fit-time behaviour is preserved."""
        data = _make_data()
        cfg = _base_cfg(data)
        spec = models.get_model("naive_similar_day")
        state = spec["fit"](data.iloc[: 7 * 288], "price", cfg)
        fc = spec["predict"](state, None, 288)
        assert len(fc) == 288

    def test_lgbm_rich_uses_input_tail(self):
        """lightgbm_rich builds features from the supplied tail."""
        pytest.importorskip("lightgbm")
        data = _make_data(days=21)
        cfg = _base_cfg(
            data, model="lightgbm_rich",
            model_params={"n_estimators": 10, "learning_rate": 0.1},
        )
        spec = models.get_model("lightgbm_rich")
        state = spec["fit"](data.iloc[: 14 * 288], "price", cfg)
        fc_stale = spec["predict"](state, None, 288)
        fc_fresh = spec["predict"](state, data, 288)
        # Fresh forecast must be anchored at the end of the data.
        assert fc_fresh.index[0] > data.index[-1]
        # And must differ from the stale one (different feature rows).
        assert not np.allclose(fc_stale.values, fc_fresh.values)


class TestDayBudgets:
    """Rolling-horizon per-day budget construction (hour-based steps)."""

    def test_midnight_start_uniform_steps(self):
        """48 half-hour steps from midnight → one budget row per day."""
        dts = np.full(96, 0.5)
        budgets = _day_budgets(0.0, dts, 400.0, 400.0)
        assert budgets == [(0, 48, 400.0), (48, 96, 400.0)]

    def test_mid_day_start_with_partial_budget(self):
        """Starting at 18:00, today's rows get the remaining budget."""
        dts = np.full(60, 0.5)  # 30 hours of half-hour steps
        budgets = _day_budgets(18.0, dts, 150.0, 400.0)
        assert budgets[0] == (0, 12, 150.0)   # 6h left of today
        assert budgets[1] == (12, 60, 400.0)  # tomorrow

    def test_telescoped_steps_cross_midnight(self):
        """Variable-length steps assign to the day of their start."""
        # 12 five-minute steps (1h) then 30-min blocks, starting 23:00.
        dts = np.concatenate([np.full(12, 1 / 12), np.full(8, 0.5)])
        budgets = _day_budgets(23.0, dts, 100.0, 400.0)
        assert budgets[0] == (0, 12, 100.0)
        assert budgets[1] == (12, 20, 400.0)

    def test_exhausted_budget_clamps_to_zero(self):
        budgets = _day_budgets(0.0, np.full(288, 1 / 12), -5.0, 400.0)
        assert budgets[0][2] == 0.0


class TestMpcLoop:
    """W1.2 — the executor's physical and economic invariants."""

    def test_physics_and_budget(self):
        """SOC in bounds; daily discharge within the cycle budget."""
        data = _make_data()
        cfg = _base_cfg(data)
        result = simulate_region_mpc(data, cfg)
        ledger_df = pd.DataFrame(result["ledger"]).set_index("timestamp")
        assert (ledger_df["soc_mwh"] >= -1e-6).all()
        assert (ledger_df["soc_mwh"] <= 200.0 + 1e-6).all()
        dt = 5.0 / 60.0
        daily_discharge = (
            (ledger_df["discharge_mw"] * dt).resample("D").sum()
        )
        assert (daily_discharge <= 400.0 + 1e-6).all()

    def test_perfect_forecaster_approaches_oracle(self):
        """With an oracle forecaster, MPC captures ≈ 1.0.

        This is the executor's end-to-end correctness proof: if the
        forecast is exact, the only gap vs the oracle is the rolling
        horizon boundary, which on smooth data is tiny.
        """
        data = _make_data(days=10)
        cfg = _base_cfg(data)
        test_start = pd.Timestamp(cfg["test_start"])
        test_end = pd.Timestamp(cfg["test_end"]) + pd.Timedelta(days=1)

        # A "model" that returns the actual future prices.
        spec = {
            "name": "cheat",
            "output": "point",
            "fit": lambda df, col, c: {"n": len(df)},
            "predict": lambda st, df, h: pd.Series(
                data["price"].iloc[len(df): len(df) + h].values
            ),
            "save": lambda st, p: None,
            "load": lambda p: None,
        }
        models.REGISTRY["cheat"] = spec
        try:
            cfg["model"] = "cheat"
            result = simulate_region_mpc(data, cfg)
        finally:
            del models.REGISTRY["cheat"]

        ledger_df = pd.DataFrame(result["ledger"]).set_index("timestamp")
        revenue = ledger_df["revenue"].sum()

        mask = (data.index >= test_start) & (data.index < test_end)
        oracle_result = oracle.compute_oracle(
            data.loc[mask, "price"], dt_hours=5.0 / 60.0
        )
        capture = revenue / oracle_result["total_revenue"]
        assert capture == pytest.approx(1.0, abs=0.02)


def _register_cheat_fan(data):
    """A deterministic fan model: scaled copies of the true future prices.

    Returns a spec ready to drop into ``models.REGISTRY``. The fan is a
    fixed multiple of the actual horizon prices per quantile, so runs are
    reproducible without training.
    """
    mults = {0.05: 0.6, 0.5: 1.0, 0.9: 1.4, 0.98: 1.8}

    def predict_fan(st, df, h):
        future = data["price"].iloc[len(df): len(df) + h].to_numpy()
        return {q: m * future for q, m in mults.items()}

    return {
        "name": "cheatfan",
        "output": "quantile",
        "fit": lambda df, col, c: {"n": len(df)},
        "predict": lambda st, df, h: pd.Series(
            data["price"].iloc[len(df): len(df) + h].values),
        "predict_fan": predict_fan,
        "save": lambda st, p: None,
        "load": lambda p: None,
    }


class TestFanReplay:
    """The test-bed fast path: record a fan, then replay dispatch over it."""

    def _cfg(self, data, mpc_over):
        cfg = _base_cfg(data, model="cheatfan")
        cfg["mpc"] = {"resolve_every": 6, "reforecast_every": 6, **mpc_over}
        return cfg

    def test_replay_reproduces_live_run_exactly(self):
        """Replaying the recorded fan reproduces the live run to the cent.

        This is the load-bearing guarantee of the test bed: the cached-fan
        fast path runs the *identical* dispatch code as a full run, so a
        replay must match a live run that used the same fan and dispatch.
        """
        data = _make_data(days=12)
        models.REGISTRY["cheatfan"] = _register_cheat_fan(data)
        try:
            cfg = self._cfg(data, {"dispatch_mode": "scenario_ev"})
            live = simulate_region_mpc(data, cfg, record_fans=True)
            replay = simulate_region_mpc(data, cfg, fan_cache=live["fans"])
        finally:
            del models.REGISTRY["cheatfan"]

        live_df = pd.DataFrame(live["ledger"]).set_index("timestamp")
        replay_df = pd.DataFrame(replay["ledger"]).set_index("timestamp")
        assert replay_df["revenue"].sum() == pytest.approx(
            live_df["revenue"].sum(), rel=1e-9)
        np.testing.assert_allclose(
            replay_df["discharge_mw"].to_numpy(),
            live_df["discharge_mw"].to_numpy())

    def test_replay_skips_the_model(self):
        """With a fan cache the model is never fit (model_state stays None)."""
        data = _make_data(days=12)
        models.REGISTRY["cheatfan"] = _register_cheat_fan(data)
        try:
            cfg = self._cfg(data, {"dispatch_mode": "scenario"})
            fans = simulate_region_mpc(data, cfg, record_fans=True)["fans"]
            replay = simulate_region_mpc(data, cfg, fan_cache=fans)
        finally:
            del models.REGISTRY["cheatfan"]
        assert replay["model_state"] is None
        assert len(fans) > 0

    def test_dispatch_modes_differ_over_same_fan(self):
        """The grid actually varies: robust and EV give different dispatch."""
        data = _make_data(days=12)
        models.REGISTRY["cheatfan"] = _register_cheat_fan(data)
        try:
            build_cfg = self._cfg(data, {"dispatch_mode": "scenario"})
            fans = simulate_region_mpc(
                data, build_cfg, record_fans=True)["fans"]
            revs = {}
            for mode in ("scenario", "scenario_ev", "scenario_cvar"):
                cfg = self._cfg(data, {"dispatch_mode": mode})
                out = simulate_region_mpc(data, cfg, fan_cache=fans)
                revs[mode] = pd.DataFrame(out["ledger"])["revenue"].sum()
        finally:
            del models.REGISTRY["cheatfan"]
        # EV trades more than the robust min, so the ledgers must differ.
        assert revs["scenario"] != pytest.approx(revs["scenario_ev"])

    def test_mean_cvar_lp_runs_over_fan(self):
        """The joint mean-CVaR LP executor produces a physically valid ledger."""
        data = _make_data(days=12)
        models.REGISTRY["cheatfan"] = _register_cheat_fan(data)
        try:
            build_cfg = self._cfg(data, {"dispatch_mode": "scenario"})
            fans = simulate_region_mpc(
                data, build_cfg, record_fans=True)["fans"]
            cfg = self._cfg(data, {"dispatch_mode": "mean_cvar",
                                   "cvar_lambda": 0.5, "cvar_alpha": 0.5})
            out = simulate_region_mpc(data, cfg, fan_cache=fans)
        finally:
            del models.REGISTRY["cheatfan"]
        led = pd.DataFrame(out["ledger"]).set_index("timestamp")
        assert len(led) > 0
        assert (led["soc_mwh"] >= -1e-6).all()
        assert (led["soc_mwh"] <= 200.0 + 1e-6).all()


class TestObserveGate:
    """observe_gate pins the current price only for genuine spikes (Entry 034/035).

    The bug: pinning step 0 to the actual price at every re-solve makes the LP
    trade the forecast residual and collapses capture. The fix gates that pin to
    prices >= observe_gate. These tests pin the gating logic by two crisp
    equivalences on identical inputs.
    """

    def _cfg(self, data, **mpc):
        cfg = _base_cfg(data, model="naive_similar_day")
        cfg["mpc"] = {"resolve_every": 1, "reforecast_every": 288, **mpc}
        return cfg

    def test_gate_above_all_prices_equals_observe_off(self):
        """A gate above every price never fires → identical to observe_present off."""
        data = _make_data(days=10)
        pmax = float(data["price"].max())
        gated = simulate_region_mpc(
            data, self._cfg(data, observe_present=True, observe_gate=pmax + 1e6))
        off = simulate_region_mpc(
            data, self._cfg(data, observe_present=False))
        g = pd.DataFrame(gated["ledger"])["revenue"].sum()
        o = pd.DataFrame(off["ledger"])["revenue"].sum()
        assert g == pytest.approx(o)

    def test_gate_zero_equals_always_observe(self):
        """gate=0 fires on every step → identical to legacy always-observe."""
        data = _make_data(days=10)
        zero = simulate_region_mpc(
            data, self._cfg(data, observe_present=True, observe_gate=0.0))
        always = simulate_region_mpc(
            data, self._cfg(data, observe_present=True))
        z = pd.DataFrame(zero["ledger"])["revenue"].sum()
        a = pd.DataFrame(always["ledger"])["revenue"].sum()
        assert z == pytest.approx(a)


class TestFanCheckpoint:
    """Long fan builds must be crash-safe and report progress (design fix)."""

    def test_fan_checkpoint_fires_periodically(self):
        """fan_checkpoint saves the accumulated fan every N reforecasts."""
        data = _make_data(days=12)
        models.REGISTRY["cheatfan"] = _register_cheat_fan(data)
        saves = []  # (num_origins) at each checkpoint call
        cfg = _base_cfg(data, model="cheatfan")
        cfg["mpc"] = {"resolve_every": 6, "reforecast_every": 6,
                      "dispatch_mode": "scenario"}
        try:
            out = simulate_region_mpc(
                data, cfg, record_fans=True,
                fan_checkpoint=(lambda f: saves.append(len(f)), 2))
        finally:
            del models.REGISTRY["cheatfan"]
        n = len(out["fans"])
        assert saves, "checkpoint never fired"
        assert saves == sorted(saves)               # monotic non-decreasing
        assert all(s % 2 == 0 for s in saves)       # fired every 2 origins
        assert saves[-1] <= n                        # never claims more than exist
