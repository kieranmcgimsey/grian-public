"""Tests for the HiGHS arbitrage LP, feasibility clamp, and oracle.

Covers campaign tickets W0.1 and W0.3: the fast LP must match the
cvxpy reference, clamped execution must never create phantom energy,
and replaying the oracle's own schedule must reproduce its revenue
(capture ratio 1.0 by construction).
"""

import numpy as np
import pandas as pd
import pytest

from grian.dispatch import battery_lp as lp
from grian.dispatch import oracle
from grian.dispatch.cvxpy_reference import schedule
from grian.evaluation.analytics import capture_report

RNG = np.random.default_rng(7)


class TestSolveLpEquivalence:
    """HiGHS LP matches the cvxpy reference implementation."""

    @pytest.mark.parametrize("dt_hours", [0.5, 5.0 / 60.0])
    def test_matches_cvxpy_objective(self, dt_hours):
        """Same optimal revenue as the cvxpy model on random prices."""
        for _ in range(5):
            prices = RNG.normal(60, 40, size=48)
            ref = schedule(prices, dt_hours=dt_hours)
            fast = lp.solve_lp(
                prices,
                dt_hours=dt_hours,
                throughput_budgets=[(0, 48, 2 * 200.0)],
            )
            assert fast["status"] == "optimal"
            assert fast["revenue"] == pytest.approx(
                ref["revenue"], rel=1e-4, abs=1e-4
            )

    def test_hand_checkable_case(self):
        """Two intervals: charge cheap, discharge dear, efficiency paid."""
        # dt=1h, power 100, capacity 200, eta = sqrt(0.85) each way.
        result = lp.solve_lp(
            np.array([0.0, 100.0]),
            dt_hours=1.0,
            throughput_budgets=[(0, 2, 400.0)],
        )
        eta = np.sqrt(0.85)
        # Charge 100 MWh at $0 (stores 100*eta), discharge it all at $100:
        # discharge power = 100*eta*eta = 85 MW → revenue $8,500.
        assert result["revenue"] == pytest.approx(100 * eta * eta * 100, rel=1e-6)

    def test_soc_continuity_and_bounds(self):
        """SOC starts at soc0, stays within [0, capacity]."""
        prices = RNG.normal(60, 40, size=96)
        result = lp.solve_lp(prices, dt_hours=0.25, soc0=150.0)
        assert result["soc"][0] == 150.0
        assert (result["soc"] >= -1e-6).all()
        assert (result["soc"] <= 200.0 + 1e-6).all()

    def test_throughput_budget_binds(self):
        """Discharge energy respects the per-window budget."""
        prices = np.tile([0.0, 500.0], 24).astype(float)
        result = lp.solve_lp(
            prices, dt_hours=0.5, throughput_budgets=[(0, 48, 100.0)]
        )
        assert result["discharge"].sum() * 0.5 <= 100.0 + 1e-6

    def test_terminal_soc_pinned(self):
        """Terminal SOC constraint is honoured."""
        prices = RNG.normal(60, 40, size=48)
        result = lp.solve_lp(prices, dt_hours=0.5, terminal_soc=0.0)
        assert result["soc"][-1] == pytest.approx(0.0, abs=1e-6)


class TestClampAction:
    """Execution clamp forbids phantom energy (trap T2)."""

    def test_discharge_limited_by_stored_energy(self):
        """An empty battery cannot discharge."""
        charge, discharge, soc = lp.clamp_action(
            0.0, 100.0, 0.0, dt_hours=0.5
        )
        assert discharge == 0.0
        assert soc == 0.0

    def test_charge_limited_by_headroom(self):
        """A full battery cannot charge."""
        charge, discharge, soc = lp.clamp_action(
            100.0, 0.0, 200.0, dt_hours=0.5
        )
        assert charge == 0.0
        assert soc == 200.0

    def test_budget_limits_discharge(self):
        """Remaining daily throughput budget caps discharge."""
        charge, discharge, soc = lp.clamp_action(
            0.0, 100.0, 200.0, dt_hours=0.5, discharge_budget_mwh=10.0
        )
        assert discharge * 0.5 <= 10.0 + 1e-9

    def test_feasible_action_untouched(self):
        """A feasible plan passes through unchanged."""
        charge, discharge, soc = lp.clamp_action(
            50.0, 0.0, 100.0, dt_hours=5.0 / 60.0
        )
        assert charge == pytest.approx(50.0)
        eta = np.sqrt(0.85)
        assert soc == pytest.approx(100.0 + 50.0 * eta * 5.0 / 60.0)


class TestMeanCvarLp:
    """Mean-CVaR dispatch: EV at lam=0, worst-tail protection at lam=1."""

    def test_lam0_matches_expected_value_plan(self):
        """lam=0 maximises E[R] — same revenue as the mean-price LP."""
        paths = np.array([[10.0, 100.0], [10.0, 40.0], [10.0, 70.0]])
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        pbar = (w[:, None] * paths).sum(axis=0)
        ref = lp.solve_lp(pbar, dt_hours=0.5, terminal_soc=0.0)
        cvar = lp.solve_mean_cvar_lp(paths, w, dt_hours=0.5, lam=0.0,
                                     terminal_soc=0.0)
        ev_rev = float((w * cvar["scenario_revenue"]).sum())
        assert ev_rev == pytest.approx(ref["revenue"], rel=1e-4)

    def test_lam1_protects_the_worst_scenario(self):
        """A risky trade (good on average, bad in the tail) is avoided at lam=1.

        Scenario B sells below the buy price, so the expected-value plan takes
        a trade that loses money in B; the pure-CVaR plan should not — its
        worst-scenario revenue must be at least the EV plan's.
        """
        paths = np.array([[10.0, 100.0], [10.0, 5.0]])   # A profits, B loses
        w = np.array([0.5, 0.5])
        ev = lp.solve_mean_cvar_lp(paths, w, dt_hours=0.5, lam=0.0,
                                   terminal_soc=0.0)
        cvar = lp.solve_mean_cvar_lp(paths, w, dt_hours=0.5, lam=1.0, alpha=0.5,
                                     terminal_soc=0.0)
        assert cvar["scenario_revenue"].min() >= ev["scenario_revenue"].min() - 1e-6
        # The tail-averse plan declines the losing trade → ~no worst-case loss.
        assert cvar["scenario_revenue"].min() >= -1e-6

    def test_single_scenario_matches_solve_lp(self):
        """One scenario reduces to the deterministic LP."""
        prices = np.array([20.0, 80.0, 30.0, 90.0])
        ref = lp.solve_lp(prices, dt_hours=0.5, terminal_soc=0.0)
        cvar = lp.solve_mean_cvar_lp(prices[None, :], np.array([1.0]),
                                     dt_hours=0.5, lam=0.5, terminal_soc=0.0)
        assert cvar["scenario_revenue"][0] == pytest.approx(ref["revenue"], rel=1e-4)
        np.testing.assert_allclose(cvar["discharge"], ref["discharge"], atol=1e-6)


class TestOracle:
    """Oracle self-consistency: replaying it captures exactly 1.0."""

    def _prices(self, days=3):
        idx = pd.date_range("2023-06-01", periods=days * 288, freq="5min")
        base = 60 + 40 * np.sin(np.arange(len(idx)) * 2 * np.pi / 288)
        return pd.Series(base + RNG.normal(0, 10, len(idx)), index=idx)

    def test_oracle_replay_capture_is_one(self):
        """Executing the oracle schedule through the honest executor
        reproduces the oracle revenue."""
        prices = self._prices()
        dt = 5.0 / 60.0
        result = oracle.compute_oracle(prices, dt_hours=dt)

        # Replay through the clamp, exactly as the runner executes.
        soc, replayed = 0.0, []
        sched = result["schedule_df"]
        day = None
        for ts, row in sched.iterrows():
            if ts.date() != day:
                day, budget_left = ts.date(), 2 * 200.0
            c, d, soc = lp.clamp_action(
                row["charge_mw"], row["discharge_mw"], soc,
                dt_hours=dt, discharge_budget_mwh=budget_left,
            )
            budget_left -= d * dt
            replayed.append(row["actual_price"] * (d - c) * dt)

        assert sum(replayed) == pytest.approx(
            result["total_revenue"], rel=1e-4
        )

    def test_capture_report_on_oracle_ledger(self):
        """capture_report scores the oracle's own ledger at 1.0."""
        prices = self._prices()
        result = oracle.compute_oracle(prices, dt_hours=5.0 / 60.0)
        ledger_df = result["schedule_df"].copy()
        ledger_df["forecast_price"] = ledger_df["actual_price"]
        report = capture_report(ledger_df, result["daily_revenue"])
        assert report["capture_ratio"] == pytest.approx(1.0, abs=1e-6)
        assert report["mean_daily_spearman"] == pytest.approx(1.0, abs=1e-9)

    def test_oracle_closed_cycles_nonnegative(self):
        """No closed cycle (SOC empty→empty) loses money (trap T7).

        compute_oracle asserts this internally on return; a clean run
        proves the invariant holds for ordinary prices.
        """
        prices = self._prices()
        result = oracle.compute_oracle(prices, dt_hours=5.0 / 60.0)
        # The window as a whole starts and ends empty, so it is one big
        # closed cycle and must be non-negative.
        assert result["total_revenue"] >= -1e-6

    def test_negative_day_from_overnight_carry_is_allowed(self):
        """A single day may be net-negative when the oracle optimally
        charges cheap overnight to discharge into the next day's spike.

        This is legitimate cross-midnight arbitrage inside a block (SOC is
        pinned to 0 only at block boundaries), not a broken oracle, so the
        closed-cycle assert must accept it. Prices: a cheap day-1 followed
        by an extreme day-2 morning spike, forcing overnight accumulation.
        """
        idx = pd.date_range("2023-06-01", periods=2 * 288, freq="5min")
        price = np.full(len(idx), 20.0)          # cheap everywhere by default
        price[288 + 6 : 288 + 18] = 5000.0       # day-2 morning spike
        prices = pd.Series(price, index=idx)
        result = oracle.compute_oracle(prices, dt_hours=5.0 / 60.0)

        daily = result["daily_revenue"]
        # Day 1 buys to bank for day 2 → net-negative; the whole cycle is +.
        assert daily.iloc[0] < 0
        assert result["total_revenue"] > 0
