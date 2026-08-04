"""Tests for probabilistic dispatch fusion (grian.sim.dispatch_prob).

The load-bearing properties: quantile levels map to a proper probability
distribution, and scenario actions fuse in decision space so that the CVaR
knob interpolates monotonically from the expected action (risk-neutral) to
the robust worst-case action (trade only where every scenario agrees).
"""

import numpy as np
import pytest

from grian.sim.dispatch_prob import (
    combine_scenario_actions,
    quantile_gate_prices,
    quantile_weights,
)


class TestQuantileWeights:
    """Quantile levels → a discrete probability distribution."""

    def test_sums_to_one(self):
        """Weights over any fan sum to 1."""
        w = quantile_weights([0.05, 0.5, 0.9, 0.98])
        assert w.sum() == pytest.approx(1.0)

    def test_median_carries_most_mass(self):
        """The central level gets the largest interval of probability."""
        w = quantile_weights([0.05, 0.5, 0.9, 0.98])
        assert w.argmax() == 1  # 0.5 sits on the widest neighbour interval

    def test_symmetric_fan_is_symmetric(self):
        """A fan symmetric about 0.5 yields symmetric weights."""
        w = quantile_weights([0.1, 0.5, 0.9])
        assert w[0] == pytest.approx(w[2])


class TestCombineScenarioActions:
    """Fusing per-scenario LP actions in decision space."""

    def _scenarios(self):
        # Two scenarios, one horizon step. Scenario A discharges, B is idle.
        charges = np.array([[0.0], [10.0]])      # A idle-charge, B charges 10
        discharges = np.array([[20.0], [0.0]])   # A discharges 20, B idle
        weights = np.array([0.5, 0.5])
        return charges, discharges, weights

    def test_ev_is_probability_weighted_net(self):
        """EV returns the probability-weighted net action, split into legs."""
        charges, discharges, w = self._scenarios()
        ch, dis = combine_scenario_actions(charges, discharges, w, "ev")
        # net = 0.5*(20-0) + 0.5*(0-10) = 5  → discharge 5, charge 0
        assert dis[0] == pytest.approx(5.0)
        assert ch[0] == pytest.approx(0.0)

    def test_no_simultaneous_charge_and_discharge(self):
        """Fused legs are never both positive in the same interval."""
        charges, discharges, w = self._scenarios()
        for mode in ("ev", "cvar"):
            ch, dis = combine_scenario_actions(charges, discharges, w, mode, lam=0.3)
            assert np.all((ch == 0) | (dis == 0))
            assert np.all(ch >= 0) and np.all(dis >= 0)

    def test_cvar_lambda0_equals_ev(self):
        """lam=0 recovers the expected action exactly."""
        charges, discharges, w = self._scenarios()
        ev = combine_scenario_actions(charges, discharges, w, "ev")
        cvar0 = combine_scenario_actions(charges, discharges, w, "cvar", lam=0.0)
        assert cvar0[0] == pytest.approx(ev[0])
        assert cvar0[1] == pytest.approx(ev[1])

    def test_cvar_lambda1_is_robust_worst_case(self):
        """lam=1 is the net robust action: min charge / min discharge."""
        charges, discharges, w = self._scenarios()
        ch, dis = combine_scenario_actions(charges, discharges, w, "cvar", lam=1.0)
        # net_robust = min(disch) - min(charge) = 0 - 0 = 0 → no trade
        assert dis[0] == pytest.approx(0.0)
        assert ch[0] == pytest.approx(0.0)

    def test_cvar_interpolates_monotonically(self):
        """Rising lam moves the net action from EV toward the worst case."""
        # Scenarios agree on direction (both discharge) but disagree on size.
        charges = np.array([[0.0], [0.0]])
        discharges = np.array([[20.0], [4.0]])
        w = np.array([0.5, 0.5])
        nets = [
            combine_scenario_actions(charges, discharges, w, "cvar", lam=lm)[1][0]
            for lm in (0.0, 0.5, 1.0)
        ]
        # EV = 12, robust = min = 4; monotonically decreasing toward 4.
        assert nets[0] == pytest.approx(12.0)
        assert nets[2] == pytest.approx(4.0)
        assert nets[0] > nets[1] > nets[2]

    def test_unknown_mode_raises(self):
        """An unrecognised fusion mode is rejected."""
        charges, discharges, w = self._scenarios()
        with pytest.raises(ValueError):
            combine_scenario_actions(charges, discharges, w, "bogus")


def test_quantile_gate_still_direction_pessimistic():
    """Regression guard: the gate compresses spreads toward the median."""
    fan = {
        0.05: np.array([10.0, 10.0]),
        0.5: np.array([20.0, 100.0]),
        0.9: np.array([30.0, 200.0]),
        0.98: np.array([40.0, 300.0]),
    }
    gated = quantile_gate_prices(fan, q_low=0.05, q_high=0.9)
    # ref = median([20, 100]) = 60. Step 0 (20 < ref) is a likely buy → take
    # the high quantile (30); step 1 (100 >= ref) is a likely sell → take the
    # low quantile (10). Only confident spread survives.
    assert gated[0] == pytest.approx(30.0)
    assert gated[1] == pytest.approx(10.0)
