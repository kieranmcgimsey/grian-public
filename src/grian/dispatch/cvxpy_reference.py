"""Battery dispatch optimisation and economic valuation.

Implements the arbitrage linear program, capture ratio, and a rolling
model-predictive-control loop that values a price forecast the way a
battery operator does.
"""

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike


def schedule(
    prices: ArrayLike,
    power_mw: float = 100.0,
    duration_hours: float = 2.0,
    efficiency: float = 0.85,
    max_cycles: int = 2,
    dt_hours: float = 0.5,
    soc0: float = 0.0,
    terminal_soc: float | None = None,
) -> dict:
    """Solve the battery arbitrage LP for a single day.

    Given a price profile, finds the charge/discharge schedule that
    maximises revenue subject to power, energy, efficiency, and cycle
    constraints.

    This is the readable cvxpy reference implementation; the hot path
    uses the equivalent HiGHS model in ``grian.dispatch.battery_lp``.

    Args:
        prices: Price vector for the scheduling horizon ($/MWh).
        power_mw: Rated power capacity (MW).
        duration_hours: Energy capacity in hours at rated power.
        efficiency: Round-trip efficiency (0–1).
        max_cycles: Maximum full charge/discharge cycles per day.
        dt_hours: Interval length in hours (0.5 for 30-min, 1/12 for 5-min).
        soc0: State of charge at the start of the horizon (MWh).
        terminal_soc: If given, pin the final SOC to this value (MWh).

    Returns:
        Dict with keys ``charge`` (MW), ``discharge`` (MW), ``soc`` (MWh),
        ``revenue`` ($), and ``status`` (solver status string).
    """
    prices = np.asarray(prices, dtype=float)
    T = len(prices)
    capacity_mwh = power_mw * duration_hours
    eta = np.sqrt(efficiency)
    dt = dt_hours

    # Decision variables
    charge = cp.Variable(T, nonneg=True)
    discharge = cp.Variable(T, nonneg=True)
    soc = cp.Variable(T + 1, nonneg=True)

    # Objective: maximize revenue (discharge - charge) * price * dt
    revenue = prices @ (discharge - charge) * dt
    objective = cp.Maximize(revenue)

    constraints = [
        # Power limits
        charge <= power_mw,
        discharge <= power_mw,
        # SOC limits
        soc <= capacity_mwh,
        # Initial SOC
        soc[0] == soc0,
    ]
    if terminal_soc is not None:
        constraints.append(soc[T] == terminal_soc)

    # SOC dynamics: soc[t+1] = soc[t] + charge[t]*eta*dt - discharge[t]/eta*dt
    for t in range(T):
        constraints.append(
            soc[t + 1] == soc[t] + charge[t] * eta * dt - discharge[t] / eta * dt
        )

    # Cycle limit: total energy throughput <= max_cycles * capacity
    constraints.append(
        cp.sum(discharge) * dt <= max_cycles * capacity_mwh
    )

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return {
        "charge": np.array(charge.value) if charge.value is not None else np.zeros(T),
        "discharge": (
            np.array(discharge.value)
            if discharge.value is not None
            else np.zeros(T)
        ),
        "soc": np.array(soc.value) if soc.value is not None else np.zeros(T + 1),
        "revenue": float(prob.value) if prob.value is not None else 0.0,
        "status": prob.status,
    }


def capture_ratio(
    revenue: float,
    perfect_revenue: float,
) -> float:
    """Capture ratio: realised revenue as a fraction of perfect foresight.

    Args:
        revenue: Revenue achieved by the model's dispatch.
        perfect_revenue: Revenue from dispatching against actual prices.

    Returns:
        Ratio in [0, 1]; 1.0 means perfect foresight was matched.
    """
    if perfect_revenue == 0:
        return 0.0
    return revenue / perfect_revenue


def rolling_mpc(
    prices_actual: ArrayLike,
    forecaster,
    horizon: int = 48,
    **battery_kwargs,
) -> dict:
    """Run a rolling model-predictive-control dispatch loop.

    At each step, obtains a forecast from *forecaster*, solves the LP,
    executes one period, then re-forecasts. Accumulates revenue and
    state-of-charge over the full period.

    Args:
        prices_actual: Full actual price series.
        forecaster: Callable that returns a price forecast from the current
            position.
        horizon: Forecast/scheduling horizon in periods.
        **battery_kwargs: Passed to :func:`schedule`.

    Returns:
        Dict with ``total_revenue``, ``soc_history``, ``actions``, and
        ``capture_ratio`` against perfect foresight.
    """
    prices_actual = np.asarray(prices_actual, dtype=float)
    T = len(prices_actual)
    dt = 0.5  # 30-minute intervals

    power_mw = battery_kwargs.get("power_mw", 100.0)
    duration_hours = battery_kwargs.get("duration_hours", 2.0)
    efficiency = battery_kwargs.get("efficiency", 0.85)
    eta = np.sqrt(efficiency)
    capacity_mwh = power_mw * duration_hours

    total_revenue = 0.0
    soc_history = [0.0]
    actions = []
    current_soc = 0.0

    for t in range(T):
        # Get forecast from current position
        forecast = forecaster(t, horizon)
        forecast = np.asarray(forecast, dtype=float)

        # Pad or truncate forecast to horizon length
        if len(forecast) < horizon:
            forecast = np.pad(
                forecast, (0, horizon - len(forecast)), constant_values=forecast[-1]
            )

        # Solve LP over forecast horizon
        result = schedule(forecast, **battery_kwargs)

        if result["status"] != "optimal":
            # If solver fails, do nothing this period
            actions.append({"charge": 0.0, "discharge": 0.0})
            soc_history.append(current_soc)
            continue

        # Execute only the first period's action
        charge_t = float(result["charge"][0])
        discharge_t = float(result["discharge"][0])

        # Clamp to feasible SOC limits
        # Energy added to SOC by charging
        energy_in = charge_t * eta * dt
        # Energy removed from SOC by discharging
        energy_out = discharge_t / eta * dt

        # Ensure SOC stays within bounds
        if current_soc + energy_in - energy_out > capacity_mwh:
            energy_in = capacity_mwh - current_soc + energy_out
            charge_t = energy_in / (eta * dt)
        if current_soc + energy_in - energy_out < 0:
            energy_out = current_soc + energy_in
            discharge_t = energy_out * eta / dt

        new_soc = current_soc + charge_t * eta * dt - discharge_t / eta * dt
        new_soc = max(0.0, min(capacity_mwh, new_soc))

        # Revenue from actual price
        period_revenue = prices_actual[t] * (discharge_t - charge_t) * dt
        total_revenue += period_revenue

        actions.append({"charge": charge_t, "discharge": discharge_t})
        current_soc = new_soc
        soc_history.append(current_soc)

    # Perfect foresight benchmark
    perfect = schedule(prices_actual, **battery_kwargs)
    perfect_rev = perfect["revenue"] if perfect["status"] == "optimal" else 0.0

    return {
        "total_revenue": total_revenue,
        "soc_history": soc_history,
        "actions": actions,
        "capture_ratio": capture_ratio(total_revenue, perfect_rev),
    }


def main() -> None:
    """Run this module as a CLI (exposes its public callables)."""
    from grian._cli import run_module_cli

    run_module_cli(globals())


if __name__ == "__main__":
    main()
