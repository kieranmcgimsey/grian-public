"""Fast battery arbitrage LP (linear program) and feasibility clamping.

A *linear program* is an optimisation with a linear objective and linear
constraints; it is solved exactly and quickly. This module casts battery
dispatch as one: choose how much to charge and discharge in every interval so
that arbitrage revenue is maximised, subject to the battery's physical limits.

This is the hot-path replacement for the cvxpy model in
``grian.dispatch.schedule``. It writes the same problem out with explicit
state-of-charge (SOC — the energy currently stored, in MWh) variables and
*sparse* constraint matrices, solved by the HiGHS solver via
``scipy.optimize.linprog``. This hand-written sparse formulation was chosen over
the readable cvxpy reference (``grian.dispatch.schedule``): it is **~10–50× faster**
and, unlike cvxpy, scales to the full-window oracle, while remaining provably
equivalent — ``tests/test_sim_lp.py`` checks the two produce the same optimum at
both resolutions. Sparse means we only store the non-zero constraint
coefficients, which is what lets the same formulation scale from a single
trading day (n_steps = 288 at 5-min resolution) up to the full-window oracle
(n_steps ≈ 35,000) without building a dense matrix that would not fit in memory.

All energy accounting uses a caller-supplied ``dt_hours`` (the interval length
in hours) — never a hardcoded interval length. Hardcoding it once silently
multiplied revenue by the wrong factor and voided a dozen early results
(experiment log, Entry 013).
"""

from collections.abc import Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def solve_lp(
    prices: np.ndarray,
    *,
    dt_hours: float | np.ndarray,
    power_mw: float = 100.0,
    capacity_mwh: float = 200.0,
    efficiency: float = 0.85,
    soc0: float = 0.0,
    terminal_soc: float | None = None,
    throughput_budgets: Sequence[tuple[int, int, float]] | None = None,
) -> dict:
    """Solve the battery arbitrage LP with HiGHS.

    Maximises ``sum(prices * (discharge - charge)) * dt_hours`` — i.e. sell
    energy when prices are high, buy when they are low — subject to power,
    energy, round-trip-efficiency, and discharge-throughput (cycle) limits.

    Args:
        prices: Price vector for the horizon ($/MWh), length ``n_steps``.
        dt_hours: Interval length in hours (1/12 for 5-min, 0.5 for 30-min), or
            an array of length ``n_steps`` for telescoping horizons with mixed
            step durations.
        power_mw: Rated charge/discharge power (MW) — the per-interval cap.
        capacity_mwh: Usable energy capacity (MWh) — the upper SOC bound.
        efficiency: Round-trip efficiency (fraction of energy surviving a full
            charge-then-discharge cycle); its square root is applied to each
            direction (see the SOC dynamics below).
        soc0: State of charge at the start of the horizon (MWh).
        terminal_soc: If given, the SOC at the end of the horizon is pinned to
            this value; if None, the terminal SOC is free.
        throughput_budgets: Optional list of ``(start, end, budget_mwh)``
            constraints: total *discharged* energy over the interval range
            ``[start, end)`` must not exceed ``budget_mwh``. Used for per-day
            cycle limits.

    Returns:
        Dict with ``charge`` (MW per step), ``discharge`` (MW per step),
        ``soc`` (MWh, length ``n_steps + 1`` including the starting ``soc0``),
        ``revenue`` ($), and ``status``.
    """
    prices = np.asarray(prices, dtype=float)
    n_steps = len(prices)
    # One-way efficiency: the round trip loses (1 - efficiency), and we split
    # that loss evenly by taking the square root and applying it once on the way
    # in (charging) and once on the way out (discharging).
    one_way_efficiency = np.sqrt(efficiency)
    # Per-step interval length in hours (broadcast a scalar to a length-n vector
    # so mixed-duration telescoped horizons work with the same code path).
    dt_per_step = np.broadcast_to(np.asarray(dt_hours, dtype=float), (n_steps,))

    # Decision variables are laid out as one flat vector:
    #   x = [ charge(n_steps) , discharge(n_steps) , soc(n_steps) ]
    # so charge[t] is at index t, discharge[t] at n_steps + t, and soc[t]
    # (the stored energy *after* interval t) at 2*n_steps + t.
    n_vars = 3 * n_steps

    # linprog minimises, so we negate the revenue we want to maximise. Charging
    # spends money (+price), discharging earns it (-price); SOC has no price.
    objective_coeffs = np.concatenate([
        prices * dt_per_step,        # charging costs money
        -prices * dt_per_step,       # discharging earns money
        np.zeros(n_steps),           # SOC does not appear in the objective
    ])

    # Equality constraints encode the SOC dynamics, one row per interval:
    #   soc[t] = soc[t-1] + eff*dt*charge[t] - (dt/eff)*discharge[t]
    #   (eff = one_way_efficiency)
    # rearranged to "= 0" (or "= soc0" for the first interval, which has no
    # soc[t-1] term). We build the sparse matrix as (row, col, value) triples.
    eq_rows, eq_cols, eq_vals, eq_rhs = [], [], [], []
    for t in range(n_steps):
        # Coefficients on charge[t], discharge[t], soc[t] in row t.
        eq_rows += [t, t, t]
        eq_cols += [t, n_steps + t, 2 * n_steps + t]
        eq_vals += [
            -one_way_efficiency * dt_per_step[t],   # energy added by charging
            dt_per_step[t] / one_way_efficiency,    # energy removed by discharging
            1.0,                                    # the soc[t] variable itself
        ]
        if t > 0:
            # Subtract the previous interval's stored energy, soc[t-1].
            eq_rows.append(t)
            eq_cols.append(2 * n_steps + t - 1)
            eq_vals.append(-1.0)
            eq_rhs.append(0.0)
        else:
            # First interval: soc[0] - (charge/discharge terms) = soc0.
            eq_rhs.append(soc0)
    A_eq = sparse.coo_matrix(
        (eq_vals, (eq_rows, eq_cols)), shape=(n_steps, n_vars)
    ).tocsr()
    b_eq = np.array(eq_rhs)

    # Inequality constraints (optional): per-window discharge-throughput caps,
    # one row per budget. Total discharged energy in the window must stay under
    # the budget: sum_t discharge[t] * dt <= budget_mwh.
    A_ub, b_ub = None, None
    if throughput_budgets:
        ineq_rows, ineq_cols, ineq_vals, ineq_rhs = [], [], [], []
        for budget_index, (start, end, budget_mwh) in enumerate(throughput_budgets):
            for t in range(start, min(end, n_steps)):
                ineq_rows.append(budget_index)
                ineq_cols.append(n_steps + t)          # the discharge[t] variable
                ineq_vals.append(dt_per_step[t])       # convert power to energy
            ineq_rhs.append(budget_mwh)
        A_ub = sparse.coo_matrix(
            (ineq_vals, (ineq_rows, ineq_cols)),
            shape=(len(throughput_budgets), n_vars),
        ).tocsr()
        b_ub = np.array(ineq_rhs)

    # Variable bounds: charge/discharge between 0 and rated power; SOC between
    # empty and capacity. Optionally pin the final SOC to terminal_soc.
    charge_bounds = [(0.0, power_mw)] * n_steps
    discharge_bounds = [(0.0, power_mw)] * n_steps
    soc_bounds = [(0.0, capacity_mwh)] * n_steps
    if terminal_soc is not None:
        soc_bounds[-1] = (terminal_soc, terminal_soc)

    solution = linprog(
        objective_coeffs,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=charge_bounds + discharge_bounds + soc_bounds,
        method="highs",
    )

    if not solution.success:
        # Infeasible / failed solve: return a do-nothing plan and the reason.
        return {
            "charge": np.zeros(n_steps),
            "discharge": np.zeros(n_steps),
            "soc": np.full(n_steps + 1, soc0),
            "revenue": 0.0,
            "status": solution.message,
        }

    # Unpack the flat solution vector back into named arrays.
    charge = solution.x[:n_steps]
    discharge = solution.x[n_steps:2 * n_steps]
    # Prepend soc0 so the SOC series has one entry per interval boundary.
    soc = np.concatenate([[soc0], solution.x[2 * n_steps:]])
    revenue = float((prices * (discharge - charge) * dt_per_step).sum())
    return {
        "charge": charge,
        "discharge": discharge,
        "soc": soc,
        "revenue": revenue,
        "status": "optimal",
    }


def solve_mean_cvar_lp(
    scenario_prices: np.ndarray,
    weights: np.ndarray,
    *,
    dt_hours: float | np.ndarray,
    lam: float = 0.5,
    alpha: float = 0.5,
    power_mw: float = 100.0,
    capacity_mwh: float = 200.0,
    efficiency: float = 0.85,
    soc0: float = 0.0,
    terminal_soc: float | None = None,
    throughput_budgets: Sequence[tuple[int, int, float]] | None = None,
) -> dict:
    """Mean-CVaR battery dispatch across price scenarios (one here-and-now plan).

    Solves a single joint LP for *one* charge/discharge plan that trades off
    expected revenue against tail risk over ``n_scenarios`` price scenarios (the
    quantile paths of the forecast fan). It maximises::

        (1 - lam) * E[revenue]  +  lam * CVaR_alpha(revenue)

    where ``revenue_s`` is the plan's revenue under scenario ``s`` and
    ``CVaR_alpha`` (Conditional Value-at-Risk, a.k.a. expected shortfall) is the
    *mean revenue over the worst ``alpha`` fraction of scenarios* — a risk
    measure that looks at how bad the tail is, not just where it starts. ``lam``
    is the risk-aversion weight (0 = pure expected value, 1 = pure worst-tail)
    and ``alpha`` the tail level (small = more tail-averse; ``alpha = 1``
    recovers the plain mean).

    CVaR is turned into linear constraints by the **Rockafellar-Uryasev**
    identity: introduce a scalar ``var`` (the Value-at-Risk threshold) and one
    non-negative shortfall variable ``shortfall_s`` per scenario measuring how
    far that scenario's revenue falls below ``var``. Then CVaR equals
    ``var - (1/alpha) * sum_s weight_s * shortfall_s`` under the constraint
    ``var - shortfall_s - revenue_s <= 0``. Unlike blending per-scenario
    actions after the fact, this optimises one plan *jointly* against the whole
    scenario distribution.

    Args:
        scenario_prices: ``(n_scenarios, n_steps)`` price paths ($/MWh), one row
            per scenario.
        weights: ``(n_scenarios,)`` scenario probabilities (need not sum to 1;
            they are normalised).
        dt_hours: Interval length(s) in hours (scalar or length-``n_steps`` array).
        lam: Risk-aversion weight in ``[0, 1]``.
        alpha: CVaR tail level in ``(0, 1]``.
        power_mw: Rated charge/discharge power (MW), as in :func:`solve_lp`.
        capacity_mwh: Usable energy capacity (MWh).
        efficiency: Round-trip efficiency.
        soc0: State of charge at the start of the horizon (MWh).
        terminal_soc: If given, pins the final SOC.
        throughput_budgets: Per-window discharge-throughput caps.

    Returns:
        Dict with ``charge`` (MW per step), ``discharge`` (MW per step),
        ``soc`` (MWh, length ``n_steps + 1``), ``scenario_revenue`` ($ per
        scenario), ``var`` (the fitted Value-at-Risk threshold), and ``status``.
    """
    price_paths = np.atleast_2d(np.asarray(scenario_prices, dtype=float))
    n_scenarios, n_steps = price_paths.shape
    scenario_probs = np.asarray(weights, dtype=float)
    scenario_probs = scenario_probs / scenario_probs.sum()   # normalise to sum 1
    one_way_efficiency = np.sqrt(efficiency)
    dt_per_step = np.broadcast_to(np.asarray(dt_hours, dtype=float), (n_steps,))
    # Probability-weighted mean price per interval — the "expected" price path
    # that drives the E[revenue] part of the objective.
    mean_price = (scenario_probs[:, None] * price_paths).sum(axis=0)

    # Decision-variable layout (one flat vector):
    #   x = [ charge(n_steps) , discharge(n_steps) , soc(n_steps) ,
    #         var(1) , shortfall(n_scenarios) ]
    # var is the VaR threshold; shortfall_s is scenario s's revenue shortfall
    # below var (the Rockafellar-Uryasev auxiliary variables).
    n_vars = 3 * n_steps + 1 + n_scenarios
    var_col = 3 * n_steps                 # index of the single ``var`` variable
    shortfall_col0 = 3 * n_steps + 1      # index of the first ``shortfall`` variable

    # Objective. We minimise the negative of
    #   (1 - lam) * E[revenue] + lam * (var - (1/alpha) * sum_s prob_s * shortfall_s).
    objective_coeffs = np.zeros(n_vars)
    # Charging costs the expected price; discharging earns it (both weighted by
    # 1 - lam, the expected-value share of the objective).
    objective_coeffs[:n_steps] = (1.0 - lam) * mean_price * dt_per_step
    objective_coeffs[n_steps:2 * n_steps] = -(1.0 - lam) * mean_price * dt_per_step
    # The CVaR term contributes + lam * var and + (lam/alpha) * sum(prob * shortfall).
    objective_coeffs[var_col] = -lam
    objective_coeffs[shortfall_col0:] = (lam / alpha) * scenario_probs

    # Equality constraints: SOC dynamics, identical to solve_lp (they only touch
    # the charge/discharge/soc variables; var and shortfall do not appear here).
    eq_rows, eq_cols, eq_vals, eq_rhs = [], [], [], []
    for t in range(n_steps):
        eq_rows += [t, t, t]
        eq_cols += [t, n_steps + t, 2 * n_steps + t]
        eq_vals += [
            -one_way_efficiency * dt_per_step[t],
            dt_per_step[t] / one_way_efficiency,
            1.0,
        ]
        if t > 0:
            eq_rows.append(t)
            eq_cols.append(2 * n_steps + t - 1)
            eq_vals.append(-1.0)
            eq_rhs.append(0.0)
        else:
            eq_rhs.append(soc0)
    A_eq = sparse.coo_matrix(
        (eq_vals, (eq_rows, eq_cols)), shape=(n_steps, n_vars)
    ).tocsr()
    b_eq = np.array(eq_rhs)

    # Inequality constraints, built as (row, col, value) triples:
    #   (a) one CVaR shortfall row per scenario:
    #         var - shortfall_s - revenue_s <= 0,  where
    #         revenue_s = sum_t price_s[t] * dt * (discharge[t] - charge[t]).
    #       Written with charge as +price and discharge as -price so the row is
    #       (+price*charge) + (-price*discharge) + var - shortfall_s <= 0.
    #   (b) then the optional per-window throughput budgets.
    ineq_rows, ineq_cols, ineq_vals, ineq_rhs = [], [], [], []
    for s in range(n_scenarios):
        for t in range(n_steps):
            ineq_rows += [s, s]
            ineq_cols += [t, n_steps + t]
            ineq_vals += [
                price_paths[s, t] * dt_per_step[t],     # +price * charge[t]
                -price_paths[s, t] * dt_per_step[t],    # -price * discharge[t]
            ]
        ineq_rows += [s, s]
        ineq_cols += [var_col, shortfall_col0 + s]
        ineq_vals += [1.0, -1.0]                        # +var, -shortfall_s
        ineq_rhs.append(0.0)
    if throughput_budgets:
        first_budget_row = n_scenarios
        for budget_index, (start, end, budget_mwh) in enumerate(throughput_budgets):
            for t in range(start, min(end, n_steps)):
                ineq_rows.append(first_budget_row + budget_index)
                ineq_cols.append(n_steps + t)           # discharge[t]
                ineq_vals.append(dt_per_step[t])
            ineq_rhs.append(budget_mwh)
    n_ineq_rows = n_scenarios + (
        len(throughput_budgets) if throughput_budgets else 0
    )
    A_ub = sparse.coo_matrix(
        (ineq_vals, (ineq_rows, ineq_cols)), shape=(n_ineq_rows, n_vars)
    ).tocsr()

    # Variable bounds: charge/discharge in [0, power], soc in [0, capacity], var
    # free (it is a threshold, can be any sign), shortfall non-negative.
    bounds = (
        [(0.0, power_mw)] * n_steps            # charge
        + [(0.0, power_mw)] * n_steps          # discharge
        + [(0.0, capacity_mwh)] * n_steps      # soc
        + [(None, None)]                       # var (free)
        + [(0.0, None)] * n_scenarios          # shortfall (>= 0)
    )
    if terminal_soc is not None:
        bounds[3 * n_steps - 1] = (terminal_soc, terminal_soc)   # last soc entry

    solution = linprog(
        objective_coeffs, A_ub=A_ub, b_ub=np.array(ineq_rhs),
        A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs",
    )
    if not solution.success:
        return {
            "charge": np.zeros(n_steps), "discharge": np.zeros(n_steps),
            "soc": np.full(n_steps + 1, soc0),
            "scenario_revenue": np.zeros(n_scenarios),
            "var": 0.0, "status": solution.message,
        }

    charge = solution.x[:n_steps]
    discharge = solution.x[n_steps:2 * n_steps]
    soc = np.concatenate([[soc0], solution.x[2 * n_steps:3 * n_steps]])
    # Realised revenue of the single chosen plan under each scenario's prices.
    scenario_revenue = (price_paths * (discharge - charge) * dt_per_step).sum(axis=1)
    return {
        "charge": charge, "discharge": discharge, "soc": soc,
        "scenario_revenue": scenario_revenue,
        "var": float(solution.x[var_col]), "status": "optimal",
    }


def clamp_action(
    charge_mw: float,
    discharge_mw: float,
    soc_mwh: float,
    *,
    dt_hours: float,
    power_mw: float = 100.0,
    capacity_mwh: float = 200.0,
    efficiency: float = 0.85,
    discharge_budget_mwh: float = float("inf"),
) -> tuple[float, float, float]:
    """Clamp a planned action so execution is physically feasible.

    An LP plan is only a plan; when we execute it against reality the true SOC
    and the remaining cycle budget may not support it. This trims the action to
    what is actually possible: discharge is limited by the energy in store and
    by the remaining daily throughput budget; charge is then limited by the
    headroom left after the (already clamped) discharge. Revenue must only ever
    be computed from these clamped values.

    Args:
        charge_mw: Planned charging power (MW).
        discharge_mw: Planned discharging power (MW).
        soc_mwh: State of charge before the action (MWh).
        dt_hours: Interval length in hours.
        power_mw: Power limit (MW).
        capacity_mwh: Energy capacity (MWh).
        efficiency: Round-trip efficiency.
        discharge_budget_mwh: Remaining discharge throughput allowed today
            (MWh of discharged energy).

    Returns:
        Tuple ``(charge_mw, discharge_mw, new_soc_mwh)`` — all feasible.
    """
    one_way_efficiency = np.sqrt(efficiency)

    # Discharge first: clamp to [0, power], then to the energy the store can
    # actually deliver, then to the remaining throughput budget. Delivering
    # ``discharge`` for ``dt`` hours draws ``discharge / efficiency * dt`` from
    # the store (efficiency loss on the way out), so the store can support at
    # most ``soc * efficiency / dt`` of delivered power.
    discharge = min(max(discharge_mw, 0.0), power_mw)
    discharge = min(discharge, soc_mwh * one_way_efficiency / dt_hours)
    discharge = min(discharge, discharge_budget_mwh / dt_hours)

    soc_after_discharge = soc_mwh - discharge / one_way_efficiency * dt_hours

    # Charge next: clamp to [0, power], then to the headroom remaining. Charging
    # ``charge`` for ``dt`` hours adds ``charge * efficiency * dt`` to the store,
    # so the headroom of ``capacity - soc_after_discharge`` allows at most
    # ``headroom / (efficiency * dt)`` of charging power.
    charge = min(max(charge_mw, 0.0), power_mw)
    headroom_mwh = capacity_mwh - soc_after_discharge
    charge = min(charge, headroom_mwh / (one_way_efficiency * dt_hours))

    new_soc = soc_after_discharge + charge * one_way_efficiency * dt_hours
    new_soc = min(max(new_soc, 0.0), capacity_mwh)   # guard float round-off
    return charge, discharge, new_soc


def resolution_dt_hours(resolution: str) -> float:
    """Interval length in hours for a resolution string.

    Args:
        resolution: Either ``"5min"`` or ``"30min"``.

    Returns:
        Interval length in hours (1/12 for 5-min, 0.5 for 30-min).
    """
    return 5.0 / 60.0 if resolution == "5min" else 0.5
