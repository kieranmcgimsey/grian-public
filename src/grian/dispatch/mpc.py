"""Receding-horizon MPC executor.

Replaces the open-loop one-LP-per-day execution with a rolling loop:
every ``resolve_every`` intervals the LP is re-solved from the true
state of charge, and every ``reforecast_every`` intervals the forecast
is regenerated from all data observed so far. This converts the
forecaster's short-lead skill — much higher than its midnight skill —
into dispatch decisions.

Cycle limits are enforced per calendar day both inside the LP (the
first partial day gets the *remaining* budget, later days a full one)
and again at execution via the shared feasibility clamp.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from grian import models as models_mod
from grian.dispatch import battery_lp as lp
from grian.dispatch import ledger as ledger_mod
from grian.dispatch.probabilistic import (
    combine_scenario_actions,
    quantile_gate_prices,
    quantile_weights,
)
from grian.evaluation import trials as trials_mod

# Dispatch modes that consume a quantile fan rather than a single price path.
# _SCENARIO_MODES maps each mode to how per-scenario actions are fused:
# None = legacy robust min, else the combine_scenario_actions mode.
_SCENARIO_MODES = {"scenario": None, "scenario_ev": "ev", "scenario_cvar": "cvar"}
# "mean_cvar" solves the joint Rockafellar-Uryasev LP (one plan across scenarios)
# rather than fusing per-scenario actions; it also needs the fan.
_FAN_MODES = {"qgate", "mean_cvar"} | set(_SCENARIO_MODES)

logger = logging.getLogger(__name__)


def _telescope(
    plan_prices: np.ndarray,
    dt_hours: float,
    fine_n: int = 12,
    coarse_block: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compress a fine-grained price path into a telescoping horizon.

    The first ``fine_n`` steps keep native resolution (these are the
    steps that may actually be executed before the next re-solve);
    the remainder is averaged into blocks of ``coarse_block`` steps.
    This cuts the LP from ~288 to ~58 variables per stage (a ~20×
    solve-time reduction) at no measurable cost, because only the
    fine steps are executed and the plan is re-solved every block.

    Args:
        plan_prices: Native-resolution price path.
        dt_hours: Native step length in hours.
        fine_n: Steps kept at native resolution.
        coarse_block: Native steps per coarse block.

    Returns:
        Tuple (prices, dts) — per-step prices and durations.
    """
    fine = plan_prices[:fine_n]
    rest = plan_prices[fine_n:]
    n_blocks = len(rest) // coarse_block
    parts_p = [fine]
    parts_dt = [np.full(len(fine), dt_hours)]
    if n_blocks:
        coarse = rest[: n_blocks * coarse_block].reshape(
            n_blocks, coarse_block
        ).mean(axis=1)
        parts_p.append(coarse)
        parts_dt.append(np.full(n_blocks, dt_hours * coarse_block))
    tail = rest[n_blocks * coarse_block:]
    if len(tail):
        parts_p.append([tail.mean()])
        parts_dt.append([len(tail) * dt_hours])
    return np.concatenate(parts_p), np.concatenate(parts_dt)


def _day_budgets(
    offset_hours: float,
    step_dts: np.ndarray,
    remaining_today_mwh: float,
    full_budget_mwh: float,
) -> list[tuple[int, int, float]]:
    """Per-calendar-day discharge budgets over variable-length steps.

    Each step is assigned to the calendar day containing its start.
    The first day gets the remaining budget, later days a full one.

    Args:
        offset_hours: Hours already elapsed in the current day.
        step_dts: Per-step durations in hours.
        remaining_today_mwh: Discharge budget left for today.
        full_budget_mwh: Full daily budget for subsequent days.

    Returns:
        List of (start, end, budget) tuples covering all steps.
    """
    starts = offset_hours + np.concatenate([[0.0], np.cumsum(step_dts)[:-1]])
    day_idx = np.floor(starts / 24.0 + 1e-9).astype(int)
    budgets = []
    for d in range(day_idx.max() + 1):
        steps = np.where(day_idx == d)[0]
        if len(steps) == 0:
            continue
        budget = max(0.0, remaining_today_mwh) if d == 0 else full_budget_mwh
        budgets.append((int(steps[0]), int(steps[-1]) + 1, budget))
    return budgets


def _forecast_from_fan(
    fan: dict, dispatch_mode: str, q_low: float, q_high: float
) -> np.ndarray:
    """Reduce a quantile fan to the single price path that drives the LP.

    ``qgate`` builds the direction-pessimistic path. Every other mode uses the
    quantile-integrated **mean** ``E[price]`` — NOT the median. The dispatch
    objective is linear in price (revenue = price x action), so its expectation
    needs the mean; and electricity prices are heavily right-skewed, so the
    median badly under-predicts spikes and the controller sells into mediocre
    intervals (buy-low / sell-mediocre). Using the mean matches the open-loop
    point forecast (``_lgbm_qmean_predict``). For scenario modes the fan itself
    still drives the per-scenario LPs; this path is only the logged/observe-
    present reference.
    """
    if dispatch_mode == "qgate":
        return quantile_gate_prices(fan, q_low, q_high)
    qs = sorted(fan)
    weights = quantile_weights(qs)
    stacked = np.vstack([np.asarray(fan[q], dtype=float) for q in qs])
    return (weights[:, None] * stacked).sum(axis=0)


def simulate_region_mpc(
    data: pd.DataFrame,
    cfg: dict,
    dispatch_fn: Any = None,
    fan_cache: dict | None = None,
    record_fans: bool = False,
    record_plans: bool = False,
    fan_checkpoint: tuple[Any, int] | None = None,
) -> dict[str, Any]:
    """Run the receding-horizon MPC simulation for one region.

    Walks the test window interval-block by interval-block:
    1. Refit the model on all observed history (weekly, at midnight).
    2. Every ``reforecast_every`` intervals, regenerate the forecast
       from data up to now (predict-from-now).
    3. Every ``resolve_every`` intervals, re-solve the LP from the true
       SOC and remaining cycle budget.
    4. Execute the block with feasibility clamping and log it.

    Args:
        data: Full dataset (train + test), DatetimeIndex, target column.
        cfg: Trial config; reads ``mpc.resolve_every`` (default 6) and
            ``mpc.reforecast_every`` (default 12) in addition to the
            standard keys.
        dispatch_fn: Ignored — present for open_loop.run_trial compatibility.
        fan_cache: Optional ``{origin_timestamp: {quantile: path}}`` of
            pre-computed quantile fans. When given, the model is never fit
            or queried — forecasts are read from the cache — so many dispatch
            variants can be replayed over one frozen forecast in seconds
            (the test-bed fast path). Requires a fan produced by a prior
            ``record_fans`` run over the same window/reforecast cadence.
        record_fans: If True, collect each reforecast's full quantile fan
            into the returned ``"fans"`` dict for later replay.
        record_plans: If True, record each solve's full net-action plan.
        fan_checkpoint: Optional ``(save_fn, every)`` — with ``record_fans`` on,
            calls ``save_fn(recorded_fans)`` every ``every`` reforecasts so a long
            build is crash-safe (partial fans survive) and its progress is logged.

    Returns:
        Dict with "ledger", "forecasts", "model_state", and (when
        ``record_fans``) "fans" — otherwise the same shape as
        open_loop.simulate_region.
    """
    target_col = cfg["target_col"]
    horizon = cfg["horizon"]
    resolution = cfg.get("resolution", "5min")
    ppd = models_mod._periods_per_day(resolution)
    dt_hours = lp.resolution_dt_hours(resolution)
    refit_days = cfg.get("refit_days", 7)
    seed = cfg.get("seed", 42)

    mpc_cfg = cfg.get("mpc", {})
    resolve_every = int(mpc_cfg.get("resolve_every", 6))
    reforecast_every = int(mpc_cfg.get("reforecast_every", 12))
    persistence_tau = float(mpc_cfg.get("persistence_tau", 0))
    persistence_gate = float(mpc_cfg.get("persistence_gate", 0))
    # observe_present pins step 0 of the plan to the *actual* current price
    # (we know it at decision time). But applied at every re-solve it makes the
    # LP trade the high-frequency forecast residual (actual vs smoothed
    # forecast) — churning cycles on mean-reversion noise and collapsing capture
    # (Entry 034). observe_gate keeps that pin ONLY when the current price is a
    # genuine spike (>= gate $/MWh) — the events open-loop under-forecasts and
    # misses — and otherwise trusts the forecast, so MPC keeps the clean daily
    # arbitrage AND grabs unforecast spikes.
    observe_present = mpc_cfg.get("observe_present", True)
    observe_gate = float(mpc_cfg.get("observe_gate", 0.0))
    # Probabilistic dispatch: "qgate" feeds the LP direction-pessimistic quantiles.
    dispatch_mode = mpc_cfg.get("dispatch_mode", "point")
    q_low = float(mpc_cfg.get("q_low", 0.1))
    q_high = float(mpc_cfg.get("q_high", 0.9))
    cvar_lambda = float(mpc_cfg.get("cvar_lambda", 0.5))
    cvar_alpha = float(mpc_cfg.get("cvar_alpha", 0.5))
    # Horizon telescoping is a speed hack for the 288-step 5-min LP. At 30-min
    # the day-ahead horizon is only ~48 steps — cheap to solve in full — and
    # coarsening it into 3-hour blocks *averages away the daily price peak*, so
    # the controller can't see the peak coming and discharges early into lesser
    # bumps (buy-low / sell-mediocre). Only telescope when the horizon is large.
    tele_fine_n = int(mpc_cfg.get("fine_n", 12 if horizon > 96 else horizon))
    # Executed steps must lie inside the telescoped plan's fine window.
    assert resolve_every <= tele_fine_n, "resolve_every must fit the fine horizon"

    dispatch_cfg = cfg.get("dispatch", {})
    power_mw = dispatch_cfg.get("power_mw", 100.0)
    efficiency = dispatch_cfg.get("efficiency", 0.85)
    capacity_mwh = power_mw * dispatch_cfg.get("duration_hours", 2.0)
    daily_budget = dispatch_cfg.get("max_cycles", 2) * capacity_mwh

    model_spec = models_mod.get_model(cfg["model"])
    np.random.seed(seed)

    forward_fn, inverse_fn = trials_mod._get_transform_pair(
        cfg.get("transform", "identity")
    )
    # Pointwise transform — safe to apply once over the full frame.
    tdata = data.copy()
    tdata[target_col] = forward_fn(tdata[target_col].values)

    test_start = pd.Timestamp(cfg["test_start"])
    test_end = pd.Timestamp(cfg["test_end"]) + pd.Timedelta(days=1)
    pos0 = int(data.index.searchsorted(test_start))
    pos_end = int(data.index.searchsorted(test_end))

    trade_ledger = ledger_mod.empty_ledger()
    all_forecasts = []
    recorded_fans: dict = {}
    recorded_plans: list = []
    model_state = None
    days_since_refit = refit_days  # force refit at start
    soc = 0.0
    throughput_today = 0.0
    current_day = None
    forecast_dollars = None
    fan_dollars = None
    forecast_origin = None
    lp_failures = 0

    pos = pos0
    while pos < pos_end:
        ts = data.index[pos]
        day = ts.normalize()

        if day != current_day:
            if current_day is not None:
                days_since_refit += 1
            current_day = day
            throughput_today = 0.0
            # Replaying a cached fan: never fit or query the model.
            if fan_cache is None and (
                model_state is None or days_since_refit >= refit_days
            ):
                import time as _time
                _t_refit = _time.time()
                model_state = model_spec["fit"](
                    tdata.iloc[:pos], target_col, cfg
                )
                days_since_refit = 0
                logger.info("refit at %s (%d rows, %.0fs)", ts, pos,
                            _time.time() - _t_refit)

        # Reforecast from all data observed so far.
        if (
            forecast_dollars is None
            or pos - forecast_origin >= reforecast_every
        ):
            if fan_cache is not None:
                # Test-bed replay: pull the frozen fan for this origin.
                fan_dollars = fan_cache[ts]
                forecast_dollars = _forecast_from_fan(
                    fan_dollars, dispatch_mode, q_low, q_high)
            elif dispatch_mode in _FAN_MODES and "predict_fan" in model_spec:
                # Probabilistic: predict_fan returns dollar-space quantile paths.
                fan_dollars = model_spec["predict_fan"](
                    model_state, tdata.iloc[:pos], horizon)
                forecast_dollars = _forecast_from_fan(
                    fan_dollars, dispatch_mode, q_low, q_high)
                if record_fans:
                    recorded_fans[ts] = {
                        float(q): np.asarray(v, dtype=float)
                        for q, v in fan_dollars.items()
                    }
                    # Checkpoint the fan to disk every N reforecasts so a long
                    # build is never an all-or-nothing in-memory gamble, and the
                    # count doubles as a progress signal.
                    if fan_checkpoint is not None:
                        save_fn, every = fan_checkpoint
                        if len(recorded_fans) % max(1, every) == 0:
                            save_fn(recorded_fans)
                            logger.info("fan checkpoint: %d origins recorded",
                                        len(recorded_fans))
            else:
                fan_dollars = None
                raw = model_spec["predict"](model_state, tdata.iloc[:pos], horizon)
                forecast_dollars = inverse_fn(np.asarray(raw.values, dtype=float))
            forecast_origin = pos
            all_forecasts.append({
                "origin": ts,
                "forecast": forecast_dollars[: reforecast_every].tolist(),
                "actual": data[target_col]
                .iloc[pos: pos + reforecast_every]
                .tolist(),
            })

        # Plan from the current state over the remaining forecast,
        # telescoped: native resolution for the first hour, 30-min
        # blocks beyond. Only fine steps are ever executed.
        shift = pos - forecast_origin
        plan_prices = forecast_dollars[shift:]

        # Persistence blend: pull the first steps of the plan
        # toward the last observed price with exponentially decaying
        # weight. On spike days the model's mean-reverting forecast
        # lags the event by up to reforecast_every intervals; the last
        # trade price is the best short-lead estimate we own. Uses only
        # already-observed data — no leakage.
        # Gate (spike latch): an ungated blend loses ~10 capture points
        # on validation — 5-min prices mean-revert hard, so trusting the
        # last price on ordinary intervals chases noise. Only override
        # the model when the last price is genuinely extreme.
        if persistence_tau > 0 and pos > pos0:
            last_price = float(data[target_col].iloc[pos - 1])
            if persistence_gate > 0 and last_price < persistence_gate:
                last_price = None
        else:
            last_price = None
        if last_price is not None:
            k = np.arange(min(len(plan_prices), int(4 * persistence_tau)))
            w = np.exp(-k / persistence_tau)
            plan_prices = plan_prices.copy()
            plan_prices[: len(k)] = (
                w * last_price + (1 - w) * plan_prices[: len(k)]
            )

        # Observe the present (gated): at decision time the current price is
        # known. Pin step 0 to it only when it is a genuine spike (>= gate);
        # otherwise trust the forecast, so frequent re-solves don't churn on the
        # forecast residual (Entry 034). gate <= 0 reproduces always-observe.
        actual_now = float(data[target_col].iloc[pos])
        observe_now = observe_present and (
            observe_gate <= 0.0 or actual_now >= observe_gate)
        if observe_now and len(plan_prices) > 0:
            plan_prices = plan_prices.copy()
            plan_prices[0] = actual_now

        tele_prices, tele_dts = _telescope(plan_prices, dt_hours, fine_n=tele_fine_n)
        offset_hours = (ts - day) / pd.Timedelta(hours=1)
        budgets = _day_budgets(
            offset_hours, tele_dts,
            daily_budget - throughput_today, daily_budget,
        )
        lp_kw = dict(power_mw=power_mw, capacity_mwh=capacity_mwh,
                     efficiency=efficiency, soc0=soc, throughput_budgets=budgets)
        if dispatch_mode == "mean_cvar" and fan_dollars is not None:
            # Joint Rockafellar-Uryasev LP: ONE plan optimised across all
            # quantile paths, trading expected revenue against tail risk
            # (cvar_lambda) at tail level cvar_alpha — the principled version
            # of the decision-space blend.
            qs, paths = [], []
            for q, path in fan_dollars.items():
                sp = np.asarray(path[shift:], dtype=float).copy()
                if observe_now and len(sp) > 0:
                    sp[0] = actual_now
                stp, stdt = _telescope(sp, dt_hours, fine_n=tele_fine_n)
                qs.append(q)
                paths.append(stp)
            order = np.argsort(qs)
            scen = np.asarray(paths)[order]
            w = quantile_weights(sorted(qs))
            result = lp.solve_mean_cvar_lp(
                scen, w, dt_hours=stdt, lam=cvar_lambda, alpha=cvar_alpha,
                **lp_kw)
        elif dispatch_mode in _SCENARIO_MODES and fan_dollars is not None:
            # Solve each quantile path on its own (each internally ranked, so the
            # cross-interval arbitrage ordering the LP relies on is preserved —
            # the property the price-space gate destroyed), then fuse the actions
            # in *decision* space: robust min, expected value, or a CVaR blend.
            qs, charges, discharges = [], [], []
            for q, path in fan_dollars.items():
                sp = np.asarray(path[shift:], dtype=float).copy()
                if observe_now and len(sp) > 0:
                    sp[0] = actual_now
                stp, stdt = _telescope(sp, dt_hours, fine_n=tele_fine_n)
                r = lp.solve_lp(stp, dt_hours=stdt, **lp_kw)
                if r["status"] == "optimal":
                    qs.append(q)
                    charges.append(r["charge"])
                    discharges.append(r["discharge"])
            if charges:
                order = np.argsort(qs)          # align rows with sorted-q weights
                charges = np.asarray(charges)[order]
                discharges = np.asarray(discharges)[order]
                fuse = _SCENARIO_MODES[dispatch_mode]
                if fuse is None:                # legacy robust: only where all agree
                    ch, dis = charges.min(axis=0), discharges.min(axis=0)
                else:
                    w = quantile_weights(sorted(qs))
                    ch, dis = combine_scenario_actions(
                        charges, discharges, w, fuse, cvar_lambda)
                result = {"charge": ch, "discharge": dis, "status": "optimal"}
            else:
                result = {"charge": np.zeros(len(tele_dts)),
                          "discharge": np.zeros(len(tele_dts)), "status": "infeasible"}
        else:
            result = lp.solve_lp(tele_prices, dt_hours=tele_dts, **lp_kw)
        planned_ok = result["status"] == "optimal"
        if not planned_ok:
            lp_failures += 1

        if record_plans and planned_ok:
            # The full multi-step plan the LP just produced (only the first
            # ``resolve_every`` steps get executed). Net action per horizon step,
            # keyed by absolute position so plans from different solves can be
            # aligned on the target interval to measure re-plan churn.
            net = np.asarray(result["charge"], dtype=float) - np.asarray(
                result["discharge"], dtype=float)
            recorded_plans.append({"pos": pos, "ts": data.index[pos],
                                   "soc0": soc, "net": net})

        # Execute this block against actual prices, clamped.
        block_end = min(pos + resolve_every, pos_end)
        for i in range(pos, block_end):
            k = i - pos
            charge = float(result["charge"][k]) if planned_ok else 0.0
            discharge = float(result["discharge"][k]) if planned_ok else 0.0
            charge, discharge, soc = lp.clamp_action(
                charge, discharge, soc,
                dt_hours=dt_hours,
                power_mw=power_mw,
                capacity_mwh=capacity_mwh,
                efficiency=efficiency,
                discharge_budget_mwh=daily_budget - throughput_today,
            )
            throughput_today += discharge * dt_hours
            ledger_mod.append_record(
                trade_ledger,
                timestamp=data.index[i],
                actual_price=float(data[target_col].iloc[i]),
                forecast_price=float(forecast_dollars[i - forecast_origin]),
                charge_mw=charge,
                discharge_mw=discharge,
                soc_mwh=soc,
                interval_minutes=dt_hours * 60.0,
            )

        pos = block_end
        if (pos - pos0) % (5 * ppd) < resolve_every:
            import time as _time
            now = _time.time()
            if not hasattr(simulate_region_mpc, "_t0"):
                simulate_region_mpc._t0 = now
            logger.info("MPC day %d/%d (%.0fs elapsed)",
                        (pos - pos0) // ppd, (pos_end - pos0) // ppd,
                        now - simulate_region_mpc._t0)

    if lp_failures:
        rate = lp_failures / max(1, (pos_end - pos0) // resolve_every)
        logger.warning("LP failures: %d (%.2f%%)", lp_failures, 100 * rate)

    result = {
        "ledger": trade_ledger,
        "forecasts": all_forecasts,
        "model_state": model_state,
    }
    if record_fans:
        result["fans"] = recorded_fans
    if record_plans:
        result["plans"] = recorded_plans
    return result
