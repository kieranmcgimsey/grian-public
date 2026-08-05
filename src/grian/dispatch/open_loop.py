"""Open-loop walk-forward executor and trial runner.

Runs a day-by-day trading simulation: for each trading day, optionally
refits the model on all available history, produces a forecast over the
horizon, runs dispatch to decide charge/discharge actions, then executes
against actual prices and logs every interval to the ledger.

The runner is domain-agnostic — it reads the target column, transform,
embargo, and dispatch function from the trial config. It never
references "price", "asinh", or any specific model by name.

Multi-region: the top-level `run_trial` function loops over all regions
in the config, running the simulation independently for each and saving
artifacts per region.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grian import models as models_mod
from grian.dispatch import ledger as ledger_mod
from grian.evaluation import trials as trials_mod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default dispatch function — wraps the existing battery LP
# ---------------------------------------------------------------------------

def battery_dispatch(
    forecast: pd.Series,
    actual_prices: pd.Series,
    cfg: dict,
    state: dict | None = None,
) -> list[dict]:
    """Dispatch a battery against forecast prices, score against actuals.

    This is the default (open-loop) dispatch strategy. It solves the
    arbitrage LP over the forecast horizon starting from the carried
    state of charge, then executes the plan with per-interval
    feasibility clamping — revenue can only come from clamped,
    physically possible actions.

    Args:
        forecast: Forecasted prices for the horizon.
        actual_prices: Actual prices for the same horizon.
        cfg: Trial config (reads dispatch.* and resolution).
        state: Mutable executor state carried across calls; uses and
            updates ``state["soc"]``. None starts from an empty battery.

    Returns:
        List of interval-level dicts with charge, discharge, soc, and
        revenue fields — one per interval in the horizon.
    """
    from grian.dispatch import battery_lp as lp

    dispatch_cfg = cfg.get("dispatch", {})
    dt_hours = lp.resolution_dt_hours(cfg.get("resolution", "5min"))
    power_mw = dispatch_cfg.get("power_mw", 100.0)
    efficiency = dispatch_cfg.get("efficiency", 0.85)
    capacity_mwh = power_mw * dispatch_cfg.get("duration_hours", 2.0)
    cycle_budget = dispatch_cfg.get("max_cycles", 2) * capacity_mwh

    soc = state["soc"] if state is not None else 0.0

    result = lp.solve_lp(
        forecast.values,
        dt_hours=dt_hours,
        power_mw=power_mw,
        capacity_mwh=capacity_mwh,
        efficiency=efficiency,
        soc0=soc,
        throughput_budgets=[(0, len(forecast), cycle_budget)],
    )
    planned_ok = result["status"] == "optimal"

    n = min(len(forecast), len(actual_prices))
    records = []
    budget_left = cycle_budget

    for i in range(n):
        charge = float(result["charge"][i]) if planned_ok else 0.0
        discharge = float(result["discharge"][i]) if planned_ok else 0.0

        charge, discharge, soc = lp.clamp_action(
            charge, discharge, soc,
            dt_hours=dt_hours,
            power_mw=power_mw,
            capacity_mwh=capacity_mwh,
            efficiency=efficiency,
            discharge_budget_mwh=budget_left,
        )
        budget_left -= discharge * dt_hours

        records.append({
            "charge_mw": charge,
            "discharge_mw": discharge,
            "soc_mwh": soc,
        })

    if state is not None:
        state["soc"] = soc

    return records


# ---------------------------------------------------------------------------
# Single-day simulation step
# ---------------------------------------------------------------------------

def _simulate_day(
    model_spec: dict,
    model_state: dict,
    day_data: pd.DataFrame,
    forecast_data: pd.DataFrame,
    cfg: dict,
    dispatch_fn: Callable,
    dispatch_state: dict | None = None,
) -> list[dict]:
    """Simulate one trading day: forecast, dispatch, execute, log.

    Args:
        model_spec: Model spec dict (from models.REGISTRY).
        model_state: Current fitted model state.
        day_data: Full data available at forecast time (for features).
        forecast_data: Actual data for the forecast horizon (for scoring).
        cfg: Trial config dict.
        dispatch_fn: Dispatch function to use.
        dispatch_state: Mutable state (e.g. SOC) carried across days;
            passed to dispatch functions that accept a fourth argument.

    Returns:
        List of ledger records for this day's intervals.
    """
    target_col = cfg["target_col"]
    horizon = cfg["horizon"]
    transform_name = cfg.get("transform", "identity")

    # Apply ablation: use transform or identity
    if cfg.get("ablations", {}).get("use_transform", True):
        _, inverse_fn = trials_mod._get_transform_pair(transform_name)
    else:
        _, inverse_fn = trials_mod._get_transform_pair("identity")

    # Produce forecast
    raw_forecast = model_spec["predict"](model_state, day_data, horizon)

    # Invert the target transform so forecasts are in original units
    forecast_values = inverse_fn(raw_forecast.values)
    forecast = pd.Series(
        forecast_values,
        index=forecast_data.index[:len(forecast_values)],
        name="forecast",
    )

    # Get actual prices for the horizon
    actuals = forecast_data[target_col].iloc[:len(forecast)]

    # Dispatch against the forecast, score against actuals. Dispatch
    # functions that accept a state argument get SOC continuity.
    import inspect

    if len(inspect.signature(dispatch_fn).parameters) >= 4:
        actions = dispatch_fn(forecast, actuals, cfg, dispatch_state)
    else:
        actions = dispatch_fn(forecast, actuals, cfg)

    # Build ledger records
    records = []
    resolution = cfg.get("resolution", "5min")
    interval_minutes = 5.0 if resolution == "5min" else 30.0

    for i in range(min(len(forecast), len(actuals), len(actions))):
        records.append({
            "timestamp": actuals.index[i],
            "actual_price": float(actuals.iloc[i]),
            "forecast_price": float(forecast.iloc[i]),
            "charge_mw": actions[i]["charge_mw"],
            "discharge_mw": actions[i]["discharge_mw"],
            "soc_mwh": actions[i]["soc_mwh"],
            "interval_minutes": interval_minutes,
        })

    return records


# ---------------------------------------------------------------------------
# Walk-forward simulation for a single region
# ---------------------------------------------------------------------------

def simulate_region(
    data: pd.DataFrame,
    cfg: dict,
    dispatch_fn: Callable = battery_dispatch,
) -> dict[str, Any]:
    """Run the full walk-forward simulation for one region.

    Walks through the test period day by day. On each day:
    1. Optionally refit the model on all history up to the embargo.
    2. Forecast the next `horizon` intervals.
    3. Dispatch and execute against actual prices.
    4. Log all intervals to the ledger.

    Args:
        data: Full dataset for this region (train + test period),
            with a DatetimeIndex and the target column.
        cfg: Trial config dict.
        dispatch_fn: Dispatch function (default: battery_dispatch).

    Returns:
        Dict with "ledger" (list of records), "forecasts" (list of
        forecast arrays), "model_state" (final fitted state).
    """
    target_col = cfg["target_col"]
    horizon = cfg["horizon"]
    resolution = cfg.get("resolution", "5min")
    ppd = models_mod._periods_per_day(resolution)
    refit_days = cfg.get("refit_days", 1)
    embargo = cfg.get("embargo", horizon)
    seed = cfg.get("seed", 42)
    transform_name = cfg.get("transform", "identity")

    # Resolve model
    model_spec = models_mod.get_model(cfg["model"])

    # Set seed for reproducibility
    np.random.seed(seed)

    # Apply ablation: use transform or identity
    if cfg.get("ablations", {}).get("use_transform", True):
        forward_fn, _ = trials_mod._get_transform_pair(transform_name)
    else:
        forward_fn, _ = trials_mod._get_transform_pair("identity")

    # Apply ablation: embargo or no embargo
    use_embargo = cfg.get("ablations", {}).get("use_embargo", True)
    effective_embargo = embargo if use_embargo else 0

    # Determine test day boundaries
    test_start = pd.Timestamp(cfg["test_start"])
    test_end = pd.Timestamp(cfg["test_end"])
    test_days = pd.date_range(test_start, test_end, freq="D")

    trade_ledger = ledger_mod.empty_ledger()
    all_forecasts = []
    model_state = None
    dispatch_state = {"soc": 0.0}  # SOC carried across the whole window
    days_since_refit = refit_days  # Force refit on first day

    for day_idx, day in enumerate(test_days):
        # Forecast window for this day
        day_start = day
        day_end = day + pd.Timedelta(days=1)

        # Find actual data for this day's horizon
        forecast_mask = (data.index >= day_start) & (data.index < day_end)
        forecast_data = data.loc[forecast_mask]

        if len(forecast_data) == 0:
            continue

        # Training cutoff: everything before the forecast window, minus embargo
        if effective_embargo > 0:
            train_end_idx = data.index.searchsorted(day_start) - effective_embargo
        else:
            train_end_idx = data.index.searchsorted(day_start)

        if train_end_idx <= 0:
            continue

        # Rolling training window: cap how far back we train. None/0 keeps the
        # expanding window (all history). A fixed lookback bounds the refit cost
        # and drops stale regime data — see docs on why ~18 months is sensible.
        lookback_days = cfg.get("train_lookback_days")
        if lookback_days:
            train_start_idx = max(0, train_end_idx - int(lookback_days) * ppd)
        else:
            train_start_idx = 0

        train_data = data.iloc[train_start_idx:train_end_idx].copy()

        # Apply ablation: leak future data
        if cfg.get("ablations", {}).get("leak_future", False):
            # Deliberately include some future data in training
            leak_end = min(train_end_idx + horizon, len(data))
            train_data = data.iloc[train_start_idx:leak_end].copy()

        # Apply target transform to training data
        train_data[target_col] = forward_fn(train_data[target_col].values)

        # Refit model if needed
        if model_state is None or days_since_refit >= refit_days:
            model_state = model_spec["fit"](train_data, target_col, cfg)
            days_since_refit = 0
            logger.debug("Refit model on day %s (%d training rows)",
                         day.date(), len(train_data))

        days_since_refit += 1

        # Simulate this day
        actual_horizon = min(horizon, len(forecast_data))
        day_records = _simulate_day(
            model_spec=model_spec,
            model_state=model_state,
            day_data=train_data,
            forecast_data=forecast_data.iloc[:actual_horizon],
            cfg=cfg,
            dispatch_fn=dispatch_fn,
            dispatch_state=dispatch_state,
        )

        # Append to ledger
        for rec in day_records:
            ledger_mod.append_record(
                trade_ledger,
                timestamp=rec["timestamp"],
                actual_price=rec["actual_price"],
                forecast_price=rec["forecast_price"],
                charge_mw=rec["charge_mw"],
                discharge_mw=rec["discharge_mw"],
                soc_mwh=rec["soc_mwh"],
                interval_minutes=rec["interval_minutes"],
            )

        # Store forecast for later analysis
        all_forecasts.append({
            "origin": day,
            "forecast": [rec["forecast_price"] for rec in day_records],
            "actual": [rec["actual_price"] for rec in day_records],
        })

        if day_idx % 30 == 0:
            logger.info("Day %d/%d (%s)", day_idx + 1, len(test_days), day.date())

    return {
        "ledger": trade_ledger,
        "forecasts": all_forecasts,
        "model_state": model_state,
    }


# ---------------------------------------------------------------------------
# Top-level trial runner — multi-region with full artifact saving
# ---------------------------------------------------------------------------

def run_trial(
    data_by_region: dict[str, pd.DataFrame],
    cfg: dict,
    dispatch_fn: Callable = battery_dispatch,
    base: str | Path = "outputs/trials",
    simulate_fn: Callable | None = None,
) -> dict[str, dict]:
    """Run a complete trial across all regions and save all artifacts.

    This is the main entry point. It:
    1. Saves the trial config to disk.
    2. Loops over each region in cfg["regions"].
    3. Runs the walk-forward simulation.
    4. Saves forecasts, ledger, metrics, and model state per region.

    Args:
        data_by_region: Dict mapping region name to its full DataFrame.
        cfg: Trial config dict (from trials.make_config).
        dispatch_fn: Dispatch function (default: battery_dispatch).
        base: Root directory for trial artifacts.
        simulate_fn: Simulation loop to use; defaults to the open-loop
            simulate_region. Pass mpc.simulate_region_mpc for MPC.

    Returns:
        Dict mapping region name to its result dict (with ledger_df,
        metrics, etc.).
    """
    trial_name = cfg["trial_name"]
    regions = cfg["regions"]

    # Save config first — even if the run fails, the config is on disk
    trials_mod.save_config(cfg, base=base)
    logger.info("Trial '%s' started — regions: %s", trial_name, regions)

    results = {}

    for region in regions:
        if region not in data_by_region:
            logger.warning("No data for region %s — skipping", region)
            continue

        logger.info("Running region %s", region)
        data = data_by_region[region]

        # Run simulation
        sim = simulate_fn if simulate_fn is not None else simulate_region
        sim_result = sim(data, cfg, dispatch_fn=dispatch_fn)

        # Convert ledger to DataFrame
        ledger_df = ledger_mod.to_dataframe(sim_result["ledger"])

        # Compute summary metrics
        metrics = ledger_mod.summarise(ledger_df)

        # Save all artifacts
        trials_mod.save_ledger(ledger_df, trial_name, region, base=base)
        trials_mod.save_metrics(metrics, trial_name, region, base=base)

        # Save forecasts as a DataFrame
        if sim_result["forecasts"]:
            forecast_records = []
            for fc in sim_result["forecasts"]:
                for i, (fprice, aprice) in enumerate(zip(fc["forecast"], fc["actual"])):
                    forecast_records.append({
                        "origin": fc["origin"],
                        "step": i,
                        "forecast": fprice,
                        "actual": aprice,
                    })
            forecast_df = pd.DataFrame(forecast_records)
            trials_mod.save_forecasts(forecast_df, trial_name, region, base=base)

        # Save model state
        if sim_result["model_state"] is not None:
            model_spec = models_mod.get_model(cfg["model"])
            model_dir = trials_mod.trial_dir(trial_name, region, base=base) / "model"
            model_spec["save"](sim_result["model_state"], model_dir)

        results[region] = {
            "ledger_df": ledger_df,
            "metrics": metrics,
            "forecasts": sim_result["forecasts"],
            "model_state": sim_result["model_state"],
        }

        logger.info(
            "Region %s done — revenue: $%.2f, MAE: %.2f",
            region, metrics["total_revenue"], metrics["mae"],
        )

    logger.info("Trial '%s' complete", trial_name)
    return results
