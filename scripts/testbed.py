"""Lightweight dispatch test bed — grid dispatch variants over a frozen fan.

The full common-window eval is slow because every dispatch variant retrains
the model and reforecasts (52 refits x a 4-model fan x tens of thousands of LP
solves). But the dispatch experiments we actually care about — robust min vs
expected value vs a CVaR blend, the ``lambda`` knob, resolve cadence — never
change the forecast. So we split the cost in two:

    build : run the walk-forward **once** per (model, short window), recording
            the full quantile fan at every reforecast origin to disk. This is
            the expensive step (model training), bounded to a short window.
    grid  : replay a whole dispatch grid over the cached fan in seconds —
            no fit, no predict — and score each against the window oracle.

Short, regime-contrasting windows (a spiky summer month and a calm month) keep
a full grid down to minutes, so promising configs can be found before
committing to a 9-hour full-window run. The fixed sub-window oracle
(oracle.py closed-cycle assert) is what makes short-window scoring honest.

Usage::

    python scripts/testbed.py build --models lightgbm_qmean
    python scripts/testbed.py grid  --models lightgbm_qmean
    python scripts/build_testbed_dashboard.py   # then open the subpage
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grian.dispatch import battery_lp as lp
from grian.dispatch import ledger as ledger_mod
from grian.dispatch import oracle
from grian.dispatch.mpc import simulate_region_mpc
from grian.evaluation.analytics import capture_report
from grian.evaluation.trials import make_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("testbed")

REGION = "SA1"
RESOLUTION = "5min"           # set from --resolution in main()
HORIZON = 288                 # day-ahead: 288 @ 5min, 48 @ 30min
SEED = 42
REFORECAST_EVERY = 6          # fixed so cached fans replay against any dispatch
REFIT_DAYS = 28              # monthly refit — sprint speed; matches the point matrix
LOOKBACK_DAYS = 548
DATA_PATH = f"data/processed/{REGION}_5min_sim.parquet"
OUT = Path("outputs/testbed")

_PPD = {"5min": 288, "30min": 48}
_STEPS_PER_30MIN = {"5min": 6, "30min": 1}

# Windows. Short regime-contrasting months for quick iteration; "full" is the
# shared 12-month common-eval span for headline, run-comparable numbers.
WINDOWS = {
    "spike_jan26": ("2026-01-01", "2026-01-31"),
    "calm_sep25": ("2025-09-01", "2025-09-30"),
    "full": ("2025-07-01", "2026-06-30"),
}

DISPATCH = {"power_mw": 100.0, "duration_hours": 2.0,
            "efficiency": 0.85, "max_cycles": 2}

# The dispatch grid replayed over each frozen fan — the probabilistic-dispatch
# ladder plus a CVaR lambda sweep from risk-neutral (EV) to worst-case (robust).
GRID = {
    "point": {"dispatch_mode": "point"},
    "robust": {"dispatch_mode": "scenario"},
    "ev": {"dispatch_mode": "scenario_ev"},
    "cvar_010": {"dispatch_mode": "scenario_cvar", "cvar_lambda": 0.10},
    "cvar_025": {"dispatch_mode": "scenario_cvar", "cvar_lambda": 0.25},
    "cvar_050": {"dispatch_mode": "scenario_cvar", "cvar_lambda": 0.50},
    "cvar_075": {"dispatch_mode": "scenario_cvar", "cvar_lambda": 0.75},
    "cvar_090": {"dispatch_mode": "scenario_cvar", "cvar_lambda": 0.90},
    # True joint mean-CVaR LP (Rockafellar-Uryasev) — a small lambda×alpha grid.
    "meancvar_l50_a50": {"dispatch_mode": "mean_cvar",
                         "cvar_lambda": 0.50, "cvar_alpha": 0.50},
    "meancvar_l50_a10": {"dispatch_mode": "mean_cvar",
                         "cvar_lambda": 0.50, "cvar_alpha": 0.10},
    "meancvar_l90_a10": {"dispatch_mode": "mean_cvar",
                         "cvar_lambda": 0.90, "cvar_alpha": 0.10},
    # Open-loop (daily-commitment) dispatch: forecast + solve once per day, no
    # reforecast churn (Entry 030). Pairs with a calibrated fan (Entry 033) to
    # test whether spike-aware, churn-free probabilistic dispatch beats the
    # ~0.25 MPC ceiling and the point forecast.
    "point_ol": {"dispatch_mode": "point", "open_loop": True},
    "ev_ol": {"dispatch_mode": "scenario_ev", "open_loop": True},
    "cvar050_ol": {"dispatch_mode": "scenario_cvar", "cvar_lambda": 0.50,
                   "open_loop": True},
    # Neutral default lambda/alpha. A full-window sweep found lambda 0.3/alpha 0.3
    # scored higher (0.559), but that beat did NOT survive cross-window validation
    # (point_ol >= mean-CVaR on both the spike and calm months) — it was tuned to
    # the eval window. Open-loop itself is the robust win; the risk-shaping
    # increment is window-specific (Entry 033).
    "meancvar_ol": {"dispatch_mode": "mean_cvar", "cvar_lambda": 0.50,
                    "cvar_alpha": 0.50, "open_loop": True},
    # THE working MPC (Entry 035): daily forecast + re-solve every interval +
    # spike-gated observe. Beats open-loop on every window (spike 0.66, calm tie,
    # full 0.58 > AR 0.543) by discharging into scarcity the forecast misses,
    # while staying provably open-loop on non-spike intervals.
    "mpc_spikegate": {"dispatch_mode": "point", "daily_forecast": True,
                      "observe_gate": 3000.0},
}


def _cfg(data: pd.DataFrame, model: str, window: str, mpc_over: dict) -> dict:
    """Build a trial config for one (model, window) with an mpc override."""
    start, end = WINDOWS[window]
    mpc_over = dict(mpc_over)
    # open_loop: forecast + solve once per horizon (no reforecast churn) — the
    # daily-commitment counterpart to the 30-min-reforecast MPC cadence.
    if mpc_over.pop("open_loop", False):
        mpc_over["resolve_every"] = HORIZON
        mpc_over["reforecast_every"] = HORIZON
    # daily_forecast: the working MPC (Entry 035) — forecast once per day (no
    # reforecast churn) but re-solve every interval with true SOC, so a
    # spike-gated observe can react to scarcity the day-ahead forecast misses.
    if mpc_over.pop("daily_forecast", False):
        mpc_over.setdefault("reforecast_every", HORIZON)
        mpc_over.setdefault("resolve_every", 1)
    return make_config({
        "trial_name": f"testbed_{model}__{window}",
        "model": model,
        "regions": [REGION],
        "resolution": RESOLUTION,
        "target_col": "price",
        "horizon": HORIZON,
        "train_lookback_days": LOOKBACK_DAYS,
        "embargo": 0,
        "seed": SEED,
        "transform": "asinh",
        "loss": "pinball",
        "model_params": _MODEL_PARAMS[model],
        "dispatch": DISPATCH,
        "refit_days": REFIT_DAYS,
        "train_start": str(data.index.min().date()),
        "train_end": str((pd.Timestamp(start) - pd.Timedelta(days=1)).date()),
        "test_start": start,
        "test_end": end,
        "ablations": {"scale_target": True, "use_transform": True,
                      "use_embargo": False, "leak_future": False},
        "mpc": {"resolve_every": REFORECAST_EVERY,
                "reforecast_every": REFORECAST_EVERY, **mpc_over},
    })


# Fan-model params (must emit a quantile fan). Mirrors run_common_eval.
_MODEL_PARAMS = {
    "lightgbm_qmean": {"n_estimators": 150, "learning_rate": 0.05,
                       "quantiles": [0.05, 0.5, 0.9, 0.98]},
    "lightgbm_qmean_weather": {"n_estimators": 150, "learning_rate": 0.05,
                               "quantiles": [0.05, 0.5, 0.9, 0.98],
                               "include_weather": True},
    "lightgbm_qmean_fourier": {"n_estimators": 150, "learning_rate": 0.05,
                               "quantiles": [0.05, 0.5, 0.9, 0.98],
                               "calendar_encoding": "fourier"},
    "lightgbm_qmean_weather_fourier": {"n_estimators": 150, "learning_rate": 0.05,
                                       "quantiles": [0.05, 0.5, 0.9, 0.98],
                                       "include_weather": True,
                                       "calendar_encoding": "fourier"},
    # Split-conformal calibrated fan: same model, quantiles widened to nominal
    # coverage on a held-out tail before dispatch reads them (Entry 033).
    "lightgbm_qmean_cal": {"n_estimators": 150, "learning_rate": 0.05,
                           "quantiles": [0.05, 0.5, 0.9, 0.98],
                           "calibrate": True, "cal_days": 28},
    # Linear quantile fan — the LEAR counterpart, same dispatch grid.
    "lear_qmean": {"quantiles": [0.05, 0.5, 0.9, 0.98], "alpha": 0.01},
    "lear_qmean_weather": {"quantiles": [0.05, 0.5, 0.9, 0.98],
                           "alpha": 0.01, "include_weather": True},
    "lear_qmean_fourier": {"quantiles": [0.05, 0.5, 0.9, 0.98],
                           "alpha": 0.01, "calendar_encoding": "fourier"},
    "lear_qmean_weather_fourier": {"quantiles": [0.05, 0.5, 0.9, 0.98],
                                   "alpha": 0.01, "include_weather": True,
                                   "calendar_encoding": "fourier"},
    # Torch (CPU) quantile fan — same grid as lear_qmean, fit in seconds via a
    # batched pinball GD fit with the Fourier calendar block unpenalised
    # CPU-pinned: the planned MPS backend diverges (Entry 036).
    "lear_qmean_torch": {"quantiles": [0.05, 0.5, 0.9, 0.98], "alpha": 0.01},
    "lear_qmean_torch_weather": {"quantiles": [0.05, 0.5, 0.9, 0.98],
                                 "alpha": 0.01, "include_weather": True},
    "lear_qmean_torch_fourier": {"quantiles": [0.05, 0.5, 0.9, 0.98],
                                 "alpha": 0.01, "calendar_encoding": "fourier"},
    "lear_qmean_torch_weather_fourier": {"quantiles": [0.05, 0.5, 0.9, 0.98],
                                         "alpha": 0.01, "include_weather": True,
                                         "calendar_encoding": "fourier"},
}


def _fan_path(model: str, window: str) -> Path:
    return OUT / "fans" / f"{model}__{window}__{RESOLUTION}.parquet"


def _window_oracle(data: pd.DataFrame, window: str) -> pd.Series:
    """Compute (or load) the perfect-foresight oracle for a window."""
    start, end = WINDOWS[window]
    mask = (data.index >= start) & (
        data.index < pd.Timestamp(end) + pd.Timedelta(days=1))
    cache = OUT / "oracle" / f"{window}_{REGION}_{RESOLUTION}.parquet"
    result = oracle.compute_oracle(
        data.loc[mask, "price"], dt_hours=lp.resolution_dt_hours(RESOLUTION),
        cache_path=cache,
        **{k: DISPATCH[k] for k in
           ("power_mw", "duration_hours", "efficiency", "max_cycles")})
    return result["daily_revenue"]


def _save_fans(fans: dict, path: Path) -> None:
    """Persist ``{origin: {quantile: path}}`` as a long-format parquet."""
    rows = []
    for origin, fan in fans.items():
        for q, arr in fan.items():
            for step, price in enumerate(arr):
                rows.append((origin, float(q), step, float(price)))
    df = pd.DataFrame(rows, columns=["origin", "quantile", "step", "price"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _load_fans(path: Path) -> dict:
    """Reconstruct ``{origin: {quantile: np.ndarray}}`` from parquet."""
    df = pd.read_parquet(path)
    fans: dict = {}
    for origin, g in df.groupby("origin"):
        fan = {}
        for q, gq in g.groupby("quantile"):
            fan[float(q)] = gq.sort_values("step")["price"].to_numpy()
        fans[pd.Timestamp(origin)] = fan
    return fans


def build(data: pd.DataFrame, models: list[str], windows: list[str],
          checkpoint_every: int = 30) -> None:
    """Record the quantile fan for each (model, window) to disk.

    The fan is checkpointed to ``<path>.partial`` every ``checkpoint_every``
    reforecasts and only moved to the final path on completion — so a long build
    is crash-safe and reports progress, and a partial (never a truncated final)
    is what survives an interrupt. Only a *complete* fan counts as cached.
    """
    for model in models:
        for window in windows:
            path = _fan_path(model, window)
            if path.exists():
                logger.info("fan cached: %s", path.name)
                continue
            partial = path.with_suffix(path.suffix + ".partial")
            cfg = _cfg(data, model, window, {"dispatch_mode": "scenario"})
            t0 = time.time()
            out = simulate_region_mpc(
                data, cfg, record_fans=True,
                fan_checkpoint=(lambda f: _save_fans(f, partial), checkpoint_every))
            _save_fans(out["fans"], path)          # final, complete
            partial.unlink(missing_ok=True)
            logger.info("built %s (%d origins) in %.0fs",
                        path.name, len(out["fans"]), time.time() - t0)


def _write_trial_artifacts(
    model: str, dispatch: str, ledger_df: pd.DataFrame, mpc_over: dict
) -> None:
    """Persist a full-window replay as a native trial for the rich dashboard.

    Writes config.json + metrics.json + ledger.parquet under the run base so
    ``build_dashboard`` shows the probabilistic executors alongside every other
    trial — one dashboard, no separate page. Full window only (the dashboard's
    capture denominator is the full-window oracle). The config mirrors what
    run_common_eval writes so the dashboard's settings panel renders correctly.
    """
    base = Path(f"outputs/trials_{RESOLUTION}") / f"{model}__{dispatch}" / REGION
    base.mkdir(parents=True, exist_ok=True)
    ledger_df.to_parquet(base / "ledger.parquet")
    err = ledger_df["actual_price"] - ledger_df["forecast_price"]
    (base / "metrics.json").write_text(json.dumps({
        "total_revenue": float(ledger_df["revenue"].sum()),
        "sharpe_ratio": ledger_mod.sharpe_ratio(ledger_df),
        "peak_drawdown": ledger_mod.peak_drawdown(ledger_df),
        "mae": float(err.abs().mean()),
        "rmse": float((err ** 2).mean() ** 0.5),
        "n_intervals": int(len(ledger_df)),
    }, indent=2))
    (base / "config.json").write_text(json.dumps({
        "trial_name": f"{model}__{dispatch}", "model": model,
        "regions": [REGION], "resolution": RESOLUTION, "horizon": HORIZON,
        "refit_days": REFIT_DAYS, "train_lookback_days": LOOKBACK_DAYS,
        "transform": "asinh", "dispatch": DISPATCH,
        "model_params": _MODEL_PARAMS.get(model, {}),
        "mpc": {"reforecast_every": REFORECAST_EVERY,
                "resolve_every": REFORECAST_EVERY, **mpc_over},
    }, indent=2))


def grid(data: pd.DataFrame, models: list[str], windows: list[str],
         executors: list[str] | None = None) -> None:
    """Replay the dispatch grid over cached fans and score each config.

    Args:
        data: Full price/feature frame.
        models: Fan models to replay (must be built already).
        windows: Windows to score.
        executors: Optional subset of GRID keys to run (e.g. point_ol,
            mpc_spikegate). None runs the whole grid. Unknown names raise.
    """
    if executors:
        unknown = [e for e in executors if e not in GRID]
        if unknown:
            raise ValueError(f"unknown executors {unknown}; choose from "
                             f"{list(GRID)}")
        active = {k: GRID[k] for k in executors}
    else:
        active = GRID
    rows = []
    for model in models:
        for window in windows:
            path = _fan_path(model, window)
            if not path.exists():
                logger.warning("no fan for %s__%s — run build first", model, window)
                continue
            fans = _load_fans(path)
            oracle_daily = _window_oracle(data, window)
            for name, mpc_over in active.items():
                cfg = _cfg(data, model, window, mpc_over)
                t0 = time.time()
                out = simulate_region_mpc(data, cfg, fan_cache=fans)
                ledger_df = ledger_mod.to_dataframe(out["ledger"])
                report = capture_report(ledger_df, oracle_daily)
                if window == "full":  # feed the single rich dashboard natively
                    _write_trial_artifacts(model, name, ledger_df, mpc_over)
                rows.append({
                    "model": model, "window": window, "dispatch": name,
                    "capture": round(report["capture_ratio"], 4),
                    "revenue": round(report["total_revenue"]),
                    "oracle": round(report["oracle_revenue"]),
                })
                logger.info("%s / %s / %-9s capture %.4f (%.1fs)",
                            model, window, name, report["capture_ratio"],
                            time.time() - t0)
    # Per-invocation results file (keyed by the model set) so multiple grids
    # can run in parallel without clobbering a shared file. The dashboards
    # merge every results_<res>*.json.
    results_dir = OUT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(models)[:80]
    (results_dir / f"results_{RESOLUTION}__{tag}.json").write_text(
        json.dumps(rows, indent=2))
    if rows:
        table = pd.DataFrame(rows).sort_values(
            ["window", "capture"], ascending=[True, False])
        logger.info("\n%s", table.to_string(index=False))


def main() -> None:
    """Dispatch to the build or grid sub-command."""
    global RESOLUTION, HORIZON, REFORECAST_EVERY, DATA_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "grid"])
    parser.add_argument("--models", default="lightgbm_qmean")
    parser.add_argument("--windows", default=",".join(WINDOWS))
    parser.add_argument("--resolution", default=RESOLUTION,
                        choices=["5min", "30min"])
    parser.add_argument(
        "--fan-cadence", type=int, default=None,
        help="Reforecast cadence in steps for BOTH build and grid replay. "
             "The executors used for the ablation (point_ol, mpc_spikegate) "
             "reforecast once per day, so daily origins suffice: pass 48 at "
             "30min for a ~48x cheaper build. Default: the 30-min cadence.")
    parser.add_argument(
        "--executors", default=None,
        help="Comma-separated subset of GRID executors to replay in `grid` "
             "(e.g. point_ol,mpc_spikegate). Skips the slow scenario/mean-CVaR "
             "variants. Default: the whole grid.")
    args = parser.parse_args()

    RESOLUTION = args.resolution
    HORIZON = _PPD[RESOLUTION]
    REFORECAST_EVERY = _STEPS_PER_30MIN[RESOLUTION]   # 30-min reforecast cadence
    if args.fan_cadence is not None:
        REFORECAST_EVERY = args.fan_cadence
        logger.info("fan cadence overridden: reforecast every %d steps "
                    "(~%.1f/day)", REFORECAST_EVERY, HORIZON / REFORECAST_EVERY)
    DATA_PATH = f"data/processed/{REGION}_{RESOLUTION}_sim.parquet"

    models = [m for m in args.models.split(",") if m]
    windows = [w for w in args.windows.split(",") if w]
    data = pd.read_parquet(DATA_PATH)
    logger.info("Data: %s -> %s (%s)", data.index.min(), data.index.max(),
                RESOLUTION)

    if args.command == "build":
        build(data, models, windows)
    else:
        executors = ([e for e in args.executors.split(",") if e]
                     if args.executors else None)
        grid(data, models, windows, executors)


if __name__ == "__main__":
    main()
