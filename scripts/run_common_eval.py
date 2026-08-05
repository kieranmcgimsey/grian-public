"""Common-window evaluation: every model configuration on one identical test set.

The old val/test paradigm scored different configs on disjoint windows, which
made them incomparable. This harness instead evaluates every configuration by
walk-forward over a *single shared span* (train on all history up to each
trading day, trade the next day), and scores each against one perfect-foresight
oracle computed over that same span.

Because capture ratio over any sub-window is simply
``sum(model daily revenue) / sum(oracle daily revenue)``, the dashboard can then
recompute capture for any test window (e.g. the most recent 12 months) live from
the saved daily series — an identical test set for every configuration, with the
window as a free parameter.

Grid: {naive_similar_day, autoregression, lightgbm_rich, lightgbm_qmean}
      × {open-loop one-shot, 30-min-reforecast MPC}.

Usage::

    python scripts/run_common_eval.py                        # full grid
    python scripts/run_common_eval.py --executors openloop   # fast subset first
    python scripts/run_common_eval.py --test-start 2024-07-01 --test-end 2026-06-30
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grian.sim import lp, oracle
from grian.sim.analytics import capture_report
from grian.sim.mpc import simulate_region_mpc
from grian.sim.runner import run_trial
from grian.sim.trials import make_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_common_eval")

REGION = "SA1"
RESOLUTION = "5min"           # overridable via --resolution {5min,30min}
HORIZON = 288                 # day-ahead: 288 @ 5min, 48 @ 30min (set from resolution)
SEED = 42
BASE_DIR = "outputs/trials"   # overridable via --base (for isolated reruns)
REFIT_DAYS = 7                # weekly; cheap now that training is a rolling window
LOOKBACK_DAYS = 548           # ~18 months rolling train window (see --lookback-days)
DATA_PATH = f"data/processed/{REGION}_5min_sim.parquet"

# Intervals per day / per 30 min, by resolution — the MPC cadences and the
# day-ahead horizon are expressed in physical time and scaled to the grid.
_PPD = {"5min": 288, "30min": 48}
_STEPS_PER_30MIN = {"5min": 6, "30min": 1}

# Default shared eval span: the most recent 12 months of available data — one
# full seasonal cycle (incl. summer spikes), the sanity-proof headline window.
# Training expands from the start of the data up to each trading day. A longer
# pool is possible but the perfect-foresight oracle costs ~18 s per 7-day block,
# and MPC re-solves every 30 min, so multi-year spans run for hours.
DEFAULT_TEST_START = "2025-07-01"
DEFAULT_TEST_END = "2026-06-30"

MODELS = {
    "naive_similar_day": {},
    "autoregression": {},   # one-hot calendar (default)
    "autoregression_fourier": {"calendar_encoding": "fourier"},
    "autoregression_ordinal": {"calendar_encoding": "ordinal"},
    # LEAR family — regularised linear on the same rich features as
    # lightgbm_rich (canonical EPF benchmark; clean linear-vs-GBM contrast).
    "lear": {"estimator": "lasso"},   # one-hot calendar (default)
    "ridge": {"estimator": "ridge"},
    "elasticnet": {"estimator": "elasticnet"},
    "lear_weather": {"estimator": "lasso", "include_weather": True},
    # Shape-preserving LEAR: Lasso on a rich raw-lag basis + calendar, no
    # mean-reverting smoothers. Tests the handoff hypothesis that LEAR's
    # smoothers flatten the peaks capture pays for (Entry 032). ols_lean is
    # the same basis with OLS — a better-specified AR — to split the "features"
    # effect from the "regularisation" effect versus AR (OLS, 3 lags).
    "lear_lean": {"estimator": "lasso", "feature_set": "lean"},
    "ols_lean": {"estimator": "ols", "feature_set": "lean"},
    # Calendar-encoding ablation for LEAR (one-hot vs Fourier vs ordinal).
    "lear_fourier": {"estimator": "lasso", "calendar_encoding": "fourier"},
    "lear_weather_fourier": {"estimator": "lasso", "include_weather": True,
                             "calendar_encoding": "fourier"},
    "lear_ordinal": {"estimator": "lasso", "calendar_encoding": "ordinal"},
    "lightgbm_rich": {"n_estimators": 300, "learning_rate": 0.05},
    "lightgbm_rich_weather": {"n_estimators": 300, "learning_rate": 0.05,
                              "include_weather": True},
    # Fourier-calendar tree ablation (cyclic sin/cos calendar features).
    "lightgbm_rich_fourier": {"n_estimators": 300, "learning_rate": 0.05,
                              "calendar_encoding": "fourier"},
    "lightgbm_rich_weather_fourier": {"n_estimators": 300, "learning_rate": 0.05,
                                      "include_weather": True,
                                      "calendar_encoding": "fourier"},
    # Spike-precursor features on — the anticipating forecast to beat the
    # baseline (Entry 030).
    "lightgbm_rich_scarcity": {"n_estimators": 300, "learning_rate": 0.05,
                               "include_scarcity": True},
    # Decision-focused training: scarcity features + a fit that up-weights
    # high-price intervals (magnitude scheme), so the model is scored where
    # capture is earned. The direct attack on the accuracy-vs-value gap
    #.
    "lightgbm_rich_dfl": {"n_estimators": 300, "learning_rate": 0.05,
                          "include_scarcity": True,
                          "sample_weighting": {"scheme": "magnitude",
                                               "strength": 1.0}},
    "lightgbm_qmean": {
        "n_estimators": 150, "learning_rate": 0.05,
        "quantiles": [0.05, 0.5, 0.9, 0.98],
    },
    "lightgbm_qmean_weather": {
        "n_estimators": 150, "learning_rate": 0.05,
        "quantiles": [0.05, 0.5, 0.9, 0.98], "include_weather": True,
    },
    # Conformal-calibrated fan (Entry 033): upper quantiles widened to nominal
    # coverage on a held-out tail so probabilistic dispatch sees spike risk.
    "lightgbm_qmean_cal": {
        "n_estimators": 150, "learning_rate": 0.05,
        "quantiles": [0.05, 0.5, 0.9, 0.98], "calibrate": True, "cal_days": 28,
    },
}

# Executors (all MPC now observes the true current price — see mpc.py). The
# cadences are physical: mpc30 = 30-min resolve + 30-min reforecast (champion),
# mpc5 = re-solve every native dispatch interval. Both are scaled to the grid so
# the same executor names mean the same wall-clock cadence at any resolution.
def build_executors(resolution: str) -> dict:
    """Executor definitions with cadences scaled to ``resolution``.

    Args:
        resolution: "5min" or "30min".

    Returns:
        Mapping of executor name to its MPC config (or ``{"mpc": None}``).
    """
    s = _STEPS_PER_30MIN[resolution]   # native steps per 30 min (6 @ 5min, 1 @ 30min)
    ppd = _PPD[resolution]             # steps per day (daily reforecast cadence)
    return {
        "openloop": {"mpc": None},
        "mpc30": {"mpc": {"resolve_every": s, "reforecast_every": s}},
        "mpc5": {"mpc": {"resolve_every": 1, "reforecast_every": s}},
        # THE working MPC (Entry 035): forecast once/day, re-solve every interval,
        # observe only genuine spikes (>= $3k). Beats open-loop by reacting to
        # scarcity the forecast misses; provably open-loop otherwise.
        "mpc_spike": {"mpc": {"resolve_every": 1, "reforecast_every": ppd,
                              "observe_gate": 3000.0}},
        # Probabilistic dispatch (quantile models only — need a fan):
        "mpc_qgate": {"mpc": {"resolve_every": s, "reforecast_every": s,
                              "dispatch_mode": "qgate", "q_low": 0.1, "q_high": 0.9}},
        "mpc_robust": {"mpc": {"resolve_every": s, "reforecast_every": s,
                               "dispatch_mode": "qgate",
                               "q_low": 0.02, "q_high": 0.98}},
        # Scenario dispatch: per-quantile LPs (each internally ranked), then fuse
        # actions in decision space — robust min / expected value / CVaR blend.
        "mpc_scenario": {"mpc": {"resolve_every": s, "reforecast_every": s,
                                 "dispatch_mode": "scenario"}},
        "mpc_ev": {"mpc": {"resolve_every": s, "reforecast_every": s,
                           "dispatch_mode": "scenario_ev"}},
        "mpc_cvar": {"mpc": {"resolve_every": s, "reforecast_every": s,
                             "dispatch_mode": "scenario_cvar", "cvar_lambda": 0.5}},
        # True mean-CVaR: one joint Rockafellar-Uryasev LP across scenarios.
        "mpc_meancvar": {"mpc": {"resolve_every": s, "reforecast_every": s,
                                 "dispatch_mode": "mean_cvar",
                                 "cvar_lambda": 0.5, "cvar_alpha": 0.5}},
    }


EXECUTORS = build_executors(RESOLUTION)
# Models that emit a quantile fan (required by the qgate/robust executors).
FAN_MODELS = {"lightgbm_qmean", "lightgbm_qmean_weather", "lightgbm_qmean_cal",
              "qra_ensemble", "lear_qmean", "lear_qmean_weather"}

DISPATCH = {
    "power_mw": 100.0, "duration_hours": 2.0,
    "efficiency": 0.85, "max_cycles": 2,
}


def common_oracle(data: pd.DataFrame, test_start: str, test_end: str) -> dict:
    """Compute (or load cached) the single oracle over the shared span."""
    mask = (data.index >= test_start) & (
        data.index < pd.Timestamp(test_end) + pd.Timedelta(days=1)
    )
    prices = data.loc[mask, "price"]
    cache = Path(BASE_DIR) / "_oracle" / REGION / "common.parquet"
    t0 = time.time()
    result = oracle.compute_oracle(
        prices,
        dt_hours=lp.resolution_dt_hours(RESOLUTION),
        cache_path=cache,
        **{k: DISPATCH[k] for k in
           ("power_mw", "duration_hours", "efficiency", "max_cycles")},
    )
    top10 = (result["daily_revenue"].sort_values().tail(10).sum()
             / result["total_revenue"])
    logger.info(
        "Oracle: $%.0f over %d days (%.1fs) — top-10-day share %.1f%%",
        result["total_revenue"], len(result["daily_revenue"]),
        time.time() - t0, 100 * top10,
    )
    return result


def run_config(data, model, executor, test_start, test_end, oracle_result):
    """Run one (model, executor) config over the shared span and score it.

    Returns ``None`` for skipped combos (probabilistic executor + point model).
    """
    mpc_cfg = EXECUTORS[executor].get("mpc")
    fan_modes = ("qgate", "scenario", "scenario_ev", "scenario_cvar")
    if (mpc_cfg and mpc_cfg.get("dispatch_mode") in fan_modes
            and model not in FAN_MODELS):
        logger.info("skip %s__%s — %s needs a quantile model", model, executor, model)
        return None
    trial_name = f"{model}__{executor}"
    ledger_path = Path(BASE_DIR) / trial_name / REGION / "ledger.parquet"

    if not ledger_path.exists():
        cfg = make_config({
            "trial_name": trial_name,
            "model": model,
            "regions": [REGION],
            "resolution": RESOLUTION,
            "horizon": HORIZON,
            "refit_days": REFIT_DAYS,
            "train_lookback_days": LOOKBACK_DAYS,
            "embargo": 0,  # deployment realism: trade next day on today's model
            "seed": SEED,
            "transform": "asinh",
            "loss": "pinball",
            "model_params": MODELS[model],
            "dispatch": DISPATCH,
            "train_start": str(data.index.min().date()),
            "train_end": str((pd.Timestamp(test_start) - pd.Timedelta(days=1)).date()),
            "test_start": test_start,
            "test_end": test_end,
            "ablations": {
                "scale_target": True, "use_transform": True,
                "use_embargo": False, "leak_future": False,
            },
            **({"mpc": EXECUTORS[executor]["mpc"]}
               if EXECUTORS[executor]["mpc"] else {}),
        })
        simulate_fn = simulate_region_mpc if EXECUTORS[executor].get("mpc") else None
        t0 = time.time()
        run_trial({REGION: data}, cfg, base=BASE_DIR, simulate_fn=simulate_fn)
        logger.info("%s ran in %.1fs", trial_name, time.time() - t0)

    ledger_df = pd.read_parquet(ledger_path)
    report = capture_report(ledger_df, oracle_result["daily_revenue"])
    logger.info(
        "%s: capture %.3f  ($%.0f / $%.0f)",
        trial_name, report["capture_ratio"],
        report["total_revenue"], report["oracle_revenue"],
    )
    return report


def main() -> None:
    """Run the requested grid over the shared span and report capture ratios."""
    global BASE_DIR, REFIT_DAYS, LOOKBACK_DAYS
    global RESOLUTION, HORIZON, DATA_PATH, EXECUTORS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--executors", default=",".join(EXECUTORS))
    parser.add_argument("--resolution", default=RESOLUTION, choices=["5min", "30min"],
                        help="Grid resolution; sets horizon, data path, MPC cadence")
    parser.add_argument("--test-start", default=DEFAULT_TEST_START)
    parser.add_argument("--test-end", default=DEFAULT_TEST_END)
    parser.add_argument("--base", default=BASE_DIR, help="Output dir (isolate reruns)")
    parser.add_argument("--refit-days", type=int, default=REFIT_DAYS)
    parser.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS,
                        help="Rolling train window in days; 0 = expanding")
    args = parser.parse_args()

    BASE_DIR = args.base
    REFIT_DAYS = args.refit_days
    LOOKBACK_DAYS = args.lookback_days or None
    RESOLUTION = args.resolution
    HORIZON = _PPD[RESOLUTION]
    DATA_PATH = f"data/processed/{REGION}_{RESOLUTION}_sim.parquet"
    EXECUTORS = build_executors(RESOLUTION)
    logger.info("Resolution %s → horizon %d, data %s", RESOLUTION, HORIZON, DATA_PATH)

    models = [m for m in args.models.split(",") if m]
    executors = [e for e in args.executors.split(",") if e]

    data = pd.read_parquet(DATA_PATH)
    logger.info("Data: %s → %s", data.index.min(), data.index.max())
    oracle_result = common_oracle(data, args.test_start, args.test_end)

    rows = []
    for executor in executors:      # open-loop first (fast), then MPC
        for model in models:
            try:
                report = run_config(
                    data, model, executor,
                    args.test_start, args.test_end, oracle_result,
                )
            except Exception:       # never let one bad config kill the matrix
                logger.exception("config %s__%s failed — skipping", model, executor)
                continue
            if report is None:      # skipped (e.g. qgate on a point model)
                continue
            rows.append({
                "config": f"{model}__{executor}", "model": model, "executor": executor,
                "capture": round(report["capture_ratio"], 4),
                "revenue": round(report["total_revenue"]),
            })

    table = pd.DataFrame(rows).sort_values("capture", ascending=False)
    logger.info("\n%s", table.to_string(index=False))


if __name__ == "__main__":
    main()
