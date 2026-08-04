"""W1.3 — MPC vs open-loop capture ratios (campaign plan Phase 1).

Runs the receding-horizon MPC executor for the baseline models on the
validation and test windows, scores against the cached oracle, and
appends to the headline table.

Usage:
    python scripts/run_mpc_trials.py [val] [test] [--models m1,m2]
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_capture_baselines import (
    BASE_DIR,
    DISPATCH,
    MODELS,
    REGION,
    RESOLUTION,
    RESULTS_MD,
    SEED,
    WINDOWS,
    window_oracle,
)

from grian.sim.analytics import capture_report
from grian.sim.mpc import simulate_region_mpc
from grian.sim.runner import run_trial
from grian.sim.trials import make_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_mpc_trials")

HORIZON = 288
MPC = {"resolve_every": 6, "reforecast_every": 12}


def run_one_mpc(data, model, window, oracle_result):
    """Run one MPC trial and score it against the oracle."""
    w = WINDOWS[window]
    trial_name = f"mpc_{model}_{window}"

    ledger_path = Path(BASE_DIR) / trial_name / REGION / "ledger.parquet"
    if not ledger_path.exists():
        cfg = make_config({
            "trial_name": trial_name,
            "model": model,
            "regions": [REGION],
            "resolution": RESOLUTION,
            "horizon": HORIZON,
            "refit_days": 7,
            "embargo": 0,
            "seed": SEED,
            "transform": "asinh",
            "loss": "pinball",
            "model_params": MODELS[model],
            "dispatch": DISPATCH,
            "mpc": MPC,
            "ablations": {
                "scale_target": True, "use_transform": True,
                "use_embargo": False, "leak_future": False,
            },
            **w,
        })
        t0 = time.time()
        run_trial({REGION: data}, cfg, base=BASE_DIR,
                  simulate_fn=simulate_region_mpc)
        logger.info("%s ran in %.1fs", trial_name, time.time() - t0)

    ledger_df = pd.read_parquet(ledger_path)
    report = capture_report(ledger_df, oracle_result["daily_revenue"])
    logger.info(
        "%s: capture %.3f  ($%.0f / $%.0f)  spearman %.3f",
        trial_name, report["capture_ratio"], report["total_revenue"],
        report["oracle_revenue"], report["mean_daily_spearman"],
    )
    return report


def main():
    """Run MPC trials for the requested windows and append results."""
    args = sys.argv[1:]
    models = ["naive_similar_day", "lightgbm_rich"]
    if "--models" in args:
        i = args.index("--models")
        models = args[i + 1].split(",")
        args = args[:i] + args[i + 2:]
    windows = args or ["val", "test"]
    data = pd.read_parquet(f"data/processed/{REGION}_5min_sim.parquet")

    import subprocess
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip() or "unknown"

    rows = []
    for window in windows:
        oracle_result = window_oracle(data, window)
        top10 = (oracle_result["daily_revenue"].sort_values().tail(10).sum()
                 / oracle_result["total_revenue"])
        for model in models:
            r = run_one_mpc(data, model, window, oracle_result)
            rows.append({
                "model": model, "executor": "mpc-30m", "window": window,
                "capture": round(r["capture_ratio"], 4),
                "revenue": round(r["total_revenue"]),
                "oracle": round(r["oracle_revenue"]),
                "spearman": round(r["mean_daily_spearman"], 3),
                "oracle_top10_share": round(top10, 3),
                "git_sha": sha,
            })

    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False))

    with RESULTS_MD.open("a") as f:
        for r in rows:
            f.write(
                f"| 2026-07-11 | {r['model']} | {r['executor']} | "
                f"{r['window']} | {r['capture']:.3f} | ${r['revenue']:,} | "
                f"${r['oracle']:,} | {r['spearman']} | "
                f"{r['oracle_top10_share']} | {r['git_sha']} |\n"
            )
    logger.info("Appended to %s", RESULTS_MD)


if __name__ == "__main__":
    main()
