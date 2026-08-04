"""Round 2 — model comparison on the accepted MPC executor (r6_f6).

Three questions from Entries 017/018, all on validation:
1. Is qmean's MPC loss a capacity artefact? (rich at 150 trees)
2. Does qmean at 150 trees improve under the fresher f6 executor?
3. Does the lead-dependent median/mean hybrid beat both?

Usage:
    python scripts/run_model_round2.py
"""
# ruff: noqa: I001 — sys.path setup must precede the script-local import

import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_capture_baselines import (
    BASE_DIR, DISPATCH, REGION, RESOLUTION, RESULTS_MD, SEED,
    WINDOWS, window_oracle,
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
logger = logging.getLogger("run_model_round2")

EXECUTOR = {"resolve_every": 6, "reforecast_every": 6, "persistence_tau": 0}

VARIANTS = [
    ("rich150", "lightgbm_rich",
     {"n_estimators": 150, "learning_rate": 0.05}),
    ("qmean150", "lightgbm_qmean",
     {"n_estimators": 150, "learning_rate": 0.05,
      "quantiles": [0.05, 0.5, 0.9, 0.98]}),
    ("qhybrid150", "lightgbm_qmean",
     {"n_estimators": 150, "learning_rate": 0.05,
      "quantiles": [0.05, 0.5, 0.9, 0.98], "mean_from_step": 12}),
]


def main():
    """Run the round-2 variants on validation and append results."""
    window = sys.argv[1] if len(sys.argv) > 1 else "val"
    data = pd.read_parquet(f"data/processed/{REGION}_5min_sim.parquet")
    oracle_result = window_oracle(data, window)
    top10 = (oracle_result["daily_revenue"].sort_values().tail(10).sum()
             / oracle_result["total_revenue"])

    import subprocess
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip() or "unknown"

    rows = []
    for tag, model, params in VARIANTS:
        trial_name = f"r2_{tag}_f6_{window}"
        ledger_path = Path(BASE_DIR) / trial_name / REGION / "ledger.parquet"
        if not ledger_path.exists():
            cfg = make_config({
                "trial_name": trial_name,
                "model": model,
                "regions": [REGION],
                "resolution": RESOLUTION,
                "horizon": 288,
                "refit_days": 7,
                "embargo": 0,
                "seed": SEED,
                "transform": "asinh",
                "loss": "pinball",
                "model_params": params,
                "dispatch": DISPATCH,
                "mpc": EXECUTOR,
                "ablations": {
                    "scale_target": True, "use_transform": True,
                    "use_embargo": False, "leak_future": False,
                },
                **WINDOWS[window],
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
        rows.append({
            "model": f"{model}[{tag}]", "executor": "mpc-r6_f6",
            "window": window,
            "capture": round(report["capture_ratio"], 4),
            "revenue": round(report["total_revenue"]),
            "oracle": round(report["oracle_revenue"]),
            "spearman": round(report["mean_daily_spearman"], 3),
            "oracle_top10_share": round(top10, 3),
            "git_sha": sha,
        })

    print("\n" + pd.DataFrame(rows).to_string(index=False))
    with RESULTS_MD.open("a") as f:
        for r in rows:
            f.write(
                f"| 2026-07-12 | {r['model']} | {r['executor']} | "
                f"{r['window']} | {r['capture']:.3f} | ${r['revenue']:,} | "
                f"${r['oracle']:,} | {r['spearman']} | "
                f"{r['oracle_top10_share']} | {r['git_sha']} |\n"
            )
    logger.info("Appended to %s", RESULTS_MD)


if __name__ == "__main__":
    main()
