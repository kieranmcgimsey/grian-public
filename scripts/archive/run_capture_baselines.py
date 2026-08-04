"""W0.4 — honest capture-ratio baselines (campaign plan §5, Phase 0).

Computes the perfect-foresight oracle for the validation and test
windows, re-runs the open-loop baselines with fixed LP physics and
sim embargo = 0, and reports capture ratios. Appends one row per
(model, window) to the headline table.

Usage:
    python scripts/run_capture_baselines.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grian.sim import lp, oracle
from grian.sim.analytics import capture_report
from grian.sim.runner import run_trial
from grian.sim.trials import make_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_capture_baselines")

REGION = "SA1"
RESOLUTION = "5min"
HORIZON = 288
SEED = 42
BASE_DIR = "outputs/trials"
RESULTS_MD = Path("outputs/plans/capture_campaign_results.md")

# Campaign windows (plan §1.2). Data ends 2024-01-30.
WINDOWS = {
    "val": {
        "train_start": "2023-01-01", "train_end": "2023-06-30",
        "test_start": "2023-07-01", "test_end": "2023-09-30",
    },
    "test": {
        "train_start": "2023-01-01", "train_end": "2023-09-30",
        "test_start": "2023-10-01", "test_end": "2024-01-30",
    },
}

MODELS = {
    "naive_similar_day": {},
    "autoregression": {},
    "lightgbm_rich": {"n_estimators": 300, "learning_rate": 0.05},
    "lightgbm_qmean": {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "quantiles": [0.05, 0.5, 0.9, 0.98],
    },
}

DISPATCH = {
    "power_mw": 100.0,
    "duration_hours": 2.0,
    "efficiency": 0.85,
    "max_cycles": 2,
}


def window_oracle(data: pd.DataFrame, window: str) -> dict:
    """Compute (or load cached) oracle for one evaluation window."""
    w = WINDOWS[window]
    mask = (data.index >= w["test_start"]) & (
        data.index < pd.Timestamp(w["test_end"]) + pd.Timedelta(days=1)
    )
    prices = data.loc[mask, "price"]
    cache = Path(BASE_DIR) / "_oracle" / REGION / f"{window}.parquet"
    t0 = time.time()
    result = oracle.compute_oracle(
        prices,
        dt_hours=lp.resolution_dt_hours(RESOLUTION),
        cache_path=cache,
        **{k: DISPATCH[k] for k in
           ("power_mw", "duration_hours", "efficiency", "max_cycles")},
    )
    logger.info(
        "Oracle[%s]: $%.0f over %d days (%.1fs) — top-10-day share %.1f%%",
        window, result["total_revenue"], len(result["daily_revenue"]),
        time.time() - t0,
        100 * result["daily_revenue"].sort_values().tail(10).sum()
        / result["total_revenue"],
    )
    return result


def run_one(data: pd.DataFrame, model: str, window: str,
            oracle_result: dict) -> dict:
    """Run one open-loop trial and score it against the oracle."""
    w = WINDOWS[window]
    trial_name = f"cap0_{model}_{window}"

    ledger_path = Path(BASE_DIR) / trial_name / REGION / "ledger.parquet"
    if not ledger_path.exists():
        cfg = make_config({
            "trial_name": trial_name,
            "model": model,
            "regions": [REGION],
            "resolution": RESOLUTION,
            "horizon": HORIZON,
            "refit_days": 7,
            # Sim embargo is 0 by design — deployment realism (plan §2.5).
            "embargo": 0,
            "seed": SEED,
            "transform": "asinh",
            "loss": "pinball",
            "model_params": MODELS[model],
            "dispatch": DISPATCH,
            "ablations": {
                "scale_target": True, "use_transform": True,
                "use_embargo": False, "leak_future": False,
            },
            **w,
        })
        t0 = time.time()
        run_trial({REGION: data}, cfg, base=BASE_DIR)
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
    """Compute oracles, run baselines, emit the headline table."""
    data = pd.read_parquet(f"data/processed/{REGION}_5min_sim.parquet")
    logger.info("Data: %d rows, %s → %s", len(data),
                data.index.min(), data.index.max())

    sha = "unknown"
    try:
        import subprocess
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        pass

    rows = []
    for window in WINDOWS:
        oracle_result = window_oracle(data, window)
        top10 = (oracle_result["daily_revenue"].sort_values().tail(10).sum()
                 / oracle_result["total_revenue"])
        for model in MODELS:
            r = run_one(data, model, window, oracle_result)
            rows.append({
                "model": model, "executor": "open-loop", "window": window,
                "capture": round(r["capture_ratio"], 4),
                "revenue": round(r["total_revenue"]),
                "oracle": round(r["oracle_revenue"]),
                "spearman": round(r["mean_daily_spearman"], 3),
                "oracle_top10_share": round(top10, 3),
                "git_sha": sha,
            })

    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False))

    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not RESULTS_MD.exists()
    with RESULTS_MD.open("a") as f:
        if header_needed:
            f.write("# Capture campaign — headline results\n\n"
                    "Append-only. One row per (model, executor, window). "
                    "See capture_campaign.md §6.4.\n\n")
            f.write("| date | model | executor | window | capture | "
                    "revenue | oracle | spearman | top10 share | sha |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| 2026-07-11 | {r['model']} | {r['executor']} | "
                f"{r['window']} | {r['capture']:.3f} | ${r['revenue']:,} | "
                f"${r['oracle']:,} | {r['spearman']} | "
                f"{r['oracle_top10_share']} | {r['git_sha']} |\n"
            )
    logger.info("Wrote %s", RESULTS_MD)

    # Also drop a machine-readable copy next to the table.
    with (RESULTS_MD.parent / "capture_baselines_w04.json").open("w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
