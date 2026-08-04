"""Scarcity-feature A/B on the champion executor.

Runs lightgbm_rich under the champion MPC config (30-min reforecast)
with and without the spike-precursor features, so the only difference
is the feature set. Validation first; pass `test` to confirm a winner.

Usage:
    python scripts/run_scarcity_ab.py [val|test]
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
logger = logging.getLogger("run_scarcity_ab")

EXECUTOR = {"resolve_every": 6, "reforecast_every": 6}
VARIANTS = [
    ("baseline", False),   # champion feature set
    ("scarcity", True),    # + spike-precursor features
]


def main():
    """Run the A/B on the requested window and append results."""
    window = sys.argv[1] if len(sys.argv) > 1 else "val"
    data = pd.read_parquet(f"data/processed/{REGION}_5min_sim.parquet")
    orc = window_oracle(data, window)
    top10 = (orc["daily_revenue"].sort_values().tail(10).sum()
             / orc["total_revenue"])

    import subprocess
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip() or "unknown"

    rows = []
    for tag, scarcity in VARIANTS:
        trial_name = f"scar_{tag}_{window}"
        ledger_path = Path(BASE_DIR) / trial_name / REGION / "ledger.parquet"
        if not ledger_path.exists():
            cfg = make_config({
                "trial_name": trial_name,
                "model": "lightgbm_rich",
                "regions": [REGION],
                "resolution": RESOLUTION,
                "horizon": 288,
                "refit_days": 7,
                "embargo": 0,
                "seed": SEED,
                "transform": "asinh",
                "loss": "pinball",
                "model_params": {"n_estimators": 300, "learning_rate": 0.05,
                                 "include_scarcity": scarcity},
                "dispatch": DISPATCH,
                "mpc": EXECUTOR,
                "ablations": {"scale_target": True, "use_transform": True,
                              "use_embargo": False, "leak_future": False},
                **WINDOWS[window],
            })
            t0 = time.time()
            run_trial({REGION: data}, cfg, base=BASE_DIR,
                      simulate_fn=simulate_region_mpc)
            logger.info("%s ran in %.1fs", trial_name, time.time() - t0)

        ledger = pd.read_parquet(ledger_path)
        rep = capture_report(ledger, orc["daily_revenue"])
        logger.info(
            "%s: capture %.4f  ($%.0f / $%.0f)  spearman %.3f",
            trial_name, rep["capture_ratio"], rep["total_revenue"],
            rep["oracle_revenue"], rep["mean_daily_spearman"],
        )
        rows.append({
            "tag": tag, "capture": round(rep["capture_ratio"], 4),
            "revenue": round(rep["total_revenue"]),
            "spearman": round(rep["mean_daily_spearman"], 3),
            "sha": sha,
        })

    delta = rows[1]["capture"] - rows[0]["capture"]
    print(f"\nscarcity effect ({window}): {delta:+.4f} capture "
          f"({rows[0]['capture']:.4f} -> {rows[1]['capture']:.4f})")

    with RESULTS_MD.open("a") as f:
        for r in rows:
            f.write(
                f"| 2026-07-12 | lightgbm_rich[{r['tag']}] | mpc-r6_f6 | "
                f"{window} | {r['capture']:.3f} | ${r['revenue']:,} | "
                f"$-- | {r['spearman']} | {round(top10, 3)} | {r['sha']} |\n"
            )


if __name__ == "__main__":
    main()
