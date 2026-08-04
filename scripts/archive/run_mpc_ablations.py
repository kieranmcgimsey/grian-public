"""W1.3/W1.4 — MPC frequency and persistence-blend ablations.

Grid over (resolve_every, reforecast_every, persistence_tau) for
lightgbm_rich on the validation window. Isolates three mechanisms:
re-solving from true SOC more often, fresher forecasts, and reacting
to the last observed price on spike onset (Entry 016/017 follow-up;
regret decomposition shows event days with sustained spikes dominate).

Usage:
    python scripts/run_mpc_ablations.py [val|test]
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
logger = logging.getLogger("run_mpc_ablations")

MODEL = "lightgbm_rich"
MODEL_PARAMS = {"n_estimators": 300, "learning_rate": 0.05}

# Baseline champion is r6_f12_p0 (capture 0.536 val). Each variant moves
# one or two knobs. persistence_tau is in intervals (6 = 30 min decay).
GRID = [
    {"resolve_every": 6, "reforecast_every": 6, "persistence_tau": 0},
    {"resolve_every": 1, "reforecast_every": 6, "persistence_tau": 0},
    {"resolve_every": 6, "reforecast_every": 12, "persistence_tau": 6},
    {"resolve_every": 1, "reforecast_every": 6, "persistence_tau": 6},
    # Gated persistence (spike latch): blend only when the last price
    # is extreme. Ungated blending lost ~10 points (Entry 018).
    {"resolve_every": 1, "reforecast_every": 6, "persistence_tau": 6,
     "persistence_gate": 300.0},
    # Final latch attempt: only genuine extremes ($1000+). g300 recovered
    # 9.5 of the 10 ungated points but still trails no-persistence.
    {"resolve_every": 1, "reforecast_every": 6, "persistence_tau": 6,
     "persistence_gate": 1000.0},
    # 5-minute re-forecast: does freshness beyond 30 min help on the
    # fast midday spikes where regret concentrates? (r1_f1 = re-solve
    # AND re-forecast every interval.)
    {"resolve_every": 1, "reforecast_every": 1, "persistence_tau": 0},
]

# The variant accepted on validation, run once on test for the headline.
ACCEPTED = {"resolve_every": 6, "reforecast_every": 6, "persistence_tau": 0}


def main():
    """Run the ablation grid (val) or the accepted variant (test)."""
    window = sys.argv[1] if len(sys.argv) > 1 else "val"
    grid = GRID if window == "val" else [ACCEPTED]
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
    for mpc_cfg in grid:
        tag = (f"r{mpc_cfg['resolve_every']}"
               f"_f{mpc_cfg['reforecast_every']}"
               f"_p{mpc_cfg['persistence_tau']}")
        if mpc_cfg.get("persistence_gate"):
            tag += f"_g{int(mpc_cfg['persistence_gate'])}"
        trial_name = f"mpcx_{tag}_{MODEL}_{window}"
        ledger_path = Path(BASE_DIR) / trial_name / REGION / "ledger.parquet"

        if not ledger_path.exists():
            cfg = make_config({
                "trial_name": trial_name,
                "model": MODEL,
                "regions": [REGION],
                "resolution": RESOLUTION,
                "horizon": 288,
                "refit_days": 7,
                "embargo": 0,
                "seed": SEED,
                "transform": "asinh",
                "loss": "pinball",
                "model_params": MODEL_PARAMS,
                "dispatch": DISPATCH,
                "mpc": mpc_cfg,
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
            "model": MODEL, "executor": f"mpc-{tag}", "window": window,
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
