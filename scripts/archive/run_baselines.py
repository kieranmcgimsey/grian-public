"""Run all baseline models and ablations on SA1 NEM data.

Downloads 5-minute dispatch prices, builds the dataset, then runs:
  1. All 4 models (naive, AR, LightGBM, MLP) with correct config.
  2. All 6 ablations for LightGBM (the main ablation target).

Results are saved to outputs/trials/ for the dashboard.

Usage:
    python scripts/run_baselines.py
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Ensure the src directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grian.data import load_demand, load_prices
from grian.sim.ablations import make_ablation_suite
from grian.sim.runner import run_trial
from grian.sim.trials import make_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_baselines")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REGION = "SA1"

# Training: 2023-01-01 to 2023-09-30 (9 months)
# Test:     2023-10-01 to 2024-01-31 (4 months)
# This gives enough history for lag features while keeping runs manageable.
TRAIN_START = "2023-01-01"
TRAIN_END = "2023-09-30"
TEST_START = "2023-10-01"
TEST_END = "2024-01-31"

# Full data range includes both train and test
DATA_START = TRAIN_START
DATA_END = TEST_END

RESOLUTION = "5min"
HORIZON = 288  # 288 × 5min = day-ahead
SEED = 42

BASE_DIR = "outputs/trials"


# ---------------------------------------------------------------------------
# Step 1: Download and prepare data
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Download and assemble the 5-minute price + demand dataset."""
    cache = "data/raw/nemosis"
    cache_file = Path("data/processed") / f"{REGION}_5min_sim.parquet"

    if cache_file.exists():
        logger.info("Loading cached dataset from %s", cache_file)
        return pd.read_parquet(cache_file)

    logger.info("Downloading 5-minute prices for %s [%s, %s]", REGION, DATA_START, DATA_END)
    prices = load_prices(REGION, DATA_START, DATA_END, cache=cache, resolution=RESOLUTION)

    logger.info("Downloading 5-minute demand for %s [%s, %s]", REGION, DATA_START, DATA_END)
    demand = load_demand(REGION, DATA_START, DATA_END, cache=cache)

    df = prices.join(demand, how="outer")

    # Forward-fill small gaps (up to 1 hour = 12 intervals)
    df = df.ffill(limit=12)

    # Drop any remaining NaN rows
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning("Dropped %d rows with NaN (%.2f%%)", n_dropped, 100 * n_dropped / n_before)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_file)
    logger.info("Cached dataset to %s (%d rows)", cache_file, len(df))

    return df


# ---------------------------------------------------------------------------
# Step 2: Run all 4 correct baselines
# ---------------------------------------------------------------------------

def run_correct_baselines(data: pd.DataFrame) -> None:
    """Run each model type with the correct (no-ablation) config."""
    models = [
        "naive_similar_day", "autoregression", "lightgbm",
        "simple_mlp", "lightgbm_rich", "lstm",
    ]

    for model_name in models:
        # Skip if already completed
        metrics_path = Path(BASE_DIR) / f"{model_name}_correct" / REGION / "metrics.json"
        if metrics_path.exists():
            logger.info("Skipping %s_correct — already exists", model_name)
            continue

        logger.info("=" * 60)
        logger.info("Running correct baseline: %s", model_name)
        logger.info("=" * 60)

        # Model-specific params
        model_params = {}
        if model_name == "lightgbm":
            model_params = {"n_estimators": 200, "learning_rate": 0.05}
        elif model_name == "lightgbm_rich":
            model_params = {"n_estimators": 300, "learning_rate": 0.05}
        elif model_name == "simple_mlp":
            model_params = {"epochs": 30, "hidden_dim": 64, "batch_size": 512}
        elif model_name == "lstm":
            model_params = {"epochs": 30, "hidden_dim": 64, "seq_len": 288, "batch_size": 256}

        cfg = make_config({
            "trial_name": f"{model_name}_correct",
            "model": model_name,
            "regions": [REGION],
            "resolution": RESOLUTION,
            "horizon": HORIZON,
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "test_start": TEST_START,
            "test_end": TEST_END,
            "refit_days": 7,  # Weekly refit — daily is too slow for a demo
            "embargo": HORIZON,
            "seed": SEED,
            "transform": "asinh",
            "loss": "pinball",
            "model_params": model_params,
            "ablations": {
                "scale_target": True,
                "use_transform": True,
                "use_embargo": True,
                "leak_future": False,
            },
        })

        t0 = time.time()
        data_by_region = {REGION: data}
        results = run_trial(data_by_region, cfg, base=BASE_DIR)

        elapsed = time.time() - t0
        metrics = results[REGION]["metrics"]
        logger.info(
            "%s done in %.1fs — revenue: $%.2f, MAE: %.2f, Sharpe: %.2f",
            model_name, elapsed,
            metrics["total_revenue"], metrics["mae"], metrics["sharpe_ratio"],
        )


# ---------------------------------------------------------------------------
# Step 3: Run ablation suite for LightGBM
# ---------------------------------------------------------------------------

def run_ablations(data: pd.DataFrame) -> None:
    """Run all 6 ablation experiments for LightGBM."""
    suite = make_ablation_suite(model="lightgbm", regions=[REGION])

    for cfg in suite:
        # Override dates and params to match our setup
        cfg["resolution"] = RESOLUTION
        cfg["horizon"] = HORIZON
        cfg["train_start"] = TRAIN_START
        cfg["train_end"] = TRAIN_END
        cfg["test_start"] = TEST_START
        cfg["test_end"] = TEST_END
        cfg["refit_days"] = 7
        cfg["seed"] = SEED
        cfg["model_params"] = {"n_estimators": 200, "learning_rate": 0.05}

        # No-reconditioning ablation keeps refit_days very large
        if "no_reconditioning" in cfg["trial_name"]:
            cfg["refit_days"] = 99999

        logger.info("=" * 60)
        logger.info("Running ablation: %s", cfg["trial_name"])
        logger.info("=" * 60)

        t0 = time.time()
        data_by_region = {REGION: data}

        # Skip the correct baseline if we already ran it
        trial_dir = Path(BASE_DIR) / cfg["trial_name"] / REGION / "metrics.json"
        if trial_dir.exists():
            logger.info("Skipping %s — already exists", cfg["trial_name"])
            continue

        results = run_trial(data_by_region, cfg, base=BASE_DIR)

        elapsed = time.time() - t0
        metrics = results[REGION]["metrics"]
        logger.info(
            "%s done in %.1fs — revenue: $%.2f, MAE: %.2f",
            cfg["trial_name"], elapsed,
            metrics["total_revenue"], metrics["mae"],
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Download data, run all baselines and ablations."""
    logger.info("Starting baseline runs")

    data = load_data()
    logger.info(
        "Dataset: %d rows, %s to %s",
        len(data), data.index.min(), data.index.max(),
    )

    run_correct_baselines(data)
    run_ablations(data)

    # Print summary
    logger.info("=" * 60)
    logger.info("All runs complete. Build the dashboard with:")
    logger.info("  python scripts/build_dashboard.py  # → outputs/dashboard/index.html")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
