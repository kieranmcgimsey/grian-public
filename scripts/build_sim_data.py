"""Rebuild the 5-minute sim dataset (price + demand) over an extended range.

Pulls SA1 price and demand via NEMOSIS for [START, END], caching the monthly
archives (existing months are reused, not re-downloaded), joins them, forward-
fills gaps up to one hour, drops residual NaNs, and writes
``data/processed/SA1_5min_sim.parquet``.

Usage::

    python scripts/build_sim_data.py                 # defaults below
    python scripts/build_sim_data.py 2023-01-01 2026-05-31
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grian.data import load_demand, load_prices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_sim_data")

REGION = "SA1"
RESOLUTION = "5min"
CACHE = "data/raw/nemosis"
OUT = Path("data/processed") / f"{REGION}_5min_sim.parquet"

DEFAULT_START = "2023-01-01"
DEFAULT_END = "2026-06-30"


def main() -> None:
    """Pull price + demand over the range and rebuild the sim parquet."""
    args = sys.argv[1:]
    start = args[0] if len(args) > 0 else DEFAULT_START
    end = args[1] if len(args) > 1 else DEFAULT_END

    logger.info("Building %s over [%s, %s]", REGION, start, end)
    prices = load_prices(REGION, start, end, cache=CACHE, resolution=RESOLUTION)
    logger.info("prices: %d rows, %s", len(prices), prices.index.max())
    demand = load_demand(REGION, start, end, cache=CACHE)
    logger.info("demand: %d rows, %s", len(demand), demand.index.max())

    df = prices.join(demand, how="outer").ffill(limit=12)
    n_before = len(df)
    df = df.dropna()
    logger.info("joined: %d rows (dropped %d NaN)", len(df), n_before - len(df))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)
    logger.info("Wrote %s: %d rows → %s", OUT, len(df), df.index.max())


if __name__ == "__main__":
    main()
