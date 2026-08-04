"""Forecast-quality report: CRPS and coverage from cached quantile fans.

Dispatch value answers "does the forecast make money"; this answers the
complementary question "is the forecast *honest*" — is the quantile fan
sharp (low CRPS) and calibrated (empirical coverage ≈ nominal)? It reads the
fans the test bed already cached, aligns each quantile path to the realised
prices, and reports CRPS + per-quantile coverage per (model, window).

Usage::

    python scripts/forecast_quality.py --resolution 30min
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grian.sim.analytics import crps_from_quantiles, quantile_coverage

REGION = "SA1"
OUT = Path("outputs/testbed")
_FREQ = {"5min": "5min", "30min": "30min"}


def _actuals_for(fan_long: pd.DataFrame, price: pd.Series, freq: str) -> pd.DataFrame:
    """Attach the realised price to each (origin, step) fan row."""
    df = fan_long.copy()
    df["target_time"] = pd.to_datetime(df["origin"]) + pd.to_timedelta(
        df["step"] * pd.Timedelta(freq))
    df["actual"] = df["target_time"].map(price)
    return df.dropna(subset=["actual"])


def _score_fan(path: Path, price: pd.Series, freq: str) -> dict | None:
    """CRPS + coverage for one cached fan file, or None if empty."""
    fan_long = pd.read_parquet(path)
    df = _actuals_for(fan_long, price, freq)
    if df.empty:
        return None
    preds = {float(q): g["price"].to_numpy()
             for q, g in df.groupby("quantile")}
    # Align actuals per quantile group (same ordering as groupby → same rows).
    actual_by_q = {float(q): g["actual"].to_numpy()
                   for q, g in df.groupby("quantile")}
    any_q = next(iter(actual_by_q))
    actual = actual_by_q[any_q]
    crps = crps_from_quantiles(actual, preds)
    coverage = quantile_coverage(actual, preds)
    return {"crps": round(crps, 2),
            "coverage": {str(q): round(c, 3) for q, c in coverage.items()},
            "n": int(len(actual))}


def main() -> None:
    """Score every cached fan for the given resolution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", default="30min", choices=["5min", "30min"])
    args = parser.parse_args()
    res = args.resolution
    freq = _FREQ[res]

    price = pd.read_parquet(
        f"data/processed/{REGION}_{res}_sim.parquet")["price"]
    rows = {}
    for path in sorted((OUT / "fans").glob(f"*__{res}.parquet")):
        key = path.stem.replace(f"__{res}", "")
        result = _score_fan(path, price, freq)
        if result:
            rows[key] = result
            cov = result["coverage"]
            print(f"{key:40} CRPS ${result['crps']:>8.2f}  "
                  f"coverage {cov}  (n={result['n']})")
    (OUT / f"forecast_quality_{res}.json").write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {OUT / f'forecast_quality_{res}.json'}")


if __name__ == "__main__":
    main()
