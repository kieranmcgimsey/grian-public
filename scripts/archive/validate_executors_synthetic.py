"""Synthetic-data validation of the two executors (open-loop and MPC).

Feeds the *real* executors a controllable forecast on a synthetic price series
whose arbitrage is purely intra-day (a daily cheap window + evening spike, no
cross-day value). With a **perfect** forecast both executors should capture
~100% of the perfect-foresight oracle; capture should degrade smoothly as
forecast noise grows. This isolates the LP + dispatch + executor loop from any
real forecasting model — a correctness check, not a performance one.

Usage::

    python scripts/validate_executors_synthetic.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grian.sim import models as models_mod
from grian.sim import oracle
from grian.sim.mpc import simulate_region_mpc
from grian.sim.runner import run_trial
from grian.sim.trials import make_config

REGION = "SA1"
RESOLUTION = "5min"
DT = 1 / 12
HORIZON = 288
DISPATCH = {"power_mw": 100.0, "duration_hours": 2.0,
            "efficiency": 0.85, "max_cycles": 2}
NOISE_LEVELS = [0, 5, 20, 50, 100]   # $/MWh std of forecast error


def synthetic_prices(days: int = 40, seed: int = 0) -> pd.DataFrame:
    """A 5-min price series with purely intra-day arbitrage value."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=days * 288, freq="5min")
    hours = idx.hour + idx.minute / 60.0
    # Smooth daily swing (cheap pre-dawn, dearer midday→evening)…
    base = 60 + 40 * np.sin((hours - 14) / 24 * 2 * np.pi)
    # …plus a sharp, lucrative evening spike 18:00–19:00.
    spike = np.where((hours >= 18) & (hours < 19), 240.0, 0.0)
    # Small day-to-day jitter, but the same expected shape every day → no
    # cross-day arbitrage, so a perfect 1-day forecast can capture ~everything.
    price = base + spike + rng.normal(0, 4, len(idx))
    return pd.DataFrame({"price": price, "demand": 1000 + price}, index=idx)


def make_synth_model(full_prices: pd.Series, sigma: float, seed: int = 0) -> dict:
    """A model that returns the ACTUAL future prices plus Gaussian noise.

    With ``sigma=0`` this is a perfect forecast (deliberate look-ahead — for
    testing only). ``transform`` is identity in the config, so predict returns
    raw dollars.
    """
    rng = np.random.default_rng(seed)

    def fit(train_df, target_col, cfg):
        return {}

    def predict(state, input_df, horizon):
        pos = full_prices.index.get_indexer([input_df.index[-1]])[0]
        future = full_prices.iloc[pos + 1: pos + 1 + horizon]
        vals = future.values.astype(float)
        if sigma > 0:
            vals = vals + rng.normal(0, sigma, len(vals))
        return pd.Series(vals, index=future.index)

    def save(state, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    def load(path):
        return {}

    return {"name": "synthetic", "fit": fit, "predict": predict,
            "save": save, "load": load}


def run_one(prices: pd.DataFrame, sigma: float, executor: str, base: str) -> float:
    """Run one (executor, noise) config and return realised revenue ($)."""
    models_mod.REGISTRY["synthetic"] = make_synth_model(prices["price"], sigma)
    cfg = make_config({
        "trial_name": f"synth_{executor}_s{int(sigma)}",
        "model": "synthetic", "regions": [REGION], "resolution": RESOLUTION,
        "horizon": HORIZON, "refit_days": 7, "train_lookback_days": None,
        "embargo": 0, "seed": 42, "transform": "identity", "loss": "pinball",
        "model_params": {}, "dispatch": DISPATCH,
        "train_start": str(prices.index.min().date()),
        "train_end": str(prices.index[288 * 2].date()),
        "test_start": str(prices.index[288 * 3].date()),
        "test_end": str(prices.index[-1].date()),
        "ablations": {"scale_target": True, "use_transform": True,
                      "use_embargo": False, "leak_future": False},
        **({"mpc": {"resolve_every": 6, "reforecast_every": 6}}
           if executor == "mpc30" else {}),
    })
    sim = simulate_region_mpc if executor == "mpc30" else None
    run_trial({REGION: prices}, cfg, base=base, simulate_fn=sim)
    ledger = pd.read_parquet(Path(base) / cfg["trial_name"] / REGION / "ledger.parquet")
    return ledger, float(ledger["revenue"].sum())


def main() -> None:
    """Run the noise sweep for both executors and print a capture table."""
    prices = synthetic_prices()
    with tempfile.TemporaryDirectory() as base:
        results = {}
        oracle_rev = None
        for executor in ("openloop", "mpc30"):
            for sigma in NOISE_LEVELS:
                ledger, rev = run_one(prices, sigma, executor, base)
                if oracle_rev is None:
                    # Oracle over exactly the executed window (aligned to ledger).
                    orc = oracle.compute_oracle(
                        prices.loc[ledger.index, "price"], dt_hours=DT,
                        **{k: DISPATCH[k] for k in
                           ("power_mw", "duration_hours", "efficiency", "max_cycles")},
                    )
                    oracle_rev = orc["total_revenue"]
                results[(executor, sigma)] = rev / oracle_rev

    print(f"\nSynthetic executor validation — oracle revenue ${oracle_rev:,.0f}")
    print("Capture ratio (realised ÷ oracle):\n")
    header = "  noise $  " + "".join(f"{e:>12}" for e in ("open-loop", "MPC 30-min"))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for sigma in NOISE_LEVELS:
        ol = results[("openloop", sigma)]
        mpc = results[("mpc30", sigma)]
        tag = "  (perfect)" if sigma == 0 else ""
        print(f"  {sigma:>6}  {ol:>11.3f} {mpc:>11.3f}{tag}")
    print()


if __name__ == "__main__":
    main()
