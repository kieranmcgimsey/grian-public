"""Balanced per-month capture evaluation across trials.

A single pooled annual capture ratio is dominated by 2-3 mega-spike days
(investigation report, Part III). This harness reports, for each
``<model>__<executor>`` trial under a base dir:

* per-calendar-month capture = sum(model revenue) / sum(oracle revenue),
* **balanced** capture = mean of the per-month ratios (equal weight per month),
* **pooled** capture = annual sum / sum (value-weighted, spike-dominated),

and writes a per-month heatmap so wins/losses are visible. Reads whatever
ledgers already exist (produced by run_common_eval or the test bed), so point
and quantile models are compared on identical footing.

Usage::

    python scripts/balanced_eval.py --base outputs/trials_30min \
        --trials autoregression__openloop lightgbm_qmean__mpc_spikegate ...
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from grian.viz import apply_style, save_fig

REGION = "SA1"


def oracle_daily(base: Path) -> pd.Series:
    """Daily perfect-foresight revenue for the shared common window."""
    p = base / "_oracle" / REGION / "common.parquet"
    return pd.read_parquet(p)["revenue"].resample("D").sum()


def monthly_capture(ledger: Path, odaily: pd.Series) -> pd.Series:
    """Per-calendar-month capture ratio for one trial ledger."""
    rev = pd.read_parquet(ledger)["revenue"].resample("D").sum()
    idx = rev.index.intersection(odaily.index)
    m_rev = rev.loc[idx].resample("MS").sum()
    m_orc = odaily.loc[idx].resample("MS").sum()
    return (m_rev / m_orc).rename(ledger.parent.parent.name)


def evaluate(base: Path, trials: list[str]) -> pd.DataFrame:
    """Build the per-month capture table (rows = trials) + balanced/pooled cols."""
    odaily = oracle_daily(base)
    rows = {}
    for t in trials:
        led = base / t / REGION / "ledger.parquet"
        if not led.exists():
            print(f"  [skip] no ledger for {t}")
            continue
        rows[t] = monthly_capture(led, odaily)
    df = pd.DataFrame(rows).T
    df.columns = [c.strftime("%Y-%m") for c in df.columns]
    # pooled = annual sum/sum, recomputed from ledgers
    pooled = {}
    for t in rows:
        led = pd.read_parquet(base / t / REGION / "ledger.parquet")
        rev = led["revenue"].resample("D").sum()
        idx = rev.index.intersection(odaily.index)
        pooled[t] = rev.loc[idx].sum() / odaily.loc[idx].sum()
    df["BALANCED"] = df[df.columns].mean(axis=1)
    df["pooled"] = pd.Series(pooled)
    return df.sort_values("BALANCED", ascending=False)


def heatmap(df: pd.DataFrame, out_name: str) -> None:
    """Per-month capture heatmap (months only), balanced/pooled appended as text."""
    import matplotlib.pyplot as plt
    apply_style()
    months = [c for c in df.columns if c not in ("BALANCED", "pooled")]
    M = df[months].values
    fig, ax = plt.subplots(figsize=(1.0 * len(months) + 4, 0.5 * len(df) + 1.5))
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(
        [f"{t}   [bal {df.loc[t, 'BALANCED']:.2f} | pool {df.loc[t, 'pooled']:.2f}]"
         for t in df.index], fontsize=8)
    for i in range(len(df)):
        for j in range(len(months)):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="black")
    fig.colorbar(im, ax=ax, shrink=0.6, label="capture ratio")
    ax.set_title("Per-month capture (balanced = mean of months; "
                 "pooled = spike-dominated annual)")
    save_fig(fig, out_name, "Per-month capture. Balanced weights each month "
             "equally; pooled is value-weighted.")
    plt.close(fig)


def main() -> None:
    """Print the balanced/pooled per-month capture table and write the heatmap."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="outputs/trials_30min")
    ap.add_argument("--trials", nargs="+", required=True)
    ap.add_argument("--out", default="balanced_capture")
    args = ap.parse_args()
    df = evaluate(Path(args.base), args.trials)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(df.to_string())
    heatmap(df, args.out)
    print(f"\nwrote outputs/figures/{args.out}.png")


if __name__ == "__main__":
    main()
