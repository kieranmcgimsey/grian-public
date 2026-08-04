"""Plot a probabilistic (quantile) price forecast — the fan the models emit.

Loads a saved quantile model (``lightgbm_qmean``), computes the full quantile fan
for a chosen forecast origin, and plots the median, the shaded inter-quantile
bands (the *uncertainty*), and the realised actual price. This is the plot the
probabilistic dispatch reasons over — where the fan is wide, a robust dispatcher
holds back; where it's tight, it trades.

Usage::

    python scripts/plot_quantile_fan.py                       # default trial/date
    python scripts/plot_quantile_fan.py --date 2026-01-15 --trial lightgbm_qmean__mpc30
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grian.sim.models import get_model  # noqa: E402
from grian.sim.trials import _get_transform_pair  # noqa: E402

DATA = "data/processed/SA1_5min_sim.parquet"
OUT = Path("outputs/figures/quantile_fan.png")


def main() -> None:
    """Compute and plot one day's quantile fan vs the actual price."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trial", default="lightgbm_qmean__mpc30")
    ap.add_argument("--date", default="2026-01-15", help="forecast origin (a day)")
    ap.add_argument("--horizon", type=int, default=288)
    args = ap.parse_args()

    data = pd.read_parquet(DATA)
    spec = get_model("lightgbm_qmean")
    state = spec["load"](f"outputs/trials/{args.trial}/SA1/model")

    origin = pd.Timestamp(args.date)
    pos = int(data.index.searchsorted(origin))
    fwd, _ = _get_transform_pair(state["transform"])
    tdata = data.copy()
    tdata["price"] = fwd(tdata["price"].values)

    fan = spec["predict_fan"](state, tdata.iloc[:pos], args.horizon)  # dollar space
    actual = data["price"].iloc[pos:pos + args.horizon].values
    qs = sorted(fan)
    hours = np.arange(len(actual)) * 5 / 60.0

    fig, ax = plt.subplots(figsize=(11, 5))
    # Shade nested inter-quantile bands from the fan (widest = most uncertain).
    lo_keys = [q for q in qs if q < 0.5]
    hi_keys = [q for q in qs if q > 0.5][::-1]
    for i, (ql, qh) in enumerate(zip(lo_keys, hi_keys)):
        ax.fill_between(hours, fan[ql][:len(actual)], fan[qh][:len(actual)],
                        color="#2563eb", alpha=0.15 + 0.12 * i,
                        label=f"q{int(ql*100):02d}–q{int(qh*100):02d}")
    med = min(qs, key=lambda q: abs(q - 0.5))
    ax.plot(hours, fan[med][:len(actual)], color="#1d4ed8", lw=1.6, label="median forecast")
    ax.plot(hours, actual, color="#111827", lw=1.4, label="actual")

    ax.set_yscale("asinh")
    ax.set_xlabel("hours ahead (day-ahead horizon)")
    ax.set_ylabel("price ($/MWh, signed-log)")
    ax.set_title(f"Quantile forecast fan — {args.trial}, origin {args.date}")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"Wrote {OUT}  (fan width q05–q98 at h+12: "
          f"${fan[qs[-1]][144] - fan[qs[0]][144]:,.0f})")


if __name__ == "__main__":
    main()
