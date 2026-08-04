"""Generate the capture-ratio campaign report: figures + markdown.

Reads the cached oracles and trial ledgers under outputs/trials/,
produces the campaign figures in outputs/figures/campaign/, and writes
a self-contained findings report to outputs/reports/.

Usage:
    python scripts/build_campaign_report.py
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grian.sim.analytics import capture_report  # noqa: E402

BASE = Path("outputs/trials")
FIGDIR = Path("outputs/figures/campaign")
REPORT = Path("outputs/reports/capture_campaign_report.md")
REGION = "SA1"

# Campaign palette
C_ORACLE = "#111111"
C_CHAMP = "#c1121f"
C_BASE = "#4361ee"
C_MUTED = "#8d99ae"
C_GOOD = "#2a9d8f"
C_BAD = "#e76f51"

STYLE = {
    "figure.figsize": (11, 5),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "figure.dpi": 130,
}


def oracle_daily(window: str) -> pd.Series:
    """Oracle daily revenue for a window."""
    df = pd.read_parquet(BASE / "_oracle" / REGION / f"{window}.parquet")
    return df["revenue"].resample("D").sum()


def oracle_schedule(window: str) -> pd.DataFrame:
    """Oracle interval schedule for a window."""
    return pd.read_parquet(BASE / "_oracle" / REGION / f"{window}.parquet")


def ledger(trial: str) -> pd.DataFrame | None:
    """Load a trial ledger, or None if absent."""
    p = BASE / trial / REGION / "ledger.parquet"
    return pd.read_parquet(p) if p.exists() else None


def savefig(fig, name: str) -> str:
    """Save a figure and return its repo-relative path."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return f"../figures/campaign/{name}.png"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_capture_bars() -> str:
    """Grouped bars: capture ratio by model/executor, val vs test."""
    rows = [
        ("naive", "open-loop", 0.452, 0.347),
        ("autoregression", "open-loop", 0.473, 0.402),
        ("lightgbm_rich", "open-loop", 0.389, 0.301),
        ("lightgbm_qmean", "open-loop", 0.407, 0.443),
        ("naive", "MPC", 0.382, 0.365),
        ("lightgbm_qmean", "MPC", 0.520, None),
        ("lightgbm_rich", "MPC\n30-min reforecast", 0.546, 0.562),
    ]
    labels = [f"{m}\n{e}" for m, e, _, _ in rows]
    val = [v for *_, v, _ in rows]
    test = [t for *_, t in rows]
    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - w / 2, val, w, label="validation (Jul–Sep 23)", color=C_BASE)
    ax.bar(x + w / 2, [t if t is not None else 0 for t in test], w,
           label="test (Oct 23–Jan 24)", color=C_CHAMP)
    ax.axhline(0.50, color="k", ls="--", lw=1, alpha=0.7)
    ax.text(len(rows) - 0.5, 0.51, "0.50 target", ha="right", fontsize=9)
    for xi, t in zip(x, test):
        if t is None:
            ax.text(xi + w / 2, 0.01, "n/r", ha="center", fontsize=7,
                    rotation=90, color=C_MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("capture ratio")
    ax.set_title("Capture ratio by model and executor")
    ax.legend()
    return savefig(fig, "01_capture_bars")


def fig_equity_curves() -> str:
    """Cumulative revenue: oracle vs champion vs baselines (test)."""
    osched = oracle_schedule("test")
    oracle_cum = osched["revenue"].cumsum()
    champ = ledger("mpcx_r6_f6_p0_lightgbm_rich_test")
    naive = ledger("cap0_naive_similar_day_test")
    rich_ol = ledger("cap0_lightgbm_rich_test")

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(oracle_cum.index, oracle_cum / 1e6, color=C_ORACLE, lw=2,
            label="oracle (perfect foresight)")
    if champ is not None:
        ax.plot(champ.index, champ["revenue"].cumsum() / 1e6, color=C_CHAMP,
                lw=1.8, label="lightgbm_rich + MPC (champion)")
    if naive is not None:
        ax.plot(naive.index, naive["revenue"].cumsum() / 1e6, color=C_BASE,
                lw=1.3, label="naive open-loop")
    if rich_ol is not None:
        ax.plot(rich_ol.index, rich_ol["revenue"].cumsum() / 1e6,
                color=C_MUTED, lw=1.3, label="lightgbm_rich open-loop")
    ax.set_ylabel("cumulative revenue ($M)")
    ax.set_title("Equity curves — test window (SA1, 100 MW / 200 MWh)")
    ax.legend(loc="upper left")
    return savefig(fig, "02_equity_curves")


def fig_spike_concentration() -> str:
    """Lorenz-style: cumulative oracle revenue vs sorted days."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for window, col in [("val", C_BASE), ("test", C_CHAMP)]:
        d = oracle_daily(window).sort_values(ascending=False).values
        frac_days = np.arange(1, len(d) + 1) / len(d)
        frac_rev = np.cumsum(d) / d.sum()
        ax.plot(frac_days * 100, frac_rev * 100, color=col, lw=2,
                label=f"{window} ({len(d)} days)")
    ax.plot([0, 100], [0, 100], color=C_MUTED, ls=":", label="uniform")
    ax.axvline(10 / 122 * 100, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("top X% of days (ranked by oracle revenue)")
    ax.set_ylabel("cumulative % of oracle revenue")
    ax.set_title("Revenue is spike-concentrated")
    ax.legend(loc="lower right")
    return savefig(fig, "03_spike_concentration")


def fig_regret_decomp() -> str:
    """Champion daily regret over the test window + top-10 callout."""
    champ = ledger("mpcx_r6_f6_p0_lightgbm_rich_test")
    od = oracle_daily("test")
    rep = capture_report(champ, od)
    regret = rep["regret_daily"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1]}
    )
    ax1.bar(regret.index, regret.values / 1e3, color=C_BAD, width=1.0)
    ax1.set_ylabel("daily regret (oracle − realised, $k)")
    ax1.set_title("Where the champion loses to the oracle (test)")

    top = rep["top10_regret"].sort_values("regret", ascending=True)
    ax2.barh([d.strftime("%b %d") for d in top.index],
             top["regret"].values / 1e3, color=C_BAD)
    ax2.set_xlabel("regret ($k)")
    ax2.set_title("Top-10 regret days")
    fig.tight_layout()
    return savefig(fig, "04_regret_decomposition")


def fig_hourly_gap() -> str:
    """Revenue by hour: champion vs oracle (test)."""
    champ = ledger("mpcx_r6_f6_p0_lightgbm_rich_test")
    osched = oracle_schedule("test")
    ours = champ.groupby(champ.index.hour)["revenue"].sum() / 1e3
    orac = osched.groupby(osched.index.hour)["revenue"].sum() / 1e3
    hours = np.arange(24)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(hours - 0.2, orac.reindex(hours).values, 0.4, label="oracle",
           color=C_ORACLE)
    ax.bar(hours + 0.2, ours.reindex(hours).values, 0.4, label="champion",
           color=C_CHAMP)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("total revenue ($k)")
    ax.set_title("Revenue by hour of day — the gap is midday (test)")
    ax.set_xticks(hours)
    ax.legend()
    return savefig(fig, "05_hourly_gap")


def fig_ablation_grid() -> str:
    """Validation capture across the MPC ablation variants."""
    variants = [
        ("open-loop", 0.389, C_MUTED),
        ("60-min\nreforecast", 0.536, C_MUTED),
        ("30-min\n(champion)", 0.546, C_CHAMP),
        ("5-min\nreforecast", 0.534, C_BAD),
        ("5-min\nre-solve only", 0.545, C_GOOD),
        ("+persist\nungated", 0.435, C_BAD),
        ("+persist\ngate $300", 0.523, C_BAD),
    ]
    labels = [v[0] for v in variants]
    caps = [v[1] for v in variants]
    cols = [v[2] for v in variants]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(variants)), caps, color=cols)
    ax.axhline(0.50, color="k", ls="--", lw=1, alpha=0.7)
    for i, c in enumerate(caps):
        ax.text(i, c + 0.005, f"{c:.3f}", ha="center", fontsize=9)
    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("capture ratio (validation)")
    ax.set_ylim(0.35, 0.58)
    ax.set_title("lightgbm_rich — MPC frequency and persistence ablations")
    return savefig(fig, "06_ablation_grid")


def fig_event_day() -> str:
    """Dec 8 event-day autopsy: price, forecast, SOC, actions."""
    champ = ledger("mpcx_r6_f6_p0_lightgbm_rich_test")
    if champ is None or "2023-12-08" not in champ.index.strftime("%Y-%m-%d"):
        return ""
    day = champ.loc["2023-12-08"]
    osched = oracle_schedule("test").loc["2023-12-08"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(day.index, day["actual_price"], color=C_ORACLE, lw=1.2,
             label="actual price")
    ax1.plot(day.index, day["forecast_price"], color=C_CHAMP, lw=1.2,
             ls="--", label="forecast")
    ax1.set_yscale("symlog")
    ax1.set_ylabel("price ($/MWh, symlog)")
    ax1.set_title("2023-12-08 — the $811k regret day: forecast missed the spike")
    ax1.legend(loc="upper left")
    ax2.plot(day.index, day["soc_mwh"], color=C_GOOD, lw=1.5,
             label="champion SOC")
    ax2.plot(osched.index, osched["soc_mwh"], color=C_MUTED, lw=1.2,
             ls=":", label="oracle SOC")
    ax2.fill_between(day.index, 0, day["discharge_mw"], color=C_CHAMP,
                     alpha=0.3, label="discharge (MW)")
    ax2.set_ylabel("SOC (MWh) / discharge")
    ax2.set_xlabel("time of day")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    return savefig(fig, "07_event_day_dec08")


def fig_skill_vs_lead() -> str:
    """Forecast error vs lead time — the MPC argument, from forecasts."""
    fc_path = BASE / "cap0_lightgbm_rich_test" / REGION / "forecasts.parquet"
    if not fc_path.exists():
        return ""
    fc = pd.read_parquet(fc_path)
    fc["abs_err"] = (fc["forecast"] - fc["actual"]).abs()
    by_step = fc.groupby("step")["abs_err"].mean()
    # 288 steps at 5 min -> hours
    hours = by_step.index * 5 / 60
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(hours, by_step.values, color=C_BASE, lw=1.5)
    ax.set_xlabel("forecast lead time (hours ahead)")
    ax.set_ylabel("mean absolute error ($/MWh)")
    ax.set_title("Why MPC works: forecast skill decays with lead time\n"
                 "(open-loop uses only the 0–24h curve; MPC re-queries at "
                 "short leads hourly)")
    ax.axvspan(0, 1, color=C_GOOD, alpha=0.15)
    ax.text(0.5, ax.get_ylim()[1] * 0.9, "MPC\nfine window", ha="center",
            fontsize=8, color=C_GOOD)
    return savefig(fig, "08_skill_vs_lead")


def main():
    """Build all figures and the markdown report."""
    plt.rcParams.update(STYLE)
    figs = {
        "capture_bars": fig_capture_bars(),
        "equity": fig_equity_curves(),
        "spike": fig_spike_concentration(),
        "regret": fig_regret_decomp(),
        "hourly": fig_hourly_gap(),
        "ablation": fig_ablation_grid(),
        "event": fig_event_day(),
        "skill": fig_skill_vs_lead(),
    }
    print("Figures written:")
    for k, v in figs.items():
        print(f"  {k}: {v}")
    # Report body is authored separately (report_body.py imports these
    # paths); here we only guarantee the figures exist and are current.
    return figs


if __name__ == "__main__":
    main()
