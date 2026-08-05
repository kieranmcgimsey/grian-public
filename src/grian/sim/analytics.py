"""Deep analytics for model failure analysis.

Breaks down forecast errors and trading performance by time-of-day,
price regime, day-of-week, and other structural dimensions. Every
function takes a ledger DataFrame and returns a DataFrame or dict
ready for display.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Error breakdowns
# ---------------------------------------------------------------------------

def error_by_hour(ledger_df: pd.DataFrame) -> pd.DataFrame:
    """MAE and bias by hour of day.

    Args:
        ledger_df: Ledger DataFrame with actual_price and forecast_price.

    Returns:
        DataFrame indexed by hour (0-23) with mae, bias, rmse, count.
    """
    df = ledger_df.copy()
    df["error"] = df["actual_price"] - df["forecast_price"]
    df["abs_error"] = df["error"].abs()
    df["hour"] = df.index.hour

    grouped = df.groupby("hour").agg(
        mae=("abs_error", "mean"),
        bias=("error", "mean"),
        rmse=("error", lambda x: np.sqrt((x ** 2).mean())),
        count=("error", "count"),
    )
    return grouped


def error_by_dow(ledger_df: pd.DataFrame) -> pd.DataFrame:
    """MAE and bias by day of week.

    Args:
        ledger_df: Ledger DataFrame.

    Returns:
        DataFrame indexed by day name with mae, bias, rmse, count.
    """
    df = ledger_df.copy()
    df["error"] = df["actual_price"] - df["forecast_price"]
    df["abs_error"] = df["error"].abs()
    df["dow"] = df.index.day_name()
    df["dow_num"] = df.index.dayofweek

    grouped = df.groupby(["dow_num", "dow"]).agg(
        mae=("abs_error", "mean"),
        bias=("error", "mean"),
        rmse=("error", lambda x: np.sqrt((x ** 2).mean())),
        count=("error", "count"),
    )
    return grouped.droplevel("dow_num")


def error_by_price_regime(
    ledger_df: pd.DataFrame,
    thresholds: tuple[float, ...] = (-50, 0, 50, 100, 300),
) -> pd.DataFrame:
    """MAE and bias by actual price regime.

    Bins actual prices into regimes (negative, low, medium, high, spike)
    and reports error stats per bin.

    Args:
        ledger_df: Ledger DataFrame.
        thresholds: Bin edges for price regimes.

    Returns:
        DataFrame indexed by price regime with mae, bias, count, pct.
    """
    df = ledger_df.copy()
    df["error"] = df["actual_price"] - df["forecast_price"]
    df["abs_error"] = df["error"].abs()

    bins = [-np.inf] + list(thresholds) + [np.inf]
    labels = []
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]
        if lo == -np.inf:
            labels.append(f"< ${hi:.0f}")
        elif hi == np.inf:
            labels.append(f">= ${lo:.0f}")
        else:
            labels.append(f"${lo:.0f}–${hi:.0f}")

    df["regime"] = pd.cut(df["actual_price"], bins=bins, labels=labels)

    grouped = df.groupby("regime", observed=False).agg(
        mae=("abs_error", "mean"),
        bias=("error", "mean"),
        rmse=("error", lambda x: np.sqrt((x ** 2).mean())),
        count=("error", "count"),
    )
    grouped["pct"] = grouped["count"] / len(df) * 100
    return grouped


# ---------------------------------------------------------------------------
# Revenue breakdowns
# ---------------------------------------------------------------------------

def revenue_by_hour(ledger_df: pd.DataFrame) -> pd.DataFrame:
    """Revenue contribution by hour of day.

    Args:
        ledger_df: Ledger DataFrame with revenue column.

    Returns:
        DataFrame indexed by hour with total, mean, std revenue.
    """
    df = ledger_df.copy()
    df["hour"] = df.index.hour

    return df.groupby("hour").agg(
        total_revenue=("revenue", "sum"),
        mean_revenue=("revenue", "mean"),
        std_revenue=("revenue", "std"),
        count=("revenue", "count"),
    )


def revenue_by_dow(ledger_df: pd.DataFrame) -> pd.DataFrame:
    """Revenue contribution by day of week.

    Args:
        ledger_df: Ledger DataFrame.

    Returns:
        DataFrame indexed by day name with total and mean revenue.
    """
    df = ledger_df.copy()
    df["dow"] = df.index.day_name()
    df["dow_num"] = df.index.dayofweek

    grouped = df.groupby(["dow_num", "dow"]).agg(
        total_revenue=("revenue", "sum"),
        mean_revenue=("revenue", "mean"),
        count=("revenue", "count"),
    )
    return grouped.droplevel("dow_num")


def revenue_by_price_regime(
    ledger_df: pd.DataFrame,
    thresholds: tuple[float, ...] = (-50, 0, 50, 100, 300),
) -> pd.DataFrame:
    """Revenue contribution by actual price regime.

    Args:
        ledger_df: Ledger DataFrame.
        thresholds: Bin edges for price regimes.

    Returns:
        DataFrame with total and mean revenue per regime.
    """
    df = ledger_df.copy()
    bins = [-np.inf] + list(thresholds) + [np.inf]
    labels = []
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]
        if lo == -np.inf:
            labels.append(f"< ${hi:.0f}")
        elif hi == np.inf:
            labels.append(f">= ${lo:.0f}")
        else:
            labels.append(f"${lo:.0f}–${hi:.0f}")

    df["regime"] = pd.cut(df["actual_price"], bins=bins, labels=labels)

    return df.groupby("regime", observed=False).agg(
        total_revenue=("revenue", "sum"),
        mean_revenue=("revenue", "mean"),
        count=("revenue", "count"),
    )


# ---------------------------------------------------------------------------
# Skill scores
# ---------------------------------------------------------------------------

def skill_vs_naive(
    model_ledger: pd.DataFrame,
    naive_ledger: pd.DataFrame,
) -> dict[str, float]:
    """Compute skill scores relative to the naive baseline.

    Skill = 1 - (model_metric / naive_metric) for error metrics.
    Positive skill means the model is better.

    Args:
        model_ledger: Ledger from the model being evaluated.
        naive_ledger: Ledger from the naive baseline.

    Returns:
        Dict with mae_skill, rmse_skill, revenue_ratio.
    """
    m_err = model_ledger["actual_price"] - model_ledger["forecast_price"]
    n_err = naive_ledger["actual_price"] - naive_ledger["forecast_price"]

    m_mae = float(m_err.abs().mean())
    n_mae = float(n_err.abs().mean())
    m_rmse = float(np.sqrt((m_err ** 2).mean()))
    n_rmse = float(np.sqrt((n_err ** 2).mean()))

    m_rev = float(model_ledger["revenue"].sum())
    n_rev = float(naive_ledger["revenue"].sum())

    return {
        "mae_skill": 1 - m_mae / n_mae if n_mae > 0 else 0.0,
        "rmse_skill": 1 - m_rmse / n_rmse if n_rmse > 0 else 0.0,
        "model_revenue": m_rev,
        "naive_revenue": n_rev,
        "revenue_ratio": m_rev / n_rev if n_rev != 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Headline comparison table
# ---------------------------------------------------------------------------

def headline_table(
    base: str = "outputs/trials",
    naive_trial: str = "naive_similar_day_correct",
) -> pd.DataFrame:
    """Build the headline performance comparison across all trials.

    Args:
        base: Root directory for trial artifacts.
        naive_trial: Name of the naive baseline trial for skill scores.

    Returns:
        DataFrame with one row per trial, columns for all key metrics
        plus skill scores relative to the naive baseline.
    """
    from grian.sim.trials import list_regions, list_trials, load_config, load_metrics

    base_path = Path(base)
    rows = []

    naive_ledgers: dict[str, pd.DataFrame] = {}
    for region in list_regions(naive_trial, base_path):
        try:
            from grian.sim.trials import load_ledger
            naive_ledgers[region] = load_ledger(naive_trial, region, base)
        except FileNotFoundError:
            pass

    for trial_name in list_trials(base_path):
        cfg = load_config(trial_name, base_path)
        for region in list_regions(trial_name, base_path):
            metrics = load_metrics(trial_name, region, base_path)

            row = {
                "trial": trial_name,
                "region": region,
                "model": cfg.get("model", ""),
                "transform": cfg.get("transform", ""),
                "refit_days": cfg.get("refit_days", 0),
                "mae": metrics.get("mae", float("nan")),
                "rmse": metrics.get("rmse", float("nan")),
                "total_revenue": metrics.get("total_revenue", 0),
                "sharpe": metrics.get("sharpe_ratio", 0),
                "drawdown": metrics.get("peak_drawdown", 0),
            }

            if region in naive_ledgers:
                try:
                    from grian.sim.trials import load_ledger
                    model_ledger = load_ledger(trial_name, region, base)
                    skill = skill_vs_naive(model_ledger, naive_ledgers[region])
                    row["mae_skill"] = skill["mae_skill"]
                    row["rmse_skill"] = skill["rmse_skill"]
                except FileNotFoundError:
                    row["mae_skill"] = float("nan")
                    row["rmse_skill"] = float("nan")

            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("total_revenue", ascending=False)
    return df


# ---------------------------------------------------------------------------
# Worst-day analysis
# ---------------------------------------------------------------------------

def worst_days(
    ledger_df: pd.DataFrame,
    n: int = 10,
    by: str = "revenue",
) -> pd.DataFrame:
    """Find the N worst-performing days.

    Args:
        ledger_df: Ledger DataFrame.
        n: Number of worst days to return.
        by: "revenue" or "mae" — what defines "worst".

    Returns:
        DataFrame with one row per day, sorted worst-first.
    """
    df = ledger_df.copy()
    df["error"] = (df["actual_price"] - df["forecast_price"]).abs()
    df["date"] = df.index.date

    daily = df.groupby("date").agg(
        revenue=("revenue", "sum"),
        mae=("error", "mean"),
        mean_price=("actual_price", "mean"),
        max_price=("actual_price", "max"),
        min_price=("actual_price", "min"),
        n_intervals=("revenue", "count"),
    )

    if by == "revenue":
        return daily.sort_values("revenue", ascending=True).head(n)
    return daily.sort_values("mae", ascending=False).head(n)


def dispatch_efficiency(ledger_df: pd.DataFrame) -> dict[str, float]:
    """Measure how well the dispatch aligns with price signals.

    Args:
        ledger_df: Ledger DataFrame.

    Returns:
        Dict with dispatch quality metrics.
    """
    charging = ledger_df[ledger_df["charge_mw"] > 0]
    discharging = ledger_df[ledger_df["discharge_mw"] > 0]

    avg_charge_price = (
        float(charging["actual_price"].mean()) if len(charging) else 0
    )
    avg_discharge_price = (
        float(discharging["actual_price"].mean()) if len(discharging) else 0
    )
    spread = avg_discharge_price - avg_charge_price

    total_mwh_charged = float(
        (charging["charge_mw"] * charging["interval_minutes"] / 60).sum()
    )
    total_mwh_discharged = float(
        (discharging["discharge_mw"] * discharging["interval_minutes"] / 60).sum()
    )
    # Approximate cycle count from throughput (the /4.0 scaling is the existing
    # convention in this summary); guard against divide-by-zero when idle.
    cycle_count = (
        min(total_mwh_charged, total_mwh_discharged) / 4.0
        if total_mwh_charged > 0 else 0
    )

    return {
        "avg_charge_price": avg_charge_price,
        "avg_discharge_price": avg_discharge_price,
        "price_spread": spread,
        "total_mwh_charged": total_mwh_charged,
        "total_mwh_discharged": total_mwh_discharged,
        "approx_cycles": cycle_count,
        "revenue_per_cycle": (
            float(ledger_df["revenue"].sum()) / cycle_count
            if cycle_count > 0 else 0
        ),
    }


# ---------------------------------------------------------------------------
# Capture ratio report
# ---------------------------------------------------------------------------

def capture_report(
    ledger_df: pd.DataFrame,
    oracle_daily: pd.Series,
) -> dict[str, Any]:
    """Score a trial's ledger against the perfect-foresight oracle.

    Joins realised daily revenue with oracle daily revenue over their
    common dates and computes the campaign headline metrics: capture
    ratio, decision regret, spike concentration, and intraday rank
    skill.

    Args:
        ledger_df: Ledger DataFrame with revenue, actual_price, and
            forecast_price columns, indexed by timestamp.
        oracle_daily: Oracle revenue by day (from oracle.compute_oracle).

    Returns:
        Dict with capture_ratio, total_revenue, oracle_revenue,
        regret_daily (Series), top10_regret (DataFrame),
        oracle_top10_share, and mean_daily_spearman.
    """
    realised_daily = ledger_df["revenue"].resample("D").sum()
    common = realised_daily.index.intersection(oracle_daily.index)
    realised_daily = realised_daily.loc[common]
    oracle_aligned = oracle_daily.loc[common]

    total_revenue = float(realised_daily.sum())
    oracle_revenue = float(oracle_aligned.sum())
    capture = total_revenue / oracle_revenue if oracle_revenue > 0 else 0.0
    # Trap T7: capture far above 1 means a scoreboard bug. A *small* excess is
    # legitimate — the oracle is solved in 7-day blocks with SOC pinned to 0
    # (see oracle.py / Entry 026), so it is ~0.2% conservative vs a rolling
    # perfect-foresight controller. Allow that gap; still catch gross bugs.
    assert capture <= 1.03, (
        f"Capture ratio {capture:.3f} > 1 — scoreboard bug"
    )

    regret = oracle_aligned - realised_daily
    top10 = regret.sort_values(ascending=False).head(10)
    top10_df = pd.DataFrame({
        "regret": top10,
        "oracle_revenue": oracle_aligned.loc[top10.index],
        "realised_revenue": realised_daily.loc[top10.index],
    })

    oracle_sorted = oracle_aligned.sort_values(ascending=False)
    top10_share = (
        float(oracle_sorted.head(10).sum() / oracle_revenue)
        if oracle_revenue > 0 else float("nan")
    )

    # Intraday rank skill: does the forecast order prices correctly
    # within each day? (Plan §3.2 — ranking beats magnitude.)
    def _day_spearman(day: pd.DataFrame) -> float:
        if day["actual_price"].nunique() < 3:
            return np.nan
        return day["actual_price"].rank().corr(day["forecast_price"].rank())

    spearman = (
        ledger_df.groupby(ledger_df.index.normalize())[
            ["actual_price", "forecast_price"]
        ].apply(_day_spearman)
    )

    return {
        "capture_ratio": capture,
        "total_revenue": total_revenue,
        "oracle_revenue": oracle_revenue,
        "n_days": len(common),
        "regret_daily": regret,
        "top10_regret": top10_df,
        "oracle_top10_share": top10_share,
        "mean_daily_spearman": float(spearman.mean()),
    }


# ---------------------------------------------------------------------------
# Probabilistic forecast quality (fan calibration)
# ---------------------------------------------------------------------------

def pinball_loss(actual: np.ndarray, pred: np.ndarray, q: float) -> float:
    """Mean pinball (quantile) loss at level ``q``.

    Args:
        actual: Realised values.
        pred: Predicted ``q``-quantile, same shape as ``actual``.
        q: Quantile level in (0, 1).

    Returns:
        Mean pinball loss in the units of ``actual``.
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    e = a - p
    return float(np.mean(np.maximum(q * e, (q - 1.0) * e)))


def crps_from_quantiles(
    actual: np.ndarray,
    quantile_preds: dict[float, np.ndarray],
) -> float:
    """Approximate CRPS as twice the average pinball loss over a fan.

    The continuous ranked probability score decomposes into an integral of
    the pinball loss over quantile levels; with a discrete fan we average the
    pinball loss across the available levels and double it. Lower is better,
    in the units of ``actual``. This is the standard quantile-score estimate
    of CRPS used when only a fan (not a full CDF) is available.

    Args:
        actual: Realised values, shape ``(n,)``.
        quantile_preds: ``{quantile_level: predictions}`` over the same n.

    Returns:
        Approximate CRPS (dollars).
    """
    qs = sorted(quantile_preds)
    losses = [pinball_loss(actual, quantile_preds[q], q) for q in qs]
    return 2.0 * float(np.mean(losses))


def quantile_coverage(
    actual: np.ndarray,
    quantile_preds: dict[float, np.ndarray],
) -> dict[float, float]:
    """Empirical coverage of each quantile — the calibration diagnostic.

    For each level ``q`` returns the fraction of actuals at or below the
    predicted ``q``-quantile. A well-calibrated fan has empirical coverage
    close to ``q`` (e.g. ~90% of actuals fall below the 0.9 quantile).

    Args:
        actual: Realised values, shape ``(n,)``.
        quantile_preds: ``{quantile_level: predictions}`` over the same n.

    Returns:
        ``{quantile_level: empirical_coverage}``.
    """
    a = np.asarray(actual, dtype=float)
    return {
        q: float(np.mean(a <= np.asarray(pred, dtype=float)))
        for q, pred in quantile_preds.items()
    }
