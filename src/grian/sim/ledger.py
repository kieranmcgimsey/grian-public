"""Append-only trade ledger and P&L computation.

The ledger is a list of trade records — one per interval. Each record
captures the timestamp, actual price, forecast price, charge/discharge
actions, state of charge, and interval revenue. The runner appends
records as the simulation progresses; nothing is ever mutated.

All P&L and summary functions are pure — they take a ledger DataFrame
and return values without side effects.
"""

from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ledger creation and appending
# ---------------------------------------------------------------------------

def empty_ledger() -> list[dict[str, Any]]:
    """Create an empty ledger (a plain list).

    Returns:
        Empty list ready to receive trade records via append_record.
    """
    return []


def append_record(
    ledger: list[dict],
    *,
    timestamp: pd.Timestamp,
    actual_price: float,
    forecast_price: float,
    charge_mw: float,
    discharge_mw: float,
    soc_mwh: float,
    interval_minutes: float = 5.0,
) -> None:
    """Append one interval's trade record to the ledger.

    Revenue is computed as (discharge - charge) * price * dt, where
    dt is the interval length in hours.

    Args:
        ledger: The list to append to (mutated in place).
        timestamp: Interval start timestamp.
        actual_price: Realised spot price ($/MWh).
        forecast_price: Model's forecast price for this interval.
        charge_mw: Charging power (MW), non-negative.
        discharge_mw: Discharging power (MW), non-negative.
        soc_mwh: State of charge after this interval (MWh).
        interval_minutes: Length of the interval in minutes.
    """
    dt_hours = interval_minutes / 60.0
    net_mw = discharge_mw - charge_mw
    revenue = actual_price * net_mw * dt_hours

    ledger.append({
        "timestamp": timestamp,
        "actual_price": actual_price,
        "forecast_price": forecast_price,
        "charge_mw": charge_mw,
        "discharge_mw": discharge_mw,
        "net_mw": net_mw,
        "soc_mwh": soc_mwh,
        "revenue": revenue,
        "interval_minutes": interval_minutes,
    })


# ---------------------------------------------------------------------------
# Convert to DataFrame
# ---------------------------------------------------------------------------

def to_dataframe(ledger: list[dict]) -> pd.DataFrame:
    """Convert a ledger list to a DataFrame indexed by timestamp.

    Args:
        ledger: List of trade record dicts.

    Returns:
        DataFrame with one row per interval, indexed by timestamp.
    """
    if not ledger:
        return pd.DataFrame(columns=[
            "actual_price", "forecast_price", "charge_mw", "discharge_mw",
            "net_mw", "soc_mwh", "revenue", "interval_minutes",
        ])
    df = pd.DataFrame(ledger)
    df = df.set_index("timestamp").sort_index()
    return df


# ---------------------------------------------------------------------------
# P&L computation — all pure functions on DataFrames
# ---------------------------------------------------------------------------

def cumulative_pnl(ledger_df: pd.DataFrame) -> pd.Series:
    """Compute the cumulative P&L (equity curve) from a ledger.

    Args:
        ledger_df: Ledger DataFrame with a "revenue" column.

    Returns:
        Series of cumulative revenue indexed by timestamp.
    """
    return ledger_df["revenue"].cumsum()


def daily_pnl(ledger_df: pd.DataFrame) -> pd.Series:
    """Compute daily P&L from a ledger.

    Args:
        ledger_df: Ledger DataFrame with a "revenue" column.

    Returns:
        Series of daily revenue indexed by date.
    """
    return ledger_df["revenue"].resample("D").sum()


def total_revenue(ledger_df: pd.DataFrame) -> float:
    """Total revenue across the entire ledger.

    Args:
        ledger_df: Ledger DataFrame with a "revenue" column.

    Returns:
        Sum of all interval revenues ($).
    """
    return float(ledger_df["revenue"].sum())


def peak_drawdown(ledger_df: pd.DataFrame) -> float:
    """Maximum drawdown of the equity curve.

    Args:
        ledger_df: Ledger DataFrame with a "revenue" column.

    Returns:
        Peak drawdown as a positive dollar amount.
    """
    equity = cumulative_pnl(ledger_df)
    running_max = equity.cummax()
    drawdown = running_max - equity
    return float(drawdown.max()) if len(drawdown) > 0 else 0.0


def sharpe_ratio(ledger_df: pd.DataFrame, periods_per_year: int = 365) -> float:
    """Annualised Sharpe ratio of daily returns.

    Uses daily P&L (not percentage returns, since we're tracking
    dollar revenue, not portfolio value changes).

    Args:
        ledger_df: Ledger DataFrame with a "revenue" column.
        periods_per_year: Number of trading days per year.

    Returns:
        Annualised Sharpe ratio, or 0.0 if insufficient data.
    """
    daily = daily_pnl(ledger_df)
    if len(daily) < 2 or daily.std() == 0:
        return 0.0
    return float(daily.mean() / daily.std() * np.sqrt(periods_per_year))


def forecast_accuracy(ledger_df: pd.DataFrame) -> dict[str, float]:
    """Compute forecast accuracy metrics from the ledger.

    Args:
        ledger_df: Ledger DataFrame with actual_price and forecast_price.

    Returns:
        Dict with mae, rmse, and mape.
    """
    actual = ledger_df["actual_price"].values
    forecast = ledger_df["forecast_price"].values
    errors = actual - forecast

    result = {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }

    # MAPE — skip zero-price intervals to avoid division by zero
    nonzero = actual != 0
    if nonzero.any():
        result["mape"] = float(np.mean(np.abs(errors[nonzero] / actual[nonzero])))
    else:
        result["mape"] = float("nan")

    return result


def summarise(ledger_df: pd.DataFrame) -> dict[str, Any]:
    """Produce a full summary of a trial's trading performance.

    Args:
        ledger_df: Ledger DataFrame.

    Returns:
        Dict with total_revenue, daily_mean, daily_std, sharpe,
        peak_drawdown, n_intervals, and forecast accuracy metrics.
    """
    daily = daily_pnl(ledger_df)
    accuracy = forecast_accuracy(ledger_df)

    return {
        "total_revenue": total_revenue(ledger_df),
        "daily_mean_revenue": float(daily.mean()) if len(daily) > 0 else 0.0,
        "daily_std_revenue": float(daily.std()) if len(daily) > 1 else 0.0,
        "sharpe_ratio": sharpe_ratio(ledger_df),
        "peak_drawdown": peak_drawdown(ledger_df),
        "n_intervals": len(ledger_df),
        "n_days": len(daily),
        **accuracy,
    }
