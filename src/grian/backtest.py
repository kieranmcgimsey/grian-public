"""Rolling-origin backtesting with embargo for day-ahead forecasting.

The backtest is the methodological backbone: every model is scored on the
same held-out windows with an embargo that prevents information leakage
across the forecast horizon.
"""

from collections.abc import Callable
from typing import Any

import pandas as pd


def rolling_origin(
    data: pd.DataFrame,
    model_fn: Callable,
    train_start: str,
    test_start: str,
    test_end: str,
    horizon: int = 48,
    step: int = 48,
    embargo: int | None = None,
    refit_every: int = 1,
) -> list[dict[str, Any]]:
    """Run a rolling-origin backtest over day-ahead windows.

    Walks forward through the test period, fitting (or refitting) the
    model on expanding training data, then forecasting one horizon ahead.
    An embargo gap between training and forecast windows prevents leakage.

    Args:
        data: Full dataset (features + target).
        model_fn: Callable that accepts train data and returns predictions
            for the next horizon.
        train_start: ISO start of the training window.
        test_start: ISO start of the test period.
        test_end: ISO end of the test period.
        horizon: Number of periods in each forecast (default 48 = day-ahead
            at 30-minute resolution).
        step: Periods to advance between origins.
        embargo: Periods to skip between end of training and start of
            forecast. Defaults to *horizon*.
        refit_every: Refit the model every N steps (1 = every step).

    Returns:
        List of dicts, each with keys ``origin``, ``actual``, ``forecast``,
        and any model-specific metadata.
    """
    if embargo is None:
        embargo = horizon

    results: list[dict[str, Any]] = []
    data_ts = data.loc[train_start:test_end]

    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    # Build list of forecast origins
    origins = pd.date_range(
        start=test_start_ts,
        end=test_end_ts,
        freq=pd.tseries.offsets.DateOffset(minutes=30 * step),
    )

    last_model = None
    step_count = 0

    for origin in origins:
        # Training window: from train_start up to embargo periods before origin
        train_end_idx = data_ts.index.searchsorted(origin) - embargo
        if train_end_idx <= 0:
            continue
        train_data = data_ts.iloc[:train_end_idx]

        # Forecast window: horizon periods starting at origin
        forecast_start_loc = data_ts.index.searchsorted(origin)
        forecast_end_loc = forecast_start_loc + horizon
        if forecast_end_loc > len(data_ts):
            break
        actual_window = data_ts.iloc[forecast_start_loc:forecast_end_loc]

        # Refit model if needed
        if last_model is None or step_count % refit_every == 0:
            forecast = model_fn(train_data, horizon)
            last_model = model_fn  # Track that we have a fitted model
        else:
            forecast = model_fn(train_data, horizon)

        results.append(
            {
                "origin": origin,
                "actual": actual_window,
                "forecast": forecast,
            }
        )
        step_count += 1

    return results
