"""Baseline forecasters: similar-day naive and autoregression.

These set the floor that every learned model must beat.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def similar_day_naive(
    prices: pd.DataFrame,
    origin: pd.Timestamp,
    horizon: int = 48,
) -> pd.Series:
    """Forecast by repeating the same day-of-week from the previous week.

    Args:
        prices: Historical price DataFrame with ``DatetimeIndex``.
        origin: Forecast origin (the last observation before the forecast).
        horizon: Number of periods to forecast.

    Returns:
        Forecast series of length *horizon*.
    """
    # Identify price column
    if "price" in prices.columns:
        price_col = "price"
    else:
        price_col = prices.columns[0]

    # Get the block of prices from same time-of-day one week ago
    hist = prices.loc[:origin, price_col]
    week_ago_start = origin - pd.Timedelta(days=7)
    week_ago_block = hist.loc[week_ago_start:]

    # Take the first `horizon` values starting after week_ago_start
    # (these correspond to same time-of-day, same day-of-week)
    forecast_values = week_ago_block.iloc[:horizon].values

    # Build forecast index: horizon periods starting at origin
    freq = pd.infer_freq(prices.index[:5])
    if freq is None:
        freq = "30min"
    forecast_index = pd.date_range(
        start=origin, periods=horizon, freq=freq,
    )

    return pd.Series(forecast_values, index=forecast_index, name="forecast")


def autoregression(
    prices: pd.DataFrame,
    origin: pd.Timestamp,
    horizon: int = 48,
    lags: int = 168,
) -> pd.Series:
    """Simple autoregressive forecast using recent price lags.

    Args:
        prices: Historical price DataFrame.
        origin: Forecast origin timestamp.
        horizon: Number of periods to forecast.
        lags: Number of lag terms to include.

    Returns:
        Forecast series of length *horizon*.
    """
    # Identify price column
    if "price" in prices.columns:
        price_col = "price"
    else:
        price_col = prices.columns[0]

    series = prices.loc[:origin, price_col].dropna()

    # Build lagged feature matrix for training
    # Each row: [y_{t-1}, y_{t-2}, ..., y_{t-lags}] -> y_t
    values = series.values
    n = len(values)
    if n <= lags:
        # Not enough data; fall back to repeating last value
        freq = pd.infer_freq(prices.index[:5]) or "30min"
        forecast_index = pd.date_range(start=origin, periods=horizon, freq=freq)
        return pd.Series(
            np.full(horizon, values[-1] if n > 0 else 0.0),
            index=forecast_index,
            name="forecast",
        )

    # Construct training data
    X_train = np.column_stack(
        [values[lags - i - 1 : n - i - 1] for i in range(lags)]
    )
    y_train = values[lags:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Iterative forecasting: predict one step, append, repeat
    recent = list(values[-lags:])
    forecasts = []
    for _ in range(horizon):
        x = np.array(recent[-lags:]).reshape(1, -1)
        pred = model.predict(x)[0]
        forecasts.append(pred)
        recent.append(pred)

    freq = pd.infer_freq(prices.index[:5]) or "30min"
    forecast_index = pd.date_range(start=origin, periods=horizon, freq=freq)
    return pd.Series(forecasts, index=forecast_index, name="forecast")
