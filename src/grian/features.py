"""Feature engineering for NEM price forecasting.

Provides net-load calculation, the clear-sky index, calendar features,
and the full feature matrix builder that notebooks 3--10 consume.
"""

import pandas as pd


def net_load(demand: pd.Series, solar: pd.Series, wind: pd.Series) -> pd.Series:
    """Compute net load: demand minus wind and solar generation.

    Args:
        demand: Regional demand series (MW).
        solar: Solar generation series (MW).
        wind: Wind generation series (MW).

    Returns:
        Net load series (MW).
    """
    return demand - wind - solar


def clear_sky_index(
    actual_ghi: pd.Series,
    clearsky_ghi: pd.Series,
) -> pd.Series:
    """Compute the clear-sky index (actual / clear-sky irradiance).

    Values are clipped to [0, 1.5] to handle sensor noise.

    Args:
        actual_ghi: Measured global horizontal irradiance (W/m²).
        clearsky_ghi: Modelled clear-sky GHI from pvlib (W/m²).

    Returns:
        Clear-sky index series.
    """
    csi = actual_ghi / clearsky_ghi.replace(0, float("nan"))
    return csi.clip(0, 1.5)


def build_matrix(
    prices: pd.DataFrame,
    demand: pd.DataFrame,
    weather: pd.DataFrame | None = None,
    renewable_forecast: pd.DataFrame | None = None,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Assemble the feature matrix for day-ahead price forecasting.

    Includes lagged prices, calendar terms (hour, day-of-week, month),
    demand and renewable forecasts. Uses only information available at
    forecast time — no future leakage.

    Args:
        prices: Price DataFrame with ``DatetimeIndex``.
        demand: Demand DataFrame.
        weather: Optional weather features.
        renewable_forecast: Optional renewable generation forecasts.
        lags: Price lag periods to include. Defaults to sensible set.

    Returns:
        Feature matrix with ``DatetimeIndex``, ready for modelling.
    """
    if lags is None:
        lags = [48, 96, 336]

    # Identify the price column (first column or 'price')
    if "price" in prices.columns:
        price_col = "price"
    else:
        price_col = prices.columns[0]

    parts: list[pd.DataFrame] = []

    # --- Lagged price features ---
    for lag in lags:
        parts.append(
            prices[[price_col]]
            .shift(lag)
            .rename(columns={price_col: f"price_lag_{lag}"})
        )

    # --- Calendar features ---
    idx = prices.index
    cal = pd.DataFrame(
        {
            "hour": idx.hour,
            "day_of_week": idx.dayofweek,
            "month": idx.month,
        },
        index=idx,
    )
    parts.append(cal)

    # --- Demand columns ---
    if demand is not None and not demand.empty:
        parts.append(demand.reindex(idx))

    # --- Weather columns ---
    if weather is not None and not weather.empty:
        parts.append(weather.reindex(idx))

    # --- Renewable forecast columns ---
    if renewable_forecast is not None and not renewable_forecast.empty:
        parts.append(renewable_forecast.reindex(idx))

    matrix = pd.concat(parts, axis=1)

    # Drop rows with NaN introduced by lagging
    matrix = matrix.dropna()

    return matrix
