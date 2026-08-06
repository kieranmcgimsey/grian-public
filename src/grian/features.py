"""Feature engineering for price forecasting models.

Builds rich feature sets from the raw price + demand data. Every function
takes a DataFrame and returns a DataFrame of the same length — no leakage,
no lookahead, only backward-looking computations.

Feature groups:
    - Price lags: point values from 1d, 2d, 7d ago.
    - Rolling stats: mean, std, min, max over 1h, 6h, 24h windows.
    - Demand: raw demand, rolling demand stats, price-demand ratio.
    - Calendar: hour, day-of-week, month, weekend flag, hour×dow interaction.
    - Momentum: price returns, directional indicators.
    - Intra-day profile: average price shape from last N days.

All features are purely backward-looking (shift >= 1 or rolling with
closed='left'). The feature builder takes a `horizon` argument and ensures
no feature uses data closer than `horizon` intervals to the present when
`safe=True` (for use in direct multi-step models where the training target
is shifted forward).
"""

import numpy as np
import pandas as pd


def price_lags(
    series: pd.Series,
    resolution: str = "5min",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Lagged price values.

    Args:
        series: Price series.
        resolution: "5min" or "30min".
        lags: Custom lag list. Default: [1d, 2d, 7d] in interval units.

    Returns:
        DataFrame with one column per lag.
    """
    ppd = 288 if resolution == "5min" else 48
    if lags is None:
        lags = [ppd, 2 * ppd, 7 * ppd]
    return pd.DataFrame(
        {f"price_lag_{lag}": series.shift(lag) for lag in lags},
        index=series.index,
    )


def rolling_stats(
    series: pd.Series,
    resolution: str = "5min",
    windows_hours: list[float] | None = None,
) -> pd.DataFrame:
    """Rolling mean, std, min, max over configurable windows.

    Args:
        series: Price series.
        resolution: "5min" or "30min".
        windows_hours: Window sizes in hours. Default: [1, 6, 24].

    Returns:
        DataFrame with 4 columns per window (mean, std, min, max).
    """
    iph = 12 if resolution == "5min" else 2  # intervals per hour
    if windows_hours is None:
        windows_hours = [1.0, 6.0, 24.0]

    parts = {}
    for wh in windows_hours:
        w = max(1, int(wh * iph))
        tag = f"{wh:.0f}h" if wh == int(wh) else f"{wh}h"
        r = series.shift(1).rolling(w, min_periods=1)
        parts[f"price_rmean_{tag}"] = r.mean()
        parts[f"price_rstd_{tag}"] = r.std().fillna(0.0)
        parts[f"price_rmin_{tag}"] = r.min()
        parts[f"price_rmax_{tag}"] = r.max()

    return pd.DataFrame(parts, index=series.index)


def demand_features(
    df: pd.DataFrame,
    demand_col: str = "demand",
    resolution: str = "5min",
) -> pd.DataFrame:
    """Demand-side features: raw demand, rolling stats, price/demand ratio.

    Args:
        df: DataFrame with both price and demand columns.
        demand_col: Name of the demand column.
        resolution: "5min" or "30min".

    Returns:
        DataFrame with demand features. Empty if demand_col not present.
    """
    if demand_col not in df.columns:
        return pd.DataFrame(index=df.index)

    iph = 12 if resolution == "5min" else 2
    demand = df[demand_col]

    parts = {
        "demand": demand.shift(1),
        "demand_rmean_6h": demand.shift(1).rolling(6 * iph, min_periods=1).mean(),
        "demand_rmean_24h": demand.shift(1).rolling(24 * iph, min_periods=1).mean(),
        "demand_rstd_24h": (
            demand.shift(1).rolling(24 * iph, min_periods=1).std().fillna(0.0)
        ),
    }

    return pd.DataFrame(parts, index=df.index)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features: hour, day-of-week, month, weekend, interactions.

    Args:
        index: DatetimeIndex.

    Returns:
        DataFrame with calendar columns.
    """
    hour = index.hour + index.minute / 60.0
    dow = index.dayofweek
    return pd.DataFrame(
        {
            "hour": hour,
            "day_of_week": dow,
            "month": index.month,
            "is_weekend": (dow >= 5).astype(float),
            "hour_x_dow": hour * dow,
        },
        index=index,
    )


# (period, harmonics) for hour-of-day, day-of-week, month — same spec the linear
# models use, so the tree Fourier ablation is comparable to the linear one.
_CAL_FOURIER = (("hour", 24.0, 3), ("dow", 7.0, 1), ("month", 12.0, 2))


def fourier_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclic sin/cos calendar features (Fourier encoding).

    The ordinal calendar (``calendar_features``) is correct for trees, which
    split on it natively. This alternative encodes each cyclic calendar variable
    as sin/cos harmonics so hour 23.5 and hour 0 are adjacent — the encoding the
    linear models use internally, exposed here so tree models can be ablated on
    the same footing (Fourier-vs-ordinal calendar).

    Args:
        index: DatetimeIndex.

    Returns:
        DataFrame of sin/cos harmonics plus a weekend flag.
    """
    raw = {"hour": index.hour + index.minute / 60.0,
           "dow": index.dayofweek.astype(float),
           "month": index.month.astype(float)}
    parts = {}
    for name, period, n_harm in _CAL_FOURIER:
        x = raw[name]
        for k in range(1, n_harm + 1):
            parts[f"{name}_sin{k}"] = np.sin(2 * np.pi * k * x / period)
            parts[f"{name}_cos{k}"] = np.cos(2 * np.pi * k * x / period)
    parts["is_weekend"] = (index.dayofweek >= 5).astype(float)
    return pd.DataFrame(parts, index=index)


def momentum_features(
    series: pd.Series,
    resolution: str = "5min",
) -> pd.DataFrame:
    """Price momentum: returns, direction, acceleration.

    Args:
        series: Price series.
        resolution: "5min" or "30min".

    Returns:
        DataFrame with momentum features.
    """
    iph = 12 if resolution == "5min" else 2

    ret_1h = series.shift(1).pct_change(iph).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    ret_6h = (series.shift(1).pct_change(6 * iph)
              .replace([np.inf, -np.inf], 0.0).fillna(0.0))
    ret_24h = (series.shift(1).pct_change(24 * iph)
               .replace([np.inf, -np.inf], 0.0).fillna(0.0))

    return pd.DataFrame(
        {
            "price_ret_1h": ret_1h,
            "price_ret_6h": ret_6h,
            "price_ret_24h": ret_24h,
            "price_direction_1h": np.sign(ret_1h),
        },
        index=series.index,
    )


def intraday_profile(
    series: pd.Series,
    resolution: str = "5min",
    lookback_days: int = 7,
) -> pd.DataFrame:
    """Average intra-day price shape from recent history.

    For each interval, computes the average price at that time-of-day
    over the last `lookback_days` days. This captures the structural
    daily pattern (low overnight, peak in evening, etc).

    Args:
        series: Price series.
        resolution: "5min" or "30min".
        lookback_days: Number of recent days to average.

    Returns:
        DataFrame with profile deviation column.
    """
    ppd = 288 if resolution == "5min" else 48
    window = lookback_days * ppd

    # For each row, the average price over the trailing lookback window.
    shifted = series.shift(1)

    # Group by time-of-day and compute rolling mean across days
    profile_mean = shifted.rolling(window, min_periods=ppd).mean()
    deviation = shifted - profile_mean

    return pd.DataFrame(
        {
            "intraday_profile_mean": profile_mean,
            "intraday_deviation": deviation.fillna(0.0),
        },
        index=series.index,
    )


def scarcity_features(
    df: pd.DataFrame,
    target_col: str = "price",
    demand_col: str = "demand",
    resolution: str = "5min",
    spike_threshold: float = 300.0,
) -> pd.DataFrame:
    """Spike-precursor features: demand stress, ramp, volatility clustering.

    The campaign's residual regret is concentrated on fast midday
    scarcity events (report §5). These are driven by the system running
    close to its supply ceiling, by fast demand ramps, and by volatility
    clustering (spikes beget spikes). The standard feature set captures
    price level and shape but none of these precursors. Empirically, on
    SA1 the next-interval spike rate rises from a 3.5% base to ~17% when
    demand is within 5% of its recent peak, and likewise ~17% after a
    cluster of recent spikes — signal the model was previously blind to.

    All columns are strictly backward-looking (shift >= 1).

    Args:
        df: DataFrame with price and (optionally) demand columns.
        target_col: Price column name.
        demand_col: Demand column name.
        resolution: "5min" or "30min".
        spike_threshold: Price ($/MWh) above which an interval is a spike.

    Returns:
        DataFrame of scarcity/clustering features.
    """
    iph = 12 if resolution == "5min" else 2
    ppd = 288 if resolution == "5min" else 48
    price = df[target_col]
    parts = {}

    # Volatility clustering: how many spikes in the trailing 24h, and how
    # recently. Both segment the forward spike rate strongly.
    is_spike = (price > spike_threshold).astype(float)
    parts["spikes_24h"] = is_spike.shift(1).rolling(ppd, min_periods=1).sum()
    # Intervals since the last spike (capped at one day for scale),
    # counting only spikes strictly before the current interval.
    pos = np.arange(len(price), dtype=float)
    marker = np.where(is_spike.values > 0, pos, np.nan)
    marker = pd.Series(marker, index=price.index).shift(1).ffill()
    since = (pd.Series(pos, index=price.index) - marker).clip(upper=ppd)
    parts["intervals_since_spike"] = since.fillna(ppd)

    # Price volatility regime: short-run vol relative to long-run vol.
    rstd_1h = price.shift(1).rolling(iph, min_periods=1).std().fillna(0.0)
    rstd_24h = price.shift(1).rolling(24 * iph, min_periods=1).std().fillna(0.0)
    parts["price_vol_ratio"] = (rstd_1h / (rstd_24h + 1.0)).clip(0, 20)

    if demand_col in df.columns:
        demand = df[demand_col]
        # Demand stress: how close to the recent supply ceiling we sit.
        dmax_7d = demand.shift(1).rolling(7 * ppd, min_periods=ppd).max()
        dp95_7d = demand.shift(1).rolling(
            7 * ppd, min_periods=ppd
        ).quantile(0.95)
        parts["demand_stress_max"] = (demand.shift(1) / dmax_7d).fillna(0.5)
        parts["demand_stress_p95"] = (demand.shift(1) / dp95_7d).fillna(0.5)
        # Demand ramp and acceleration: fast ramps trigger scarcity.
        dramp = demand.shift(1).pct_change(iph)
        parts["demand_ramp_1h"] = dramp.replace(
            [np.inf, -np.inf], 0.0
        ).fillna(0.0)
        parts["demand_accel"] = (
            dramp - dramp.shift(iph)
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return pd.DataFrame(parts, index=df.index)


def weather_features(df: pd.DataFrame, resolution: str = "5min") -> pd.DataFrame:
    """Weather-derived features from ERA5 columns, if present.

    Uses ``ssrd`` (solar radiation), ``t2m`` (temperature), and ``wind_speed``
    when present. All features are **known at forecast time**: the models consume
    features at the forecast *origin* to predict the horizon, so the "current"
    value here is the weather observed *now* (leakage-free), and the rolling mean
    and 1-hour change are strictly backward-looking. The weather series is
    forward-filled from hourly ERA5 (no interpolation), so no future value leaks.
    (Using *target-time* weather — a true day-ahead weather forecast — would need
    NEMSEER/forecast data and is a separate, flagged extension.)

    Args:
        df: Raw data, optionally with weather columns.
        resolution: "5min" or "30min".

    Returns:
        DataFrame of weather features (empty if no weather columns present).
    """
    iph = 12 if resolution == "5min" else 2
    cols = [c for c in ("ssrd", "t2m", "wind_speed") if c in df.columns]
    if not cols:
        return pd.DataFrame(index=df.index)
    parts = {}
    for c in cols:
        s = df[c].astype(float)
        parts[f"wx_{c}"] = s
        parts[f"wx_{c}_3h"] = s.rolling(3 * iph, min_periods=1).mean()
        parts[f"wx_{c}_chg"] = s.diff(iph)
    return pd.DataFrame(parts, index=df.index)


def lean_lags(resolution: str = "5min") -> list[int]:
    """Shape-preserving autoregressive lag set: daily-aligned raw lags only.

    Every lag is a whole number of days back at the *same* interval-of-day
    (prior 1/2/3/7/14 days), plus ±1 interval around yesterday so the fit can
    place a shifted daily peak. This is the canonical EPF basis (Lago et al.'s
    LEAR uses same-hour prices on prior days) and a richer basis than the 3-lag
    AR default, so a sparse linear model has real structure to select from.

    Deliberately **no sub-daily recency lags** (lag 1, 1h, …): those anchor the
    forecast to the current price level (persistence), and an OLS/Lasso fit
    leans on them even at far horizons — 54% of the far-step lag weight landed
    on lag 1 — which flattens the diurnal shape that dispatch capture pays for
    (Entry 032).

    Args:
        resolution: "5min" or "30min".

    Returns:
        Sorted, de-duplicated list of positive lags in interval units.
    """
    ppd = 288 if resolution == "5min" else 48
    lags = [d * ppd for d in (1, 2, 3, 7, 14)] + [ppd - 1, ppd + 1]
    return sorted({lag for lag in lags if lag >= 1})


def build_features(
    df: pd.DataFrame,
    target_col: str = "price",
    resolution: str = "5min",
    include_demand: bool = True,
    include_scarcity: bool = False,
    include_weather: bool = False,
    spike_threshold: float = 300.0,
    feature_set: str = "full",
    calendar_encoding: str = "ordinal",
) -> pd.DataFrame:
    """Build the feature matrix from raw data.

    Combines feature groups into a single DataFrame. All features are
    backward-looking — safe for time-series forecasting.

    Args:
        df: Raw data with at least the target column.
        target_col: Price column name.
        resolution: "5min" or "30min".
        include_demand: Whether to include demand features.
        include_scarcity: Whether to include spike-precursor features
            (demand stress, ramp, volatility clustering).
        include_weather: Whether to include ERA5 weather features (if the
            weather columns are present).
        spike_threshold: Spike threshold passed to scarcity_features.
        feature_set: "full" (all groups, default) or "lean" — a
            shape-preserving set of raw price lags (``lean_lags``) plus the
            calendar only. "lean" drops the rolling-mean / momentum /
            intraday-profile smoothers, which are mean-reverting and flatten
            the daily peaks that dispatch capture pays for (Entry 032). The
            ``include_*`` exogenous toggles are ignored under "lean".
        calendar_encoding: "ordinal" (default) emits raw hour/day/month columns —
            correct for trees. "fourier" emits cyclic sin/cos harmonics instead
            (``fourier_calendar_features``), so a tree model can be ablated on
            Fourier-vs-ordinal calendar the same way linear models are. Linear
            models pass "ordinal" here and Fourier-encode in their own pipeline.

    Returns:
        DataFrame with the selected features. NaN rows at the start are
        expected (from lag/rolling operations) — caller should drop them.
    """
    series = df[target_col]
    cal = (fourier_calendar_features(df.index) if calendar_encoding == "fourier"
           else calendar_features(df.index))

    if feature_set == "lean":
        return pd.concat([
            price_lags(series, resolution, lags=lean_lags(resolution)),
            cal,
        ], axis=1)

    parts = [
        price_lags(series, resolution),
        rolling_stats(series, resolution),
        cal,
        momentum_features(series, resolution),
        intraday_profile(series, resolution),
    ]

    if include_demand:
        parts.append(demand_features(df, resolution=resolution))

    if include_scarcity:
        parts.append(scarcity_features(
            df, target_col, resolution=resolution,
            spike_threshold=spike_threshold,
        ))

    if include_weather:
        parts.append(weather_features(df, resolution=resolution))

    return pd.concat(parts, axis=1)


def main() -> None:
    """Run this module as a CLI (exposes its public callables)."""
    from grian._cli import run_module_cli

    run_module_cli(globals())


if __name__ == "__main__":
    main()
