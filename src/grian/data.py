"""Data loading and alignment for NEM prices, demand, and weather.

All AEMO timestamps are interval-ending. This module shifts them back by
one interval on load so that every timestamp marks the *start* of the
interval, matching ERA5 and other interval-beginning conventions. This
decision is applied once here and nowhere else.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_5MIN = pd.Timedelta(minutes=5)
_30MIN = pd.Timedelta(minutes=30)


def _shift_to_interval_start(df: pd.DataFrame, freq: pd.Timedelta) -> pd.DataFrame:
    """Shift AEMO interval-ending timestamps to interval-start."""
    df = df.copy()
    df.index = df.index - freq
    return df


def _drop_intervention_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only non-intervention rows; where all are interventions, keep them."""
    if "INTERVENTION" not in df.columns:
        return df
    non_interv = df[df["INTERVENTION"] == 0]
    if len(non_interv) == 0:
        return df.drop(columns=["INTERVENTION"])
    return non_interv.drop(columns=["INTERVENTION"])


def load_prices(
    region: str,
    start: str,
    end: str,
    cache: str | Path = "data/raw/nemosis",
    resolution: str = "5min",
) -> pd.DataFrame:
    """Load dispatch or trading prices from NEMOSIS.

    Args:
        region: NEM region identifier (e.g. ``'SA1'``).
        start: ISO start date string.
        end: ISO end date string.
        cache: Directory for the NEMOSIS file cache.
        resolution: ``'5min'`` for DISPATCHPRICE, ``'30min'`` for TRADINGPRICE.

    Returns:
        DataFrame with a ``DatetimeIndex`` (interval-start) and column ``price``.
    """
    from nemosis import dynamic_data_compiler

    table = "DISPATCHPRICE" if resolution == "5min" else "TRADINGPRICE"
    freq = _5MIN if resolution == "5min" else _30MIN

    start_fmt = pd.Timestamp(start).strftime("%Y/%m/%d %H:%M:%S")
    end_fmt = pd.Timestamp(end).strftime("%Y/%m/%d %H:%M:%S")

    cols = ["SETTLEMENTDATE", "REGIONID", "RRP"]
    if table == "DISPATCHPRICE":
        cols.append("INTERVENTION")

    cache = str(Path(cache))
    Path(cache).mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s for %s [%s, %s]", table, region, start, end)
    df = dynamic_data_compiler(
        start_fmt,
        end_fmt,
        table,
        cache,
        select_columns=cols,
        filter_cols=["REGIONID"],
        filter_values=[(region,)],
    )

    if "INTERVENTION" in df.columns:
        df = _drop_intervention_duplicates(df)

    df = df.set_index("SETTLEMENTDATE").sort_index()
    df = df.drop(columns=["REGIONID"], errors="ignore")
    df = df.rename(columns={"RRP": "price"})
    df = _shift_to_interval_start(df, freq)
    df.index.name = "timestamp"
    return df


def load_demand(
    region: str,
    start: str,
    end: str,
    cache: str | Path = "data/raw/nemosis",
) -> pd.DataFrame:
    """Load 5-minute regional demand from NEMOSIS.

    Args:
        region: NEM region identifier.
        start: ISO start date string.
        end: ISO end date string.
        cache: Directory for the NEMOSIS file cache.

    Returns:
        DataFrame with a ``DatetimeIndex`` (interval-start) and column ``demand``.
    """
    from nemosis import dynamic_data_compiler

    start_fmt = pd.Timestamp(start).strftime("%Y/%m/%d %H:%M:%S")
    end_fmt = pd.Timestamp(end).strftime("%Y/%m/%d %H:%M:%S")

    cache = str(Path(cache))
    Path(cache).mkdir(parents=True, exist_ok=True)

    logger.info("Downloading DISPATCHREGIONSUM for %s [%s, %s]", region, start, end)
    df = dynamic_data_compiler(
        start_fmt,
        end_fmt,
        "DISPATCHREGIONSUM",
        cache,
        select_columns=[
            "SETTLEMENTDATE",
            "REGIONID",
            "INTERVENTION",
            "TOTALDEMAND",
        ],
        filter_cols=["REGIONID"],
        filter_values=[(region,)],
    )

    df = _drop_intervention_duplicates(df)
    df = df.set_index("SETTLEMENTDATE").sort_index()
    df = df.drop(columns=["REGIONID"], errors="ignore")
    df = df.rename(columns={"TOTALDEMAND": "demand"})
    df = _shift_to_interval_start(df, _5MIN)
    df.index.name = "timestamp"
    return df


def load_era5(
    region: str,
    start: str,
    end: str,
    cache: str | Path = "data/raw/era5",
) -> pd.DataFrame:
    """Load ERA5 weather reanalysis (irradiance, temperature, wind).

    Requires a Copernicus CDS account with an API key configured in
    ``~/.cdsapirc``. If the key is missing, raises ``FileNotFoundError``
    with setup instructions.

    Args:
        region: NEM region identifier (used to select lat/lon bounds).
        start: ISO start date string.
        end: ISO end date string.
        cache: Directory for cached NetCDF files.

    Returns:
        DataFrame with hourly ``DatetimeIndex`` and columns
        ``ssrd`` (surface solar radiation), ``t2m`` (2m temperature),
        ``u100``/``v100`` (100m wind components).
    """
    import xarray as xr

    region_bounds = {
        "SA1": {"lat": (-38.0, -26.0), "lon": (129.0, 141.0)},
        "NSW1": {"lat": (-37.5, -28.0), "lon": (141.0, 154.0)},
        "VIC1": {"lat": (-39.5, -34.0), "lon": (141.0, 150.0)},
        "QLD1": {"lat": (-29.0, -10.0), "lon": (138.0, 154.0)},
        "TAS1": {"lat": (-44.0, -40.0), "lon": (144.0, 149.0)},
    }

    if region not in region_bounds:
        raise ValueError(
            f"Unknown region {region!r}. "
            f"Expected one of {list(region_bounds)}"
        )

    bounds = region_bounds[region]
    cache_dir = Path(cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    nc_path = cache_dir / f"{region}_{start_ts:%Y%m%d}_{end_ts:%Y%m%d}.nc"

    if not nc_path.exists():
        cdsapi_rc = Path.home() / ".cdsapirc"
        if not cdsapi_rc.exists():
            raise FileNotFoundError(
                "ERA5 download requires a CDS API key. Create an account at "
                "https://cds.climate.copernicus.eu and save your key to ~/.cdsapirc:\n"
                "  url: https://cds.climate.copernicus.eu/api/v2\n"
                "  key: <UID>:<API-KEY>"
            )

        import cdsapi

        client = cdsapi.Client()
        years = [str(y) for y in range(start_ts.year, end_ts.year + 1)]
        months = [f"{m:02d}" for m in range(1, 13)]
        days = [f"{d:02d}" for d in range(1, 32)]
        hours = [f"{h:02d}:00" for h in range(24)]

        logger.info("Downloading ERA5 for %s [%s, %s]", region, start, end)
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "surface_solar_radiation_downwards",
                    "2m_temperature",
                    "100m_u_component_of_wind",
                    "100m_v_component_of_wind",
                ],
                "year": years,
                "month": months,
                "day": days,
                "time": hours,
                "area": [
                    bounds["lat"][1], bounds["lon"][0],
                    bounds["lat"][0], bounds["lon"][1],
                ],
                "format": "netcdf",
            },
            str(nc_path),
        )

    ds = xr.open_dataset(nc_path)

    rename_map = {}
    for var in ds.data_vars:
        lower = var.lower()
        if "ssrd" in lower or "solar" in lower:
            rename_map[var] = "ssrd"
        elif "t2m" in lower or "2m" in lower or "temperature" in lower:
            rename_map[var] = "t2m"
        elif "u100" in lower or "100m_u" in lower:
            rename_map[var] = "u100"
        elif "v100" in lower or "100m_v" in lower:
            rename_map[var] = "v100"

    ds = ds.rename(rename_map)
    df = ds.mean(dim=["latitude", "longitude"]).to_dataframe()
    df = df[["ssrd", "t2m", "u100", "v100"]]
    df.index.name = "timestamp"
    df = df.loc[start:end]

    return df


def _check_gaps(df: pd.DataFrame, freq: pd.Timedelta) -> pd.DataFrame:
    """Detect and report gaps in a time series, then reindex to fill them."""
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    n_missing = len(full_idx) - len(df)
    if n_missing > 0:
        pct = 100 * n_missing / len(full_idx)
        logger.warning("Found %d missing intervals (%.2f%%)", n_missing, pct)
    df = df.reindex(full_idx)
    df.index.name = "timestamp"
    return df


def build_dataset(
    region: str,
    start: str,
    end: str,
    cache: str | Path = "data/raw/nemosis",
    out_dir: str | Path = "data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build aligned 5-minute and 30-minute datasets for a region.

    Loads prices and demand, aligns timestamps (interval-start), handles
    the INTERVENTION flag, detects and fills missing intervals, then
    writes tidy parquet files to *out_dir*.

    Args:
        region: NEM region identifier.
        start: ISO start date string.
        end: ISO end date string.
        cache: Directory for the NEMOSIS file cache.
        out_dir: Directory for output parquet files.

    Returns:
        Tuple of (5-minute DataFrame, 30-minute DataFrame).
    """
    prices_5min = load_prices(region, start, end, cache, resolution="5min")
    demand_5min = load_demand(region, start, end, cache)

    df_5min = prices_5min.join(demand_5min, how="outer")
    df_5min = _check_gaps(df_5min, _5MIN)

    prices_30min = load_prices(region, start, end, cache, resolution="30min")
    df_30min = prices_30min.copy()
    demand_30min = demand_5min.resample("30min").mean()
    df_30min = df_30min.join(demand_30min, how="outer")
    df_30min = _check_gaps(df_30min, _30MIN)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path_5 = out_dir / f"{region}_5min.parquet"
    path_30 = out_dir / f"{region}_30min.parquet"
    df_5min.to_parquet(path_5)
    df_30min.to_parquet(path_30)
    logger.info(
        "Wrote %s (%d rows) and %s (%d rows)",
        path_5, len(df_5min), path_30, len(df_30min),
    )

    return df_5min, df_30min
