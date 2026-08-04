"""Pull ERA5 weather for SA1 and merge it into the 5-min sim dataset.

Downloads surface solar radiation, 2 m temperature, and 100 m wind for the SA1
region and adds ``ssrd``, ``t2m``, ``wind_speed`` columns (hourly ERA5 resampled
to 5 min) to ``data/processed/SA1_5min_sim.parquet``.

The request is split into full-year (2023–2025) and partial-2026 parts so we
never ask CDS for not-yet-published dates. ERA5 is *reanalysis* (actual weather,
~5-day lag): using it as a forecast feature is the documented "proxy" assumption
(weather is highly forecastable day-ahead).

Usage::

    python scripts/build_weather_data.py            # download + merge
    python scripts/build_weather_data.py --probe     # parse the cached probe only
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REGION = "SA1"
# SA1 bounds (lat -38…-26, lon 129…141) as CDS area [North, West, South, East].
AREA = [-26.0, 129.0, -38.0, 141.0]
VARS = ["surface_solar_radiation_downwards", "2m_temperature",
        "100m_u_component_of_wind", "100m_v_component_of_wind"]
CACHE = Path("data/raw/era5")
SIM_PATH = Path(f"data/processed/{REGION}_5min_sim.parquet")


def _pull(name: str, years: list[str], months: list[str]) -> Path:
    """Retrieve one ERA5 NetCDF (cached by name)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{REGION}_{name}.nc"
    if out.exists():
        return out
    import cdsapi

    cdsapi.Client().retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis", "variable": VARS,
            "year": years, "month": months,
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": AREA, "data_format": "netcdf",
        },
        str(out),
    )
    return out


def _parse(nc_path: Path) -> pd.DataFrame:
    """NetCDF (or CDS zip of netcdfs) → hourly DataFrame of ssrd, t2m, wind_speed."""
    import zipfile

    if zipfile.is_zipfile(nc_path):
        # CDS returns a zip of per-stream netcdfs (instantaneous + accumulated).
        with zipfile.ZipFile(nc_path) as z:
            names = [n for n in z.namelist() if n.endswith(".nc")]
            tmp = nc_path.parent / f"{nc_path.stem}_x"
            tmp.mkdir(exist_ok=True)
            paths = [z.extract(n, tmp) for n in names]
        ds = xr.merge([xr.open_dataset(p) for p in paths], compat="override")
    else:
        ds = xr.open_dataset(nc_path)
    # Collapse the ERA5/ERA5T preliminary split if present (non-overlapping in
    # time, so a skipna mean keeps the one valid value per timestep).
    if "expver" in ds.dims:
        ds = ds.mean(dim="expver", skipna=True)
    for extra in ("number",):
        if extra in ds.dims:
            ds = ds.mean(dim=extra, skipna=True)
    # Spatial average over the region.
    ds = ds.mean(dim=[d for d in ("latitude", "longitude") if d in ds.dims])
    df = ds.to_dataframe()
    time_col = "valid_time" if "valid_time" in df.index.names else "time"
    df = df.reset_index().set_index(time_col)
    df.index.name = "timestamp"
    keep = {c: c for c in ("ssrd", "t2m", "u100", "v100") if c in df.columns}
    df = df[list(keep)].astype(float)
    df["wind_speed"] = np.sqrt(df["u100"] ** 2 + df["v100"] ** 2)
    return df[["ssrd", "t2m", "wind_speed"]].sort_index()


def main() -> None:
    """Download ERA5 (unless --probe), merge into the sim parquet."""
    if "--probe" in sys.argv:
        probe = Path(sys.argv[sys.argv.index("--probe") + 1])
        print(_parse(probe).head())
        return

    # One request per month keeps each well under CDS's per-request cost limit.
    # 2026 only goes to June (later months not yet published).
    ym = [(y, f"{m:02d}") for y in ("2023", "2024", "2025") for m in range(1, 13)]
    ym += [("2026", f"{m:02d}") for m in range(1, 7)]
    parts = [_pull(f"{y}_{m}", [y], [m]) for y, m in ym]
    weather = pd.concat([_parse(p) for p in parts]).sort_index()
    weather = weather[~weather.index.duplicated(keep="first")]
    # ERA5 is UTC; NEM/AEMO market data is AEST (UTC+10, no DST). Shift so solar
    # peaks at local midday, not UTC midday — else the feature is ~10h misaligned.
    weather.index = weather.index + pd.Timedelta(hours=10)
    print(f"ERA5 hourly (→AEST): {len(weather)} rows to {weather.index.max()}")

    sim = pd.read_parquet(SIM_PATH)
    # Hourly → 5-min by FORWARD-FILL only (no interpolation): each 5-min step
    # takes the most recent *published* hourly value. Interpolating would blend
    # in the next hour's value — a ~55-min look-ahead leak. Forward-fill keeps
    # every weather feature strictly known-at-time.
    w5 = weather.reindex(sim.index, method="ffill")
    for col in ("ssrd", "t2m", "wind_speed"):
        sim[col] = w5[col].values
    n_nan = int(sim[["ssrd", "t2m", "wind_speed"]].isna().any(axis=1).sum())
    sim = sim.ffill().bfill()  # only the very first pre-dawn rows, if any
    sim.to_parquet(SIM_PATH)
    print(f"Merged weather (ffill) into {SIM_PATH}: cols={list(sim.columns)}, "
          f"rows={len(sim)}, filled {n_nan} edge NaNs")


if __name__ == "__main__":
    main()
