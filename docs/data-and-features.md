# grian — Input Data & Feature Schemas

A thorough reference for every input the models see: the raw data schema, its
lineage and units, the target transform, and the full engineered feature matrix
(exact column names, formulas, windows, and leakage guarantees).

Region: **SA1** (South Australia). Everything generalises to other NEM regions.

---

## 1. Raw processed dataset

The single source table the pipeline consumes:

- `data/processed/SA1_5min_sim.parquet` — native 5-minute grid.
- `data/processed/SA1_30min_sim.parquet` — 30-minute grid (mean of each six
  5-min intervals = the NEM 30-minute settlement price; see §6).

Both share one schema: a `DatetimeIndex` named `timestamp` plus five float
columns.

| Column | Meaning | Units | Source | Typical range (SA1) |
|---|---|---|---|---|
| `price` | Regional reference price (RRP) | $/MWh | AEMO via NEMOSIS | −1000 … 20300 |
| `demand` | Operational total demand | MW | AEMO via NEMOSIS | ~−300 … 3300 |
| `ssrd` | Surface solar radiation downwards | J·m⁻² (hourly accumulation) | ERA5 | 0 … 3.9e6 |
| `t2m` | 2-metre air temperature | **Kelvin** | ERA5 | 278 … 309 (≈ 5–36 °C) |
| `wind_speed` | 100-metre wind speed magnitude | m/s | ERA5 (derived) | 0 … 15 |

Notes:
- **Index is interval-*beginning*.** AEMO publishes interval-*ending*
  timestamps; `data.py::_shift_to_interval_start` subtracts one interval on load,
  so `timestamp` marks the *start* of each interval. This is applied once, at
  ingest, and never again.
- **Timezone:** the index is tz-naive AEST (UTC+10). ERA5 is fetched in UTC and
  shifted **+10 h** on merge so solar noon aligns (verified: `ssrd` peaks ~hour
  13). Price/demand are already AEST.
- **`price` floor/cap:** the NEM market price floor is −$1000/MWh (negative
  prices are real — oversupply). The upper values reflect scarcity-price events
  near the market price cap.
- **Negative `demand`** occurs occasionally: "operational demand" nets off
  distributed rooftop PV, which in SA can exceed grid load midday.
- Span: **2023-01-01 → 2026-06-29**. 5-min = 367,488 rows; 30-min = 61,248 rows.

---

## 2. Data lineage & transforms applied at ingest

```
AEMO NEMOSIS  ── DISPATCHPRICE.RRP ───────────┐
             ── DISPATCHREGIONSUM.TOTALDEMAND ─┤
                                               ├─► interval-shift ─► merge ─► processed parquet
ERA5 (CDS)   ── ssrd, t2m, u100, v100 ─────────┘   (+10h, ffill, wind derive)
```

- **Price** (`data.py::load_price`): NEMOSIS `DISPATCHPRICE`, columns
  `SETTLEMENTDATE, REGIONID, RRP` → `price`; intervention duplicates dropped;
  interval-shifted.
- **Demand** (`load_demand`): NEMOSIS `DISPATCHREGIONSUM.TOTALDEMAND` → `demand`;
  interval-shifted; gap-checked and reindexed to a complete 5-min grid.
- **Weather** (`load_era5`): ERA5 `reanalysis-era5-single-levels`, hourly,
  spatially averaged over the region's lat/lon box. Raw variables:
  `surface_solar_radiation_downwards` (`ssrd`), `2m_temperature` (`t2m`),
  `100m_u/v_component_of_wind` (`u100`, `v100`).
  - `wind_speed = sqrt(u100² + v100²)` (100 m wind magnitude).
  - Hourly ERA5 is **forward-filled** onto the 5-min grid (no interpolation, so
    no future value leaks into a sub-hour interval).

---

## 3. Target & transform

- **Target:** `price` ($/MWh).
- **Model space:** the inverse hyperbolic sine, `asinh(price)` (`np.arcsinh`),
  inverted with `np.sinh` before scoring. `asinh` behaves like `log` for large
  |price| but is defined for the **negative and zero** prices the NEM has, which
  `log1p` cannot handle. All reported errors and revenues are inverted back to
  **dollars**.
- Quantile models forecast quantiles in transformed space, invert each quantile
  (monotone transform preserves order), then integrate to a dollar-space mean.

---

## 4. Engineered feature matrix (`sim/features.py::build_features`)

All features are **strictly backward-looking**: every column uses `shift(≥1)` or
a rolling window on already-shifted data, so the value at time *t* depends only
on data observed **before** *t*. Rows at the very start contain NaNs (from
lags/rolling) and are dropped by the caller.

`iph` = intervals per hour (12 @ 5min, 2 @ 30min); `ppd` = intervals per day
(288 @ 5min, 48 @ 30min).

### 4.1 Price lags — `price_lags` (3 cols, always on)
Point price value at fixed day offsets.

| Column | Definition | 5-min shift | 30-min shift |
|---|---|---|---|
| `price_lag_{1d}` | `price.shift(ppd)` | 288 | 48 |
| `price_lag_{2d}` | `price.shift(2·ppd)` | 576 | 96 |
| `price_lag_{7d}` | `price.shift(7·ppd)` | 2016 | 336 |

### 4.2 Rolling price stats — `rolling_stats` (12 cols, always on)
Mean / std / min / max over 1 h, 6 h, 24 h windows on `price.shift(1)`.

`price_rmean_{w}`, `price_rstd_{w}`, `price_rmin_{w}`, `price_rmax_{w}` for
`w ∈ {1h, 6h, 24h}`. (std NaNs filled with 0.)

### 4.3 Calendar — `calendar_features` (5 cols, always on)

| Column | Definition |
|---|---|
| `hour` | hour + minute/60 (0–23.92) |
| `day_of_week` | 0 = Mon … 6 = Sun |
| `month` | 1–12 |
| `is_weekend` | 1.0 if dow ≥ 5 |
| `hour_x_dow` | `hour · day_of_week` interaction |

### 4.4 Momentum — `momentum_features` (4 cols, always on)
Percentage returns of `price.shift(1)` over 1 h / 6 h / 24 h, plus sign.

`price_ret_1h`, `price_ret_6h`, `price_ret_24h`, `price_direction_1h`
(= `sign(ret_1h)`). Inf/NaN → 0.

### 4.5 Intraday profile — `intraday_profile` (2 cols, always on)
The recent average daily shape and the deviation from it.

| Column | Definition |
|---|---|
| `intraday_profile_mean` | rolling mean of `price.shift(1)` over `7·ppd` |
| `intraday_deviation` | `price.shift(1) − intraday_profile_mean` |

### 4.6 Demand — `demand_features` (4 cols, on when `include_demand=True`, default)
Empty if no `demand` column.

| Column | Definition |
|---|---|
| `demand` | `demand.shift(1)` (last observed demand) |
| `demand_rmean_6h` | rolling mean over 6 h |
| `demand_rmean_24h` | rolling mean over 24 h |
| `demand_rstd_24h` | rolling std over 24 h |

### 4.7 Scarcity / spike-precursors — `scarcity_features` (opt-in, `include_scarcity=True`)
Spike-clustering and demand-stress signals. **3 columns** (price-only) or
**7 columns** when `demand` is present. `spike_threshold` default $300/MWh.

| Column | Definition |
|---|---|
| `spikes_24h` | count of intervals with `price>threshold` in trailing `ppd` |
| `intervals_since_spike` | intervals since last spike, capped at `ppd` |
| `price_vol_ratio` | `rstd_1h / (rstd_24h + 1)`, clipped [0,20] |
| `demand_stress_max` | `demand / rolling 7-day max demand` |
| `demand_stress_p95` | `demand / rolling 7-day 95th-pct demand` |
| `demand_ramp_1h` | 1-hour % change in demand |
| `demand_accel` | change in the ramp (2nd difference) |

### 4.8 Weather — `weather_features` (9 cols, on when `include_weather=True`)
For each of `ssrd`, `t2m`, `wind_speed` (present columns only):

| Column | Definition |
|---|---|
| `wx_{c}` | current value (observed at forecast origin) |
| `wx_{c}_3h` | rolling 3-hour mean |
| `wx_{c}_chg` | 1-hour difference (`diff(iph)`) |

> **Leakage caveat (important):** weather features are **origin-time** — the
> weather *observed now* — not a day-ahead weather *forecast*. They are
> leakage-free (no future value enters), but a true forward weather forecast
> would need NEMSEER/BOM forecast data and is a flagged future extension. This
> is why current weather lift is modest/mixed.

---

## 5. Feature sets per model

| Model | Base¹ | demand | scarcity | weather | Total cols |
|---|:--:|:--:|:--:|:--:|:--:|
| `naive_similar_day` | — (uses raw series, not this matrix) | | | | — |
| `autoregression` | — (AR on the series) | | | | — |
| `lightgbm_rich` | ✓ | ✓ | flag | flag | **30** |
| `lightgbm_rich_weather` | ✓ | ✓ | flag | ✓ | **39** |
| `lightgbm_qmean` | ✓ | ✓ | — | flag | **30** |
| `lightgbm_qmean_weather` | ✓ | ✓ | — | ✓ | **39** |

¹ Base = price_lags(3) + rolling_stats(12) + calendar(5) + momentum(4) +
intraday(2) = **26**. +demand(4)=30, +weather(9)=39, +scarcity(3 or 7) when
enabled. LightGBM trains **one booster per forecast step** (strided across the
288/48-step day-ahead horizon); quantile models train one booster per
(step, quantile).

---

## 6. Resolution handling (5-min ↔ 30-min)

- The 30-min series is `resample('30min').mean()` of every column — for `price`
  this is exactly the NEM 30-min settlement price (average of six 5-min dispatch
  prices).
- All interval-denominated feature parameters scale with resolution via `iph`
  and `ppd`, so the *same* feature definitions hold at either grid.
- Day-ahead horizon = `ppd` (288 @ 5min, 48 @ 30min).

---

## 7. Downstream forecast schemas

**Point forecast** (naive, AR, rich): a length-`horizon` price vector (dollars),
anchored at the forecast origin.

**Quantile fan** (`predict_fan`, qmean family): `{quantile_level: np.ndarray}`
over the horizon, dollar-space, quantiles `[0.05, 0.5, 0.9, 0.98]` (monotone per
step). This fan drives probabilistic dispatch (scenario / EV / CVaR) and the
CRPS / coverage diagnostics.

**Cached fan (test bed)** — `outputs/testbed/fans/{model}__{window}__{res}.parquet`,
long format: `origin` (reforecast timestamp), `quantile`, `step` (0…horizon−1),
`price` ($). Reconstructs to `{origin: {quantile: array}}` for dispatch replay.

**Ledger** (per trial, `.../SA1/ledger.parquet`): one row per interval with
`timestamp, actual_price, forecast_price, charge_mw, discharge_mw, soc_mwh,
net_mw, revenue, interval_minutes`.

---

## 8. Leakage guarantees (summary)

1. Every feature uses `shift(≥1)` or rolling-on-shifted — no same-interval or
   future data.
2. AEMO interval-shift applied once at ingest (interval-ending → beginning).
3. Weather is forward-filled from hourly (never back-filled) and used at
   origin-time only.
4. The rolling-origin backtest trains only on history up to each trading day and
   trades the *next* day (the common-window eval uses `embargo=0` for
   deployment realism — safe because train and trade windows never overlap; the
   backtest module also supports a horizon-length embargo). Integration tests
   (`test_no_future_leakage`, `test_leakage_improves_mae`) prove that injecting a
   future value changes the score — the guard against silent leakage.
