"""Static HTML dashboard builder for trial results.

Reads all trial artifacts from disk (``outputs/trials/``), decimates them into
a compact bundle, and writes a single self-contained ``index.html`` with Plotly
inlined and every series embedded as JSON. The page is fully interactive
client-side — no server, no re-reads, opens instantly at ``file://``.

Memory strategy (the reason the static bundle is fast):

* Summary metrics for every trial come from ``metrics.json`` — no ledger reads.
* Equity and daily-revenue curves are resampled to **daily** resolution over
  the full window (one point per day).
* The 5-minute forecast-vs-actual detail is kept only for the **most recent
  window** (``buffer_days``, default 14; the page opens showing the last 7),
  stored on a regular time grid as ``t0`` + ``dt`` so timestamps cost almost
  nothing.
* Forecast fans are kept only for origins inside the recent window.
* Deep-analytics tables are pre-aggregated here, so the page ships small
  summaries rather than raw ledgers.

Build with::

    python scripts/build_dashboard.py
    # → outputs/dashboard/index.html
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from grian.sim import analytics as _an
from grian.sim.trials import (
    get_git_sha,
    list_regions,
    load_config,
    load_forecasts,
    load_ledger,
    load_metrics,
)

DEFAULT_VIEW_DAYS = 14
DEFAULT_BUFFER_DAYS = 14
_TEMPLATE_PATH = Path(__file__).parent / "assets" / "dashboard.html"

# Reforecast/resolve cadences (in 30-min steps) for the executors whose trials
# are written *without* a config.json (run_common_eval). Used only to synthesise
# a stub config so those trials still appear in the dashboard. None = open-loop
# (one day-ahead plan, no reforecast).
_STUB_EXECUTOR_MPC: dict[str, dict | None] = {
    "openloop": None,
    "mpc30": {"reforecast_every": 1, "resolve_every": 1, "dispatch_mode": "point"},
    "mpc_spike": {"reforecast_every": 48, "resolve_every": 1,
                  "dispatch_mode": "point", "observe_gate": 3000.0},
}
_DEFAULT_BATTERY = {"power_mw": 100.0, "duration_hours": 2.0,
                    "efficiency": 0.85, "max_cycles": 2}


def _all_trial_names(base: Path) -> list[str]:
    """List every trial dir with results, config.json or not.

    ``list_trials`` only returns trials that have a ``config.json`` (testbed
    output); ``run_common_eval`` writes a ledger + metrics but no config. The
    dashboard should show both, so enumerate any dir with a region ``metrics.json``.

    Args:
        base: Trials root directory.

    Returns:
        Sorted trial names (excludes the ``_oracle`` helper dir).
    """
    if not base.exists():
        return []
    names = []
    for d in base.iterdir():
        if not d.is_dir() or d.name.startswith("_"):
            continue
        if any(r.is_dir() and (r / "metrics.json").exists() for r in d.iterdir()):
            names.append(d.name)
    return sorted(names)


def _stub_config(name: str) -> dict:
    """Synthesise a minimal config for a trial that has no ``config.json``.

    The trial name is ``<model>__<executor>`` and the model name is
    self-describing (``lear_weather_fourier`` → LEAR + weather + Fourier), so the
    settings the dashboard needs can be recovered from it. Mirrors the shape the
    testbed writes (see :func:`_trial_settings`).

    Args:
        name: Trial directory name.

    Returns:
        A config dict good enough for ``_trial_settings`` and model metadata.
    """
    model = name.split("__", 1)[0]
    executor = name.rsplit("__", 1)[1] if "__" in name else "openloop"
    params: dict = {}
    if "weather" in model:
        params["include_weather"] = True
    if "fourier" in model:
        params["calendar_encoding"] = "fourier"
    elif "ordinal" in model:
        params["calendar_encoding"] = "ordinal"
    return {
        "model": model,
        "resolution": "30min",
        "horizon": 48,
        "refit_days": 28,
        "train_lookback_days": 548,
        "transform": "asinh",
        "model_params": params,
        "dispatch": _DEFAULT_BATTERY,
        "mpc": _STUB_EXECUTOR_MPC.get(executor),
    }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _clean(series: pd.Series, ndigits: int) -> list:
    """Round a series to a JSON-safe list, mapping NaN to ``None`` (chart gap)."""
    return [None if pd.isna(v) else round(float(v), ndigits) for v in series]


# ---------------------------------------------------------------------------
# Per-trial decimation
# ---------------------------------------------------------------------------

def _daily_series(ledger: pd.DataFrame) -> dict:
    """Daily equity curve and daily revenue over the full window."""
    daily = ledger["revenue"].resample("D").sum()
    dates = [d.strftime("%Y-%m-%d") for d in daily.index]
    return {
        "dates": dates,
        "equity": _clean(daily.cumsum(), 2),
        "revenue": _clean(daily, 2),
    }


def _recent_block(ledger: pd.DataFrame, buffer_days: int) -> dict | None:
    """High-resolution recent window on a regular 5-min grid (gaps → None)."""
    if ledger.empty:
        return None
    end = ledger.index.max()
    start = end - pd.Timedelta(days=buffer_days)
    win = ledger.loc[ledger.index >= start]
    if win.empty:
        return None
    minutes = int(win["interval_minutes"].iloc[0]) if "interval_minutes" in win else 5
    dt = pd.Timedelta(minutes=minutes)
    grid = pd.date_range(win.index.min().floor(dt), end, freq=dt)
    win = win.reindex(grid)
    return {
        "t0": grid[0].isoformat(),
        "dt_min": minutes,
        "n": len(grid),
        "actual": _clean(win["actual_price"], 2),
        "forecast": _clean(win["forecast_price"], 2),
        "soc": _clean(win["soc_mwh"], 2),
        "net": _clean(win["net_mw"], 2),
    }


def _fans(forecasts: pd.DataFrame, cutoff: pd.Timestamp) -> list:
    """Committed day-ahead forecast trajectories for origins on/after ``cutoff``.

    Only *full* trajectories are returned. An every-interval MPC saves a
    one-step forecast per origin (no committed day-ahead plan); those degenerate
    fans are dropped, so the day-ahead-fan section shows only configs that
    actually commit a trajectory (open-loop / daily-reforecast).
    """
    recent = forecasts[forecasts["origin"] >= cutoff]
    groups = list(recent.groupby("origin"))
    if not groups:
        return []
    max_len = max(len(g) for _, g in groups)
    if max_len < 2:                      # only near-term steps → no day-ahead plan
        return []
    fans = []
    for origin, grp in groups:
        if len(grp) < 0.8 * max_len:     # skip partial/degenerate origins
            continue
        grp = grp.sort_values("step")
        fans.append({
            "origin": pd.Timestamp(origin).strftime("%Y-%m-%d"),
            "forecast": _clean(grp["forecast"], 2),
            "actual": _clean(grp["actual"], 2),
        })
    return fans


def _analytics_block(ledger: pd.DataFrame) -> dict:
    """Pre-aggregated deep-analytics tables for the full window."""
    ebh = _an.error_by_hour(ledger)
    rbh = _an.revenue_by_hour(ledger)
    eff = _an.dispatch_efficiency(ledger)
    worst_rev = _an.worst_days(ledger, n=10, by="revenue").reset_index()
    worst_mae = _an.worst_days(ledger, n=10, by="mae").reset_index()

    def _worst(df: pd.DataFrame) -> list:
        return [
            {
                "date": pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
                "revenue": round(float(r["revenue"]), 2),
                "mae": round(float(r["mae"]), 2),
                "max_price": round(float(r["max_price"]), 2),
                "min_price": round(float(r["min_price"]), 2),
            }
            for _, r in df.iterrows()
        ]

    return {
        "error_by_hour": {
            "hour": [int(h) for h in ebh.index],
            "mae": _clean(ebh["mae"], 2),
            "bias": _clean(ebh["bias"], 2),
        },
        "revenue_by_hour": {
            "hour": [int(h) for h in rbh.index],
            "total_revenue": _clean(rbh["total_revenue"], 2),
        },
        "dispatch_efficiency": {k: round(float(v), 2) for k, v in eff.items()},
        "worst_revenue": _worst(worst_rev),
        "worst_mae": _worst(worst_mae),
    }


def _common_oracle_daily(base_path: Path, region: str) -> pd.Series | None:
    """Daily revenue of the single shared perfect-foresight oracle for a region.

    In the common-window paradigm every configuration is evaluated over one
    identical span, and there is one oracle over that span at
    ``<base>/_oracle/<region>/common.parquet``. Because capture ratio over any
    sub-window is ``sum(model daily) / sum(oracle daily)``, embedding the
    oracle's daily revenue lets the dashboard recompute capture for any test
    window client-side.

    Args:
        base_path: Root directory holding per-trial artifact folders.
        region: NEM region identifier.

    Returns:
        Daily oracle revenue indexed by date, or ``None`` if not yet computed.
    """
    path = base_path / "_oracle" / region / "common.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)["revenue"].resample("D").sum()


def _common_price_daily(base_path: Path, region: str) -> pd.Series | None:
    """Daily *maximum* actual price for a region, on the shared oracle date axis.

    Read from the same oracle ``common.parquet`` (its ``actual_price`` column) so
    it aligns exactly with the daily oracle/revenue series. The dashboard overlays
    this under the equity curves so the reader can see each price spike land on the
    day the cumulative-revenue curves step up. Daily *max* (not mean) because a
    spike is a single half-hour and the max is what makes the money.

    Args:
        base_path: Root directory holding per-trial artifact folders.
        region: NEM region identifier.

    Returns:
        Daily max price indexed by date, or ``None`` if not yet computed.
    """
    path = base_path / "_oracle" / region / "common.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)["actual_price"].resample("D").max()


_TESTBED_FANS = Path("outputs/testbed/fans")


def _quantile_fans(model, sim_price, resolution="30min", n_days=6):
    """Recent day-ahead quantile fans for a model, from the saved testbed fans.

    The ablation testbed persists the *real* quantile fan for every quantile
    model over the full window (``outputs/testbed/fans/<model>__full__<res>.parquet``,
    long format ``origin, quantile, step, price`` in dollar space). We slice the
    last ``n_days`` daily origins and pair each with the realised prices — no
    on-the-fly model reload, so it works for every quantile model at the right
    resolution.

    Args:
        model: Model name (e.g. ``lightgbm_qmean_weather_fourier``).
        sim_price: 30-min price series (dollar space) for the realised actuals.
        resolution: Grid resolution; selects the fan file and steps-per-day.
        n_days: Number of most-recent daily origins to expose.

    Returns:
        List of ``{date, q{level}: [...], actual: [...]}`` (dollar space), or None.
    """
    fan_path = _TESTBED_FANS / f"{model}__full__{resolution}.parquet"
    if sim_price is None or not fan_path.exists():
        return None
    try:
        df = pd.read_parquet(fan_path)
        ppd = 48 if resolution == "30min" else 288
        origins = sorted(df["origin"].unique())[-n_days:]
        fans = []
        for origin in origins:
            o_ts = pd.Timestamp(origin)
            pos = int(sim_price.index.searchsorted(o_ts))
            actual = sim_price.iloc[pos:pos + ppd]
            if actual.empty:
                continue
            sub = df[df["origin"] == origin]
            # Always emit the full day-ahead horizon; pad the realised prices with
            # NaN for origins near the data end so the fan isn't truncated to a
            # few hours.
            pad = [float("nan")] * (ppd - len(actual))
            actual_padded = list(actual.to_numpy()) + pad
            row = {"date": o_ts.strftime("%Y-%m-%d"),
                   "actual": _clean(pd.Series(actual_padded), 2)}
            for q, grp in sub.groupby("quantile"):
                arr = grp.sort_values("step")["price"].to_numpy()[:ppd]
                row[f"q{int(round(float(q) * 100)):02d}"] = _clean(pd.Series(arr), 2)
            fans.append(row)
        return fans or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Bundle assembly
# ---------------------------------------------------------------------------

def _trial_settings(cfg: dict, executor: str) -> dict:
    """Human-readable hyperparameters for one trial, for the settings panel.

    Captures the knobs needed to interpret a trial's plots — crucially the
    *reforecast cadence*, which is what makes open-loop (one day-ahead forecast)
    differ from MPC (re-forecast every N minutes from fresh data).

    Args:
        cfg: The trial's config dict.
        executor: Executor name (trial-name suffix).

    Returns:
        Flat dict of display-ready settings.
    """
    res = cfg.get("resolution", "5min")
    interval_min = 5 if res == "5min" else 30
    mpc = cfg.get("mpc") or {}
    params = cfg.get("model_params") or {}
    disp = cfg.get("dispatch") or {}
    if executor == "openloop" or not mpc:
        reforecast, resolve = "once/day (day-ahead)", "—"
    else:
        rf, rs = mpc.get("reforecast_every"), mpc.get("resolve_every")
        reforecast = f"every {int(rf) * interval_min} min" if rf else "?"
        resolve = f"every {int(rs) * interval_min} min" if rs else "?"
    battery = (f"{disp.get('power_mw', '?')} MW / {disp.get('duration_hours', '?')}h"
               f" / η {disp.get('efficiency', '?')} / {disp.get('max_cycles', '?')} cyc"
               ) if disp else None
    # Linear/AR models default to one-hot calendar when unset in model_params.
    encoding = params.get("calendar_encoding")
    model = str(cfg.get("model", ""))
    if encoding is None and any(
        k in model for k in ("lear", "ridge", "elasticnet", "autoregression")):
        encoding = "onehot (default)"
    return {
        "resolution": res,
        "horizon": cfg.get("horizon"),
        "refit_days": cfg.get("refit_days"),
        "lookback_days": cfg.get("train_lookback_days"),
        "transform": cfg.get("transform"),
        "reforecast": reforecast,
        "resolve": resolve,
        "dispatch_mode": mpc.get("dispatch_mode", "point"),
        "cvar_lambda": mpc.get("cvar_lambda"),
        "cvar_alpha": mpc.get("cvar_alpha"),
        "calendar_encoding": encoding,
        "weather": bool(params.get("include_weather", False)),
        "quantiles": params.get("quantiles"),
        "battery": battery,
    }


def build_bundle(
    base: str | Path = "outputs/trials",
    region_filter: str | None = None,
    buffer_days: int = DEFAULT_BUFFER_DAYS,
) -> dict:
    """Read every trial artifact and assemble the decimated data bundle.

    Args:
        base: Root directory holding per-trial artifact folders.
        region_filter: If set, keep only this NEM region.
        buffer_days: Days of 5-minute detail to embed for the recent window.

    Returns:
        A JSON-serialisable dict with ``meta``, ``trials`` (summary rows), and
        ``detail`` (per trial/region daily series, recent block, fans,
        analytics).
    """
    base_path = Path(base)
    trials: list[dict] = []
    detail: dict[str, dict] = {}
    naive_mae: dict[str, float] = {}
    oracle_series: dict[str, pd.Series | None] = {}
    # 30-min price series for the realised actuals in the quantile-fan overlay.
    sim30_path = Path("data/processed/SA1_30min_sim.parquet")
    sim30_price = (pd.read_parquet(sim30_path)["price"]
                   if sim30_path.exists() else None)
    qfan_by_model: dict[str, list | None] = {}   # cache: fans are per-model

    def oracle_for(region: str) -> pd.Series | None:
        if region not in oracle_series:
            oracle_series[region] = _common_oracle_daily(base_path, region)
        return oracle_series[region]

    for name in _all_trial_names(base_path):
        try:
            cfg = load_config(name, base_path)
        except FileNotFoundError:
            cfg = _stub_config(name)   # run_common_eval trials: derive from name
        for region in list_regions(name, base_path):
            if region_filter and region != region_filter:
                continue
            try:
                metrics = load_metrics(name, region, base_path)
            except FileNotFoundError:
                continue

            model = str(cfg.get("model", ""))
            # Executor is encoded as the trial-name suffix: "<model>__<executor>".
            executor = name.rsplit("__", 1)[1] if "__" in name else "openloop"
            row = {
                "name": name,
                "region": region,
                "model": model,
                "executor": executor,
                "mae": metrics.get("mae"),
                "rmse": metrics.get("rmse"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "peak_drawdown": metrics.get("peak_drawdown"),
                "cfg": _trial_settings(cfg, executor),
            }
            trials.append(row)
            if "naive" in model.lower() and metrics.get("mae"):
                naive_mae[region] = float(metrics["mae"])

            key = f"{name}::{region}"
            block: dict = {}
            cutoff = None
            odaily = oracle_for(region)
            try:
                ledger = load_ledger(name, region, base_path)
                if not ledger.empty:
                    block["daily"] = _daily_series(ledger)
                    block["recent"] = _recent_block(ledger, buffer_days)
                    block["analytics"] = _analytics_block(ledger)
                    cutoff = ledger.index.max() - pd.Timedelta(days=buffer_days)
                    # Daily revenue aligned to the shared oracle date axis so the
                    # dashboard can recompute capture over any test sub-window.
                    if odaily is not None:
                        aligned = ledger["revenue"].resample("D").sum().reindex(
                            odaily.index).fillna(0.0)
                        row["daily"] = _clean(aligned, 2)
            except FileNotFoundError:
                pass
            if cutoff is not None:
                try:
                    forecasts = load_forecasts(name, region, base_path)
                    block["fans"] = _fans(forecasts, cutoff)
                except FileNotFoundError:
                    pass
            # Quantile fan for probabilistic models (qmean / qra) — the real
            # saved testbed fan, loaded once per model and shared across executors.
            if sim30_price is not None and ("qmean" in model or "qra" in model):
                if model not in qfan_by_model:
                    qfan_by_model[model] = _quantile_fans(model, sim30_price)
                if qfan_by_model[model]:
                    block["qfan"] = qfan_by_model[model]
            if block:
                detail[key] = block

    # Forecast skill vs the naive baseline in the same region (whole-pool MAE).
    for row in trials:
        base_mae = naive_mae.get(row["region"])
        if base_mae and row.get("mae"):
            row["mae_skill"] = round(1 - row["mae"] / base_mae, 4)
        else:
            row["mae_skill"] = None

    # One shared oracle daily-revenue series per region (capture denominator).
    common: dict[str, dict] = {}
    for region, series in oracle_series.items():
        if series is not None:
            block = {
                "dates": [d.strftime("%Y-%m-%d") for d in series.index],
                "oracle": _clean(series, 2),
            }
            price_max = _common_price_daily(base_path, region)
            if price_max is not None:
                block["price_max"] = _clean(price_max.reindex(series.index), 2)
            common[region] = block

    meta = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "git_sha": get_git_sha(),
        "view_days": DEFAULT_VIEW_DAYS,
        "buffer_days": buffer_days,
        "default_window_months": 12,
        "n_trials": len(trials),
    }
    return {
        "meta": meta,
        "trials": trials,
        "detail": detail,
        "common": common,
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def build_dashboard(
    base: str | Path = "outputs/trials",
    out: str | Path = "outputs/dashboard/index.html",
    region_filter: str | None = None,
    buffer_days: int = DEFAULT_BUFFER_DAYS,
) -> Path:
    """Build the bundle and write the self-contained dashboard HTML.

    Args:
        base: Root directory holding per-trial artifact folders.
        out: Destination HTML path (parent dirs are created).
        region_filter: If set, restrict the dashboard to one NEM region.
        buffer_days: Days of 5-minute recent detail to embed.

    Returns:
        The path written.
    """
    from plotly.offline import get_plotlyjs

    bundle = build_bundle(base, region_filter, buffer_days)
    html = (
        _TEMPLATE_PATH.read_text(encoding="utf-8")
        .replace("/*__PLOTLY_JS__*/", get_plotlyjs())
        .replace("/*__DATA__*/", json.dumps(bundle, allow_nan=False))
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    path = build_dashboard()
    print(f"Wrote {path}")
