"""Baseline forecasters: similar-day naive and autoregression."""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grian.models._shared import (
    _build_lag_features,
    _calendar_features,
    _linear_preprocessor,
    _periods_per_day,
)
from grian.models.params import ARParams, NaiveParams, default_lags
from grian.models.spec import ModelSpec


def _naive_fit(train_df: pd.DataFrame, target_col: str, cfg: Mapping[str, Any]) -> dict:
    """Naive model "fitting" — just stores a reference to the data.

    Args:
        train_df: Training DataFrame with target column.
        target_col: Name of the target column.
        cfg: Trial config dict.

    Returns:
        State dict containing the training series and resolution.
    """
    return {
        "series": train_df[target_col].copy(),
        "target_col": target_col,
        "resolution": cfg.get("resolution", "5min"),
    }


def _naive_predict(state: dict, input_df: pd.DataFrame, horizon: int) -> pd.Series:
    """Produce a forecast by repeating last week's same-day profile.

    Args:
        state: State dict from _naive_fit.
        input_df: Data up to the forecast origin; the target column
            is used as the history to repeat. Falls back to fit-time data.
        horizon: Number of periods to forecast.

    Returns:
        Series of length `horizon` with the naive forecast.
    """
    series = state["series"]
    # Predict-from-now: when the runner supplies data up
    # to the forecast origin, use it — the frozen fit-time series drifts
    # up to refit_days stale and breaks day-of-week alignment.
    target_col = state.get("target_col")
    if input_df is not None and target_col in getattr(input_df, "columns", []):
        if len(input_df) > 0:
            series = input_df[target_col]
    ppd = _periods_per_day(state["resolution"])
    # 7 days back in periods
    lookback = 7 * ppd
    if len(series) < lookback:
        # Not enough history — repeat the last available values
        values = series.iloc[-horizon:].values
    else:
        values = series.iloc[-lookback: -lookback + horizon].values

    if len(values) < horizon:
        values = np.pad(values, (0, horizon - len(values)),
                        constant_values=values[-1] if len(values) > 0 else 0.0)

    return pd.Series(values[:horizon], name="forecast")


def _naive_save(state: dict, path: str | Path) -> None:
    """Persist naive model state to disk.

    Args:
        state: State dict from _naive_fit.
        path: Directory to write into.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state["series"].to_frame().to_parquet(path / "naive_history.parquet")
    with open(path / "naive_meta.json", "w") as f:
        json.dump({"resolution": state["resolution"]}, f)


def _naive_load(path: str | Path) -> dict:
    """Restore naive model state from disk.

    Args:
        path: Directory to read from.

    Returns:
        State dict matching what _naive_fit produces.
    """
    path = Path(path)
    df = pd.read_parquet(path / "naive_history.parquet")
    with open(path / "naive_meta.json") as f:
        meta = json.load(f)
    return {"series": df.iloc[:, 0], "resolution": meta["resolution"]}


NAIVE_SIMILAR_DAY = ModelSpec(
    name="naive_similar_day",
    output="point",
    fit=_naive_fit,
    predict=_naive_predict,
    save=_naive_save,
    load=_naive_load,
    params=NaiveParams,
)


def _ar_fit(train_df: pd.DataFrame, target_col: str, cfg: Mapping[str, Any]) -> dict:
    """Fit a linear autoregressive model on lagged target values.

    Calendar terms are encoded for a *linear* fit: one-hot by default, or
    Fourier / ordinal via ``model_params.calendar_encoding`` — the same
    treatment as the LEAR family (ordinal integers mislead a linear model).

    Args:
        train_df: Training DataFrame with target column.
        target_col: Name of the target column.
        cfg: Trial config dict (reads model_params.lags, .calendar_encoding).

    Returns:
        State dict with the fitted sklearn pipeline, lag config, etc.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    p = ARParams.model_validate(cfg.get("model_params") or {})
    ppd = _periods_per_day(cfg.get("resolution", "5min"))
    lags = p.lags if p.lags is not None else default_lags(ppd)
    calendar_encoding = p.calendar_encoding

    series = train_df[target_col]
    lag_df = _build_lag_features(series, lags)
    cal_df = _calendar_features(series.index)
    X = pd.concat([lag_df, cal_df], axis=1).dropna()
    y = series.reindex(X.index)
    feature_cols = list(X.columns)

    model = Pipeline([
        ("pre", _linear_preprocessor(feature_cols, calendar_encoding)),
        ("est", LinearRegression()),
    ])
    model.fit(X, y.values)   # DataFrame in — named-column calendar encoding

    return {
        "model": model,
        "lags": lags,
        "feature_cols": feature_cols,
        "calendar_encoding": calendar_encoding,
        "resolution": cfg.get("resolution", "5min"),
        "target_col": target_col,
        "last_values": series.iloc[-max(lags):].copy(),
    }


def _ar_predict(state: dict, input_df: pd.DataFrame, horizon: int) -> pd.Series:
    """Produce an iterative AR forecast, conditioned on the live recent tail.

    Predict-from-now: the lags are seeded from ``input_df`` (the actual prices up
    to the forecast origin) whenever the caller supplies them — this is what lets
    the model re-forecast under MPC. Only if ``input_df`` is missing or too short
    does it fall back to the training tail frozen at fit time. (Without this the
    forecast is identical at every origin between refits, so the near-term value
    an MPC records is a flat line — see experiment_log Entry 038.)

    Args:
        state: State dict from _ar_fit.
        input_df: DataFrame whose target column carries the recent observations.
        horizon: Number of periods to forecast.

    Returns:
        Series of length `horizon`.
    """
    model = state["model"]
    lags = state["lags"]
    feature_cols = state["feature_cols"]
    target_col = state.get("target_col")

    tail = state["last_values"]
    n_needed = max(lags)
    if (input_df is not None and target_col is not None
            and target_col in getattr(input_df, "columns", [])
            and len(input_df) >= n_needed):
        tail = input_df[target_col].tail(n_needed)   # seed from prices-to-now
    recent = list(tail.values)

    last_ts = tail.index[-1]
    freq = "5min" if state["resolution"] == "5min" else "30min"
    future_idx = pd.date_range(start=last_ts, periods=horizon + 1, freq=freq)[1:]
    cal = _calendar_features(future_idx)

    forecasts = []
    for i in range(horizon):
        have_lags = len(recent) >= max(lags)
        row = {f"lag_{lag}": (recent[-lag] if have_lags else 0.0) for lag in lags}
        row.update(cal.iloc[i].to_dict())
        # One row, columns in the fitted order — the pipeline encodes calendar.
        x = pd.DataFrame([row], columns=feature_cols)
        pred = float(model.predict(x)[0])
        forecasts.append(pred)
        recent.append(pred)

    return pd.Series(forecasts, index=future_idx, name="forecast")


def _ar_save(state: dict, path: str | Path) -> None:
    """Persist AR model state.

    Args:
        state: State dict from _ar_fit.
        path: Directory to write into.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "ar_model.pkl", "wb") as f:
        pickle.dump(state["model"], f)
    state["last_values"].to_frame().to_parquet(path / "ar_last_values.parquet")
    with open(path / "ar_meta.json", "w") as f:
        json.dump({
            "lags": state["lags"],
            "feature_cols": state["feature_cols"],
            "calendar_encoding": state.get("calendar_encoding", "onehot"),
            "resolution": state["resolution"],
            "target_col": state.get("target_col"),
        }, f)


def _ar_load(path: str | Path) -> dict:
    """Restore AR model state.

    Args:
        path: Directory to read from.

    Returns:
        State dict matching what _ar_fit produces.
    """
    path = Path(path)
    with open(path / "ar_model.pkl", "rb") as f:
        model = pickle.load(f)
    last_values = pd.read_parquet(path / "ar_last_values.parquet").iloc[:, 0]
    with open(path / "ar_meta.json") as f:
        meta = json.load(f)
    return {
        "model": model,
        "lags": meta["lags"],
        "feature_cols": meta["feature_cols"],
        "calendar_encoding": meta.get("calendar_encoding", "onehot"),
        "resolution": meta["resolution"],
        "target_col": meta.get("target_col"),
        "last_values": last_values,
    }


AUTOREGRESSION = ModelSpec(
    name="autoregression",
    output="point",
    fit=_ar_fit,
    predict=_ar_predict,
    save=_ar_save,
    load=_ar_load,
    params=ARParams,
)


def main() -> None:
    """Run this module as a CLI (exposes its public callables)."""
    from grian._cli import run_module_cli

    run_module_cli(globals())


if __name__ == "__main__":
    main()
