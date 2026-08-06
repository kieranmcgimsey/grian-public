"""Gradient-boosted-tree forecasters: LightGBM point, rich, and quantile models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grian.models._shared import (
    _apply_conformal,
    _build_lag_features,
    _calendar_features,
    _conformal_fan_adjustments,
    _decision_weights,
    _periods_per_day,
    _quantile_weights,
)
from grian.models.params import (
    LGBMParams,
    LGBMQMeanParams,
    LGBMRichParams,
    default_lags,
)
from grian.models.spec import ModelSpec


def _lgbm_fit(train_df: pd.DataFrame, target_col: str, cfg: Mapping[str, Any]) -> dict:
    """Fit a LightGBM model for direct multi-step forecasting.

    Each horizon step gets its own model (direct strategy), which
    avoids the error accumulation of iterative forecasting.

    Args:
        train_df: Training DataFrame with target column.
        target_col: Name of the target column.
        cfg: Trial config dict (reads model_params for LightGBM kwargs).

    Returns:
        State dict with fitted boosters (one per horizon step).
    """
    import lightgbm as lgb

    p = LGBMParams.model_validate(cfg.get("model_params") or {})
    ppd = _periods_per_day(cfg.get("resolution", "5min"))
    horizon = cfg.get("horizon", ppd)
    lags = p.lags if p.lags is not None else default_lags(ppd)

    series = train_df[target_col]
    lag_df = _build_lag_features(series, lags)
    cal_df = _calendar_features(series.index)
    X = pd.concat([lag_df, cal_df], axis=1)

    # Typed defaults live on LGBMParams; the fit only wires the config loss into
    # the LightGBM objective (pinball → median-quantile, huber → huber).
    lgb_params = p.to_lgb_kwargs()
    loss = cfg.get("loss", "pinball")
    if loss == "pinball":
        lgb_params["objective"] = "quantile"
        lgb_params["alpha"] = 0.5
    elif loss == "huber":
        lgb_params["objective"] = "huber"

    # Direct multi-step: one model per horizon step
    # For efficiency, model one step per hour and interpolate the rest.
    # At 5min resolution, 12 steps = 1 hour → ~24 models for day-ahead.
    intervals_per_hour = max(1, 60 // (1440 // ppd))
    step_stride = p.step_stride if p.step_stride is not None else intervals_per_hour
    model_steps = list(range(0, horizon, step_stride))
    if model_steps[-1] != horizon - 1:
        model_steps.append(horizon - 1)

    boosters = {}
    for step in model_steps:
        # Target is the value `step` periods ahead
        y_step = series.shift(-step)
        mask = X.notna().all(axis=1) & y_step.notna()
        X_clean = X.loc[mask]
        y_clean = y_step.loc[mask]

        if len(X_clean) < 50:
            continue

        booster = lgb.LGBMRegressor(**lgb_params)
        booster.fit(X_clean.values, y_clean.values)
        boosters[step] = booster

    return {
        "boosters": boosters,
        "model_steps": model_steps,
        "lags": lags,
        "feature_cols": list(X.columns),
        "resolution": cfg.get("resolution", "5min"),
        "lgb_params": lgb_params,
        "last_values": series.iloc[-max(lags):].copy(),
    }


def _lgbm_predict(state: dict, input_df: pd.DataFrame, horizon: int) -> pd.Series:
    """Produce a direct multi-step forecast from fitted LightGBM models.

    For horizon steps that don't have a dedicated model, linearly
    interpolates between the nearest modelled steps.

    Args:
        state: State dict from _lgbm_fit.
        input_df: Not used directly — features built from state.
        horizon: Number of periods to forecast.

    Returns:
        Series of length `horizon`.
    """
    boosters = state["boosters"]
    lags = state["lags"]
    series = state["last_values"]

    # Build features from the last available data
    lag_vals = [float(series.iloc[-lag]) if lag <= len(series) else 0.0
                for lag in lags]

    last_ts = series.index[-1]
    freq = "5min" if state["resolution"] == "5min" else "30min"
    future_idx = pd.date_range(start=last_ts, periods=horizon + 1, freq=freq)[1:]

    predictions = np.full(horizon, np.nan)

    for step, booster in boosters.items():
        if step >= horizon:
            continue
        cal = _calendar_features(future_idx[step:step + 1])
        x = np.array(lag_vals + cal.iloc[0].values.tolist()).reshape(1, -1)
        predictions[step] = float(booster.predict(x)[0])

    # Interpolate gaps between modelled steps
    valid_mask = ~np.isnan(predictions)
    if valid_mask.any():
        xp = np.where(valid_mask)[0]
        fp = predictions[valid_mask]
        predictions = np.interp(np.arange(horizon), xp, fp)

    return pd.Series(predictions, index=future_idx, name="forecast")


def _lgbm_save(state: dict, path: str | Path) -> None:
    """Persist LightGBM model state.

    Args:
        state: State dict from _lgbm_fit.
        path: Directory to write into.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save each booster as a native LightGBM text file
    for step, booster in state["boosters"].items():
        booster.booster_.save_model(str(path / f"booster_{step}.txt"))

    state["last_values"].to_frame().to_parquet(path / "lgbm_last_values.parquet")
    with open(path / "lgbm_meta.json", "w") as f:
        json.dump({
            "model_steps": state["model_steps"],
            "lags": state["lags"],
            "feature_cols": state["feature_cols"],
            "resolution": state["resolution"],
            "lgb_params": state["lgb_params"],
        }, f)


def _lgbm_load(path: str | Path) -> dict:
    """Restore LightGBM model state.

    Args:
        path: Directory to read from.

    Returns:
        State dict matching what _lgbm_fit produces.
    """
    import lightgbm as lgb

    path = Path(path)
    with open(path / "lgbm_meta.json") as f:
        meta = json.load(f)

    boosters = {}
    for step in meta["model_steps"]:
        booster_path = path / f"booster_{step}.txt"
        if booster_path.exists():
            booster = lgb.Booster(model_file=str(booster_path))
            boosters[step] = booster

    last_values = pd.read_parquet(path / "lgbm_last_values.parquet").iloc[:, 0]

    return {
        "boosters": boosters,
        "model_steps": meta["model_steps"],
        "lags": meta["lags"],
        "feature_cols": meta["feature_cols"],
        "resolution": meta["resolution"],
        "lgb_params": meta["lgb_params"],
        "last_values": last_values,
    }


LIGHTGBM = ModelSpec(
    name="lightgbm",
    output="point",
    fit=_lgbm_fit,
    predict=_lgbm_predict,
    save=_lgbm_save,
    load=_lgbm_load,
    params=LGBMParams,
)


def _lgbm_rich_fit(
    train_df: pd.DataFrame, target_col: str, cfg: Mapping[str, Any]
) -> dict:
    """Fit LightGBM with the full feature set from features.py.

    Args:
        train_df: Training DataFrame with target and demand columns.
        target_col: Name of the target column.
        cfg: Trial config dict.

    Returns:
        State dict with fitted boosters and feature metadata.
    """
    import lightgbm as lgb

    from grian.features import build_features
    p = LGBMRichParams.model_validate(cfg.get("model_params") or {})
    ppd = _periods_per_day(cfg.get("resolution", "5min"))
    horizon = cfg.get("horizon", ppd)
    resolution = cfg.get("resolution", "5min")

    X = build_features(train_df, target_col, resolution,
                       include_scarcity=p.include_scarcity,
                       include_weather=p.include_weather,
                       calendar_encoding=p.calendar_encoding)
    series = train_df[target_col]

    lgb_params = p.to_lgb_kwargs()
    loss = cfg.get("loss", "pinball")
    if loss == "pinball":
        lgb_params["objective"] = "quantile"
        lgb_params["alpha"] = 0.5
    elif loss == "huber":
        lgb_params["objective"] = "huber"

    # Decision-focused training: optional per-sample weights that up-weight
    # high-price intervals so the fit chases spike timing, not just MAE.
    weighting = p.sample_weighting
    transform = cfg.get("transform", "identity")

    intervals_per_hour = max(1, 60 // (1440 // ppd))
    step_stride = p.step_stride if p.step_stride is not None else intervals_per_hour
    model_steps = list(range(0, horizon, step_stride))
    if model_steps[-1] != horizon - 1:
        model_steps.append(horizon - 1)

    boosters = {}
    feature_cols = list(X.columns)
    for step in model_steps:
        y_step = series.shift(-step)
        mask = X.notna().all(axis=1) & y_step.notna()
        X_clean = X.loc[mask]
        y_clean = y_step.loc[mask]

        if len(X_clean) < 50:
            continue

        sample_weight = (
            _decision_weights(y_clean.values, transform, weighting)
            if weighting else None
        )
        booster = lgb.LGBMRegressor(**lgb_params)
        booster.fit(X_clean.values, y_clean.values, sample_weight=sample_weight)
        boosters[step] = booster

    return {
        "boosters": boosters,
        "model_steps": model_steps,
        "feature_cols": feature_cols,
        "resolution": resolution,
        "lgb_params": lgb_params,
        "target_col": target_col,
        "include_scarcity": p.include_scarcity,
        "include_weather": p.include_weather,
        "calendar_encoding": p.calendar_encoding,
        "train_tail": train_df.tail(max(288 * 7, 2016) + 1).copy(),
    }


def _lgbm_rich_predict(state: dict, input_df: pd.DataFrame, horizon: int) -> pd.Series:
    """Predict using LightGBM with rich features.

    Rebuilds the full feature vector from the stored training tail,
    then queries each booster for its step.

    Args:
        state: State dict from _lgbm_rich_fit.
        input_df: Data up to the forecast origin; features are built
            from its tail. Falls back to the stored fit-time tail.
        horizon: Number of periods to forecast.

    Returns:
        Series of length `horizon`.
    """
    from grian.features import build_features

    boosters = state["boosters"]
    tail = state["train_tail"]
    resolution = state["resolution"]
    target_col = state["target_col"]

    # Predict-from-now: use the freshest data the caller
    # can see, so rolling/momentum features reflect the true origin
    # instead of the last refit. Falls back to the stored tail when the
    # supplied frame is too short for the 7-day lag features.
    n_tail = len(tail)
    if (
        input_df is not None
        and target_col in getattr(input_df, "columns", [])
        and len(input_df) >= n_tail
    ):
        tail = input_df.tail(n_tail)

    X = build_features(tail, target_col, resolution,
                       include_scarcity=state.get("include_scarcity", False),
                       include_weather=state.get("include_weather", False),
                       calendar_encoding=state.get("calendar_encoding", "ordinal"))
    last_row = X.iloc[-1:]

    last_ts = tail.index[-1]
    freq = "5min" if resolution == "5min" else "30min"
    future_idx = pd.date_range(start=last_ts, periods=horizon + 1, freq=freq)[1:]

    predictions = np.full(horizon, np.nan)
    for step, booster in boosters.items():
        if step >= horizon:
            continue
        predictions[step] = float(booster.predict(last_row)[0])

    valid_mask = ~np.isnan(predictions)
    if valid_mask.any():
        xp = np.where(valid_mask)[0]
        fp = predictions[valid_mask]
        predictions = np.interp(np.arange(horizon), xp, fp)

    return pd.Series(predictions, index=future_idx, name="forecast")


def _lgbm_rich_save(state: dict, path: str | Path) -> None:
    """Persist rich LightGBM state."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for step, booster in state["boosters"].items():
        booster.booster_.save_model(str(path / f"booster_{step}.txt"))
    state["train_tail"].to_parquet(path / "train_tail.parquet")
    with open(path / "lgbm_rich_meta.json", "w") as f:
        json.dump({
            "model_steps": state["model_steps"],
            "feature_cols": state["feature_cols"],
            "resolution": state["resolution"],
            "lgb_params": state["lgb_params"],
            "target_col": state["target_col"],
            "include_scarcity": state.get("include_scarcity", False),
            "include_weather": state.get("include_weather", False),
            "calendar_encoding": state.get("calendar_encoding", "ordinal"),
        }, f)


def _lgbm_rich_load(path: str | Path) -> dict:
    """Restore rich LightGBM state."""
    import lightgbm as lgb

    path = Path(path)
    with open(path / "lgbm_rich_meta.json") as f:
        meta = json.load(f)

    boosters = {}
    for step in meta["model_steps"]:
        bp = path / f"booster_{step}.txt"
        if bp.exists():
            booster = lgb.Booster(model_file=str(bp))
            boosters[step] = booster

    train_tail = pd.read_parquet(path / "train_tail.parquet")
    return {
        "boosters": boosters,
        "model_steps": meta["model_steps"],
        "feature_cols": meta["feature_cols"],
        "resolution": meta["resolution"],
        "lgb_params": meta["lgb_params"],
        "target_col": meta.get("target_col", "price"),
        "include_scarcity": meta.get("include_scarcity", False),
        "include_weather": meta.get("include_weather", False),
        "calendar_encoding": meta.get("calendar_encoding", "ordinal"),
        "train_tail": train_tail,
    }


LIGHTGBM_RICH = ModelSpec(
    name="lightgbm_rich",
    output="point",
    fit=_lgbm_rich_fit,
    predict=_lgbm_rich_predict,
    save=_lgbm_rich_save,
    load=_lgbm_rich_load,
    params=LGBMRichParams,
)


def _lgbm_qmean_fit(
    train_df: pd.DataFrame, target_col: str, cfg: Mapping[str, Any]
) -> dict:
    """Fit LightGBM quantile boosters for dollar-space mean forecasting.

    Same rich feature set as lightgbm_rich, but one booster per
    (step, quantile level). The predict step integrates the quantiles
    to a conditional mean in dollar space — the quantity the linear
    dispatch objective actually needs.

    Args:
        train_df: Training DataFrame (target already transformed).
        target_col: Name of the target column.
        cfg: Trial config dict; model_params.quantiles sets the levels.

    Returns:
        State dict with per-step-per-quantile boosters and metadata.
    """
    import lightgbm as lgb

    from grian.features import build_features
    p = LGBMQMeanParams.model_validate(cfg.get("model_params") or {})
    ppd = _periods_per_day(cfg.get("resolution", "5min"))
    horizon = cfg.get("horizon", ppd)
    resolution = cfg.get("resolution", "5min")
    quantiles = sorted(p.quantiles)
    include_weather = p.include_weather
    calendar_encoding = p.calendar_encoding

    X = build_features(train_df, target_col, resolution,
                       include_weather=include_weather,
                       calendar_encoding=calendar_encoding)
    series = train_df[target_col]

    # Quantile boosters: shared kwargs here, per-quantile alpha set in the loop.
    lgb_params = p.to_lgb_kwargs()
    lgb_params["objective"] = "quantile"

    intervals_per_hour = max(1, 60 // (1440 // ppd))
    step_stride = p.step_stride if p.step_stride is not None else intervals_per_hour
    model_steps = list(range(0, horizon, step_stride))
    if model_steps[-1] != horizon - 1:
        model_steps.append(horizon - 1)

    # Optional split-conformal calibration: hold out a recent tail, fit the
    # boosters on the rest, and per (step, quantile) shift the fan so empirical
    # coverage meets nominal (widening under-covering quantiles). Off by
    # default. In dollar space, because that is where dispatch reads the fan —
    # and it is the upper quantiles' under-coverage of spikes we most want to
    # correct (Entry 033).
    calibrate = p.calibrate
    transform = cfg.get("transform", "identity")
    n = len(X)
    n_cal = min(p.cal_days * ppd, n // 4) if calibrate else 0
    is_cal = np.arange(n) >= (n - n_cal)

    boosters = {}
    calib_forecasts: dict = {}   # populated only when `calibrate`; empty otherwise
    for step in model_steps:
        y_step = series.shift(-step)
        base = (X.notna().all(axis=1) & y_step.notna()).values
        fit_mask = base & ~is_cal
        if fit_mask.sum() < 50:
            continue
        for tau in quantiles:
            booster = lgb.LGBMRegressor(**{**lgb_params, "alpha": tau})
            booster.fit(X.values[fit_mask], y_step.values[fit_mask])
            boosters[(step, tau)] = booster
        if calibrate:
            cal_mask = base & is_cal
            if cal_mask.sum() >= 20:
                calib_forecasts[step] = (X.values[cal_mask],
                                         y_step.values[cal_mask])

    adjustments = (
        _conformal_fan_adjustments(boosters, quantiles, model_steps,
                                   calib_forecasts, transform)
        if calibrate else None
    )

    return {
        "boosters": boosters,
        "model_steps": model_steps,
        "quantiles": quantiles,
        "resolution": resolution,
        "target_col": target_col,
        "transform": transform,
        "include_weather": include_weather,
        "calendar_encoding": calendar_encoding,
        "mean_from_step": p.mean_from_step,
        "conformal_adjustments": adjustments,
        "train_tail": train_df.tail(max(288 * 7, 2016) + 1).copy(),
    }


def _lgbm_qmean_predict(state: dict, input_df: pd.DataFrame, horizon: int) -> pd.Series:
    """Forecast the conditional mean in dollar space via quantiles.

    Per step: predict all quantile levels (in transformed space),
    invert the target transform per quantile (monotone transforms
    commute with quantiles), sort to repair crossings, and
    integrate with midpoint weights to a dollar mean. The result is
    re-forward-transformed so the runner's standard inversion recovers
    the dollar mean exactly.

    Args:
        state: State dict from _lgbm_qmean_fit.
        input_df: Data up to the forecast origin (predict-from-now).
        horizon: Number of periods to forecast.

    Returns:
        Series of length horizon (in transformed space).
    """
    from grian.evaluation.trials import _get_transform_pair
    from grian.features import build_features

    boosters = state["boosters"]
    tail = state["train_tail"]
    target_col = state["target_col"]
    quantiles = state["quantiles"]
    resolution = state["resolution"]
    # Lead-dependent blend (Entry 017): at short leads the predictive
    # distribution has collapsed and the median is the better point
    # forecast; at long leads skew dominates and the LP needs the mean.
    # Steps below mean_from_step use the median booster's output.
    mean_from_step = int(state.get("mean_from_step", 0))

    n_tail = len(tail)
    if (
        input_df is not None
        and target_col in getattr(input_df, "columns", [])
        and len(input_df) >= n_tail
    ):
        tail = input_df.tail(n_tail)

    X = build_features(tail, target_col, resolution,
                       include_weather=state.get("include_weather", False),
                       calendar_encoding=state.get("calendar_encoding", "ordinal"))
    last_row = X.iloc[-1:].values

    forward_fn, inverse_fn = _get_transform_pair(state["transform"])
    weights = _quantile_weights(quantiles)

    step_means = {}
    for step in state["model_steps"]:
        preds = {
            tau: float(boosters[(step, tau)].predict(last_row)[0])
            for tau in quantiles
            if (step, tau) in boosters
        }
        if not preds:
            continue
        if step < mean_from_step and 0.5 in preds:
            step_means[step] = float(inverse_fn(np.asarray([preds[0.5]]))[0])
        else:
            dollars = np.sort(inverse_fn(np.asarray(list(preds.values()))))
            step_means[step] = float(weights[: len(dollars)] @ dollars)

    predictions = np.full(horizon, np.nan)
    for step, mean in step_means.items():
        if step < horizon:
            predictions[step] = mean
    valid = ~np.isnan(predictions)
    if valid.any():
        xp = np.where(valid)[0]
        predictions = np.interp(np.arange(horizon), xp, predictions[valid])

    last_ts = tail.index[-1]
    freq = "5min" if resolution == "5min" else "30min"
    future_idx = pd.date_range(start=last_ts, periods=horizon + 1, freq=freq)[1:]

    # Back to transformed space so the runner's inversion is a no-op
    # round trip (it always applies inverse_fn to predict output).
    return pd.Series(forward_fn(predictions), index=future_idx, name="forecast")


def _lgbm_qmean_predict_fan(
    state: dict, input_df: pd.DataFrame, horizon: int
) -> dict[float, np.ndarray]:
    """Return the full quantile *fan* in dollar space (for probabilistic dispatch).

    Same booster evaluation as ``_lgbm_qmean_predict`` but keeps each quantile
    separate instead of integrating to a mean. Crossings are repaired by sorting.

    Args:
        state: State dict from ``_lgbm_qmean_fit``.
        input_df: Data up to the forecast origin.
        horizon: Number of periods to forecast.

    Returns:
        ``{tau: np.ndarray(horizon)}`` — dollar-space quantile forecasts.
    """
    from grian.evaluation.trials import _get_transform_pair
    from grian.features import build_features

    boosters = state["boosters"]
    tail = state["train_tail"]
    target_col = state["target_col"]
    qs = sorted(state["quantiles"])
    resolution = state["resolution"]

    n_tail = len(tail)
    if (input_df is not None and target_col in getattr(input_df, "columns", [])
            and len(input_df) >= n_tail):
        tail = input_df.tail(n_tail)

    X = build_features(tail, target_col, resolution,
                       include_weather=state.get("include_weather", False),
                       calendar_encoding=state.get("calendar_encoding", "ordinal"))
    last_row = X.iloc[-1:].values
    _, inverse_fn = _get_transform_pair(state["transform"])

    fan = {q: np.full(horizon, np.nan) for q in qs}
    for step in state["model_steps"]:
        if step >= horizon:
            continue
        preds = [boosters[(step, tau)].predict(last_row)[0]
                 for tau in qs if (step, tau) in boosters]
        if len(preds) != len(qs):
            continue
        dollars = np.sort(inverse_fn(np.asarray(preds, dtype=float)))
        for i, q in enumerate(qs):
            fan[q][step] = dollars[i]
    for q in qs:
        arr = fan[q]
        valid = ~np.isnan(arr)
        fan[q] = (np.interp(np.arange(horizon), np.where(valid)[0], arr[valid])
                  if valid.any() else np.zeros(horizon))
    adj = state.get("conformal_adjustments")
    if adj:
        # JSON round-trips integer step keys to strings — normalise back.
        adj = {int(k): v for k, v in adj.items()}
        fan = _apply_conformal(fan, adj, qs, horizon)
    return fan


def _lgbm_qmean_save(state: dict, path: str | Path) -> None:
    """Persist quantile-mean LightGBM state."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for (step, tau), booster in state["boosters"].items():
        booster.booster_.save_model(str(path / f"booster_{step}_{tau}.txt"))
    state["train_tail"].to_parquet(path / "train_tail.parquet")
    with open(path / "lgbm_qmean_meta.json", "w") as f:
        json.dump({
            "model_steps": state["model_steps"],
            "quantiles": state["quantiles"],
            "resolution": state["resolution"],
            "target_col": state["target_col"],
            "transform": state["transform"],
            "calendar_encoding": state.get("calendar_encoding", "ordinal"),
            "conformal_adjustments": state.get("conformal_adjustments"),
        }, f)


def _lgbm_qmean_load(path: str | Path) -> dict:
    """Restore quantile-mean LightGBM state."""
    import lightgbm as lgb

    path = Path(path)
    with open(path / "lgbm_qmean_meta.json") as f:
        meta = json.load(f)
    boosters = {}
    for step in meta["model_steps"]:
        for tau in meta["quantiles"]:
            bp = path / f"booster_{step}_{tau}.txt"
            if bp.exists():
                boosters[(step, tau)] = lgb.Booster(model_file=str(bp))
    train_tail = pd.read_parquet(path / "train_tail.parquet")
    return {**meta, "boosters": boosters, "train_tail": train_tail}


LIGHTGBM_QMEAN = ModelSpec(
    name="lightgbm_qmean",
    output="point",
    fit=_lgbm_qmean_fit,
    predict=_lgbm_qmean_predict,
    predict_fan=_lgbm_qmean_predict_fan,   # quantile fan for probabilistic dispatch
    save=_lgbm_qmean_save,
    load=_lgbm_qmean_load,
    params=LGBMQMeanParams,
)


def main() -> None:
    """Run this module as a CLI (exposes its public callables)."""
    from grian._cli import run_module_cli

    run_module_cli(globals())


if __name__ == "__main__":
    main()
