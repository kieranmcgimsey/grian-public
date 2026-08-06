"""Linear forecasters: the LEAR family (Lasso) and the batched quantile-LEAR (torch)."""

from __future__ import annotations

import json
import logging
import pickle
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grian.models._shared import (
    _LINEAR_BY_NAME,
    _apply_conformal,
    _l1_penalty,
    _leading_calendar_columns,
    _linear_preprocessor,
    _make_linear_estimator,
    _periods_per_day,
    _pinball_loss,
    _quantile_weights,
)

logger = logging.getLogger(__name__)

def _linear_fit(
    train_df: pd.DataFrame, target_col: str, cfg: Mapping[str, Any]
) -> dict:
    """Fit a regularised linear model per horizon step on rich features.

    Args:
        train_df: Training DataFrame with the target (and demand) columns.
        target_col: Name of the target column.
        cfg: Trial config; model_params.estimator picks lasso/ridge/elasticnet
            (defaults from the model name), with optional include_weather.

    Returns:
        State dict with per-step fitted pipelines and feature metadata.
    """
    from sklearn.pipeline import Pipeline

    from grian.features import build_features
    from grian.models.params import LinearParams

    p = LinearParams.model_validate(cfg.get("model_params") or {})
    ppd = _periods_per_day(cfg.get("resolution", "5min"))
    horizon = cfg.get("horizon", ppd)
    resolution = cfg.get("resolution", "5min")
    kind = p.estimator or _LINEAR_BY_NAME.get(cfg.get("model", ""), "lasso")
    # Linear models one-hot the calendar by default (ordinal is wrong for a
    # linear fit); "fourier" and "ordinal" are alternatives via model_params.
    calendar_encoding = p.calendar_encoding
    include_weather = p.include_weather
    include_scarcity = p.include_scarcity
    feature_set = p.feature_set

    X = build_features(train_df, target_col, resolution,
                       include_scarcity=include_scarcity,
                       include_weather=include_weather,
                       feature_set=feature_set)
    series = train_df[target_col]
    feature_cols = list(X.columns)

    intervals_per_hour = max(1, 60 // (1440 // ppd))
    step_stride = p.step_stride if p.step_stride is not None else intervals_per_hour
    model_steps = list(range(0, horizon, step_stride))
    if model_steps[-1] != horizon - 1:
        model_steps.append(horizon - 1)

    models = {}
    for step in model_steps:
        y_step = series.shift(-step)
        mask = X.notna().all(axis=1) & y_step.notna()
        X_clean = X.loc[mask]
        y_clean = y_step.loc[mask]
        if len(X_clean) < 50:
            continue
        pipe = Pipeline([
            ("pre", _linear_preprocessor(feature_cols, calendar_encoding)),
            ("est", _make_linear_estimator(
                kind, n_alphas=p.n_alphas, max_iter=p.max_iter,
                l1_ratio=p.l1_ratio, seed=cfg.get("seed", 42))),
        ])
        pipe.fit(X_clean, y_clean.values)   # DataFrame in: named-column encoding
        models[step] = pipe

    return {
        "models": models,
        "model_steps": model_steps,
        "feature_cols": feature_cols,
        "resolution": resolution,
        "target_col": target_col,
        "estimator": kind,
        "calendar_encoding": calendar_encoding,
        "include_weather": include_weather,
        "include_scarcity": include_scarcity,
        "feature_set": feature_set,
        "train_tail": train_df.tail(max(288 * 7, 2016) + 1).copy(),
    }


def _linear_predict(state: dict, input_df: pd.DataFrame, horizon: int) -> pd.Series:
    """Predict with the per-step linear pipelines (predict-from-now).

    Args:
        state: State dict from _linear_fit.
        input_df: Data up to the forecast origin (freshest tail used).
        horizon: Number of periods to forecast.

    Returns:
        Series of length `horizon`.
    """
    from grian.features import build_features

    models = state["models"]
    tail = state["train_tail"]
    resolution = state["resolution"]
    target_col = state["target_col"]

    n_tail = len(tail)
    if (input_df is not None
            and target_col in getattr(input_df, "columns", [])
            and len(input_df) >= n_tail):
        tail = input_df.tail(n_tail)

    X = build_features(tail, target_col, resolution,
                       include_scarcity=state.get("include_scarcity", False),
                       include_weather=state.get("include_weather", False),
                       feature_set=state.get("feature_set", "full"))
    last_row = X.iloc[-1:]   # DataFrame — named columns for the encoder

    last_ts = tail.index[-1]
    freq = "5min" if resolution == "5min" else "30min"
    future_idx = pd.date_range(start=last_ts, periods=horizon + 1, freq=freq)[1:]

    predictions = np.full(horizon, np.nan)
    for step, pipe in models.items():
        if step < horizon:
            predictions[step] = float(pipe.predict(last_row)[0])

    valid = ~np.isnan(predictions)
    if valid.any():
        predictions = np.interp(
            np.arange(horizon), np.where(valid)[0], predictions[valid])
    return pd.Series(predictions, index=future_idx, name="forecast")


def _linear_save(state: dict, path: str | Path) -> None:
    """Persist linear-model state (pipelines pickled together)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "linear_models.pkl", "wb") as f:
        pickle.dump(state["models"], f)
    state["train_tail"].to_parquet(path / "train_tail.parquet")
    with open(path / "linear_meta.json", "w") as f:
        json.dump({
            "model_steps": state["model_steps"],
            "feature_cols": state["feature_cols"],
            "resolution": state["resolution"],
            "target_col": state["target_col"],
            "estimator": state["estimator"],
            "calendar_encoding": state.get("calendar_encoding", "onehot"),
            "include_weather": state.get("include_weather", False),
            "include_scarcity": state.get("include_scarcity", False),
            "feature_set": state.get("feature_set", "full"),
        }, f)


def _linear_load(path: str | Path) -> dict:
    """Restore linear-model state."""
    path = Path(path)
    with open(path / "linear_models.pkl", "rb") as f:
        models = pickle.load(f)
    with open(path / "linear_meta.json") as f:
        meta = json.load(f)
    return {
        "models": models,
        "model_steps": meta["model_steps"],
        "feature_cols": meta["feature_cols"],
        "resolution": meta["resolution"],
        "target_col": meta.get("target_col", "price"),
        "estimator": meta.get("estimator", "lasso"),
        "calendar_encoding": meta.get("calendar_encoding", "onehot"),
        "include_weather": meta.get("include_weather", False),
        "include_scarcity": meta.get("include_scarcity", False),
        "feature_set": meta.get("feature_set", "full"),
        "train_tail": pd.read_parquet(path / "train_tail.parquet"),
    }


LINEAR = {
    "name": "linear",
    "output": "point",
    "fit": _linear_fit,
    "predict": _linear_predict,
    "save": _linear_save,
    "load": _linear_load,
}


def _lear_qmean_fit(
    train_df: pd.DataFrame, target_col: str, cfg: Mapping[str, Any]
) -> dict:
    """Fit linear quantile regressors per (step, quantile) on rich features.

    .. deprecated::
        Use ``lear_qmean_torch`` (:data:`LEAR_QMEAN_TORCH`) instead. This
        sklearn implementation is **not fit for the task** on two counts: it
        solves one HiGHS LP per (step, quantile) — ~100 sequential CPU fits per
        refit, ~8 min/refit and ~10-15 h to build a single fan — and it cannot
        exempt the Fourier calendar columns from the L1 penalty (the phase-reg
        wart). The torch reimplementation fixes both: one batched
        pinball fit in seconds, Fourier block left unpenalised. Kept only to
        reproduce the pre-torch ablation and as a teaching contrast.

    Args:
        train_df: Training DataFrame (target already transformed upstream).
        target_col: Name of the target column.
        cfg: Trial config; model_params.quantiles sets levels, .alpha the L1
            strength, .include_weather toggles weather features.

    Returns:
        State dict with per-(step, tau) pipelines and metadata.
    """
    from sklearn.linear_model import QuantileRegressor
    from sklearn.pipeline import Pipeline

    from grian.features import build_features
    from grian.models.params import LearParams

    warnings.warn(
        "lear_qmean (sklearn QuantileRegressor) is deprecated: ~8 min/refit and "
        "it regularizes the Fourier calendar block. Use lear_qmean_torch instead.",
        DeprecationWarning, stacklevel=2)
    logger.warning("lear_qmean is DEPRECATED (slow CPU LPs; Fourier penalised) "
                   "— prefer lear_qmean_torch.")

    p = LearParams.model_validate(cfg.get("model_params") or {})
    ppd = _periods_per_day(cfg.get("resolution", "5min"))
    horizon = cfg.get("horizon", ppd)
    resolution = cfg.get("resolution", "5min")
    quantiles = sorted(p.quantiles)
    alpha = p.alpha
    calendar_encoding = p.calendar_encoding
    include_weather = p.include_weather

    X = build_features(train_df, target_col, resolution,
                       include_weather=include_weather)
    series = train_df[target_col]
    feature_cols = list(X.columns)

    intervals_per_hour = max(1, 60 // (1440 // ppd))
    step_stride = p.step_stride if p.step_stride is not None else intervals_per_hour
    model_steps = list(range(0, horizon, step_stride))
    if model_steps[-1] != horizon - 1:
        model_steps.append(horizon - 1)

    pipelines = {}
    for step in model_steps:
        y_step = series.shift(-step)
        mask = X.notna().all(axis=1) & y_step.notna()
        X_clean, y_clean = X.loc[mask], y_step.loc[mask].values
        if len(X_clean) < 50:
            continue
        for tau in quantiles:
            pipe = Pipeline([
                ("pre", _linear_preprocessor(feature_cols, calendar_encoding)),
                ("qr", QuantileRegressor(quantile=tau, alpha=alpha,
                                         solver="highs")),
            ])
            pipe.fit(X_clean, y_clean)
            pipelines[(step, tau)] = pipe

    return {
        "pipelines": pipelines,
        "model_steps": model_steps,
        "quantiles": quantiles,
        "resolution": resolution,
        "target_col": target_col,
        "transform": cfg.get("transform", "identity"),
        "calendar_encoding": calendar_encoding,
        "include_weather": include_weather,
        "train_tail": train_df.tail(max(288 * 7, 2016) + 1).copy(),
    }


def _lear_qmean_last_row(state, input_df):
    """Build the origin-time feature row for prediction (predict-from-now)."""
    from grian.features import build_features

    tail = state["train_tail"]
    target_col = state["target_col"]
    n_tail = len(tail)
    if (input_df is not None and target_col in getattr(input_df, "columns", [])
            and len(input_df) >= n_tail):
        tail = input_df.tail(n_tail)
    X = build_features(tail, target_col, state["resolution"],
                       include_weather=state.get("include_weather", False))
    freq = "5min" if state["resolution"] == "5min" else "30min"
    future_start = tail.index[-1]
    return X.iloc[-1:], future_start, freq   # DataFrame — named-column encoding


def _lear_qmean_predict_fan(
    state: dict, input_df: pd.DataFrame, horizon: int
) -> dict[float, np.ndarray]:
    """Return the dollar-space quantile fan from the linear quantile models."""
    from grian.evaluation.trials import _get_transform_pair

    pipelines = state["pipelines"]
    qs = sorted(state["quantiles"])
    last_row, _, _ = _lear_qmean_last_row(state, input_df)
    _, inverse_fn = _get_transform_pair(state["transform"])

    fan = {q: np.full(horizon, np.nan) for q in qs}
    for step in state["model_steps"]:
        if step >= horizon:
            continue
        if not all((step, tau) in pipelines for tau in qs):
            continue
        preds = [pipelines[(step, tau)].predict(last_row)[0] for tau in qs]
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


def _lear_qmean_predict(state: dict, input_df: pd.DataFrame, horizon: int) -> pd.Series:
    """Point forecast: integrate the fan quantiles to a dollar-space mean."""
    fan = _lear_qmean_predict_fan(state, input_df, horizon)
    qs = sorted(state["quantiles"])
    weights = _quantile_weights(qs)
    stacked = np.vstack([fan[q] for q in qs])
    mean = (weights[:, None] * stacked).sum(axis=0)
    _, future_start, freq = _lear_qmean_last_row(state, input_df)
    idx = pd.date_range(start=future_start, periods=horizon + 1, freq=freq)[1:]
    return pd.Series(mean, index=idx, name="forecast")


def _lear_qmean_save(state: dict, path: str | Path) -> None:
    """Persist quantile-LEAR state."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "lear_qmean_models.pkl", "wb") as f:
        pickle.dump(state["pipelines"], f)
    state["train_tail"].to_parquet(path / "train_tail.parquet")
    with open(path / "lear_qmean_meta.json", "w") as f:
        json.dump({
            "model_steps": state["model_steps"],
            "quantiles": state["quantiles"],
            "resolution": state["resolution"],
            "target_col": state["target_col"],
            "transform": state["transform"],
            "calendar_encoding": state.get("calendar_encoding", "onehot"),
            "include_weather": state.get("include_weather", False),
        }, f)


def _lear_qmean_load(path: str | Path) -> dict:
    """Restore quantile-LEAR state."""
    path = Path(path)
    with open(path / "lear_qmean_models.pkl", "rb") as f:
        pipelines = pickle.load(f)
    with open(path / "lear_qmean_meta.json") as f:
        meta = json.load(f)
    return {
        "pipelines": pipelines,
        "model_steps": meta["model_steps"],
        "quantiles": meta["quantiles"],
        "resolution": meta["resolution"],
        "target_col": meta.get("target_col", "price"),
        "transform": meta.get("transform", "identity"),
        "calendar_encoding": meta.get("calendar_encoding", "onehot"),
        "include_weather": meta.get("include_weather", False),
        "train_tail": pd.read_parquet(path / "train_tail.parquet"),
    }


LEAR_QMEAN = {          # DEPRECATED — see _lear_qmean_fit; use LEAR_QMEAN_TORCH
    "name": "lear_qmean",
    "output": "quantile",
    "fit": _lear_qmean_fit,
    "predict": _lear_qmean_predict,
    "predict_fan": _lear_qmean_predict_fan,
    "save": _lear_qmean_save,
    "load": _lear_qmean_load,
}


def _lear_qmean_torch_fit(
    train_df: pd.DataFrame, target_col: str, cfg: Mapping[str, Any]
) -> dict:
    """Fit the batched linear quantile fan (torch, CPU).

    A single linear layer maps preprocessed features to every
    ``(step, quantile)`` output at once, trained by full-batch Adam on the
    pinball loss with an L1 penalty (Fourier calendar block exempt). Feature
    encoding reuses :func:`_linear_preprocessor` for parity with the sklearn
    ``lear_qmean``; the standardised design matrix is winsorised for stable
    descent. Pinned to CPU — see the module note on the MPS autograd bug.

    Args:
        train_df: Training DataFrame (target already transformed upstream).
        target_col: Name of the target column.
        cfg: Trial config; ``model_params`` reads quantiles, alpha,
            calendar_encoding, include_weather, and optional epochs/lr/
            feature_clip.

    Returns:
        State dict with the fitted weight matrix, preprocessor, and metadata.
    """
    import time

    import torch

    from grian.features import build_features
    from grian.models.params import LearTorchParams

    p = LearTorchParams.model_validate(cfg.get("model_params") or {})
    ppd = _periods_per_day(cfg.get("resolution", "5min"))
    horizon = cfg.get("horizon", ppd)
    resolution = cfg.get("resolution", "5min")
    quantiles = sorted(p.quantiles)
    alpha = p.alpha
    calendar_encoding = p.calendar_encoding
    include_weather = p.include_weather
    epochs = p.epochs
    lr = p.lr
    feature_clip = p.feature_clip
    seed = cfg.get("seed", 42)

    torch.manual_seed(seed)
    np.random.seed(seed)

    X = build_features(train_df, target_col, resolution,
                       include_weather=include_weather)
    series = train_df[target_col]
    feature_cols = list(X.columns)

    intervals_per_hour = max(1, 60 // (1440 // ppd))
    step_stride = p.step_stride if p.step_stride is not None else intervals_per_hour
    model_steps = list(range(0, horizon, step_stride))
    if model_steps[-1] != horizon - 1:
        model_steps.append(horizon - 1)

    # Fit the shared preprocessor on rows with all features present, then
    # build one design matrix and a multi-step target matrix with a NaN mask
    # (samples near the series tail lack their far-horizon targets). Winsorise
    # the standardised features so heavy-tailed spike features can't wreck the
    # descent conditioning (see module note).
    feat_mask = X.notna().all(axis=1)
    pre = _linear_preprocessor(feature_cols, calendar_encoding)
    pre.fit(X.loc[feat_mask])
    idx = X.index[feat_mask]
    X_design = np.clip(pre.transform(X.loc[feat_mask]).astype(np.float32),
                       -feature_clip, feature_clip)

    Y = np.full((len(idx), len(model_steps)), np.nan, dtype=np.float32)
    for j, step in enumerate(model_steps):
        Y[:, j] = series.shift(-step).reindex(idx).to_numpy()
    Y_valid = ~np.isnan(Y)

    # Standardise targets once (globally over valid entries). Quantiles are
    # equivariant under this affine map, so we predict standardised and invert.
    y_flat = Y[Y_valid]
    y_mean = float(y_flat.mean()) if y_flat.size else 0.0
    y_std = float(y_flat.std()) or 1.0
    # Transform-space envelope: predictions are clamped here at predict time so
    # an extrapolation can never feed inf (e.g. sinh overflow) into dispatch.
    y_min = float(y_flat.min()) if y_flat.size else 0.0
    y_max = float(y_flat.max()) if y_flat.size else 0.0
    Y_std = (Y - y_mean) / y_std
    Y_std[~Y_valid] = 0.0

    n_calendar = _leading_calendar_columns(pre)

    n_features = X_design.shape[1]
    n_steps = len(model_steps)
    n_q = len(quantiles)
    n_out = n_steps * n_q

    device = torch.device("cpu")   # MPS autograd diverges here — see module note
    Xt = torch.tensor(X_design, device=device)
    Yt = torch.tensor(Y_std, device=device)
    Mt = torch.tensor(Y_valid.astype(np.float32), device=device)
    taus = torch.tensor(quantiles, dtype=torch.float32, device=device)

    layer = torch.nn.Linear(n_features, n_out).to(device)
    optimizer = torch.optim.Adam(layer.parameters(), lr=lr)

    t0 = time.time()
    layer.train()
    final_loss = float("nan")
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = layer(Xt).reshape(-1, n_steps, n_q)
        loss = _pinball_loss(pred, Yt, Mt, taus)
        penalty = _l1_penalty(layer.weight, n_calendar, alpha)
        total = loss + penalty
        total.backward()
        optimizer.step()
        final_loss = float(loss.detach())
        if epoch % 100 == 0:
            logger.debug("lear_qmean_torch epoch %d/%d pinball=%.4f",
                         epoch, epochs, final_loss)

    elapsed = time.time() - t0
    logger.info(
        "lear_qmean_torch refit: %d rows, %d feats, %d outputs, %d epochs, "
        "pinball=%.4f, %.1fs on %s",
        len(idx), n_features, n_out, epochs, final_loss, elapsed, device.type)

    return {
        "weight": layer.weight.detach().numpy().astype(np.float32),
        "bias": layer.bias.detach().numpy().astype(np.float32),
        "pre": pre,
        "n_features": n_features,
        "feature_clip": feature_clip,
        "model_steps": model_steps,
        "quantiles": quantiles,
        "y_mean": y_mean,
        "y_std": y_std,
        "y_min": y_min,
        "y_max": y_max,
        "resolution": resolution,
        "target_col": target_col,
        "transform": cfg.get("transform", "identity"),
        "calendar_encoding": calendar_encoding,
        "include_weather": include_weather,
        "train_tail": train_df.tail(max(288 * 7, 2016) + 1).copy(),
    }


def _lear_qmean_torch_predict_fan(
    state: dict, input_df: pd.DataFrame, horizon: int
) -> dict[float, np.ndarray]:
    """Return the dollar-space quantile fan from the torch linear model.

    Mirrors :func:`_lear_qmean_predict_fan`: forward the origin-time feature
    row through the linear layer, de-standardise, clamp to the training
    transform-space envelope, invert the target transform, sort quantiles per
    step, interpolate skipped steps, apply any conformal adjustment.
    """
    from grian.evaluation.trials import _get_transform_pair

    qs = sorted(state["quantiles"])
    n_q = len(qs)
    model_steps = state["model_steps"]
    last_row, _, _ = _lear_qmean_last_row(state, input_df)
    _, inverse_fn = _get_transform_pair(state["transform"])

    clip = state.get("feature_clip", 5.0)
    x = np.clip(state["pre"].transform(last_row).astype(np.float32), -clip, clip)
    logits = x @ state["weight"].T + state["bias"]           # (1, n_out)
    out = logits.reshape(len(model_steps), n_q)
    out = out * state["y_std"] + state["y_mean"]             # de-standardise
    # Guard the inverse transform: never predict outside the historical
    # transform-space envelope, so sinh (etc.) can't overflow into the LP.
    out = np.clip(out, state.get("y_min", -np.inf), state.get("y_max", np.inf))

    fan = {q: np.full(horizon, np.nan) for q in qs}
    for j, step in enumerate(model_steps):
        if step >= horizon:
            continue
        dollars = np.sort(inverse_fn(out[j].astype(float)))
        for i, q in enumerate(qs):
            fan[q][step] = dollars[i]
    for q in qs:
        arr = fan[q]
        valid = ~np.isnan(arr)
        fan[q] = (np.interp(np.arange(horizon), np.where(valid)[0], arr[valid])
                  if valid.any() else np.zeros(horizon))
    adj = state.get("conformal_adjustments")
    if adj:
        adj = {int(k): v for k, v in adj.items()}
        fan = _apply_conformal(fan, adj, qs, horizon)
    return fan


def _lear_qmean_torch_predict(
    state: dict, input_df: pd.DataFrame, horizon: int
) -> pd.Series:
    """Point forecast: integrate the fan quantiles to a dollar-space mean."""
    fan = _lear_qmean_torch_predict_fan(state, input_df, horizon)
    qs = sorted(state["quantiles"])
    weights = _quantile_weights(qs)
    stacked = np.vstack([fan[q] for q in qs])
    mean = (weights[:, None] * stacked).sum(axis=0)
    _, future_start, freq = _lear_qmean_last_row(state, input_df)
    idx = pd.date_range(start=future_start, periods=horizon + 1, freq=freq)[1:]
    return pd.Series(mean, index=idx, name="forecast")


def _lear_qmean_torch_save(state: dict, path: str | Path) -> None:
    """Persist torch quantile-LEAR state (weights + preprocessor + meta)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "lear_qmean_torch_weights.npz",
             weight=state["weight"], bias=state["bias"])
    with open(path / "lear_qmean_torch_pre.pkl", "wb") as f:
        pickle.dump(state["pre"], f)
    state["train_tail"].to_parquet(path / "train_tail.parquet")
    with open(path / "lear_qmean_torch_meta.json", "w") as f:
        json.dump({
            "n_features": state["n_features"],
            "feature_clip": state.get("feature_clip", 5.0),
            "model_steps": state["model_steps"],
            "quantiles": state["quantiles"],
            "y_mean": state["y_mean"],
            "y_std": state["y_std"],
            "y_min": state.get("y_min"),
            "y_max": state.get("y_max"),
            "resolution": state["resolution"],
            "target_col": state["target_col"],
            "transform": state["transform"],
            "calendar_encoding": state.get("calendar_encoding", "onehot"),
            "include_weather": state.get("include_weather", False),
        }, f)


def _lear_qmean_torch_load(path: str | Path) -> dict:
    """Restore torch quantile-LEAR state."""
    path = Path(path)
    weights = np.load(path / "lear_qmean_torch_weights.npz")
    with open(path / "lear_qmean_torch_pre.pkl", "rb") as f:
        pre = pickle.load(f)
    with open(path / "lear_qmean_torch_meta.json") as f:
        meta = json.load(f)
    return {
        "weight": weights["weight"],
        "bias": weights["bias"],
        "pre": pre,
        "n_features": meta["n_features"],
        "feature_clip": meta.get("feature_clip", 5.0),
        "model_steps": meta["model_steps"],
        "quantiles": meta["quantiles"],
        "y_mean": meta["y_mean"],
        "y_std": meta["y_std"],
        "y_min": meta.get("y_min"),
        "y_max": meta.get("y_max"),
        "resolution": meta["resolution"],
        "target_col": meta.get("target_col", "price"),
        "transform": meta.get("transform", "identity"),
        "calendar_encoding": meta.get("calendar_encoding", "onehot"),
        "include_weather": meta.get("include_weather", False),
        "train_tail": pd.read_parquet(path / "train_tail.parquet"),
    }


LEAR_QMEAN_TORCH = {
    "name": "lear_qmean_torch",
    "output": "quantile",
    "fit": _lear_qmean_torch_fit,
    "predict": _lear_qmean_torch_predict,
    "predict_fan": _lear_qmean_torch_predict_fan,
    "save": _lear_qmean_torch_save,
    "load": _lear_qmean_torch_load,
}


def main() -> None:
    """Run this module as a CLI (exposes its public callables)."""
    from grian._cli import run_module_cli

    run_module_cli(globals())


if __name__ == "__main__":
    main()
