"""Model specifications for the trading simulation.

Every model is a plain dict with a fixed set of function keys:

    {
        "name":    str,
        "output":  "point" | "quantile",
        "fit":     fn(train_df, target_col, cfg) -> state,
        "predict": fn(state, input_df, horizon) -> pd.Series | pd.DataFrame,
        "save":    fn(state, path) -> None,
        "load":    fn(path) -> state,
    }

No classes, no inheritance, no registration. Adding a new model means
writing four functions and assembling them into a dict.

State is always a plain dict containing whatever the model needs to
produce forecasts — fitted weights, learned parameters, or just a
reference to the training data. The runner never inspects state
internals.

Models included:
    - naive_similar_day: Repeat same day-of-week from last week.
    - autoregression: Linear AR with configurable lag depth.
    - lightgbm: Gradient-boosted trees via LightGBM.
    - simple_mlp: Two-layer feedforward net via PyTorch.

This is the **test bench** registry (used by the walk-forward backtest, executors,
and dashboard). It is separate from the notebook *curriculum* models in
``grian.models`` — same subject, different half of the repo.
"""

import json
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ===================================================================
# Helpers shared across models
# ===================================================================

def _periods_per_day(resolution: str) -> int:
    """Number of intervals in one calendar day.

    Args:
        resolution: "5min" or "30min".

    Returns:
        288 for 5min, 48 for 30min.
    """
    return 288 if resolution == "5min" else 48


def _build_lag_features(
    series: pd.Series,
    lags: list[int],
) -> pd.DataFrame:
    """Build a DataFrame of lagged values from a single series.

    Args:
        series: The time series to lag.
        lags: List of lag periods (e.g. [288, 576, 2016] for 5min).

    Returns:
        DataFrame with one column per lag, named "lag_{n}".
    """
    parts = {f"lag_{lag}": series.shift(lag) for lag in lags}
    return pd.DataFrame(parts, index=series.index)


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Extract calendar features from a datetime index.

    Args:
        index: DatetimeIndex to extract from.

    Returns:
        DataFrame with hour, day_of_week, month columns.
    """
    return pd.DataFrame(
        {
            "hour": index.hour + index.minute / 60.0,
            "day_of_week": index.dayofweek,
            "month": index.month,
        },
        index=index,
    )


# ===================================================================
# Model 1: Naive similar-day
# ===================================================================
# Repeats the price profile from the same day-of-week one week ago.
# This is the absolute floor — every other model must beat it.

def _naive_fit(train_df, target_col, cfg):
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


def _naive_predict(state, input_df, horizon):
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
    # Predict-from-now (campaign W1.1): when the runner supplies data up
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


def _naive_save(state, path):
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


def _naive_load(path):
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


NAIVE_SIMILAR_DAY = {
    "name": "naive_similar_day",
    "output": "point",
    "fit": _naive_fit,
    "predict": _naive_predict,
    "save": _naive_save,
    "load": _naive_load,
}


# ===================================================================
# Model 2: Autoregression
# ===================================================================
# Linear regression on lagged price values. Simple, interpretable,
# and fast to fit. Uses sklearn's LinearRegression.

def _ar_fit(train_df, target_col, cfg):
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

    ppd = _periods_per_day(cfg.get("resolution", "5min"))
    default_lags = [ppd, 2 * ppd, 7 * ppd]
    params = cfg.get("model_params", {})
    lags = params.get("lags", default_lags)
    calendar_encoding = params.get("calendar_encoding", "onehot")

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


def _ar_predict(state, input_df, horizon):
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


def _ar_save(state, path):
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


def _ar_load(path):
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


AUTOREGRESSION = {
    "name": "autoregression",
    "output": "point",
    "fit": _ar_fit,
    "predict": _ar_predict,
    "save": _ar_save,
    "load": _ar_load,
}


# ===================================================================
# Regularised linear models on the rich feature set (LEAR family)
# ===================================================================
# LEAR (Lasso-Estimated AutoRegressive; Lago et al.) is the canonical
# electricity-price-forecasting benchmark: a sparse linear model on a
# large set of lagged prices, calendar terms, and exogenous features,
# with L1 regularisation selecting which lags matter. Here it shares the
# *exact* feature matrix as lightgbm_rich, so "LEAR vs LightGBM" is a
# clean linear-vs-GBM comparison on identical inputs. The same scaffold
# yields Ridge (L2) and ElasticNet (L1+L2) by swapping the estimator.
# Direct strategy: one pipeline per horizon step, like lightgbm_rich.

_LINEAR_BY_NAME = {"lear": "lasso", "ridge": "ridge", "elasticnet": "elasticnet"}


def _make_linear_estimator(kind: str, params: dict):
    """Build the per-step sklearn estimator for a linear model.

    Args:
        kind: One of "lasso" (LEAR), "ridge", "elasticnet", or "ols".
        params: model_params (reads n_alphas, l1_ratio, max_iter, seed).

    Returns:
        An unfitted sklearn regressor with regularisation chosen by CV.
    """
    import inspect

    from sklearn.linear_model import (
        ElasticNetCV,
        LassoCV,
        LinearRegression,
        RidgeCV,
    )

    n_alphas = params.get("n_alphas", 20)
    max_iter = params.get("max_iter", 2000)
    seed = params.get("seed", 42)
    # How to say "use this many alphas along the regularisation path" changed
    # across the scikit-learn versions this project supports (>=1.4,<2): up to
    # 1.6 the argument is ``n_alphas=<int>``; 1.7 removed it and now takes the
    # count as an int via ``alphas=<int>`` (older versions reject an int there).
    # Pick whichever the installed version's signature accepts, so a fresh
    # install on a newer scikit-learn doesn't break the whole LEAR family.
    if "n_alphas" in inspect.signature(LassoCV).parameters:
        n_alphas_kwarg = {"n_alphas": n_alphas}
    else:
        n_alphas_kwarg = {"alphas": n_alphas}
    if kind == "lasso":
        return LassoCV(cv=3, max_iter=max_iter, random_state=seed,
                       n_jobs=-1, **n_alphas_kwarg)
    if kind == "ridge":
        # RidgeCV takes an explicit array of alphas in every supported version.
        return RidgeCV(alphas=np.logspace(-3, 3, n_alphas))
    if kind == "elasticnet":
        return ElasticNetCV(cv=3, l1_ratio=params.get("l1_ratio", 0.5),
                            max_iter=max_iter, random_state=seed, n_jobs=-1,
                            **n_alphas_kwarg)
    return LinearRegression()


# Calendar columns produced by build_features and how linear models encode them.
# Trees split on ordinal ints fine; linear models must not read month=12 as
# "12x month=1", so we one-hot (default) or Fourier-encode the cyclic calendar.
_CALENDAR_COLS = ("hour", "day_of_week", "month")
# (period, n_harmonics) aligned with _CALENDAR_COLS for the Fourier encoding.
_FOURIER_SPEC = ((24.0, 3), (7.0, 1), (12.0, 2))


def _fourier_calendar(arr):
    """Map calendar columns to sin/cos harmonics (cyclic encoding).

    Columns must be ordered as ``_CALENDAR_COLS``; each gets the harmonics in
    ``_FOURIER_SPEC`` so that, e.g., hour 23.5 and hour 0 are adjacent. This is
    a smooth, low-dimensional alternative to one-hot for linear models.

    Args:
        arr: ``(n, k)`` array of the calendar columns (k ≤ 3).

    Returns:
        ``(n, 2·sum(harmonics))`` array of sin/cos features.
    """
    arr = np.asarray(arr, dtype=float)
    feats = []
    for j, (period, n_harm) in enumerate(_FOURIER_SPEC[: arr.shape[1]]):
        x = arr[:, j]
        for k in range(1, n_harm + 1):
            feats.append(np.sin(2 * np.pi * k * x / period))
            feats.append(np.cos(2 * np.pi * k * x / period))
    return np.column_stack(feats)


def _linear_preprocessor(feature_cols: list[str], calendar_encoding: str):
    """ColumnTransformer that encodes the calendar and scales the rest.

    Args:
        feature_cols: All columns in the feature matrix.
        calendar_encoding: "onehot" (default), "fourier", or "ordinal".

    Returns:
        An unfitted ColumnTransformer operating on named DataFrame columns.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        FunctionTransformer,
        OneHotEncoder,
        StandardScaler,
    )

    cal = [c for c in _CALENDAR_COLS if c in feature_cols]
    num = StandardScaler()
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", num)])

    if calendar_encoding == "ordinal":
        # Legacy: scale everything, keep the ordinal calendar + interaction.
        return ColumnTransformer([("num", num_pipe, list(feature_cols))],
                                 remainder="drop")

    # onehot/fourier drop the ordinal interaction (meaningless once encoded).
    other = [c for c in feature_cols if c not in cal and c != "hour_x_dow"]
    if calendar_encoding == "fourier":
        cal_tf = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("fourier", FunctionTransformer(_fourier_calendar)),
            ("scale", StandardScaler()),
        ])
    else:  # onehot
        cal_tf = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [("cal", cal_tf, cal), ("num", num_pipe, other)], remainder="drop")


def _linear_fit(train_df, target_col, cfg):
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

    from grian.sim.features import build_features

    ppd = _periods_per_day(cfg.get("resolution", "5min"))
    horizon = cfg.get("horizon", ppd)
    resolution = cfg.get("resolution", "5min")
    params = cfg.get("model_params", {})
    kind = params.get("estimator") or _LINEAR_BY_NAME.get(
        cfg.get("model", ""), "lasso")
    # Linear models one-hot the calendar by default (ordinal is wrong for a
    # linear fit); "fourier" and "ordinal" are alternatives via model_params.
    calendar_encoding = params.get("calendar_encoding", "onehot")
    include_weather = params.get("include_weather", False)
    include_scarcity = params.get("include_scarcity", False)
    feature_set = params.get("feature_set", "full")

    X = build_features(train_df, target_col, resolution,
                       include_scarcity=include_scarcity,
                       include_weather=include_weather,
                       feature_set=feature_set)
    series = train_df[target_col]
    feature_cols = list(X.columns)

    intervals_per_hour = max(1, 60 // (1440 // ppd))
    step_stride = params.get("step_stride", intervals_per_hour)
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
            ("est", _make_linear_estimator(kind, params)),
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


def _linear_predict(state, input_df, horizon):
    """Predict with the per-step linear pipelines (predict-from-now).

    Args:
        state: State dict from _linear_fit.
        input_df: Data up to the forecast origin (freshest tail used).
        horizon: Number of periods to forecast.

    Returns:
        Series of length `horizon`.
    """
    from grian.sim.features import build_features

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


def _linear_save(state, path):
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


def _linear_load(path):
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


# --- Quantile LEAR: linear quantile regression → a fan for probabilistic dispatch
# One L1-regularised quantile regressor per (step, quantile) on the rich
# features. This gives the LEAR family a full quantile fan (like lightgbm_qmean),
# so a *linear* model can also drive scenario / EV / CVaR dispatch — the linear
# counterpart to the GBM quantile model, on identical inputs.

def _lear_qmean_fit(train_df, target_col, cfg):
    """Fit linear quantile regressors per (step, quantile) on rich features.

    .. deprecated::
        Use ``lear_qmean_torch`` (:data:`LEAR_QMEAN_TORCH`) instead. This
        sklearn implementation is **not fit for the task** on two counts: it
        solves one HiGHS LP per (step, quantile) — ~100 sequential CPU fits per
        refit, ~8 min/refit and ~10-15 h to build a single fan — and it cannot
        exempt the Fourier calendar columns from the L1 penalty (the phase-reg
        wart, HANDOFF.md §4). The torch reimplementation fixes both: one batched
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

    from grian.sim.features import build_features

    warnings.warn(
        "lear_qmean (sklearn QuantileRegressor) is deprecated: ~8 min/refit and "
        "it regularizes the Fourier calendar block. Use lear_qmean_torch instead.",
        DeprecationWarning, stacklevel=2)
    logger.warning("lear_qmean is DEPRECATED (slow CPU LPs; Fourier penalised) "
                   "— prefer lear_qmean_torch. See HANDOFF.md §3-4.")

    ppd = _periods_per_day(cfg.get("resolution", "5min"))
    horizon = cfg.get("horizon", ppd)
    resolution = cfg.get("resolution", "5min")
    params = cfg.get("model_params", {})
    quantiles = sorted(params.get("quantiles", [0.05, 0.5, 0.9, 0.98]))
    alpha = params.get("alpha", 0.01)
    calendar_encoding = params.get("calendar_encoding", "onehot")
    include_weather = params.get("include_weather", False)

    X = build_features(train_df, target_col, resolution,
                       include_weather=include_weather)
    series = train_df[target_col]
    feature_cols = list(X.columns)

    intervals_per_hour = max(1, 60 // (1440 // ppd))
    step_stride = params.get("step_stride", intervals_per_hour)
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
    from grian.sim.features import build_features

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


def _lear_qmean_predict_fan(state, input_df, horizon):
    """Return the dollar-space quantile fan from the linear quantile models."""
    from grian.sim.trials import _get_transform_pair

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


def _lear_qmean_predict(state, input_df, horizon):
    """Point forecast: integrate the fan quantiles to a dollar-space mean."""
    fan = _lear_qmean_predict_fan(state, input_df, horizon)
    qs = sorted(state["quantiles"])
    weights = _quantile_weights(qs)
    stacked = np.vstack([fan[q] for q in qs])
    mean = (weights[:, None] * stacked).sum(axis=0)
    _, future_start, freq = _lear_qmean_last_row(state, input_df)
    idx = pd.date_range(start=future_start, periods=horizon + 1, freq=freq)[1:]
    return pd.Series(mean, index=idx, name="forecast")


def _lear_qmean_save(state, path):
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


def _lear_qmean_load(path):
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


# ===================================================================
# Model 2b: LEAR quantile fan (torch, CPU) — the fast reimplementation
# ===================================================================
#
# The sklearn `lear_qmean` fits one HiGHS LP per (step, quantile) — ~100
# sequential CPU solves per refit, ~10-15 h to build a full fan. This model
# collapses all of them into ONE batched gradient fit: a single linear layer
#   W : (n_features) -> (n_steps * n_quantiles)
# trained with the pinball (quantile) loss summed over every output column.
# ~13 s per refit instead of ~8 min — a ~40x speedup. The win is the BATCHING
# (one autograd fit vs 100 LPs), not the device.
#
# WHY CPU, NOT MPS. The handoff planned this for MPS. It was implemented and
# validated on MPS — and MPS gives WRONG gradients here: on the real feature
# matrix the objective *ascends* every epoch and the fit diverges (pinball
# 0.44 -> 9.9 over 300 epochs), while the identical code/seed on CPU converges
# cleanly (0.44 -> 0.15). Reproduced, isolated to the MPS autograd backend (a
# maximum-free pinball still diverges), and written up in experiment_log.md
# Entry 036. The fit is therefore pinned to CPU; a linear model is cheap enough
# there. Revisit MPS only if a future torch release fixes the backend.
#
# Three choices keep it faithful and numerically robust:
#   * Features go through the SAME `_linear_preprocessor` (one-hot/Fourier
#     calendar + standardised numerics), fit once per refit, so this and the
#     sklearn `lear_qmean` see identical inputs and the ablation stays fair.
#   * The standardised design matrix is WINSORISED to +/-`feature_clip` std.
#     The price-return features (`price_ret_*`) are heavy-tailed — a tiny
#     denominator at a spike sends them to ~200 std even after scaling, which
#     wrecks the conditioning of gradient descent (the LP solver didn't care).
#     Clipping bounds each feature's leverage; QuantileRegressor needs no such
#     guard, so this is torch-only and does not touch the sklearn model.
#   * The L1 penalty matches QuantileRegressor's `alpha * ||coef||_1` on the
#     real (lag/weather) predictors, but the whole CALENDAR block is EXEMPT —
#     left unpenalised, for either encoding. Penalising the one-hot dummies
#     shrinks the hour-of-day profile flat (a flat forecast has no arbitrage, so
#     dispatch capture collapses to ~0); penalising sin/cos as `|a|+|b|` biases
#     the harmonic phase (the "Fourier hurts LEAR" wart, HANDOFF.md §4). The
#     calendar is low-dimensional and deterministic, so L1 there buys no useful
#     sparsity — it is reserved for selecting among the many lag/weather columns.
#     (sklearn's QuantileRegressor can't exempt columns, which is why the CPU
#     `lear_qmean` still penalises its calendar and carries the wart.)


def _leading_calendar_columns(preprocessor):
    """Number of leading output columns produced by the calendar (`cal`) branch.

    ``_linear_preprocessor`` always lists its ``cal`` transformer first (Fourier
    sin/cos terms, or one-hot dummies), so the calendar block occupies the
    leading output columns. These are exempt from the L1 penalty: the calendar
    is a low-dimensional, deterministic signal that L1 should not shrink away —
    penalising the one-hot dummies flattens the intraday price shape (no
    arbitrage), and penalising sin/cos as ``|a|+|b|`` biases the harmonic phase
    (HANDOFF.md §4). L1 is reserved for selecting among the many lag/weather
    predictors. The ``ordinal`` legacy encoding has no ``cal`` branch → 0.

    Args:
        preprocessor: A *fitted* ``_linear_preprocessor`` ColumnTransformer.

    Returns:
        Count of leading calendar columns (0 if there is no ``cal`` branch).
    """
    cal = preprocessor.output_indices_.get("cal")
    return 0 if cal is None else int(cal.stop - cal.start)


def _pinball_loss(pred, y_true, mask, taus):
    """Masked, quantile-weighted pinball loss over all output columns.

    Args:
        pred: ``(B, S, Q)`` tensor of quantile predictions (standardised space).
        y_true: ``(B, S)`` tensor of standardised targets.
        mask: ``(B, S)`` boolean/float tensor, 0 where the target is missing.
        taus: ``(Q,)`` tensor of quantile levels.

    Returns:
        Scalar tensor: mean pinball loss over valid ``(sample, step, quantile)``.
    """
    import torch

    diff = y_true.unsqueeze(-1) - pred                     # (B, S, Q)
    loss = torch.maximum(taus * diff, (taus - 1.0) * diff)  # (B, S, Q)
    loss = loss * mask.unsqueeze(-1)
    denom = mask.sum() * taus.shape[0]
    return loss.sum() / torch.clamp(denom, min=1.0)


def _l1_penalty(weight, n_calendar, alpha):
    """L1 penalty on the real features, exempting the leading calendar block.

    Mirrors ``QuantileRegressor``'s ``alpha * ||coef||_1`` for every coefficient
    EXCEPT the leading ``n_calendar`` calendar columns, which are left
    unpenalised (see :func:`_leading_calendar_columns` for why). ``weight`` rows
    are the ``n_steps * n_quantiles`` outputs; each row is one regressor,
    matching the per-model sklearn penalty.

    Args:
        weight: ``(n_out, n_features)`` linear-layer weight matrix.
        n_calendar: Number of leading calendar columns to exempt (0 if none).
        alpha: L1 strength (per output, as in sklearn).

    Returns:
        Scalar tensor penalty.
    """
    if n_calendar:
        return alpha * weight[:, n_calendar:].abs().sum()
    return alpha * weight.abs().sum()


def _lear_qmean_torch_fit(train_df, target_col, cfg):
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

    from grian.sim.features import build_features

    ppd = _periods_per_day(cfg.get("resolution", "5min"))
    horizon = cfg.get("horizon", ppd)
    resolution = cfg.get("resolution", "5min")
    params = cfg.get("model_params", {})
    quantiles = sorted(params.get("quantiles", [0.05, 0.5, 0.9, 0.98]))
    alpha = params.get("alpha", 0.01)
    calendar_encoding = params.get("calendar_encoding", "onehot")
    include_weather = params.get("include_weather", False)
    epochs = params.get("epochs", 400)
    lr = params.get("lr", 0.005)
    feature_clip = params.get("feature_clip", 5.0)
    seed = cfg.get("seed", 42)

    torch.manual_seed(seed)
    np.random.seed(seed)

    X = build_features(train_df, target_col, resolution,
                       include_weather=include_weather)
    series = train_df[target_col]
    feature_cols = list(X.columns)

    intervals_per_hour = max(1, 60 // (1440 // ppd))
    step_stride = params.get("step_stride", intervals_per_hour)
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


def _lear_qmean_torch_predict_fan(state, input_df, horizon):
    """Return the dollar-space quantile fan from the torch linear model.

    Mirrors :func:`_lear_qmean_predict_fan`: forward the origin-time feature
    row through the linear layer, de-standardise, clamp to the training
    transform-space envelope, invert the target transform, sort quantiles per
    step, interpolate skipped steps, apply any conformal adjustment.
    """
    from grian.sim.trials import _get_transform_pair

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


def _lear_qmean_torch_predict(state, input_df, horizon):
    """Point forecast: integrate the fan quantiles to a dollar-space mean."""
    fan = _lear_qmean_torch_predict_fan(state, input_df, horizon)
    qs = sorted(state["quantiles"])
    weights = _quantile_weights(qs)
    stacked = np.vstack([fan[q] for q in qs])
    mean = (weights[:, None] * stacked).sum(axis=0)
    _, future_start, freq = _lear_qmean_last_row(state, input_df)
    idx = pd.date_range(start=future_start, periods=horizon + 1, freq=freq)[1:]
    return pd.Series(mean, index=idx, name="forecast")


def _lear_qmean_torch_save(state, path):
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


def _lear_qmean_torch_load(path):
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


# ===================================================================
# Model 3: LightGBM
# ===================================================================
# Gradient-boosted trees — the strong tabular baseline. Trained on
# lagged prices + calendar features. Fast to fit, handles non-linear
# relationships, and serves as the main model for ablation studies.

def _lgbm_fit(train_df, target_col, cfg):
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

    ppd = _periods_per_day(cfg.get("resolution", "5min"))
    horizon = cfg.get("horizon", ppd)
    default_lags = [ppd, 2 * ppd, 7 * ppd]
    lags = cfg.get("model_params", {}).get("lags", default_lags)

    series = train_df[target_col]
    lag_df = _build_lag_features(series, lags)
    cal_df = _calendar_features(series.index)
    X = pd.concat([lag_df, cal_df], axis=1)

    # Default LightGBM parameters — reasonable for price forecasting
    lgb_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": -1,
        "n_jobs": -1,
    }

    # Wire config loss into LightGBM objective
    loss = cfg.get("loss", "pinball")
    if loss == "pinball":
        lgb_params["objective"] = "quantile"
        lgb_params["alpha"] = 0.5
    elif loss == "huber":
        lgb_params["objective"] = "huber"
    # Override with any user-provided params (excluding our internal keys)
    user_params = {
        k: v for k, v in cfg.get("model_params", {}).items()
        if k not in ("lags",)
    }
    lgb_params.update(user_params)

    # Direct multi-step: one model per horizon step
    # For efficiency, model one step per hour and interpolate the rest.
    # At 5min resolution, 12 steps = 1 hour → ~24 models for day-ahead.
    intervals_per_hour = max(1, 60 // (1440 // ppd))
    step_stride = cfg.get("model_params", {}).get("step_stride", intervals_per_hour)
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


def _lgbm_predict(state, input_df, horizon):
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


def _lgbm_save(state, path):
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


def _lgbm_load(path):
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


LIGHTGBM = {
    "name": "lightgbm",
    "output": "point",
    "fit": _lgbm_fit,
    "predict": _lgbm_predict,
    "save": _lgbm_save,
    "load": _lgbm_load,
}


# ===================================================================
# Model 4: Simple MLP
# ===================================================================
# Two-layer feedforward network via PyTorch. Kept deliberately simple
# as a baseline for later deep learning experiments. Uses MPS on Apple
# Silicon when available, falls back to CPU.

def _get_device():
    """Pick the best available torch device (MPS > CPU).

    Returns:
        torch.device for MPS if available, else CPU.
    """
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mlp_fit(train_df, target_col, cfg):
    """Fit a two-layer MLP for direct multi-step forecasting.

    .. deprecated::
        The MLP is not maintained — it was never developed far enough to be
        competitive, and its ``predict`` still has the frozen-tail bug (it does
        not condition on ``input_df``, so it can't reforecast under MPC; see
        :func:`_ar_predict` for the fix pattern). Kept for reference only; no
        results depend on it. Use the LightGBM or LEAR families instead.

    Args:
        train_df: Training DataFrame with target column.
        target_col: Name of the target column.
        cfg: Trial config dict (reads model_params for MLP config).

    Returns:
        State dict with model weights, scaler params, and metadata.
    """
    import torch
    import torch.nn as nn

    warnings.warn(
        "simple_mlp is deprecated and unmaintained (its predict has the "
        "frozen-tail bug and can't reforecast under MPC). Use LightGBM or LEAR.",
        DeprecationWarning, stacklevel=2)

    ppd = _periods_per_day(cfg.get("resolution", "5min"))
    default_lags = [ppd, 2 * ppd, 7 * ppd]
    lags = cfg.get("model_params", {}).get("lags", default_lags)

    params = cfg.get("model_params", {})
    hidden_dim = params.get("hidden_dim", 128)
    epochs = params.get("epochs", 50)
    lr = params.get("lr", 1e-3)
    batch_size = params.get("batch_size", 256)
    seed = cfg.get("seed", 42)

    torch.manual_seed(seed)
    np.random.seed(seed)

    series = train_df[target_col]
    lag_df = _build_lag_features(series, lags)
    cal_df = _calendar_features(series.index)
    X = pd.concat([lag_df, cal_df], axis=1).dropna()
    y = series.reindex(X.index)

    # Standardise inputs
    X_mean = X.mean().values.astype(np.float32)
    X_std = X.std().values.astype(np.float32)
    X_std[X_std == 0] = 1.0
    y_mean = float(y.mean())
    y_std = float(y.std()) or 1.0

    X_norm = (X.values - X_mean) / X_std
    y_norm = (y.values - y_mean) / y_std

    device = _get_device()
    n_features = X_norm.shape[1]

    model = nn.Sequential(
        nn.Linear(n_features, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Wire config loss into PyTorch criterion
    loss_name = cfg.get("loss", "pinball")
    if loss_name == "pinball":
        loss_fn = nn.SmoothL1Loss()
    else:
        loss_fn = nn.MSELoss()

    X_t = torch.tensor(X_norm, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_norm, dtype=torch.float32, device=device).unsqueeze(1)

    model.train()
    for epoch in range(epochs):
        # Mini-batch training
        perm = torch.randperm(len(X_t), device=device)
        for start in range(0, len(X_t), batch_size):
            idx = perm[start:start + batch_size]
            pred = model(X_t[idx])
            loss = loss_fn(pred, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Move weights to CPU for serialisation
    model_cpu = model.cpu()

    return {
        "model_state_dict": model_cpu.state_dict(),
        "n_features": n_features,
        "hidden_dim": hidden_dim,
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "lags": lags,
        "resolution": cfg.get("resolution", "5min"),
        "last_values": series.iloc[-max(lags):].copy(),
    }


def _mlp_predict(state, input_df, horizon):
    """Produce an iterative forecast from the fitted MLP.

    Args:
        state: State dict from _mlp_fit.
        input_df: Not used directly — features built from state.
        horizon: Number of periods to forecast.

    Returns:
        Series of length `horizon`.
    """
    import torch
    import torch.nn as nn

    n_features = state["n_features"]
    hidden_dim = state["hidden_dim"]

    model = nn.Sequential(
        nn.Linear(n_features, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    lags = state["lags"]
    recent = list(state["last_values"].values)
    X_mean = state["X_mean"]
    X_std = state["X_std"]
    y_mean = state["y_mean"]
    y_std = state["y_std"]

    last_ts = state["last_values"].index[-1]
    freq = "5min" if state["resolution"] == "5min" else "30min"
    future_idx = pd.date_range(start=last_ts, periods=horizon + 1, freq=freq)[1:]
    cal = _calendar_features(future_idx)

    forecasts = []
    with torch.no_grad():
        for i in range(horizon):
            lag_vals = [recent[-lag] if lag <= len(recent) else 0.0
                        for lag in lags]
            cal_vals = cal.iloc[i].values.tolist()
            raw = np.array(lag_vals + cal_vals, dtype=np.float32)
            x = (raw - X_mean) / X_std
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            pred_norm = float(model(x_t)[0, 0])
            pred = pred_norm * y_std + y_mean
            forecasts.append(pred)
            recent.append(pred)

    return pd.Series(forecasts, index=future_idx, name="forecast")


def _mlp_save(state, path):
    """Persist MLP model state.

    Args:
        state: State dict from _mlp_fit.
        path: Directory to write into.
    """
    import torch

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(state["model_state_dict"], path / "mlp_weights.pt")
    state["last_values"].to_frame().to_parquet(path / "mlp_last_values.parquet")
    with open(path / "mlp_meta.json", "w") as f:
        json.dump({
            "n_features": state["n_features"],
            "hidden_dim": state["hidden_dim"],
            "X_mean": state["X_mean"].tolist(),
            "X_std": state["X_std"].tolist(),
            "y_mean": state["y_mean"],
            "y_std": state["y_std"],
            "lags": state["lags"],
            "resolution": state["resolution"],
        }, f)


def _mlp_load(path):
    """Restore MLP model state.

    Args:
        path: Directory to read from.

    Returns:
        State dict matching what _mlp_fit produces.
    """
    import torch

    path = Path(path)
    model_state_dict = torch.load(path / "mlp_weights.pt", weights_only=True)
    last_values = pd.read_parquet(path / "mlp_last_values.parquet").iloc[:, 0]
    with open(path / "mlp_meta.json") as f:
        meta = json.load(f)

    return {
        "model_state_dict": model_state_dict,
        "n_features": meta["n_features"],
        "hidden_dim": meta["hidden_dim"],
        "X_mean": np.array(meta["X_mean"], dtype=np.float32),
        "X_std": np.array(meta["X_std"], dtype=np.float32),
        "y_mean": meta["y_mean"],
        "y_std": meta["y_std"],
        "lags": meta["lags"],
        "resolution": meta["resolution"],
        "last_values": last_values,
    }


SIMPLE_MLP = {
    "name": "simple_mlp",
    "output": "point",
    "fit": _mlp_fit,
    "predict": _mlp_predict,
    "save": _mlp_save,
    "load": _mlp_load,
}


# ===================================================================
# Model 5: LightGBM with rich features
# ===================================================================
# Same gradient-boosted tree engine, but using the full feature set
# from features.py: rolling stats, demand, momentum, intra-day
# profiles. This is the "LightGBM done properly" baseline.

def _decision_weights(y_transformed, transform, params):
    """Per-sample fit weights that up-weight high-price (scarcity) intervals.

    Decision-focused training. Dispatch *capture* is earned in price spikes,
    but MAE / median-pinball loss is dominated by the flat 95% of intervals, so
    an accuracy-optimal forecaster smooths the spikes it most needs to call
    (HANDOFF §0 / Entry 030). Weighting the fit toward high-price intervals
    trades a little average accuracy for better spike timing — the thing
    capture actually rewards.

    Weights are computed in dollar space (the target is inverted out of its
    modelling transform first), so a config is comparable across transforms.

    Args:
        y_transformed: Target values in modelling space (e.g. asinh dollars).
        transform: Name of the target transform ("asinh", "log1p", "identity").
        params: ``sample_weighting`` sub-dict. Keys:
            ``scheme`` — "magnitude" (bounded log-dollar ramp, default) or
            "quantile" (flat boost above a price percentile);
            ``strength`` — weight gain (default 1.0);
            ``scale`` — dollar scale for the magnitude ramp (default 300.0);
            ``q`` — percentile cut for the quantile scheme (default 0.9).

    Returns:
        1-D float array of non-negative weights, mean-normalised to 1.0 so the
        effective learning rate is unchanged versus an unweighted fit.
    """
    from grian.sim.trials import _get_transform_pair

    _, inverse_fn = _get_transform_pair(transform)
    dollars = np.asarray(inverse_fn(np.asarray(y_transformed, dtype=float)),
                         dtype=float)
    scheme = params.get("scheme", "magnitude")
    strength = float(params.get("strength", 1.0))

    if scheme == "quantile":
        thresh = float(np.quantile(dollars, float(params.get("q", 0.9))))
        w = np.where(dollars >= thresh, 1.0 + strength, 1.0)
    else:  # "magnitude": gentle, bounded ramp in dollar space
        scale = float(params.get("scale", 300.0))
        w = 1.0 + strength * np.log1p(np.maximum(0.0, dollars) / scale)

    w = np.clip(w, 1e-6, None)
    mean = float(w.mean()) or 1.0
    return w / mean


def _lgbm_rich_fit(train_df, target_col, cfg):
    """Fit LightGBM with the full feature set from features.py.

    Args:
        train_df: Training DataFrame with target and demand columns.
        target_col: Name of the target column.
        cfg: Trial config dict.

    Returns:
        State dict with fitted boosters and feature metadata.
    """
    import lightgbm as lgb

    from grian.sim.features import build_features

    ppd = _periods_per_day(cfg.get("resolution", "5min"))
    horizon = cfg.get("horizon", ppd)
    resolution = cfg.get("resolution", "5min")
    include_scarcity = cfg.get("model_params", {}).get(
        "include_scarcity", False
    )
    include_weather = cfg.get("model_params", {}).get("include_weather", False)
    calendar_encoding = cfg.get("model_params", {}).get(
        "calendar_encoding", "ordinal")

    X = build_features(train_df, target_col, resolution,
                       include_scarcity=include_scarcity,
                       include_weather=include_weather,
                       calendar_encoding=calendar_encoding)
    series = train_df[target_col]

    lgb_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": -1,
        "n_jobs": -1,
    }

    loss = cfg.get("loss", "pinball")
    if loss == "pinball":
        lgb_params["objective"] = "quantile"
        lgb_params["alpha"] = 0.5
    elif loss == "huber":
        lgb_params["objective"] = "huber"

    _non_lgb = ("lags", "step_stride", "include_scarcity", "include_weather",
                "sample_weighting")
    user_params = {
        k: v for k, v in cfg.get("model_params", {}).items()
        if k not in _non_lgb
    }
    lgb_params.update(user_params)

    # Decision-focused training: optional per-sample weights that up-weight
    # high-price intervals so the fit chases spike timing, not just MAE.
    weighting = cfg.get("model_params", {}).get("sample_weighting")
    transform = cfg.get("transform", "identity")

    intervals_per_hour = max(1, 60 // (1440 // ppd))
    step_stride = cfg.get("model_params", {}).get("step_stride", intervals_per_hour)
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
        "include_scarcity": include_scarcity,
        "include_weather": include_weather,
        "calendar_encoding": calendar_encoding,
        "train_tail": train_df.tail(max(288 * 7, 2016) + 1).copy(),
    }


def _lgbm_rich_predict(state, input_df, horizon):
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
    from grian.sim.features import build_features

    boosters = state["boosters"]
    tail = state["train_tail"]
    resolution = state["resolution"]
    target_col = state["target_col"]

    # Predict-from-now (campaign W1.1): use the freshest data the caller
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


def _lgbm_rich_save(state, path):
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


def _lgbm_rich_load(path):
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


def _quantile_weights(taus):
    """Integration weights so that sum(w * q_tau) approximates the mean.

    Uses the piecewise-constant quantile-function approximation of
    E[X] = integral of Q(tau) over [0, 1]: each quantile represents the
    probability mass between the midpoints of its neighbouring levels.

    Args:
        taus: Sorted quantile levels in (0, 1).

    Returns:
        Array of weights summing to 1.
    """
    taus = np.asarray(taus, dtype=float)
    mids = (taus[1:] + taus[:-1]) / 2.0
    edges = np.concatenate([[0.0], mids, [1.0]])
    return np.diff(edges)


def _lgbm_qmean_fit(train_df, target_col, cfg):
    """Fit LightGBM quantile boosters for dollar-space mean forecasting.

    Same rich feature set as lightgbm_rich, but one booster per
    (step, quantile level). The predict step integrates the quantiles
    to a conditional mean in dollar space — the quantity the linear
    dispatch objective actually needs (campaign plan §3.2, trap T3).

    Args:
        train_df: Training DataFrame (target already transformed).
        target_col: Name of the target column.
        cfg: Trial config dict; model_params.quantiles sets the levels.

    Returns:
        State dict with per-step-per-quantile boosters and metadata.
    """
    import lightgbm as lgb

    from grian.sim.features import build_features

    ppd = _periods_per_day(cfg.get("resolution", "5min"))
    horizon = cfg.get("horizon", ppd)
    resolution = cfg.get("resolution", "5min")
    params = cfg.get("model_params", {})
    quantiles = sorted(params.get("quantiles", [0.05, 0.5, 0.9, 0.98]))
    include_weather = params.get("include_weather", False)
    calendar_encoding = params.get("calendar_encoding", "ordinal")

    X = build_features(train_df, target_col, resolution,
                       include_weather=include_weather,
                       calendar_encoding=calendar_encoding)
    series = train_df[target_col]

    lgb_params = {
        "n_estimators": params.get("n_estimators", 150),
        "learning_rate": params.get("learning_rate", 0.05),
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": -1,
        "n_jobs": -1,
        "objective": "quantile",
    }

    intervals_per_hour = max(1, 60 // (1440 // ppd))
    step_stride = params.get("step_stride", intervals_per_hour)
    model_steps = list(range(0, horizon, step_stride))
    if model_steps[-1] != horizon - 1:
        model_steps.append(horizon - 1)

    # Optional split-conformal calibration: hold out a recent tail, fit the
    # boosters on the rest, and per (step, quantile) shift the fan so empirical
    # coverage meets nominal (widening under-covering quantiles). Off by
    # default. In dollar space, because that is where dispatch reads the fan —
    # and it is the upper quantiles' under-coverage of spikes we most want to
    # correct (Entry 033).
    calibrate = params.get("calibrate", False)
    transform = cfg.get("transform", "identity")
    n = len(X)
    n_cal = min(int(params.get("cal_days", 28)) * ppd, n // 4) if calibrate else 0
    is_cal = np.arange(n) >= (n - n_cal)

    boosters = {}
    calib_forecasts = {} if calibrate else None
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
        "mean_from_step": params.get("mean_from_step", 0),
        "conformal_adjustments": adjustments,
        "train_tail": train_df.tail(max(288 * 7, 2016) + 1).copy(),
    }


def _conformal_fan_adjustments(boosters, quantiles, model_steps,
                               calib_forecasts, transform):
    """Per-step conformal shifts (dollar space) for a quantile fan.

    Split-conformal (Entry 033): for each modelled step, predict the held-out
    calibration rows at every quantile, invert to dollars, and use
    :class:`ConformalWrapper` to size the shift that brings each quantile to
    nominal coverage. Reused at predict time by ``_apply_conformal``.

    Args:
        boosters: ``{(step, tau): booster}`` fitted on the non-calibration rows.
        quantiles: Sorted quantile levels.
        model_steps: Steps that have boosters.
        calib_forecasts: ``{step: (X_cal, y_cal_transformed)}``.
        transform: Target transform name (for inversion to dollars).

    Returns:
        ``{step: list[float]}`` — one dollar shift per quantile, or ``{}``.
    """
    from grian.models.conformal import ConformalWrapper
    from grian.sim.trials import _get_transform_pair

    _, inverse_fn = _get_transform_pair(transform)
    adjustments = {}
    for step in model_steps:
        if step not in (calib_forecasts or {}):
            continue
        x_cal, y_cal = calib_forecasts[step]
        preds = [inverse_fn(boosters[(step, tau)].predict(x_cal))
                 for tau in quantiles if (step, tau) in boosters]
        if len(preds) != len(quantiles):
            continue
        qf = np.sort(np.column_stack(preds), axis=1)     # (n_cal, nq), dollars
        actual = inverse_fn(np.asarray(y_cal, dtype=float))
        cw = ConformalWrapper(list(quantiles)).calibrate(qf, actual)
        adjustments[step] = [float(a) for a in cw.adjustments_]
    return adjustments


def _apply_conformal(fan, adjustments, quantiles, horizon):
    """Shift a dollar-space fan by stored conformal adjustments in place.

    Lower quantiles shift down, upper quantiles shift up (widening toward
    nominal coverage); each step column is re-sorted to stay monotone.

    Args:
        fan: ``{tau: np.ndarray(horizon)}`` dollar-space fan.
        adjustments: ``{step: list[float]}`` from ``_conformal_fan_adjustments``.
        quantiles: Sorted quantile levels (aligned to the adjustment lists).
        horizon: Forecast length.

    Returns:
        The adjusted fan (same dict, mutated).
    """
    if not adjustments:
        return fan
    qs = list(quantiles)
    for step, adj in adjustments.items():
        if step >= horizon or len(adj) != len(qs):
            continue
        col = np.array([fan[q][step] for q in qs], dtype=float)
        for i, q in enumerate(qs):
            col[i] += (-adj[i] if q <= 0.5 else adj[i])
        col.sort()
        for i, q in enumerate(qs):
            fan[q][step] = col[i]
    return fan


def _lgbm_qmean_predict(state, input_df, horizon):
    """Forecast the conditional mean in dollar space via quantiles.

    Per step: predict all quantile levels (in transformed space),
    invert the target transform per quantile (monotone transforms
    commute with quantiles), sort to repair crossings (trap T8), and
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
    from grian.sim.features import build_features
    from grian.sim.trials import _get_transform_pair

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


def _lgbm_qmean_predict_fan(state, input_df, horizon):
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
    from grian.sim.features import build_features
    from grian.sim.trials import _get_transform_pair

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


def _lgbm_qmean_save(state, path):
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


def _lgbm_qmean_load(path):
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


LIGHTGBM_QMEAN = {
    "name": "lightgbm_qmean",
    "output": "point",
    "fit": _lgbm_qmean_fit,
    "predict": _lgbm_qmean_predict,
    "predict_fan": _lgbm_qmean_predict_fan,   # quantile fan for probabilistic dispatch
    "save": _lgbm_qmean_save,
    "load": _lgbm_qmean_load,
}


LIGHTGBM_RICH = {
    "name": "lightgbm_rich",
    "output": "point",
    "fit": _lgbm_rich_fit,
    "predict": _lgbm_rich_predict,
    "save": _lgbm_rich_save,
    "load": _lgbm_rich_load,
}


# ===================================================================
# Model 6: LSTM
# ===================================================================
# Sequence model that processes the last N intervals as a window.
# Captures temporal dependencies that feedforward models miss.
# Uses MPS on Apple Silicon, falls back to CPU elsewhere.

def _lstm_fit(train_df, target_col, cfg):
    """Fit an LSTM on windowed price sequences.

    .. deprecated::
        The LSTM is not maintained — it was never developed far enough to be
        competitive, and its ``predict`` still has the frozen-tail bug (it does
        not condition on ``input_df``, so it can't reforecast under MPC; see
        :func:`_ar_predict` for the fix pattern). Kept for reference only; no
        results depend on it. Use the LightGBM or LEAR families instead.

    Args:
        train_df: Training DataFrame with target column.
        target_col: Name of the target column.
        cfg: Trial config dict.

    Returns:
        State dict with model weights and normalization params.
    """
    import torch
    import torch.nn as nn

    warnings.warn(
        "lstm is deprecated and unmaintained (its predict has the frozen-tail "
        "bug and can't reforecast under MPC). Use LightGBM or LEAR.",
        DeprecationWarning, stacklevel=2)

    params = cfg.get("model_params", {})
    seq_len = params.get("seq_len", 288)
    hidden_dim = params.get("hidden_dim", 64)
    num_layers = params.get("num_layers", 2)
    epochs = params.get("epochs", 30)
    lr = params.get("lr", 0.001)
    batch_size = params.get("batch_size", 256)
    dropout = params.get("dropout", 0.1)
    seed = cfg.get("seed", 42)

    torch.manual_seed(seed)
    np.random.seed(seed)

    series = train_df[target_col].values.astype(np.float32)

    # Standardise
    y_mean = float(np.nanmean(series))
    y_std = float(np.nanstd(series)) or 1.0
    normed = (series - y_mean) / y_std

    # Build supervised windows: X[i] = normed[i:i+seq_len], y[i] = normed[i+seq_len]
    n = len(normed) - seq_len
    if n < 100:
        return {"empty": True}

    X_all = np.stack([normed[i:i + seq_len] for i in range(n)])
    y_all = normed[seq_len:seq_len + n]

    device = _get_device()

    model = nn.LSTM(
        input_size=1,
        hidden_size=hidden_dim,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout if num_layers > 1 else 0.0,
    )
    head = nn.Linear(hidden_dim, 1)
    model.to(device)
    head.to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=lr,
    )
    loss_name = cfg.get("loss", "pinball")
    loss_fn = nn.SmoothL1Loss() if loss_name == "pinball" else nn.MSELoss()

    X_t = torch.tensor(X_all, dtype=torch.float32).unsqueeze(-1)
    y_t = torch.tensor(y_all, dtype=torch.float32).unsqueeze(-1)

    model.train()
    head.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X_t))
        for start in range(0, len(X_t), batch_size):
            idx = perm[start:start + batch_size]
            xb = X_t[idx].to(device)
            yb = y_t[idx].to(device)
            out, (h_n, _) = model(xb)
            pred = head(out[:, -1, :])
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model_cpu = model.cpu()
    head_cpu = head.cpu()

    ppd = _periods_per_day(cfg.get("resolution", "5min"))
    default_lags = [ppd, 2 * ppd, 7 * ppd]
    lags = params.get("lags", default_lags)

    return {
        "lstm_state_dict": model_cpu.state_dict(),
        "head_state_dict": head_cpu.state_dict(),
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "seq_len": seq_len,
        "y_mean": y_mean,
        "y_std": y_std,
        "resolution": cfg.get("resolution", "5min"),
        "last_values": pd.Series(
            train_df[target_col].iloc[-max(seq_len, max(lags)):].values,
            index=train_df.index[-max(seq_len, max(lags)):],
            name=target_col,
        ),
        "lags": lags,
    }


def _lstm_predict(state, input_df, horizon):
    """Produce an iterative LSTM forecast.

    Feeds the last `seq_len` values through the LSTM, gets one
    prediction, appends it, and slides the window forward.

    Args:
        state: State dict from _lstm_fit.
        input_df: Not used — features from state.
        horizon: Number of periods to forecast.

    Returns:
        Series of length `horizon`.
    """
    if state.get("empty"):
        return pd.Series(np.zeros(horizon), name="forecast")

    import torch
    import torch.nn as nn

    seq_len = state["seq_len"]
    hidden_dim = state["hidden_dim"]
    num_layers = state["num_layers"]
    dropout = state["dropout"]
    y_mean = state["y_mean"]
    y_std = state["y_std"]

    model = nn.LSTM(
        input_size=1, hidden_size=hidden_dim, num_layers=num_layers,
        batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
    )
    head = nn.Linear(hidden_dim, 1)
    model.load_state_dict(state["lstm_state_dict"])
    head.load_state_dict(state["head_state_dict"])
    model.eval()
    head.eval()

    recent = list(state["last_values"].values[-seq_len:])
    normed = [(v - y_mean) / y_std for v in recent]

    last_ts = state["last_values"].index[-1]
    freq = "5min" if state["resolution"] == "5min" else "30min"
    future_idx = pd.date_range(start=last_ts, periods=horizon + 1, freq=freq)[1:]

    forecasts = []
    with torch.no_grad():
        for _ in range(horizon):
            window = np.array(normed[-seq_len:], dtype=np.float32)
            x_t = torch.tensor(window).unsqueeze(0).unsqueeze(-1)
            out, _ = model(x_t)
            pred_norm = float(head(out[:, -1, :])[0, 0])
            pred = pred_norm * y_std + y_mean
            forecasts.append(pred)
            normed.append(pred_norm)

    return pd.Series(forecasts, index=future_idx, name="forecast")


def _lstm_save(state, path):
    """Persist LSTM state."""
    import torch

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if not state.get("empty"):
        torch.save(state["lstm_state_dict"], path / "lstm_weights.pt")
        torch.save(state["head_state_dict"], path / "lstm_head.pt")
        state["last_values"].to_frame().to_parquet(path / "lstm_last_values.parquet")
    with open(path / "lstm_meta.json", "w") as f:
        json.dump({
            "hidden_dim": state.get("hidden_dim", 64),
            "num_layers": state.get("num_layers", 2),
            "dropout": state.get("dropout", 0.1),
            "seq_len": state.get("seq_len", 288),
            "y_mean": state.get("y_mean", 0.0),
            "y_std": state.get("y_std", 1.0),
            "resolution": state.get("resolution", "5min"),
            "lags": state.get("lags", [288, 576, 2016]),
            "empty": state.get("empty", False),
        }, f)


def _lstm_load(path):
    """Restore LSTM state."""
    import torch

    path = Path(path)
    with open(path / "lstm_meta.json") as f:
        meta = json.load(f)

    if meta.get("empty"):
        return {"empty": True}

    return {
        "lstm_state_dict": torch.load(path / "lstm_weights.pt", weights_only=True),
        "head_state_dict": torch.load(path / "lstm_head.pt", weights_only=True),
        "hidden_dim": meta["hidden_dim"],
        "num_layers": meta["num_layers"],
        "dropout": meta["dropout"],
        "seq_len": meta["seq_len"],
        "y_mean": meta["y_mean"],
        "y_std": meta["y_std"],
        "resolution": meta["resolution"],
        "lags": meta["lags"],
        "last_values": pd.read_parquet(path / "lstm_last_values.parquet").iloc[:, 0],
    }


LSTM = {
    "name": "lstm",
    "output": "point",
    "fit": _lstm_fit,
    "predict": _lstm_predict,
    "save": _lstm_save,
    "load": _lstm_load,
}


# ===================================================================
# Model registry
# ===================================================================
# Look up a model spec dict by name string. The runner uses this to
# resolve the "model" field in the trial config.

REGISTRY: dict[str, dict] = {
    "naive_similar_day": NAIVE_SIMILAR_DAY,
    "autoregression": AUTOREGRESSION,          # one-hot calendar (default)
    "autoregression_fourier": AUTOREGRESSION,  # Fourier calendar via model_params
    "autoregression_ordinal": AUTOREGRESSION,  # legacy ordinal (for comparison)
    # LEAR family — regularised linear on the rich feature set (estimator
    # inferred from the name; see _LINEAR_BY_NAME).
    "lear": LINEAR,                    # one-hot calendar (default)
    "lear_weather": LINEAR,            # weather on via model_params.include_weather
    "lear_fourier": LINEAR,            # Fourier (cyclic) calendar via model_params
    "lear_weather_fourier": LINEAR,    # weather + Fourier (full ablation cell)
    "lear_ordinal": LINEAR,            # legacy ordinal calendar (for comparison)
    "ridge": LINEAR,
    "elasticnet": LINEAR,
    # Shape-preserving linear models: raw price-lag basis + calendar, no
    # mean-reverting smoothers (feature_set="lean"). Tests whether LEAR's
    # sub-AR capture is the smoothers flattening peaks (Entry 032).
    "lear_lean": LINEAR,               # Lasso on the lean basis
    "ols_lean": LINEAR,                # OLS on the lean basis (richer AR)
    # DEPRECATED sklearn quantile LEAR — impractical (~8 min/refit) and it
    # regularizes the Fourier block. Superseded by lear_qmean_torch below; kept
    # only to reproduce the pre-torch ablation. See _lear_qmean_fit docstring.
    "lear_qmean": LEAR_QMEAN,          # linear quantile fan (probabilistic LEAR)
    "lear_qmean_weather": LEAR_QMEAN,
    "lear_qmean_fourier": LEAR_QMEAN,
    "lear_qmean_weather_fourier": LEAR_QMEAN,   # weather + Fourier quantile LEAR
    # Torch (CPU) reimplementation of the quantile fan: one batched pinball fit
    # instead of ~100 CPU LPs, with the Fourier calendar block left unpenalised.
    # ~40x faster and correct — the canonical LEAR quantile model going forward,
    # and what fills the LEAR-quantile ablation cells (HANDOFF.md §3-4). Pinned
    # to CPU: the planned MPS backend diverges (Entry 036).
    "lear_qmean_torch": LEAR_QMEAN_TORCH,
    "lear_qmean_torch_weather": LEAR_QMEAN_TORCH,
    "lear_qmean_torch_fourier": LEAR_QMEAN_TORCH,
    "lear_qmean_torch_weather_fourier": LEAR_QMEAN_TORCH,
    "lightgbm": LIGHTGBM,
    "lightgbm_rich": LIGHTGBM_RICH,
    # Same spec as lightgbm_rich; weather features are switched on via
    # model_params.include_weather (see scripts/run_common_eval.py).
    "lightgbm_rich_weather": LIGHTGBM_RICH,
    # Fourier (cyclic sin/cos) calendar features for trees via
    # model_params.calendar_encoding — the tree side of the calendar ablation.
    "lightgbm_rich_fourier": LIGHTGBM_RICH,
    "lightgbm_rich_weather_fourier": LIGHTGBM_RICH,
    # Spike-precursor (scarcity) features on via model_params.include_scarcity —
    # the anticipating forecast meant to break the capture ceiling (Entry 030).
    "lightgbm_rich_scarcity": LIGHTGBM_RICH,
    # Decision-focused loss: scarcity features + high-price sample weighting
    # (model_params.sample_weighting). Attacks the accuracy-vs-capture gap.
    "lightgbm_rich_dfl": LIGHTGBM_RICH,
    "lightgbm_qmean": LIGHTGBM_QMEAN,
    # Same spec; weather on via model_params.include_weather.
    "lightgbm_qmean_weather": LIGHTGBM_QMEAN,
    "lightgbm_qmean_fourier": LIGHTGBM_QMEAN,          # Fourier-calendar quantile GBM
    "lightgbm_qmean_weather_fourier": LIGHTGBM_QMEAN,  # weather + Fourier
    # Split-conformal calibrated fan (model_params.calibrate) — widens the
    # under-covering upper quantiles so scenario/CVaR dispatch sees spike risk
    # the raw fan misses (Entry 033).
    "lightgbm_qmean_cal": LIGHTGBM_QMEAN,
    # DEPRECATED — never developed enough to be competitive; their predict has
    # the frozen-tail bug (can't reforecast under MPC) and is left unfixed. No
    # results depend on them (old runs are parked in outputs/_archive_valtest_trials/).
    # Registered so the code still loads/tests; see the fit docstrings.
    "simple_mlp": SIMPLE_MLP,   # deprecated
    "lstm": LSTM,               # deprecated
}


def get_model(name: str) -> dict:
    """Look up a model spec by name.

    Args:
        name: Model name string (must be a key in REGISTRY).

    Returns:
        Model spec dict with fit/predict/save/load functions.

    Raises:
        KeyError: If the model name is not registered.
    """
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown model {name!r}. "
            f"Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]
