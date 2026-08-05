"""Shared helpers used across the model modules."""

import numpy as np
import pandas as pd


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


_CALENDAR_COLS = ("hour", "day_of_week", "month")


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


def _leading_calendar_columns(preprocessor):
    """Number of leading output columns produced by the calendar (`cal`) branch.

    ``_linear_preprocessor`` always lists its ``cal`` transformer first (Fourier
    sin/cos terms, or one-hot dummies), so the calendar block occupies the
    leading output columns. These are exempt from the L1 penalty: the calendar
    is a low-dimensional, deterministic signal that L1 should not shrink away —
    penalising the one-hot dummies flattens the intraday price shape (no
    arbitrage), and penalising sin/cos as ``|a|+|b|`` biases the harmonic phase
    L1 is reserved for selecting among the many lag/weather
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


def _get_device():
    """Pick the best available torch device (MPS > CPU).

    Returns:
        torch.device for MPS if available, else CPU.
    """
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _decision_weights(y_transformed, transform, params):
    """Per-sample fit weights that up-weight high-price (scarcity) intervals.

    Decision-focused training. Dispatch *capture* is earned in price spikes,
    but MAE / median-pinball loss is dominated by the flat 95% of intervals, so
    an accuracy-optimal forecaster smooths the spikes it most needs to call
    (Entry 030). Weighting the fit toward high-price intervals
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
