"""Deprecated neural forecasters (simple MLP, LSTM) — kept but not competitive."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from grian.sim.models._common import (
    _build_lag_features,
    _calendar_features,
    _get_device,
    _periods_per_day,
)


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
