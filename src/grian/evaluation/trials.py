"""Trial configuration, artifact I/O, and reproducibility tracking.

A trial is a single experiment run — one model, one config, one set of
results. The config is frozen at creation time and written to disk as
JSON so any result can be reproduced from its config.json alone.

Design:
    - Configs are plain dicts (no custom classes).
    - Every trial gets a directory under outputs/trials/{name}/{region}/.
    - Git SHA and timestamp are captured automatically.
    - Parent trial lineage supports hyperparameter search provenance.
"""

import json
import subprocess
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Target transforms — (forward, inverse) pairs
# ---------------------------------------------------------------------------

def _get_transform_pair(name: str):
    """Look up a named (forward, inverse) transform pair.

    Args:
        name: One of "asinh", "log1p", or "identity".

    Returns:
        Tuple of (forward_fn, inverse_fn).

    Raises:
        KeyError: If the transform name is not recognised.
    """
    import numpy as np

    registry = {
        "asinh": (np.arcsinh, np.sinh),
        "log1p": (np.log1p, np.expm1),
        "identity": (lambda x: x, lambda x: x),
    }
    if name not in registry:
        raise KeyError(
            f"Unknown transform {name!r}. "
            f"Expected one of {list(registry)}"
        )
    return registry[name]


# ---------------------------------------------------------------------------
# Git SHA capture
# ---------------------------------------------------------------------------

def get_git_sha() -> str:
    """Return the current git HEAD SHA, or 'unknown' if not in a repo.

    Returns:
        40-character hex SHA string, or "unknown".
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    # --- Identity ---
    "trial_name": "unnamed",
    "parent_trial": None,

    # --- Data ---
    "regions": ["SA1"],
    "resolution": "5min",
    "target_col": "price",
    "train_start": "2020-01-01",
    "train_end": "2023-06-30",
    "test_start": "2023-07-01",
    "test_end": "2024-06-30",

    # --- Model ---
    "model": "naive_similar_day",
    "model_params": {},

    # --- Forecast ---
    "horizon": 288,
    "transform": "asinh",
    "loss": "pinball",

    # --- Simulation ---
    "refit_days": 1,
    # Rolling training window in days. None keeps the expanding window (all
    # history); a fixed value (e.g. 548 ≈ 18 months) bounds refit cost and
    # drops stale-regime data.
    "train_lookback_days": None,
    "embargo": 288,
    "seed": 42,

    # --- Ablation flags ---
    # These are checked by the runner to deliberately introduce mistakes.
    # In a correct trial they should all be at their "safe" default.
    "ablations": {
        "scale_target": True,
        "use_transform": True,
        "use_embargo": True,
        "leak_future": False,
    },

    # --- Dispatch ---
    "dispatch": {
        "power_mw": 100.0,
        "duration_hours": 2.0,
        "efficiency": 0.85,
        "max_cycles": 2,
    },

    # --- MPC executor (used when the trial runs mpc.simulate_region_mpc) ---
    "mpc": {
        "resolve_every": 6,
        "reforecast_every": 12,
        "persistence_tau": 0,
        "persistence_gate": 0,
    },
}


# ---------------------------------------------------------------------------
# Config creation and freezing
# ---------------------------------------------------------------------------

def make_config(overrides: dict | None = None) -> dict[str, Any]:
    """Create a trial config by merging overrides onto the defaults.

    The result is stamped with the current git SHA and UTC timestamp.
    Nested dicts (model_params, ablations, dispatch) are merged one
    level deep — keys not present in overrides keep their defaults.

    Args:
        overrides: Dict of values to override. Keys must exist in the
            default config (typo protection).

    Returns:
        Complete config dict, ready to freeze and save.

    Raises:
        KeyError: If an override key is not in the default config.
    """
    cfg = deepcopy(DEFAULT_CONFIG)

    if overrides:
        for key, value in overrides.items():
            if key not in cfg:
                raise KeyError(
                    f"Unknown config key {key!r}. "
                    f"Valid keys: {sorted(cfg)}"
                )
            # Merge nested dicts one level deep
            if isinstance(cfg[key], dict) and isinstance(value, dict):
                cfg[key].update(value)
            else:
                cfg[key] = value

    # Stamp with provenance
    cfg["git_sha"] = get_git_sha()
    cfg["timestamp"] = datetime.now(UTC).isoformat()

    return cfg


# ---------------------------------------------------------------------------
# Trial directory management
# ---------------------------------------------------------------------------

def trial_dir(
    trial_name: str,
    region: str | None = None,
    base: str | Path = "outputs/trials",
) -> Path:
    """Return the artifact directory for a trial (and optionally a region).

    Args:
        trial_name: Name of the trial.
        region: Optional NEM region. If given, returns the region subfolder.
        base: Root directory for all trials.

    Returns:
        Path to the trial (or trial/region) directory.
    """
    path = Path(base) / trial_name
    if region:
        path = path / region
    return path


def save_config(cfg: dict, base: str | Path = "outputs/trials") -> Path:
    """Write a trial config to disk as config.json.

    Creates the trial directory if it doesn't exist.

    Args:
        cfg: Complete trial config dict.
        base: Root directory for all trials.

    Returns:
        Path to the written config.json file.
    """
    out_dir = trial_dir(cfg["trial_name"], base=base)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    return path


def load_config(trial_name: str, base: str | Path = "outputs/trials") -> dict:
    """Load a trial config from disk.

    Args:
        trial_name: Name of the trial to load.
        base: Root directory for all trials.

    Returns:
        The trial config dict.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    path = trial_dir(trial_name, base=base) / "config.json"
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Artifact save/load — parquet and JSON
# ---------------------------------------------------------------------------

def save_forecasts(
    forecasts: pd.DataFrame,
    trial_name: str,
    region: str,
    base: str | Path = "outputs/trials",
) -> Path:
    """Save all forecasts for a trial/region to parquet.

    Args:
        forecasts: DataFrame with forecast origins as rows and horizon
            steps as columns (or a long-format frame with origin + step).
        trial_name: Name of the trial.
        region: NEM region identifier.
        base: Root directory for all trials.

    Returns:
        Path to the written parquet file.
    """
    out_dir = trial_dir(trial_name, region, base)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "forecasts.parquet"
    forecasts.to_parquet(path)
    return path


def save_actuals(
    actuals: pd.DataFrame,
    trial_name: str,
    region: str,
    base: str | Path = "outputs/trials",
) -> Path:
    """Save matched actuals for a trial/region to parquet.

    Args:
        actuals: DataFrame aligned to the forecasts.
        trial_name: Name of the trial.
        region: NEM region identifier.
        base: Root directory for all trials.

    Returns:
        Path to the written parquet file.
    """
    out_dir = trial_dir(trial_name, region, base)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "actuals.parquet"
    actuals.to_parquet(path)
    return path


def save_ledger(
    ledger_df: pd.DataFrame,
    trial_name: str,
    region: str,
    base: str | Path = "outputs/trials",
) -> Path:
    """Save the trade ledger for a trial/region to parquet.

    Args:
        ledger_df: DataFrame of trade records.
        trial_name: Name of the trial.
        region: NEM region identifier.
        base: Root directory for all trials.

    Returns:
        Path to the written parquet file.
    """
    out_dir = trial_dir(trial_name, region, base)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ledger.parquet"
    ledger_df.to_parquet(path)
    return path


def save_metrics(
    metrics: dict,
    trial_name: str,
    region: str,
    base: str | Path = "outputs/trials",
) -> Path:
    """Save summary metrics for a trial/region to JSON.

    Args:
        metrics: Dict of metric name → value.
        trial_name: Name of the trial.
        region: NEM region identifier.
        base: Root directory for all trials.

    Returns:
        Path to the written metrics.json file.
    """
    out_dir = trial_dir(trial_name, region, base)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    return path


def load_metrics(
    trial_name: str,
    region: str,
    base: str | Path = "outputs/trials",
) -> dict:
    """Load summary metrics for a trial/region from JSON.

    Args:
        trial_name: Name of the trial.
        region: NEM region identifier.
        base: Root directory for all trials.

    Returns:
        Dict of metric name → value.
    """
    path = trial_dir(trial_name, region, base) / "metrics.json"
    with open(path) as f:
        return json.load(f)


def load_forecasts(
    trial_name: str,
    region: str,
    base: str | Path = "outputs/trials",
) -> pd.DataFrame:
    """Load forecasts for a trial/region from parquet.

    Args:
        trial_name: Name of the trial.
        region: NEM region identifier.
        base: Root directory for all trials.

    Returns:
        Forecast DataFrame.
    """
    path = trial_dir(trial_name, region, base) / "forecasts.parquet"
    return pd.read_parquet(path)


def load_ledger(
    trial_name: str,
    region: str,
    base: str | Path = "outputs/trials",
) -> pd.DataFrame:
    """Load the trade ledger for a trial/region from parquet.

    Args:
        trial_name: Name of the trial.
        region: NEM region identifier.
        base: Root directory for all trials.

    Returns:
        Ledger DataFrame.
    """
    path = trial_dir(trial_name, region, base) / "ledger.parquet"
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Trial discovery — scan disk for completed trials
# ---------------------------------------------------------------------------

def list_trials(base: str | Path = "outputs/trials") -> list[str]:
    """List all trial names that have a config.json on disk.

    Args:
        base: Root directory for all trials.

    Returns:
        Sorted list of trial name strings.
    """
    base = Path(base)
    if not base.exists():
        return []
    return sorted(
        d.name
        for d in base.iterdir()
        if d.is_dir() and (d / "config.json").exists()
    )


def list_regions(
    trial_name: str,
    base: str | Path = "outputs/trials",
) -> list[str]:
    """List all regions that have results for a given trial.

    Args:
        trial_name: Name of the trial.
        base: Root directory for all trials.

    Returns:
        Sorted list of region strings.
    """
    root = trial_dir(trial_name, base=base)
    if not root.exists():
        return []
    return sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and (d / "metrics.json").exists()
    )


def main() -> None:
    """Run this module as a CLI (exposes its public callables)."""
    from grian._cli import run_module_cli

    run_module_cli(globals())


if __name__ == "__main__":
    main()
