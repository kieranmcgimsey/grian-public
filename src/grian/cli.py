"""The ``grian`` command-line tool (Fire + Rich).

Commands:
    grian models                 List registered models.
    grian describe <model>       Show a model's typed hyperparameter schema.
    grian configs                List the example run configs.
    grian run <config.yaml>      Reproduce a full trial from a YAML config.
    grian tune <config.yaml> <space.yaml>
                                 Optimise a model's hyperparameters (Optuna).

Runs are fully reproducible: the config carries the seed and data window, and
every RNG is seeded before the trial. ``run``/``tune`` need the processed
region parquet under ``data/processed/`` (see ``scripts/build_sim_data.py``);
``models``/``describe``/``configs`` need no data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from grian._cli import console


def models() -> None:
    """List every registered model and its typed params class."""
    from rich.table import Table

    from grian.models import REGISTRY, params_for

    table = Table(title="Registered models")
    table.add_column("model", style="bold")
    table.add_column("output")
    table.add_column("params class", style="cyan")
    for name in sorted(REGISTRY):
        table.add_row(name, REGISTRY[name].get("output", "point"),
                      params_for(name).__name__)
    console().print(table)


def describe(model: str) -> None:
    """Show a model's tunable hyperparameters, types, and defaults.

    Args:
        model: A registered model name (see ``grian models``).
    """
    from rich.table import Table

    from grian.models import params_for

    cls = params_for(model)
    table = Table(title=f"{model} — {cls.__name__}")
    table.add_column("field", style="bold")
    table.add_column("type", style="cyan")
    table.add_column("default")
    for name, info in cls.model_fields.items():
        if info.default_factory is not None:      # type: ignore[truthy-function]
            default = repr(info.default_factory())
        else:
            default = repr(info.default)
        ann = getattr(info.annotation, "__name__", str(info.annotation))
        table.add_row(name, ann, default)
    console().print(table)


def configs(directory: str = "configs") -> None:
    """List the example run configs under ``directory``."""
    from grian.experiment import ExperimentConfig

    con = console()
    for path in sorted(Path(directory).glob("*.yaml")):
        try:
            cfg = ExperimentConfig.from_yaml(path)
            con.print(f"[bold]{path}[/bold]  →  {cfg.model} "
                      f"({cfg.executor}, {cfg.resolution})")
        except Exception as exc:                    # noqa: BLE001 — surface + continue
            con.print(f"[red]{path}: {exc}[/red]")


def _load_region_data(cfg) -> tuple[Any, Any]:
    """Load the processed region frame and the perfect-foresight oracle.

    Raises:
        FileNotFoundError: If the processed parquet for the region is absent.
    """
    import pandas as pd

    from grian.dispatch.oracle import compute_oracle
    from grian.models._shared import _periods_per_day

    region = cfg.regions[0]
    path = Path(f"data/processed/{region}_{cfg.resolution}_sim.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Build it with scripts/build_sim_data.py "
            f"(needs the NEMOSIS/ERA5 caches)."
        )
    data = pd.read_parquet(path)
    window = data.loc[cfg.test_start:cfg.test_end]
    dt_hours = 24.0 / _periods_per_day(cfg.resolution)
    oracle = compute_oracle(
        window[cfg.target_col], dt_hours=dt_hours,
        power_mw=cfg.dispatch.power_mw, duration_hours=cfg.dispatch.duration_hours,
        efficiency=cfg.dispatch.efficiency, max_cycles=cfg.dispatch.max_cycles,
    )
    return data, oracle["daily_revenue"]


def run(config: str, seed: int | None = None,
        base: str = "outputs/trials") -> dict:
    """Reproduce a full trial from a YAML config and print its capture report.

    Args:
        config: Path to an experiment YAML (see ``configs/``).
        seed: Optional override of the config's seed.
        base: Output directory for trial artifacts.

    Returns:
        The capture report dict.
    """
    from grian.experiment import ExperimentConfig

    cfg = ExperimentConfig.from_yaml(config)
    if seed is not None:
        cfg = cfg.model_copy(update={"seed": seed})
    con = console()
    con.print(f"[bold]Running[/bold] {cfg.model} ({cfg.executor}), seed={cfg.seed}")
    data, oracle_rev = _load_region_data(cfg)
    report = cfg.run(data, oracle_rev, base=base)
    con.print(f"[green]capture_ratio = {report['capture_ratio']:.4f}[/green]  "
              f"(${report.get('total_revenue', 0):,.0f} / "
              f"${report.get('oracle_revenue', 0):,.0f})")
    return report


def _space_from_yaml(path: str) -> dict:
    """Parse a search-space YAML into typed tuning dimensions.

    Format (per field):
        learning_rate: {type: float, low: 0.001, high: 0.3, log: true}
        n_estimators:  {type: int, low: 50, high: 600}
        calendar_encoding: {type: categorical, choices: [onehot, fourier]}
    """
    import yaml

    from grian.tuning import Categorical, Float, Int

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    space: dict[str, Any] = {}
    for field, spec in raw.items():
        kind = spec.get("type")
        if kind == "float":
            space[field] = Float(spec["low"], spec["high"], spec.get("log", False))
        elif kind == "int":
            space[field] = Int(spec["low"], spec["high"], spec.get("log", False))
        elif kind == "categorical":
            space[field] = Categorical(spec["choices"])
        else:
            raise ValueError(f"{field}: unknown dim type {kind!r}")
    return space


def tune(config: str, space: str, sampler: str = "tpe",
         n_trials: int = 50, seed: int = 42,
         base: str = "outputs/tuning") -> dict:
    """Optimise a model's hyperparameters over a search space.

    Args:
        config: Base experiment YAML — the fixed part of the run.
        space: Search-space YAML (see ``_space_from_yaml`` for the format).
        sampler: ``tpe`` (default), ``gp`` (Bayesian GP), ``random``, ``grid``.
        n_trials: Number of trials to evaluate.
        seed: Sampler seed (reproducible search).
        base: Output directory for per-trial artifacts.

    Returns:
        Dict with ``best_params`` and ``best_value`` (capture ratio).
    """
    from grian.experiment import ExperimentConfig
    from grian.tuning import experiment_objective
    from grian.tuning import tune as run_tune

    cfg = ExperimentConfig.from_yaml(config)
    search_space = _space_from_yaml(space)
    con = console()
    con.print(f"[bold]Tuning[/bold] {cfg.model} — {sampler} × {n_trials} trials "
              f"over {sorted(search_space)}")
    data, oracle_rev = _load_region_data(cfg)
    objective = experiment_objective(cfg, data, oracle_rev, base=base)
    study = run_tune(objective, search_space, sampler=sampler,
                     n_trials=n_trials, seed=seed)
    con.print(f"[green]best capture_ratio = {study.best_value:.4f}[/green]")
    con.print(f"best params: {study.best_params}")
    return {"best_params": study.best_params, "best_value": study.best_value}


def main() -> None:
    """Entry point for the ``grian`` console script."""
    import fire

    fire.Fire({
        "models": models,
        "describe": describe,
        "configs": configs,
        "run": run,
        "tune": tune,
    })


if __name__ == "__main__":
    main()
