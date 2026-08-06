"""Unified hyperparameter tuning over the typed model params, via Optuna.

One interface, several optimisers — Optuna is modern, actively maintained, and
extensible (swap the sampler, add pruning, persist to storage, resume a study).
It replaces the hand-rolled grid/random/GP strategies.

The search *space* is declared with small typed dimensions
(:class:`Float` / :class:`Int` / :class:`Categorical`) keyed by model-params
field name, so it lines up one-to-one with the frozen params classes in
:mod:`grian.models.params`. The *objective* is any ``(params: dict) -> float``
callable — decoupled from data so it is trivially testable, with
:func:`experiment_objective` wiring the real capture-ratio run.

Example:
    >>> from grian.tuning import tune, Float, Int
    >>> space = {"learning_rate": Float(1e-3, 0.3, log=True),
    ...          "n_estimators": Int(50, 600)}
    >>> study = tune(objective, space, sampler="gp", n_trials=40)
    >>> study.best_params
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

Objective = Callable[[dict[str, Any]], float]


@dataclass(frozen=True)
class Float:
    """A continuous dimension, optionally log-scaled."""

    low: float
    high: float
    log: bool = False


@dataclass(frozen=True)
class Int:
    """An integer dimension, optionally log-scaled."""

    low: int
    high: int
    log: bool = False


@dataclass(frozen=True)
class Categorical:
    """A discrete choice among fixed values."""

    choices: list[Any] = field(default_factory=list)


Dimension = Float | Int | Categorical


def _suggest(trial, name: str, dim: Dimension) -> Any:
    """Ask an Optuna trial for one value of a typed dimension."""
    if isinstance(dim, Float):
        return trial.suggest_float(name, dim.low, dim.high, log=dim.log)
    if isinstance(dim, Int):
        return trial.suggest_int(name, dim.low, dim.high, log=dim.log)
    return trial.suggest_categorical(name, dim.choices)


def _grid_search_space(space: dict[str, Dimension]) -> dict[str, list]:
    """Build an explicit grid from enumerable dimensions (for GridSampler)."""
    grid: dict[str, list] = {}
    for name, dim in space.items():
        if isinstance(dim, Categorical):
            grid[name] = list(dim.choices)
        elif isinstance(dim, Int):
            grid[name] = list(range(dim.low, dim.high + 1))
        else:
            raise ValueError(
                f"grid search needs Int/Categorical dims; {name!r} is a Float. "
                "Use sampler='random' or 'tpe'/'gp' for continuous spaces."
            )
    return grid


def _make_sampler(sampler: str, space: dict[str, Dimension], seed: int):
    """Construct an Optuna sampler by name."""
    from optuna import samplers

    if sampler == "tpe":
        return samplers.TPESampler(seed=seed)
    if sampler == "gp":
        return samplers.GPSampler(seed=seed)
    if sampler == "random":
        return samplers.RandomSampler(seed=seed)
    if sampler == "grid":
        return samplers.GridSampler(_grid_search_space(space), seed=seed)
    raise ValueError(
        f"Unknown sampler {sampler!r}. Choose tpe | gp | random | grid."
    )


def tune(
    objective: Objective,
    space: dict[str, Dimension],
    *,
    sampler: str = "tpe",
    n_trials: int = 50,
    direction: str = "maximize",
    seed: int = 42,
    study_name: str | None = None,
    storage: str | None = None,
    show_progress_bar: bool = False,
):
    """Optimise ``objective`` over ``space`` with the chosen Optuna sampler.

    Args:
        objective: ``(params) -> float`` — evaluate one hyperparameter set.
        space: Field name → :class:`Dimension` (Float/Int/Categorical).
        sampler: ``"tpe"`` (default), ``"gp"`` (Bayesian GP), ``"random"``, or
            ``"grid"``.
        n_trials: Number of evaluations (ignored/bounded by ``grid``).
        direction: ``"maximize"`` (e.g. capture ratio) or ``"minimize"``.
        seed: Seed for the sampler (reproducible search).
        study_name: Optional name; with ``storage`` enables resume/persistence.
        storage: Optional Optuna storage URL (e.g. an SQLite path).
        show_progress_bar: Show Optuna's progress bar.

    Returns:
        The completed :class:`optuna.study.Study` (``best_params``,
        ``best_value``, ``trials``).
    """
    import optuna

    def _objective(trial: optuna.Trial) -> float:
        params = {name: _suggest(trial, name, dim) for name, dim in space.items()}
        return objective(params)

    study = optuna.create_study(
        direction=direction,
        sampler=_make_sampler(sampler, space, seed),
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
    )
    study.optimize(_objective, n_trials=n_trials,
                   show_progress_bar=show_progress_bar)
    return study


def experiment_objective(
    base_config,
    data,
    oracle_daily_revenue,
    *,
    base: str = "outputs/tuning",
    metric: str = "capture_ratio",
) -> Objective:
    """Build a capture-ratio objective that tunes an :class:`ExperimentConfig`.

    Each call overrides ``base_config.model_params`` with the sampled values,
    runs the trial on ``data``, and returns the scored metric — so
    :func:`tune` can drive real dispatch backtests.

    Args:
        base_config: The :class:`~grian.experiment.ExperimentConfig` to vary.
        data: Prepared region frame (as for :meth:`ExperimentConfig.run`).
        oracle_daily_revenue: Capture denominator (perfect-foresight revenue).
        base: Root dir for per-trial artifacts.
        metric: Key of the capture report to optimise (default capture ratio).

    Returns:
        An objective callable for :func:`tune`.
    """
    def objective(params: dict[str, Any]) -> float:
        merged = {**base_config.model_params, **params}
        trial_cfg = base_config.model_copy(update={"model_params": merged})
        name = "tune_" + "_".join(f"{k}{v}" for k, v in sorted(params.items()))
        report = trial_cfg.run(data, oracle_daily_revenue, base=base,
                               trial_name=name)
        return float(report[metric])

    return objective


__all__ = [
    "Categorical", "Dimension", "Float", "Int", "Objective",
    "experiment_objective", "tune",
]


def main() -> None:
    """Run this module as a CLI (exposes its public callables)."""
    from grian._cli import run_module_cli

    run_module_cli(globals())


if __name__ == "__main__":
    main()
