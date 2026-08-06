"""On-disk hyperparameter search: run every candidate as a full, saved trial.

This module's niche is the *artifact* search driver — :func:`run_search`
evaluates each point as a complete trial with its own ``config.json`` and
ledger on disk, nothing ephemeral. For fast, in-memory tuning over the typed
model params (with a shared objective) prefer :mod:`grian.tuning`, the unified
Optuna interface.

Each strategy is a plain pull function ``fn(space, past_results) -> params | None``:
    grid_strategy    — Exhaustive grid over discrete values.
    random_strategy  — Random sampling from continuous/discrete ranges.
    bayesian_strategy — Optuna Gaussian-process (GP) optimisation.

The Bayesian strategy delegates to Optuna's ``GPSampler`` rather than a
hand-rolled sklearn GP — it handles categoricals and the acquisition properly.
The driver (:func:`run_search`) asks the strategy for the next point, runs a
trial, records the result, and repeats until the strategy returns None.
"""

import itertools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from grian.dispatch import open_loop as runner_mod
from grian.evaluation import trials as trials_mod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search space specification
# ---------------------------------------------------------------------------
# A search space is a dict mapping parameter names to either:
#   - A list of discrete values:      {"n_estimators": [100, 300, 500]}
#   - A (min, max) tuple for floats:   {"lr": (0.001, 0.3)}
#   - A (min, max, "log") tuple:       {"lr": (0.001, 0.3, "log")}
#
# Parameter names use dot notation for nesting:
#   {"model_params.lr": (0.001, 0.3)}
# The driver unpacks these into the trial config.


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dot-separated keys.

    Args:
        d: Dict to modify in place.
        dotted_key: Dot-separated key path (e.g. "model_params.lr").
        value: Value to set.
    """
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _sample_value(spec, rng: np.random.Generator) -> Any:
    """Sample a single value from a search space spec entry.

    Args:
        spec: List of values, or (min, max) tuple, or (min, max, "log").
        rng: Numpy random generator.

    Returns:
        A sampled parameter value.
    """
    if isinstance(spec, list):
        return spec[rng.integers(len(spec))]
    elif isinstance(spec, tuple):
        lo, hi = spec[0], spec[1]
        log_scale = len(spec) > 2 and spec[2] == "log"
        if log_scale:
            return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        elif isinstance(lo, int) and isinstance(hi, int):
            return int(rng.integers(lo, hi + 1))
        else:
            return float(rng.uniform(lo, hi))
    else:
        return spec


# ---------------------------------------------------------------------------
# Strategy 1: Grid search
# ---------------------------------------------------------------------------

def grid_strategy(space: dict, past_results: list[dict]) -> dict | None:
    """Exhaustive grid search over discrete parameter values.

    Only supports list-valued space entries (not continuous ranges).
    Returns None when all combinations have been evaluated.

    Args:
        space: Search space dict (all values must be lists).
        past_results: List of previous result dicts.

    Returns:
        Next parameter dict, or None if grid is exhausted.
    """
    keys = sorted(space.keys())
    all_values = [space[k] for k in keys]
    all_combos = list(itertools.product(*all_values))

    # Find which combos have already been tried
    tried = set()
    for result in past_results:
        combo = tuple(result["params"].get(k) for k in keys)
        tried.add(combo)

    for combo in all_combos:
        if combo not in tried:
            return dict(zip(keys, combo))

    return None


# ---------------------------------------------------------------------------
# Strategy 2: Random search
# ---------------------------------------------------------------------------

def random_strategy(
    space: dict,
    past_results: list[dict],
    seed: int = 42,
    max_evals: int = 50,
) -> dict | None:
    """Random sampling from the search space.

    Supports both discrete (list) and continuous (tuple) specs.
    Returns None after max_evals evaluations.

    Args:
        space: Search space dict.
        past_results: List of previous result dicts.
        seed: Random seed (offset by number of past results).
        max_evals: Maximum number of evaluations.

    Returns:
        Next parameter dict, or None if budget exhausted.
    """
    if len(past_results) >= max_evals:
        return None

    rng = np.random.default_rng(seed + len(past_results))
    return {key: _sample_value(spec, rng) for key, spec in space.items()}


# ---------------------------------------------------------------------------
# Strategy 3: Bayesian optimisation (GP-based)
# ---------------------------------------------------------------------------

def _distribution(spec):
    """Map a search-space spec to an Optuna distribution.

    Args:
        spec: A list of choices, a ``(lo, hi)`` range, or ``(lo, hi, "log")``.

    Returns:
        The matching ``optuna.distributions`` object.
    """
    from optuna.distributions import (
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
    )

    if isinstance(spec, list):
        return CategoricalDistribution(spec)
    lo, hi = spec[0], spec[1]
    log = len(spec) > 2 and spec[2] == "log"
    if isinstance(lo, int) and isinstance(hi, int) and not log:
        return IntDistribution(lo, hi)
    return FloatDistribution(lo, hi, log=log)


def bayesian_strategy(
    space: dict,
    past_results: list[dict],
    seed: int = 42,
    max_evals: int = 50,
    n_initial: int = 5,
) -> dict | None:
    """Bayesian optimisation via Optuna's Gaussian-process sampler.

    Replaces a hand-rolled sklearn GP + expected-improvement loop. Optuna's
    ``GPSampler`` models the objective properly (categoricals included, no
    hash-encoding hack) and optimises the acquisition itself. The driver's
    pull interface is stateless, so each call rebuilds a study from
    ``past_results``, tells it the observations, and asks for the next point.

    Args:
        space: Search space dict (lists / ``(lo, hi)`` / ``(lo, hi, "log")``).
        past_results: List of previous result dicts.
        seed: Random seed (reproducible search).
        max_evals: Maximum number of evaluations.
        n_initial: Random start-up evaluations before the GP takes over.

    Returns:
        Next parameter dict, or None if the budget is exhausted.
    """
    if len(past_results) >= max_evals:
        return None

    # Initial random exploration phase
    if len(past_results) < n_initial:
        return random_strategy(space, past_results, seed=seed, max_evals=max_evals)

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    distributions = {key: _distribution(spec) for key, spec in space.items()}
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.GPSampler(seed=seed, n_startup_trials=n_initial),
    )
    for result in past_results:
        params = {k: result["params"][k] for k in space if k in result["params"]}
        if len(params) != len(space):
            continue                        # skip incomplete observations
        study.add_trial(optuna.trial.create_trial(
            params=params, distributions=distributions,
            value=float(result.get("metric", 0.0)),
        ))
    trial = study.ask(distributions)
    return dict(trial.params)


# ---------------------------------------------------------------------------
# Search driver
# ---------------------------------------------------------------------------

def run_search(
    base_config: dict,
    space: dict,
    data_by_region: dict,
    strategy_fn: Callable = random_strategy,
    metric_key: str = "total_revenue",
    base: str | Path = "outputs/trials",
    **strategy_kwargs,
) -> list[dict]:
    """Run a hyperparameter search, evaluating each point as a full trial.

    Args:
        base_config: Base trial config to build from.
        space: Search space dict (param names → value specs).
        data_by_region: Dict mapping region → DataFrame.
        strategy_fn: Strategy function to call for the next params.
        metric_key: Which metric to optimise (from ledger.summarise).
        base: Root artifact directory.
        **strategy_kwargs: Passed through to the strategy function.

    Returns:
        List of result dicts, each with "trial_name", "params",
        "metric", and "metrics" (full summary dict).
    """
    search_name = base_config.get("trial_name", "search")
    results = []
    eval_idx = 0

    while True:
        # Ask strategy for the next point
        next_params = strategy_fn(space, results, **strategy_kwargs)
        if next_params is None:
            logger.info("Search '%s' complete — %d evaluations",
                        search_name, len(results))
            break

        # Build trial config for this evaluation
        trial_name = f"{search_name}_{eval_idx:03d}"
        overrides = dict(base_config)
        overrides["trial_name"] = trial_name
        overrides["parent_trial"] = search_name

        # Apply search parameters using dot notation
        for dotted_key, value in next_params.items():
            _set_nested(overrides, dotted_key, value)

        cfg = trials_mod.make_config(overrides)

        logger.info(
            "Search eval %d: %s — params: %s",
            eval_idx, trial_name, next_params,
        )

        # Run the trial
        trial_results = runner_mod.run_trial(data_by_region, cfg, base=base)

        # Aggregate metric across regions (mean)
        region_metrics = [r["metrics"][metric_key] for r in trial_results.values()]
        avg_metric = float(np.mean(region_metrics)) if region_metrics else 0.0

        results.append({
            "trial_name": trial_name,
            "params": next_params,
            "metric": avg_metric,
            "metrics": {
                region: r["metrics"] for region, r in trial_results.items()
            },
        })

        eval_idx += 1

    return results


def main() -> None:
    """Run this module as a CLI (exposes its public callables)."""
    from grian._cli import run_module_cli

    run_module_cli(globals())


if __name__ == "__main__":
    main()
