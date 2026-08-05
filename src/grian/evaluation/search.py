"""Hyperparameter search strategies.

Each search strategy is a plain function with the signature:

    fn(space, past_results) -> next_params | None

The search driver calls the strategy repeatedly, running a full trial
for each set of parameters. Every evaluation is a complete trial with
its own config.json and artifacts on disk — nothing is ephemeral.

Strategies included:
    grid_strategy    — Exhaustive grid over discrete values.
    random_strategy  — Random sampling from continuous/discrete ranges.
    bayesian_strategy — Gaussian-process-based Bayesian optimisation.

The driver (`run_search`) orchestrates the loop: it asks the strategy
for the next point, runs a trial, records the result, and repeats
until the strategy returns None or the budget is exhausted.
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

def bayesian_strategy(
    space: dict,
    past_results: list[dict],
    seed: int = 42,
    max_evals: int = 50,
    n_initial: int = 5,
) -> dict | None:
    """Bayesian optimisation using a Gaussian process surrogate.

    Uses sklearn's GaussianProcessRegressor to model the objective
    surface. Selects the next point by maximising expected improvement.
    Falls back to random sampling for the first `n_initial` evaluations.

    Args:
        space: Search space dict.
        past_results: List of previous result dicts.
        seed: Random seed.
        max_evals: Maximum number of evaluations.
        n_initial: Number of random initial evaluations before GP kicks in.

    Returns:
        Next parameter dict, or None if budget exhausted.
    """
    if len(past_results) >= max_evals:
        return None

    # Initial random exploration phase
    if len(past_results) < n_initial:
        return random_strategy(space, past_results, seed=seed, max_evals=max_evals)

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern

    rng = np.random.default_rng(seed + len(past_results))
    keys = sorted(space.keys())

    # Encode past results into X (params) and y (metric)
    X_obs = []
    y_obs = []
    for result in past_results:
        row = []
        for k in keys:
            val = result["params"].get(k, 0)
            row.append(float(val) if not isinstance(val, str) else hash(val) % 1000)
        X_obs.append(row)
        # Minimise negative revenue (i.e. maximise revenue)
        y_obs.append(-result.get("metric", 0.0))

    X_obs = np.array(X_obs)
    y_obs = np.array(y_obs)

    # Fit GP
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5),
        n_restarts_optimizer=5,
        random_state=seed,
    )
    gp.fit(X_obs, y_obs)

    # Generate candidate points and evaluate expected improvement
    n_candidates = 1000
    candidates = []
    for _ in range(n_candidates):
        point = {key: _sample_value(spec, rng) for key, spec in space.items()}
        row = [
            float(point[k]) if not isinstance(point[k], str)
            else hash(point[k]) % 1000
            for k in keys
        ]
        candidates.append((point, row))

    X_cand = np.array([c[1] for c in candidates])
    mu, sigma = gp.predict(X_cand, return_std=True)

    # Expected improvement
    best_y = y_obs.min()
    with np.errstate(divide="ignore", invalid="ignore"):
        improvement = best_y - mu
        Z = improvement / (sigma + 1e-8)
        from scipy.stats import norm
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-8] = 0.0

    best_idx = np.argmax(ei)
    return candidates[best_idx][0]


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
