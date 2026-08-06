"""Tests for the unified Optuna tuning interface (grian.tuning)."""

import optuna
import pytest

from grian.tuning import Categorical, Float, Int, tune

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _bowl(p):
    """Concave objective peaking at n=5, lr≈0.1, scheme='b'."""
    return (-((p["n_estimators"] - 5) ** 2) / 25
            - (p["learning_rate"] - 0.1) ** 2 * 100
            + (1.0 if p["scheme"] == "b" else 0.0))


_SPACE = {
    "n_estimators": Int(1, 10),
    "learning_rate": Float(1e-3, 0.3, log=True),
    "scheme": Categorical(["a", "b"]),
}


@pytest.mark.parametrize("sampler", ["random", "tpe", "gp"])
def test_samplers_optimise(sampler):
    """Each sampler improves on the objective and picks the good category."""
    study = tune(_bowl, _SPACE, sampler=sampler, n_trials=25, seed=0)
    assert study.best_params["scheme"] == "b"
    assert study.best_value > 0.0            # beats the a-scheme floor


def test_grid_enumerates_and_wins():
    """Grid search enumerates the discrete space and finds the maximum."""
    space = {"n_estimators": Int(1, 6), "scheme": Categorical(["a", "b"])}
    study = tune(lambda p: p["n_estimators"] + (1.0 if p["scheme"] == "b" else 0.0),
                 space, sampler="grid", n_trials=100, seed=0)
    assert study.best_params == {"n_estimators": 6, "scheme": "b"}
    assert len(study.trials) == 12           # 6 × 2 grid, fully enumerated


def test_grid_rejects_continuous_dims():
    """Grid needs enumerable dims; a Float should be a clear error."""
    with pytest.raises(ValueError, match="grid search needs"):
        tune(_bowl, {"learning_rate": Float(0.0, 1.0)}, sampler="grid", n_trials=3)


def test_unknown_sampler():
    """An unknown sampler name is rejected."""
    with pytest.raises(ValueError, match="Unknown sampler"):
        tune(_bowl, _SPACE, sampler="magic", n_trials=1)


def test_dimensions_are_frozen():
    """Typed dims are immutable value objects."""
    from dataclasses import FrozenInstanceError

    d = Float(0.0, 1.0)
    with pytest.raises(FrozenInstanceError):
        d.low = 5.0
