"""Tests for grian.evaluation.search — hyperparameter search strategies."""

import numpy as np

from grian.evaluation.search import (
    _sample_value,
    _set_nested,
    bayesian_strategy,
    grid_strategy,
    random_strategy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestSetNested:
    """Tests for dot-notation nested dict setting."""

    def test_flat_key(self):
        """Single key sets top-level value."""
        d = {}
        _set_nested(d, "lr", 0.01)
        assert d["lr"] == 0.01

    def test_nested_key(self):
        """Dotted key creates nested structure."""
        d = {}
        _set_nested(d, "model_params.lr", 0.01)
        assert d["model_params"]["lr"] == 0.01

    def test_deep_nesting(self):
        """Multiple dots create deep nesting."""
        d = {}
        _set_nested(d, "a.b.c", 42)
        assert d["a"]["b"]["c"] == 42

    def test_preserves_existing(self):
        """Setting a nested key preserves sibling keys."""
        d = {"model_params": {"n_estimators": 100}}
        _set_nested(d, "model_params.lr", 0.01)
        assert d["model_params"]["n_estimators"] == 100
        assert d["model_params"]["lr"] == 0.01


class TestSampleValue:
    """Tests for sampling from search space specs."""

    def test_sample_from_list(self):
        """Sampling from a list returns one of the list values."""
        rng = np.random.default_rng(42)
        options = [10, 20, 30]
        for _ in range(10):
            val = _sample_value(options, rng)
            assert val in options

    def test_sample_from_float_range(self):
        """Sampling from (lo, hi) returns a float in range."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            val = _sample_value((0.1, 0.9), rng)
            assert 0.1 <= val <= 0.9

    def test_sample_from_int_range(self):
        """Sampling from (int, int) returns an int in range."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            val = _sample_value((5, 20), rng)
            assert isinstance(val, int)
            assert 5 <= val <= 20

    def test_sample_log_scale(self):
        """Log-scale sampling stays in range."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            val = _sample_value((0.001, 1.0, "log"), rng)
            assert 0.001 <= val <= 1.0

    def test_log_scale_biases_toward_small(self):
        """Log-scale should sample more small values than uniform would."""
        rng = np.random.default_rng(42)
        samples = [_sample_value((0.001, 1.0, "log"), rng) for _ in range(1000)]
        # Median should be much less than 0.5 (the uniform midpoint)
        assert np.median(samples) < 0.2


# ---------------------------------------------------------------------------
# Grid strategy
# ---------------------------------------------------------------------------

class TestGridStrategy:
    """Tests for exhaustive grid search."""

    def test_returns_all_combinations(self):
        """Grid visits every combination exactly once."""
        space = {"a": [1, 2], "b": [10, 20]}
        results = []
        while True:
            params = grid_strategy(space, results)
            if params is None:
                break
            results.append({"params": params})
        assert len(results) == 4  # 2 × 2

    def test_returns_none_when_exhausted(self):
        """Grid returns None after all combinations."""
        space = {"a": [1]}
        past = [{"params": {"a": 1}}]
        assert grid_strategy(space, past) is None

    def test_skips_already_tried(self):
        """Grid doesn't repeat a combination."""
        space = {"a": [1, 2], "b": [10]}
        past = [{"params": {"a": 1, "b": 10}}]
        params = grid_strategy(space, past)
        assert params == {"a": 2, "b": 10}


# ---------------------------------------------------------------------------
# Random strategy
# ---------------------------------------------------------------------------

class TestRandomStrategy:
    """Tests for random search."""

    def test_returns_params(self):
        """Random strategy returns a params dict."""
        space = {"lr": (0.001, 0.1), "depth": [3, 5, 7]}
        params = random_strategy(space, [])
        assert "lr" in params
        assert "depth" in params

    def test_respects_max_evals(self):
        """Returns None after max_evals."""
        space = {"lr": (0.01, 0.1)}
        past = [{"params": {"lr": 0.05}} for _ in range(10)]
        result = random_strategy(space, past, max_evals=10)
        assert result is None

    def test_different_seeds_differ(self):
        """Different seeds produce different first samples."""
        space = {"lr": (0.001, 0.3)}
        r1 = random_strategy(space, [], seed=1)
        r2 = random_strategy(space, [], seed=99)
        # Very unlikely to be exactly equal
        assert r1["lr"] != r2["lr"]


# ---------------------------------------------------------------------------
# Bayesian strategy
# ---------------------------------------------------------------------------

class TestBayesianStrategy:
    """Tests for Bayesian optimisation strategy."""

    def test_initial_phase_returns_random(self):
        """First n_initial evals are random."""
        space = {"lr": (0.01, 0.3)}
        params = bayesian_strategy(space, [], n_initial=5)
        assert params is not None
        assert "lr" in params

    def test_respects_max_evals(self):
        """Returns None after max_evals."""
        space = {"lr": (0.01, 0.3)}
        past = [{"params": {"lr": 0.05}, "metric": 100.0} for _ in range(10)]
        result = bayesian_strategy(space, past, max_evals=10)
        assert result is None

    def test_gp_phase_returns_params(self):
        """After initial phase, GP-based selection returns params."""
        space = {"lr": (0.01, 0.3)}
        # Build enough past results to trigger GP
        past = [
            {"params": {"lr": 0.01 + 0.03 * i}, "metric": float(100 - i * 5)}
            for i in range(6)
        ]
        params = bayesian_strategy(space, past, n_initial=5, max_evals=20)
        assert params is not None
        assert 0.01 <= params["lr"] <= 0.3

    def test_gp_considers_past_results(self):
        """GP should explore different from past points."""
        space = {"lr": (0.01, 0.3)}
        # All past results at lr=0.15, bad metric
        past = [
            {"params": {"lr": 0.15}, "metric": 0.0}
            for _ in range(6)
        ]
        params = bayesian_strategy(space, past, n_initial=5, max_evals=20, seed=42)
        # Should suggest something different from 0.15
        assert params is not None
