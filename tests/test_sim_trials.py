"""Tests for grian.sim.trials — config, artifact I/O, reproducibility."""

import json

import numpy as np
import pandas as pd
import pytest

from grian.sim.trials import (
    DEFAULT_CONFIG,
    _get_transform_pair,
    get_git_sha,
    list_regions,
    list_trials,
    load_config,
    load_forecasts,
    load_ledger,
    load_metrics,
    make_config,
    save_config,
    save_forecasts,
    save_ledger,
    save_metrics,
    trial_dir,
)

# ---------------------------------------------------------------------------
# Transform pairs
# ---------------------------------------------------------------------------

class TestTransformPairs:
    """Tests for target transform (forward, inverse) pairs."""

    @pytest.mark.parametrize("name", ["asinh", "log1p", "identity"])
    def test_transform_round_trip(self, name):
        """forward(inverse(x)) == x for all named transforms."""
        forward, inverse = _get_transform_pair(name)
        # log1p domain: x > -1, so use non-negative values for it
        if name == "log1p":
            x = np.array([0.0, 1.0, 10.0, 100.0, 0.5])
        else:
            x = np.array([0.0, 1.0, 10.0, 100.0, -5.0])
        np.testing.assert_allclose(inverse(forward(x)), x, atol=1e-10)

    @pytest.mark.parametrize("name", ["asinh", "log1p", "identity"])
    def test_inverse_round_trip(self, name):
        """inverse(forward(x)) == x for all named transforms."""
        forward, inverse = _get_transform_pair(name)
        # log1p domain requires x > -1
        x = np.array([0.0, 1.0, 10.0, 100.0, 0.5])
        np.testing.assert_allclose(forward(inverse(x)), x, atol=1e-10)

    def test_unknown_transform_raises(self):
        """Requesting an unknown transform raises KeyError."""
        with pytest.raises(KeyError, match="Unknown transform"):
            _get_transform_pair("nonexistent")

    def test_asinh_compresses_spikes(self):
        """asinh should compress large values more than small ones."""
        forward, _ = _get_transform_pair("asinh")
        small = forward(np.array([50.0]))[0]
        large = forward(np.array([15000.0]))[0]
        # The ratio of transformed values should be much less than raw
        assert large / small < 15000.0 / 50.0


# ---------------------------------------------------------------------------
# Git SHA
# ---------------------------------------------------------------------------

class TestGitSha:
    """Tests for git SHA capture."""

    def test_returns_string(self):
        """get_git_sha returns a string."""
        sha = get_git_sha()
        assert isinstance(sha, str)

    def test_sha_length(self):
        """SHA is either 40 hex chars or 'unknown'."""
        sha = get_git_sha()
        assert sha == "unknown" or (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha))


# ---------------------------------------------------------------------------
# Config creation
# ---------------------------------------------------------------------------

class TestMakeConfig:
    """Tests for trial config creation and validation."""

    def test_default_config_has_all_keys(self):
        """make_config() with no overrides produces a complete config."""
        cfg = make_config()
        for key in DEFAULT_CONFIG:
            assert key in cfg

    def test_overrides_applied(self):
        """Scalar overrides are applied correctly."""
        cfg = make_config({"trial_name": "my_trial", "seed": 123})
        assert cfg["trial_name"] == "my_trial"
        assert cfg["seed"] == 123

    def test_nested_merge(self):
        """Nested dict overrides merge, not replace."""
        cfg = make_config({"ablations": {"leak_future": True}})
        # leak_future should be overridden
        assert cfg["ablations"]["leak_future"] is True
        # Other ablation keys should keep defaults
        assert cfg["ablations"]["use_embargo"] is True

    def test_unknown_key_raises(self):
        """Config rejects unknown keys (typo protection)."""
        with pytest.raises(KeyError, match="Unknown config key"):
            make_config({"typo_key": 42})

    def test_git_sha_stamped(self):
        """Config includes a git SHA."""
        cfg = make_config()
        assert "git_sha" in cfg
        assert isinstance(cfg["git_sha"], str)

    def test_timestamp_stamped(self):
        """Config includes a UTC timestamp."""
        cfg = make_config()
        assert "timestamp" in cfg
        assert "T" in cfg["timestamp"]  # ISO format

    def test_config_is_independent_copy(self):
        """Modifying a config doesn't affect DEFAULT_CONFIG."""
        cfg = make_config()
        cfg["seed"] = 999
        assert DEFAULT_CONFIG["seed"] == 42


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------

class TestArtifactIO:
    """Tests for saving and loading trial artifacts."""

    def test_save_load_config(self, tmp_path):
        """Config round-trips through save/load."""
        cfg = make_config({"trial_name": "test_trial"})
        save_config(cfg, base=tmp_path)
        loaded = load_config("test_trial", base=tmp_path)
        assert loaded["trial_name"] == "test_trial"
        assert loaded["seed"] == cfg["seed"]
        assert loaded["git_sha"] == cfg["git_sha"]

    def test_config_json_is_valid(self, tmp_path):
        """Saved config is valid JSON."""
        cfg = make_config({"trial_name": "json_test"})
        path = save_config(cfg, base=tmp_path)
        with open(path) as f:
            data = json.load(f)
        assert data["trial_name"] == "json_test"

    def test_save_load_metrics(self, tmp_path):
        """Metrics round-trip through save/load."""
        metrics = {"mae": 12.5, "total_revenue": 1000.0}
        save_metrics(metrics, "trial_a", "SA1", base=tmp_path)
        loaded = load_metrics("trial_a", "SA1", base=tmp_path)
        assert loaded["mae"] == 12.5
        assert loaded["total_revenue"] == 1000.0

    def test_save_load_forecasts(self, tmp_path):
        """Forecasts round-trip through save/load."""
        df = pd.DataFrame({
            "origin": pd.Timestamp("2023-07-01"),
            "step": [0, 1, 2],
            "forecast": [50.0, 55.0, 48.0],
            "actual": [51.0, 54.0, 49.0],
        })
        save_forecasts(df, "trial_b", "SA1", base=tmp_path)
        loaded = load_forecasts("trial_b", "SA1", base=tmp_path)
        assert len(loaded) == 3
        assert list(loaded.columns) == list(df.columns)

    def test_save_load_ledger(self, tmp_path):
        """Ledger round-trips through save/load."""
        idx = pd.date_range("2023-07-01", periods=5, freq="5min")
        df = pd.DataFrame({
            "actual_price": [50, 51, 52, 53, 54],
            "revenue": [10, -5, 20, -3, 15],
        }, index=idx)
        df.index.name = "timestamp"
        save_ledger(df, "trial_c", "SA1", base=tmp_path)
        loaded = load_ledger("trial_c", "SA1", base=tmp_path)
        assert len(loaded) == 5

    def test_trial_dir_with_region(self, tmp_path):
        """trial_dir returns region subfolder when region is given."""
        path = trial_dir("my_trial", "SA1", base=tmp_path)
        assert path == tmp_path / "my_trial" / "SA1"

    def test_trial_dir_without_region(self, tmp_path):
        """trial_dir returns trial folder when region is None."""
        path = trial_dir("my_trial", base=tmp_path)
        assert path == tmp_path / "my_trial"


# ---------------------------------------------------------------------------
# Trial discovery
# ---------------------------------------------------------------------------

class TestTrialDiscovery:
    """Tests for listing trials and regions on disk."""

    def test_list_trials_empty(self, tmp_path):
        """list_trials on empty dir returns empty list."""
        assert list_trials(base=tmp_path) == []

    def test_list_trials_finds_trials(self, tmp_path):
        """list_trials finds trial dirs that have config.json."""
        cfg = make_config({"trial_name": "trial_x"})
        save_config(cfg, base=tmp_path)
        assert "trial_x" in list_trials(base=tmp_path)

    def test_list_regions_finds_regions(self, tmp_path):
        """list_regions finds region subdirs with metrics.json."""
        metrics = {"mae": 10.0}
        save_metrics(metrics, "trial_y", "SA1", base=tmp_path)
        save_metrics(metrics, "trial_y", "NSW1", base=tmp_path)
        save_config(make_config({"trial_name": "trial_y"}), base=tmp_path)
        regions = list_regions("trial_y", base=tmp_path)
        assert "SA1" in regions
        assert "NSW1" in regions

    def test_list_regions_empty(self, tmp_path):
        """list_regions returns empty for nonexistent trial."""
        assert list_regions("nonexistent", base=tmp_path) == []
