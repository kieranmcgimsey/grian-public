"""Tests for the typed, YAML-driven experiment config (grian.experiment)."""

import pytest

from grian.experiment import ExperimentConfig


def test_to_overrides_materialises_defaults():
    """model_params is normalised through the typed schema (defaults filled)."""
    cfg = ExperimentConfig(model="lightgbm_qmean", model_params={"n_estimators": 200})
    ov = cfg.to_overrides()
    assert ov["model_params"]["n_estimators"] == 200
    assert ov["model_params"]["num_leaves"] == 31          # default materialised
    assert ov["model_params"]["calendar_encoding"] == "ordinal"
    assert ov["seed"] == 42


def test_unknown_model_and_bad_params_rejected():
    """Unknown model or a typo in model_params fails fast at construction."""
    with pytest.raises(Exception, match="Unknown model"):
        ExperimentConfig(model="nope")
    with pytest.raises(Exception, match="extra|forbid|permitted"):
        ExperimentConfig(model="lightgbm_rich", model_params={"n_estimatorz": 1})


def test_executor_validation_and_mpc():
    """executor is constrained; MPC settings flow into the overrides."""
    with pytest.raises(Exception, match="executor"):
        ExperimentConfig(model="naive_similar_day", executor="turbo")
    cfg = ExperimentConfig(model="lightgbm_qmean", executor="mpc",
                           mpc={"resolve_every": 3, "reforecast_every": 6})
    assert cfg.to_overrides()["mpc"]["resolve_every"] == 3


def test_yaml_round_trip(tmp_path):
    """A config written to YAML reloads to an identical set of overrides."""
    cfg = ExperimentConfig(
        model="lear_weather", model_params={"include_weather": True},
        resolution="30min", horizon=48,
    )
    path = cfg.to_yaml(tmp_path / "c.yaml")
    assert ExperimentConfig.from_yaml(path).to_overrides() == cfg.to_overrides()


def test_provenance_records_versions():
    """Provenance stamps the git SHA and key library versions."""
    prov = ExperimentConfig(model="naive_similar_day").provenance()
    assert "git_sha" in prov and "grian" in prov and "pydantic" in prov


def test_shipped_configs_are_valid():
    """Every configs/*.yaml loads and validates."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    files = sorted(root.glob("*.yaml"))
    assert files, "no example configs found"
    for path in files:
        ExperimentConfig.from_yaml(path)  # raises on any invalid config
