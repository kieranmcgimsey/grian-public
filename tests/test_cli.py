"""Smoke tests for the data-free CLI commands (grian.cli)."""

import pytest

from grian import cli


def test_models_and_describe_and_configs_run():
    """The introspection commands render without error."""
    cli.models()
    cli.describe("lightgbm_qmean")
    cli.configs()


def test_describe_unknown_model():
    """describe surfaces a clear error for an unknown model."""
    with pytest.raises(KeyError, match="Unknown model"):
        cli.describe("not_a_model")


def test_space_from_yaml(tmp_path):
    """A search-space YAML parses into the typed tuning dimensions."""
    from grian.tuning import Categorical, Float, Int

    path = tmp_path / "space.yaml"
    path.write_text(
        "learning_rate: {type: float, low: 0.001, high: 0.3, log: true}\n"
        "n_estimators: {type: int, low: 50, high: 600}\n"
        "calendar_encoding: {type: categorical, choices: [onehot, fourier]}\n"
    )
    space = cli._space_from_yaml(str(path))
    assert isinstance(space["learning_rate"], Float) and space["learning_rate"].log
    assert isinstance(space["n_estimators"], Int)
    assert isinstance(space["calendar_encoding"], Categorical)
    assert space["calendar_encoding"].choices == ["onehot", "fourier"]


def test_run_reports_missing_data(tmp_path):
    """`grian run` fails clearly when the processed parquet is absent."""
    from grian.experiment import ExperimentConfig

    cfg = ExperimentConfig(model="naive_similar_day", regions=["ZZ1"])
    path = cfg.to_yaml(tmp_path / "c.yaml")
    with pytest.raises(FileNotFoundError, match="Missing"):
        cli.run(str(path))
