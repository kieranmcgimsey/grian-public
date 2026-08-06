"""Typed, YAML-driven experiment configuration for reproducible runs.

An :class:`ExperimentConfig` is the strongly-typed, immutable spec for a single
run — model + typed hyperparameters + seed + data window + dispatch physics — that
the CLI loads from a YAML file to reproduce a trial exactly, down to the seed.

It is a thin authoring/reproducibility layer over the engine: :meth:`to_overrides`
produces the plain dict the existing :func:`grian.evaluation.trials.make_config`
consumes, so ``config.json`` provenance, the runner, and every scorer are unchanged.
Model hyperparameters are validated against the model's typed params class
(:func:`grian.models.params_for`); an unknown key raises rather than silently
misconfiguring the run.
"""

from __future__ import annotations

import importlib.metadata as ilmd
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from grian.models import params_for, seed_everything


class _Frozen(BaseModel):
    """Frozen, typo-protected base."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DispatchConfig(_Frozen):
    """Battery physics for the dispatch LP."""

    power_mw: float = 100.0
    duration_hours: float = 2.0
    efficiency: float = 0.85
    max_cycles: int = 2


class MPCConfig(BaseModel):
    """Receding-horizon MPC settings.

    Core cadence fields are typed; ``extra="allow"`` keeps the experimental
    dispatch-mode / scenario / CVaR knobs usable without enumerating every one.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    resolve_every: int = 6
    reforecast_every: int = 12
    persistence_tau: float = 0
    persistence_gate: float = 0


class ExperimentConfig(_Frozen):
    """A complete, reproducible run specification.

    Attributes:
        model: A registered model name (see :data:`grian.models.REGISTRY`).
        model_params: Hyperparameter overrides, validated against the model's
            typed params class; defaults fill the rest.
        seed: Master seed applied to every RNG before the run.
        executor: ``"openloop"`` (one forecast/solve per day) or ``"mpc"``.
    """

    model: str = "naive_similar_day"
    model_params: dict[str, Any] = Field(default_factory=dict)
    seed: int = 42

    regions: list[str] = Field(default_factory=lambda: ["SA1"])
    resolution: str = "5min"
    target_col: str = "price"
    horizon: int = 288
    transform: str = "asinh"
    loss: str = "pinball"

    train_start: str = "2020-01-01"
    train_end: str = "2023-06-30"
    test_start: str = "2023-07-01"
    test_end: str = "2024-06-30"
    refit_days: int = 1
    train_lookback_days: int | None = None
    embargo: int = 288

    dispatch: DispatchConfig = Field(default_factory=DispatchConfig)
    executor: str = "openloop"
    mpc: MPCConfig | None = None

    @model_validator(mode="after")
    def _validate_model_params(self) -> ExperimentConfig:
        """Reject unknown model, and validate model_params against its schema."""
        params_for(self.model).model_validate(self.model_params)
        if self.executor not in ("openloop", "mpc"):
            raise ValueError(
                f"executor must be 'openloop' or 'mpc', got {self.executor!r}"
            )
        return self

    # -- typed accessors -----------------------------------------------------

    def params(self) -> Any:
        """Return the model's hyperparameters as their typed params object."""
        return params_for(self.model).model_validate(self.model_params)

    def to_overrides(self) -> dict[str, Any]:
        """Return the overrides dict for :func:`make_config` (engine wire form).

        ``model_params`` is normalised through the typed schema so defaults are
        materialised — two configs that differ only in unset keys produce an
        identical ``config.json``.
        """
        overrides: dict[str, Any] = {
            "model": self.model,
            "model_params": self.params().model_dump(),
            "seed": self.seed,
            "regions": list(self.regions),
            "resolution": self.resolution,
            "target_col": self.target_col,
            "horizon": self.horizon,
            "transform": self.transform,
            "loss": self.loss,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "refit_days": self.refit_days,
            "train_lookback_days": self.train_lookback_days,
            "embargo": self.embargo,
            "dispatch": self.dispatch.model_dump(),
        }
        if self.executor == "mpc" and self.mpc is not None:
            overrides["mpc"] = self.mpc.model_dump()
        return overrides

    def provenance(self) -> dict[str, str]:
        """Return library versions + git SHA to stamp alongside a run."""
        from grian.evaluation.trials import get_git_sha

        pkgs = ("grian", "pydantic", "numpy", "pandas", "scikit-learn",
                "lightgbm", "torch", "optuna")
        versions: dict[str, str] = {}
        for pkg in pkgs:
            try:
                versions[pkg] = ilmd.version(pkg)
            except ilmd.PackageNotFoundError:
                continue
        versions["git_sha"] = get_git_sha()
        return versions

    # -- (de)serialisation ---------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load and validate an experiment config from a YAML file."""
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> Path:
        """Write the (normalised) config to a YAML file; return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.model_dump(), fh, sort_keys=False)
        return path

    # -- execution -----------------------------------------------------------

    def simulate_fn(self):
        """Return the simulation loop for this executor (None = open-loop)."""
        if self.executor == "mpc":
            from grian.dispatch.mpc import simulate_region_mpc

            return simulate_region_mpc
        return None

    def run(
        self,
        data,
        oracle_daily_revenue,
        base: str | Path = "outputs/trials",
        trial_name: str | None = None,
    ) -> dict[str, Any]:
        """Seed, run the trial on ``data``, and score it against the oracle.

        Args:
            data: Prepared price/demand (+weather) frame for the region.
            oracle_daily_revenue: Perfect-foresight daily revenue (the capture
                denominator), e.g. ``compute_oracle(...)["daily_revenue"]``.
            base: Root directory for trial artifacts.
            trial_name: Optional trial name; defaults to ``model__executor``.

        Returns:
            The capture report dict (``capture_ratio``, revenues, …).
        """
        from grian.dispatch.open_loop import run_trial
        from grian.evaluation.analytics import capture_report
        from grian.evaluation.trials import make_config

        seed_everything(self.seed)
        name = trial_name or f"{self.model}__{self.executor}"
        overrides = self.to_overrides()
        overrides["trial_name"] = name
        cfg = make_config(overrides)
        region = self.regions[0]
        run_trial({region: data}, cfg, base=base, simulate_fn=self.simulate_fn())

        ledger_path = Path(base) / name / region / "ledger.parquet"
        import pandas as pd

        ledger_df = pd.read_parquet(ledger_path)
        return capture_report(ledger_df, oracle_daily_revenue)


def main() -> None:
    """Run this module as a CLI (exposes its public callables)."""
    from grian._cli import run_module_cli

    run_module_cli(globals())


if __name__ == "__main__":
    main()
