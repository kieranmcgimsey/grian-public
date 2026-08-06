"""Typed, immutable hyperparameter models for the forecasters.

Every model ``fit`` used to read its hyperparameters as scattered
``model_params.get(key, default)`` calls with the defaults hardcoded inline (and
duplicated across the three LightGBM variants). This module replaces that with
one frozen `pydantic` model per family, so:

* every tunable is an explicit, typed **input** with its default defined once;
* the surface is immutable and typo-protected (``extra="forbid"`` → an unknown
  key raises, rather than silently doing nothing like a dict ``.get`` miss);
* a variety of tuning interfaces share the same type — construct directly, parse
  a dict/YAML with :meth:`~pydantic.BaseModel.model_validate`, or produce a
  variant with :meth:`~pydantic.BaseModel.model_copy` ``(update=...)``.

The wire format stays a plain dict (``cfg["model_params"]``) so ``config.json``
provenance, the search driver's dotted-key overrides, and existing scripts keep
working; each ``fit`` parses that dict into its typed params at the top.

Resolution-dependent defaults (``lags``, ``step_stride``) default to ``None``
here and are resolved inside the fit against the trial resolution — they cannot
be static because they depend on the periods-per-day of the run.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Default quantile fan shared by the probabilistic models.
_DEFAULT_QUANTILES = [0.05, 0.5, 0.9, 0.98]
# Default lag basis, expressed in *days*; the fit scales by periods-per-day.
_DEFAULT_LAG_DAYS = (1, 2, 7)


class _Params(BaseModel):
    """Base for all model params: frozen and typo-protected."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class SampleWeighting(_Params):
    """Decision-focused per-sample fit weights (see ``_decision_weights``)."""

    scheme: str = "magnitude"   # "magnitude" (log-dollar ramp) | "quantile"
    strength: float = 1.0
    scale: float = 300.0        # dollar scale for the magnitude ramp
    q: float = 0.9              # percentile cut for the quantile scheme


class NaiveParams(_Params):
    """The similar-day naive baseline has no tunable hyperparameters."""


class ARParams(_Params):
    """Autoregression on lagged prices + calendar."""

    lags: list[int] | None = None            # None → [ppd, 2·ppd, 7·ppd]
    calendar_encoding: str = "onehot"        # onehot | fourier | ordinal


class LinearParams(_Params):
    """LEAR family (Lasso/Ridge/ElasticNet/OLS) on the rich feature set."""

    estimator: str | None = None             # None → inferred from model name
    calendar_encoding: str = "onehot"
    include_weather: bool = False
    include_scarcity: bool = False
    feature_set: str = "full"                # full | lean
    step_stride: int | None = None           # None → one model per hour
    n_alphas: int = 20                       # CV path length (Lasso/ElasticNet)
    max_iter: int = 2000
    l1_ratio: float = 0.5                    # ElasticNet only


class LearParams(_Params):
    """Deprecated sklearn quantile-LEAR (one HiGHS LP per step × quantile)."""

    quantiles: list[float] = Field(default_factory=lambda: list(_DEFAULT_QUANTILES))
    alpha: float = 0.01                      # L1 strength
    calendar_encoding: str = "onehot"
    include_weather: bool = False
    step_stride: int | None = None


class LearTorchParams(_Params):
    """Batched torch quantile-LEAR — the canonical quantile linear model."""

    quantiles: list[float] = Field(default_factory=lambda: list(_DEFAULT_QUANTILES))
    alpha: float = 0.01
    calendar_encoding: str = "onehot"
    include_weather: bool = False
    epochs: int = 400
    lr: float = 0.005
    feature_clip: float = 5.0                # winsorise standardised features
    step_stride: int | None = None


class LGBMParams(_Params):
    """LightGBM point forecaster (one direct booster per horizon step).

    ``lags`` and ``step_stride`` are feature/loop controls; the rest map onto
    ``LGBMRegressor`` kwargs via :meth:`to_lgb_kwargs`. The loss → objective
    wiring (``pinball``/``huber``) is applied in the fit from ``cfg["loss"]``.
    ``lgb_extra`` is an explicit, typed escape hatch for forwarding any other
    LightGBM kwarg (e.g. ``reg_lambda``) without weakening typo protection on
    the named fields.
    """

    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    lags: list[int] | None = None
    step_stride: int | None = None
    lgb_extra: dict[str, Any] = Field(default_factory=dict)

    def to_lgb_kwargs(self) -> dict[str, Any]:
        """Return the ``LGBMRegressor`` kwargs (objective wired by the fit)."""
        kwargs: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "verbosity": -1,
            "n_jobs": -1,
        }
        kwargs.update(self.lgb_extra)
        return kwargs


class LGBMRichParams(LGBMParams):
    """LightGBM on the full feature set, with optional decision-focused weights."""

    calendar_encoding: str = "ordinal"
    include_weather: bool = False
    include_scarcity: bool = False
    sample_weighting: SampleWeighting | None = None


class LGBMQMeanParams(LGBMParams):
    """LightGBM quantile boosters integrated to a dollar-space mean.

    One booster per (step, quantile). ``n_estimators`` defaults lower (150) than
    the point model; ``max_depth`` inherits ``-1`` (LightGBM's own default, so
    the fitted trees are unchanged from the pre-typed code).
    """

    n_estimators: int = 150
    calendar_encoding: str = "ordinal"
    include_weather: bool = False
    quantiles: list[float] = Field(default_factory=lambda: list(_DEFAULT_QUANTILES))
    calibrate: bool = False                  # split-conformal fan calibration
    cal_days: int = 28                       # calibration holdout length
    mean_from_step: int = 0                  # first step to integrate the mean


class MLPParams(_Params):
    """Deprecated two-layer MLP (kept for reference; frozen-tail predict)."""

    lags: list[int] | None = None
    hidden_dim: int = 128
    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 256


class LSTMParams(_Params):
    """Deprecated LSTM sequence model (kept for reference)."""

    seq_len: int = 288
    hidden_dim: int = 64
    num_layers: int = 2
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 256
    dropout: float = 0.1


def default_lags(periods_per_day: int) -> list[int]:
    """Resolve the resolution-dependent default lag basis.

    Args:
        periods_per_day: Intervals per day at the trial resolution.

    Returns:
        The default lag list ``[1·ppd, 2·ppd, 7·ppd]``.
    """
    return [d * periods_per_day for d in _DEFAULT_LAG_DAYS]


__all__ = [
    "ARParams",
    "LGBMParams",
    "LGBMQMeanParams",
    "LGBMRichParams",
    "LSTMParams",
    "LearParams",
    "LearTorchParams",
    "LinearParams",
    "MLPParams",
    "NaiveParams",
    "SampleWeighting",
    "default_lags",
]


def main() -> None:
    """Run this module as a CLI (exposes its public callables)."""
    from grian._cli import run_module_cli

    run_module_cli(globals())


if __name__ == "__main__":
    main()
