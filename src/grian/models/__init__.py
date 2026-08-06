"""Model registry for the trading simulation.

Each model is a plain dict of four functions (``fit``/``predict``/``save``/``load``);
quantile models add ``predict_fan``. The registry maps model names to those dicts and
``get_model`` looks one up. Implementations live in the sibling modules
(``baselines``, ``linear``, ``gradient_boosting``, ``neural``); shared helpers in
``_shared``.
"""

from grian.models._shared import (
    _CALENDAR_COLS,
    _FOURIER_SPEC,
    _LINEAR_BY_NAME,
    _apply_conformal,
    _build_lag_features,
    _calendar_features,
    _conformal_fan_adjustments,
    _decision_weights,
    _fourier_calendar,
    _get_device,
    _l1_penalty,
    _leading_calendar_columns,
    _linear_preprocessor,
    _make_linear_estimator,
    _periods_per_day,
    _pinball_loss,
    _quantile_weights,
    seed_everything,
)
from grian.models.baselines import AUTOREGRESSION, NAIVE_SIMILAR_DAY
from grian.models.gradient_boosting import LIGHTGBM, LIGHTGBM_QMEAN, LIGHTGBM_RICH
from grian.models.linear import LEAR_QMEAN, LEAR_QMEAN_TORCH, LINEAR
from grian.models.neural import LSTM, SIMPLE_MLP
from grian.models.params import (
    ARParams,
    LearParams,
    LearTorchParams,
    LGBMParams,
    LGBMQMeanParams,
    LGBMRichParams,
    LinearParams,
    LSTMParams,
    MLPParams,
    NaiveParams,
    SampleWeighting,
)

__all__ = [
    "REGISTRY", "get_model", "PARAMS_FOR", "params_for", "seed_everything",
    "NAIVE_SIMILAR_DAY", "AUTOREGRESSION", "LINEAR", "LEAR_QMEAN", "LEAR_QMEAN_TORCH",
    "LIGHTGBM", "LIGHTGBM_RICH", "LIGHTGBM_QMEAN", "SIMPLE_MLP", "LSTM",
    "NaiveParams", "ARParams", "LinearParams", "LearParams", "LearTorchParams",
    "LGBMParams", "LGBMRichParams", "LGBMQMeanParams", "MLPParams", "LSTMParams",
    "SampleWeighting",
    "_CALENDAR_COLS", "_FOURIER_SPEC", "_LINEAR_BY_NAME", "_apply_conformal",
    "_build_lag_features", "_calendar_features", "_conformal_fan_adjustments",
    "_decision_weights", "_fourier_calendar", "_get_device", "_l1_penalty",
    "_leading_calendar_columns", "_linear_preprocessor", "_make_linear_estimator",
    "_periods_per_day", "_pinball_loss", "_quantile_weights",
]

# Model family (a spec's canonical ``name``) → its typed params class.
_PARAMS_BY_SPEC_NAME: dict[str, type] = {
    "naive_similar_day": NaiveParams,
    "autoregression": ARParams,
    "linear": LinearParams,
    "lear_qmean": LearParams,
    "lear_qmean_torch": LearTorchParams,
    "lightgbm": LGBMParams,
    "lightgbm_rich": LGBMRichParams,
    "lightgbm_qmean": LGBMQMeanParams,
    "simple_mlp": MLPParams,
    "lstm": LSTMParams,
}


REGISTRY: dict[str, dict] = {
    "naive_similar_day": NAIVE_SIMILAR_DAY,
    "autoregression": AUTOREGRESSION,          # one-hot calendar (default)
    "autoregression_fourier": AUTOREGRESSION,  # Fourier calendar via model_params
    "autoregression_ordinal": AUTOREGRESSION,  # legacy ordinal (for comparison)
    # LEAR family — regularised linear on the rich feature set (estimator
    # inferred from the name; see _LINEAR_BY_NAME).
    "lear": LINEAR,                    # one-hot calendar (default)
    "lear_weather": LINEAR,            # weather on via model_params.include_weather
    "lear_fourier": LINEAR,            # Fourier (cyclic) calendar via model_params
    "lear_weather_fourier": LINEAR,    # weather + Fourier (full ablation cell)
    "lear_ordinal": LINEAR,            # legacy ordinal calendar (for comparison)
    "ridge": LINEAR,
    "elasticnet": LINEAR,
    # Shape-preserving linear models: raw price-lag basis + calendar, no
    # mean-reverting smoothers (feature_set="lean"). Tests whether LEAR's
    # sub-AR capture is the smoothers flattening peaks (Entry 032).
    "lear_lean": LINEAR,               # Lasso on the lean basis
    "ols_lean": LINEAR,                # OLS on the lean basis (richer AR)
    # DEPRECATED sklearn quantile LEAR — impractical (~8 min/refit) and it
    # regularizes the Fourier block. Superseded by lear_qmean_torch below; kept
    # only to reproduce the pre-torch ablation. See _lear_qmean_fit docstring.
    "lear_qmean": LEAR_QMEAN,          # linear quantile fan (probabilistic LEAR)
    "lear_qmean_weather": LEAR_QMEAN,
    "lear_qmean_fourier": LEAR_QMEAN,
    "lear_qmean_weather_fourier": LEAR_QMEAN,   # weather + Fourier quantile LEAR
    # Torch (CPU) reimplementation of the quantile fan: one batched pinball fit
    # instead of ~100 CPU LPs, with the Fourier calendar block left unpenalised.
    # ~40x faster and correct — the canonical LEAR quantile model going forward,
    # and what fills the LEAR-quantile ablation cells Pinned
    # to CPU: the planned MPS backend diverges (Entry 036).
    "lear_qmean_torch": LEAR_QMEAN_TORCH,
    "lear_qmean_torch_weather": LEAR_QMEAN_TORCH,
    "lear_qmean_torch_fourier": LEAR_QMEAN_TORCH,
    "lear_qmean_torch_weather_fourier": LEAR_QMEAN_TORCH,
    "lightgbm": LIGHTGBM,
    "lightgbm_rich": LIGHTGBM_RICH,
    # Same spec as lightgbm_rich; weather features are switched on via
    # model_params.include_weather (see scripts/run_common_eval.py).
    "lightgbm_rich_weather": LIGHTGBM_RICH,
    # Fourier (cyclic sin/cos) calendar features for trees via
    # model_params.calendar_encoding — the tree side of the calendar ablation.
    "lightgbm_rich_fourier": LIGHTGBM_RICH,
    "lightgbm_rich_weather_fourier": LIGHTGBM_RICH,
    # Spike-precursor (scarcity) features on via model_params.include_scarcity —
    # the anticipating forecast meant to break the capture ceiling (Entry 030).
    "lightgbm_rich_scarcity": LIGHTGBM_RICH,
    # Decision-focused loss: scarcity features + high-price sample weighting
    # (model_params.sample_weighting). Attacks the accuracy-vs-capture gap.
    "lightgbm_rich_dfl": LIGHTGBM_RICH,
    "lightgbm_qmean": LIGHTGBM_QMEAN,
    # Same spec; weather on via model_params.include_weather.
    "lightgbm_qmean_weather": LIGHTGBM_QMEAN,
    "lightgbm_qmean_fourier": LIGHTGBM_QMEAN,          # Fourier-calendar quantile GBM
    "lightgbm_qmean_weather_fourier": LIGHTGBM_QMEAN,  # weather + Fourier
    # Split-conformal calibrated fan (model_params.calibrate) — widens the
    # under-covering upper quantiles so scenario/CVaR dispatch sees spike risk
    # the raw fan misses (Entry 033).
    "lightgbm_qmean_cal": LIGHTGBM_QMEAN,
    # DEPRECATED — never developed enough to be competitive; their predict has
    # the frozen-tail bug (can't reforecast under MPC) and is left unfixed. No
    # results depend on them (old runs are parked in outputs/_archive_valtest_trials/).
    # Registered so the code still loads/tests; see the fit docstrings.
    "simple_mlp": SIMPLE_MLP,   # deprecated
    "lstm": LSTM,               # deprecated
}


def get_model(name: str) -> dict:
    """Look up a model spec by name.

    Args:
        name: Model name string (must be a key in REGISTRY).

    Returns:
        Model spec dict with fit/predict/save/load functions.

    Raises:
        KeyError: If the model name is not registered.
    """
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown model {name!r}. "
            f"Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]


# Registered model name (including aliases) → its typed params class.
PARAMS_FOR: dict[str, type] = {
    name: _PARAMS_BY_SPEC_NAME[spec["name"]] for name, spec in REGISTRY.items()
}


def params_for(name: str) -> type:
    """Return the typed params class for a registered model name.

    Args:
        name: A key in :data:`REGISTRY` (canonical or alias).

    Returns:
        The `pydantic` params class the model's ``fit`` parses ``model_params``
        into (e.g. :class:`~grian.models.params.LGBMQMeanParams`).

    Raises:
        KeyError: If the model name is not registered.
    """
    if name not in PARAMS_FOR:
        raise KeyError(
            f"Unknown model {name!r}. Available: {sorted(PARAMS_FOR)}"
        )
    return PARAMS_FOR[name]
