"""Tests for the typed model hyperparameters (grian.models.params).

Pins three things: the typed defaults still equal the old hardcoded literals
(behaviour-preserving), the surface is typo-protected (``extra="forbid"``), and
``params_for`` resolves every registered model — including aliases.
"""

import pytest

from grian.models import PARAMS_FOR, REGISTRY, params_for
from grian.models.params import (
    ARParams,
    LearTorchParams,
    LGBMParams,
    LGBMQMeanParams,
    LGBMRichParams,
    LinearParams,
    default_lags,
)


def test_lgbm_defaults_match_legacy():
    """The typed LightGBM kwargs equal the pre-refactor hardcoded dict."""
    assert LGBMParams().to_lgb_kwargs() == {
        "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
        "max_depth": -1, "min_child_samples": 20, "subsample": 0.8,
        "colsample_bytree": 0.8, "verbosity": -1, "n_jobs": -1,
    }


def test_family_specific_defaults():
    """Each family keeps its own defaults (qmean is lighter, ordinal calendar)."""
    assert LGBMQMeanParams().n_estimators == 150
    assert LGBMQMeanParams().calendar_encoding == "ordinal"
    assert LGBMQMeanParams().quantiles == [0.05, 0.5, 0.9, 0.98]
    assert LGBMRichParams().calendar_encoding == "ordinal"
    assert LGBMRichParams().sample_weighting is None
    assert LinearParams().calendar_encoding == "onehot"
    assert LinearParams().feature_set == "full"
    assert LearTorchParams().epochs == 400 and LearTorchParams().alpha == 0.01


def test_frozen_and_typo_protected():
    """Unknown keys raise; instances are immutable."""
    with pytest.raises(Exception, match="extra|forbid|permitted"):
        LGBMParams.model_validate({"n_estimatorss": 10})
    p = LGBMParams()
    with pytest.raises(Exception):
        p.n_estimators = 5  # frozen


def test_lgb_extra_escape_hatch():
    """Arbitrary LightGBM kwargs pass through the typed escape hatch."""
    p = LGBMParams.model_validate({"lgb_extra": {"reg_lambda": 0.3}})
    assert p.to_lgb_kwargs()["reg_lambda"] == 0.3


def test_resolution_dependent_lags():
    """lags/step_stride default to None and resolve against the resolution."""
    assert ARParams().lags is None
    assert default_lags(288) == [288, 576, 2016]
    assert default_lags(48) == [48, 96, 336]


def test_params_for_covers_every_model():
    """Every registered name (aliases included) maps to a params class."""
    assert set(PARAMS_FOR) == set(REGISTRY)
    assert params_for("lightgbm_qmean_weather_fourier") is LGBMQMeanParams
    assert params_for("lear_weather") is LinearParams
    assert params_for("autoregression_ordinal") is ARParams
    with pytest.raises(KeyError, match="Unknown model"):
        params_for("does_not_exist")
