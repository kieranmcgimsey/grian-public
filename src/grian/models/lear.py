"""LASSO-estimated autoregressive (LEAR) models.

Per-hour LASSO models that form the strong linear benchmark a network
must beat. Feature selection via L1 regularisation reveals which inputs
drive price at each delivery hour.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV


class LEAR:
    """Per-hour LASSO autoregressive model for day-ahead price forecasting.

    Fits one LASSO regression per delivery half-hour (or a single global
    model), using lagged prices, calendar features, and exogenous inputs.
    """

    def __init__(
        self,
        alpha: float | None = None,
        per_hour: bool = True,
        refit_window: int | None = None,
        n_alphas: int = 20,
        cv: int = 5,
        seed: int = 42,
    ):
        """Initialise the LEAR model.

        Args:
            alpha: LASSO regularisation strength.  When *None* (default),
                ``LassoCV`` selects the best alpha via cross-validation.
            per_hour: If True, fit a separate model per delivery half-hour.
            refit_window: Rolling window size for refitting (None = expanding).
            n_alphas: Number of alphas to test in the CV path.
            cv: Number of cross-validation folds.
            seed: Random seed for reproducibility.
        """
        self.alpha = alpha
        self.per_hour = per_hour
        self.refit_window = refit_window
        self.n_alphas = n_alphas
        self.cv = cv
        self.seed = seed
        self.models_: dict[int, LassoCV] = {}
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LEAR":
        """Fit the model(s) on training data.

        When ``per_hour`` is True, one LassoCV model is fitted for each
        of the 48 half-hours in a day.  Otherwise a single global model
        is fitted on all training rows.

        Args:
            X: Feature matrix with a ``DatetimeIndex``.
            y: Target price series aligned with *X*.

        Returns:
            Self, for method chaining.
        """
        X = X.copy()
        y = y.reindex(X.index)
        self.feature_names_ = list(X.columns)

        if self.per_hour:
            # Determine the half-hour slot (0-47) from the index
            hh = X.index.hour * 2 + X.index.minute // 30
            for slot in range(48):
                mask = hh == slot
                X_slot = X.loc[mask]
                y_slot = y.loc[mask]
                if len(X_slot) < 10:
                    continue
                model = self._make_lasso()
                model.fit(X_slot.values, y_slot.values)
                self.models_[slot] = model
        else:
            model = self._make_lasso()
            model.fit(X.values, y.values)
            self.models_[0] = model

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate day-ahead point forecasts.

        Args:
            X: Feature matrix for the forecast window.

        Returns:
            Predicted price series with the same index as *X*.
        """
        preds = np.empty(len(X))
        if self.per_hour:
            hh = X.index.hour * 2 + X.index.minute // 30
            for slot in range(48):
                mask = hh == slot
                if not mask.any():
                    continue
                if slot in self.models_:
                    preds[mask] = self.models_[slot].predict(X.loc[mask].values)
                else:
                    # Fall back to global mean if slot model missing
                    preds[mask] = 0.0
        else:
            preds = self.models_[0].predict(X.values)

        return pd.Series(preds, index=X.index, name="forecast")

    def selected_features(self) -> dict[int, list[str]]:
        """Return the features with non-zero coefficients per hour.

        Returns:
            Dict mapping delivery half-hour slot (0-47) to a list of
            selected feature names.
        """
        result: dict[int, list[str]] = {}
        for slot, model in self.models_.items():
            coefs = model.coef_
            nonzero = np.nonzero(coefs)[0]
            result[slot] = [self.feature_names_[i] for i in nonzero]
        return result

    def coef_matrix(self) -> pd.DataFrame:
        """Return a DataFrame of coefficients (rows = slots, cols = features).

        Returns:
            DataFrame of shape ``(n_slots, n_features)`` with LASSO
            coefficients.  Zero entries were dropped by regularisation.
        """
        rows = {}
        for slot, model in sorted(self.models_.items()):
            rows[slot] = model.coef_
        return pd.DataFrame(rows, index=self.feature_names_).T

    def _make_lasso(self) -> LassoCV:
        """Create a fresh LassoCV instance with the model's hyper-parameters."""
        kwargs: dict = {
            "n_alphas": self.n_alphas,
            "cv": self.cv,
            "random_state": self.seed,
            "max_iter": 5000,
        }
        if self.alpha is not None:
            kwargs["alphas"] = [self.alpha]
        return LassoCV(**kwargs)
