"""Quantile regression averaging (QRA) for forecast combination.

Combines quantile forecasts from multiple models (LEAR, GBT, neural net)
into a single calibrated probabilistic forecast using quantile regression
on the cross-validated predictions.
"""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import QuantileRegressor


class QRA:
    """Quantile regression averaging combiner.

    Fits a quantile regression at each level to optimally weight the
    input models' quantile forecasts.
    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        alpha: float = 0.0,
        solver: str = "highs",
    ):
        """Initialise the QRA combiner.

        Args:
            quantiles: Target quantile levels.  Defaults to the full set
                from ``config.yaml``.
            alpha: L1 regularisation strength for the quantile regression.
                Zero means no regularisation.
            solver: Solver for ``QuantileRegressor`` (default ``"highs"``).
        """
        self.quantiles = quantiles or [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        self.alpha = alpha
        self.solver = solver
        self.models_: dict[int, QuantileRegressor] = {}
        self.model_names_: list[str] = []

    def fit(
        self,
        forecasts: dict[str, ArrayLike],
        actual: ArrayLike,
    ) -> "QRA":
        """Fit the combination weights from cross-validated forecasts.

        For each quantile level *q*, a ``QuantileRegressor`` learns the
        optimal weighting of each model's *q*-th quantile forecast to
        minimise pinball loss against actuals.

        Args:
            forecasts: Dict mapping model name to quantile forecast arrays,
                each of shape ``(n, len(self.quantiles))``.
            actual: Actual values, shape ``(n,)``.

        Returns:
            Self, for method chaining.
        """
        self.model_names_ = sorted(forecasts.keys())
        actual = np.asarray(actual).ravel()
        n = len(actual)

        # Stack model forecasts: for each quantile index, build the
        # design matrix from each model's forecast at that quantile.
        forecast_arrays = {
            name: np.asarray(forecasts[name]) for name in self.model_names_
        }

        for qi, q in enumerate(self.quantiles):
            # Design matrix: each column is one model's q-th quantile forecast
            X = np.column_stack(
                [forecast_arrays[name][:n, qi] for name in self.model_names_]
            )
            model = QuantileRegressor(
                quantile=q,
                alpha=self.alpha,
                solver=self.solver,
                fit_intercept=True,
            )
            model.fit(X, actual)
            self.models_[qi] = model

        return self

    def predict(self, forecasts: dict[str, ArrayLike]) -> np.ndarray:
        """Produce combined quantile forecasts.

        Args:
            forecasts: Dict mapping model name to quantile forecast arrays,
                each of shape ``(n, len(self.quantiles))``.

        Returns:
            Combined quantile forecasts, shape ``(n, len(self.quantiles))``.
        """
        forecast_arrays = {
            name: np.asarray(forecasts[name]) for name in self.model_names_
        }
        n = len(next(iter(forecast_arrays.values())))
        result = np.empty((n, len(self.quantiles)))

        for qi in range(len(self.quantiles)):
            X = np.column_stack(
                [forecast_arrays[name][:n, qi] for name in self.model_names_]
            )
            result[:, qi] = self.models_[qi].predict(X)

        # Enforce monotonicity: quantiles must be non-decreasing
        for i in range(n):
            result[i] = np.sort(result[i])

        return result

    def weights(self) -> dict[int, dict[str, float]]:
        """Return the fitted combination weights per quantile level.

        Returns:
            Dict mapping quantile index to a dict of model-name -> weight.
        """
        w: dict[int, dict[str, float]] = {}
        for qi, model in self.models_.items():
            w[qi] = {
                name: float(coef)
                for name, coef in zip(self.model_names_, model.coef_)
            }
        return w
