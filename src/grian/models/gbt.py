"""LightGBM quantile regression for day-ahead price forecasting."""

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore[assignment]


class GBTQuantile:
    """Gradient-boosted tree model producing quantile price forecasts.

    Wraps LightGBM with quantile regression objective.  One model per
    quantile level produces a full predictive distribution fan.
    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        seed: int = 42,
    ):
        """Initialise the GBT quantile model.

        Args:
            quantiles: Target quantile levels.  Defaults to a standard set.
            n_estimators: Number of boosting rounds.
            learning_rate: Step size shrinkage.
            max_depth: Maximum tree depth.
            num_leaves: Maximum number of leaves per tree.
            min_child_samples: Minimum samples in a leaf.
            subsample: Row sub-sampling fraction.
            colsample_bytree: Column sub-sampling fraction.
            seed: Random seed for reproducibility.
        """
        self.quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.seed = seed
        self.models_: dict[float, lgb.LGBMRegressor] = {}
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GBTQuantile":
        """Fit one LightGBM model per quantile level.

        Args:
            X: Feature matrix.
            y: Target price series.

        Returns:
            Self, for method chaining.
        """
        if lgb is None:
            raise ImportError("LightGBM is required: pip install lightgbm")

        self.feature_names_ = list(X.columns)
        X_arr = X.values
        y_arr = np.asarray(y)

        for q in self.quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.seed,
                verbosity=-1,
                n_jobs=-1,
            )
            model.fit(X_arr, y_arr)
            self.models_[q] = model

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict quantiles for the forecast window.

        Args:
            X: Feature matrix.

        Returns:
            DataFrame with one column per quantile level, indexed like *X*.
        """
        X_arr = X.values
        cols = {}
        for q in self.quantiles:
            cols[f"q{q:.2f}"] = self.models_[q].predict(X_arr)

        return pd.DataFrame(cols, index=X.index)

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance averaged across quantile models.

        Returns:
            DataFrame with columns ``feature`` and ``importance``,
            sorted descending by importance.
        """
        importances = np.zeros(len(self.feature_names_))
        for model in self.models_.values():
            importances += model.feature_importances_

        importances /= len(self.models_)
        df = pd.DataFrame(
            {"feature": self.feature_names_, "importance": importances}
        )
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
