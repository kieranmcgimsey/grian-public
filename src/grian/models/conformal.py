"""Conformal prediction wrapper for coverage calibration.

Wraps any quantile forecaster to provide finite-sample coverage
guarantees via split conformal prediction. The conformity scores are
computed on a calibration set and used to adjust quantile widths.
"""

import numpy as np
from numpy.typing import ArrayLike


class ConformalWrapper:
    """Coverage-calibrated wrapper for quantile forecasts.

    Uses split conformal prediction to adjust quantile forecasts so that
    empirical coverage matches the nominal level.  For each quantile *q*,
    the adjustment is computed as the ``(1 - q)``-th quantile of the
    conformity scores on the calibration set.
    """

    def __init__(self, quantiles: list[float] | None = None):
        """Initialise the conformal wrapper.

        Args:
            quantiles: Quantile levels to calibrate.
        """
        self.quantiles = quantiles or [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        self.adjustments_: np.ndarray | None = None
        self.scores_: np.ndarray | None = None

    def calibrate(
        self,
        quantile_forecasts: ArrayLike,
        actual: ArrayLike,
    ) -> "ConformalWrapper":
        """Compute conformity scores on a calibration set.

        For each quantile level, the conformity score measures how far
        the actual value falls outside the predicted quantile.  Lower
        quantiles need a *downward* adjustment (actual should be above
        the lower bound), and upper quantiles need an *upward* adjustment
        (actual should be below the upper bound).

        Args:
            quantile_forecasts: Quantile predictions, shape ``(n, q)``.
            actual: Actual values, shape ``(n,)``.

        Returns:
            Self, for method chaining.
        """
        qf = np.asarray(quantile_forecasts)
        actual = np.asarray(actual).ravel()
        n, nq = qf.shape

        # Conformity scores: signed residual for each quantile
        # For lower quantiles (q < 0.5): score = predicted_q - actual
        #   (positive when prediction is too high, i.e. under-covers)
        # For upper quantiles (q > 0.5): score = actual - predicted_q
        #   (positive when prediction is too low, i.e. under-covers)
        # For the median (q = 0.5): use the absolute residual sign convention
        scores = np.empty((n, nq))
        for qi, q in enumerate(self.quantiles):
            if q <= 0.5:
                # Lower quantile: shift prediction downward to cover more
                scores[:, qi] = qf[:, qi] - actual
            else:
                # Upper quantile: shift prediction upward to cover more
                scores[:, qi] = actual - qf[:, qi]

        self.scores_ = scores

        # Adjustment = quantile of scores at the appropriate conformal level
        # Use ceil((n+1)*level)/n correction for finite-sample validity
        adjustments = np.empty(nq)
        for qi, q in enumerate(self.quantiles):
            # The conformal quantile level for guaranteed coverage
            level = np.ceil((n + 1) * (1 - q if q <= 0.5 else q)) / n
            level = min(level, 1.0)
            adjustments[qi] = np.quantile(scores[:, qi], level)

        self.adjustments_ = adjustments
        return self

    def adjust(self, quantile_forecasts: ArrayLike) -> np.ndarray:
        """Apply conformal adjustments to new quantile forecasts.

        Lower quantiles are shifted downward and upper quantiles are
        shifted upward so that empirical coverage meets or exceeds the
        nominal level.

        Args:
            quantile_forecasts: Raw quantile predictions, shape ``(n, q)``.

        Returns:
            Adjusted quantile forecasts with calibrated coverage.
        """
        if self.adjustments_ is None:
            raise RuntimeError("Call calibrate() before adjust().")

        qf = np.asarray(quantile_forecasts).copy()

        for qi, q in enumerate(self.quantiles):
            if q <= 0.5:
                # Widen lower bound downward
                qf[:, qi] -= self.adjustments_[qi]
            else:
                # Widen upper bound upward
                qf[:, qi] += self.adjustments_[qi]

        # Enforce monotonicity after adjustment
        for i in range(qf.shape[0]):
            qf[i] = np.sort(qf[i])

        return qf


def main() -> None:
    """Run this module as a CLI (exposes its public callables)."""
    from grian._cli import run_module_cli

    run_module_cli(globals())


if __name__ == "__main__":
    main()
