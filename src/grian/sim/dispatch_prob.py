"""Probabilistic dispatch: turn a quantile *fan* into robust prices for the LP.

The deterministic LP takes a single price vector. To make it uncertainty-aware
without changing the LP, we feed it a **direction-pessimistic** price vector built
from the forecast fan (see docs/probabilistic-dispatch-explained.md):

* steps that look like a **sell** (price above the horizon median) get a **low**
  quantile — don't bank on the upside;
* steps that look like a **buy** (below the median) get a **high** quantile — don't
  bank on the price dropping.

The LP then only trades when the spread survives the pessimistic view — a
probabilistic no-trade band. Widening the quantiles (q_low→0.02, q_high→0.98)
turns the same mechanism into a robust/worst-case dispatch.
"""

import numpy as np


def _nearest_quantile(fan: dict, target: float) -> float:
    """Key of ``fan`` whose quantile level is nearest ``target``."""
    return min(fan, key=lambda q: abs(q - target))


def quantile_gate_prices(
    fan: dict[float, np.ndarray],
    q_low: float = 0.1,
    q_high: float = 0.9,
) -> np.ndarray:
    """Direction-pessimistic price vector from a quantile fan.

    Args:
        fan: ``{quantile_level: price_array}`` over the horizon (dollar space).
        q_low: Target lower quantile for likely-sell steps (nearest available used).
        q_high: Target upper quantile for likely-buy steps.

    Returns:
        A single price vector for ``lp.solve_lp`` with spreads compressed in the
        direction that matters, so only confident arbitrage survives.
    """
    med_key = _nearest_quantile(fan, 0.5)
    lo_key = _nearest_quantile(fan, q_low)
    hi_key = _nearest_quantile(fan, q_high)
    p50 = np.asarray(fan[med_key], dtype=float)
    lo = np.asarray(fan[lo_key], dtype=float)
    hi = np.asarray(fan[hi_key], dtype=float)

    ref = float(np.median(p50))
    # Above the day's median → likely discharge → be pessimistic (low quantile).
    # Below → likely charge → be pessimistic (high quantile).
    return np.where(p50 >= ref, lo, hi)


def quantile_weights(quantiles: list[float]) -> np.ndarray:
    """Discrete probability mass for each quantile level (sorted order).

    Maps quantile *levels* to a probability distribution over the
    corresponding price paths: each level gets the mass of the interval
    running halfway to each neighbour, with the tails extending to 0 and 1.
    A fan ``[0.05, 0.5, 0.9, 0.98]`` becomes weights that sum to 1 with the
    median carrying the most mass — the natural weighting for taking an
    expected action across scenario paths.

    Args:
        quantiles: Quantile levels present in the fan (any order).

    Returns:
        Weights aligned with ``sorted(quantiles)``, summing to 1.
    """
    qs = sorted(quantiles)
    edges = [0.0] + [(a + b) / 2 for a, b in zip(qs, qs[1:])] + [1.0]
    w = np.diff(edges)
    return w / w.sum()


def combine_scenario_actions(
    charges: np.ndarray,
    discharges: np.ndarray,
    weights: np.ndarray,
    mode: str,
    lam: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse per-scenario LP actions into one dispatch under uncertainty.

    Each scenario is a quantile price *path* already solved by its own LP
    (so each path's cross-interval ranking is preserved — the property the
    price-space gate destroyed). This fuses their actions in *decision*
    space:

    * ``ev`` — probability-weighted expected net action (risk-neutral).
    * ``cvar`` — a ``lam`` blend from the expected action (``lam=0``) toward
      the robust worst-case action (``lam=1``), where the worst case takes
      the minimum charge and minimum discharge across scenarios (trade only
      where every scenario agrees). ``lam`` is the risk-aversion knob.

    Net actions are recombined into non-negative charge/discharge (a convex
    blend of feasible plans stays feasible for the linear battery), avoiding
    simultaneous charge and discharge.

    Args:
        charges: ``(n_scenarios, horizon)`` charge power per scenario (MW).
        discharges: ``(n_scenarios, horizon)`` discharge power per scenario.
        weights: Per-scenario probabilities, aligned rows, summing to 1.
        mode: ``"ev"`` or ``"cvar"``.
        lam: Risk-aversion weight in ``[0, 1]`` for ``cvar``.

    Returns:
        ``(charge, discharge)`` arrays over the horizon.
    """
    nets = discharges - charges
    w = weights / weights.sum()
    net_ev = (w[:, None] * nets).sum(axis=0)
    if mode == "ev":
        net = net_ev
    elif mode == "cvar":
        net_robust = discharges.min(axis=0) - charges.min(axis=0)
        net = (1.0 - lam) * net_ev + lam * net_robust
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"unknown scenario mode {mode!r}")
    return np.maximum(-net, 0.0), np.maximum(net, 0.0)
