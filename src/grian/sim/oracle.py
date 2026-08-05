"""Perfect-foresight oracle for capture ratio.

The oracle is the frozen denominator: a single
LP over the full evaluation window against actual prices, with SOC
continuous across days, soc0 = 0, correct dt, and a per-calendar-day
discharge-throughput constraint enforcing the cycle limit.

The result is cached to parquet — it never changes while the data and
battery spec are fixed.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from grian.sim import lp

logger = logging.getLogger(__name__)


def compute_oracle(
    prices: pd.Series,
    *,
    dt_hours: float,
    power_mw: float = 100.0,
    duration_hours: float = 2.0,
    efficiency: float = 0.85,
    max_cycles: int = 2,
    block_days: int = 7,
    cache_path: str | Path | None = None,
) -> dict:
    """Compute perfect-foresight dispatch over a full price window.

    Args:
        prices: Actual prices, DatetimeIndex, one row per interval.
        dt_hours: Interval length in hours.
        power_mw: Battery power (MW).
        duration_hours: Battery duration (h); capacity = power * duration.
        efficiency: Round-trip efficiency.
        max_cycles: Full cycles allowed per calendar day.
        block_days: Solve in blocks of this many calendar days with SOC
            pinned to 0 at block boundaries. The full-window LP scales
            superlinearly in HiGHS (~20 s for 7 days, ~3 min for 21);
            weekly blocks keep the oracle tractable and cost almost
            nothing — overnight carry across a midnight boundary is
            worth ~0 under a per-day cycle limit.
        cache_path: If given, load from / save to this parquet path.
            A sidecar ``.meta.json`` stores the battery spec; a spec
            mismatch invalidates the cache.

    Returns:
        Dict with ``schedule_df`` (charge/discharge/soc/revenue per
        interval), ``daily_revenue`` (Series by date), ``total_revenue``.
    """
    capacity_mwh = power_mw * duration_hours
    spec = {
        "block_days": block_days,
        "dt_hours": dt_hours,
        "power_mw": power_mw,
        "capacity_mwh": capacity_mwh,
        "efficiency": efficiency,
        "max_cycles": max_cycles,
        "start": str(prices.index[0]),
        "end": str(prices.index[-1]),
        "n": len(prices),
    }

    if cache_path is not None:
        cache_path = Path(cache_path)
        meta_path = cache_path.with_suffix(".meta.json")
        if cache_path.exists() and meta_path.exists():
            if json.loads(meta_path.read_text()) == spec:
                schedule_df = pd.read_parquet(cache_path)
                return _package(schedule_df)
            logger.info("Oracle cache spec mismatch — recomputing")

    # Group interval positions by calendar day, then solve in blocks of
    # block_days with SOC = 0 at block boundaries and a per-day
    # discharge-throughput budget inside each block.
    days = prices.index.normalize()
    positions = pd.Series(np.arange(len(prices)), index=days)
    day_bounds = [
        (int(idx.iloc[0]), int(idx.iloc[-1]) + 1)
        for _, idx in positions.groupby(level=0)
    ]

    charge = np.zeros(len(prices))
    discharge = np.zeros(len(prices))
    soc = np.zeros(len(prices))
    for b in range(0, len(day_bounds), block_days):
        block = day_bounds[b: b + block_days]
        start, end = block[0][0], block[-1][1]
        budgets = [
            (d0 - start, d1 - start, max_cycles * capacity_mwh)
            for d0, d1 in block
        ]
        result = lp.solve_lp(
            prices.values[start:end],
            dt_hours=dt_hours,
            power_mw=power_mw,
            capacity_mwh=capacity_mwh,
            efficiency=efficiency,
            soc0=0.0,
            terminal_soc=0.0,
            throughput_budgets=budgets,
        )
        if result["status"] != "optimal":
            raise RuntimeError(f"Oracle LP failed: {result['status']}")
        charge[start:end] = result["charge"]
        discharge[start:end] = result["discharge"]
        soc[start:end] = result["soc"][1:]
        logger.debug("Oracle block %d-%d done", b, b + len(block))

    schedule_df = pd.DataFrame(
        {
            "actual_price": prices.values,
            "charge_mw": charge,
            "discharge_mw": discharge,
            "soc_mwh": soc,
            "revenue": prices.values * (discharge - charge) * dt_hours,
        },
        index=prices.index,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        schedule_df.to_parquet(cache_path)
        meta_path.write_text(json.dumps(spec, indent=2))

    return _package(schedule_df)


def _package(schedule_df: pd.DataFrame) -> dict:
    """Assemble the oracle result dict from a schedule DataFrame.

    Args:
        schedule_df: Interval-level oracle schedule with a revenue column.

    Returns:
        Dict with schedule_df, daily_revenue, and total_revenue.
    """
    daily = schedule_df["revenue"].resample("D").sum()
    _assert_closed_cycles_nonnegative(schedule_df)
    return {
        "schedule_df": schedule_df,
        "daily_revenue": daily,
        "total_revenue": float(schedule_df["revenue"].sum()),
    }


def _assert_closed_cycles_nonnegative(
    schedule_df: pd.DataFrame, tol: float = 1.0
) -> None:
    """Assert the oracle never loses money over any closed cycle.

    Perfect foresight cannot lose money over a cycle that starts and ends
    empty: doing nothing earns exactly zero, so the optimum is >= 0. The
    honest granularity is the *closed cycle*, **not** the calendar day —
    SOC is pinned to 0 only at block boundaries, so a single day can be
    net-negative when the oracle optimally charges cheap overnight to
    discharge into the next morning's spike (a net-buyer day). We therefore
    split wherever SOC returns to empty and check each such segment: each
    is independently feasible-to-skip, so at the optimum each must be >= 0.

    Args:
        schedule_df: Interval-level schedule with ``soc_mwh`` and ``revenue``.
        tol: Dollar tolerance absorbing float round-off in the segment sums.
    """
    soc = schedule_df["soc_mwh"].to_numpy()
    rev = schedule_df["revenue"].to_numpy()
    ends = np.flatnonzero(np.abs(soc) < 1e-6)  # intervals that finish empty
    start = 0
    for end in ends:
        seg = float(rev[start : end + 1].sum())
        assert seg >= -tol, (
            f"Oracle closed cycle {schedule_df.index[start]}..."
            f"{schedule_df.index[end]} lost ${-seg:.2f}"
        )
        start = end + 1
