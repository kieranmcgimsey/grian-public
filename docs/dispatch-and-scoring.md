# Dispatch and scoring — the physics and the metric

This is the guide to the part of the system that must be *exactly* right: the
battery model, the two executors, the oracle, and the capture ratio. Every
number the campaign reports rests on this being correct, and it was catastrophically
wrong for the repo's first dozen experiments (Entry 013). Read this before you
touch anything under `dispatch/battery_lp.py`, `dispatch/oracle.py`, `dispatch/mpc.py`, or `open_loop.battery_dispatch`.

## The battery

| Parameter | Value | In code |
|---|---|---|
| Power | 100 MW charge/discharge | `dispatch.power_mw` |
| Energy | 200 MWh (2 h) | `power_mw × dispatch.duration_hours` |
| Round-trip efficiency | 0.85 (√0.85 each way) | `dispatch.efficiency` |
| Cycle limit | 2 full cycles/day = 400 MWh discharge throughput/day | `dispatch.max_cycles` |
| Interval | 5 min → `dt = 1/12 h` | from `cfg["resolution"]` |

SOC evolves as `soc[t+1] = soc[t] + charge·η·dt − discharge/η·dt`, bounded to
`[0, 200]`. Revenue for an interval is `actual_price × (discharge − charge) × dt`.

## The arbitrage LP (`lp.solve_lp`)

Given a price vector, `solve_lp` finds the charge/discharge schedule that
maximises `Σ price·(discharge − charge)·dt` subject to power, energy,
efficiency, and per-window discharge-throughput (cycle) limits. It is a sparse
HiGHS linear program — ~10–50× faster than the cvxpy reference in
`grian/dispatch/cvxpy_reference.py`, and it scales from one day (T=288) to the full-window
oracle (T≈35 000) without densifying.

```python
result = lp.solve_lp(
    prices,                       # np.ndarray, length T
    dt_hours=1/12,                # or an array of per-step durations (telescoping)
    power_mw=100, capacity_mwh=200, efficiency=0.85,
    soc0=current_soc,             # plan from the TRUE state, not 0
    terminal_soc=None,            # or pin the end SOC (oracle blocks use 0)
    throughput_budgets=[(0, T, 400.0)],   # (start, end, budget_mwh) per day
)
# -> {"charge", "discharge", "soc" (len T+1), "revenue", "status"}
```

The cvxpy version (`grian.dispatch.cvxpy_reference.schedule`) is the *readable reference*; a
unit test asserts the two agree to 1e-4 on random price vectors at both
resolutions. Keep them in sync if you change the model.

## The feasibility clamp

`lp.clamp_action` is the single gate every executed action passes through. It
makes a planned `(charge, discharge)` physically possible given the current SOC
and the remaining daily throughput budget:

```python
charge, discharge, new_soc = lp.clamp_action(
    charge_mw, discharge_mw, soc_mwh,
    dt_hours=1/12, power_mw=100, capacity_mwh=200, efficiency=0.85,
    discharge_budget_mwh=budget_left,   # remaining cycles today
)
```

Discharge is capped by stored energy and remaining budget; charge by the
headroom left after discharge. **Revenue is computed only from clamped values.**
This is trap T2: the original executor monetised the LP's raw schedule and
clamped only the *recorded* SOC, so the battery got paid for energy it never
held. Never book revenue from an unclamped plan.

## The oracle (`oracle.compute_oracle`) — the denominator

Perfect-foresight dispatch over a whole window against *actual* prices, with SOC
continuous across days, `soc0 = 0`, correct `dt`, and a per-calendar-day cycle
budget. This is the capture-ratio denominator and it is **frozen** — changing
its definition invalidates every capture number.

```python
orc = oracle.compute_oracle(
    prices,                       # pd.Series, DatetimeIndex
    dt_hours=lp.resolution_dt_hours("5min"),
    power_mw=100, duration_hours=2, efficiency=0.85, max_cycles=2,
    block_days=7,                 # solve in weekly blocks, SOC pinned 0 at seams
    cache_path="outputs/trials/_oracle/SA1/val.parquet",
)
# -> {"schedule_df", "daily_revenue" (Series by day), "total_revenue"}
```

Why weekly blocks: the full-window HiGHS LP scales superlinearly (~20 s for 7
days, ~3 min for 21). Weekly blocks with SOC pinned to 0 at the seams keep it
tractable at negligible cost — overnight carry across a midnight is worth ≈0
under a 2-cycle/day limit. The result is cached with a spec sidecar; a spec
mismatch (different battery or window) triggers a recompute.

**The self-consistency test (the anchor of the whole scoreboard):** replay the
oracle's own schedule through `clamp_action` and you recover its revenue to
1e-4 — capture ratio 1.000 by construction. This is
`tests/test_sim_lp.py::TestOracle`. If it ever fails, the planner, executor, and
scorer have drifted out of agreement; fix that before trusting any result.

## The open-loop executor (`open_loop.battery_dispatch`)

The default. Once per day: forecast → one LP solve from the carried SOC →
execute all 288 intervals with `clamp_action`, carrying SOC and the daily budget
across days. Simple and fast; the honest baseline.

## The MPC loop (`mpc.simulate_region_mpc`)

The champion executor. It walks the window and, as it goes:

1. **Refits** the model weekly (at midnight, on all data so far).
2. **Re-forecasts** every `reforecast_every` intervals, from *all data observed
   up to now* (predict-from-now). This is the value engine: a 1-hour-ahead
   forecast is far sharper than an 18-hour-ahead one.
3. **Re-solves** the LP every `resolve_every` intervals from the *true* SOC and
   remaining budget, over a **telescoped** horizon (native 5-min resolution for
   the first hour, 30-min blocks beyond — this cuts the LP from ~288 to ~58
   variables at no accuracy cost because only the near steps are ever executed).
4. **Executes** each block through `clamp_action` and books it to the ledger.

Config (`cfg["mpc"]`):

```python
{"resolve_every": 6,        # re-solve LP every 6 intervals (30 min)
 "reforecast_every": 6,     # re-forecast every 6 intervals (30 min) — champion
 "persistence_tau": 0,      # last-price blend decay (0 = off; rejected, Entry 018)
 "persistence_gate": 0}     # only blend above this price ($) if tau > 0
```

What the ablations established (Entries 018–019):
- **Re-forecast frequency pays**, and pays far more on test (fast midday spikes)
  than validation: 60→30 min was +1 pt val but +5.4 pt test.
- **Re-solve frequency alone does nothing** — without a new forecast the LP
  reproduces its plan.
- **Persistence blending is rejected** — 5-min NEM prices mean-revert too hard;
  blending toward the last price chases noise. Gating to extremes only breaks
  even, because the model's recency features already carry that signal.

## Capture ratio and regret (`analytics.capture_report`)

```python
rep = capture_report(ledger_df, oracle_daily)
```

- `capture_ratio` = realised revenue ÷ oracle revenue over the common days.
  **Asserted ≤ 1** (trap T7).
- `regret_daily` = oracle − realised, per day. Where the money leaks.
- `top10_regret` = the ten worst days, with per-day capture. Always read this.
- `oracle_top10_share` = how concentrated the oracle's revenue is (context for
  how noisy the capture number is).
- `mean_daily_spearman` = mean within-day rank correlation of forecast vs
  actual. Diagnostic: high rank skill + low capture ⇒ magnitude problem.

## The trap register (the ways to void your results)

These are the failure modes that already bit this repo — the ones that live in
this layer:

### Trap 1: the dt bug

`dt` was hardcoded to 0.5 h (30 min) in the LP while sims ran at 5-min
resolution. The planner believed each interval moved 6× more energy than physics
allows — it "filled" the battery in 4 intervals and treated a day as 144 hours.
**Every executor and LP must take `dt` from `cfg["resolution"]` via
`lp.resolution_dt_hours`.** Grep for literal `0.5` near any dispatch code in
review. This single bug voided Entries 001–012.

### Trap 2: phantom energy

Revenue only ever from clamped actions (above). The ledger asserts SOC bounds on
every record; never bypass it.

### Trap 3: median vs mean

A model trained with pinball loss at α=0.5 predicts the *median* of a
right-skewed price distribution — it understates spikes, and the LP (which sizes
cycles by magnitude) under-trades. `lightgbm_qmean` integrates a quantile set to
the *mean* in dollar space to fix this. Do not feed medians to the LP and expect
spike value.

### Trap 5: embargo confusion

Sim embargo = 0 (deployment realism); backtest embargo = horizon (selection
hygiene). Different knobs. Do not unify them.

### Trap 7: capture > 1

Impossible by definition — it means the oracle is wrong or the executor is
cheating. `capture_report` asserts it. A negative oracle *day* is likewise
impossible (doing nothing earns 0) and is asserted in `oracle.py`.

## The tests that pin all this

- `tests/test_sim_lp.py` — HiGHS ≡ cvxpy; clamp feasibility; oracle replay = 1.0.
- `tests/test_sim_mpc.py` — predict-from-now; per-day budgets; **perfect
  forecaster through MPC captures 1.0** (the end-to-end executor proof).

Run them after any change to this layer: `pytest tests/test_sim_lp.py
tests/test_sim_mpc.py -q`.
