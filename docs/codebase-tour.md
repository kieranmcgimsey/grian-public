# Learning the grian codebase — a progressive tour

*A staged reading plan for actually understanding the code, not just the ideas.
Where things live, what to read in what order, which function to open next, and a
small thing to run at the end of each stage so the knowledge sticks. Work through
the stages in order; each one assumes the last.*

This is the **map and the route**. Two companions go with it:
[`architecture.md`](architecture.md) is the *reference* wiring diagram (open it
when a stage says to), and [`internals-deep-dive.md`](internals-deep-dive.md)
teaches the *concepts and maths* behind what the code is doing (read a chapter of
it alongside the matching stage). This document's job is to walk you through the
*source* in a sensible order.

A tip that applies throughout: **follow the call, don't just read the file.** When
a function calls another, jump to it, understand it, then come back. Your editor's
"go to definition" is the most important tool here. And keep a scratch notebook or
REPL open — every stage ends with something small to run.

---

## Part 0 — Orientation (30 minutes)

### The one-paragraph mental model

grian forecasts South-Australian electricity prices and dispatches a battery to
trade them, scoring the result against a perfect-foresight benchmark. There are
**two halves that share one library**: a teaching **curriculum** (the ten
notebooks plus the modules directly under `src/grian/`) and a **simulation
environment** (`src/grian/sim/`) where the real research — the "capture-ratio
campaign" — happens. Almost everything interesting lives in `sim/`.

### The shape of the repository

```
src/grian/          the library
  config.py         load the YAML config, find the repo root
  data.py           load & clean NEM prices/demand + ERA5 weather
  features.py       curriculum feature builder (net load, clear-sky index)
  backtest.py       leakage-safe rolling-origin cross-validation
  metrics.py        mae, pinball_loss, crps  (curriculum scoring)
  dispatch.py       the readable cvxpy battery LP (reference twin of sim/lp.py)
  viz.py            the only plotting helpers
  models/           curriculum forecasters (baselines, lear, gbt, nn, qra, conformal)
  sim/              THE TEST BENCH — read this last but spend the most time here
notebooks/01–10     the curriculum, thin; they import the library and narrate
tests/              pytest — doubles as executable documentation (Part 6)
scripts/            the command-line drivers that actually run campaigns
config.yaml         the single source of run parameters
docs/               you are here
outputs/            everything a run produces (gitignored, regenerable)
data/               cached raw + processed data (gitignored)
```

### First things to actually do

1. Skim [`../README.md`](../README.md) and [`docs/README.md`](README.md) for the
   framing — just the opening sections.
2. Read Chapter 1 of [`internals-deep-dive.md`](internals-deep-dive.md) for the
   problem and the vocabulary (NEM, arbitrage, dispatch, oracle, capture ratio).
3. Run the checks so you know the code works on your machine:
   ```bash
   ruff check .
   pytest -q
   ```
   `pytest` is slow (some tests fit real models). That's expected.

**You can now:** say what grian does in two sentences and name the two halves.

---

## Where do I find…? (a lookup table to keep handy)

| I want to understand / change… | Go to |
|---|---|
| What one run's parameters are | `config.yaml`, and `sim/trials.py` → `DEFAULT_CONFIG` |
| How data is loaded and cleaned | `src/grian/data.py`; the timestamp shift is `_shift_to_interval_start` |
| What columns a model sees | `sim/features.py` → `build_features` |
| How a model is trained/queried | `sim/models.py` → the model's `fit`/`predict` (find it via `REGISTRY`) |
| The battery optimiser (the maths) | `sim/lp.py` → `solve_lp` |
| The perfect-foresight benchmark | `sim/oracle.py` → `compute_oracle` |
| The simple trading loop | `sim/runner.py` → `simulate_region` |
| The smart (re-planning) trading loop | `sim/mpc.py` → `simulate_region_mpc` |
| How trades are recorded | `sim/ledger.py` → `append_record`, `summarise` |
| How the capture ratio is computed | `sim/analytics.py` → `capture_report` |
| Risk-aware / probabilistic dispatch | `sim/dispatch_prob.py`, and `lp.py` → `solve_mean_cvar_lp` |
| How a whole campaign is launched | `scripts/run_common_eval.py`, `scripts/testbed.py` |
| The results dashboard | `sim/dashboard.py` (built by `scripts/build_dashboard.py`) |
| How "no leakage" is proven | `tests/test_backtest.py` + `sim/ablations.py` |

---

## Part 1 — The vocabulary of the code: config, transforms, the model interface

**Goal:** learn the three conventions that every other file assumes.

**Read, in this order:**

1. `sim/trials.py`, top to `make_config`. Focus on:
   - `_get_transform_pair` — the `(forward, inverse)` price transforms
     (`asinh`/`log1p`/`identity`). Everything models a *transformed* price and
     inverts before scoring; this is where that lives.
   - `DEFAULT_CONFIG` — the full menu of knobs (data window, model, horizon,
     dispatch physics, MPC settings, ablation flags). Read every key and its
     comment; this dict *is* the interface to a run.
   - `make_config` — deep-merges your overrides onto the defaults and stamps the
     git commit hash. Note it rejects unknown keys (typo protection).
2. `sim/models.py`, **just the module docstring and the `REGISTRY` dict at the
   bottom** (near the end of the file). Don't read the model bodies yet. The point
   to absorb: *a model is a plain dictionary of four functions* —
   `fit`, `predict`, `save`, `load` (quantile models add `predict_fan`). The
   registry maps a name string to one of these dicts.

**Trace this:** find `get_model` at the bottom of `models.py`. See how a name like
`"lightgbm_rich"` becomes a dict. That dict is the whole contract between the
trading loop and any model.

**Now open** [`architecture.md`](architecture.md) and read "The model registry"
and "The config schema" — it says the same things more formally.

**You can now:** open any config and know what it controls, and explain the
model interface without looking.

---

## Part 2 — Features: what the model actually looks at

**Goal:** understand how raw price/demand/weather becomes model input, and why
every column is backward-looking.

**Read** `sim/features.py` top to bottom — it's short and self-contained. Read the
per-group builders first (`price_lags`, `rolling_stats`, `demand_features`,
`calendar_features`, `momentum_features`, `intraday_profile`, `scarcity_features`,
`weather_features`), then `build_features`, which assembles them.

**As you read, notice:**
- Every feature uses `.shift(1)` or a rolling window that excludes the present.
  That is the no-leakage rule in code form — find an example and convince yourself
  it can't see the future.
- `calendar_features` vs `fourier_calendar_features` — two encodings of "what time
  is it." Chapter 5 of `internals-deep-dive.md` explains why this choice is subtle.
- `lean_lags` and the `feature_set="lean"` branch — a deliberately *smaller*
  feature set, and the docstring explains the finding that motivated it.

**Trace this:** pick `build_features` and follow the `include_weather=True` path
into `weather_features`. Read its docstring on why the weather is leakage-free.

**Run this checkpoint** (in a REPL):
```python
import pandas as pd
from grian.sim.features import build_features
df = pd.read_parquet("data/processed/SA1_30min_sim.parquet")  # if present
X = build_features(df.head(2000), "price", "30min")
print(X.columns.tolist()); print(X.tail())
```
Look at the actual column names and the NaNs at the top (from the lags).

**You can now:** list the feature groups and prove none of them leak.

---

## Part 3 — The battery optimiser and the benchmark (the maths core)

**Goal:** understand how a price forecast becomes a charge/discharge plan, and how
the "perfect" benchmark is computed.

**Read** `sim/lp.py` in full — it is the mathematical heart and now heavily
commented. Read alongside Chapter 12 of `internals-deep-dive.md` (it derives the
same objective and constraints). Key functions:
- `solve_lp` — the arbitrage linear program. Trace the four pieces: the objective
  (`objective_coeffs`), the SOC-dynamics equality constraints
  (`eq_rows/eq_cols/eq_vals`), the throughput inequality, and the variable bounds.
  Understand the variable layout comment: `x = [charge, discharge, soc]`.
- `clamp_action` — why a *plan* must be trimmed to what the real battery can do
  before any revenue is booked (trap T2).
- `resolution_dt_hours` — tiny, but it's the antidote to the "hardcoded dt" bug
  (trap T1) that voided early results. Everything derives `dt` from config.
- `solve_mean_cvar_lp` — skip on the first pass; come back in Part 5.

**Then read** `sim/oracle.py` → `compute_oracle`. It is just `solve_lp` run against
the *actual* prices with perfect foresight, in weekly blocks for speed. Read the
`_assert_closed_cycles_nonnegative` docstring — it's a lovely sanity check.

**Run this checkpoint:**
```python
import numpy as np
from grian.sim.lp import solve_lp
prices = np.array([20, 20, 300, 300, 20, 20], dtype=float)  # cheap, dear, cheap
r = solve_lp(prices, dt_hours=0.5)
print(r["charge"], r["discharge"], r["revenue"])
```
Confirm it charges into the cheap intervals and discharges into the dear ones.

**You can now:** explain the battery LP line by line and describe the oracle.

---

## Part 4 — The open-loop trading loop (the spine)

**Goal:** trace one complete, simple trial from config to capture ratio. This is
the backbone; the MPC loop in Part 5 is a variation on it.

**Read** `sim/runner.py` in this order:
1. `run_trial` (the main entry point) — it saves the config, loops over regions,
   calls a `simulate_fn`, then saves the ledger/metrics/forecasts/model. Notice
   `simulate_fn` defaults to `simulate_region` but can be swapped for the MPC loop
   — that one parameter is the whole open-loop-vs-MPC switch.
2. `simulate_region` — the open-loop walk-forward: for each day, optionally refit,
   forecast, then dispatch.
3. `_simulate_day` and `battery_dispatch` — the per-day mechanics that call the LP
   and write to the ledger.

**Then read** `sim/ledger.py` (short) and `sim/analytics.py` →
`capture_report` and `skill_vs_naive`. `capture_report` is where the headline
number is finally computed: realised daily revenue ÷ oracle daily revenue.

**Trace the whole spine now**, function by function, with `internals-deep-dive.md`
Chapters 9–10 open:

```
run_trial                        (runner.py) — save config, loop regions
  └─ simulate_region             (runner.py) — walk day by day
       ├─ model["fit"]           (models.py) — refit every refit_days
       ├─ model["predict"]       (models.py) — forecast the horizon from data-so-far
       ├─ _get_transform_pair    (trials.py) — invert asinh → dollars
       ├─ battery_dispatch       (runner.py)
       │    ├─ solve_lp          (lp.py)     — plan the day
       │    └─ clamp_action      (lp.py)     — make each action feasible
       └─ ledger.append_record   (ledger.py) — book revenue
  └─ ledger.summarise            (ledger.py) — P&L, Sharpe, drawdown
  └─ analytics.capture_report    (analytics.py) — capture ratio vs the oracle
```

Also skim [`architecture.md`](architecture.md) "Data flow of one trial" — it's the
same picture as a diagram.

**Run this checkpoint:** run one open-loop trial end to end. The quickest path is a
tiny script or REPL session using `make_config` + `run_trial` on a slice of data;
or read `scripts/run_common_eval.py` to see how the campaign wires it, and run its
smallest configuration.

**You can now:** narrate exactly how a config becomes a capture ratio.

---

## Part 5 — The MPC loop and probabilistic dispatch (where the campaign lives)

**Goal:** understand the *closed-loop* executor and the risk-aware dispatch modes —
the parts that produced the champion result.

**Read** `sim/mpc.py` → `simulate_region_mpc`. It's the biggest single function in
the sim; take it in chunks. Read Chapter 13 of `internals-deep-dive.md` first so
you know the story (the `observe_present` bug and the `$3000` spike gate). As you
read the function, find:
- The refit / reforecast cadence logic (the `while pos < pos_end` loop head).
- `observe_gate` — the spike gate; grep for `observe_now` and read the comment
  block above it. This is the single most important design decision in the sim.
- `_telescope` — the speed hack for long horizons, and the comment on why it's
  disabled at 30-min resolution.
- The `dispatch_mode` branches near the bottom: `point`, the scenario modes, and
  `mean_cvar`.

**Then read** `sim/dispatch_prob.py` (short): `quantile_gate_prices`,
`quantile_weights`, and `combine_scenario_actions` (the `ev`/`cvar` fusion). Now
go *back* to `lp.py` → `solve_mean_cvar_lp` and read it with
[`dispatch-under-uncertainty-maths.md`](dispatch-under-uncertainty-maths.md) open —
that doc derives the Rockafellar-Uryasev CVaR linearisation the function
implements.

**Trace this:** in `simulate_region_mpc`, follow one quantile fan from
`model["predict_fan"]` → `_forecast_from_fan` → the `mean_cvar` branch →
`solve_mean_cvar_lp`. That's a forecast distribution becoming a risk-aware plan.

**You can now:** explain why aggressive MPC first *lost* money, how the spike gate
fixed it, and how the CVaR objective trades expected revenue against tail risk.

---

## Part 6 — Read the models in depth, one family at a time

Only now open the big model bodies in `sim/models.py`. Don't read it top to
bottom; read **one family, end to end (fit → predict → save → load), then run it**,
before moving to the next. Suggested order, simplest first:

1. **`naive_similar_day`** (`_naive_fit`/`_naive_predict`) — the floor. Tiny.
   Notice "predict-from-now": `predict` reads the freshest data it's handed, not the
   frozen training tail.
2. **`autoregression`** (`_ar_fit`/`_ar_predict`) — linear regression on lags,
   iterated. See recursive multi-step forecasting in code.
3. **LEAR family** (`_linear_fit`/`_linear_predict`, spec `LINEAR`) — one
   regularised linear pipeline *per horizon step*. Read `_linear_preprocessor` and
   the calendar-encoding logic; this is where the Fourier "wart" (Chapter 5) lives
   in `_l1_penalty` and `_leading_calendar_columns`.
4. **`lightgbm_rich`** (`_lgbm_rich_fit`) — gradient-boosted trees on the full
   feature set. Read `_decision_weights` for the spike-weighting variant.
5. **`lightgbm_qmean`** (`_lgbm_qmean_fit`, `_lgbm_qmean_predict`,
   `_lgbm_qmean_predict_fan`) — the quantile model. Read `_quantile_weights`
   (integrating the fan to a mean) and the conformal-calibration helpers
   (`_conformal_fan_adjustments`/`_apply_conformal`). Chapters 7–8 of
   `internals-deep-dive.md` pair with this.
6. **`lear_qmean_torch`** (`_lear_qmean_torch_fit`) — the batched-gradient quantile
   linear model, and a good war story (the MPS-diverges comment block is worth
   reading in the source).

For each family, ask the same four questions: what features does it build, what
does it fit (and with what hyperparameters), how does `predict` reconstruct a
forecast, and does it honour predict-from-now?

**You can now:** open any model and explain its exact training recipe.

---

## Part 7 — The scaffolding: search, ablations, dashboard, scripts

The supporting cast, read as needed rather than cover to cover:

- `sim/ablations.py` — the "wrong on purpose" configs (no transform, no embargo,
  future leakage). Pair with `tests/test_backtest.py`: together they *prove* each
  safeguard matters.
- `sim/search.py` — grid/random/Bayesian hyperparameter search. Skim `run_search`.
- `sim/dashboard.py` — builds the static HTML results explorer from saved trials.
  You mostly *use* this (`scripts/build_dashboard.py`) rather than read it.
- `scripts/` — the actual command-line drivers. `run_common_eval.py` (point
  models over the common window) and `testbed.py` (quantile-fan builds + replay)
  are how campaigns are launched; `build_sim_data.py` / `build_weather_data.py`
  prepare the processed parquet. Read one script's `argparse` block to see the
  real knobs.

**You can now:** launch and extend a campaign, and know where every output lands.

---

## Part 6.5 — Use the tests as documentation

The `tests/` directory is the most reliable, always-current description of what the
code *guarantees*. When a module confuses you, open its test file and read the
assertions — they're worked examples with expected answers.

Highest-value ones to read:
- `tests/test_sim_lp.py` — tiny, exact LP examples (charge-cheap/discharge-dear,
  efficiency accounting, throughput caps). The fastest way to build LP intuition.
- `tests/test_backtest.py` — the leakage test: it injects a future value and
  asserts the score gets *worse*. This is how "no leakage" is proven, not claimed.
- `tests/test_sim_mpc.py` — the observe-gate and fan-checkpoint behaviour.
- `tests/test_sim_models.py` — per-model fit/predict contracts, including
  predict-from-now.

Run a single file while reading it:
```bash
pytest tests/test_sim_lp.py -q
```

---

## A suggested schedule

If you have a week of study evenings, a comfortable pace:

| Session | Cover | Deliverable to yourself |
|---|---|---|
| 1 | Part 0 + Part 1 | Explain the two halves and the model interface. |
| 2 | Part 2 | List the feature groups; run the feature checkpoint. |
| 3 | Part 3 | Explain the LP; run the tiny arbitrage example. |
| 4 | Part 4 | Narrate the open-loop spine end to end. |
| 5 | Part 5 | Explain the spike gate and CVaR dispatch. |
| 6 | Part 6 + 6.5 | Explain two model families from the source; read their tests. |
| 7 | Part 7 | Launch one small campaign and open its dashboard. |

Two habits make all of it stick: **follow every call into its definition**, and
**run the checkpoint at the end of each part** rather than only reading. The code is
small and honest — once the spine in Part 4 is clear, everything else is a variation
on it.

---

## Where to go deeper

- Concepts, methods, and the maths behind everything here:
  [`internals-deep-dive.md`](internals-deep-dive.md).
- The formal wiring reference: [`architecture.md`](architecture.md).
- Physics, the oracle, and the traps: [`dispatch-and-scoring.md`](dispatch-and-scoring.md).
- Running and reading experiments: [`running-experiments.md`](running-experiments.md).
- Adding a model / feature / executor knob safely: [`extending.md`](extending.md).
- The lab notebook of what was tried and what failed:
  [`../outputs/experiment_log.md`](../outputs/experiment_log.md).
