# grian — repository roadmap

A top-level study map of the whole codebase: what each piece is, how the pieces
connect, and the concepts to be able to explain from memory. Work through it in
the order below; every section points at the files that back the claims so you
can read the source, not just the summary.

---

## 1. What grian is, in one breath

grian takes **raw Australian NEM electricity-market data** and builds up to
**probabilistic day-ahead price forecasts** and, on top of them, a
**forecast-driven battery dispatcher** whose success is scored against a
**perfect-foresight oracle**. The headline number is the **capture ratio**:
realised arbitrage revenue ÷ oracle revenue for a 100 MW / 200 MWh South
Australian (SA1) battery. Champion today: **0.546 validation / 0.562 test**
(`lightgbm_rich` under 30-minute-reforecast MPC).

The repo has **two halves that share the same library**:

| Half | Lives in | Purpose |
|---|---|---|
| **Curriculum** | `notebooks/01`–`10` | Teaches the pipeline: data → market → features → forecast → money. Thin notebooks that import from `src/grian/`. |
| **Simulation environment** | `src/grian/sim/` | A walk-forward battery-trading test bench: models, dispatch executors, oracle, ledger, trials, sweeps. This is where the capture-ratio campaign runs. |

Read `docs/README.md` for the simulator's one-paragraph mental model and
`README.md` for the curriculum framing before diving in.

---

## 2. Directory map (where to look)

```
src/grian/            The library — all reusable logic
  config.py           load_config(), repo_root(); config.yaml is the single source of params
  data.py             NEM price/demand + ERA5 weather loading, timestamp shift, gap checks
  features.py         net_load, clear_sky_index, build_matrix (curriculum feature set)
  backtest.py         rolling_origin() — leakage-safe rolling-origin CV with embargo
  metrics.py          mae, relative_mae, pinball_loss, crps
  dispatch.py         schedule() battery LP, capture_ratio(), rolling_mpc()  (curriculum-side)
  viz.py              apply_style(), save_fig() — the ONLY plotting helpers
  models/             Curriculum forecasters
    baselines.py      similar-day naive + AR — the floor every model must beat
    lear.py           LASSO-per-hour linear benchmark (LEAR)
    gbt.py            LightGBM quantile regression
    nn.py             day-ahead quantile network (PyTorch, sorted outputs)
    qra.py            quantile regression averaging — combine forecasts
    conformal.py      conformal wrapper for finite-sample coverage
  sim/                The trading test bench (campaign lives here) — see §5
notebooks/            The 10-notebook curriculum (thin; narrate + visualise)
tests/                pytest: core (backtest/metrics/dispatch) + sim_* suites
scripts/              run_*.py drivers for baselines, round-2 models, MPC, ablations, sweeps
docs/                 Simulator onboarding: architecture, running-experiments, dispatch-and-scoring, extending
  learning_guides/    Illustrated 16-chapter learning guide: PDFs, markdown source, figures, build.py
config.yaml           Region, date windows, battery profile, horizon, quantiles, seed
outputs/              figures, models, reports, trials, plans, experiment_log.md
data/                 Raw caches (NEMOSIS/NEMSEER/ERA5) + processed parquet (gitignored)
```

---

## 3. The core library — study these first

These modules are the shared spine. Understand them and both halves make sense.

- **`config.py`** — `load_config()` reads `config.yaml`; `repo_root()` anchors
  all paths. Everything reproducible flows from one config.
- **`data.py`** — Loads prices/demand/weather. Two decisions baked in here and
  documented once: **AEMO timestamps are interval-ending and get shifted back
  one interval on load** (`_shift_to_interval_start`); intervention duplicates
  are dropped. `build_dataset()` assembles the aligned frame; ERA5 stubs
  cleanly if no CDS key.
- **`features.py`** — `net_load = demand − solar − wind` (the real price
  driver), `clear_sky_index` (weather → generation), `build_matrix` (the
  design matrix). All backward-looking.
- **`backtest.py`** — `rolling_origin()`: the leakage-safe evaluation harness.
  Rolling-origin folds with an **embargo the length of the horizon** so a fold's
  training data can't peek past the forecast boundary.
- **`metrics.py`** — `mae`, `relative_mae` (vs naive baseline = skill),
  `pinball_loss` (per-quantile), `crps` (whole-distribution score).
- **`dispatch.py`** — `schedule()` is the battery arbitrage LP; `capture_ratio()`
  the headline; `rolling_mpc()` the receding-horizon loop (curriculum version).

---

## 4. The curriculum arc — notebooks 01–10

There are **10 notebooks** (not more). Each opens with objectives, narrates every
step with inline plots, sets analytical exercises, and writes a short report.
The arc: **data → market understanding → features → forecast → money.**

| # | Title | Core idea to be able to explain |
|---|---|---|
| 01 | Get and tame NEM data | Data landscape, interval-ending timestamps, caching, no re-pull |
| 02 | The shape of the market | Stylised facts: spikes, negatives, duck curve, seasonality |
| 03 | How price forms | Marginal pricing, net load, the price-setting unit |
| 04 | Weather and generation | Clear-sky index, turning weather into generation |
| 05 | Framing, features and baselines | Day-ahead framing + **leakage-safe backtesting** |
| 06 | Classical baselines | LEAR: LASSO per delivery hour |
| 07 | Machine learning | LightGBM quantiles + day-ahead quantile network |
| 08 | Probabilistic forecasting | QRA, conformal prediction, CRPS, calibration |
| 09 | From forecast to money | Battery LP, capture ratio, MPC — the payoff |
| 10 | Capstone | Full pipeline + structural-residual model |

Every model notebook (06–08) closes by pushing its forecast through a dispatch
harness and reporting **capture ratio** alongside CRPS and relative MAE — the
forecast is a means to dispatch value, not an end.

---

## 5. The simulation environment (`src/grian/sim/`) — the campaign engine

A walk-forward, plain-functions-and-dicts test bench. Nothing is a heavyweight
class; every run is reproducible from its `config.json`. Read
`docs/architecture.md` for the data flow. The pipeline per trial:

**model forecasts → executor dispatches under constraints → ledger records every
interval → analytics/capture ratio score it against the oracle.**

| Module | Role |
|---|---|
| `models.py` | Model registry. Each model is a dict of function keys (`fit`/`predict`/`save`/`load`). Naive, AR, LightGBM (`lgbm`) live here. |
| `features.py` | Rich feature groups: price lags, rolling stats, demand, calendar, momentum, intraday profile, **scarcity** — all backward-looking. |
| `lp.py` | Fast battery LP (HiGHS via `scipy.linprog`, sparse SOC formulation). Hot-path replacement for the cvxpy version. `clamp_action` enforces feasibility. |
| `oracle.py` | **Perfect-foresight oracle** = the frozen denominator. One LP over the whole window against *actual* prices, SOC continuous across days, per-day cycle limit. |
| `runner.py` | Walk-forward loop: for each day, optionally refit, forecast, dispatch, execute against actuals, log. `run_trial()` / `simulate_region()`. |
| `mpc.py` | Receding-horizon MPC executor: re-solve the LP every `resolve_every` intervals, re-forecast every `reforecast_every`. Closes the loop vs open-loop one-LP-per-day. |
| `ledger.py` | Append-only trade ledger (one record per interval, never mutated) + P&L, drawdown, Sharpe, revenue, `summarise()`. |
| `analytics.py` | Failure analysis: error/revenue by hour, day-of-week, price regime; `skill_vs_naive`, `headline_table`, `worst_days`, `capture_report`. |
| `trials.py` | Trial config, frozen `config.json`, git SHA capture, artifact I/O (forecasts/actuals/ledger/metrics). Reproducibility layer. |
| `search.py` | Hyperparameter search strategies: grid, random, Bayesian; `run_search`. |
| `ablations.py` | **Deliberate failures** — wrong loss, no transform, no reconditioning, future leakage, no embargo — to prove experimentally that each mistake costs score. |
| `dashboard.py` | Static HTML builder to compare trials/regions/ablations from disk. |

---

## 6. Concepts to be able to explain cold

These recur everywhere; make sure you can state the *why*, not just the *what*.

1. **No leakage / embargo.** Rolling-origin CV with an embargo the length of the
   horizon; a unit test proves injecting a future value degrades the score.
   (`backtest.py`, `tests/test_backtest.py`, `sim/ablations.py`.)
2. **asinh target transform.** Model the inverse hyperbolic sine of price (tames
   spikes and admits negatives), always invert before scoring so errors are in
   dollars. (`trials.py` `_get_transform_pair`.)
3. **Capture ratio.** Realised revenue ÷ perfect-foresight oracle revenue — the
   single honest headline. Understand why the oracle must be a *frozen*
   denominator and the traps that voided earlier results
   (`docs/dispatch-and-scoring.md`).
4. **Open-loop vs MPC.** One LP per day on a stale forecast vs a receding-horizon
   loop that re-solves from true SOC and re-forecasts. Why 30-minute reforecast
   is the sweet spot (5-min is worse, 0.534; see git log / experiment log).
5. **Probabilistic scoring.** Pinball loss per quantile, CRPS over the whole
   distribution, calibration/coverage, and how conformal prediction guarantees
   finite-sample coverage on top of any quantile model.
6. **Honest benchmarks.** Every model reports skill vs a similar-day naive
   baseline and vs AEMO pre-dispatch, on one fixed held-out window.
7. **Battery physics in the LP.** SOC dynamics, round-trip efficiency (0.85),
   power/energy limits, and the per-calendar-day cycle (throughput) constraint.
8. **Reproducibility.** Seeds, pinned versions, cached downloads (never re-pull),
   git SHA per trial, config frozen to disk. CPU / Apple Silicon only.

---

## 7. How to spend the study time (suggested order)

1. **Frame it (30 min).** `README.md` + `docs/README.md` — the two halves and the
   mental model. Know the champion number and what it measures.
2. **Core library (2 hrs).** Read `config.py` → `data.py` → `features.py` →
   `backtest.py` → `metrics.py` → `dispatch.py`. These carry every big decision.
3. **Curriculum arc (2 hrs).** Skim notebooks 01–10 for the *narrative*; read
   05 (leakage/framing), 08 (probabilistic), 09 (money) closely.
4. **Simulator (2 hrs).** `docs/architecture.md`, then `sim/runner.py` →
   `sim/lp.py` → `sim/oracle.py` → `sim/mpc.py` → `sim/analytics.py`.
5. **The campaign (1 hr).** `outputs/reports/capture_campaign_report.md`,
   `outputs/plans/capture_campaign.md`, and `outputs/experiment_log.md` — the
   record of what was tried, what failed, and why. Cross-check with `git log`.
6. **Prove it works (30 min).** Read `tests/` — especially the leakage test and
   the sim integration tests — and run `pytest -q` and `ruff check .`.

---

## 8. Commands

```bash
ruff check .        # lint (Google-style docstrings enforced)
pytest -q           # full test suite (core + sim)
```

Scripts under `scripts/` (`run_baselines.py`, `run_model_round2.py`,
`run_mpc_trials.py`, `run_mpc_ablations.py`, `run_scarcity_ab.py`) are the
campaign drivers; `build_campaign_report.py` regenerates the report.
