# Architecture — how the simulator is wired

Read this before touching the code. It gives you the mental model, the data
flow, and the three extension points, so that when you open a module you know
what it is responsible for and what it must not do.

## Design philosophy

The sim is deliberately **underengineered**: plain functions, plain dicts, no
classes, no inheritance, no framework. A model is a dict of four functions. A
config is a dict. A ledger is a list of dicts. State is passed explicitly, never
hidden in objects. Readability and reproducibility beat cleverness every time. When you extend it, match that
style (see [extending.md](extending.md)).

Everything is written to disk under `outputs/trials/<trial_name>/<region>/`, and
every run is reproducible from its `config.json` alone (which carries the git
SHA that produced it).

## Module map

The library is three concerns — **`models/`** (forecast) → **`dispatch/`** (trade)
→ **`evaluation/`** (score) — plus shared `config.py`, `data.py`, `features.py`,
`plotting.py`, and `dashboard.py`.

| Module | Responsibility | Key entry points |
|---|---|---|
| `models/` | The forecaster **registry** — a dict of `fit`/`predict`/`save`/`load` per model across `baselines`/`linear`/`gradient_boosting`/`neural`, `conformal` calibration, shared helpers in `_shared` | `REGISTRY`, `get_model` |
| `features.py` | Backward-looking feature groups | `build_features` and the per-group builders |
| `dispatch/battery_lp.py` | The fast HiGHS arbitrage LP and the feasibility clamp | `solve_lp`, `clamp_action`, `resolution_dt_hours` |
| `dispatch/cvxpy_reference.py` | The readable cvxpy reference LP the fast one is tested against | `schedule`, `capture_ratio` |
| `dispatch/oracle.py` | Perfect-foresight benchmark (the capture-ratio denominator) | `compute_oracle` |
| `dispatch/open_loop.py` | The **open-loop** walk-forward loop (one forecast/solve per day) | `simulate_region`, `run_trial`, `battery_dispatch` |
| `dispatch/mpc.py` | The **receding-horizon MPC** executor (re-solve + re-forecast) | `simulate_region_mpc` |
| `dispatch/probabilistic.py` | Scenario / EV / CVaR dispatch over a quantile fan | `combine_scenario_actions`, `quantile_weights` |
| `dispatch/ledger.py` | Append-only trade log and pure P&L functions | `append_record`, `summarise`, `to_dataframe` |
| `evaluation/analytics.py` | Error breakdowns and the **capture report** | `capture_report`, `error_by_hour`, … |
| `evaluation/trials.py` | Config schema, defaults, freezing, artifact save/load (git SHA + timestamp) | `make_config`, `save_ledger`, `_get_transform_pair` |
| `evaluation/search.py` | Pluggable hyperparameter search strategies | `run_search`, `bayesian_strategy` |
| `evaluation/ablations.py` | Preconfigured "wrong on purpose" trial configs | `make_ablation_suite` |
| `dashboard.py` | Static HTML dashboard builder over the saved artifacts | `build_dashboard` (run `scripts/build_dashboard.py`) |

`config.py` and `data.py` (NEM/ERA5 loading) are the shared base; `plotting.py`
holds the plot style.

## The two executors

This is the most important structural fact. There are **two** ways to turn
forecasts into dispatch, and a trial picks one:

```
                          ┌─────────────────────────────┐
   run_trial(cfg, ...)    │  simulate_fn (a parameter)  │
        │                 └─────────────────────────────┘
        ▼                     │                     │
   for each region ───────────┤                     │
                              ▼                     ▼
                    open_loop.simulate_region   mpc.simulate_region_mpc
                    (OPEN-LOOP, default)      (RECEDING-HORIZON MPC)
                              │                     │
                    one forecast + one LP     re-solve every 30 min,
                    solve per day, execute    re-forecast every 30-60 min,
                    the whole day blind       from the TRUE state of charge
                              │                     │
                              └────────┬────────────┘
                                       ▼
                             lp.clamp_action  (shared feasibility gate)
                                       ▼
                             ledger.append_record  (revenue booked here)
```

- **Open-loop** (`open_loop.simulate_region`, the default): forecast the day at
  midnight, solve the LP once, execute all 288 intervals against actual prices.
  Simple, fast, and the right baseline. This is what `battery_dispatch` drives.
- **MPC** (`mpc.simulate_region_mpc`, pass as `simulate_fn`): walk the window in
  small blocks, re-solving the LP from the true SOC every `resolve_every`
  intervals and regenerating the forecast every `reforecast_every` intervals.
  This converts the forecaster's short-lead skill into money and is where the
  campaign's gains live. See
  [dispatch-and-scoring.md](dispatch-and-scoring.md#the-mpc-loop).

Both executors funnel through the *same* `lp.clamp_action` and the *same*
ledger, so revenue is always physically feasible and always scored identically.

`run_trial` takes `simulate_fn` as a parameter; the campaign scripts pass
`simulate_region_mpc` for MPC trials and let it default for open-loop.

## The model registry (extension point #1)

A model is a dict with a fixed interface, registered by name:

```python
NAIVE_SIMILAR_DAY = {
    "name": "naive_similar_day",
    "output": "point",
    "fit":     fn(train_df, target_col, cfg) -> state,      # returns any dict
    "predict": fn(state, input_df, horizon)  -> pd.Series,  # length `horizon`
    "save":    fn(state, path) -> None,                     # serialise to a dir
    "load":    fn(path) -> state,                           # deserialise
}
REGISTRY["naive_similar_day"] = NAIVE_SIMILAR_DAY
```

The one rule that is easy to get wrong: **`predict` must forecast from the end
of the `input_df` it is handed**, not from data frozen at `fit` time. The runner
and MPC loop both hand `predict` all data observed up to the forecast origin;
a model that ignores it and reads `state["train_tail"]` will silently forecast
from stale data (this was Entry 014 — it cost the naive model its day-of-week
alignment). See [extending.md](extending.md#adding-a-model).

Registered models today: `naive_similar_day`, `autoregression`, `lightgbm`,
`lightgbm_rich`, `lightgbm_qmean`, `simple_mlp`, `lstm`.

## The config schema (`trials.DEFAULT_CONFIG`)

`make_config(overrides)` deep-merges your overrides onto `DEFAULT_CONFIG` and
stamps the git SHA + timestamp. It **rejects unknown keys** (typo protection),
so if you add a config knob you must add it to `DEFAULT_CONFIG` first. Nested
dicts (`dispatch`, `mpc`, `ablations`, `model_params`) merge one level deep.

The blocks that matter:

```python
{
  "model": "lightgbm_rich",          # a REGISTRY key
  "resolution": "5min",              # drives dt everywhere — never hardcode dt
  "horizon": 288,                    # forecast/plan length in intervals
  "transform": "asinh",              # target transform, inverted before scoring
  "refit_days": 7,                   # how often to refit the model
  "embargo": 0,                      # 0 for the trading sim (see below)
  "test_start"/"test_end": ...,      # the window to simulate
  "dispatch": {"power_mw": 100, "duration_hours": 2,
               "efficiency": 0.85, "max_cycles": 2},
  "mpc": {"resolve_every": 6, "reforecast_every": 6,
          "persistence_tau": 0, "persistence_gate": 0},  # MPC executor only
  "ablations": {"use_transform": True, "use_embargo": False,
                "leak_future": False, ...},              # "wrong on purpose"
}
```

**Embargo is 0 for the trading sim.** Embargo is hygiene for model-*selection*
backtests (it stops CV folds leaking near-boundary info). In a sequential
trading simulation there is no such leak — at midnight the operator legitimately
knows everything up to midnight — so blinding the model to yesterday just throws
away information. This is trap T5; do not "fix" the sim embargo to match the
backtest embargo. See [dispatch-and-scoring.md](dispatch-and-scoring.md).

## Data flow of one trial

```
config.json ──► run_trial ──► simulate_fn (open-loop or MPC)
                                   │
   data (5-min price + demand) ────┤   per forecast origin:
                                   │     1. refit model (every refit_days)
                                   │     2. predict horizon from data-so-far
                                   │     3. invert transform → dollar forecast
                                   │     4. solve LP from true SOC
                                   │     5. clamp actions → feasible
                                   │     6. append to ledger
                                   ▼
                            ledger.parquet ──► analytics.capture_report(ledger, oracle_daily)
                            forecasts.parquet         │
                            metrics.json              ▼
                            model/                capture ratio, regret, Spearman
```

The oracle (`oracle.compute_oracle`) is computed once per window and cached to
`outputs/trials/_oracle/<region>/<window>.parquet`; it is the fixed denominator
for every capture ratio.

## Where state lives (and where it must not)

- **Model state** — returned by `fit`, passed to `predict`. Explicit.
- **Dispatch state** — SOC and daily throughput. In open-loop it lives in a
  `dispatch_state` dict the runner threads across days; in MPC it lives in
  locals of `simulate_region_mpc`. Either way it is carried *continuously across
  the whole window* — the battery does not teleport to empty at midnight.
- **Nothing global.** No module-level mutable state, no singletons. Two trials
  can run in the same process without interfering (the registry is read-only
  after import).

## Next

- To *run* something: [running-experiments.md](running-experiments.md).
- To understand the *physics and the metric*:
  [dispatch-and-scoring.md](dispatch-and-scoring.md).
- To *extend* it: [extending.md](extending.md).
