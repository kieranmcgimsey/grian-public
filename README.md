# grian

**Forecasting Australian electricity prices and trading a battery against them —
built from raw NEM data up to probabilistic forecasts and battery dispatch value,
scored against a perfect-foresight oracle.**

grian is two halves that share one library:

- **A curriculum** — ten notebooks (`notebooks/01`–`10`) from data ingestion to a
  calibrated forecast driving a battery arbitrage optimiser.
- **A simulation environment** (`src/grian/sim/`) — a walk-forward battery-trading
  test bench with a receding-horizon MPC dispatcher, a perfect-foresight oracle,
  and **capture ratio** (your revenue ÷ the oracle's) as the headline metric.

> ### 🏆 Headline result
> On a held-out year, the champion — `lightgbm_qmean_weather_fourier` (a
> probabilistic quantile GBM) under **spike-gated MPC** — captures **≈ 0.55** of a
> perfect-foresight oracle's revenue for a 100 MW / 200 MWh South Australian
> battery.
>
> The finding that shaped the project: **the most *accurate* forecaster is the
> *worst* trader.** Capture lives in the rare price spikes, not in average error —
> so quantile GBM > point tree > LEAR, even though LEAR has the best MAE.
>
> Explore it in the **[interactive dashboard](https://kieranmcgimsey.github.io/grian-public/)**; read
> the story in the **[experiment log](outputs/experiment_log.md)** (Entries 001–038).

---

## Contents

- [Quick start](#quick-start)
- [The curriculum (notebooks)](#the-curriculum-notebooks)
- [The simulation environment](#the-simulation-environment)
- [The capture-ratio campaign](#the-capture-ratio-campaign)
- [Documentation & reports](#documentation--reports)
- [Running the simulation](#running-the-simulation)
- [Repository layout](#repository-layout)
- [Development](#development)

---

## Quick start

```bash
pip install -e ".[dev]"      # or:  uv pip install -e ".[dev]"
pytest -q                    # 308 tests, run cold on synthetic fixtures
```

- **Just want the results?** Open **[`outputs/dashboard/index.html`](https://kieranmcgimsey.github.io/grian-public/)**
  in any browser — no server. Every model × every dispatch system, a capture
  matrix, a balanced model-selection view, forecast overlays, and a
  probabilistic-fan explorer.
- **New to the code?** Start at **[docs/README.md](docs/README.md)** (the onboarding
  hub), or **[docs/codebase-tour.md](docs/codebase-tour.md)** for a staged,
  read-it-in-this-order tour.

**Data.** The small **processed** parquet (SA1 price/demand/weather) is committed,
so the analysis runs cold. The ~3 GB of **raw** NEMOSIS/ERA5 caches are gitignored
and re-fetchable from AEMO's public archive (`scripts/build_sim_data.py`). ERA5
weather (notebook 4+) needs a free [Copernicus CDS](https://cds.climate.copernicus.eu/)
key in `~/.cdsapirc`; it stubs cleanly if the key is missing.

---

## The curriculum (notebooks)

Ten thin notebooks — they import the library, narrate, and visualise. Run in order;
each builds on the last.

| # | Title | What you learn |
|---|-------|----------------|
| 01 | Get and tame NEM data | Data landscape, timestamp conventions, caching |
| 02 | The shape of the market | Stylised facts of electricity prices |
| 03 | How price forms | Marginal pricing, net load, the price setter |
| 04 | Weather and generation | Clear-sky index, generation forecasting |
| 05 | Framing, features and baselines | Day-ahead framing, leakage-safe backtesting |
| 06 | Classical baselines | LEAR: LASSO per-hour models |
| 07 | Machine learning | LightGBM quantiles, day-ahead quantile network |
| 08 | Probabilistic forecasting | QRA, conformal prediction, CRPS |
| 09 | From forecast to money | Battery LP, capture ratio, MPC |
| 10 | Capstone | Full pipeline, structural-residual model |

---

## The simulation environment

A walk-forward, plain-functions-and-dicts test bench in `src/grian/sim/`: models
forecast, an executor dispatches the battery under real physics, a ledger records
every interval, and capture ratio scores it against the oracle. Everything is saved
to disk and reproducible from a frozen `config.json`.

<details>
<summary><b>What it does, in detail</b></summary>

- Loads real NEM dispatch prices + demand (NEMOSIS) and ERA5 weather.
- Walks a test window day-by-day, refitting models on a rolling window.
- Forecasts day-ahead prices (288 five-minute or 48 half-hour intervals).
- Dispatches a 100 MW / 200 MWh battery two ways: **open-loop** (one forecast + LP
  solve per day) or **receding-horizon MPC** (re-solve from the true state of
  charge, re-forecast from fresh data on a cadence).
- Scores against a **perfect-foresight oracle** — capture ratio = realised revenue
  ÷ oracle revenue — with regret decomposition and intraday rank skill.
- Logs every interval to an append-only ledger; all actions pass a feasibility
  clamp, so booked revenue is always physically real.
- Runs deliberate-mistake **ablations** (wrong loss, no transform, future leakage,
  no embargo) to prove each safeguard matters.
- Provides hyperparameter search (grid / random / Bayesian) and builds a
  self-contained static HTML dashboard.

</details>

<details>
<summary><b>The models</b></summary>

| Model | Type | Strategy |
|-------|------|----------|
| `naive_similar_day` | Baseline | Repeats same weekday from last week — the floor. |
| `autoregression` | Linear | LinearRegression on lagged prices + calendar, iterated for multi-step. |
| `lear` (+ `_weather`, `_fourier`, …) | Linear (Lasso) | L1-penalised per-step regression on the rich feature set — the classic EPF benchmark. Best *accuracy*, poor *capture*. Also `ridge`, `elasticnet`. |
| `lear_qmean_torch` | Linear quantile fan | Batched-GD pinball fit (one linear layer → all quantiles). Calendar block unpenalised. The canonical quantile-LEAR (supersedes the ~40× slower sklearn `lear_qmean`). |
| `lightgbm_rich` | Tree ensemble | LightGBM on the full feature set (rolling stats, demand, momentum). Best *point* tree. |
| `lightgbm_qmean` (+ `_weather_fourier`, `_cal`) | Quantile GBM | Quantile boosters → a full fan (probabilistic dispatch) and an integrated mean (point). **`lightgbm_qmean_weather_fourier` is the champion.** `_cal` adds split-conformal calibration. |
| `simple_mlp`, `lstm` | Neural (deprecated) | Never competitive here; kept as code, no results. |

Models are plain dicts with `fit`, `predict`, `save`, `load` keys — no classes, no
inheritance. Adding one is a dict and four functions.

</details>

<details>
<summary><b>The ablations (deliberate mistakes)</b></summary>

| Ablation | What's wrong | Expected effect |
|----------|-------------|-----------------|
| `wrong_loss` | MSE on raw prices | Spike-dominated gradients |
| `no_transform` | No asinh compression | Heavy-tailed target, unstable training |
| `no_reconditioning` | Train once, never refit | Distribution drift over time |
| `future_leakage` | Future data in training | Suspiciously good results |
| `no_embargo` | Zero gap between train/test | Subtle autocorrelation leak |

</details>

---

## The capture-ratio campaign

The effort to dispatch the battery above 50% of oracle revenue — and the honest
record of how it actually went. Full narrative:
[findings report](outputs/reports/capture_campaign_report.md); every bug and dead
end: [experiment log](outputs/experiment_log.md) (Entries 001–038).

**The four things that mattered, in order:**

1. The LP had `dt` hardcoded to 30 min on 5-min data — every earlier revenue figure
   was physically fictional.
2. Models were forecasting from week-stale, embargoed data.
3. MPC converts a forecaster's short-lead skill into ~15–20 capture points — but
   only if the forecaster *has* short-lead skill.
4. Forecast freshness has an optimum: a 30-min re-forecast beats both 60-min (stale)
   and 5-min (jittery — the executor dithers).

<details>
<summary><b>Numbers: the original val/test phase → the current common-window champion</b></summary>

The original val/test phase (5-min data), Entries 013–020:

| Model | Executor | Validation | Test (held out) |
|---|---|---|---|
| autoregression | open-loop | 0.473 | 0.402 |
| lightgbm_rich | open-loop | 0.389 | 0.301 |
| lightgbm_qmean | open-loop | 0.407 | 0.443 |
| **lightgbm_rich** | **MPC (30-min reforecast)** | **0.546** | **0.562** |

Evaluation then moved to a **single common held-out year at 30-min resolution**,
scored by **balanced (per-month) capture**, with a 16-cell model × dispatch
ablation. **Latest champion:** `lightgbm_qmean_weather_fourier` under **spike-gated
MPC**, ≈ 0.55 balanced; a probabilistic quantile-LEAR (`lear_qmean_torch`) fills the
linear tier. The always-current state lives in the
[dashboard](https://kieranmcgimsey.github.io/grian-public/) and Entries 030–038.

</details>

---

## Documentation & reports

<details>
<summary><b>Documentation index</b> — onboarding guides, the deep dive, the learning guide</summary>

| Doc | Read it when you need to… |
|---|---|
| [docs/README.md](docs/README.md) | Orient — the one-paragraph mental model and the guide map. **Start here.** |
| [docs/codebase-tour.md](docs/codebase-tour.md) | A staged, read-in-this-order tour of the source itself. |
| [docs/internals-deep-dive.md](docs/internals-deep-dive.md) | The whole repo in one pass — raw maths behind every method and every training recipe. |
| [docs/architecture.md](docs/architecture.md) | How the pieces fit: modules, data flow, the model registry, config schema. |
| [docs/running-experiments.md](docs/running-experiments.md) | Run a trial, an MPC trial, an ablation, or a sweep — and read the results correctly. |
| [docs/dispatch-and-scoring.md](docs/dispatch-and-scoring.md) | Battery physics, the oracle, the MPC loop, how capture / regret are computed. |
| [docs/dispatch-under-uncertainty-maths.md](docs/dispatch-under-uncertainty-maths.md) | The scenario / robust / CVaR / mean-CVaR dispatch maths. |
| [docs/extending.md](docs/extending.md) | Add a model, feature group, or executor knob without breaking the physics. |
| [docs/techniques-roadmap.md](docs/techniques-roadmap.md) | A roadmap of every technique used, with file pointers. |
| [docs/learning_guides/](docs/learning_guides/) | The full illustrated **16-chapter learning guide** (PDF). |
| [REPO_ROADMAP.md](REPO_ROADMAP.md) · [scripts/README.md](scripts/README.md) | Top-level study map; the entry-point scripts. |

</details>

<details>
<summary><b>Reports & results</b> — the dashboard, the campaign report, the MPC investigation</summary>

| Artifact | What it is |
|---|---|
| [`outputs/dashboard/index.html`](https://kieranmcgimsey.github.io/grian-public/) | **The interactive results dashboard** — models × dispatch, curated view, forecast & fan explorers. |
| [`outputs/experiment_log.md`](outputs/experiment_log.md) | The lab notebook — Entries 001–038, every bug/failure/finding with root-cause write-ups. |
| [`outputs/reports/capture_campaign_report.md`](outputs/reports/capture_campaign_report.md) | The capture-ratio campaign findings report. |
| [`outputs/reports/mpc_investigation_report.pdf`](outputs/reports/mpc_investigation_report.pdf) | 15-page MPC investigation (the `observe_present` bug → spike-gated fix). |
| [`outputs/figures/`](outputs/figures/) | Ablation heatmaps and per-month capture figures. |

</details>

---

## Running the simulation

Full walkthroughs are in
[docs/running-experiments.md](docs/running-experiments.md). The short version:

```bash
python scripts/build_sim_data.py     # extend SA1 price+demand to ~present
python scripts/run_common_eval.py    # models × {open-loop, 30-min MPC}, scored vs the oracle
python scripts/build_dashboard.py    # → outputs/dashboard/index.html
```

Every configuration is scored on **one identical test window** by walk-forward
(train on all history to date, trade the next day) against a single cached oracle.
The dashboard slider recomputes capture/regret over any sub-window.

<details>
<summary><b>Run a custom trial (open-loop or MPC)</b></summary>

```python
from grian.sim.trials import make_config
from grian.sim.runner import run_trial

cfg = make_config({
    "trial_name": "my_experiment",
    "model": "lightgbm_rich",
    "regions": ["SA1"],
    "resolution": "5min",
    "horizon": 288,
    "test_start": "2023-07-01",     # validation window — tune here
    "test_end": "2023-09-30",
    "refit_days": 7,
    "embargo": 0,                   # trading sim → 0 (see docs/architecture.md)
    "transform": "asinh",
    "model_params": {"n_estimators": 300, "learning_rate": 0.05},
    "mpc": {"resolve_every": 6, "reforecast_every": 6},
})

# Open-loop (default executor):
results = run_trial({"SA1": data}, cfg, base="outputs/trials")

# ...or receding-horizon MPC (the champion executor):
from grian.sim.mpc import simulate_region_mpc
results = run_trial({"SA1": data}, cfg, base="outputs/trials",
                    simulate_fn=simulate_region_mpc)
```

Then score against the oracle with `analytics.capture_report` — see
[docs/running-experiments.md](docs/running-experiments.md#recipe-5--score-a-trial-against-the-oracle).

</details>

<details>
<summary><b>Run a hyperparameter search</b></summary>

```python
from grian.sim.search import run_search, bayesian_strategy

space = {
    "model_params.n_estimators": [100, 200, 500],
    "model_params.learning_rate": (0.01, 0.2, "log"),
    "model_params.num_leaves": [15, 31, 63],
}

best = run_search(
    base_config=cfg,
    space=space,
    data_by_region=data_by_region,
    strategy_fn=bayesian_strategy,
    metric_key="total_revenue",
    base="outputs/trials",
)
```

</details>

<details>
<summary><b>Add a new model</b></summary>

Create a dict with four functions and register it:

```python
# In src/grian/sim/models.py

def _my_fit(train_df, target_col, cfg):
    """Train and return a state dict."""
    ...

def _my_predict(state, input_df, horizon):
    """Return a pd.Series of length `horizon`."""
    ...

def _my_save(state, path): ...
def _my_load(path): ...

MY_MODEL = {
    "name": "my_model", "output": "point",
    "fit": _my_fit, "predict": _my_predict,
    "save": _my_save, "load": _my_load,
}
REGISTRY["my_model"] = MY_MODEL
```

Then use `"model": "my_model"` in any trial config.

</details>

---

## Repository layout

```
docs/                # Onboarding guides for the simulator (start here)
docs/learning_guides/# Illustrated 16-chapter learning guide (PDFs, source, figures)
src/grian/           # Core library: data, features, backtest, models, dispatch, viz
src/grian/sim/       # Simulation environment (module map below)
notebooks/           # The 10-notebook curriculum
tests/               # pytest suite (308 tests)
scripts/             # Run scripts (see scripts/README.md)
config.yaml          # Region, date windows, paths, seeds
data/processed/      # Small committed working parquet; raw caches are gitignored
outputs/             # Figures, reports, experiment log, trial artifacts (trials gitignored)
```

<details>
<summary><b><code>src/grian/sim/</code> module map</b></summary>

| Module | Responsibility |
|---|---|
| `trials.py` | Config schema, artifact save/load, reproducibility |
| `models.py` | The model registry (`fit`/`predict`/`save`/`load` dicts) |
| `features.py` | Backward-looking feature groups |
| `runner.py` | Open-loop walk-forward loop (`simulate_region`, `run_trial`) |
| `lp.py` | Fast HiGHS arbitrage LP + feasibility clamp |
| `oracle.py` | Perfect-foresight benchmark (the capture-ratio denominator) |
| `mpc.py` | Receding-horizon MPC executor |
| `ledger.py` | Append-only trade log and pure P&L functions |
| `analytics.py` | Error breakdowns + `capture_report` |
| `ablations.py` | "Wrong on purpose" trial configs |
| `search.py` | Pluggable hyperparameter search |
| `dashboard.py` | Static HTML dashboard builder |

</details>

---

## Development

```bash
ruff check .          # lint (Google-style docstrings enforced)
pytest -q             # 308 tests
pre-commit install    # optional: lint on commit
```

Config lives in `config.yaml` (single source of run parameters). Default region is
**SA1** (South Australia) — high wind/solar, frequent spikes and negatives.

## License

[MIT](LICENSE) © 2026 Kieran McGimsey.
