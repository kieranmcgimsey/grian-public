# grian

**Forecasting Australian electricity prices and trading a battery against them —
from raw NEM data up to probabilistic forecasts and battery dispatch value, scored
against a perfect-foresight oracle.**

grian is a **walk-forward battery-trading simulation environment**
(`src/grian/`): models forecast day-ahead NEM prices, a receding-horizon MPC
dispatcher trades a 100 MW / 200 MWh battery against them under real physical
constraints, a ledger records every interval, and **capture ratio** — your revenue
÷ a perfect-foresight oracle's — is the headline metric.

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
> the record in the **[experiment log](outputs/experiment_log.md)** (Entries 001–038).

---

## Contents

- [Quick start](#quick-start)
- [The simulation environment](#the-simulation-environment)
- [The capture-ratio campaign](#the-capture-ratio-campaign)
- [Documentation](#documentation)
- [Running the simulation](#running-the-simulation)
- [Repository layout](#repository-layout)
- [Development](#development)

---

## Quick start

```bash
pip install -e ".[dev]"      # or:  uv pip install -e ".[dev]"
pytest -q                    # 301 tests, run cold on synthetic fixtures
```

- **Just want the results?** Open **[`outputs/dashboard/index.html`](https://kieranmcgimsey.github.io/grian-public/)**
  in any browser — no server. Every model × every dispatch system, a capture
  matrix, a balanced model-selection view, forecast overlays, and a
  probabilistic-fan explorer.
- **New to the code?** Start at **[docs/README.md](docs/README.md)** (the onboarding
  hub), then **[docs/architecture.md](docs/architecture.md)** for how the pieces fit.

**Data.** The small **processed** parquet (SA1 price/demand/weather) is committed,
so the analysis runs cold. The ~3 GB of **raw** NEMOSIS/ERA5 caches are gitignored
and re-fetchable from AEMO's public archive (`scripts/build_sim_data.py`). ERA5
weather needs a free [Copernicus CDS](https://cds.climate.copernicus.eu/) key in
`~/.cdsapirc`; it stubs cleanly if the key is missing.

---

## The simulation environment

A walk-forward, plain-functions-and-dicts test bench in `src/grian/`: models
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
| `lear_qmean_torch` | Linear quantile fan | Batched-GD pinball fit (one linear layer → all quantiles). Calendar block unpenalised. The canonical quantile-LEAR. |
| `lightgbm_rich` | Tree ensemble | LightGBM on the full feature set (rolling stats, demand, momentum). Best *point* tree. |
| `lightgbm_qmean` (+ `_weather_fourier`, `_cal`) | Quantile GBM | Quantile boosters → a full fan (probabilistic dispatch) and an integrated mean (point). **`lightgbm_qmean_weather_fourier` is the champion.** `_cal` adds split-conformal calibration. |
| `simple_mlp`, `lstm` | Neural (deprecated) | Never competitive here; retained and flagged deprecated (they carry the frozen-tail bug and produce no results). |

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
record of how it actually went. Every bug, dead end, and finding is written up in
the [experiment log](outputs/experiment_log.md) (Entries 001–038).

**The four things that mattered, in order:**

1. The LP had `dt` hardcoded to 30 min on 5-min data — every earlier revenue figure
   was physically fictional.
2. Models were forecasting from week-stale, embargoed data.
3. MPC converts a forecaster's short-lead skill into ~15–20 capture points — but
   only if the forecaster *has* short-lead skill.
4. The winning lever is the **dispatch objective**, not the forecast: a $3000 spike
   gate that reacts to observed scarcity the day-ahead forecast can't call.

Evaluation runs on a **single common held-out year at 30-minute resolution**, scored
by **balanced (per-month) capture**, with a 16-cell model × dispatch ablation.
**Champion:** `lightgbm_qmean_weather_fourier` under **spike-gated MPC**, ≈ 0.55
balanced / 0.585 pooled; a probabilistic quantile-LEAR (`lear_qmean_torch`) fills the
linear tier. The always-current state lives in the
[dashboard](https://kieranmcgimsey.github.io/grian-public/).

---

## Documentation

| Doc | Read it when you need to… |
|---|---|
| [docs/README.md](docs/README.md) | Orient — the one-paragraph mental model and the guide map. **Start here.** |
| [docs/architecture.md](docs/architecture.md) | How the pieces fit: modules, data flow, the model registry, config schema. |
| [docs/running-experiments.md](docs/running-experiments.md) | Run a trial, an MPC trial, an ablation, or a sweep — and read the results correctly. |
| [docs/dispatch-and-scoring.md](docs/dispatch-and-scoring.md) | Battery physics, the oracle, the MPC loop, how capture / regret are computed. |
| [docs/data-and-features.md](docs/data-and-features.md) | The data sources, timestamp/weather handling, and every feature group. |
| [docs/extending.md](docs/extending.md) | Add a model, feature group, or executor knob without breaking the physics. |
| [scripts/README.md](scripts/README.md) | The entry-point scripts. |
| [`outputs/dashboard/index.html`](https://kieranmcgimsey.github.io/grian-public/) | The interactive results dashboard. |
| [`outputs/experiment_log.md`](outputs/experiment_log.md) | The engineering record — every bug/failure/finding, with root-cause write-ups. |

---

## Running the simulation

Full walkthroughs are in
[docs/running-experiments.md](docs/running-experiments.md). The short version:

```bash
python scripts/build_sim_data.py     # build the SA1 price+demand parquet
python scripts/run_common_eval.py    # models × {open-loop, mpc30, spike-gated MPC}, scored vs the oracle
python scripts/build_dashboard.py    # → outputs/dashboard/index.html
```

Every configuration is scored on **one identical held-out year** by walk-forward
(train on the trailing 18 months, trade the next day) against a single cached
oracle. The dashboard slider recomputes capture/regret over any sub-window.

<details>
<summary><b>Run a custom trial (open-loop or MPC)</b></summary>

```python
import pandas as pd
from grian.evaluation.trials import make_config
from grian.dispatch.open_loop import run_trial
from grian.dispatch.mpc import simulate_region_mpc

data = pd.read_parquet("data/processed/SA1_30min_sim.parquet")

cfg = make_config({
    "trial_name": "my_experiment",
    "model": "lightgbm_qmean_weather_fourier",
    "regions": ["SA1"],
    "resolution": "30min",
    "horizon": 48,
    "test_start": "2025-07-01", "test_end": "2026-06-30",
    "refit_days": 28,
    "train_lookback_days": 548,
    "embargo": 0,                   # trading sim → 0 (see docs/architecture.md)
    "transform": "asinh",
    # Spike-gated MPC: forecast daily, re-solve every interval, gate at $3000.
    "mpc": {"resolve_every": 1, "reforecast_every": 48, "observe_gate": 3000},
})

# Open-loop: omit simulate_fn.  Spike-gated MPC: pass simulate_region_mpc.
results = run_trial({"SA1": data}, cfg, base="outputs/trials",
                    simulate_fn=simulate_region_mpc)
```

Then score against the oracle with `analytics.capture_report` — see
[docs/running-experiments.md](docs/running-experiments.md#recipe-5--score-a-trial-against-the-oracle).

</details>

<details>
<summary><b>Add a new model</b></summary>

Create a dict with four functions and register it:

```python
# In src/grian/models/ (e.g. gradient_boosting.py)

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
src/grian/
  config.py  data.py  features.py  plotting.py  dashboard.py
  models/            # forecasters + registry
  dispatch/          # battery LP, oracle, executors (open-loop, MPC), ledger
  evaluation/        # scoring, trials, sweeps, ablations
tests/               # pytest suite (301 tests)
scripts/             # Run scripts (see scripts/README.md)
config.yaml          # Region, date windows, paths, seeds
data/processed/      # Small committed working parquet; raw caches are gitignored
outputs/             # Experiment log + dashboard (trial artifacts gitignored)
```

<details>
<summary><b><code>src/grian/</code> module map</b></summary>

| Module | Responsibility |
|---|---|
| `models/` | Forecasters + registry: baselines, linear, gradient_boosting, neural, conformal; `__init__` has `REGISTRY` + `get_model` |
| `features.py` | Backward-looking feature groups |
| `dispatch/battery_lp.py` | Fast HiGHS arbitrage LP + feasibility clamp |
| `dispatch/cvxpy_reference.py` | Readable cvxpy reference LP (the fast one is tested against it) |
| `dispatch/oracle.py` | Perfect-foresight benchmark (the capture-ratio denominator) |
| `dispatch/open_loop.py` | Open-loop walk-forward loop (`simulate_region`, `run_trial`) |
| `dispatch/mpc.py` | Receding-horizon MPC executor |
| `dispatch/probabilistic.py` | Scenario / EV / CVaR dispatch over a quantile fan |
| `dispatch/ledger.py` | Append-only trade log and pure P&L functions |
| `evaluation/analytics.py` | Error breakdowns + `capture_report` |
| `evaluation/trials.py` | Config schema, artifact save/load, reproducibility |
| `evaluation/search.py` | Pluggable hyperparameter search |
| `evaluation/ablations.py` | "Wrong on purpose" trial configs |
| `dashboard.py` | Static HTML dashboard builder |

</details>

---

## Development

```bash
ruff check .          # lint (Google-style docstrings enforced)
pytest -q             # 301 tests
pre-commit install    # optional: lint on commit
```

Config lives in `config.yaml` (single source of run parameters). Default region is
**SA1** (South Australia) — high wind/solar, frequent spikes and negatives.

## License

[MIT](LICENSE) © 2026 Kieran McGimsey.
