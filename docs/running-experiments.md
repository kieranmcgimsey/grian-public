# Running experiments

A task-oriented walkthrough. Every command here is runnable from the repo root.
Assumes you have installed the package (`pip install -e ".[dev]"`) and that the
processed dataset `data/processed/SA1_5min_sim.parquet` exists (the campaign
scripts read it directly; `scripts/run_baselines.py` builds it on first run).

## The two windows

Every capture-ratio result is reported on one of two fixed windows. **This
separation is load-bearing — respect it or your numbers are not trustworthy.**

| Window | Dates | Purpose |
|---|---|---|
| **validation** | 2023-07-01 → 2023-09-30 (92 days) | Tune everything here. Model selection, hyperparameters, executor knobs. Run as often as you like. |
| **test** (held out) | 2023-10-01 → 2024-01-30 (122 days) | Touch **once per accepted technique**, for the headline number. |

Why so strict: battery revenue is spike-concentrated — on validation the top 10
of 92 days carry 52% of the oracle's revenue. A metric resting on ~10 days
overfits to those specific days fast, so repeated peeking at test silently
inflates your result. The campaign scripts encode both windows; you select with
a `val`/`test` argument.

## Recipe 1 — reproduce the honest baselines

```bash
python scripts/run_capture_baselines.py
```

This computes (and caches) the perfect-foresight oracle for both windows, runs
the open-loop baselines (`naive`, `autoregression`, `lightgbm_rich`,
`lightgbm_qmean`), scores each against the oracle, and appends rows to
[`outputs/plans/capture_campaign_results.md`](../outputs/plans/capture_campaign_results.md).
Expect ~20–40 min (the first oracle solve dominates; it is cached afterward).

What you should see (open-loop): AR ~0.47 val / ~0.40 test is the best
open-loop; `lightgbm_rich` is *worse* (0.39/0.30) despite the best rank skill —
that is the median-forecast problem (Entry 015), and the reason MPC exists.

## Recipe 2 — run the champion (MPC)

```bash
python scripts/run_mpc_trials.py val            # validation
python scripts/run_mpc_trials.py test           # held-out (do once)
python scripts/run_mpc_trials.py val --models lightgbm_qmean   # a subset
```

This drives the same models through `mpc.simulate_region_mpc`. The champion is
`lightgbm_rich` under MPC: **0.546 val / 0.562 test**. Compare against the
open-loop numbers from Recipe 1 to see MPC's ~15–20-point lift — and note it
makes the *naive* model worse, because MPC only pays when the forecaster has
real short-lead skill (Entry 016).

Runtime note: a 122-day MPC run is a few minutes thanks to the telescoped LP
horizon and HiGHS. Re-forecasting is the bottleneck, not the LP.

## Recipe 3 — the executor ablation grid

```bash
python scripts/run_mpc_ablations.py val    # frequency + persistence grid
python scripts/run_mpc_ablations.py test   # the accepted variant, once
```

This isolates the executor knobs (`resolve_every`, `reforecast_every`,
`persistence_tau`, `persistence_gate`). Findings you should reproduce: 30-min
re-forecast beats 60-min (decisively on test); re-solve frequency alone does
nothing; persistence blending hurts unless gated to extremes, and even then only
breaks even. See Entries 018–019 for the mechanisms.

## Recipe 4 — a custom trial from Python

```python
import pandas as pd
from grian.sim.trials import make_config
from grian.sim.runner import run_trial
from grian.sim.mpc import simulate_region_mpc

data = pd.read_parquet("data/processed/SA1_5min_sim.parquet")

cfg = make_config({
    "trial_name": "my_experiment",
    "model": "lightgbm_rich",
    "regions": ["SA1"],
    "resolution": "5min",
    "horizon": 288,
    "test_start": "2023-07-01", "test_end": "2023-09-30",
    "refit_days": 7,
    "embargo": 0,                       # trading sim → 0 (see architecture.md)
    "transform": "asinh",
    "model_params": {"n_estimators": 300, "learning_rate": 0.05},
    "dispatch": {"power_mw": 100.0, "duration_hours": 2.0,
                 "efficiency": 0.85, "max_cycles": 2},
    "mpc": {"resolve_every": 6, "reforecast_every": 6},
})

# Open-loop: omit simulate_fn.  MPC: pass simulate_region_mpc.
results = run_trial({"SA1": data}, cfg,
                    base="outputs/trials",
                    simulate_fn=simulate_region_mpc)
```

`run_trial` writes `config.json`, `ledger.parquet`, `forecasts.parquet`,
`metrics.json`, and `model/` under `outputs/trials/my_experiment/SA1/`. It does
**not** compute capture ratio — do that next.

## Recipe 5 — score a trial against the oracle

```python
import pandas as pd
from grian.sim import lp, oracle
from grian.sim.analytics import capture_report

data = pd.read_parquet("data/processed/SA1_5min_sim.parquet")
prices = data.loc["2023-07-01":"2023-09-30", "price"]

orc = oracle.compute_oracle(
    prices, dt_hours=lp.resolution_dt_hours("5min"),
    cache_path="outputs/trials/_oracle/SA1/val.parquet",
)

ledger = pd.read_parquet("outputs/trials/my_experiment/SA1/ledger.parquet")
rep = capture_report(ledger, orc["daily_revenue"])

print(f"capture   {rep['capture_ratio']:.3f}")
print(f"revenue   ${rep['total_revenue']:,.0f} of ${rep['oracle_revenue']:,.0f}")
print(f"spearman  {rep['mean_daily_spearman']:.3f}")   # intraday rank skill
print(rep["top10_regret"])                             # worst days
```

`capture_report` returns: `capture_ratio`, `total_revenue`, `oracle_revenue`,
`regret_daily` (Series), `top10_regret` (DataFrame), `oracle_top10_share`, and
`mean_daily_spearman`. It **asserts capture ≤ 1** — a violation means a
scoreboard bug (trap T7), not a good model.

## Recipe 6 — a hyperparameter sweep

```python
from grian.sim.search import run_search, bayesian_strategy

space = {
    "model_params.n_estimators": [100, 200, 500],
    "model_params.learning_rate": (0.01, 0.2, "log"),
    "model_params.num_leaves": [15, 31, 63],
}
best = run_search(base_config=cfg, space=space,
                  data_by_region={"SA1": data},
                  strategy_fn=bayesian_strategy,
                  metric_key="total_revenue", base="outputs/trials")
```

Sweep on **validation** only. Note `metric_key` operates on `metrics.json`
fields (revenue, MAE, Sharpe); to select on *capture ratio* you currently score
with `capture_report` after the run (a `capture`-aware search hook is a
reasonable extension — see [extending.md](extending.md)).

## Recipe 7 — regenerate the report figures

```bash
python scripts/build_campaign_report.py
```

Rebuilds all 8 figures under `outputs/figures/campaign/` from the current trial
ledgers. Run this after any trial that should change the headline, then review
[`outputs/reports/capture_campaign_report.md`](../outputs/reports/capture_campaign_report.md).

## Recipe 8 — the dashboard

```bash
python scripts/build_dashboard.py            # → outputs/dashboard/index.html
```

Builds a single self-contained HTML page (Plotly inlined, no server) — open it
directly in a browser and rebuild when new trials land. Interactive comparison
of all saved trials: sortable summary table, equity curves, forecast-vs-actual
overlays (recent 5-min window with a rangeslider), day-ahead forecast fans,
daily revenue, ablation comparison, and deep analytics. Only the most recent
window is kept at 5-min resolution; everything else is daily or pre-aggregated,
so the page stays small and fast.

## Reading results correctly

- **Capture ratio is the headline.** Revenue in dollars is scale; capture is
  the honest fraction of achievable value. `~0.50` is market-average;
  `0.65–0.75` is strong.
- **Spearman is the diagnostic, not the goal.** High rank skill with low capture
  means the forecast ranks prices well but understates spike *magnitude* — the
  LP sizes cycles by magnitude, so it under-trades (Entry 015).
- **Always look at `top10_regret`.** A capture number is an average over a
  window whose P&L lives in ~10 days. Two models with the same capture can fail
  on completely different days.
- **Sub-1-point differences on test are noise** given the spike concentration.
  Do not crown a champion on a 0.3-point test edge.

## Where results are recorded

- Headline table (append-only, git SHAs): `outputs/plans/capture_campaign_results.md`
- Per-trial artifacts: `outputs/trials/<name>/SA1/` (gitignored — reproducible)
- Narrative + figures: `outputs/reports/capture_campaign_report.md`
- Failures & surprises: `outputs/experiment_log.md` (**mandatory** — see
  [extending.md](extending.md#the-experiment-log-rule))
