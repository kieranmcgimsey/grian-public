# Running experiments

A task-oriented walkthrough. Every command runs from the repo root; assumes
`pip install -e ".[dev]"` and the processed dataset
`data/processed/SA1_30min_sim.parquet` (build it with `scripts/build_sim_data.py`).
The active entry points are listed in [`scripts/README.md`](../scripts/README.md).

## The common window

Every capture-ratio result is scored on **one identical held-out window** so the
numbers are directly comparable: **2025-07-01 → 2026-06-30** (the most recent 12
months), at **30-minute** resolution, against a single cached perfect-foresight
oracle. Walk-forward: for each day, train on the trailing ~548 days (18 months),
refit every 28 days, trade the next day.

**Why one window, and why balanced capture.** Battery revenue is
spike-concentrated — the top ~10 days are ~45% of the year's oracle revenue — so a
single *pooled* ratio is dominated by a couple of mega-spike days and overfits
fast. Rank models on **balanced** capture (the mean of the 12 per-month ratios),
report pooled alongside, and confirm anything that looks like a win on
regime-contrasting **sub-windows** (a spiky month, a calm month) before believing
it. A knob whose edge doesn't survive a fresh sub-window is a fit, not a result.

## Recipe 1 — evaluate the point models

```bash
python scripts/run_common_eval.py --resolution 30min --base outputs/trials_30min \
  --executors openloop,mpc30,mpc_spike
```

Computes (and caches) the oracle for the window, walks every point model
(`naive_similar_day`, `autoregression`, the `lear*` family, `lightgbm_rich*`)
through each executor, and writes per-trial ledgers under `outputs/trials_30min/`.
`--models` restricts the set; `--test-start/--test-end` change the window;
`--refit-days`/`--lookback-days` change the cadence (defaults 28 / 548).

The executors: **openloop** (one forecast + LP solve per day), **mpc30**
(re-forecast every 30 min — the over-reactive baseline), and **mpc_spike** (the
champion executor: forecast once a day, re-solve every interval from the true
state of charge, and observe the live price only above the **$3000 spike gate**).

## Recipe 2 — build and score the quantile fans

Probabilistic models emit a *fan*; the dispatch mode is independent of the
forecast, so `testbed.py` builds the fan **once** and replays every executor over
it (seconds instead of a refit each).

```bash
# build the fans (daily cadence)
python scripts/testbed.py build \
  --models lightgbm_qmean_weather_fourier,lear_qmean_torch_fourier \
  --windows full --resolution 30min --fan-cadence 48

# replay the dispatch executors over the frozen fans
python scripts/testbed.py grid \
  --models lightgbm_qmean_weather_fourier,lear_qmean_torch_fourier \
  --windows full --resolution 30min --fan-cadence 48 \
  --executors point_ol,mpc_spikegate
```

The champion is **`lightgbm_qmean_weather_fourier` under spike-gated MPC ≈ 0.55
balanced / 0.585 pooled**. Building a fan is the expensive step (~minutes/refit on
CPU); the grid replay is cheap because it reuses the cached fan.

## Recipe 3 — the balanced-capture leaderboard

```bash
python scripts/balanced_eval.py --base outputs/trials_30min \
  --trials <trial names> --out my_eval
```

Scores the named trials by **balanced** (per-month) and **pooled** capture against
the oracle and writes a per-month heatmap. Rank on balanced; treat sub-1-point
gaps as noise.

## Recipe 4 — a custom trial from Python

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
    "embargo": 0,                       # trading sim → 0 (see architecture.md)
    "transform": "asinh",
    "dispatch": {"power_mw": 100.0, "duration_hours": 2.0,
                 "efficiency": 0.85, "max_cycles": 2},
    # Spike-gated MPC: forecast daily, re-solve every interval, gate at $3000.
    "mpc": {"resolve_every": 1, "reforecast_every": 48, "observe_gate": 3000},
})

# Open-loop: omit simulate_fn.  Spike-gated MPC: pass simulate_region_mpc.
results = run_trial({"SA1": data}, cfg, base="outputs/trials",
                    simulate_fn=simulate_region_mpc)
```

`run_trial` writes `config.json`, `ledger.parquet`, `forecasts.parquet`,
`metrics.json`, and `model/` under `outputs/trials/my_experiment/SA1/`. It does
**not** compute capture ratio — do that next.

## Recipe 5 — score a trial against the oracle

```python
import pandas as pd
from grian.dispatch import battery_lp as lp, oracle
from grian.evaluation.analytics import capture_report

data = pd.read_parquet("data/processed/SA1_30min_sim.parquet")
prices = data.loc["2025-07-01":"2026-06-30", "price"]

orc = oracle.compute_oracle(
    prices, dt_hours=lp.resolution_dt_hours("30min"),
    cache_path="outputs/trials/_oracle/SA1/common.parquet",
)

ledger = pd.read_parquet("outputs/trials/my_experiment/SA1/ledger.parquet")
rep = capture_report(ledger, orc["daily_revenue"])

print(f"capture   {rep['capture_ratio']:.3f}")
print(f"revenue   ${rep['total_revenue']:,.0f} of ${rep['oracle_revenue']:,.0f}")
print(f"spearman  {rep['mean_daily_spearman']:.3f}")   # intraday rank skill
print(rep["top10_regret"])                             # worst days
```

`capture_report` returns `capture_ratio`, `total_revenue`, `oracle_revenue`,
`regret_daily`, `top10_regret`, `oracle_top10_share`, and `mean_daily_spearman`.
It **asserts capture ≤ 1.03** — a larger value is a scoreboard bug, not
a good model (the small margin absorbs the oracle's ~0.2% block conservatism).

## Recipe 6 — the dashboard

```bash
python scripts/build_dashboard.py --base outputs/trials_30min --region SA1
```

Builds a single self-contained HTML page (Plotly inlined, no server) →
`outputs/dashboard/index.html`. Interactive comparison of all saved trials: the
capture matrix, a balanced model-selection view, equity curves, forecast-vs-actual
overlays, day-ahead fans, and a window slider that recomputes capture / regret over
any sub-window.

## Reading results correctly

- **Capture ratio is the headline.** Revenue in dollars is scale; capture is the
  honest fraction of achievable value. `~0.50` is market-average; `0.65–0.75` is
  strong.
- **Balanced over pooled.** Rank on the mean of the 12 monthly ratios; pooled is a
  high-variance "did you catch the two big days".
- **Spearman is the diagnostic, not the goal.** High rank skill with low capture
  means the forecast ranks prices well but understates spike *magnitude* — the LP
  sizes cycles by magnitude, so it under-trades (Entry 015).
- **Always look at `top10_regret`.** A capture number is an average over a window
  whose P&L lives in ~10 days.
- **Sub-1-point differences are noise** given the spike concentration; confirm any
  win on a fresh sub-window before believing it. Do not crown a champion on a
  0.3-point edge.

## Where results are recorded

- Per-trial artifacts: `outputs/trials_30min/<name>/SA1/` (gitignored — reproducible)
- Failures & surprises: `outputs/experiment_log.md` (**mandatory** — see
  [extending.md](extending.md#the-experiment-log-rule))
