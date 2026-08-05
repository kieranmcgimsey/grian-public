# CLAUDE.md — operating manual for agents

**Read this first.** grian is a walk-forward battery-trading simulation
environment for NEM electricity-price forecasting. The package is `grian`; the
core is `src/grian/` (models → dispatch → evaluation) — models forecast prices, an MPC dispatcher
trades a battery against them, and **capture ratio** (revenue ÷ a
perfect-foresight oracle's) is the headline metric.

## Repository map

| Path | What it holds |
|---|---|
| `src/grian/models/` | Forecasters + registry (baselines, linear, gradient_boosting, neural, conformal) |
| `src/grian/dispatch/` | Battery LP, oracle, executors (open-loop, MPC), probabilistic dispatch, ledger |
| `src/grian/evaluation/` | Scoring (analytics/capture), trial orchestration, sweeps, ablations |
| `src/grian/` | Shared: config, data, features, plotting, dashboard |
| `tests/` | pytest suite covering the LP/oracle physics, backtest, metrics, dispatch |
| `config.yaml` | Region, date windows, paths, seeds — single source of run parameters |
| `data/` | Raw caches (NEMOSIS, ERA5) and processed parquet. Raw gitignored |
| `outputs/` | The experiment log and the built dashboard (trial artifacts gitignored) |

## Engineering rules

- **Fat library, thin everything else.** All reusable logic lives in `src/grian/`;
  scripts and docs import, run, and explain.
- **No leakage.** Use the shared rolling-origin backtest with an embargo the
  length of the horizon. A unit test must prove that injecting a future value
  degrades the score.
- **Honest benchmarks.** Every model is scored against a perfect-foresight oracle
  and a similar-day naive baseline on one fixed held-out window. (An AEMO
  pre-dispatch comparison is intended future work — NEMSEER is not wired in.)
- **Target transform.** Model the inverse hyperbolic sine of price; always invert
  before scoring, so errors are in dollars.
- **Determinism.** Set seeds, pin versions, cache every download, never re-pull.
- **CPU / Apple Silicon only.** No CUDA-only dependencies. Use MPS where torch
  benefits; keep everything runnable on a laptop.
- **AEMO timestamps.** Interval-ending. Shifted back by one interval on load in
  `data.py`. This decision is applied once and documented once.
- **Log failures.** Every bug, dead end, or surprising result goes in
  `outputs/experiment_log.md` with a root-cause write-up.

## Conventions

- Every public module, class, and function has a Google-style docstring, enforced
  by `ruff` with pydocstyle rules.
- Figures are saved to `outputs/figures/` using the shared style from `viz.py`.

## Development

```bash
ruff check .        # lint (docstrings enforced)
pytest -q           # tests
```

## Collaboration

- The repo is private under `kieranmcgimsey`. Do not push to any org remote.
- Verify `git remote -v` before pushing.
