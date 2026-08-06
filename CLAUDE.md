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
mypy                # type-check src/grian/models
pytest -q           # tests
```

## Typed params, configs, and the CLI

- **Hyperparameters are typed, not dict keys.** Each model family has a frozen
  `pydantic` params model in `src/grian/models/params.py` (`extra="forbid"`);
  every `fit` parses `cfg["model_params"]` into it. Defaults live there, once.
  `grian.models.params_for(name)` maps a model to its class.
- **Runs are typed YAML.** `src/grian/experiment.py` `ExperimentConfig` is loaded
  from `configs/*.yaml` and reproduces a trial down to the seed; it produces the
  engine `cfg` via `make_config` (the engine core stays dict-based — typed at the
  boundary, dicts at the core).
- **Tuning** is one Optuna interface (`src/grian/tuning.py`): `tpe`/`gp`/`grid`/
  `random` over the typed fields.
- **CLI**: `grian` (fire + rich) — `models`, `describe <model>`, `configs`,
  `run <cfg.yaml>`, `tune <cfg.yaml> <space.yaml>`. Every module is also runnable
  as `python -m grian.<module>`.
- sklearn is kept only where it *is* the model (LEAR/Ridge/ElasticNet); new infra
  (tuning, configs) does not use it.

## Collaboration

- The repo is private under `kieranmcgimsey`. Do not push to any org remote.
- Verify `git remote -v` before pushing.
