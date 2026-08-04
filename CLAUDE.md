# CLAUDE.md — operating manual for agents

**Read this first.** This repository is a teaching curriculum for NEM
electricity price forecasting. The package is called `grian`. The ten notebooks
build from raw data to probabilistic price forecasts and battery dispatch value.

## Repository map

| Path | What it holds |
|---|---|
| `src/grian/` | The library: config, data, features, backtest, metrics, dispatch, viz, models/ |
| `notebooks/` | Ten notebooks (01–10), run in order. Each imports from `src/grian/`. |
| `tests/` | pytest suite covering backtest, metrics, dispatch. |
| `config.yaml` | Region, date windows, paths, seeds — single source of run parameters. |
| `data/` | Raw caches (NEMOSIS, NEMSEER, ERA5) and processed parquet. Gitignored. |
| `outputs/` | Figures, saved models, reports. Partially gitignored. |

## Engineering rules

- **Thin notebooks.** All reusable logic lives in `src/grian/`; notebooks import,
  explain, and visualise.
- **No leakage.** Use the shared rolling-origin backtest with an embargo the
  length of the horizon. A unit test must prove that injecting a future value
  degrades the score.
- **Honest benchmarks.** Every model reports skill against a similar-day naive
  baseline and against AEMO pre-dispatch, on one fixed held-out period.
- **Target transform.** Model the inverse hyperbolic sine of price; always
  invert before scoring, so errors are in dollars.
- **Determinism.** Set seeds, pin versions, cache every download, never re-pull.
- **CPU / Apple Silicon only.** No CUDA-only dependencies. Use MPS where torch
  benefits; keep everything runnable on a laptop.
- **AEMO timestamps.** Interval-ending. Shifted back by one interval on load in
  `data.py`. This decision is applied once and documented once.

## Conventions

- Every public module, class, and function has a Google-style docstring.
  Enforced by `ruff` with pydocstyle rules.
- Every notebook opens with objectives and prerequisites, narrates each step,
  ends with a summary, and writes a short report to `outputs/reports/`.
- Every notebook sets 1–3 exercises with worked solutions in a later cell.
- Figures are saved to `outputs/figures/` using shared style from `viz.py`.

## Development

```bash
ruff check .        # lint (docstrings enforced)
pytest -q           # tests
```

## Collaboration

- The repo is private under `kieranmcgimsey`. Do not push to any org remote.
- Verify `git remote -v` before pushing.
