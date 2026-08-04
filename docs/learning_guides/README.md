# grian learning guides

The illustrated, book-length companion to the codebase: **16 chapters** (plus
cover, table of contents, and glossary) that build the whole subject from raw NEM
data to decision-focused learning. This folder holds the source, the figures, the
per-chapter PDFs, the merged PDF, and the build script — everything needed to read
or rebuild the guide.

## Contents of this folder

| Path | What it is |
|---|---|
| `grian_learning_guide.pdf` | The full guide, all chapters merged into one PDF. **Read this.** |
| `pdf/` | Per-chapter PDFs (`00_cover` … `16_…`, `99_glossary`) if you want them individually. |
| `src/` | The markdown source for every chapter (`NN_title.md`). |
| `figures/` | Conceptual figures (PNG), embedded into the PDFs at build time. |
| `gen_figures.py` | Regenerates every figure in `figures/`. |
| `build.py` | Renders `src/*.md` + `figures/` into `pdf/*.pdf` and the merged guide. |

### Rebuilding

```bash
python docs/learning_guides/gen_figures.py   # regenerate figures (only if changed)
python docs/learning_guides/build.py         # rebuild per-chapter + merged PDFs
```

Both scripts resolve their own paths relative to this folder — run them from
anywhere.

## Chapter map and reading status

Each chapter maps to a part of the codebase. The **Code status** column is the
honest state of the *repository* against the chapter — it tells you whether the
concept you are reading about is already implemented, so you know what to expect
when you open the source.

- ✅ **Built** — a shipped notebook and/or library module implements this directly.
- 🔧 **In the sim** — no dedicated notebook, but the idea is embodied in the
  simulation environment / capture-ratio campaign.
- 📖 **Guide ahead of code** — the chapter teaches the concept, but the codebase
  does **not** implement it yet. This is the development frontier (see below).

| # | Chapter | Maps to | Code status |
|---|---|---|---|
| 01 | Data ingestion | Notebook 01, `grian/data.py` | ✅ Built |
| 02 | Price distribution | Notebook 02 | ✅ Built |
| 03 | Price formation | Notebook 03, `grian/features.py` (`net_load`) | ✅ Built |
| 04 | Weather and renewables | Notebook 04, `grian/features.py` (`clear_sky_index`) | ✅ Built |
| 05 | Framing and baselines | Notebook 05, `grian/backtest.py`, `models/baselines.py` | ✅ Built |
| 06 | Classical models | Notebook 06, `models/lear.py` | ✅ Built |
| 07 | Machine learning | Notebook 07, `models/gbt.py`, `models/nn.py` | ✅ Built |
| 08 | Probabilistic forecasting | Notebook 08, `metrics.py`, `models/qra.py`, `models/conformal.py` | ✅ Built |
| 09 | Forecast to money | Notebook 09, `grian/dispatch.py`, `sim/lp.py` | ✅ Built |
| 10 | Capstone | Notebook 10 (full pipeline, structural-residual model) | ✅ Built |
| 11 | FCAS and co-optimisation | — (frequency-control markets alongside energy) | 📖 Guide ahead of code |
| 12 | Bidding and participation | — (offer curves, rebidding, market participation) | 📖 Guide ahead of code |
| 13 | Realised revenue | `sim/ledger.py`, `sim/runner.py`, capture-ratio campaign | 🔧 In the sim |
| 14 | Failure analysis | `sim/analytics.py`, `outputs/experiment_log.md` | 🔧 In the sim |
| 15 | Live operations | `sim/mpc.py` (receding-horizon reforecast/resolve) | 🔧 In the sim |
| 16 | Decision-focused learning | — (train the forecaster on downstream dispatch value) | 📖 Guide ahead of code |
| 99 | Glossary | Cross-cutting reference | — |

## Where development currently is

The **codebase and the notebook curriculum stop at chapter 10** — data through a
forecast driving a battery LP. Everything past that is the **simulation
environment and the capture-ratio campaign**, which currently sits at **0.546
validation / 0.562 test** capture (champion `lightgbm_rich` under 30-minute-
reforecast MPC).

Mapping that onto the guide:

- **Chapters 01–10 — implemented and stable.** Read alongside the matching
  notebook; the code exists and is tested.
- **Chapters 13–15 — implemented in the simulator, not as notebooks.** Realised
  revenue (13), failure analysis (14), and live/receding-horizon operation (15)
  are exactly what the `sim/` package and the campaign do. Read these against
  `src/grian/sim/` and the campaign report, not a notebook.
- **Chapters 11, 12, 16 — the frontier: guide is ahead of the code.** FCAS
  co-optimisation, market bidding/participation, and decision-focused learning
  are written up but **not yet built** in the repository (verified: no
  corresponding modules or tests exist). When these concepts land in code, this
  table is the first thing to update — so the reading and the repository stay in
  lockstep.

**Keeping this honest:** when a new concept is added to the codebase, flip its
row from 📖 to ✅/🔧 and point the "Maps to" column at the new module. That way the
guide always tells you which chapters are backed by real, runnable code and which
are still ahead of it.
