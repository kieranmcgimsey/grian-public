# grian docs — start here

This directory is the practical onboarding path for the **grian simulation
environment**: the walk-forward battery-trading test bench in `src/grian/sim/`.
If you have been handed a ticket against this repo, read the guide that matches
your task and follow it — each is a walkthrough with runnable commands, not a
reference dump.

The notebook curriculum (`notebooks/01`–`10`) is documented separately in the
top-level [README](../README.md), and the full illustrated learning guide (16
chapters) lives under [`learning_guides/`](learning_guides/) — see its
[index](learning_guides/README.md) for the chapter map and reading status. These
docs are about the *simulator* and the *capture-ratio campaign* built on top of
it.

## The one-paragraph mental model

The simulator dispatches a 100 MW / 200 MWh battery in South Australia (SA1)
and measures how much of a **perfect-foresight oracle's** arbitrage revenue it
captures. A *model* forecasts future prices; a *dispatch executor* turns those
forecasts into charge/discharge decisions under real physical constraints; a
*ledger* records every 5-minute interval; and *capture ratio* = realised
revenue ÷ oracle revenue is the single headline metric. The whole thing is
plain functions and dicts, everything is written to disk, and every run is
reproducible from its `config.json`.

## Every file in this directory

Each is a self-contained walkthrough. Start with `architecture.md`; reach for
`internals-deep-dive.md` when you want the whole repo explained in one dense pass.

**Onboarding & orientation**

| File | Read it when you need to… |
|---|---|
| [architecture.md](architecture.md) | Understand how the pieces fit: modules, data flow, the model registry, the config schema, where state lives. **Read this first.** |
| [codebase-tour.md](codebase-tour.md) | A **progressive, staged reading plan** for learning the source itself — where to find things, what to read in what order, which call to trace next, and a small thing to run at the end of each stage. |
| [running-experiments.md](running-experiments.md) | Run a trial, an MPC trial, an ablation, or a sweep — and read the results correctly. |
| [extending.md](extending.md) | Add a model, a feature group, or an executor knob without breaking the physics or the reproducibility guarantees. |
| [data-and-features.md](data-and-features.md) | Understand the data sources, the timestamp/weather handling, and every feature group `sim/features.py` builds. |

**Dispatch, scoring, and the maths**

| File | Read it when you need to… |
|---|---|
| [dispatch-and-scoring.md](dispatch-and-scoring.md) | Understand the battery physics, the oracle, the MPC loop, and how capture ratio / regret are computed. The traps that voided every earlier result live here. |
| [dispatch-under-uncertainty-maths.md](dispatch-under-uncertainty-maths.md) | The ladder of dispatch objectives — point → worst-case → EV → mean-CVaR → two-stage recourse — with the maths and worked examples. |
| [probabilistic-dispatch-explained.md](probabilistic-dispatch-explained.md) | A gentler tour of quantile fans and how a distribution (not a point) drives the battery. |
| [executors-and-dispatch-explained.md](executors-and-dispatch-explained.md) | The executor zoo end to end: open-loop, MPC, the spike gate, and the scenario/EV/CVaR modes. |
| [quantile-mpc-postmortem.md](quantile-mpc-postmortem.md) | The post-mortem on the quantile-gate MPC that failed — why it scrambled price rankings. |

**Study maps & the deep guide**

| File | Read it when you need to… |
|---|---|
| [techniques-roadmap.md](techniques-roadmap.md) | A novice-friendly roadmap of **every** technique used to forecast and dispatch — models, features, calendar encodings, quantiles, calibration, metrics, backtest, model selection, executors — with file pointers. |
| [internals-deep-dive.md](internals-deep-dive.md) | The **single-document deep dive**: the entire repo in one pass — raw maths behind every method, every training recipe with its exact hyperparameters, every finding with its number, and a self-test. Built from the two roadmaps and grounded in the source. |
| [learning_guides/](learning_guides/) | The illustrated, book-length companion — **16 chapters** (source markdown, per-chapter and merged PDFs, figures, build scripts). See its [index](learning_guides/README.md) for the chapter map and reading status. |

## The campaign, in three documents

The capture-ratio campaign — the effort that took the battery from a fictional
scoreboard to a genuine 0.56 capture on held-out data — is recorded in three
places you will be pointed back to constantly:

1. **[outputs/plans/capture_campaign.md](../outputs/plans/capture_campaign.md)** —
   the plan of record. Frozen metric definitions, the phase ladder (P0–P5),
   the ticket board, and the **trap register** (T1–T11). If you are executing a
   ticket, it is specified here.
2. **[outputs/reports/capture_campaign_report.md](../outputs/reports/capture_campaign_report.md)** —
   the findings report with figures. The "why" behind the current design.
3. **[outputs/experiment_log.md](../outputs/experiment_log.md)** — every bug,
   dead end, and surprising result, written as curriculum. Entries 013–019 are
   the campaign. **Adding to this log is mandatory** — see
   [extending.md](extending.md#the-experiment-log-rule).

## Golden rules (violate these and your results are void)

1. **Validation-first.** Tune on the validation window (Jul–Sep 2023). Touch
   the test window (Oct 2023–Jan 2024) once per accepted technique. See
   [running-experiments.md](running-experiments.md#the-two-windows).
2. **Physics from config, never hardcoded.** The interval length `dt` comes
   from `cfg["resolution"]`. Hardcoding it is the bug that voided a dozen
   earlier results (Entry 013). See
   [dispatch-and-scoring.md](dispatch-and-scoring.md#trap-1-the-dt-bug).
3. **Revenue only from clamped actions.** Never monetise a plan that has not
   passed `lp.clamp_action`. See
   [dispatch-and-scoring.md](dispatch-and-scoring.md#the-feasibility-clamp).
4. **Log every failure.** In `outputs/experiment_log.md`, with the standing
   template.
5. **Push only to the personal remote** `kieranmcgimsey/grian`. Check
   `git remote -v` before pushing.
