# grian docs — start here

This directory is the practical onboarding path for the **grian simulation
environment**: the walk-forward battery-trading test bench in `src/grian/`.
If you have been handed a ticket against this repo, read the guide that matches
your task and follow it — each is a walkthrough with runnable commands, not a
reference dump.

These docs are about the **grian simulation environment** and the capture-ratio
campaign built on it.

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

Each is a self-contained walkthrough. Start with `architecture.md` for how the
pieces fit, then reach for the specific guide you need.

**Onboarding & orientation**

| File | Read it when you need to… |
|---|---|
| [architecture.md](architecture.md) | Understand how the pieces fit: modules, data flow, the model registry, the config schema, where state lives. **Read this first.** |
| [running-experiments.md](running-experiments.md) | Run a trial, an MPC trial, an ablation, or a sweep — and read the results correctly. |
| [extending.md](extending.md) | Add a model, a feature group, or an executor knob without breaking the physics or the reproducibility guarantees. |
| [data-and-features.md](data-and-features.md) | Understand the data sources, the timestamp/weather handling, and every feature group `features.py` builds. |

**Dispatch & scoring**

| File | Read it when you need to… |
|---|---|
| [dispatch-and-scoring.md](dispatch-and-scoring.md) | Understand the battery physics, the oracle, the MPC loop, and how capture ratio / regret are computed. The traps that voided every earlier result live here. |

## The campaign record

The capture-ratio campaign — the effort that took the battery from a fictional
scoreboard to a genuine ≈0.55 capture on held-out data — is recorded in
**[outputs/experiment_log.md](../outputs/experiment_log.md)**: every bug, dead end,
and finding, with root-cause write-ups (Entries 001–038). **Adding to this log is
mandatory** — see [extending.md](extending.md#the-experiment-log-rule).

## Golden rules (violate these and your results are void)

1. **One common window, confirmed across regimes.** Score every configuration on
   the same held-out year (2025-07 → 2026-06) by balanced (per-month) capture, and
   confirm any apparent win on regime-contrasting sub-windows before believing it.
   See [running-experiments.md](running-experiments.md#the-common-window).
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
