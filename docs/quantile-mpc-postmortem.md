# Quantile-based MPC: what failed, why, and what's next

A postmortem of the probabilistic-dispatch attempts, and the roadmap. Pairs with
`probabilistic-dispatch-explained.md` (the concepts) and experiment_log Entry 025.

---

## The goal (recap)

Point-forecast MPC over-reacts: it treats a noisy forecast as certain and
whipsaws (buys high, sells low). The fix is to make dispatch **uncertainty-aware**
— trade decisively when the forecast fan is tight, hold back when it's wide.
`lightgbm_qmean` already emits that fan (q05, q50, q90, q98 per step). The
question was *how the dispatch should consume it.* Two attempts failed; a third
looks right.

---

## Attempt 1 — `mpc_qgate` (the quantile gate) — **FAILED**

### How it works

The LP takes a single price vector. To make it pessimistic without touching the
LP, the gate **rewrote each interval's price with a quantile**, chosen by whether
the interval looked like a buy or a sell:

```
for each step t:
    if median[t] >= day-median:   # looks like a SELL step
        price[t] = q10[t]         #   → use a LOW quantile (don't bank on the upside)
    else:                         # looks like a BUY step
        price[t] = q90[t]         #   → use a HIGH quantile (don't bank on the drop)
```

The intent: compress the spread so the LP only trades when the arbitrage survives
the pessimistic view — a "probabilistic no-trade band."

### Why it failed

- **Capture 0.032** (vs 0.362 for the same model under point MPC).
- It did **not** stop trading — near-identical volume — it **mistimed**. Realised
  avg charge/discharge went from **$7.6 / $83.5** (spread +$76) under point MPC to
  **$43.5 / $57.3** (spread **+$13.9**) under the gate. It bought *high* and sold
  *low*.

**Root cause: it scrambles the cross-interval price ranking.** The LP decides
*which* intervals to charge/discharge purely from the **relative order** of prices
across the horizon — charge the cheapest, discharge the dearest. The gate replaces
each interval's price with a quantile, and because **forecast uncertainty varies
across intervals**, that replacement is *uneven*: a very-uncertain cheap interval
(wide fan → high q90) can be lifted *above* a confident expensive one. The ordering
inverts, so the LP charges intervals that aren't actually cheap and discharges ones
that aren't actually dear. Compressing a spread is fine; **reordering the intervals
is fatal.**

---

## Attempt 2 — `mpc_robust` (tail-quantile gate) — **FAILED worse**

Identical mechanism, but with tail quantiles (q02 / q98) for "more robustness."
Wider, more uneven substitution → **more** ranking scramble → **capture −0.040**
(lost money). Same disease, higher dose.

**Lesson (Entry 025):** never distort the cross-interval ranking the LP depends on.
A correct robust dispatch must reason over *coherent, internally-ranked* price
paths — not per-interval quantile swaps.

---

## Next up 1 — `mpc_scenario` (per-scenario LPs) — **built, validating**

### How it works

Treat the fan as a set of **whole scenarios** — each quantile level is one coherent
price *path* across the horizon (the q05 path, q50 path, q90, q98). Then:

```
for each quantile path:
    solve the FULL LP on that path      # each path is internally ranked → coherent plan
robust action (this block) =
    min over paths of charge,  min over paths of discharge
```

The `min` means **"trade only to the extent every scenario agrees."** If the
pessimistic q05 path *and* the optimistic q98 path both want to discharge here, the
spread is robust → act. If they disagree, `min → 0` → hold. Crucially, **no path's
ranking is ever distorted** — each LP sees a coherent price path — so the failure
mode of the gate can't happen.

### Evidence so far

Smoke test (16-day window): bought at **−$63.6** (paid to charge on deep negative
troughs), sold at **$117**, realised spread **+$180.7** — far better timing than
point MPC's +$76. It's a *sniper*: sits out the ambiguous middle, fires only on
deep unanimous opportunities.

**Open caveat:** it trades *very* selectively (high spread, low volume). Capture =
revenue ÷ oracle, so if it under-trades, capture could still trail point MPC
*despite* the beautiful spread. **A full-window capture number is running now**
(`lightgbm_qmean` mpc30 vs mpc_scenario, 90-day summer window) — that decides
whether it's actually good or just tasteful.

---

## Next up 2 — full 2-stage scenario / CVaR LP (the principled version)

`mpc_scenario`'s `min-across-scenarios` is a robust *heuristic* — effectively the
worst-case (infinitely risk-averse) limit, which explains the under-trading. The
principled version is a **two-stage stochastic program**:

- **One shared here-and-now action** (the present is common to all scenarios) +
  **per-scenario recourse** (future actions can differ once you learn the outcome).
- Objective: **maximise `E[revenue] − λ·CVaR`**, where CVaR penalises the worst-α%
  scenarios and **λ tunes risk appetite** (λ=0 risk-neutral … λ→∞ = the `min`
  heuristic above).

This uses the scenario *probabilities* and lets you **dial selectivity** with λ,
instead of the all-or-nothing `min`. It's a bigger LP (per-scenario SOC dynamics +
non-anticipativity + CVaR variables via the Rockafellar–Uryasev formulation) —
tracked as `v2-prob`.

---

## Refinements on the shortlist

1. **Soften the `min`** in `mpc_scenario` toward a probability-weighted / λ-tunable
   combination — likely fixes the under-trading without the full 2-stage build.
2. **Tune the quantile set** — 4 quantiles is coarse; more levels = a smoother fan.
3. **Calibrate the fan** (conformal prediction, Ch 08) — the whole approach assumes
   the fan's width is *honest*; if q90 is exceeded far more than 10% of the time,
   every hedge misfires. Worth checking calibration (CRPS/coverage) before trusting
   any of this.

---

## Bottom line

- The two **gate** approaches are **junk** (rank-scrambling) — kept only as negative
  baselines.
- **`mpc_scenario`** is the first *correct* design — it preserves rankings and
  behaves sensibly — but it is **not yet validated at scale**. The head-to-head
  capture (running) is the deciding number; if it under-trades, softening the `min`
  or the full CVaR LP is the fix.
