# Probabilistic dispatch: quantile-gating and robust MPC — a novice's guide

This is the sequel to [`executors-and-dispatch-explained.md`](executors-and-dispatch-explained.md).
That one explained the LP, open-loop, and MPC. This one explains **why a point
forecast makes MPC self-destruct**, and the two fixes: **quantile-gated trading**
and **scenario / robust MPC**. No optimisation background assumed.

---

## 1. The problem in one sentence

> MPC re-plans constantly and **treats each forecast as if it were certain** — so
> when the forecast is a noisy guess, MPC bets the farm on noise, buys high, sells
> low, and loses to a battery that just committed to a simple plan (open-loop).

We proved this: with a **perfect** forecast MPC captures ~1.0; as forecast noise
grows, MPC collapses (0.37 at high noise) while open-loop degrades gently (0.67).
The disease is **over-trust**, not the LP (the LP is correct). The cure is to give
the dispatcher a sense of **how sure** the forecast is — i.e. a *probability
distribution* over future prices, and a dispatch that respects it.

---

## 2. What a probabilistic forecast actually is

A **point** forecast says: "6pm price = \$180." A **probabilistic** forecast says:
"6pm price is *most likely* ~\$120, but there's a 10% chance it's below \$40 and a
10% chance it's above \$400." It hands you **quantiles** instead of one number:

```
q10  q25  q50(median)  q75  q90
 40   80     120       210  400
```

The gap between the low and high quantiles is the **uncertainty**. A *confident*
forecast has a narrow fan (q10≈q90); an *unsure* one has a wide fan. grian's
quantile models (`lightgbm_qmean`, the new `qra_ensemble`) already produce this
fan — we were just **throwing it away** by collapsing to the mean before dispatch.

> **QRA (quantile regression averaging)** — the ensemble: take the quantile fans
> from several models and blend them per-quantile into one better-calibrated fan.
> Averaging reduces each model's idiosyncratic error, like a poll-of-polls.

---

## 3. Fix #1 — Quantile-gated trading (the cheap, intuitive one)

Recall the **efficiency tax**: with 0.85 round-trip you must sell at **≥ 1.18×**
your buy price just to break even. So you should only trade on a spread you're
*confident* clears that bar.

**Quantile-gating** feeds the LP a **direction-pessimistic** price vector:

- For intervals you might **sell into**, use a **low** quantile (say q25) — *don't
  count on the upside being as big as the median hopes.*
- For intervals you might **buy at**, use a **high** quantile (say q75) — *don't
  count on the price dropping as much as you hope.*

Now the LP only acts when the spread survives the pessimistic view. It's a
**probabilistic no-trade band**: on a shaky "spike coming" signal (wide fan, low
q25), the pessimistic sell-price is small, the spread doesn't clear the tax, and
the LP **sits on its hands** — exactly the buy-high mistake we wanted to stop.

**Worked example.** Median forecast says: buy now \$100, sell later \$130.

- *Point MPC:* 130 ≥ 100×1.18 = 118? Yes → **charges** at \$100. If the spike
  fizzles and you sell at actual \$95 → loss.
- *Quantile-gated:* it's a wide, unsure fan — q75(buy) = \$110, q25(sell) = \$108.
  108 ≥ 110×1.18 = 130? **No** → **does nothing.** Dodged the bad trade.

One knob: how pessimistic (which quantiles). q10/q90 = very cautious; q40/q60 =
barely cautious. It's a small change on top of the existing LP.

---

## 4. Fix #2 — Scenario / robust MPC (the principled one)

Instead of one pessimistic price vector, use the **whole fan** as a set of
possible futures ("**scenarios**") and optimise across all of them at once.

### The two-stage idea (here-and-now vs recourse)

At the current moment you must pick **one** action *now* — you can't act
differently in different scenarios, because the present is shared. But the
*future* actions can differ per scenario (you'll adapt once you learn which one
you're in). So:

```
choose ONE action for now (shared across all scenarios)
   +  a tailored plan for each scenario's future
so as to maximise  EXPECTED revenue across scenarios.
```

Because the *now* action must be good **on average across all futures**, it
naturally **hedges**: it won't charge hard on a spike that only 1 of 5 scenarios
predicts. When scenarios agree (narrow fan), it acts decisively; when they
disagree (wide fan), it holds back. Uncertainty-awareness falls out of the maths
for free.

### Being risk-averse: CVaR

Maximising the *average* still lets a strategy love a coin-flip that's great on
average but occasionally catastrophic. Batteries (and traders) don't want the
catastrophic tail. So we add a penalty on the **worst** scenarios:

```
maximise   E[revenue]  −  λ · CVaR_α(loss)
```

- **CVaR_α** ("conditional value at risk") = the average outcome in the worst
  **α%** of scenarios (e.g. worst 20%). Penalising it means "make the bad case
  not-too-bad," not just "make the average good."
- **λ** dials risk appetite: λ=0 is risk-neutral (pure expected value); large λ is
  very cautious (hug the worst case, like robust optimisation).

This is the general tool; quantile-gating (#3) is essentially its cheap, hand-set
special case.

---

## 5. Why this fixes the MPC over-reaction

Point MPC fails because it re-plans on a single noisy number and bets fully on it.
Both fixes make it **bet in proportion to its confidence**:

- Confident forecast (narrow fan) → trades decisively → keeps MPC's real edge
  (adapting to genuine changes).
- Unsure forecast (wide fan) → holds → stops the whipsaw and the efficiency-taxed
  churn.

So MPC should stop losing to open-loop when forecasts are noisy, and still win
when they're sharp. That's the hypothesis these runs test.

---

## 6. The catch: it needs a *calibrated* fan

Garbage in, garbage out. If the model says "q90 = \$400" but real prices exceed
\$400 half the time, the fan lies and the hedging misfires. That's why
**calibration** and **CRPS** (a score for the whole distribution, not just the
mean) matter — see Ch 08. And it's why **QRA + conformal** (which re-calibrate the
fan to have honest coverage) are the right forecast side to pair with this.

---

## 7. Where this sits in the bigger idea (decision-focused learning)

A model trained to minimise *forecast error* (MAE) is not trained to *make good
trades* — the two objectives differ (Ch 16). Uncertainty-aware dispatch is one
half of closing that gap from the **decision** side: even with an imperfect
forecast, use its *uncertainty* to make robust decisions. The other half is
training the forecaster on downstream **dollars** rather than error — the frontier
we're not building yet, but this is a step toward it.

---

## 8. Glossary

- **Quantile / qN** — the value the price falls below N% of the time. q90 = a high
  price you'd only exceed 10% of the time.
- **Fan** — the spread of quantiles; its width = forecast uncertainty.
- **QRA** — quantile regression averaging; an ensemble that blends model fans.
- **Conformal prediction** — post-hoc recalibration so the fan has honest coverage.
- **Quantile-gating** — feed the LP direction-pessimistic quantiles so it only
  trades on robust spreads.
- **Scenario** — one possible future price path, drawn from the fan.
- **Here-and-now vs recourse** — the shared present action vs per-scenario future
  actions in a two-stage stochastic program.
- **CVaR_α** — average outcome in the worst α% of scenarios; a tail-risk measure.
- **λ (lambda)** — how strongly we penalise the bad tail (risk appetite).
- **Robust optimisation** — optimise against the worst case (≈ large λ).

---

**Read next / alongside:** Ch 08 (probabilistic forecasting, CRPS, QRA,
conformal) and Ch 16 (decision-focused learning). Code will land in
`src/grian/sim/dispatch_prob.py` (quantile-gate + scenario/robust LP) and a new
`qra_ensemble` model in `src/grian/sim/models.py`.
