# How dispatch works: forecasts, the LP, and executors — a novice's guide

This explains, from the ground up, what actually happens when grian turns a
**price forecast** into **battery trades** and scores the result. No prior
optimisation background assumed. By the end you'll understand what "open-loop"
and "MPC 30-min" mean, why a **linear program (LP)** is solved, and how the
maths works — with real numbers.

---

## 1. The one-sentence version

> We own a big battery. Electricity prices swing wildly through the day. We use a
> **forecast** of future prices to decide **when to buy (charge)** and **when to
> sell (discharge)**, and an **optimiser (the LP)** works out the best schedule
> given the battery's physical limits. An **executor** is the *policy* for how
> often we re-plan. **Capture ratio** grades us against a perfect-foresight
> trader.

The whole pipeline:

```mermaid
flowchart LR
    A[Raw NEM data<br/>5-min prices + demand] --> B[Model / forecast<br/>predict next 24h of prices]
    B --> C[Executor<br/>open-loop or MPC]
    C --> D[LP solver<br/>best charge/discharge plan<br/>given battery physics]
    D --> E[Execute vs ACTUAL prices<br/>log every 5-min interval]
    E --> F[Ledger<br/>revenue, SOC, actions]
    F --> G[Capture ratio<br/>your revenue ÷ oracle revenue]
```

Everything below is just these boxes, slowly.

---

## 2. The asset: a 100 MW / 200 MWh battery

We simulate a grid battery in South Australia (region **SA1**). Its physical
specs are fixed (this is a real-ish "big battery"):

| Property | Value | Meaning |
|---|---|---|
| **Power** | 100 MW | Fastest it can charge or discharge. Like the width of the pipe. |
| **Energy capacity** | 200 MWh | How much it can store. Like the size of the tank. |
| **Duration** | 2 hours | 200 MWh ÷ 100 MW = 2 h to fill or empty at full power. |
| **Round-trip efficiency** | 0.85 | Store 100 units, get 85 back. 15% is lost to heat. |
| **Max cycles/day** | 2 | Can fully charge+discharge at most twice a day (battery wear limit). |
| **Interval** | 5 min | The market dispatches every 5 minutes. dt = 1/12 hour. |

**Why these matter:** every one of these becomes a *constraint* in the LP. You
can't discharge faster than 100 MW, can't store more than 200 MWh, can't cheat
the efficiency loss, and can't cycle more than twice a day.

### The efficiency detail (important for the maths)

The 0.85 round-trip is split evenly across the two directions using a square
root: η = √0.85 ≈ **0.922**.

- **Charging:** to put 1 MWh *into storage*, you must buy 1/η ≈ 1.085 MWh from the
  grid. (You lose ~8% on the way in.)
- **Discharging:** to deliver 1 MWh *to the grid*, you must drain 1/η ≈ 1.085 MWh
  from storage. (You lose ~8% on the way out.)
- Round trip: buy 1.085 in, deliver 0.922 of what you stored out → 0.85 overall.

This is why you can't make money from tiny price wiggles: the spread has to beat
the ~15% efficiency tax first.

---

## 3. The job: arbitrage

Electricity can't easily be stored at grid scale — except in a battery. NEM
prices are extraordinarily volatile: often **negative** (you get *paid* to
consume) in the sunny middle of the day when solar floods the grid, and
occasionally **$16,600/MWh** (the market price cap) during a scarcity event on a
hot evening.

The trade is simple in spirit:

> **Buy low, sell high.** Charge when prices are cheap/negative, discharge when
> prices spike.

The hard part is *timing* under uncertainty — you don't know today what tonight's
price will be. That's what the forecast is for, and what makes this interesting.

---

## 4. The forecast: what the model produces

A **model** (naive, autoregression, LightGBM, …) looks at history up to *now* and
predicts the price for each of the next **288 five-minute intervals** (= 24 hours,
"day-ahead"). That's the **horizon**.

```
forecast = [ p̂₀, p̂₁, p̂₂, …, p̂₂₈₇ ]     ($/MWh, one per 5-min step)
```

The models differ only in *how* they predict:

- **naive_similar_day** — "tomorrow 6pm looks like the most recent similar day's
  6pm." No fitting. The floor everyone must beat.
- **autoregression (AR)** — a linear formula on recent price lags + time-of-day.
- **lightgbm_rich** — gradient-boosted trees on a rich feature set (lags, rolling
  stats, demand, calendar, momentum, scarcity signals).

Crucially, the forecast is **just a vector of numbers**. It does not decide any
trades. That's the executor + LP's job. A model that forecasts *accurately* is
not automatically a model that *trades well* — remember this, it's the punchline
later.

---

## 5. The LP: the optimiser that turns a forecast into a plan

Given a price vector (forecast, or for the oracle, the *actual* prices), the LP
answers one question:

> **What charge/discharge schedule maximises revenue, without breaking any of the
> battery's physical limits?**

"LP" = **Linear Program**: an optimisation where both the thing you're maximising
and all the constraints are *linear* (straight-line) in the decision variables.
Linear problems can be solved to the *global optimum* fast and reliably (grian
uses the HiGHS solver). That's why we bother formulating it this way instead of
guessing.

### 5.1 The decision variables

For each 5-minute step `t` (t = 0 … T−1), the solver chooses three numbers:

| Variable | Units | Bounds |
|---|---|---|
| `charge[t]` | MW | 0 … 100 |
| `discharge[t]` | MW | 0 … 100 |
| `soc[t]` — state of charge *after* step t | MWh | 0 … 200 |

### 5.2 The objective (what we maximise)

```
maximise   Σ_t  price[t] · ( discharge[t] − charge[t] ) · dt
```

Read it plainly: for each interval, **money in** is `price × discharge × dt`
(selling), **money out** is `price × charge × dt` (buying). `dt = 1/12` h for
5-min steps converts MW to MWh. Sum over the horizon = total revenue.

*(In the code the solver minimises the negative of this, which is the same thing.
See `src/grian/sim/lp.py`.)*

### 5.3 The constraints (the battery's physics)

**(a) Storage bookkeeping (SOC dynamics)** — the tank level after each step:

```
soc[t] = soc[t−1] + η · dt · charge[t]  −  (dt / η) · discharge[t]
```

Charging *adds* η·(energy bought); discharging *removes* (energy delivered)/η.
The efficiency loss lives here. Start empty: `soc[−1] = soc0 = 0`.

**(b) Power limits:** `0 ≤ charge[t] ≤ 100`, `0 ≤ discharge[t] ≤ 100` MW.

**(c) Energy limits:** `0 ≤ soc[t] ≤ 200` MWh. Can't overfill or go below empty.

**(d) Cycle limit (throughput budget):** total energy discharged in a calendar
day can't exceed 2 cycles:

```
Σ_(t in day)  discharge[t] · dt   ≤   2 × 200 = 400 MWh per day
```

That's the whole model. Linear objective, linear constraints → an LP.

### 5.4 A worked example (real arithmetic)

Imagine a simplified day. Ignore efficiency for a second to build intuition, then
we'll add it back.

> Prices: **$20/MWh all morning**, then a **$300/MWh spike for 2 hours** in the
> evening, $60 otherwise.

The LP will:

1. **Charge** at 100 MW through the cheap morning until the tank is full
   (200 MWh takes 2 hours at 100 MW).
2. **Hold.**
3. **Discharge** at 100 MW into the $300 spike (200 MWh empties in 2 hours).

Without efficiency:
```
Cost to charge:   200 MWh × $20   = $4,000
Revenue selling:  200 MWh × $300  = $60,000
Net profit:       $56,000   (one cycle)
```

Now **with** the 0.85 efficiency:
```
To store 200 MWh you must BUY 200 / 0.922 = 216.9 MWh
  → charge cost = 216.9 × $20  = $4,338
From 200 MWh stored you DELIVER 200 × 0.922 = 184.4 MWh
  → sell revenue = 184.4 × $300 = $55,320
Net profit ≈ $50,982   (one cycle)
```

The efficiency tax cost ~$5k here — fine, because the spread ($20 → $300) is
enormous. If the spread were only $20 → $25, the tax would eat the profit and the
LP would correctly choose to **do nothing**. The cycle limit (2/day) means if
there were *two* separate spikes, it could capture both but not a third.

This is exactly what the LP does automatically, across 288 intervals, respecting
every constraint simultaneously — something you can't eyeball.

---

## 6. Executors: open-loop vs MPC — the *re-planning policy*

Here's the crux the dashboard compares. The model and the LP are the same; the
**executor** decides *how often you re-plan as the day unfolds and reality
arrives.*

### 6.1 Open-loop (plan once, commit)

```
At 00:00:  forecast the whole day  →  solve ONE LP  →  get a 288-step plan
All day:   execute that plan exactly, come what may.
```

Simple. But the plan is only as good as the *morning's* forecast. If at 6pm the
real price does something the morning forecast didn't expect, tough — you're
already committed to whatever you planned at midnight.

Think of it as **writing your whole day's to-do list at breakfast and refusing to
look at your phone until midnight.**

### 6.2 MPC 30-min (receding horizon — re-plan as you go)

**MPC = Model Predictive Control.** Every 30 minutes it throws away the old plan
and makes a fresh one from the *current true state*:

```
Every 30 min (reforecast_every = 6 steps):
   1. Regenerate the forecast using all data observed up to now.
   2. Re-solve the LP starting from the battery's ACTUAL current SOC.
   3. Execute only the next 30 min of that new plan.
   4. 30 min later, repeat.
```

Two knobs control it (both counted in 5-minute steps):

- `reforecast_every = 6` → make a fresh forecast every 30 min.
- `resolve_every = 6` → re-solve the LP every 30 min.

Think of it as **re-checking your phone every 30 minutes and rewriting the rest of
your to-do list based on what actually happened.**

> **Note on "30-min":** the market itself settles every **5 minutes** (has since
> Oct 2021). The "30" is *how often the controller re-plans*, not the settlement
> interval. The battery still acts every 5 minutes — it just re-optimises its plan
> every 30. Re-planning every 5 min was tested and was *worse* (it chases noise).

### 6.3 A timeline showing the difference

Say the morning forecast expected an evening peak at **7pm**, but the real spike
arrives at **6:30pm** and is bigger than forecast.

- **Open-loop:** committed at midnight to discharge at 7pm. At 6:30pm it's still
  holding (or half-committed), partly misses the real spike. Locked in.
- **MPC:** at 6:00pm it re-forecasts, now sees the spike is imminent and larger,
  re-solves, and starts discharging into it. It *adapts*.

That's the theory of why MPC should win. **But** — and this is the finding from
grian's own runs — MPC only helps if the *fresh* forecasts are actually better
information. With a **weak or flat forecast**, re-planning every 30 minutes just
**whipsaws** the battery (buy, change your mind, sell low, change again), and MPC
can do *worse* than open-loop — even lose money. A confident-but-wrong forecast,
acted on repeatedly, is dangerous.

---

## 7. Scoring: the oracle and capture ratio

How good is "$X of revenue"? Meaningless on its own — it depends on how much
opportunity the market offered. So we compare to a **perfect-foresight oracle**:

> The **oracle** runs the *same LP on the ACTUAL prices* (not a forecast). It's
> the most money any trader could *possibly* make with this exact battery — an
> omniscient upper bound.

Then:

```
capture ratio = your revenue ÷ oracle revenue
```

- **1.0** = you matched a trader who knew the future perfectly. (Impossible in
  practice.)
- **0.5** = you captured half the theoretically available arbitrage value.
- **negative** = you actually *lost* money (bought high, sold low). Yes, this
  happens with bad forecasts + aggressive re-planning.

Two companion numbers on the dashboard:

- **Regret ($)** = oracle − you = dollars left on the table.
- **Regret (%)** = 1 − capture.

**Why a ratio?** It normalises out the raw size of the window's opportunity, so
you can compare configs. It does **not** normalise out the *regime* — a spikier
year has different dynamics — which is why comparing capture across *different
time windows* is only roughly fair.

---

## 8. The options, in one place

| Knob | What it does | grian default |
|---|---|---|
| **model** | how prices are forecast | naive / AR / lightgbm_rich / lightgbm_qmean |
| **executor** | re-planning policy | open-loop, or MPC 30-min |
| **reforecast_every** | how often MPC makes a fresh forecast | 6 steps = 30 min |
| **resolve_every** | how often MPC re-solves the LP | 6 steps = 30 min |
| **refit_days** | how often the model is *retrained* on new history | 7 (campaign) / 28 (recent runs, for speed) |
| **horizon** | how far ahead each plan looks | 288 steps = 24 h |
| **power / capacity / efficiency / cycles** | battery physics → LP constraints | 100 MW / 200 MWh / 0.85 / 2 |

Note **refit** (retraining the model) is different from **reforecast**
(re-running the *existing* model on newer inputs). MPC reforecasts constantly but
only refits every `refit_days`.

---

## 9. Why the results can surprise you (grian's own findings)

Three things grian's common-window evaluation found — each a lesson:

1. **Accuracy ≠ trading value.** LightGBM forecasts prices most *accurately*
   (best MAE) yet a simpler model captured *more revenue*. Capture rewards getting
   the **timing/ranking** of highs and lows right, not minimising average error.
   A forecast that nails the level but blurs the daily *shape* trades poorly.
2. **MPC is not free lunch.** Re-planning helps only when fresh forecasts add real
   signal. On weak forecasts it whipsaws and can go negative.
3. **The window and the refit cadence change everything.** The same config wins on
   one year and loses on another. That's why the dashboard makes the test window a
   slider rather than baking one number into folklore.

---

## 10. Glossary

- **Arbitrage** — buy low, sell high on the same asset across time.
- **SOC (state of charge)** — how full the battery is, in MWh.
- **LP (linear program)** — optimise a linear objective under linear constraints;
  solvable to a global optimum efficiently.
- **Horizon** — how many steps ahead a plan covers (288 = day-ahead).
- **Executor** — the policy for *when* to re-plan (open-loop vs MPC).
- **MPC (model predictive control)** — re-solve the plan every few steps from the
  true current state (receding horizon).
- **Open-loop** — solve once, execute without correction.
- **Reforecast** — re-run the existing model on newer data.
- **Refit** — retrain the model itself on more history.
- **Oracle** — the LP run on actual (perfectly known) prices; the revenue ceiling.
- **Capture ratio** — your revenue ÷ oracle revenue.
- **Regret** — revenue you left on the table vs the oracle.
- **η (eta)** — √(round-trip efficiency) ≈ 0.922; the per-direction loss factor.

---

**Where to go next in the code:**

- `src/grian/sim/lp.py` — the LP (Section 5) in ~120 lines.
- `src/grian/sim/oracle.py` — the perfect-foresight oracle (Section 7).
- `src/grian/sim/mpc.py` — the MPC executor (Section 6.2).
- `src/grian/sim/runner.py` — the open-loop walk-forward loop.
- `scripts/run_common_eval.py` — runs every model × executor on one window.
