# Experiment Log

This log documents bugs, modelling errors, surprising results, and lessons
learned during the development and training of the grian simulation environment.
Each entry is written to teach — if you encounter a similar issue, the reasoning
and resolution (or lack thereof) should help you avoid the same trap.

**Rule: every modelling failure, unexpected result, or code bug discovered during
experimentation must be logged here with a full explanation of the what, why,
and how. This is not optional. The log exists to make failures as valuable as
successes.**

---

## Entry 001: LightGBM step stride miscalculation (288 boosters instead of 25)

**Date:** 2026-07-08
**Component:** `src/grian/sim/models.py` — `_lgbm_fit`
**Severity:** Performance — training 11.5x slower than intended

### What happened

The LightGBM direct multi-step model trains one booster per forecast step. To
keep training tractable for day-ahead (288 five-minute intervals), we stride
across steps — training one booster per hour and interpolating the rest. The
stride calculation was:

```python
step_stride = max(1, horizon // ppd)
```

When `horizon == ppd == 288`, this gives `288 // 288 = 1` — a stride of 1,
meaning 288 separate boosters per refit. Each refit took ~30 seconds, and with
weekly refitting over 123 test days, the run was killed after 30+ minutes with
no end in sight.

### Why it happened

The formula confused "horizon steps" with "days." It was trying to compute
"how many days does the horizon span" but the numerator and denominator were
both in intervals, cancelling out. For day-ahead at 5-minute resolution,
horizon/ppd = 1 (one day), which is not a useful stride.

### The fix

Replace with a resolution-aware stride:

```python
intervals_per_hour = max(1, 60 // (1440 // ppd))
step_stride = intervals_per_hour  # 12 for 5min, 2 for 30min
```

This gives 25 boosters (288/12 = 24, plus the final step) — a 11.5x speedup.
Training dropped from 30+ minutes to ~5 minutes per trial.

### Lesson

When writing stride/step calculations, always sanity-check the result for your
actual resolution. "Does this give me a reasonable number of models?" is a
one-line assert that would have caught this immediately. Dimensional analysis
on the formula would also have shown that horizon/ppd is dimensionless, not
a step count.

---

## Entry 002: LightGBM loss function not wired to config

**Date:** 2026-07-08
**Component:** `src/grian/sim/models.py` — `_lgbm_fit`
**Severity:** Correctness — `wrong_loss` ablation produced identical results

### What happened

The ablation `lightgbm_wrong_loss` was designed to demonstrate the effect of
using MSE loss on heavy-tailed NEM prices. It sets `cfg["loss"] = "mse"` while
the correct baseline uses `cfg["loss"] = "pinball"`. Both trials produced
identical results: revenue -$348,803, MAE 70.19.

### Why it happened

The `loss` field in the config was never read by the LightGBM model code.
LightGBM always used its default objective (`regression`, which is L2/MSE).
The config field was metadata only — it described the *intent* but had no
effect on the actual training.

This is a classic "configuration theatre" bug: the config looks correct, the
ablation looks properly specified, but the setting is never consumed. Without
checking that the ablation actually changed the output, it's invisible.

### The fix

Wire the config `loss` field into LightGBM's `objective` parameter:

```python
loss = cfg.get("loss", "pinball")
if loss == "pinball":
    lgb_params["objective"] = "quantile"
    lgb_params["alpha"] = 0.5  # median regression — robust to outliers
elif loss == "huber":
    lgb_params["objective"] = "huber"
# else: default "regression" (MSE)
```

Similarly, the MLP model was hardcoded to `nn.MSELoss()`. Fixed to use
`nn.SmoothL1Loss()` (Huber) when `loss == "pinball"`, which is more robust
to NEM price outliers.

### Lesson

Every ablation must be verified by checking that the output *actually changes*.
If two configs produce identical results, either the ablation flag isn't being
consumed, or the difference is too small to measure. The first case is a bug;
the second means the ablation is uninformative at the current scale. Both need
to be caught before reporting results.

A good practice: after adding any config field, grep for it in the codebase.
If it only appears in config construction and never in model/runner code, it's
dead config.

---

## Entry 003: LightGBM produces systematically bad forecasts (negative revenue)

**Date:** 2026-07-08
**Component:** `src/grian/sim/models.py` — `_lgbm_predict`
**Severity:** Model quality — worse than doing nothing

### What happened

The `lightgbm_correct` baseline produced -$348,803 revenue over the 4-month
test period, compared to +$463K for naive similar-day and +$456K for linear AR.
A more complex model performed dramatically worse than trivial baselines.

### Analysis

Examining the forecast data reveals the root cause:

| Metric | Forecast | Actual |
|--------|----------|--------|
| Mean   | $9.03    | $32.83 |
| Std    | $26.52   | $280.05 |
| Max    | $103.69  | $16,600 |

The LightGBM forecasts are:

1. **Systematically biased low** — mean $9 vs actual $33. The model
   underestimates the price level.

2. **Dramatically under-dispersed** — std $27 vs $280. The model cannot
   capture the extreme volatility of NEM prices (spikes to $16,600,
   troughs to -$993).

3. **Monotonically smoothed** — each day's forecast is a slowly varying
   curve, because all 25 boosters use the same lag features (yesterday,
   2 days ago, 7 days ago) and only differ in calendar features for
   their specific forecast step.

### Why this causes negative revenue

The battery dispatch LP optimises charge/discharge schedules based on the
forecast price profile. When the forecast is smooth and low-variance, the LP
sees small arbitrage opportunities and makes small, confident bets. But actual
prices spike wildly — the battery ends up charged during surprise spikes
(missing discharge revenue) and discharged during surprise troughs (buying
expensive power back).

The naive model works because it simply repeats last week's prices, which
naturally includes the volatility and spike structure. The LP can schedule
around realistic price patterns even if the specific timing is wrong.

### Why the model is bad (not a bug)

This is not a code bug — the model is doing exactly what it was told. The
problem is that basic LightGBM with 3 lag features and calendar dummies is
insufficient for NEM price forecasting. The direct multi-step approach with
hourly interpolation compounds the problem: each booster independently
regresses to the conditional mean, and interpolation smooths out any remaining
structure.

### What would fix it

- **More features**: demand, temperature, interconnector flows, renewable
  generation, time-of-day price curves from recent history.
- **Recency features**: rolling means/stds over recent hours, not just
  point lags from yesterday/last week.
- **Quantile regression**: predict percentiles, not the mean. The dispatch
  LP benefits more from knowing "price could spike to $500" than "mean
  price is $40."
- **Different forecasting strategy**: instead of direct multi-step with
  interpolation, use a recursive approach that updates features as it
  forecasts forward, or model intra-day price shape as a profile rather
  than 25 independent points.

### Lesson

**More complex does not mean better.** A model is only as good as its features
and its match to the loss landscape. LightGBM is a powerful learner, but with
uninformative features it learns uninformative patterns. The naive baseline
wins because it implicitly encodes the full intra-day price structure.

Before investing in model complexity, always ask: "What information does this
model have access to that the baseline doesn't?" If the answer is "nothing
new, just a fancier function approximator," the model will likely underperform.

---

## Entry 004: `no_reconditioning` ablation — worst revenue but best MAE

**Date:** 2026-07-08
**Component:** `src/grian/sim/runner.py` — refit logic
**Severity:** Informational — expected result with educational nuance

### What happened

The `lightgbm_no_reconditioning` trial (train once on history, never refit)
produced the worst revenue at -$689,393 but had the lowest MAE at 67.59 among
LightGBM variants.

| Trial | Revenue | MAE |
|-------|---------|-----|
| correct | -$348,803 | 70.19 |
| no_reconditioning | -$689,393 | 67.59 |
| no_transform | -$230,918 | 81.25 |

### Why MAE and revenue can disagree

MAE measures average absolute forecast error across all intervals equally.
Revenue depends on forecast errors *at the moments that matter* — the price
spikes and troughs where the battery acts. A model can have lower average
error but systematically misforecast the high-value intervals.

The no-reconditioning model learned the training period's price distribution
well (lower MAE) but couldn't adapt to shifts in the test period's price
dynamics (worse dispatch timing, worse revenue). The refitted model has
slightly higher average error but makes better-timed dispatch decisions.

### Lesson

**Never optimise a forecasting model on MAE alone when the downstream task is
dispatch/trading.** The metric that matters is the one aligned with the
economic objective — in this case, revenue or Sharpe ratio. MAE is a useful
diagnostic, but it weights all intervals equally while revenue weights them
by price magnitude and dispatch action.

This is a general principle: the gap between forecast accuracy and decision
quality is where most value is lost or found.

---

## Entry 005: libomp architecture mismatch on Apple Silicon

**Date:** 2026-07-08
**Component:** LightGBM runtime dependency
**Severity:** Environment — blocking error on import

### What happened

```
OSError: dlopen(libomp.dylib): Library not loaded: @rpath/libomp.dylib
```

LightGBM requires `libomp` (OpenMP runtime) for parallel tree building. On
this Apple Silicon Mac, Homebrew installed the x86_64 version to
`/usr/local/Cellar/libomp/` (via Rosetta) but LightGBM expects an arm64
binary at `/opt/homebrew/opt/libomp/lib/libomp.dylib`.

### The fix

Copy the arm64 libomp bundled with scikit-learn:

```bash
cp /path/to/env/lib/python3.11/site-packages/sklearn/.dylibs/libomp.dylib \
   /opt/homebrew/opt/libomp/lib/libomp.dylib
```

### Lesson

On Apple Silicon, always verify that native dependencies match the
architecture. `file /path/to/lib.dylib` shows `arm64` or `x86_64`. When
Homebrew installs a Rosetta (x86_64) version, conda or pip may have the
correct arm64 version bundled with another package.

---

## Entry 006: `wrong_loss` ablation identical to correct (pre-fix)

**Date:** 2026-07-08
**Component:** `scripts/run_baselines.py`
**Severity:** Experimental validity

### What happened

The `lightgbm_wrong_loss` ablation ran before the loss-wiring fix (Entry 002)
was applied. Its results are identical to `lightgbm_correct` because both
used the same default LightGBM objective. The saved artifacts for this trial
are not valid — they do not demonstrate the effect of a wrong loss function.

### Resolution

This trial needs to be re-run after the fix. Delete
`outputs/trials/lightgbm_wrong_loss/` and re-run with the corrected code.
Until then, do not include `lightgbm_wrong_loss` in ablation comparisons.

### Lesson

When a code fix changes model behaviour, all previously-run trials that were
affected by the bug must be invalidated and re-run. Saved artifacts carry no
provenance about which version of the code produced them unless you explicitly
record it. The `git_sha` field in trial configs helps — but only if you check
it.

---

## Entry 007: Future leakage ablation makes results *worse*, not better

**Date:** 2026-07-08
**Component:** `src/grian/sim/runner.py` — `simulate_region`
**Severity:** Informational — counter-intuitive result

### What happened

The `lightgbm_future_leakage` ablation, which deliberately includes future
data in the training set, produced *worse* results than the correct baseline:

| Trial | Revenue | MAE |
|-------|---------|-----|
| correct | -$348,803 | 70.19 |
| future_leakage | -$402,722 | 72.36 |

This is the opposite of the expected outcome. Leakage is supposed to make
results suspiciously *good* — the model should memorise future prices and
produce unrealistically accurate forecasts.

### Why it happened

The leakage in this implementation extends the training set by `horizon`
intervals past the embargo boundary. For LightGBM with lag features
(yesterday, 2 days ago, 7 days ago), this means the model sees some
overlapping data but doesn't directly memorise the forecast target.

The lag features are historical values — they don't include the actual future
prices being forecasted. The leakage adds `horizon = 288` extra intervals
(one day) of training data. This slightly shifts the training distribution
but doesn't give the model direct access to the answers.

For leakage to be dramatically visible, the model would need features that
directly encode the target (e.g., "price at time t+k" as a feature for
predicting price at time t+k), or the horizon would need to overlap with the
lag structure so that a lagged feature at prediction time is actually a
future value.

In this case, the extra training data from the overlapping window just adds
noise — slightly worsening the model. The leakage is there but it's too
subtle to exploit with these simple features.

### Why there is no fix

This isn't a code bug. The ablation correctly leaks future data. The lesson
is that leakage severity depends on the interaction between:

1. **How** the leak happens (training set extension vs feature contamination)
2. **What features** the model uses (lag features vs direct future access)
3. **How far** the leak extends relative to the forecast horizon

### Lesson

**Not all leakage is created equal.** The textbook example — where leakage
produces magically good results — requires that the leaked information
actually reaches the model's decision boundary. Extending the training set
by one day, when the model uses week-old lags, is a very mild form of
leakage.

The most dangerous leakage is in feature engineering: when a feature
calculated at prediction time accidentally uses information from the future
(e.g., a daily average that includes the afternoon you're trying to
forecast). This form is invisible in the training pipeline and produces
exactly the "too good to be true" results.

If your leakage ablation doesn't make results dramatically better, it
doesn't mean leakage is harmless — it means your ablation isn't leaking
through the right channel.

---

## Entry 008: `no_embargo` and `future_leakage` produce identical results

**Date:** 2026-07-08
**Component:** `src/grian/sim/runner.py` — embargo and leakage logic
**Severity:** Informational — reveals structural limitation

### What happened

| Trial | Revenue | MAE | Sharpe |
|-------|---------|-----|--------|
| future_leakage | -$402,722 | 72.36 | -19.1 |
| no_embargo | -$402,722 | 72.36 | -19.1 |

These two ablations, which test different forms of data contamination,
produced byte-identical results.

### Why it happened

Both ablations allow the model to train on data closer to (or overlapping
with) the forecast window. With LightGBM's lag features at [288, 576, 2016]
intervals (1, 2, and 7 days back), the difference between "embargo = 0" and
"train extends horizon intervals past the embargo" is negligible — in both
cases the model sees very similar training data and the lag features don't
directly encode the future.

The key insight is that **both ablations operate on the training set
boundary**, not on the feature construction. Since the features are
historical lags, shifting the training cutoff by a day doesn't change the
feature values at the prediction point. The model produces the same forecast
regardless.

### Lesson

Embargo and leakage ablations are most informative when:

1. The model uses features with **short-range temporal dependence** (e.g.,
   rolling means over the last hour, not weekly lags).
2. The leakage **contaminates the feature space**, not just the training set
   extent.
3. The model has enough capacity to **memorise** near-boundary patterns.

For a model with only weekly lags, these two ablations are effectively testing
the same thing. Future work should include a feature-level leakage ablation
(e.g., accidentally using the day-ahead price as a feature) to demonstrate
the more dangerous form.

---

## Entry 009: All LightGBM variants produce negative revenue

**Date:** 2026-07-08
**Component:** `src/grian/sim/models.py` — LightGBM model
**Severity:** Model quality — systematic failure

### What happened

Every LightGBM trial — correct baseline and all five ablations — produced
negative revenue. The naive baseline earns +$463K and simple AR earns +$456K,
yet the gradient-boosted tree model consistently loses money.

| Trial | Revenue |
|-------|---------|
| naive_similar_day | +$463,403 |
| autoregression | +$456,041 |
| simple_mlp | +$262,134 |
| lightgbm_no_transform | -$230,918 |
| lightgbm_correct | -$348,803 |
| lightgbm_wrong_loss | -$348,803 |
| lightgbm_future_leakage | -$402,722 |
| lightgbm_no_embargo | -$402,722 |
| lightgbm_no_reconditioning | -$689,393 |

### Root cause

The LightGBM direct multi-step model has a fundamental feature poverty
problem. It uses three lag features (1 day, 2 days, 7 days) and calendar
dummies. These features cannot capture:

- **Intra-day price shape** (the daily pattern of peaks and troughs)
- **Recent momentum** (prices trending up or down over the last few hours)
- **Supply-side signals** (demand, temperature, renewable generation)
- **Volatility clustering** (spikes beget spikes in NEM)

The naive model wins because repeating last week's intra-day price profile
is a better forecast than the conditional mean of three point lags. The AR
model wins because its iterative forecasting preserves temporal structure.

### Why this matters

This is the most important result in the simulation. It demonstrates that
**model complexity without feature richness is worse than simplicity.**
LightGBM is a powerful learner, but it's being asked to predict 288 future
prices from 3 historical values and a day-of-week flag. No amount of
gradient boosting can overcome that information deficit.

### What would fix it

The LightGBM model needs demand-side and supply-side features, recent price
statistics (rolling mean, std, min, max over the last N hours), and ideally
a better forecasting strategy than predict-25-points-and-interpolate.

This is future work for the curriculum — the current result is pedagogically
valuable as-is.

---

## Entry 010: Rich features turn LightGBM from loss-maker to market-beater

**Date:** 2026-07-08
**Component:** `src/grian/sim/models.py` — `lightgbm_rich`, `src/grian/sim/features.py`
**Severity:** Model quality — positive result

### What happened

The basic LightGBM model (`lightgbm_correct`) with only 3 price lags and calendar
dummies produced -$348,803 in revenue — worse than doing nothing. Adding ~30 rich
features (rolling stats, demand, momentum, intra-day profiles) via `lightgbm_rich`
flipped the result to +$557,599 — a 20% improvement over even the naive baseline
($463,403). MAE also dropped from 70.19 to 52.98 (25% better).

### Why it happened

The basic LightGBM had too few features to distinguish price regimes. With only
yesterday/2d/7d lags and hour/dow dummies, the model produced overly smooth
forecasts that couldn't capture intra-day shape or respond to recent volatility.
The LP dispatch then made bad charge/discharge decisions based on those flat
forecasts.

The rich feature set gives the model structural information it needs:
- **Rolling stats** (1h/6h/24h mean, std, min, max): recent volatility and level
- **Demand features**: demand is the primary driver of NEM spot prices
- **Momentum**: rate of change catches trending markets
- **Intra-day profile**: captures the structural daily pattern

### The fix

Created `features.py` with backward-looking feature groups and wired `lightgbm_rich`
to use them. No change to the tree architecture — same 300 estimators, same
direct multi-step approach.

### Lesson

In electricity price forecasting, the feature set matters more than the model
architecture. A simple gradient-boosted tree with rich features crushes a
sophisticated tree with impoverished features. This is a general ML principle,
but NEM prices make it vivid: the market has strong structural patterns (demand
cycles, ramp events, price regimes) that simple lag features cannot capture.

Before reaching for a more complex model (LSTM, transformer, etc.), exhaust the
feature engineering space with trees. Trees are fast to train, easy to interpret,
and expose feature importance directly.

---

## Entry 011: LSTM OOM killed — full dataset on MPS at once

**Date:** 2026-07-08
**Component:** `src/grian/sim/models.py` — `_lstm_fit`
**Severity:** Environment — training crashed with exit code 137 (OOM)

### What happened

The LSTM training process was killed by the OS (exit code 137 = SIGKILL from
OOM killer) during the walk-forward simulation. No error traceback was printed
because SIGKILL is instant — Python doesn't get a chance to handle it.

### Why it happened

The training code allocated the entire dataset as a single tensor on MPS (Apple
Silicon GPU):

```python
X_t = torch.tensor(X_all, device=device).unsqueeze(-1)  # ~113k × 288 × 1
y_t = torch.tensor(y_all, device=device).unsqueeze(-1)  # ~113k × 1
```

For 113k training sequences of length 288 in float32, that's ~125 MB for X alone.
But the LSTM forward pass creates hidden states and gradients that multiply memory
usage by 3-5×, easily exceeding the MPS shared memory budget, especially when
the model is refit every 7 days (17 refits, each allocating a fresh full-size
tensor).

### The fix

Keep data on CPU, move only the current batch to device:

```python
X_t = torch.tensor(X_all).unsqueeze(-1)  # stays on CPU
# In training loop:
xb = X_t[idx].to(device)  # only batch_size × 288 × 1 on MPS
```

### Lesson

Never put your full training set on GPU/MPS. Even if it fits in VRAM for a
feedforward model, recurrent models (LSTM, GRU) expand memory linearly with
sequence length through their hidden state chain. Always use batch-wise device
transfer. The pattern `data.to(device)` in PyTorch tutorials works for MNIST
but fails at scale.

Exit code 137 with no traceback is the signature of an OOM kill — check
`dmesg` or system logs to confirm, and suspect memory whenever a training
process dies silently.

---

## Entry 012: LSTM underperforms naive baseline — autoregressive decode on raw price

**Date:** 2026-07-08
**Component:** `src/grian/sim/models.py` — `lstm`
**Severity:** Model quality — negative result

### What happened

The 2-layer LSTM with autoregressive decoding produced -$185k revenue (vs naive's
+$463k) and MAE 67.6 (vs naive's 65.2). The LSTM is the worst-performing model
by revenue and barely worse than naive on MAE.

### Why it happened

Three compounding problems:

1. **Autoregressive error accumulation.** The LSTM predicts one step, feeds that
   prediction back as input, and repeats 288 times. Errors compound with each
   step — by the end of the forecast horizon, the model has drifted far from
   reality. This is the fundamental weakness of iterative decoding for long
   horizons.

2. **No exogenous features.** The LSTM sees only past prices. It has no demand
   data, no calendar features, no rolling statistics — the very features that
   made `lightgbm_rich` successful. An LSTM with only price history is learning
   a univariate time series model, which is provably weaker than a model with
   structural inputs.

3. **Mean-reverting forecasts.** The standardised LSTM predictions tend toward
   the training mean over long horizons, producing flat forecasts in the same
   way the basic LightGBM did — leading to the same bad dispatch decisions.

### The fix (or: Why there is no fix yet)

The LSTM architecture is not fundamentally broken — it's being asked to do too
much with too little. To make it competitive:
- Use direct multi-step output (predict all 288 steps at once, not iteratively)
- Add exogenous features as additional input channels
- Use a seq2seq encoder-decoder architecture instead of raw autoregression

These are future work. The current result is pedagogically valuable: it shows
that architecture alone (LSTM vs tree vs linear) matters less than features and
forecasting strategy.

### Lesson

Never use autoregressive decoding for long-horizon forecasting. Error compounds
exponentially with horizon length. Direct multi-step (one output per horizon
step) or encoder-decoder architectures avoid this entirely. The LightGBM direct
approach — one booster per hour, interpolate between — is the right pattern for
day-ahead electricity price forecasting.

---

## Template for future entries

```
## Entry NNN: [Short descriptive title]

**Date:** YYYY-MM-DD
**Component:** [file path or module]
**Severity:** [Performance | Correctness | Model quality | Environment | Experimental validity]

### What happened
[Factual description of the observed behaviour]

### Why it happened
[Root cause analysis — what in the code/config/data caused this]

### The fix (or: Why there is no fix)
[What was changed, or why the issue is fundamental/expected]

### Lesson
[The generalizable takeaway — what should a reader learn from this?]
```

## Entry 013: LP planned a fictional battery — dt hardcoded to 30 minutes

**Date:** 2026-07-11
**Component:** `src/grian/dispatch.py` — `schedule`
**Severity:** Correctness — all prior sim revenue figures void

### What happened

Preparing the capture-ratio campaign (see `outputs/plans/capture_campaign.md`),
code inspection found `dt = 0.5` hardcoded inside the dispatch LP while every
trial ran at 5-minute resolution (`dt = 1/12`). The LP believed each interval
moved six times more energy than physically possible: it could "fill" the
200 MWh battery in 4 intervals (20 real minutes at 100 MW stores ~31 MWh) and
treated the 288-interval day as 144 hours. The ledger, meanwhile, scored
revenue at the correct dt. Planner physics and scoring physics disagreed in
every trial run to date.

Two adjacent defects compounded it:

1. **Phantom energy at execution.** `runner.battery_dispatch` monetised the
   LP's raw charge/discharge values and only clamped the *recorded* SOC. The
   battery could be paid for discharging energy it did not hold.
2. **No SOC continuity.** The LP hardcoded `soc[0] == 0`, so each day began
   with a magically empty battery regardless of yesterday's final state, and
   `rolling_mpc` re-planned from a wrong state at every step.

### Why it happened

`schedule()` was written for the 30-minute notebooks (Chapters 5–9) and
imported unchanged into the 5-minute simulation. Nothing asserted that the
planner's dt matched the data's. The bug was invisible in relative model
comparisons — every model was handicapped by the same fictional physics —
which is exactly why it survived twelve experiment-log entries: nothing ever
compared revenue against a physical upper bound.

### The fix

- `schedule()` gained `dt_hours`, `soc0`, `terminal_soc` parameters.
- New `grian/sim/lp.py`: sparse HiGHS formulation of the same LP
  (10–50× faster, scales to the full-window oracle) plus `clamp_action`,
  the single shared feasibility clamp. Equivalence with the cvxpy model is
  unit-tested at both resolutions.
- `runner.battery_dispatch` now executes with per-interval clamping, carries
  SOC across days, and enforces the daily cycle budget at execution.
- New `grian/sim/oracle.py`: perfect-foresight LP over the full window with
  per-calendar-day cycle budgets — the capture-ratio denominator. Replaying
  the oracle's schedule through the honest executor reproduces its revenue
  to 1e-4 (capture = 1.0 by construction).

### Lesson

**Without an upper bound, you cannot see absolute nonsense.** Twelve entries
of careful relative analysis (model A beats model B) sat on top of physics
that was wrong by a factor of six, because nothing was ever compared to the
best physically achievable number. The capture ratio is not just the campaign
headline — it is the assertion that catches this whole bug class. Its unit
test ("replay the oracle, get exactly 1.0") pins planner, executor, and
scorer to the same physics.

---

## Entry 014: Models forecast from the last refit, not from now

**Date:** 2026-07-11
**Component:** `src/grian/sim/models.py` — `_naive_predict`, `_lgbm_rich_predict`
**Severity:** Model quality — systematic staleness between refits

### What happened

Both production models ignored the `input_df` argument of `predict()` and
built forecasts from state frozen at fit time (`state["series"]`,
`state["train_tail"]`). With `refit_days = 7`, forecasts drifted up to seven
days stale between refits:

- The "similar-day" naive model repeated the profile from 7 days before the
  *end of training data*, not 7 days before the forecast day — so on most
  days it repeated an 8-to-13-day-old profile with broken day-of-week
  alignment. It was only genuinely "same day last week" on refit days.
- `lightgbm_rich`'s headline features — rolling 1h/6h/24h price statistics
  and momentum (Entry 010) — were computed from a data tail that never
  advanced between refits. The model's most valuable features described a
  week-old market.

On top of this, the runner's embargo (288 intervals) blinded even the refit
model to the most recent 24 hours (see below).

### Why it happened

The `predict(state, input_df, horizon)` interface was designed for
predict-from-now, but both implementations took the shortcut of reading their
fit-time state, and the runner happened to pass a `day_data` frame the models
never touched. No test asserted that the forecast origin tracks the input.

The embargo is a separate conceptual error: embargo is hygiene for
*model-selection backtests* (it stops CV folds from leaking near-boundary
information). In a sequential trading simulation there is no such leakage —
at midnight the operator legitimately knows everything up to midnight.
Applying the backtest embargo to the sim threw away the most informative day
of history at every forecast.

### The fix

`predict()` in both models now uses the tail of `input_df` when it is
provided and long enough (falling back to fit-time state otherwise), and
campaign trials set `embargo: 0`. Unit tests pin the forecast origin to the
end of the supplied data.

### Lesson

An interface that *permits* fresh data does not *guarantee* it is used. For
any forecaster in a walk-forward loop, test the origin: hand it a longer
history and assert the forecast changes accordingly. And keep backtest
hygiene (embargo) strictly out of deployment simulation — one guards model
selection, the other measures operations; confusing them costs exactly the
recency information that rich features exist to exploit.

---

## Entry 015: Honest baselines — best rank skill, worst revenue

**Date:** 2026-07-11
**Component:** W0.4 re-baseline (`scripts/run_capture_baselines.py`)
**Severity:** Model quality — campaign baseline result

### What happened

First capture ratios under the fixed scoreboard (honest physics, SOC
continuity, embargo 0, predict-from-now), open-loop executor:

| Model | Val capture | Val Spearman | Test capture | Test Spearman |
|---|---|---|---|---|
| naive_similar_day | 0.451 | 0.566 | 0.347 | 0.575 |
| autoregression | 0.473 | 0.665 | 0.402 | 0.668 |
| lightgbm_rich | 0.389 | 0.632 | **0.301** | **0.688** |

Oracle: $11.99M over 92 validation days (51.7% in the top 10 days);
$7.71M over 122 test days (30.7% in the top 10). No open-loop model
reaches 0.50.

### The headline anomaly

`lightgbm_rich` has the **best** within-day rank correlation on test
(Spearman 0.688) and the **worst** capture ratio (0.301). It knows the
*shape* of the day better than any baseline, yet earns least. Under the
broken physics of Entry 010 it had looked like the winner; with honest
execution it is a clear loser.

### Why

The model is trained with pinball loss at α = 0.5 on asinh-transformed
price: it predicts the **median of a heavily right-skewed distribution in
a compressed space**. Both choices systematically shrink forecast spikes
(the median ignores the tail; the asinh inversion of a central estimate
under-disperses further). The LP allocates its two daily cycles by expected
*magnitude*, not rank alone — a spread that looks like $150 when reality
offers $900 makes the LP cycle timidly or spend its budget on the wrong
window. The naive model, which repeats an actual historical day — spikes
included — feeds the LP realistic magnitudes even though its ranking is
mediocre.

Also notable: the validation window (winter, Jul–Sep) is *more*
spike-concentrated than the summer test window (51.7% vs 30.7% top-10
share), and every model captures ~7–10 points less on test. Techniques must
be judged on both regimes, not tuned to one.

### What this confirms

Campaign plan §3.2 (the LP needs the conditional mean, not the median —
trap T3) is now empirically demonstrated: rank skill without magnitude
calibration does not monetise. Phase 2 (quantile set → dollar-space mean)
targets exactly this gap, and Phase 1 (MPC) should lift all models by
converting short-lead skill into recourse.

### Lesson

A forecast is an input to a decision, not an end product. The two models
bracket the failure modes: `lightgbm_rich` ranks well but lies about
magnitude; `naive` gets magnitudes right on average but ranks poorly. The
money is in having both, and neither pinball-median training nor MAE
selection optimises for that combination.

---

## Entry 016: MPC clears 0.50 — recourse pays exactly where short-lead skill exists

**Date:** 2026-07-12
**Component:** `src/grian/sim/mpc.py` — W1.3 results
**Severity:** Model quality — campaign milestone

### What happened

The receding-horizon MPC executor (re-solve every 30 min from true SOC,
reforecast hourly from all observed data, telescoped 24-h horizon):

| Model | Executor | Val capture | Test capture | Val→Test Spearman |
|---|---|---|---|---|
| lightgbm_rich | open-loop | 0.389 | 0.301 | 0.632 / 0.688 |
| lightgbm_rich | **mpc-30m** | **0.536** | **0.509** | 0.839 / 0.826 |
| naive_similar_day | open-loop | 0.451 | 0.347 | 0.566 / 0.575 |
| naive_similar_day | mpc-30m | 0.382 | 0.365 | 0.566 / 0.575 |

lightgbm_rich gains ~15–21 points from MPC and crosses the 0.50 target on
both windows. The naive model **loses** 7 points on validation from the
same executor (and gains ~2 on test — nothing like lightgbm's jump).

### Why the asymmetry

MPC converts the *slope* of the skill-vs-lead-time curve into revenue.
lightgbm_rich's most important features are rolling 1h/6h/24h price
statistics and momentum: at 1-hour lead its forecast is far sharper than
at 18-hour lead, so every re-solve acts on genuinely better information —
intraday Spearman jumps from 0.63 to 0.84. The naive forecast is last
week's profile at every lead time: the skill curve is flat, so hourly
re-solving just makes the LP chase spike timings that jitter with each
refresh, paying the 15% round-trip efficiency toll on churned decisions
and spending cycle budget early.

**MPC is not a free upgrade — it amplifies a forecaster's short-lead
information advantage, including an advantage of zero into a deficit.**

### What did not need to change

No model was retrained; no new data was added. The entire gain comes from
the executor asking the same model fresh questions and re-planning from
the true state. This is the cheapest 15 points on the board, and it was
gated entirely on Phase-0/W1.1 plumbing (honest physics, predict-from-now).

### Remaining gap and next lever

At 0.51 test, the gap to the 0.65 target is ~$1.1M/window. The regret is
concentrated where forecast *magnitude* fails (Entry 015): pinball-median
forecasts still understate spikes. `lightgbm_qmean` (quantile → dollar-mean
integration, W2.1) is running next; per plan §3.6 it targets +3–8 points.

### Lesson

Decompose "forecast quality" into level skill, rank skill, and skill decay
with lead time. Each monetises through a different mechanism: level through
cycle sizing, rank through cycle placement, and skill decay only through an
executor with recourse. A dispatch stack leaves money on the table when any
one of the three is strong but the mechanism that monetises it is missing.

---

## Entry 017: Mean-space forecasting helps open-loop, not under MPC

**Date:** 2026-07-12
**Component:** `lightgbm_qmean` (W2.1) results
**Severity:** Model quality — technique partially rejected

### What happened

`lightgbm_qmean` (quantiles {0.05, 0.5, 0.9, 0.98} per step, inverted to
dollars, sorted, integrated with midpoint weights to a conditional mean):

| Executor | Model | Val | Test |
|---|---|---|---|
| open-loop | lightgbm_rich (median) | 0.389 | 0.301 |
| open-loop | lightgbm_qmean (mean) | 0.407 | **0.443** |
| mpc-30m | lightgbm_rich (median) | **0.536** | 0.509 |
| mpc-30m | lightgbm_qmean (mean) | 0.520 | — (rejected on val) |

Open-loop, the dollar-mean forecast adds +1.8 points on validation and a
striking +14.2 on test. Under MPC it *loses* 1.6 points on validation, so
per the validation-first rule it was not run on test and the campaign
champion remains lightgbm_rich + MPC.

### Why the flip

The mean-vs-median gap matters in proportion to how wide and skewed the
predictive distribution is. At 12–24 h lead (open-loop), spike probability
mass is diffuse: the median says "calm", the mean correctly says "hold
energy". At 1–4 h lead (MPC reforecasts hourly), the distribution has
collapsed around the outcome — median ≈ mean — so the integration adds
little, and what remains is a fair fight between calibration error in the
tail quantiles and estimation noise.

Two confounders to note honestly: qmean uses 150 trees/booster (vs 300 for
rich) to keep fit time bounded, and its 0.98 tail quantile is a crude tail
model. A worker could re-run at matched capacity before fully closing the
book (ticket note added to plan §5 W2.1).

### Lesson

Techniques interact with the executor. The value of probabilistic
calibration is largest exactly where uncertainty is largest — long leads,
open-loop, sparse recourse. Measure a forecasting improvement under the
executor you will actually deploy, not the one that's easiest to run:
open-loop evidence overstated qmean's value for the MPC stack by an order
of magnitude.

---

## Entry 018: Frequency ablations — freshness pays, raw persistence backfires

**Date:** 2026-07-12
**Component:** `scripts/run_mpc_ablations.py` — W1.3/W1.4 grid, validation
**Severity:** Model quality — one accepted knob, one instructive failure

### What happened

Grid over MPC frequencies and the persistence blend, lightgbm_rich,
validation window (champion baseline r6_f12: 0.536):

| Variant | resolve | reforecast | persistence | Capture |
|---|---|---|---|---|
| r6_f6 | 30 min | **30 min** | off | **0.546** |
| r1_f6 | 5 min | 30 min | off | 0.545 |
| r6_f12_p6 | 30 min | 60 min | τ=6, ungated | 0.435 |
| r1_f6_p6 | 5 min | 30 min | τ=6, ungated | 0.428 |

### Finding 1: information, not recourse frequency

Doubling reforecast frequency (60→30 min) adds +1.0 point; increasing
LP re-solve frequency 6× on top of that adds nothing (0.545 vs 0.546).
Without new information, re-solving reproduces the same plan. The MPC
gain chain is: fresh features → fresh forecast → different plan. Spending
compute on solves instead of forecasts is pushing on a rope.

### Finding 2: the persistence blend loses 10 points

Blending the first hour of every plan toward the last traded price —
motivated by the Dec 8 event-day autopsy, where the model forecast $25
during a $15k spike — *destroys* a tenth of the oracle value. The
mechanism inverts: 5-minute NEM prices are violently mean-reverting in
normal operation (single-interval dips to −$900 and blips to +$300 that
vanish immediately). An ungated blend makes the LP chase every transient:
charging into dips that revert before the energy arrives, discharging
into blips that die, burning cycle budget and efficiency all day. The
rare spike-onset gain is overwhelmed by daily noise-chasing losses.

The Dec 8 intuition wasn't wrong — reacting to a real event onset is
worth ~$430K on that day alone — but "react to the last price" must be
**conditional on the price being extreme**.

### Follow-up: gating recovers most of the loss but does not beat clean MPC

A spike-latch gate (blend only when the last price exceeds a threshold)
was added and re-run on validation:

| Variant | Capture (val) |
|---|---|
| no persistence (accepted r6_f6) | 0.546 |
| persistence, ungated | 0.435 |
| persistence, gate $300 | 0.523 |

Gating at $300 recovers 8.8 of the ~11 lost points, confirming the
mechanism diagnosis — the damage was body-of-distribution noise-chasing,
not the tail reaction. But even gated it trails clean MPC (0.523 <
0.546), because the 30-minute re-forecast already incorporates the last
observed prices through the model's recency features, so an explicit
last-price blend is largely redundant and its residual noise-chasing is
pure cost. **Persistence blending is rejected for the stack.** A stricter
$1000 gate was queued to test whether a pure extreme-event latch clears
the bar; result recorded in the campaign results table.

### Lesson

The same signal can be the best or the worst input depending on the
regime. Last-price persistence is near-optimal during sustained extreme
events and anti-informative during normal mean-reverting chop. Averaging
a regime-dependent signal into all regimes converts a targeted edge into
a broad tax. When a mechanism is justified by tail events, gate it to
fire only in the tail — but first check the model isn't already using
that signal, or the gate buys nothing.

---

## Entry 019: Re-forecast freshness — small on validation, decisive on test

**Date:** 2026-07-12
**Component:** MPC `reforecast_every`; W1.3 follow-up
**Severity:** Model quality — new champion

### What happened

Tightening the MPC re-forecast interval from 60 to 30 minutes
(`reforecast_every` 12 → 6, re-solve held at 30 min):

| Window | 60-min reforecast | 30-min reforecast | Delta |
|---|---|---|---|
| validation | 0.536 | 0.546 | +1.0 pt |
| **test** | **0.508** | **0.562** | **+5.4 pt** |

The change that looked marginal on validation is the single largest
executor-tuning gain on the test window — larger than its validation
signal predicted by 5×. New champion: **lightgbm_rich + MPC (30-min
reforecast, 30-min re-solve) = 0.546 val / 0.562 test**, strictly
better than the prior champion on both windows.

### Why the asymmetry between windows

Freshness value is not about how *concentrated* revenue is (validation is
more concentrated: top-10-day share 51.7 % vs test's 30.7 %) — it is
about how *fast* the value-carrying events move. Validation's spikes are
disproportionately evening-ramp/duck-curve events that are largely
resolved 60 minutes ahead, so a 60-min-stale forecast already sees them.
The test window's regret is dominated by fast midday constraint/outage
events (Entry 016 hourly gap; the Dec-8 $16.5k spike), where the price
moves within a couple of intervals and 30 minutes of forecast staleness
means missing the onset entirely. Halving staleness converts directly
into catching those onsets.

### The methodological trap this exposes

Had we accepted the +1pt validation signal as "marginal, skip it," we
would have left 5.4 points ($416k/window) on the test window. A technique's
value must be read on the regime that resembles deployment, and its
*mechanism* reasoned about explicitly — "fresher forecasts help fast
events, and our regret is on fast events" would have predicted the test
result that the validation number alone hid. This is the same lesson as
Entry 017 (measure under the deployment executor) applied to a tuning
knob rather than a model.

### Resolved: freshness has an optimum — 5-min re-forecast is *worse*

The open question was whether re-forecasting every 5 minutes (rather than
30) helps further on the fastest events. Measured on validation:

| Re-forecast interval | Capture (val) | Spearman |
|---|---|---|
| 60 min | 0.536 | 0.839 |
| **30 min (champion)** | **0.546** | 0.869 |
| 5 min | 0.534 | **0.895** |

5-minute re-forecasting is **worse** than 30-minute (0.534 vs 0.546),
even though it produces the **best** intraday rank skill of any variant
(Spearman 0.895). Freshness is not monotone — there is an optimum near
30 minutes.

**Why over-freshness hurts.** The model refits only weekly; between
refits, re-forecasting every 5 minutes from marginally-shifted rolling
features produces a *jittery* forecast whose fine structure changes each
step. Because the MPC re-solves against it, the LP keeps revising its
charge/discharge intentions, and the executor — which commits the next
block before the plan changes again — ends up dithering: beginning to
charge, then reversing, paying the 15% round-trip efficiency toll on
churned decisions. This is the **same churn mechanism as the persistence
blend (Entry 018)**, milder but the same shape: over-reacting to a
fresh-but-noisy signal converts responsiveness into cost. The tell is
the same too — the highest Spearman (best ranking) coincides with lower
capture, exactly as it did for the median-vs-mean case (Entry 015):
rank skill that the executor cannot bank is not revenue.

So the champion cadence (30-min re-forecast, 30-min re-solve) is a
genuine optimum: fresh enough to catch fast events that a 60-min forecast
misses, stable enough that the plan does not dither. It is also the
cheaper choice — 5-min re-forecasting is 6× the forecast cost (the MPC
bottleneck) for a *worse* result.

### Lesson

The size of a tuning effect on one window is not its size on another when
the windows differ in the *character* of their extreme events (60→30 min:
+1 pt val, +5.4 pt test). And freshness, like reactivity, has an interior
optimum: past a point, a more responsive forecast just makes the executor
churn through its efficiency losses. Whenever a knob raises rank skill
(Spearman) but not capture, suspect churn — the same signature recurs
across median-vs-mean (015), persistence (018), and re-forecast frequency
(019). Reason about the mechanism, and remember the executor has to *bank*
skill for it to count.

---

## Entry 020: Quantile-mean forecasting closed out — capacity-controlled, still loses under MPC

**Date:** 2026-07-12
**Component:** `lightgbm_qmean`; round-2 comparison (`run_model_round2.py`)
**Severity:** Model quality — technique definitively rejected for the MPC stack

### What happened

Entry 017 left a caveat: `lightgbm_qmean` (dollar-space mean) lost to
`lightgbm_rich` (median) under MPC, but used 150 trees/booster vs the
champion's 300, so the loss might have been a capacity artifact. Round 2
controls for that — all variants at 150 trees, champion executor (r6_f6),
validation:

| Variant | Point forecast | Capture (val) | Spearman |
|---|---|---|---|
| rich150 | median of asinh-price | **0.548** | 0.866 |
| qmean150 | dollar-space mean (quantile integral) | 0.540 | 0.860 |
| qhybrid150 | median < 1h lead, mean beyond | 0.522 | 0.864 |

For reference the 300-tree champion is 0.546 — so `rich` at 150 trees
(0.548) equals it: **capacity above 150 trees buys nothing** (a useful
efficiency finding — half the trees, ~2× faster, same capture).

### Why qmean loses even at matched capacity

The capacity caveat is disproved: at 150 trees, median (0.548) still beats
mean (0.540). The dollar-space mean is the *right* target for the LP in
principle (§3.2), and it wins handily **open-loop** (+14 pts on test,
Entry 017) — but under MPC the hourly re-forecast collapses the predictive
distribution to where median ≈ mean, and the mean estimate then carries the
extra variance of four quantile heads (including a crude 0.98 tail) with no
compensating signal. Net: slightly worse.

### Why the lead-dependent hybrid is *worst*

The hybrid switches the point forecast from median (steps < 12) to
integrated mean (steps ≥ 12). That switch is a **discontinuity in the
forecast profile** at the 1-hour mark: median and mean differ by a few
dollars under skew, so the horizon has an artificial ~$5–15 step at step 12.
The LP is a price-*difference* engine — it reads that manufactured kink as a
tiny arbitrage and schedules against it. Blending two estimators along a
horizon that an optimizer will differentiate is worse than either alone.

### Resolution

Quantile-mean forecasting is **rejected for the MPC stack** and removed from
the champion path. It remains the best *open-loop* model and a good teaching
example of "the right target for the decision is executor-dependent." The
champion stays `lightgbm_rich` (median) + MPC.

### Lesson

Two lessons. (1) Always capacity-control a model comparison before concluding
the architecture matters — but here the control *confirmed* the effect rather
than dissolving it, which is the stronger outcome. (2) Do not stitch two
forecasters together along an axis a downstream optimizer differentiates; the
seam becomes a phantom signal. A blend must be smooth in whatever the decision
layer is sensitive to — here, price *differences* across the horizon.

---

---

## Entry 021: Common-window evaluation — MPC underperforms open-loop on 2025–26 data

The val/test paradigm scored each configuration on a *different* window, which
made captures incomparable and the naming confusing. Replaced it with a single
shared-window evaluation (`scripts/run_common_eval.py`): every config is scored
by walk-forward over one identical span against one oracle, and the HTML
dashboard recomputes capture over any sub-window as `Σ model daily ÷ Σ oracle
daily`. Data was extended via NEMOSIS to 2023-01 → 2026-06-29. Default span: the
most recent 12 months (2025-07 → 2026-06), oracle revenue $29.16M, top-10-day
share 44%.

### Results (12-month window, monthly refit)

| Model | open-loop | 30-min MPC |
|---|---|---|
| autoregression | **0.440** | −0.070 |
| naive_similar_day | 0.414 | 0.266 |
| lightgbm_rich | 0.389 | 0.351 |

Three surprises, all opposite to the 2023 campaign (Entries 013–020):

1. **MPC lost to open-loop for every model** — AR under MPC went *negative*
   (−$2.0M). Re-solving every 30 min on a mediocre forecast whipsaws the battery.
2. **Linear AR beat lightgbm_rich on capture** (0.440 vs 0.389) despite far worse
   forecast accuracy (MAE skill 33% vs 49%). Capture rewards *timing/rank*, not
   absolute error — the decision-focused gap made explicit.
3. **lightgbm_rich had the best MAE skill and Sharpe but not the best capture.**

### Caveat (why this is not yet a clean conclusion)

Two things changed at once vs the campaign: the **window** (2025–26 vs 2023) and
**refit cadence** (`refit_days=28` monthly, chosen for tractability — weekly
refits over a year run ~4× longer). The monthly cadence likely disadvantages both
lightgbm and the MPC executor (which leans on fresh forecasts). The comparison is
*internally* fair (all configs identical treatment), but the divergence from the
campaign is unattributed until a weekly-refit rerun isolates cadence from regime.

### Lesson

Accuracy ≠ trading value, and an executor that wins on one window/cadence can
lose on another. Bake the test window and refit cadence into the comparison, not
into folklore — hence the interactive window slider.

---

## Entry 022: Synthetic validation — executors are correct; MPC amplifies forecast noise

To decide whether MPC's real-world underperformance (Entry 021) was a bug or a
property, we fed the *real* open-loop and MPC executors a controllable forecast
on a synthetic price series with purely intra-day arbitrage (daily cheap window +
evening spike, no cross-day value). Forecast = actual future + Gaussian noise of
increasing σ. Script: `scripts/validate_executors_synthetic.py`.

| noise σ ($/MWh) | open-loop capture | MPC 30-min capture |
|---|---|---|
| 0 (perfect) | **1.000** | **0.999** |
| 5 | 0.960 | 0.959 |
| 20 | 0.886 | 0.835 |
| 50 | 0.838 | 0.747 |
| 100 | 0.666 | **0.366** |

### Conclusions

1. **Both executors are correct.** A perfect forecast captures ~100% of the
   oracle under both open-loop and MPC — the LP, SOC accounting, dispatch, and
   both executor loops are sound. The poor real-world MPC numbers are not a bug.
2. **MPC amplifies forecast noise.** It degrades *faster* than open-loop as σ
   grows (at σ=100, MPC keeps only 0.366 vs open-loop 0.666). Re-solving every
   30 min on a noisy signal whipsaws the battery — buy, reverse, sell low. This
   is exactly why MPC helped on the cleaner 2023 regime but hurt on the noisier
   2026 data.

### Lesson

MPC is a *forecast-quality amplifier*, not a free win. It pays off only when the
marginal re-forecast carries real signal; on noisy forecasts, commitment
(open-loop) is safer. Pick the executor to match the forecast quality, and prefer
open-loop when the model is weak.

---

## Entry 023: Harness validation, rolling-window training, and weekly-refit does not rescue MPC

### Harness faithfully reproduces the campaign (exact)

Before trusting the common-window harness on new data, we reran it on the *old*
2023 test window (train 2023-01→09, test 2023-10→2024-01, weekly refit) and
compared to the archived campaign trials:

| Quantity | Archived campaign | Common-eval harness |
|---|---|---|
| 2023 test oracle revenue | $7,707,632 | **$7,707,632** (exact) |
| naive open-loop capture | 0.3473 | **0.3473** (exact, $2,676,924) |
| autoregression open-loop | 0.4016 | **0.402** |

Exact reproduction of the oracle and open-loop capture confirms the LP, dispatch,
and capture computation are faithful — the new-window results are real, not a
harness artifact.

### Weekly refit does not rescue MPC (2026 window)

Re-running the 2026 window with weekly (7-day) refit instead of monthly (28-day),
under the *expanding* training window:

| Config | monthly refit | weekly refit |
|---|---|---|
| naive MPC | 0.266 | 0.266 (naive doesn't refit — unchanged) |
| autoregression MPC | −0.070 | −0.022 (better, still loses money) |

Weekly refit helps MPC only marginally and it still bleeds money on a weak
forecast. Combined with Entry 022 (synthetic: MPC amplifies forecast noise), the
MPC underperformance on 2026 is a forecast-quality/regime effect, not a refit
cadence artifact.

### Rolling training window adopted (18 months)

Switched the runner from an *expanding* window (train on all history to each
trading day) to a configurable **rolling window** (`train_lookback_days`, default
548 ≈ 18 months). Rationale:

- Electricity prices are seasonal; <12 months never sees the target season.
- 12 months = one instance of each season (thin for rare summer spikes);
  18 months ≈ 1.5× coverage without dragging in stale regime data (the SA1 grid
  changes fast year-on-year with new solar/wind/batteries).
- Bonus: refit cost becomes constant rather than growing, which makes weekly
  refit over a year tractable (the reason monthly was used before).

An overnight weekly + 18-month-rolling grid (4 models × 2 executors) is running to
produce the canonical 2026 numbers under this cadence.

---

## Entry 024: "Observe the present" — MPC was forecasting the current price

Following the buy-high/sell-low finding (Entry 021/022), inspection of `mpc.py`
showed the MPC LP was fed the *forecast* for the entire horizon **including the
current interval** (`plan_prices = forecast_dollars[shift:]`), while (inconsistently)
re-solving from the *true* current SOC. So it decided the immediate charge/discharge
on an **estimate of the present price** — which a real battery never does (the
current dispatch price is observed at decision time).

**Fix:** pin `plan_prices[0]` to the actual observed price; only genuinely-future
steps stay forecast. Gated by `mpc.observe_present` (default on; off reproduces the
legacy behaviour). Also added a realistic `mpc5` executor (5-min resolve).

**Synthetic sanity (capture vs forecast-noise σ), old → fixed:**

| σ ($/MWh) | open-loop | MPC old | MPC fixed |
|---|---|---|---|
| 0 | 1.000 | 0.999 | 0.999 |
| 20 | 0.886 | 0.835 | 0.841 |
| 50 | 0.838 | 0.747 | 0.761 |
| 100 | 0.666 | 0.366 | **0.477** |

The fix improves MPC at every noise level (dramatically at σ=100). The residual gap
vs open-loop is the *future*-forecast error — committing 6 steps ahead on forecast
under 30-min resolve; `mpc5` (re-solve every interval) should shrink it further.

**Lesson:** an MPC must observe, not forecast, the present. Feeding it a nowcast of
the current price is a fidelity bug that manifests as immediate mistrades.

---

## Entry 025: Quantile-gate scrambles interval rankings; weather plumbing (zip + TZ)

**Quantile-gated dispatch failed — instructively.** On real data `lightgbm_qmean`
under the direction-pessimistic gate captured **0.032** vs **0.362** for the same
model under point MPC. Diagnosis (dispatch_efficiency): it did *not* stop trading
(134,633 vs 138,109 MWh discharged) — it **mistimed**. Realised avg charge/discharge
went from $7.6/$83.5 (spread +$76) under point MPC to **$43.5/$57.3 (spread +$13.9)**
under the gate.

Root cause: the gate replaces each interval's price with a low/high quantile keyed
on whether it's above/below the horizon median. When forecast **uncertainty varies
across intervals**, that compression is uneven and **reorders which intervals look
cheap vs dear**, so the LP charges/discharges the wrong ones. The heuristic is
wrong; a correct robust dispatch must preserve each scenario's internal ranking
(→ per-scenario LP / CVaR, not per-interval quantile substitution).

**Weather plumbing bugs found and fixed:**
1. CDS returns multi-variable ERA5 as a **zip of per-stream netcdfs**, not a plain
   `.nc` — xarray couldn't open it. Fix: unzip and `xr.merge` the inner files.
2. **Timezone**: ERA5 is UTC; NEM/AEMO data is AEST (UTC+10). Unshifted, solar
   peaked at "midnight". Fix: shift ERA5 +10h. Verified: ssrd now peaks at hour 13.

**Lesson:** validate a probabilistic-dispatch heuristic against the *realised*
charge/discharge prices, not just capture — and never distort cross-interval price
rankings the LP depends on.

## Entry 026: Oracle "negative day" was a false alarm — the invariant is per closed cycle

`compute_oracle` asserted every *calendar day*'s revenue ≥ 0 ("doing nothing earns
zero, so perfect foresight can't lose money" — trap T7). This tripped on the
Jan–Mar 2026 sub-window. Diagnosis: on **2026-03-19** the oracle charged 687.5 MWh
but discharged only 400 MWh, ending the day at **SOC 200 MWh** — it was a *net
buyer*, banking cheap overnight energy to discharge into the next morning's spike.

Root cause: the oracle solves in **7-day blocks with SOC pinned to 0 only at block
boundaries**, so it can legitimately carry charge across midnight *inside* a block.
A single day can then be net-negative while the block is positive. The day-level
assert encodes the wrong granularity. Fix: assert non-negativity over each **closed
cycle** — split wherever SOC returns to empty; each such segment is
feasible-to-skip, so at the optimum each must be ≥ 0. The full-window oracle was
always clean (it never needed the false assert); only off-phase sub-windows tripped
it, which is exactly what the lightweight test bed needs.

**Lesson:** a "can't lose money" invariant only holds at the granularity where the
state (SOC) starts and ends neutral. Assert at that boundary (the block/cycle), not
a convenient calendar unit. New tests: a synthetic cheap-day → spike-day case proves
a net-negative *day* is accepted while the closed cycle stays positive.

## Entry 027: Probabilistic dispatch done right — decision-space fusion + a fan-cache test bed

After the quantile-gate failure (Entry 025), the correct approach fuses actions in
**decision space**, not price space. Each quantile *path* is solved by its own LP
(so every scenario keeps its internal cross-interval ranking), then the per-scenario
actions are combined:
- **robust** — min charge / min discharge (trade only where all scenarios agree);
- **EV** — probability-weighted expected net action (weights from the quantile
  levels via trapezoidal mass);
- **CVaR(λ)** — a blend from EV (λ=0, risk-neutral) to robust (λ=1, worst-case),
  giving a single tunable risk knob.
A convex blend of feasible battery plans stays feasible, and netting charge/discharge
avoids simultaneous both-legs.

**Fan-cache test bed.** Forecasters and executors are disentangled: the fan is a pure
function of (model, data, reforecast cadence, horizon) — no dependence on dispatch
mode. So `simulate_region_mpc` was extended to **record** fans during a run and
**replay** from a cached fan (skipping fit/predict), keeping one dispatch code path.
A parity test confirms a replay reproduces a live run **to the cent**. This lets the
whole dispatch grid (robust/EV/CVaR-λ) be swept over one frozen forecast in seconds
instead of retraining per executor.

**Lesson:** the LP relies on *relative* prices — respect each scenario's ranking and
combine *decisions*, not prices. And cache the expensive, executor-independent thing
(the forecast) so dispatch experiments are cheap.

## Entry 028: 30-min pivot — the repeated-training trap and an operational cache bug

**Executive pivot for the fast-iteration sprint:** rolled the eval from 5-min to
**30-min** resolution (mean of each six 5-min prices = the NEM 30-min settlement
price). LP horizon 288→48 (superlinear solve speedup), 6× fewer intervals/solves.
All configs run at 30-min so they stay comparable to each other; NEM settled at
30-min until Oct 2021, so it is defensible. Naive openloop over a month dropped from
hours (5-min) to **0.4s**.

**But the bottleneck moved to training.** At 30-min the LP is cheap, so the ~52
weekly refits dominate — and `run_common_eval` **retrains the quantile fan
separately for each executor** (openloop, mpc30, scenario, ev, cvar), i.e. up to 5×
the same fit. Fix: split the matrix — point configs via `run_common_eval`, and the
whole probabilistic family via the **fan-cache test bed** (one fit per model feeds
every dispatch variant). Also dropped weekly→**monthly** refit for a further ~4×.

**Operational bug (logged as a lesson):** a smoke test wrote a *one-month* naive
ledger into the real `outputs/trials_30min` base; the full run then **skipped**
recomputing it (it keys off ledger existence) and scored naive on January only
(0.652 vs the correct full-window 0.494). Fix: delete + recompute. **Lesson:** never
smoke-test into a production output dir; a resume-by-existence runner will silently
serve stale partial results.

## Entry 029: True mean-CVaR dispatch LP + linear (LEAR) family + calendar encoding

**Executor drill (no retraining — pure replay over cached fans):**
- **`mpc_meancvar` — the correct risk-aware dispatch.** Replaces the earlier
  decision-space λ-blend (`scenario_cvar`) with a single joint
  **Rockafellar-Uryasev LP** that optimises *one* here-and-now plan across all
  quantile scenarios: maximise ``(1-λ)·E[R] + λ·CVaR_α(R)`` via auxiliary VaR +
  per-scenario shortfall variables. `λ` = risk weight (0 = expected value,
  1 = worst-tail), `α` = tail level. Tests prove λ=0 reproduces the
  expected-value (mean-price) plan and λ=1 declines a trade that loses in the
  tail (worst-scenario revenue ≥ the EV plan's). `lp.solve_mean_cvar_lp`.
- Because forecasters and executors are disentangled, the whole executor grid —
  point / robust / EV / CVaR-blend(λ) / mean-CVaR-LP(λ,α) — is swept as a
  **replay over the cached fan** (seconds–minutes, zero retraining).

**LEAR family (canonical EPF benchmark) added** — Lasso / Ridge / ElasticNet and
a linear *quantile* fan (`lear_qmean`, QuantileRegressor per step×quantile) on the
**same rich features as lightgbm_rich**, for a clean linear-vs-GBM contrast on
identical inputs, including under probabilistic dispatch.

**Calendar encoding fixed for linear models.** Ordinal integers (`month`=12 read
as 12×`month`=1) are wrong for a linear fit. All regression-on-dates models
(LEAR family + the autoregression baseline) now default to **one-hot**, with a
**Fourier** (cyclic sin/cos harmonics: hour K=3, dow K=1, month K=2 → 12 smooth
features) and legacy ordinal option for ablation. Trees keep ordinal (correct for
threshold splits). **Normalisation audit:** every scale-sensitive model already
standardises (LEAR pipelines, MLP X&y, LSTM y); LightGBM needs none; OLS is
scale-equivariant — no hidden bug.

## Entry 030: "Controllers stink" — NOT a dispatch bug; it's forecast spike-timing

**Symptom.** On the 30-min common window, no model beats the dumb baselines and
MPC is *worse* than open-loop: open-loop capture ≈ 0.49–0.53 (AR best at 0.531,
naive 0.494, qmean 0.509), but MPC ("mpc30") ≈ 0.25. Better MAE (lightgbm) does
**not** buy more capture than AR. Looked like a bug.

**Hypotheses tested and rejected (measured over the cached qmean fan, full year):**
1. **Telescoping flattens peaks.** `_telescope` averaged the horizon beyond 6h
   into 3h blocks; at 30-min that muddies spikes. Fixed to solve the full 48-step
   horizon (telescope only when horizon > 96). Capture 0.254 → 0.253 — **no
   effect.** Kept anyway (correct, removes a confound).
2. **MPC used the median, open-loop the mean.** `_forecast_from_fan` returned the
   q0.5 path; for a linear objective the correct point is the quantile-integrated
   **mean** (spike-aware). Fixed. Capture 0.254 → 0.259 at resolve=1 — **no
   effect on its own.** Kept (genuine correctness fix).
3. **Reforecast cadence.** Sweep (with fixes 1–2, resolve every 3h): reforecast
   every 30min → 0.364, every 6h → 0.433, once/day → 0.486, open-loop → 0.51.
   **Less reforecasting is monotonically better** — the classic "MPC amplifies
   forecast noise". But slowing it is a band-aid: it removes MPC's reason to
   exist (reacting to unforecast spikes). Spike-latch (persistence blend) to make
   an *observed* spike extrapolate: negligible (0.364 → 0.399).

**Definitive test — perfect foresight.** Fed the *actual* future prices as the
forecast (identity transform) through the real 30-min pipeline:
**open-loop capture 0.980, MPC 1.002.** The dispatch/data/scoring path is
therefore **correct** — no off-by-one, no transform error, no scoreboard bug.
The ~0.5 ceiling is a *forecast-skill* ceiling, not a controller bug.

**Root cause.** The forecasts lack interval-level **spike timing**, and spikes
are where the oracle earns ~half its money. Evidence: qmean's forecast tops out
at **$1,913** while actuals reach **$20,300** — it never predicts the big spikes;
and the MPC "forecast" correlates best with actual at **lag +2** (≈1h stale), i.e.
it is a *lagging persistence copy* — it looks high-fidelity on the dashboard
precisely because it replays recent prices, but it cannot *anticipate*. So
frequent reforecasting churns on a lagging signal (worse than open-loop), and
MAE gains land on the flat body, not the paying tail. This is the
accuracy-vs-decision-value gap (curriculum Ch 16), not a code defect.

**Consequences / next steps.** (a) The path to beating the baseline is a forecast
that *anticipates* spikes — scarcity/precursor features (built, but
`include_scarcity=False` in these runs) and decision-focused training — not more
control tuning. (b) MPC only helps once the reforecast adds forward information.
(c) Minor: perfect MPC scored **1.002 > 1** because the oracle is solved in 7-day
blocks with SOC pinned to 0 (Entry 026), making it ~0.2% conservative vs a
rolling perfect-foresight controller; the `capture <= 1` assert is thus slightly
too strict for perfect-forecast diagnostics (harmless for real models).

## Entry 031: Scarcity features alone hurt capture — the objective, not the features, is the lever

Direct test of Entry 030 step 1. Ran `lightgbm_rich_scarcity` (rich features +
demand-stress / ramp / volatility-clustering precursors, `include_scarcity=True`)
open-loop on the 30-min common window (2025-07 → 2026-06, 364 days).

**Result: open-loop capture 0.415** — *below* the open-loop autoregression bar
(0.531) **and below plain `lightgbm_rich` open-loop (0.489)**. Adding the
spike-precursor features made capture **worse**.

**Reading.** The precursors carry real signal, but under a median/MAE objective
the extra columns mostly add variance the tree spends on the flat body; they do
not buy spike *timing*, because an accuracy-optimal fit is still rewarded for
smoothing the 5% of intervals that pay. This is the sharpest evidence yet that
the ceiling is set by the **training objective**, not the feature set — you
cannot feature-engineer your way past a loss that doesn't value the tail.

**Consequence → decision-focused training (Entry 030 step 2), now built.** New
`models._decision_weights` computes per-sample fit weights that up-weight
high-price intervals (bounded log-dollar "magnitude" ramp, or a flat "quantile"
boost above a price percentile), mean-normalised to 1 so the effective learning
rate is unchanged. Weights are computed in **dollar** space (target inverted out
of its asinh transform first) so a config is comparable across transforms. Wired
into `_lgbm_rich_fit` via `sample_weight`, gated on
`model_params.sample_weighting` (off by default — existing runs untouched).
Registered as `lightgbm_rich_dfl` (scarcity features + magnitude weighting,
strength 1.0). Unit tests pin the mechanism (monotone in price, mean-1, tail-only
for the quantile scheme, no-op on flat prices) plus end-to-end fit/predict:
`tests/test_sim_decision_weights.py`.

**Result: `lightgbm_rich_dfl` open-loop capture 0.452** (364 days). Better than
scarcity-alone (0.415) but still **below plain `lightgbm_rich` (0.489) and the
autoregression bar (0.531)**. So magnitude weighting at strength 1.0 recovers
part of what the scarcity features cost, but does not clear the bar.

**Reading.** A median/quantile-0.5 GBM structurally cannot emit $20k spikes
(Entry 030: forecast max ≈ $1.9k vs actual $20.3k); up-weighting high-price rows
raises the high end a little and adds some positive bias, but does not manufacture
interval-level spike *anticipation*. Sample weighting nudges the objective; it
does not change what a point regressor can represent. Open questions worth a
sharp, isolated test (not a broad sweep): (a) weighting on **plain rich** (no
scarcity, the stronger 0.489 base) at higher strength; (b) the **quantile**
scheme (flat tail boost); (c) the more likely real fix — stop asking a point GBM
to predict spike magnitude and instead model the tail explicitly (upper-quantile
or scarcity-probability head) and let dispatch act on P(spike), i.e. couple
decision-focused training with the probabilistic path (Entry 027) rather than a
point forecast.

## Entry 032: Why LEAR is famous yet loses here — MAE and capture are inversely ranked

Stepping back from fancier methods to make the *simple* ones work. Ranked every
open-loop trial on the 30-min common window by **both** capture and MAE:

| model (open-loop) | capture | MAE ($) |
|---|---|---|
| **autoregression** (OLS, 3 lags) | **0.543** | 83.2 |
| elasticnet | 0.518 | 77.0 |
| lear (Lasso, full features) | 0.516 | **76.8** |
| ridge | 0.513 | 77.2 |
| lightgbm_qmean | 0.509 | 62.5 |
| lightgbm_qmean_weather | 0.501 | **59.9** (best MAE) |
| lightgbm_rich | 0.489 | 63.2 |

**The finding: MAE and capture are inversely ranked among the serious models.**
The best-MAE models (the GBMs, ~60) have middling capture; the best-capture model
(AR) has the *worst* MAE of the group (83). Lower average error actively trades
away dispatch value.

**This answers "why is LEAR popular if it can't do much here?"** LEAR is doing
exactly what made it the canonical EPF benchmark (Lago et al. 2021): it has
excellent MAE (76.8, beating AR's 83.2). The literature scores day-ahead
forecasts on **MAE / RMSE / pinball**, and on that scoreboard LEAR reliably beats
naive and even neural nets, cheaply and robustly. It is popular because it wins
the *accuracy* game. This repo scores **dispatch capture**, which pays for spike
*timing*, not average closeness — a different objective — so the accuracy ranking
and the value ranking come apart (the accuracy-vs-decision-value gap, Ch 16).

**Why AR (dumb) beats LEAR (sophisticated) on capture.** AR and LEAR share the
same 3 price lags (`price_lags` default `[1d, 2d, 7d]`) and one-hot calendar; LEAR
adds the rolling-mean / momentum / intraday-profile **smoothers** and swaps OLS
for Lasso. Both additions pull the forecast toward the recent mean — great for
MAE, fatal for the peak *shape* that capture rewards. The calendar-encoding
ablation corroborates: `lear` (one-hot) 0.516 > `lear_ordinal` 0.471 >
`lear_fourier` 0.434 — smoothing the calendar too (Fourier) hurts most.

**Make-the-simple-model-work experiment (built).** Added `feature_set="lean"` to
`build_features`: a shape-preserving basis of *raw* price lags (`lean_lags`: recent
intervals, last few hours, same time on prior days ±1, and the weekly lag) plus
calendar — **no smoothers**. Plumbed through the linear fit/predict/save/load
(stored in state so predict rebuilds the identical columns). Two registry entries
decompose AR-vs-LEAR into its two channels:
* `lear_lean` — Lasso on the lean basis (isolates the **smoother** effect: does
  dropping the mean-reverting features recover capture?);
* `ols_lean` — OLS on the lean basis (a **better-specified AR**: does a richer raw
  lag set beat AR's 3 lags?).
Tests: `tests/test_sim_features.py::TestLeanFeatureSet` (no smoothers, richer lag
basis, leakage-free).

**Result v1 (recency-inclusive lean basis) — hypothesis REJECTED.** First
`lean_lags` = recent intervals + hourly + daily lags. Open-loop capture:
`lear_lean` **0.446** (worse than `lear` 0.516), `ols_lean` **0.492** (worse than
AR 0.543). Dropping the smoothers *hurt* — the handoff's "smoothers flatten the
peaks" story is wrong, at least on its own.

**Diagnosis — recency anchoring, not smoothing.** Inspected the `ols_lean`
per-step coefficients (`|coef|` share by lag). Even at **step 47** (≈23.5 h
ahead), **54% of the lag weight sat on lag 1** — the price 30 min ago. A
direct-strategy linear model given a recency lag leans on it at *every* horizon,
so the far-ahead forecast is dragged toward the current level (persistence),
flattening the diurnal shape. AR wins because its lags are *all* daily-aligned
(`[1d, 2d, 7d]` = same interval-of-day on prior days) — there is no recency lag
to anchor to, so it preserves the time-of-day peak shape that capture rewards.
The smoothers were not the problem; the *recency* lags were.

**Correction (v2) — daily-aligned lean basis.** Rewrote `lean_lags` to
same-interval-of-day lags only: prior 1/2/3/7/14 days plus ±1 around yesterday
(to place a shifted peak), no sub-daily lags. This is the faithful Lago LEAR
design. Re-ran `lear_lean` / `ols_lean`.

**Result v2 (daily-aligned) — the correction helped, but AR still wins.**
`lear_lean` **0.446 → 0.484**, `ols_lean` **0.492 → 0.509** (dropping the recency
lags recovered +0.02–0.04, confirming the anchoring diagnosis). But neither beats
AR (0.543) or `lear` (0.516).

**Conclusion — AR's minimal 3-lag OLS is the linear ceiling.** Every added degree
of freedom *dilutes* capture:

| linear model | lags | estimator | capture |
|---|---|---|---|
| **autoregression** | 3 daily `[1d,2d,7d]` | OLS | **0.543** |
| lear (full) | 3 daily + smoothers | Lasso | 0.516 |
| ols_lean | 7 daily `[1d±1,2d,3d,7d,14d]` | OLS | 0.509 |
| lear_lean | 7 daily | Lasso | 0.484 |

More lags (7 vs 3) → worse; the extra distant/neighbour lags (3d, 14d, ±1) smear
the daily peak instead of sharpening it. Regularisation → worse (Lasso < OLS on
the same basis: 0.484 < 0.509), because L1 shrinks toward the flat mean — good for
MAE, bad for peak timing. Smoothers → worse. **The capture-optimal linear model
is the *simplest* one**, which is why the "dumb" AR sits on top. This is the same
accuracy-vs-value gap as Entry 031 seen from the linear side: sophistication that
lowers MAE (LEAR, more lags, L1) trades away the peak timing capture rewards.

**Takeaway for the campaign.** Do not look for capture gains inside the linear
family — AR is already its ceiling, and the residual gap to the oracle (0.543 →
1.0) is *spikes*, which no daily-shape point model can call (the forecast-skill
ceiling, Entry 030). The `feature_set="lean"` knob + `lear_lean`/`ols_lean` are
kept as the documented controls that establish this ceiling.

## Entry 033: Conformal-calibrated fan + open-loop probabilistic dispatch

The point-forecast ceiling is AR 0.543 (Entry 032); the residual is spikes. A
point model can't call spikes, but a *quantile fan* can carry spike **risk** in
its upper tail — if that tail is honest. Two problems block it today: (1) the raw
fan's upper quantile badly under-covers (q0.98 forecast maxes ≈ $1.9k vs actual
$20.3k, Entry 030), so scenario/CVaR dispatch is timid; (2) every probabilistic
executor runs under 30-min reforecast **churn** (Entry 030), which caps testbed
capture at ~0.25 (`point` 0.254 > every fan mode — robust/ev/cvar 0.18–0.24).

**Built two things to attack both.**

* *Calibration (problem 1).* Wired the existing `ConformalWrapper` into the
  quantile-fan fit as **split conformal** (`model_params.calibrate`,
  `_conformal_fan_adjustments` / `_apply_conformal`): hold out a recent tail, fit
  boosters on the rest, and per (step, quantile) shift the fan in **dollar
  space** to nominal coverage. On synthetic spiky data the q0.98 path widened
  2.4× ($287 → $686 mean) while q0.05 barely moved — exactly the spike-awareness
  the raw fan lacks. Registered `lightgbm_qmean_cal`; persisted through
  save/load. Tests: `tests/test_sim_conformal_fan.py`.
* *Churn (problem 2).* Added **open-loop** dispatch variants to the testbed grid
  (`open_loop` → forecast + solve once per horizon, no reforecast): `point_ol`,
  `ev_ol`, `cvar050_ol`, `meancvar_ol`. Replay over the frozen fan, so free.

**Results (full 30-min common window, same oracle $25.18M as AR — directly
comparable).**

| dispatch | baseline fan | calibrated fan |
|---|---|---|
| **meancvar_ol** (open-loop joint mean-CVaR LP) | **0.5451** | 0.495 |
| point_ol (open-loop point) | 0.537 | 0.461 |
| ev_ol (open-loop expected value) | 0.453 | 0.353 |
| point (30-min MPC) | 0.254 | 0.265 |
| all other MPC fan modes | 0.18–0.24 | 0.16–0.25 |

**Three findings.**

1. **Open-loop ≈ 2× the MPC ceiling (0.25 → 0.54).** The fan executors were never
   the problem — the 30-min **reforecast churn** was (Entry 030). Committing to a
   day-ahead plan and not churning on a lagging reforecast recovers ~0.28 of
   capture. This is the single biggest dispatch lever found.
2. **`meancvar_ol` 0.5451 edges past AR 0.543** — the *first* configuration to
   clear the point bar. The true joint mean-CVaR LP (Rockafellar–Uryasev) over
   open-loop scenarios adds ≈ +0.8 pt over open-loop point dispatch (0.537). Small
   but real, and it means the fan carries a little decision-useful tail signal the
   point forecast throws away.
3. **Conformal calibration HURT dispatch** (meancvar_ol 0.545 → 0.495; every
   open-loop mode dropped). Widening the upper quantile to nominal *coverage*
   raises the predicted spike *magnitude* everywhere the tail is fat, but adds no
   spike *timing* — so CVaR/EV dispatch over-positions for spikes that do not
   arrive (false-positive discharges) and mistimes. **Calibrated-for-coverage ≠
   calibrated-for-decisions**: an honest negative for split conformal here, even
   though coverage itself improves (`scripts/forecast_quality.py`). Calibration is
   kept (correct, off by default) but is not on the value path.

**Takeaway.** The dispatch win is *open-loop* + *mean-CVaR*, not calibration. This
is the counterpart to the forecast-side lesson (Entry 031/032): the metric rewards
timing/commitment, and machinery that optimises a proxy (MAE; coverage) trades
value away.

**λ/α sweep (open-loop mean-CVaR, free replay over the baseline fan).**

| λ \ α | 0.1 | 0.3 | 0.5 |
|---|---|---|---|
| 0.3 | 0.547 | **0.559** | 0.529 |
| 0.5 | 0.533 | 0.526 | 0.545 |
| 0.7 | 0.517 | 0.527 | 0.558 |
| 0.9 | 0.448 | 0.470 | 0.507 |

(λ=0 pure EV = 0.537.) **Best 0.5587 at λ=0.3, α=0.3 — AR + 1.6 pt, the first
clear beat of the bar.** Mild risk-aversion helps (a little CVaR weight on the
tail improves timing); heavy risk-aversion (λ=0.9) is too conservative and hurts.
The surface is **noisy** (several mild configs cluster 0.545–0.559 with no clean
monotonicity), and this is tuned **on the eval window**, so the honest claim is
"open-loop mean-CVaR with *mild* risk-aversion beats AR, ≈0.55–0.56," not that
0.5587 is a precise optimum. The `meancvar_ol` testbed entry is set to λ=0.3,
α=0.3.

**Cross-window validation — the mean-CVaR beat does NOT generalise.** The λ/α
above was tuned on the full window, so I replayed `point_ol` vs mean-CVaR on the
two regime-contrasting sub-windows (their own oracles):

| window | point_ol | mean-CVaR λ0.3α0.3 | mean-CVaR λ0.5α0.5 |
|---|---|---|---|
| spike_jan26 | **0.576** | 0.575 | 0.574 |
| calm_sep25 | **0.648** | 0.632 | 0.631 |
| full | 0.537 | 0.559 | 0.545 |

On **both** held-out regimes, open-loop **point** matches or beats mean-CVaR — in
the calm month clearly (0.648 vs 0.632), because sizing tail risk that never
arrives costs money. The 0.559 "beat" was **specific to the full window it was
tuned on**. So the honest correction: mean-CVaR's increment over point is fragile
/ window-specific and does *not* durably beat AR. The testbed `meancvar_ol` entry
is reset to the neutral λ0.5/α0.5 default.

**Where this leaves the campaign (honest version).** The robust, generalising
dispatch win is **open-loop itself** — commit a day-ahead plan, don't churn on a
lagging reforecast: 0.54–0.65 across windows vs ~0.25 under 30-min MPC. On top of
that, open-loop *point* dispatch (~0.537 full) roughly ties AR (0.543); clever
risk-shaping (mean-CVaR) and coverage-calibration each add at most a
non-generalising bump and often hurt. This is the same lesson as Entry 032 from
the dispatch side: the point forecast + a committed plan is a robust ceiling, and
proxy-optimising machinery (MAE, coverage, in-sample-tuned risk aversion) trades
value away. The one durable, transferable result is: **kill the reforecast
churn.**

## Entry 034: MPC's collapse is `observe_present`, not reforecast churn — a real bug

Prompted by "an MPC must be able to beat open-loop." It must — MPC has strictly
more information (true state + fresh forecast), so at worst it degrades to
open-loop. MPC scoring *half* of open-loop (0.25 vs 0.54) is a bug, not physics.
Perfect-foresight MPC = 1.0 (Entry 030), so the LP/execution is sound; the defect
is in how the receding horizon uses the forecast. Ran a decomposition (cached fan,
point dispatch, full window):

**Sanity 1 — reforecast vs re-solve (fix one, vary the other).**

| | resolve 48 | 24 | 12 | 6 | 1 |
|---|---|---|---|---|---|
| reforecast=48 | 0.537 | 0.539 | 0.493 | 0.486 | **0.213** |

Fixing resolve=1 and varying reforecast 48→1 *rose* 0.213→0.259. So **re-solving
frequently is the killer; reforecasting is not** — the exact opposite of the
handoff's "MPC amplifies forecast noise / reforecast churn" story (Entry 030),
which was **wrong**.

**Sanity 2 — budget?** Removing the per-day throughput cap (max_cycles 2→100)
left resolve=1 at 0.218. **Not the budget.**

**Sanity 3 — `observe_present`?** DP optimality says resolve=1 on a *fixed*
forecast must reproduce open-loop. It does — but only with `observe_present`
**off**:

| config | observe=ON | observe=OFF |
|---|---|---|
| open-loop (48,48) | 0.537 | 0.538 |
| re-solve (48,1) | **0.213** | **0.539** ← = open-loop |
| full-MPC (1,1) | 0.259 | 0.437 |

**Root cause: `observe_present`.** It pins plan step 0 to the *actual* current
price (correct in spirit — you know it at decision time), but step 0 is the true
price while steps 1..47 stay the *smoothed* forecast. At every re-solve the LP
therefore sees step 0 as unusually cheap/expensive vs its own forecast and trades
that gap — i.e. it **churns cycles on the high-frequency forecast residual**
(mean-reversion noise) instead of executing the daily arbitrage. Harmless at
resolve=48 (pins one step per day); lethal at resolve=1. It was added in Entry 024
to fix a forecast-the-present buy-high/sell-low bug — the fix over-corrected.

**Fix — spike-gated observe (`observe_gate`, Entry 035 next).** Keep the true
current price only when it is a genuine spike (≥ gate $/MWh) — the events
open-loop under-forecasts and misses — and trust the forecast otherwise. On normal
steps MPC then equals open-loop (DP-consistent); on spikes it discharges into the
real price. Implemented in `mpc.py` (all three dispatch branches).

## Entry 035: Spike-gated MPC beats open-loop — cross-validated, the deliverable works

Config: **reforecast=48 (day-start forecast), resolve=1 (re-solve every interval
with true SOC), `observe_gate` ≈ $2–3k** (pin step 0 to the actual price only when
it is a genuine spike). `lightgbm_qmean` fan, point dispatch.

**Gate sweep on full (reforecast=48, resolve=1):** 0→0.213, 500→0.452, 1000→0.513,
OFF→0.539, **2000→0.576**, 3000→0.569, 5000→0.568, 8000→0.576. Everything in
[2000, 8000] beats open-loop 0.539; below ~1500 it degrades toward the churn.

**Cross-validation — beats-or-ties open-loop on all three windows:**

| window | open-loop | gate 2000 | gate 3000 | gate 5000 |
|---|---|---|---|---|
| spike_jan26 | 0.579 | **0.656** | **0.656** | 0.579 |
| calm_sep25 | 0.652 | 0.652 | 0.652 | 0.652 |
| full | 0.539 | **0.576** | 0.569 | 0.568 |

**It generalises** (unlike the mean-CVaR λ/α of Entry 033) because the mechanism is
causal, not a fitted knob: on a spike, MPC discharges into a price the forecast
never predicted (+7.7 pt in the spiky month); on every non-spike interval it is
*provably* open-loop (the calm month, with no ≥gate spikes, is an exact tie — zero
harm). gate 5000 misses the Jan spikes (they sit ~$2–4k) so it reverts to
open-loop there — hence gate ≈ 2500–3000 is the robust choice (catches genuine
scarcity, ignores noise). **On full, 0.576 > AR 0.543 > open-loop 0.539.**

**Resolution of the whole campaign.** The deliverable is forecast-driven MPC, and
it now works and beats every bar: the "MPC is hopeless" conclusion was a bug
(`observe_present` churning the forecast residual), not physics — exactly as the
user predicted ("an MPC must be able to beat this"). The value comes from what MPC
is *for*: reacting to observed scarcity the day-ahead forecast cannot call, while
committing to the clean day-ahead arbitrage the rest of the time. Reforecasting
intraday does *not* help (daily forecast is best); the lever is state/price
feedback via the gated observe, not fresher forecasts.

## Entry 036: Balanced per-month ablation — quantile GBM + Fourier wins; Fourier hurts LEAR

Due-diligence sweep with a **balanced** metric (mean of the 12 per-month capture
ratios; `scripts/balanced_eval.py`), because a pooled annual ratio is dominated by
~2 mega-spike days (Entry 035). Factorial {LEAR, LightGBM} × {point, quantile} ×
weather{off,on} × Fourier{off,on}, each × {open-loop, spike-gated MPC}; 13 of 16
cells (the 4 LEAR-quantile cells deferred — sklearn QuantileRegressor is ~10-15h/fan
on CPU, see HANDOFF §3 for the MPS plan).

**Best: `lightgbm_qmean_weather_fourier` + spike-gated MPC = 0.554 balanced /
0.585 pooled** (open-loop 0.558 / 0.614). Ranking: quantile GBM > point tree > LEAR.

Ablation reads (balanced, spike-gated): **Fourier helps trees (rich 0.509→0.521)
and quantile-GBM (with weather 0.522→0.554) but HURTS LEAR (0.484→0.426).** That
LEAR result is the phase-regularisation artifact: L1 penalises `|sin|+|cos|`, which
is not rotationally invariant, so it biases the harmonic's phase (trees/Ridge are
immune). Fix = group-Lasso on each {sinₖ,cosₖ} pair (HANDOFF §4). Weather: hurts
trees, helps LEAR, helps quantile-GBM only *with* Fourier. Figures:
`outputs/figures/ablation_{spikegated,openloop}.png`.

**Metric note.** Report balanced (equal-weight per month) *and* pooled — the pooled
number is a high-variance "did you catch the 2 big days"; balanced is the fairer
model comparison.

## Entry 037: The MPS quantile-LEAR plan fails (wrong gradients) — CPU batching fills the deferred cells; Fourier *helps* LEAR once the calendar is unpenalised

Closes the four deferred LEAR-quantile ablation cells (Entry 036, HANDOFF §3). The
plan was to reimplement the fan on MPS; the reality had three twists worth
recording, each a genuine failure caught and fixed.

**1. MPS gives wrong gradients — the device was never the point.** The model is a
single linear layer `W: features → (steps × quantiles)` trained by full-batch Adam
on the pinball loss — one autograd fit replacing sklearn's ~100 sequential HiGHS
LPs. Implemented and validated on MPS as planned. On real features **MPS diverges**:
the training objective *ascends* every epoch (pinball 0.44 → 9.9 over 300 epochs),
while the **identical code, seed and data on CPU converge cleanly** (0.44 → 0.15).
Reproduced, isolated to the MPS autograd backend (a `maximum`-free pinball still
diverges; not an lr issue — diverges at every lr). Lesson: **the ~40x speedup was
the *batching* (1 fit vs 100 LPs), not the device.** The model is pinned to CPU
(`torch.device("cpu")`); ~13 s/refit, a full fan in ~3.5 min. Revisit MPS only if a
torch release fixes the backend. This is why the model is `lear_qmean_torch`, not
`_mps`: naming it for a device it can't use would mislead.

**2. Heavy-tailed features wreck GD conditioning (the LP didn't care).** The
`price_ret_*` features hit ~200 std even after `StandardScaler` (a tiny denominator
at a spike), and the *first* full build blew up: a diverged fit produced asinh-space
predictions so large that `sinh` overflowed to `inf`, crashing the dispatch LP
(`linprog: c must not contain inf`). Two guards: winsorise the standardised design
matrix to ±5 σ (bounds each feature's leverage — GD-specific, sklearn's LP needs
no such guard), and clamp predictions to the historical transform-space envelope so
no extrapolation can ever feed `inf` into dispatch.

**3. Don't regularise the calendar — for *either* encoding.** Per the brief, the
Fourier calendar block is exempt from the L1 penalty (penalising sin/cos as
`|a|+|b|` biases the phase — the Entry 036 wart). The subtle bug: exempting *only*
Fourier left the **one-hot** calendar dummies penalised, and `alpha` shrank the
hour-of-day profile flat — median fan spread **$5** vs Fourier's $416, so dispatch
saw no arbitrage and onehot capture was **~0.00**. Fix: exempt the *whole* calendar
block (`_leading_calendar_columns` via the fitted ColumnTransformer's
`output_indices_`), both encodings. The calendar is low-dimensional and
deterministic — L1 is for selecting among the lag/weather predictors, not for
shrinking the daily shape. onehot spread jumped $5 → $322; capture ~0 → 0.479.

**Results (balanced / pooled, spike-gated MPC).** All four cells now land in a
coherent LEAR tier:

| model | balanced | pooled |
|---|---|---|
| lear_qmean_torch_fourier | **0.493** | 0.526 |
| lear_qmean_torch (onehot) | 0.479 | 0.511 |
| lear_qmean_torch_weather_fourier | 0.479 | 0.508 |
| lear_qmean_torch_weather | 0.469 | 0.501 |

**Fourier now HELPS LEAR (0.479 → 0.493), reversing Entry 036's "Fourier hurts
LEAR" (0.484 → 0.426).** That confirms the wart was the *penalty*, not the encoding:
unpenalise the calendar and the cyclic encoding is a net win. Weather hurts the
quantile LEAR (−0.01), consistent with weather hurting trees. The ranking is intact:
**quantile GBM (best 0.554) > quantile LEAR (0.493) ≈ point lear_weather (0.500) >
autoregression (0.470) > the old penalised-Fourier LEAR (0.426).** So the
probabilistic LEAR is a respectable linear tier, still clearly under the quantile
GBM. Per-month heatmap: `outputs/figures/lear_torch_cells.png`.

## Entry 038: The autoregression forecast was frozen at fit time — a predict-from-now bug that flatlines under MPC

Spotted on the dashboard: `autoregression__mpc30` shows a dead-flat forecast line.

**Diagnosis.** `_ar_predict` seeded its lags from `state["last_values"]` — the
series tail captured at *fit* time — and never read `input_df`. So the model
produced the **identical** multi-step trajectory at every origin between refits,
regardless of the live price. Under open-loop (one forecast/day, whole trajectory
shown) this looks like a plausible daily curve, so it hid; under **mpc30**
(re-forecast every interval, only the near-term value recorded) every interval got
the same number → a flat line, and capture collapsed (mpc30 ≈ 0.27 vs open-loop
0.47). Empirically the executed forecast had **1 distinct value across 673
intervals**; every other model had 625–667. The AR was never actually
reforecasting — the whole premise of MPC (predict-from-now) was a no-op for it.

**Audit — is anything else affected?** Two ways, agreeing. (a) Empirical: only the
three `autoregression*` variants had a degenerate (`distinct≤2`) executed forecast.
(b) Code: every `predict` was checked for whether it reads `input_df`. Clean —
`naive`, the LEAR family (`_linear_predict`), `_lgbm_rich_predict`,
`_lgbm_qmean_predict`, and `lear_qmean_torch` all condition on `input_df`. The same
frozen-tail pattern *does* exist in `_lgbm_predict` (plain `lightgbm`),
`_mlp_predict` and `_lstm_predict` — but **those three run in zero trials**, so
nothing to re-run.

**Fix.** `_ar_predict` now seeds its lags (and the future calendar origin) from
`input_df[target_col].tail(max(lags))` when the caller supplies it, falling back to
the frozen tail only if it doesn't. `target_col` is threaded through fit/save/load.
Regression test: shifting the recent input moves the forecast, and it is not a flat
line. Verified on the rebuilt `autoregression__openloop` — the 18:00 forecast now
takes **359 distinct values across 364 days** (spanning \$82–\$1256, reacting to
each day's prices) where the bug repeated one trajectory. The `autoregression*`
trials were deleted and re-run (30 min, refit 28, lookback 548, full window); the
mpc30 executed forecast went from **1 distinct value to 648** and now tracks the
daily cycle.

**The capture barely moved — and that's the honest result.** Balanced capture:
open-loop 0.47→**0.474**, mpc_spike 0.47→**0.476**, mpc30 0.27→**0.246**. The stale
forecast was not *cheating*: arbitrage value comes from the daily price *shape*,
which the frozen trajectory reproduced about as well as the live one, so correcting
it leaves open-loop/mpc_spike essentially unchanged. mpc30 actually dips slightly —
a nice illustration of Entry 034: mpc30 is the buggy `observe_present` executor, and
a correct, *volatile* forecast churns its dispatch a touch more than a stable-wrong
flat line did. AR now shows every point model's normal pattern (mpc30 ≪ mpc_spike ≈
open-loop) instead of a degenerate flat forecast. So the fix is about
**correctness/honesty of the forecast**, not a capture win — the baseline's standing
is unchanged.

**Deprecation.** `simple_mlp` and `lstm` were never developed far enough to compete
and carry the same latent frozen-tail bug. Per the call, they are **deprecated**
(fit warns; docstrings + registry note) but **left in the code**; their only
artifacts were already parked in `outputs/_archive_valtest_trials/`. The plain
`lightgbm` base shares the latent bug but is kept as-is (unused, and `lightgbm_rich`
is the real tree model).

**Lesson.** "Predict-from-now" is a contract every forecaster must honour to be
MPC-compatible — read the live tail, don't bake the training tail into the state.
A model can look fine open-loop and be silently inert under receding-horizon
control; the tell is a low distinct-value count in the executed forecast.

**Deprecation.** sklearn `lear_qmean` is now deprecated (warns + docstring) — it is
~8 min/refit *and* structurally can't exempt calendar columns from its L1 penalty,
so it can't shed the wart. `lear_qmean_torch` supersedes it.

**Infra.** Fan-cadence is now a first-class `testbed.py --fan-cadence` flag (retires
the scratch monkeypatch, HANDOFF §6), and `grid --executors point_ol,mpc_spikegate`
replays just the ablation executors instead of the whole slow grid.
