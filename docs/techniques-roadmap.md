# Learning roadmap — every technique used to forecast prices and dispatch the battery

**Purpose.** This is a *roadmap*, not the finished lesson. It lists — in teaching
order — every modelling, forecasting, calibration, evaluation, and dispatch
technique the grian project actually used, why we used it, where it lives in the
code, and the key choices and findings. A stronger model (or a patient human) can
expand each bullet into long-form prose for a **data/stats novice**: assume the
reader knows basic Python and high-school maths, nothing more. Define every term
on first use (quantile, pinball loss, walk-forward, capture ratio, CVaR, …).

**How to use it.** Each section has: *what it is* → *why grian uses it* → *how it's
implemented (file pointers)* → *the choices and the findings* → *expand on*. Follow
the file pointers and the linked docs/experiment-log entries to write from the
source, not from memory. Ground every claim in a number or a file.

**Companion material already in the repo** (don't duplicate — reference and go
deeper): [`docs/README.md`](README.md) (mental model), [`docs/architecture.md`](architecture.md),
[`docs/dispatch-and-scoring.md`](dispatch-and-scoring.md),
[`docs/dispatch-under-uncertainty-maths.md`](dispatch-under-uncertainty-maths.md),
[`docs/probabilistic-dispatch-explained.md`](probabilistic-dispatch-explained.md),
the 16-chapter [`docs/learning_guides/`](learning_guides/) PDF, the
[`outputs/experiment_log.md`](../outputs/experiment_log.md) (Entries 001–038 — the
narrative of what was tried and what failed), and the interactive
[`outputs/dashboard/index.html`](../outputs/dashboard/index.html).

---

## 0. The problem and the one-metric mental model

- **What.** Forecast South-Australian (SA1) electricity prices, then dispatch a
  100 MW / 200 MWh battery to arbitrage them; score against a perfect-foresight
  **oracle**. Headline metric: **capture ratio** = your revenue ÷ oracle revenue.
- **Why this framing.** A forecast is only worth its *decision value*. Capture
  ratio measures dollars kept, not forecast error — the whole project is a case
  study in why those two differ.
- **Expand on:** what NEM electricity prices look like (5-min settlement,
  interval-ending timestamps, negative prices, a market price cap ~\$16–17k, and
  violent scarcity spikes); why a battery is a price-arbitrage machine;
  perfect-foresight oracle as the denominator. Pointers: `src/grian/sim/oracle.py`,
  [`docs/dispatch-and-scoring.md`](dispatch-and-scoring.md), the "AEMO timestamps"
  note in [`CLAUDE.md`](../CLAUDE.md).

---

## 1. Data and the target variable

- **Data sources.** NEM price/demand (NEMOSIS/NEMSEER), ERA5 weather. Cached to
  disk, never re-pulled (determinism). Pointers: `src/grian/data.py`, notebooks
  01–04, [`docs/data-and-features.md`](data-and-features.md).
- **Resolution pivot.** The project ran at 5-min then pivoted to **30-min**
  (mean-resampled) for speed; horizon = 48 steps = one day-ahead. Explain the
  trade-off (spike resolution vs compute). Pointer: `SA1_30min_sim.parquet`.
- **Target transform: inverse hyperbolic sine (asinh).** Model `asinh(price)`,
  always invert before scoring so errors are in dollars.
  - *Why asinh not log:* prices go **negative**; `log1p` can't. asinh ≈ log for
    large \|x\| but is smooth through zero and handles negatives.
  - *Why transform at all:* compresses the enormous spike range so the model
    isn't dominated by a handful of \$15k intervals; a variance-stabilising trick.
  - Pointers: `_get_transform_pair` in `src/grian/sim/trials.py`; the "Target
    transform" rule in [`CLAUDE.md`](../CLAUDE.md).
- **Expand on:** why standardisation/scaling matters for linear + neural models
  but not trees; the difference between modelling the level vs the shape.

---

## 2. Feature engineering

Everything below is built in `src/grian/sim/features.py`; see
[`docs/data-and-features.md`](data-and-features.md).

- **Price lags** — the autoregressive backbone (yesterday, last week, same
  interval). The "lean" feature set is raw lags only.
- **Rolling statistics** — rolling mean/max/std over recent windows (captures
  regime / recent volatility).
- **Momentum** — returns, direction, acceleration (`price_ret_1h/6h/24h`). *Gotcha
  to teach:* these are heavy-tailed — a tiny denominator at a spike sends them to
  ~200 standard deviations even after scaling, which wrecks gradient-descent
  conditioning (Entry 037). Winsorising fixed it for the torch model.
- **Demand** — the physical driver of scarcity.
- **Weather (ERA5)** — solar irradiance, temperature, wind speed. *Finding:*
  weather **hurts trees, helps LEAR, and helps the quantile GBM only with
  Fourier** (Entry 036). Teach why more features can hurt (variance/overfitting,
  and target-time vs forecast-time weather leakage — see the note in
  `features.py`).
- **Scarcity / spike-precursor features** — engineered to *anticipate* spikes
  (`intervals_since_spike`, reserve proxies). *Finding:* scarcity features alone
  **hurt** capture (Entry 030/031) — the objective, not the features, was the
  lever.
- **Calendar features** — hour, day-of-week, month, weekend flag. Their *encoding*
  is a whole sub-topic → §3.
- **Expand on:** feature leakage and how the backtest embargo prevents it (§7);
  why "richer" isn't "better"; the lean-vs-rich experiment (Entry 032).

---

## 3. Calendar encodings — a deceptively deep topic

- **Three encodings** (`calendar_encoding` in `model_params`):
  - **Ordinal** — integer hour/dow/month. Correct for **trees** (they split on
    thresholds); *misleading for linear models* (implies hour 23 ≈ hour 0 + 23).
  - **One-hot** — a dummy per category. Default for linear/LEAR; no false ordering.
  - **Fourier** — cyclic sin/cos harmonics so hour 23.5 and hour 0 are adjacent.
    Smooth, low-dimensional; `fourier_calendar_features` / `_fourier_calendar`.
- **The phase-regularisation "wart"** (Entry 036/037 — a great teaching story).
  For an L1 (Lasso) linear model, penalising `|a·sin| + |b·cos|` is **not**
  rotationally invariant: it biases the harmonic's *phase* toward clock-aligned
  positions and can zero one of the pair. Empirically "Fourier hurts LEAR" — an
  artifact of the *penalty*, not the encoding. Fixes, in order of correctness:
  group-Lasso on each {sin,cos} pair (penalise amplitude); or **exempt the whole
  calendar block from the penalty** (what the torch model does). Once unpenalised,
  **Fourier helps LEAR** (0.479→0.493). Pointers: `_linear_preprocessor`,
  `_leading_calendar_columns`, `_l1_penalty` in `models.py`.
- **Expand on:** what "rotationally invariant" means with a picture; why trees are
  immune (no coefficient penalty); why Ridge (L2) is immune (penalises amplitude).

---

## 4. The models (the forecasters)

All models are plain dicts with `fit/predict/save/load` — no classes, no
inheritance (`src/grian/sim/models.py`, `REGISTRY`). Teach the interface first.

- **naive_similar_day** — repeat the most recent same-weekday/time. The floor
  every learned model must beat. *Teaching point:* it's a pure target-time lookup,
  so it forecasts identically no matter when you run it (open-loop == MPC) — but
  its *dispatch* still differs.
- **autoregression (AR)** — linear regression on price lags + calendar, iterated
  recursively for multi-step. Teach recursive vs direct multi-step forecasting and
  mean-reversion (a stable AR decays to the mean → flat far-horizon forecast).
  *Bug worth teaching (Entry 038):* it originally forecast from a frozen
  fit-time tail and ignored live data — fine open-loop, flatlines under MPC. The
  "predict-from-now" contract every MPC-compatible model must honour.
- **LEAR family (linear)** — L1-penalised (Lasso) linear regression on the rich
  feature set, one model per horizon step. Also Ridge (L2), ElasticNet (L1+L2),
  OLS. The classic electricity-price-forecasting (EPF) benchmark. `_linear_fit`.
  *Key finding:* LEAR has the **best forecast accuracy** (MAE-skill) yet the
  **worst capture** — it's a great forecaster and a poor trader (§8, Entry 032).
- **LightGBM (gradient-boosted trees)** — `lightgbm_rich` (point) on the full
  feature set. Trees model the nonlinear scarcity *thresholds* linear models
  can't → catch spikes → higher capture. Teach boosting at a high level.
- **Quantile models ("qmean")** — the probabilistic tier. §5.
- **Decision-focused / scarcity / calibrated variants** — `lightgbm_rich_dfl`
  (high-price sample weighting), `lightgbm_rich_scarcity`, `lightgbm_qmean_cal`
  (§6). Each attacks the accuracy-vs-capture gap a different way.
- **Deprecated** — `simple_mlp`, `lstm` (never competitive; carry the frozen-tail
  bug; kept as code, no results). Teach *why* they underperformed here (small
  data, spiky target, no clear inductive-bias advantage) rather than pretending
  deep learning always wins.
- **Expand on:** the fit/predict/save/load contract; why "functions + dicts, no
  classes" (the code-style choice); per-step vs single multi-output models.

---

## 5. Quantiles, the pinball loss, and the fan ("qmean")

- **What a quantile forecast is.** Instead of one predicted price, predict several
  quantile levels (the τ=0.05, 0.5, 0.9, 0.98 prices) → a **fan** = the predictive
  distribution. Teach quantile vs mean; why the levels are **asymmetric** (heavy
  on the upper tail — spike risk is what dispatch needs to see).
- **Pinball (quantile) loss.** `pinball(τ,y,ŷ)=max(τ·(y−ŷ),(τ−1)·(y−ŷ))` — an
  asymmetric absolute loss whose minimiser is the τ-quantile. Derive it; show the
  gradient is bounded. Pointers: `_pinball_loss` (torch), sklearn `QuantileRegressor`.
- **Two implementations, one idea.**
  - `lightgbm_qmean` — a gradient-boosted model per (step, quantile).
  - `lear_qmean_torch` — **one linear layer** `features → (steps × quantiles)`
    trained by batched gradient descent on the summed pinball loss. This replaced
    sklearn's ~100 sequential HiGHS LP solves per refit (~8 min → ~13 s, a ~40×
    speedup). *The speedup is the batching, not the device.*
  - **A real war story (Entry 037):** it was planned for the MPS (Apple GPU)
    backend; MPS computed **wrong gradients** and the fit diverged while identical
    CPU code converged. Teach: verify numerics, don't trust the accelerator; a
    model can look fine on toy data and diverge on real features.
- **From fan to point forecast.** Integrate the quantiles: `E[X]=∫₀¹Q(τ)dτ ≈
  Σ wₜ·Q(τ)` (`_quantile_weights`). Because price is right-skewed the integrated
  mean ≠ the median — this matters. So a `qmean` model serves *both* the
  probabilistic dispatch (whole fan) and point dispatch (integrated mean).
- **Non-crossing.** Quantiles are sorted per step so a higher level is never below
  a lower one. Explain why crossing happens and why it's a problem.
- **Expand on:** why quantile GBM beat point tree here (fan feeds dispatch AND the
  integrated mean respects skew); the "qmean" naming (q = quantile, mean =
  integrated point).

---

## 6. Calibration — does the stated 90% actually cover 90%?

- **The problem.** A raw quantile fan is often **over-confident**: the nominal
  90%/98% bands under-cover, so scenario/CVaR dispatch under-prices spike risk.
- **Split-conformal calibration** (`lightgbm_qmean_cal`, `calibrate`/`cal_days`,
  `_conformal_fan_adjustments`/`_apply_conformal`). On a held-out recent tail,
  measure the empirical coverage and *widen* the quantiles to hit nominal. Teach
  conformal prediction at the intuition level (distribution-free coverage from a
  calibration set) and the finite-sample guarantee.
- **Finding.** Calibration helped scenario/CVaR see spike risk (Entry 033) but the
  risk-shaping increment did **not** survive cross-window validation — it was tuned
  to the eval window. A cautionary tale about knobs that fit the test set.
- **Expand on:** reliability diagrams; coverage vs sharpness; why calibration is a
  *decision*-quality fix, not an accuracy fix.

---

## 7. Evaluation — test periods, backtest, and leakage

- **Walk-forward (rolling-origin) backtest.** Train on all history to a day, trade
  the next day, roll forward. The *only* honest way to score a time-series
  strategy. Pointers: `src/grian/sim/runner.py`, `backtest`, `test_backtest.py`.
- **No leakage — the embargo.** An embargo the length of the horizon prevents the
  train window from touching the test window. There is a **unit test that injects
  a future value and proves it degrades the score** (`leak_future` ablation) — a
  great thing to teach: how you *prove* you're not cheating.
- **Refit cadence and rolling train window.** Refit every N days (28 here) on a
  rolling lookback (548 days). Teach the compute-vs-freshness trade-off.
- **The common evaluation window.** Every configuration is scored on **one
  identical span** (2025-07-01 → 2026-06-30) against one oracle, so numbers are
  comparable; the dashboard recomputes capture over any sub-window live from the
  saved daily revenue series. Pointer: `scripts/run_common_eval.py` docstring.
- **The old sin (why this matters).** Earlier work scored configs on *disjoint*
  windows → incomparable, and some results were voided by leakage. Teach the
  before/after. Pointers: [`docs/dispatch-and-scoring.md`](dispatch-and-scoring.md)
  ("the traps that voided every earlier result").
- **Expand on:** train/val/test vs walk-forward; why a single fixed held-out
  period per the honest-benchmark rule ([`CLAUDE.md`](../CLAUDE.md)); determinism
  (seeds, pinned versions, cached downloads).

---

## 8. Metrics — and why the "best" model depends on which one

- **Forecast-error metrics.** MAE, RMSE (penalises big misses more), and **MAE
  skill** = 1 − MAE/naive-MAE (positive = beats naive). `src/grian/sim/metrics.py`.
- **Decision-value metrics.** **Capture ratio** = revenue ÷ oracle revenue;
  **regret** = oracle − you (dollars, and %). `src/grian/sim/analytics.py`.
- **Balanced vs pooled capture** (the metric that mattered). A single pooled
  annual capture is dominated by ~2 mega-spike days (top-10 days ≈ 45% of oracle
  revenue) → high variance. **Balanced** = mean of the 12 per-month capture ratios
  (equal weight/month) is the fairer model comparison. Pointer:
  `scripts/balanced_eval.py`.
- **Risk metrics.** Sharpe (mean/std of daily revenue), peak drawdown.
- **THE headline finding (Entry 032, visible in the dashboard).** MAE and capture
  are **inversely ranked**: `lear_weather` has the best MAE-skill (~0.63) and the
  *worst* capture (~0.50). Capture lives in the spikes; MAE lives in the calm
  body. Teach *why the most accurate forecaster is not the best trader.*
- **Expand on:** proper scoring rules; value-weighted vs equal-weighted metrics;
  why you must pick your metric before you pick your model.

---

## 9. Model selection — choosing on robustness, not a single number

- **Best-of-family across periods.** Rank models by mean-across-periods capture
  *and* worst-period capture (not just the full-year number), so you don't pick a
  model that only wins the spike month. The dashboard's "Balanced model selection"
  view (periods trajectory + a normalised metrics radar) operationalises this.
- **The curated "key models" set** — one representative per family × output
  (LEAR/tree × point/quantile) + the champion + a baseline, chosen on the
  period-robust score, filtering out dominated and buggy variants.
- **Expand on:** overfitting the eval window; why a flat, high line across periods
  beats a spiky one; multi-objective selection (capture, consistency, Sharpe,
  drawdown) via the radar.

---

## 10. Dispatch — turning a forecast into trades

The forecast is only half the system; the **executor** decides charge/discharge.

- **Battery physics + the LP.** Power/energy limits, round-trip efficiency, cycle
  limits, state-of-charge dynamics; a linear program maximises arbitrage revenue
  over the horizon subject to those constraints. Pointers: `src/grian/sim/lp.py`,
  [`docs/dispatch-and-scoring.md`](dispatch-and-scoring.md).
- **Open-loop vs MPC (receding horizon).** Open-loop: forecast the day once at
  midnight, solve one LP, commit. MPC: re-solve every interval from the *true*
  state of charge and re-forecast on a cadence. Teach receding-horizon control.
- **Point vs probabilistic executors** (`dispatch_mode`, `src/grian/sim/dispatch_prob.py`):
  - **point** — dispatch on a single price path (the forecast, or the fan's
    integrated mean).
  - **scenario / robust** — solve the LP on each quantile path (each keeps its own
    price ranking) and fuse: min-action where all agree.
  - **EV** — probability-weighted mean action over the fan.
  - **CVaR (λ sweep)** — shade dispatch toward the bad revenue tail; λ dials
    expected-value → worst-case. Teach **Conditional Value-at-Risk**.
  - **mean-CVaR** — one joint Rockafellar–Uryasev LP trading mean vs the CVaR of
    the worst-α tail. Pointer:
    [`docs/dispatch-under-uncertainty-maths.md`](dispatch-under-uncertainty-maths.md).
- **The winning executor: spike-gated MPC.** Forecast once per day, re-solve every
  interval, but only *observe* (react to) the live price when it clears a \$3000
  gate — so it catches genuine scarcity the day-ahead forecast missed while staying
  provably open-loop the rest of the time. `observe_gate` in `src/grian/sim/mpc.py`.
- **A real bug worth a whole lesson (Entry 034/035).** The prior MPC collapsed to
  half of open-loop — traced to `observe_present` churning the forecast residual,
  not "reforecast churn." The fix (spike-gated observe) beats open-loop on every
  regime. *Intraday reforecasting does not help; state/price feedback does.*
- **A subtlety the dashboard makes visible.** The recorded forecast is
  dispatch-mode *independent* — robust/EV/CVaR all read the same forecast and
  differ only in how they trade it; only the *re-forecast cadence* changes the
  forecast line. Teach the difference between the forecast and the decision.
- **Expand on:** why the non-gated intraday MPC (mpc30) is *worse* than a
  stable-wrong flat forecast (it churns); the "which executor for which model"
  matrix in the dashboard.

---

## 11. Hyperparameters and tuning

- **What each family exposes.** LightGBM: `n_estimators` (150), `learning_rate`
  (0.05), plus the quantile levels; LEAR/Lasso: `alpha` (L1 strength, 0.01);
  `lear_qmean_torch`: `alpha`, `lr` (0.005), `epochs` (400), `feature_clip` (±5 σ
  winsorising); executors: `cvar_lambda`, `cvar_alpha`, `observe_gate`. Pointers:
  `_MODEL_PARAMS` in `scripts/testbed.py` and `run_common_eval.py`.
- **How tuning was actually done.** Mostly *principled defaults + a small factorial
  ablation* rather than a big blind search: the 16-cell {LEAR,LightGBM} × {point,
  quantile} × weather{off,on} × Fourier{off,on} matrix, each × {open-loop,
  spike-gated MPC} (Entry 036). Teach ablation as hypothesis-driven tuning; note
  the CVaR λ/α sweep and *why its winner didn't generalise* (§6).
- **Numerical-stability choices** (torch model): winsorise features, standardise
  the target, learning-rate/epochs picked to *converge*, an envelope clamp on
  predictions so a bad extrapolation can't feed `inf` into the LP. Teach that "make
  it converge and stay finite" is part of hyperparameter choice.
- **Expand on:** grid vs random vs Bayesian search and why we didn't need them
  here; regularisation strength intuition; the bias-variance dial.

---

## 12. Engineering practices that make the results trustworthy

- **Determinism & caching.** Seeds set, versions pinned, every download cached,
  never re-pulled. CPU/Apple-Silicon only. ([`CLAUDE.md`](../CLAUDE.md) rules.)
- **Everything on disk, reproducible from `config.json`.** A trial = config +
  ledger + metrics + (optional) model + forecasts. The dashboard reads these.
- **Checkpointing & telemetry.** Long fan builds checkpoint to `.partial` and
  promote only on completion; per-refit telemetry at INFO. Born from a real
  10-hour, invisible, unrecoverable build (Entry 035 infra note).
- **The experiment log as a lab notebook.** Every failure gets an educational
  write-up in [`outputs/experiment_log.md`](../outputs/experiment_log.md) — the
  single best artifact to teach "how research actually goes."
- **Thin notebooks, fat library.** All logic in `src/grian/sim/`; notebooks import,
  explain, visualise. Tests in `tests/` prove the physics and the no-leakage claim.
- **Expand on:** why reproducibility is a scientific requirement, not a nicety; how
  the config schema makes every run auditable.

---

## 13. The through-line — lessons to leave the reader with

1. **Decision value ≠ forecast accuracy.** The best forecaster (LEAR) is a poor
   trader; capture lives in the spikes.
2. **Predict-from-now is a contract.** A model that bakes in its training tail is
   silently inert under MPC (the AR flatline).
3. **The objective is the lever, not the features.** Scarcity features alone hurt;
   spike-aware *dispatch* (the gate) is what won.
4. **Verify numerics.** MPS gave wrong gradients; the accelerator is not free
   correctness. Winsorise heavy tails; clamp before the LP.
5. **Pick the metric before the model.** Pooled vs balanced capture changes the
   ranking; a single number over one period is a trap.
6. **Regularisation has geometry.** L1 on sin/cos biases phase — exempt or
   group-penalise the calendar.
7. **Beware knobs tuned to the test set.** The CVaR risk-shaping win didn't
   survive cross-window validation.

---

### Suggested chapter order for the finished document

Data & target (§0–1) → Features (§2) → Calendar encodings (§3) → Models (§4) →
Quantiles & the fan (§5) → Calibration (§6) → Backtest & leakage (§7) → Metrics
(§8) → Model selection (§9) → Dispatch & executors (§10) → Hyperparameters (§11)
→ Engineering & reproducibility (§12) → Lessons (§13). Each chapter should end
with a runnable pointer (a script or a dashboard view) so the reader can *see* the
technique, and a "what went wrong" box drawn from the experiment log.
