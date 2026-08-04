# Capture Ratio Campaign — Plan of Record

**Branch:** `capture-campaign` · **Written:** 2026-07-11 · **Author:** planning agent (Fable)

**Goal:** dispatch the Mannum-class battery (100 MW / 200 MWh, SA1) at a
capture ratio consistently above 0.50, targeting 0.65–0.75. Capture ratio =
realised arbitrage revenue ÷ perfect-foresight oracle revenue, same battery,
same constraints, same test window.

This document is the single source of truth for the campaign. It is written so
that a worker with no context can pick up any ticket and execute it. Read
§1–§4 before touching code. §5 is the ticket board. §7 is the trap register —
read it before starting *any* ticket.

---

## 1. The metric, frozen

Nothing matters until the scoreboard is trustworthy and fixed. These
definitions are frozen; changing them requires a new entry in this section
with a dated justification, and invalidates all previous headline numbers.

### 1.1 Battery

| Parameter | Value |
|---|---|
| Power | 100 MW (charge and discharge) |
| Energy | 200 MWh (2 h duration) |
| Round-trip efficiency | 0.85 (√0.85 each way) |
| Cycle limit | 2 full cycles per calendar day (discharge throughput ≤ 400 MWh/day) |
| Market | SA1 energy only (no FCAS in the headline metric) |
| Price-taker | Yes, for the headline metric (price-maker is a later, separate metric) |

### 1.2 Test window and data

- Data: `data/processed/SA1_5min_sim.parquet` — 5-minute `price` and `demand`,
  2023-01-01 → 2024-01-30.
- **Held-out test window: 2023-10-01 → 2024-01-30** (~122 days, spike season).
- **Validation window: 2023-07-01 → 2023-09-30** (train on Jan–Jun 2023).
  All model selection, hyperparameter tuning, and technique iteration happens
  on validation. The test window is touched once per technique family, for the
  headline number. (Contamination note: baseline test-window revenues were
  observed before this rule existed — Entries 001–012 in the experiment log.
  Discipline applies from here forward.)

### 1.3 The oracle (denominator)

Perfect-foresight revenue over the test window: a single LP over the full
window against **actual** prices, with

- SOC continuous across the whole window, `soc[0] = 0`,
- per-calendar-day discharge-throughput constraint (2 cycles/day),
- correct `dt = 1/12 h` for 5-minute intervals,
- the same efficiency and power limits as the agent.

If the full-window LP is too slow, solve in blocks of 7 days with SOC forced
to 0 at block boundaries and report the delta vs the per-day oracle once —
overnight carry is worth little under a 2-cycle limit, but we use the harder
oracle so the capture ratio is honest, not flattering.

### 1.4 The agent (numerator)

Realised revenue from the walk-forward simulation, where every executed
action is **physically feasible**: charge/discharge clamped so SOC stays in
[0, 200] MWh with correct dt, SOC carried continuously across days, cycle
budget enforced per calendar day at execution time (not just inside the LP).
Revenue = actual_price × (discharge − charge) × dt, summed.

### 1.5 Headline reporting

Every trial reports: capture ratio (headline), total revenue, oracle revenue,
daily decision regret series (oracle_day − realised_day), MAE (diagnostic
only), and the top-10 regret days. Capture ratio > 1 is a bug by definition —
assert it.

---

## 2. Audit of the current simulator — why existing numbers are void

Confirmed by code inspection on 2026-07-11. All revenue figures in experiment
log Entries 001–012 were produced under these defects and are **not capture
ratios and not physically meaningful revenue**. They remain valid as relative
model comparisons only in a loose sense.

1. **dt hardcoded to 0.5 h in the LP** (`src/grian/dispatch.py`, `schedule()`),
   while trials run at 5-minute resolution. The LP believes each interval
   moves 6× more energy than reality: it can fill the 200 MWh battery in 4
   intervals (20 real minutes at 100 MW would give 33 MWh, not 200) and
   believes the day is 144 hours long. Every schedule it produced was
   optimised for a fictional battery. Meanwhile the ledger scores revenue at
   the correct dt — so numerator physics and planner physics disagree.

2. **No feasibility coupling at execution** (`runner.battery_dispatch`): the
   LP's charge/discharge values are logged and monetised unclamped; only the
   *recorded* SOC is clamped. The battery can earn revenue discharging energy
   it does not hold. Phantom energy, phantom dollars.

3. **`soc[0] == 0` hardcoded in the LP**: no way to re-plan from the actual
   state of charge. This breaks SOC continuity across days and makes
   `rolling_mpc` re-plan from a wrong state every step.

4. **Open-loop daily execution**: one forecast at midnight, one LP solve, the
   whole day executed blind. This is not MPC. All intraday information —
   observed prices, demand, the model's own error so far — is discarded.

5. **Embargo applied to a trading simulation**: the runner trains and builds
   features only on data ending `embargo = 288` intervals (24 h) before the
   forecast origin. Embargo is backtest hygiene for *model selection* — in a
   sequential trading sim there is no leakage in using all data up to the
   origin, because lag features are strictly backward-looking. The current
   setup blinds the model to the most recent 24 h — exactly the data its
   recency features (Entry 010) feed on. In deployment you would obviously
   know yesterday's prices. **Sim embargo should be 0**; keep embargo in the
   rolling-origin *backtest* used for hyperparameter selection.

6. **No oracle, no capture ratio anywhere in the sim pipeline.**
   `dispatch.capture_ratio` exists but nothing calls it.

Consequence: Phase 0 (fix the scoreboard) precedes all modelling work. Any
capture-ratio result produced before Phase 0 merges is void.

---

## 3. Theory of the problem — where capture ratio is won and lost

This section is the reasoning that orders the phases. Workers: read it so you
don't optimise the wrong thing.

### 3.1 Revenue is spike-concentrated

Battery arbitrage revenue in the NEM concentrates in a handful of extreme
days. First Phase-0 analytic: compute the oracle's daily revenue and report
what fraction comes from the top 10 days (expect > 40%). This number tells
you the variance of the capture ratio itself: if 10 days carry the P&L, a
technique's measured capture on 122 days has wide error bars. Corollary:
never conclude from a single spike day; always look at the regret
decomposition across all spike days.

### 3.2 What the LP actually needs from the forecast

The dispatch LP has a **linear** objective. Three consequences, each of which
saves us from a tempting mistake:

- **It needs the conditional mean, not the median.** NEM prices are severely
  right-skewed. A pinball-α=0.5 (median) forecast systematically undervalues
  spike-prone intervals — the median of a distribution with a 5% chance of
  $10,000 is boring; the mean is not. Our current models predict the median
  of asinh-price and invert — two compounding downward biases (median <
  mean under right skew; inverse-transform of a central estimate ignores
  Jensen). The battery holds energy for spikes precisely in proportion to
  the mean. Fix: forecast a quantile set and integrate to a dollar-space
  mean (§5 W2.1).

- **Scenario optimisation without recourse degenerates.** For a linear
  objective and a fixed (open-loop) schedule, maximising expected revenue
  over N price scenarios is *identical* to optimising against the mean
  price path. Copula scenario machinery adds nothing in that setting. The
  value of scenarios appears only with (a) multi-stage recourse — which we
  approximate much more cheaply by frequent re-solving (MPC), (b) risk
  shaping (CVaR against missing the day's spike), or (c) nonlinear settings
  (price-maker, FCAS co-optimisation). **Do not build the copula/stochastic
  MPC layer before MPC + mean-forecasting are in and measured.** Expected
  standalone gain in price-taker mode with hourly re-solve: small (1–3
  points). It is Phase 4, not Phase 2.

- **What matters is intraday price *ranking*, magnitude second.** The LP
  chooses *when* to cycle within the horizon. A forecast that ranks the
  evening peak above the afternoon peak beats a lower-MAE forecast that
  misranks them (proven empirically by Entry 004). Metrics to track per
  trial: rank correlation (Spearman) between forecast and actual within each
  day, and hit-rate on "was the true top-12-interval window inside the
  forecast's top-24".

### 3.3 Forecast skill vs lead time is the MPC argument

At midnight, tonight's spike is genuinely uncertain — it depends on the
evening demand ramp, solar drop-off, and unit outages that reveal themselves
during the day. By 16:00, most of that evidence is on the table, and
short-lead forecasts (1–4 h) are far more skilful than 18-h-ahead ones. An
open-loop midnight schedule converts *only* midnight skill into money; a
receding-horizon loop converts the whole skill-vs-lead-time curve. This is
the single largest expected gain on the board (§5 W1.x) and requires no new
model — the same forecaster, re-queried with fresh features.

Mechanism detail that makes this work: the fitted model is reused; only the
*features* are recomputed on data up to the current interval, and the LP is
re-solved from the current SOC. Model refit stays weekly.

### 3.4 Feature ceiling and what actually predicts SA1 spikes

Entry 010 showed features >> architecture. Current feature set: price lags,
rolling stats, demand, momentum, calendar, intraday profile. Known missing
signals, in rough order of expected value for *spike ranking*:

1. **AEMO pre-dispatch / P5MIN forecasts (via NEMSEER)** — embeds the bid
   stack, planned outages, and interconnector limits; information we cannot
   reconstruct from price history. Also doubles as the mandated benchmark.
2. **Demand *forecast*** (pre-dispatch demand), not just trailing demand.
3. **Rooftop PV / large solar output** — SA duck curve; the evening ramp is
   the spike engine.
4. **Interconnector flows/limits (Heywood)** — SA islanding risk = extreme
   prices.
5. Temperature (ERA5 already cached per CLAUDE.md map for the main package).

These require data plumbing (NEMSEER/NEMOSIS pulls, cached per determinism
rules). That's Phase 3 — after MPC and mean-forecasting, because those two
multiply the value of every feature added later.

### 3.5 Decision-focused learning is last, and why

DFL (SPO+, cvxpylayers) retrains the forecaster on decision regret. It is
the right idea and the most expensive to build and debug (piecewise-constant
LP solutions → vanishing gradients → QP smoothing or surrogate losses). Its
gain is bounded by how much the *loss function* — not the information set —
is the binding constraint. After W2.1 (mean-space forecasting) much of the
loss-mismatch is already gone. Sequence it after Phases 1–3 have been
measured; implement SPO+ first (simpler: linear surrogate, no diff-through-LP),
cvxpylayers-QP second. Realistic expectation: single-digit points.

### 3.6 Expected trajectory (honest estimates, to be falsified)

| After phase | Expected capture (test) | Basis |
|---|---|---|
| P0 fixed scoreboard, open-loop naive | 0.20–0.40 | naive repeats last week's shape; open-loop wastes intraday info. **Actual: 0.347 test / 0.451 val.** |
| P0, open-loop lightgbm_rich | 0.30–0.50 | Entry 010 relative result, honest physics unknown. **Actual: 0.301 test / 0.389 val — below naive; median forecasts starve the LP of spike magnitude (T3 confirmed). AR is the best open-loop at 0.402 test / 0.473 val.** |
| P1 MPC (hourly re-solve, fresh features) | +0.10–0.20 over open-loop | lead-time skill curve, §3.3 |
| P2 mean-space quantile forecast | +0.03–0.08 | spike-day mean vs median gap |
| P3 exogenous features (NEMSEER etc.) | +0.03–0.10 | new information, mostly on spike days |
| P4 CVaR/scenario MPC | +0.01–0.03 | recourse mostly captured by P1 |
| P5 DFL fine-tune | +0.02–0.05 | residual loss mismatch |

If P0+P1+P2 don't clear 0.50 on validation, stop and diagnose with the regret
decomposition before adding machinery — the plan says where the money is
expected; the regret table says where it actually is.

---

## 4. Architecture decisions (for implementers)

- **`dispatch.schedule()` gains parameters** `dt_hours`, `soc0`,
  `terminal_soc` (None = free), `cycle_budget_mwh` (replaces max_cycles
  semantics at call sites that need partial-day budgets). Backwards-compatible
  defaults (`dt_hours=0.5`, `soc0=0`) so notebooks keep working.
- **Solver: swap cvxpy for `scipy.optimize.linprog(method="highs")`** in the
  hot path. The LP is small (3T vars, ~2900 for a 24-h horizon at 5-min) but
  MPC solves it thousands of times; HiGHS is ~10–50× faster than
  cvxpy+ECOS with zero modelling loss. Keep the cvxpy version as the readable
  reference implementation; unit-test that both give the same objective on
  random price vectors (tolerance 1e-4 relative). New module:
  `src/grian/sim/lp.py`.
- **Oracle in `src/grian/sim/oracle.py`**: full-window HiGHS LP with per-day
  cycle constraints (§1.3). Cache the result to
  `outputs/trials/_oracle/SA1/oracle.parquet` (schedule + daily revenue) —
  it never changes while the data and battery spec are fixed.
- **MPC executor in `src/grian/sim/mpc.py`**, a new dispatch loop that
  *replaces* `runner.simulate_region`'s per-day loop for MPC trials (keep the
  old path for open-loop ablations). Config keys:
  `mpc: {resolve_every: 6, reforecast_every: 12, horizon: 288, persistence_blend: 0}`.
  State carried across the whole window: `soc`, per-day discharge throughput.
  Cycle constraint inside the horizon is per-calendar-day: first partial day
  gets the *remaining* budget, subsequent days a full budget.
- **Feasibility clamp** is one shared function (`lp.clamp_action`) used by
  every executor, and the ledger asserts `0 ≤ soc ≤ capacity` and per-day
  throughput ≤ budget on every record.
- **Capture ratio wiring**: `ledger.summarise` stays pure; a new
  `analytics.capture_report(ledger_df, oracle_daily)` joins realised vs oracle
  daily revenue and emits capture ratio, regret series, top-regret table.
- Models are reused untouched: registry interface
  `fit(train_df, target_col, cfg) -> state`, `predict(state, data, horizon)`
  forecasts the next `horizon` intervals after the end of `data`. MPC calls
  `predict` with `data` = everything observed up to now. **Verify per model**
  that predict really keys off the end of `data` (W1.1 acceptance test).

---

## 5. Ticket board

Execute in order within a phase; phases are strictly ordered (P0 → P1 → P2 →
P3 → …). Every ticket ends with: tests green (`pytest -q`), `ruff check .`
clean, results appended to `outputs/experiment_log.md` (success or failure —
log rule is mandatory), and a commit.

### Phase 0 — Fix the scoreboard (void → trustworthy)

- **W0.1 — LP correctness.** Add `dt_hours`, `soc0`, `terminal_soc`,
  `cycle_budget_mwh` to `dispatch.schedule`; write `sim/lp.py` with the HiGHS
  implementation + `clamp_action`. Tests: HiGHS ≡ cvxpy objective on 20
  random price vectors (both dt=0.5 and dt=1/12); a hand-checkable 4-interval
  case; SOC never violated; zero-price vector → zero action.
- **W0.2 — Honest executor.** Fix `runner.battery_dispatch`: correct dt from
  `cfg["resolution"]`, clamp actions with `lp.clamp_action`, carry SOC across
  days (runner owns the state, passes `soc0` in), enforce daily cycle budget
  at execution. Test: inject an infeasible schedule and assert the ledger
  never monetises phantom energy; multi-day sim conserves energy.
- **W0.3 — Oracle + capture report.** `sim/oracle.py` (§1.3, cached),
  `analytics.capture_report`. Tests: replaying the oracle's own schedule
  through the honest executor reproduces its revenue (capture = 1.000 ± 1e-6);
  per-day vs full-window oracle delta reported once in the log.
- **W0.4 — Re-baseline.** Set `embargo: 0` for sim trials (keep the
  backtest embargo untouched — §2.5). Re-run `naive_similar_day`,
  `autoregression`, `lightgbm_rich` open-loop on **validation** and **test**;
  report capture ratios + spike-concentration analytic (§3.1). This is the
  honest baseline table. Log entry mandatory, including the old-vs-new
  revenue comparison so the dt bug's effect is documented (Entry 013+).

### Phase 1 — Receding-horizon MPC

- **W1.1 — Predict-from-now verification.** For each registry model, test
  that `predict(state, data_up_to_t, h)` forecasts t+1…t+h (not
  midnight-anchored). Fix or wrap models that assume day boundaries
  (suspects: `lightgbm_rich`'s per-step calendar features — verify the step
  index is relative, not hour-of-day-anchored; the naive model likely needs
  an offset wrapper).
- **W1.2 — MPC executor** (`sim/mpc.py`, spec in §4). Start with
  `resolve_every=6` (30 min), `reforecast_every=12` (1 h). Runtime guard:
  full 122-day run must stay under ~30 min wall; if not, profile before
  optimising blindly (HiGHS should make LP time negligible; feature
  recomputation is the likely hotspot — build features incrementally or
  vectorise the per-origin tail).
- **W1.3 — MPC vs open-loop.** Same models as W0.4, MPC mode, validation
  first. Ablations: resolve_every ∈ {1, 6, 36}, reforecast_every ∈ {12, 288}
  — this isolates "re-solving from true SOC" vs "fresh information", which
  the write-up must separate. Then one test-window run for the headline.
- **W1.4 — Persistence blend (cheap short-lead skill).** For lead times
  < 1 h, blend forecast toward last observed price with exponentially
  decaying weight; tune the decay on validation only. Small ticket, possibly
  free points, possibly nothing — log either way.

### Phase 2 — Forecast the mean, in dollars

- **W2.1 — Quantile set → dollar mean.** Extend `lightgbm_rich` to fit
  quantiles {0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99} (one booster set
  per quantile, reuse the stride trick from Entry 001; consider
  `n_estimators=150` per quantile for runtime). Invert asinh per quantile
  (monotone transform ⇒ quantiles commute with inversion — this is exactly
  why we integrate in dollar space), enforce non-crossing by sorting, then
  integrate: mean ≈ Σ wᵢ·qᵢ with trapezoid weights, plus a tail term for
  q>0.99 (fit a constant tail multiplier on validation). Dispatch on this
  mean. Compare against dispatching on the median — the delta is the
  Jensen/skew effect, log it.
- **W2.2 — Spike-ranking diagnostics.** Add Spearman-per-day and
  top-window hit-rate (§3.2) to `capture_report`. No modelling — pure
  measurement, feeds every later phase.

### Phase 3 — Exogenous information

- **W3.1 — NEMSEER pre-dispatch plumbing.** Pull P5MIN and 30-min
  pre-dispatch price + demand forecasts for SA1 over 2023-01→2024-01, cache
  to `data/raw/nemseer/` (determinism rules: cache, never re-pull). Align
  lead times carefully: at sim time t, only pre-dispatch runs *issued* ≤ t
  are visible. This is the leakage hot-zone of the whole campaign — see trap
  T6.
- **W3.2 — AEMO-as-feature + AEMO-as-benchmark.** (a) Add latest-visible
  pre-dispatch price for each horizon interval as a feature; (b) run the
  mandated benchmark: MPC dispatching directly on the AEMO pre-dispatch
  forecast. The benchmark is a headline table row forever after.
- **W3.3 — Solar/rooftop PV + interconnector features** via NEMOSIS
  (DISPATCHREGIONSUM, rooftop PV actuals), same caching rules. Measure via
  validation capture, keep only features that move it.

### Phase 4 — Risk-aware MPC (only after P1–P3 measured)

- **W4.1 — CVaR-flavoured dispatch.** Quantile forecasts already exist
  (W2.1). Implement either (a) chance-constrained charging (don't be empty
  when q95 of the next 4 h exceeds a threshold) or (b) two-point scenario LP
  {mean path, spike path} with probability weights. Keep it small; measure;
  the §3.2 degeneracy argument predicts modest gains — falsify it.
- **W4.2 — Copula scenarios** only if W4.1 shows risk shaping is binding.

### Phase 5 — Decision-focused learning

- **W5.1 — SPO+ linear correction head** on frozen GBT quantile features:
  train a linear layer mapping quantile vector → dispatch-price vector with
  SPO+ loss against realised prices, validation-early-stopped on capture.
  No diff-through-LP needed.
- **W5.2 — cvxpylayers QP-smoothed end-to-end** only if W5.1 shows a real
  gradient signal. Budget-box this: it's the most likely time sink on the
  board.

### Phase 6 — out of headline scope (parking lot)

FCAS co-optimisation, nempy price-maker re-clearing, 10-band offer
construction, Mannum actual-dispatch anchor validation. Each changes the
metric definition (new oracle needed) — do not mix into the 0.50 campaign.

---

## 6. Experimental discipline

1. **Validation-first.** Every technique is tuned and accepted/rejected on
   the validation window. One test-window run per accepted technique, for
   the headline table. No exceptions, no "quick look".
2. **Every result → experiment log** (`outputs/experiment_log.md`), including
   failures, with the standing template. Failures are curriculum content.
3. **Ablation verification** (Entry 002's lesson): after wiring any config
   flag, prove the output changes. Two identical result files = a bug or an
   uninformative ablation; both block merging.
4. **Headline table** lives at `outputs/plans/capture_campaign_results.md`:
   one row per (model, executor, window), columns: capture, revenue, oracle
   revenue, Spearman-rank, top-10-regret share, runtime, git SHA. Append-only.
5. **Determinism**: seeds from config, cached data only, `git_sha` in every
   trial config (already implemented — check it when comparing).
6. **Runtime budget**: a validation run should stay ≤ ~15 min, test ≤ ~30 min
   on the M-series laptop. If a ticket blows this, profiling comes before
   more compute.

## 7. Trap register

- **T1 — dt mismatch (the void-maker).** Any new executor/LP must take dt
  from config. Grep for `0.5` near dispatch code in review.
- **T2 — phantom energy.** Revenue may only ever be computed from *clamped*
  actions. The ledger asserts SOC bounds; never bypass it.
- **T3 — median vs mean.** Anything trained with pinball α=0.5 predicts the
  median. Don't feed medians to the LP and expect spike value (§3.2).
- **T4 — scenario-LP degeneracy.** Expected-value scenario optimisation with
  a linear objective and no recourse ≡ mean-path LP. Don't build it expecting
  otherwise (§3.2).
- **T5 — embargo confusion.** Sim embargo = 0 (deployment realism); backtest
  embargo = horizon (selection hygiene). They are different knobs; don't
  "fix" one to match the other.
- **T6 — issued-time leakage (Phase 3).** Pre-dispatch forecasts must be
  joined on *issue time ≤ now*, not on target time. AEMO files index by
  target interval; the naive join is silently leaky and will look brilliant.
  Test: a feature-leakage ablation that joins naively must score
  suspiciously better — if it doesn't, the join is wrong in both arms
  (Entry 007's lesson).
- **T7 — capture > 1 or negative oracle days.** Assert capture ≤ 1.0 +ε.
  Oracle daily revenue < 0 is impossible (doing nothing earns 0) — assert.
- **T8 — quantile crossing.** Sort quantiles after inversion before
  integrating (W2.1).
- **T9 — solver failure handling.** On LP failure, hold (no action), log a
  counter; > 0.1% failure rate = investigate, don't ignore.
- **T10 — test-set erosion.** The 122-day window's P&L rests on ~10 days
  (§3.1). Repeated peeking overfits to those specific spikes faster than
  classical intuition suggests. Validation-first is load-bearing.
- **T11 — MPS/OOM and libomp.** Torch models: batch-wise device transfer
  only (Entry 011). LightGBM on Apple Silicon: arm64 libomp fix in memory
  (`reference-libomp-fix`).

## 8. Status ledger (update as tickets land)

| Ticket | Status | Result / log entry |
|---|---|---|
| W0.1 | done 2026-07-11 | `sim/lp.py` HiGHS LP + `clamp_action`; cvxpy `schedule` gained dt/soc0/terminal params; equivalence tests in `test_sim_lp.py`. Log Entry 013. |
| W0.2 | done 2026-07-11 | Honest executor: clamped actions, SOC carried across days, daily budget at execution. |
| W0.3 | done 2026-07-11 | `sim/oracle.py` (weekly blocks — full-window LP scales superlinearly, see §1.3 fallback) + `analytics.capture_report`; oracle-replay capture = 1.0 test green. |
| W0.4 | done 2026-07-11 | Log Entry 015. Open-loop: naive 0.451 val / 0.347 test; AR 0.473 / 0.402; lgbm_rich 0.389 / 0.301. Oracle $11.99M val (top-10 share 51.7%), $7.71M test (30.7%). Headline anomaly: lgbm_rich has best test Spearman (0.688) and worst capture — trap T3 empirically confirmed. |
| W1.1 | done 2026-07-11 | Predict-from-now fixed in naive + lightgbm_rich (both ignored `input_df`; forecasts were up to 7 days stale between refits — Log Entry 014). AR/MLP/LSTM still unverified for MPC use. |
| W1.2 | done 2026-07-11 | `sim/mpc.py`; perfect-forecaster-through-MPC captures 1.0 ± 0.02 (end-to-end executor proof). |
| W1.3 | done 2026-07-12 | Log Entry 016. **lightgbm_rich + MPC: 0.536 val / 0.509 test — 0.50 target cleared on both windows.** Naive under MPC *drops* (0.382 val): recourse amplifies short-lead skill, including zero skill into a deficit. W1.3 frequency ablations (resolve/reforecast grid) still open. |
| W1.4 | pending | Persistence blend — likely subsumed by hourly reforecasting; low priority. |
| W2.1 | done 2026-07-12 | Log Entry 017. Open-loop: 0.407 val / 0.443 test (vs 0.389/0.301 median — big open-loop win). Under MPC: 0.520 val < 0.536 → rejected for the MPC stack. Caveat: 150 trees/booster vs 300, crude 0.98 tail — re-run at matched capacity before final judgement. |

**Campaign headline (2026-07-12): lightgbm_rich + mpc-30m = 0.536 val / 0.509 test — the 0.50 objective is met on both windows.** Next levers toward 0.65, in expected-value order: W1.3 frequency ablations (resolve_every=1 may add 1–3 pts), W2.1 matched-capacity re-run, then Phase 3 (NEMSEER pre-dispatch features — the largest untapped information source, plus the mandated AEMO benchmark).
