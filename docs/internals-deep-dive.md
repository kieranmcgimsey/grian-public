# grian — internals deep dive

*A friendly, from-first-principles walk through the entire grian project: what it
does, how every piece works, the maths underneath each method, and the exact recipe
used to train each model. A single-document way to get properly up to speed on the
codebase internals. It assumes you can read Python and remember high-school algebra —
nothing more. Every piece of jargon is defined the first time it appears.*

**How to read this.** Go top to bottom once, slowly — each chapter builds on the last.
The early chapters set up the problem and the vocabulary; the middle chapters teach the
forecasting; the later chapters teach the trading and the maths of deciding under
uncertainty. The two appendices (a self-test and a "numbers to memorise" sheet) are for
quick revision once you've read the rest. Throughout, file pointers like `sim/models.py` tell you where to read
the real thing.

A note on units: prices are in **dollars per megawatt-hour** (\$/MWh), energy in
megawatt-hours (MWh), power in megawatts (MW), and $\Delta t$ ("delta-t") is the length
of one time interval measured in hours — $1/12$ of an hour for 5-minute data, $0.5$ for
30-minute data.

---

## Chapter 1 — The problem, in plain words

### 1.1 What is actually going on

Australia runs a wholesale electricity market called the **NEM** (National Electricity
Market). Electricity can't easily be stored at grid scale, so supply and demand must
match instant by instant, and the *price* of electricity is recalculated constantly —
every **5 minutes** — for each region of the grid. grian focuses on one region: **SA1**,
which is South Australia. South Australia is a good test case because it has a lot of
wind and solar, which makes its prices wild: they swing negative when the sun floods the
grid with cheap power, and they **spike** violently — up to the market's legal ceiling of
about **\$16,000–17,000/MWh** — when supply gets tight. (A "spike," here, just means a
brief period of extremely high price.)

Now imagine you own a big battery: **100 MW** of power (how fast it can charge or
discharge) and **200 MWh** of energy storage (how much it can hold) — a "2-hour" battery,
because at full power it takes 2 hours to fill or empty. A battery in an electricity market
is a money machine of one specific kind: it does **arbitrage**. *Arbitrage* means buying
something cheap and selling it dear. The battery **charges** (buys energy from the grid)
when prices are low and **discharges** (sells energy back) when prices are high. The whole
game is deciding *when* to charge and *when* to discharge. That decision is called
**dispatch** — literally, dispatching the battery to charge or discharge.

### 1.2 Why we need a forecast

Here's the catch: to decide whether to charge *now*, you need to know whether prices later
today will be high enough to make it worth it. You don't know the future, so you have to
**forecast** it — predict the price for each of the next several hours. grian forecasts a
full **day ahead**: the next 288 five-minute prices (or 48 half-hour prices). So the project
has two halves that chain together:

1. **Forecasting** — predict tomorrow's price curve from data available today.
2. **Dispatch** — turn that forecast into charge/discharge decisions for the battery.

A forecast is only useful insofar as it leads to *good trades*. This is the single most
important idea in the whole project, and we'll return to it many times: **a forecast that
looks accurate on paper can still trade badly, and vice versa.**

### 1.3 How we keep score: the capture ratio and the oracle

To measure success we need a yardstick. The perfect yardstick is: *how much money could a
battery have made if it knew the future exactly?* We compute that by pretending we have a
crystal ball — we take the **actual** prices that happened and find the single best
possible charge/discharge schedule for them. That best-possible schedule is called the
**oracle** (as in "an oracle that knows the future"). The oracle's revenue is the most any
battery could ever have earned. It's a fixed number for a given stretch of history, so we
call it the **frozen denominator**.

Our headline metric is the **capture ratio**:

$$\text{capture ratio} = \frac{\text{revenue our strategy actually earned}}{\text{revenue the perfect-foresight oracle earned}} \in [0, 1].$$

A capture ratio of $0.55$ means our forecast-driven battery captured 55% of the
theoretically perfect profit. It can't exceed 1 (you can't beat perfect foresight), and a
do-nothing battery scores 0. Because the number is a *fraction of the same oracle*, we can
compare wildly different strategies on one honest scale.

One technical wrinkle worth stating up front, because it quietly trips people up: AEMO (the
market operator) labels each price with the time the interval *ends* — a price stamped
14:05 is really the price *for* the five minutes from 14:00 to 14:05. On load, grian
**shifts every timestamp back one interval** so labels mean "interval start"
(`data.py`, done once, documented once). Get this wrong and every lag and every label is off
by one — a subtle bug that would quietly corrupt everything downstream.

**The champion result to remember:** grian's best system captures about **0.55** of the
oracle — specifically **0.554 balanced / 0.585 pooled** (we'll define those two flavours of
capture in Chapter 10). The earlier headline was **0.546 / 0.562**. And the single most
surprising finding — the one that reshaped the whole project — is that the *most accurate*
forecaster turned out to be the *worst trader*. Chapter 10 explains why.

---

## Chapter 2 — A tour of the codebase

grian is one Python library (`src/grian/`) with two audiences:

- **The curriculum** — ten teaching notebooks (`notebooks/01`–`10`) that walk from raw data
  to a battery earning money, plus the library modules they lean on. The notebooks are
  deliberately *thin*: all the real logic lives in the library, and the notebooks import it,
  narrate it, and draw pictures.
- **The simulation environment** (`src/grian/sim/`) — a serious, walk-forward battery-trading
  test bench. This is where the "capture-ratio campaign" (the research effort that pushed the
  number from a fictional score to a real 0.55) actually runs. When this document talks about
  models, executors, and the oracle, it almost always means the code in `sim/`.

You don't need to memorise the directory tree, but it helps to know the cast of characters in
`sim/`. Every one of these is a plain module of functions and dictionaries — no elaborate
class hierarchies, by deliberate design (the code style is "functional, readable, academic
test bench").

| Module | Its job, in one sentence |
|---|---|
| `models.py` | The model registry — each forecasting model is just a dictionary of four functions (`fit`, `predict`, `save`, `load`). |
| `features.py` | Turns raw price/demand/weather into the columns the models learn from. |
| `lp.py` | The battery optimiser — given a price forecast, it computes the best charge/discharge plan. |
| `oracle.py` | The perfect-foresight benchmark (the frozen denominator). |
| `runner.py` | The simple "open-loop" trading loop: once a day, forecast, plan, execute. |
| `mpc.py` | The smarter "closed-loop" trading loop that re-plans as the day unfolds. |
| `ledger.py` | An append-only record of every trade, plus profit/risk summaries. |
| `analytics.py` | Scoring: capture ratio, skill vs the naive baseline, worst-day analysis. |
| `trials.py` | Bookkeeping: config files, the price transform, saving every run reproducibly. |
| `dispatch_prob.py` | The "probabilistic" dispatch helpers that trade off risk. |
| `ablations.py` | Deliberately broken versions used to *prove* that each safeguard matters. |
| `dashboard.py` | Builds a static HTML dashboard to compare runs. |

The chapters that follow open up each of these in turn.

---

## Chapter 3 — The data and the target variable

### 3.1 Where the data comes from

Three sources, all cached to disk and never re-downloaded (re-pulling would break
reproducibility — you'd silently get different data on a different day):

- **NEMOSIS** — a public tool that pulls AEMO's archive of *actual* historical prices and
  demand. This is our ground truth.
- **NEMSEER** — AEMO's *pre-dispatch* forecasts (the market operator's own published guess of
  near-future prices). Useful as an "honest benchmark" to beat.
- **ERA5** — a global weather *reanalysis* dataset (a best-estimate reconstruction of past
  weather). grian uses three of its variables: `ssrd` (surface solar radiation — how sunny it
  was), `t2m` (air temperature 2 m above ground), and `wind_speed`. ERA5 comes in UTC time,
  so grian shifts it to Australian Eastern time (+10 hours) — after which solar radiation
  peaks around local 1 pm, as it should.

### 3.2 Five-minute vs thirty-minute

The market settles every 5 minutes, giving 288 intervals per day. Running everything at
5-minute resolution is slow, so the project **pivoted to 30-minute data** (48 intervals/day)
by averaging each half-hour ("mean-resampling"). This is a genuine trade-off: 30-minute data
blurs the very sharpest spikes, but it makes both the optimiser and the model re-fitting about
six times cheaper, which is what let the research move quickly. The main campaign evaluation
runs at 30 minutes.

### 3.3 The target transform: modelling asinh(price)

We don't ask the model to predict the raw price. We ask it to predict a *transformed* price,
and we transform back to dollars before scoring. The transform grian uses is the **inverse
hyperbolic sine**, written **asinh** (or $\sinh^{-1}$). Its formula and its inverse are:

$$\mathrm{asinh}(p) = \ln\!\Big(p + \sqrt{p^2 + 1}\Big), \qquad \sinh(x) = \frac{e^x - e^{-x}}{2} = \mathrm{asinh}^{-1}(x).$$

Why do this at all? Two reasons, and it's worth understanding both because the choice comes up
constantly.

- **Reason 1 — tame the spikes (variance stabilisation).** Electricity prices spend most of
  their time around \$50–100 but occasionally hit \$15,000. If we fed raw prices to a model,
  the loss function (the thing the model tries to minimise) would be utterly dominated by a
  handful of enormous spike values, and the model would neglect the ordinary days. asinh
  *compresses* large values: for big $|p|$ it behaves like $\mathrm{sign}(p)\cdot\ln(2|p|)$,
  a logarithm, which squashes \$15,000 and \$150 much closer together. This is called
  *variance stabilisation* — making the spread of the data more uniform so the model can learn
  from all of it.

- **Reason 2 — handle negative prices.** The obvious spike-taming transform is the logarithm,
  but $\ln$ (and even $\log(1+p)$) is undefined for negative numbers — and electricity prices
  *go negative*. asinh is defined and *smooth* for all real numbers, positive and negative, and
  passes cleanly through zero. That's exactly what a market with negative prices needs.

The rule that goes with the transform: **always invert before scoring**, so every error you
report is in real dollars, not in "asinh units." In the code, `_get_transform_pair` in
`trials.py` returns the pair `(np.arcsinh, np.sinh)`; there are also `log1p` and `identity`
options, but **asinh is the default**.

One subtle place this matters: the quantile models (Chapter 7) predict in asinh space,
invert *each* prediction to dollars, combine them into a dollar figure, and then
*re-apply* asinh so that the runner's standard "invert once at the end" step produces exactly
the right dollar number. It's a round-trip that keeps the accounting consistent.

### 3.4 A note on scaling

Related to transforms is **scaling** — rescaling each input feature to a common range.
Linear models and neural networks are sensitive to the scale of their inputs (a feature
measured in thousands will dominate one measured in fractions), so grian *standardises* their
inputs (subtract the mean, divide by the standard deviation). **Tree-based models don't need
this** — they only ever ask "is this feature above or below some threshold?", which is
unaffected by scale. This asymmetry explains a lot of the design later on.

---

## Chapter 4 — Feature engineering

### 4.1 What a "feature" is

A **feature** is one input column the model gets to look at when it makes a prediction. If you
want a model to predict the price at 6 pm, you might give it: the price at 6 pm yesterday, the
average price over the last hour, the current demand, whether it's a weekend, and so on. The
craft of choosing and computing these columns is **feature engineering**.

There is one iron rule: every feature must be **backward-looking** — computed only from
information available *at or before* the moment we forecast from. Using future information, even
by accident, is called **leakage**, and it makes a model look brilliant in testing and fail in
reality. In grian, every feature uses `shift(1)` or greater (look at least one interval into the
past) or a rolling window that excludes the present. All of this lives in `sim/features.py`, and
`build_features(...)` assembles the groups.

### 4.2 The feature groups

Here is the full menu. Read the "what it captures" column — that's the intuition worth
internalising.

| Group | Example columns | What it captures |
|---|---|---|
| **Price lags** | `price_lag_{1d, 2d, 7d}` | The autoregressive backbone: the price at this same time yesterday, two days ago, a week ago. ("Autoregressive" = predicting a series from its own past values.) |
| **Rolling statistics** | mean/std/min/max over 1 h, 6 h, 24 h | The recent *regime* — is the market calm or volatile right now? |
| **Demand** | demand now, its 6 h/24 h averages, its 24 h volatility | Demand is the physical driver of scarcity: high demand pushes prices up. |
| **Calendar** | `hour`, `day_of_week`, `month`, `is_weekend`, `hour × dow` | The repeating daily/weekly/seasonal shape of prices. (How we *encode* these is a whole topic — Chapter 5.) |
| **Momentum** | `price_ret_{1h,6h,24h}`, `price_direction_1h` | Trend and acceleration — is the price climbing or falling, and how fast? ("Return" = the percentage change over a window.) |
| **Intraday profile** | `intraday_profile_mean`, `intraday_deviation` | The average price *shape* over the day (low overnight, evening peak) from the last 7 days, and how today deviates from it. |
| **Scarcity** (optional) | `spikes_24h`, `intervals_since_spike`, `price_vol_ratio`, `demand_stress_max/p95`, `demand_ramp_1h`, `demand_accel` | *Precursors* of spikes: is the system running near its supply ceiling? Have spikes been clustering? |
| **Weather** (optional) | `wx_{ssrd,t2m,wind_speed}` plus 3 h averages and 1 h changes | Renewable supply (sun, wind) and temperature-driven demand. |

### 4.3 "Full" vs "lean": a cautionary tale about features

There are two feature sets. The default, `"full"`, includes everything above. The alternative,
`"lean"`, uses *only* raw price lags aligned to whole days (the price at this same interval 1,
2, 3, 7, and 14 days ago, plus one interval either side of yesterday) and the calendar. Nothing
else — no rolling averages, no momentum, no intraday smoother.

Why would you ever throw features away? Because of a real finding (logged as Entry 032 in the
experiment log). The "full" set is full of *mean-reverting smoothers* — rolling averages that,
by construction, pull toward the recent average. A linear model leans on these and on the most
recent price so heavily that its forecast becomes a nearly flat line — **54% of the far-horizon
weight landed on the single most recent lag**. A flat forecast has no peaks, and (as we'll see)
**the peaks are exactly where a battery makes its money.** The "lean" set is the classic
electricity-price-forecasting basis (from the academic work of Lago and colleagues): give a
sparse linear model whole-day-aligned lags and let it reconstruct the daily shape, rather than
handing it a smoother that flattens everything.

### 4.4 The heavy-tailed momentum gotcha

Momentum features are *returns* — percentage changes. Percentages have a nasty property: if the
denominator (the earlier price) is tiny, the percentage explodes. At a price spike this happens,
and a return can reach **~200 standard deviations even after standardisation**. For a
tree model this is harmless (trees only compare thresholds). But for a model trained by gradient
descent (Chapter 6), a single input that large wrecks the numerical conditioning and the training
*diverges* — the loss grows instead of shrinks. The fix is **winsorising**: clipping every
standardised feature to a maximum of ±5 standard deviations, so no single outlier can dominate.
(Entry 037 — this bit us and is worth remembering.)

### 4.5 Findings you should be able to recite

- **Weather is not a free win.** It *hurts* the tree models, *helps* the linear model, and helps
  the quantile tree model **only when combined with Fourier calendar features** (Chapter 5).
  The lesson: adding a feature can *increase* error by adding variance (more ways to overfit).
  Also, grian's weather is measured *at the forecast moment*, not a true day-ahead weather
  forecast — which would need forecast data (NEMSEER) and is left as future work.
- **Scarcity precursors are real but didn't help the forecaster.** The signals genuinely predict
  spikes — on SA1 the chance of a spike in the next interval rises from a **3.5% baseline to
  ~17%** when demand is within 5% of its recent peak, and similarly after a recent cluster of
  spikes. Yet adding them to the forecaster **did not improve capture** (Entry 030/031). This is
  a big theme: the lever that mattered turned out to be the *trading objective*, not the
  features. Hold that thought for Chapter 13.

---

## Chapter 5 — Calendar encodings (deeper than it looks)

Prices follow the clock and the calendar: cheap overnight, an evening peak, different on
weekends, seasonal over the year. The question is how to *feed* "hour of day" to a model. There
are three ways, and choosing wrong quietly damages the model.

- **Ordinal** — just the integer: hour = 0, 1, 2, …, 23. This is **correct for trees**, because
  a tree only ever asks "is hour ≥ 17?" — it never does arithmetic on the number. But it is
  **wrong for linear models**, which *do* arithmetic: to a linear model, "hour 23" is 23× "hour
  1," and hour 23 and hour 0 look maximally far apart even though they're adjacent on the clock.
- **One-hot** — one yes/no column per category (a column "is it hour 17?", another "is it hour
  18?", …). This is the safe **default for linear models**: no false ordering. The cost is many
  columns, and it still doesn't know that hour 23 and hour 0 are neighbours.
- **Fourier** — encode each cyclic quantity as a few sine and cosine waves, so the encoding
  literally wraps around: hour 23.5 and hour 0 sit right next to each other. For a quantity $x$
  with period $P$ (24 for hours, 7 for days, 12 for months), the harmonics are

  $$\phi_k(x) = \Big(\sin\tfrac{2\pi k x}{P},\ \cos\tfrac{2\pi k x}{P}\Big), \quad k = 1, 2, \ldots, H.$$

  grian uses $H = 3$ harmonics for hour (period 24), $H = 1$ for day-of-week (period 7), and
  $H = 2$ for month (period 12). Fourier is smooth and low-dimensional, and — importantly — the
  *same* spec is available to the tree models too, so a fair "Fourier vs ordinal" comparison can
  be run on both model types.

### The phase-regularisation "wart" — a genuinely instructive bug

This one rewards a careful explanation, because it's the kind of subtlety that separates people
who *use* methods from people who *understand* them.

Linear models like LEAR use **L1 regularisation** (also called **Lasso**). Regularisation means
adding a penalty for large coefficients, to prevent overfitting; the "L1" penalty specifically
adds up the *absolute values* of the coefficients, $\sum_j |w_j|$, which has the nice side effect
of pushing many coefficients to exactly zero (automatic feature selection).

Now, a Fourier feature is a *pair*: $a\sin(\cdot) + b\cos(\cdot)$. Together, $a$ and $b$ encode
both the **amplitude** (how big the daily swing is) and the **phase** (what time of day the peak
sits). The L1 penalty charges $|a| + |b|$. Here's the problem: $|a| + |b|$ is **not rotationally
invariant**. If you rotate the phase (slide the peak to a different hour), $|a| + |b|$ changes
even though the *size* of the wave didn't. So the penalty secretly prefers certain phases — the
ones aligned to the axes — and will happily zero out one of the pair, *forcing* the peak to a
convenient clock position rather than the true one. The visible symptom was "**Fourier hurts
LEAR**" (capture dropped from 0.484 to 0.426). That drop was an artifact of the *penalty*, not of
the encoding.

Why are other models immune?
- **Trees** have no coefficient penalty at all, so there's nothing to distort.
- **Ridge** regression uses the **L2** penalty $a^2 + b^2$, which equals the *squared amplitude* —
  a quantity that doesn't change when you rotate the phase. Ridge is phase-neutral.

The fixes, from most to least principled: (1) **group-Lasso** — penalise the pair *together* by
its combined length $\sqrt{a^2+b^2}$ (the amplitude), which is rotation-invariant; (2) simply
**exempt the whole calendar block** from the penalty (what grian's torch model does); (3) use L2
just for the calendar. Once the calendar is left unpenalised, **Fourier helps** (the quantile
LEAR rose 0.479 → 0.493).

A final subtlety that cost real debugging time: it must be the *entire* calendar block that's
exempt. Exempting only the sine/cosine columns while still penalising the one-hot dummies
squashed the hour-of-day pattern flat — the model's price fan spread collapsed to **\$5 instead
of \$416** — and a flat forecast, again, has no arbitrage and captures nothing. The relevant code
is `_linear_preprocessor`, `_leading_calendar_columns`, and `_l1_penalty` in `models.py`.

---

## Chapter 6 — The forecasting models and their exact recipes

### 6.1 The shared interface

Every model in grian is a plain dictionary with four (sometimes five) functions:

```python
{ "name": str, "output": "point" | "quantile",
  "fit":     fn(train_df, target_col, cfg) -> state,   # learn from history
  "predict": fn(state, input_df, horizon)  -> Series,  # a single price path
  "predict_fan": fn(...) -> {tau: array},              # (quantile models) a whole distribution
  "save" / "load": ... }
```

"State" is whatever the model needs to remember — fitted weights, trained trees, or just a
reference to the recent data. The trading loop never looks inside; it just calls these functions.
Adding a new model means writing four functions, nothing more. (This dicts-and-functions style is
a deliberate choice: readable, hackable, no inheritance to trace through.)

Two ideas apply to *all* the learned models:

- **Direct multi-step forecasting.** To predict 48 steps ahead we fit a *separate* model for each
  step (a model for "1 step ahead," another for "2 steps ahead," …). The alternative — predict one
  step, feed it back in, predict the next — is called *iterative* or *recursive* forecasting, and
  it accumulates errors (each prediction is built on the last prediction's mistakes). Direct
  forecasting avoids that. For efficiency grian fits one model per *hour* and linearly interpolates
  the in-between steps.

- **The "predict-from-now" contract.** When the model predicts, it seeds its features from the
  *freshest* data the trading loop hands it — the actual prices right up to the moment we're
  forecasting from. This sounds obvious, but getting it wrong is a classic bug: a model that
  instead uses the data frozen at training time will produce the *identical* forecast every time
  we ask, no matter how much has happened since. Under the re-planning loop (MPC, Chapter 13) that
  makes the near-term forecast a flat, useless line. (This is Entry 038, the "AR flatline" bug.
  The old neural models never got this fix and are deprecated because of it.)

Now the models themselves, from simplest to most sophisticated.

### 6.2 The naive baseline: `naive_similar_day`

The floor that every real model must beat. It simply **repeats the price curve from the same day
of the week, one week ago** (7 days × intervals-per-day back). No learning at all. Because it just
looks up a past target, it produces the same forecast regardless of when you run it — so for the
*forecast*, open-loop and MPC are identical (though the resulting *trades* still differ, because
the battery's state differs). We use this baseline as the denominator when we compute **skill**
(Chapter 10). Never skip the naive baseline: if you can't beat "last week, same day," you have
nothing.

### 6.3 Autoregression (AR)

A plain linear regression on price lags (default: 1, 2, and 7 days back) plus the calendar,
predicted **recursively** (the one recursive model in the set). A stable AR model *mean-reverts* —
its forecasts decay toward the long-run average as the horizon grows, so far-ahead predictions go
flat. Simple, fast, interpretable, and a useful sanity anchor. Variants let you switch the
calendar encoding.

### 6.4 The LEAR family (regularised linear models)

**LEAR** stands for **Lasso-Estimated AutoRegressive**, and it is *the* standard academic
benchmark in electricity-price forecasting (from Lago et al.). In one line: a linear model on a
big set of lagged prices + calendar + optional weather, with **L1/Lasso regularisation** deciding
which lags matter. grian gives it the *exact same feature matrix* as the main tree model, so "LEAR
vs LightGBM" is a clean linear-vs-trees comparison on identical inputs.

The training recipe (`_linear_fit`, spec `LINEAR`):
- One pipeline **per horizon step** (direct forecasting).
- The pipeline first runs `_linear_preprocessor`, which one-hot- or Fourier-encodes the calendar
  and **standardises** the numeric features (with median imputation for gaps), then fits the
  estimator.
- The estimator is chosen by the model's name: `lear` → **`LassoCV`** (L1), `ridge` → **`RidgeCV`**
  (L2), `elasticnet` → **`ElasticNetCV`** (a mix of L1 and L2), otherwise plain least squares.
  The "CV" suffix means the regularisation strength is chosen automatically by **cross-validation**
  (trying several values and keeping the best). Exact settings:
  `LassoCV(cv=3, n_alphas=20, max_iter=2000, random_state=42)` — three-fold CV over 20 candidate
  penalty strengths; `RidgeCV(alphas=logspace(-3, 3, 20))`; `ElasticNetCV(l1_ratio=0.5, cv=3, …)`.
- Variants: `lear` (one-hot calendar), `lear_weather`, `lear_fourier`, `lear_weather_fourier`,
  and the `lean`-basis versions `lear_lean` / `ols_lean`.

**The finding that defines the project:** LEAR is the **most accurate** forecaster by error
metrics (its MAE-skill is about 0.63) but the **worst trader** (capture about 0.50). Chapter 10
explains *why* — briefly, its L1 penalty and smoothers flatten exactly the peaks a battery needs.

### 6.5 Gradient-boosted trees: `lightgbm_rich`

**LightGBM** is a fast implementation of **gradient-boosted decision trees**. Let's unpack that.
A *decision tree* splits data by asking yes/no questions ("is demand > 1400?"). *Boosting* builds
many small trees in sequence, where each new tree is trained to fix the errors the previous trees
made — the ensemble "boosts" itself toward accuracy. The "gradient" part means each tree fits the
gradient (the direction of steepest error reduction) of the loss. Trees shine here because they
naturally model **non-linear thresholds** — "when demand crosses this level, prices jump" — which
is exactly how scarcity spikes behave, and which a straight-line linear model can't represent.

The recipe (`_lgbm_rich_fit`, spec `LIGHTGBM_RICH`) — memorise these numbers, they come up:
`n_estimators = 300` (300 trees), `learning_rate = 0.05` (how much each tree contributes),
`num_leaves = 31`, `max_depth = -1` (unlimited depth), `min_child_samples = 20`,
`subsample = 0.8`, `colsample_bytree = 0.8` (each tree sees 80% of rows and 80% of columns, for
robustness). The training objective comes from the config: `pinball` sets LightGBM's objective to
`quantile` at `alpha = 0.5`, which is the **median** (predicting the median with pinball loss at
$\tau = 0.5$ is the same as minimising ½ × MAE); `huber` uses a robust squared-ish loss. Calendar
is ordinal by default. Variants add weather, Fourier, scarcity features, or the decision-focused
weighting of §6.7.

### 6.6 The quantile tree: `lightgbm_qmean` (the probabilistic tier)

So far every model predicts *one* number per interval — a **point forecast**. But we're uncertain,
and it's more useful to predict a *distribution*: not just "the price will be \$200" but "there's a
5% chance it's below \$40, a 50% chance below \$120, a 10% chance above \$800." We represent that
distribution with a handful of **quantiles**. (A *quantile* $Q_\tau$ is the value below which a
fraction $\tau$ of outcomes fall: the 0.9 quantile is the price we'd exceed only 10% of the time.)
The set of quantile predictions across the horizon is called a **fan** — plot them and they spread
out like a fan.

`lightgbm_qmean` fits **one tree model per (step, quantile)**. The default quantile levels are
$\tau \in \{0.05, 0.5, 0.9, 0.98\}$ — deliberately **asymmetric**, packed toward the high end,
because the upside spikes are the risk a battery most needs to see. The recipe mirrors the point
tree but with `n_estimators = 150` (fewer trees — there are far more models to fit), same tree
shape, `objective = "quantile"`, and crucially **`alpha = τ`** set per model so each one learns its
own quantile.

This model does double duty, which is where its name comes from — "**q**uantile" plus integrated
"**mean**":
- Its `predict_fan` returns the whole fan (all quantiles), which feeds the risk-aware dispatch of
  Chapter 13.
- Its `predict` collapses the fan to a single dollar mean (the maths is in Chapter 7). It even has
  a *lead-dependent* trick (`mean_from_step`): at very short horizons the future is nearly certain,
  so the **median** is the better point estimate; at long horizons the skew matters, so it uses the
  **mean**.

The overall **champion** model is `lightgbm_qmean_weather_fourier` — the quantile tree with both
weather and Fourier calendar features switched on.

### 6.7 The fast quantile-linear model: `lear_qmean_torch` (a war story)

Can we give the *linear* model a fan too, so a linear model can also drive risk-aware dispatch?
Yes — but the obvious way is painfully slow, and the story of fixing it teaches two lessons.

The obvious way (`lear_qmean`, now deprecated) fits scikit-learn's `QuantileRegressor` once per
(step, quantile). Each fit solves a small **linear program** (an optimisation, Chapter 12) via a
solver called HiGHS. There are ~100 of them per refit, and they run one after another: about **8
minutes per refit, and 10–15 hours to build one fan.** Unusable.

The fix (`lear_qmean_torch`) collapses all ~100 fits into **one**. Instead of 100 separate linear
regressions, it trains a *single* linear layer

$$W:\ \mathbb{R}^{n_\text{features}} \to \mathbb{R}^{n_\text{steps}\times n_\text{quantiles}}$$

that outputs *every* (step, quantile) at once, using **gradient descent** (specifically the Adam
optimiser in PyTorch) on the summed pinball loss. This drops the time to **~13 seconds per refit
(~3.5 minutes per fan) — about 40× faster.** The exact recipe: `epochs = 400`, `lr = 0.005`, L1
strength `alpha = 0.01`, features winsorised to `feature_clip = ±5σ`, target standardised once
(quantiles survive a linear rescaling, so we predict in standardised space and invert), and
predictions **clamped to the training range** so a wild extrapolation can't feed infinity into the
optimiser. The L1 penalty matches sklearn's but **exempts the whole calendar block** — the fix from
Chapter 5, which is *why* torch-Fourier helps where sklearn-Fourier hurt.

**Lesson one: the speed-up was the *batching*, not the hardware.** The plan was to run this on the
Mac's GPU (Apple's "MPS" backend). It was tried — and MPS computed **wrong gradients** here: the
loss *rose* every epoch and the fit diverged (pinball loss went 0.44 → 9.9), while the *identical
code and seed on the CPU converged cleanly* (0.44 → 0.15). The bug was isolated to the MPS autograd
backend, and the model is pinned to the CPU. **Lesson two: verify your numerics — an accelerator is
not free correctness.** A linear model is cheap enough on CPU anyway.

### 6.8 Decision-focused, scarcity, and calibrated variants

Three attempts to close the "accurate but poor trader" gap, each worth a sentence:
- **`lightgbm_rich_dfl`** ("decision-focused learning") — during training it **weights high-price
  intervals more heavily** (`_decision_weights`) so the model tries harder to nail the spikes that
  actually earn money, accepting slightly worse average accuracy. Two weighting schemes: a smooth
  `magnitude` ramp $w = 1 + \text{strength}\cdot\ln(1 + \max(0, p)/\text{scale})$ (scale = \$300),
  or a flat `quantile` boost above the 90th price percentile. Weights are computed in dollars and
  normalised to average 1 so the learning rate is unchanged.
- **`lightgbm_rich_scarcity`** — the scarcity features from §4. Alone, they *hurt* capture.
- **`lightgbm_qmean_cal`** — a *calibrated* fan (Chapter 8).

### 6.9 The deprecated neural models

A small feedforward network (`simple_mlp`) and an LSTM (`lstm`, a recurrent net for sequences)
exist in the code but are **deprecated**: never competitive, and carrying the frozen-tail bug (§6.1)
so they can't re-forecast under MPC. The honest lesson is *why* deep learning didn't win here —
**small data, a spiky target, and no obvious structural advantage** over gradient-boosted trees on
this kind of tabular data. Deep learning is not automatically better.

---

## Chapter 7 — Quantiles, the pinball loss, and the fan

We met quantiles in §6.6. This chapter gives them their maths, because the whole probabilistic and
risk-aware machinery rests on it.

### 7.1 The pinball (quantile) loss

To *train* a model to predict the $\tau$-quantile, we need a loss function whose best answer is
exactly that quantile. That function is the **pinball loss** (a.k.a. quantile loss). With error
$e = y - \hat{y}$ (actual minus predicted):

$$\rho_\tau(y, \hat{y}) = \max\big(\tau\, e,\ (\tau - 1)\, e\big) = \begin{cases} \tau\,(y - \hat{y}) & \text{if } y \ge \hat{y} \ \text{(we under-predicted)} \\ (1 - \tau)\,(\hat{y} - y) & \text{if } y < \hat{y} \ \text{(we over-predicted)} \end{cases}$$

The trick is the **asymmetry**. For the 0.9 quantile, under-predicting is penalised at weight 0.9
and over-predicting at only 0.1 — so the model is pushed to predict a *high* value, one it will
exceed only 10% of the time. That's precisely the definition of the 0.9 quantile. At $\tau = 0.5$
the two weights are equal and the loss becomes symmetric absolute error, whose best answer is the
median. The gradient is bounded (always $\tau$ or $\tau-1$), which makes it robust to spike
outliers — another reason it suits this domain. In the code: the torch model implements it directly
(`_pinball_loss`), LightGBM sets `objective="quantile", alpha=τ`, and sklearn uses
`QuantileRegressor`.

### 7.2 Turning the fan back into a mean

The dispatch optimiser needs a *single* expected price per interval, not a whole fan (its revenue
is linear in price, so it needs the expected value $\mathbb{E}[p]$). How do we get a mean from a
handful of quantiles? By integrating the quantile function — a standard identity says the mean is
the area under the quantile curve:

$$\mathbb{E}[X] = \int_0^1 Q(\tau)\, d\tau \approx \sum_i w_i\, Q(\tau_i).$$

grian approximates that integral with the **midpoint rule** (`_quantile_weights`): each quantile is
given the slice of probability lying between the midpoints of its neighbours. With edges at
$\big[0,\ \tfrac{\tau_1+\tau_2}{2},\ \tfrac{\tau_2+\tau_3}{2},\ \ldots,\ 1\big]$, the weight of
quantile $i$ is the width of its slice, $w_i = \text{edge}_{i+1} - \text{edge}_i$.

Here's why this matters and isn't just bookkeeping: because prices are **right-skewed** (a long
tail of high spikes), the mean sits *above* the median. If dispatch used the median it would
systematically under-price the spikes and sell into merely-okay intervals — "buy low, sell
mediocre." Using the integrated mean respects the skew.

### 7.3 Non-crossing

Because each quantile is fitted by a *separate* model, nothing stops the "0.9" prediction from
coming out *below* the "0.5" prediction on some interval — a nonsensical **crossing**. grian
repairs this the cheap way: **sort** the quantiles within each interval so they're monotone. Simple,
and it's enough.

---

## Chapter 8 — Calibration: does "90%" really mean 90%?

A fan is only trustworthy if its stated probabilities are honest. If the model says "90% chance the
price is below \$400" but prices actually exceed \$400 *thirty* percent of the time, the fan is
**over-confident** — its bands are too narrow — and any risk-aware dispatch built on it will
under-hedge. Checking and fixing this is **calibration**.

**How we check.** A fan is *calibrated* if, for every level $\tau$, actual outcomes fall below the
$\tau$-quantile exactly a fraction $\tau$ of the time: $\Pr[y \le Q_\tau] = \tau$. Plot empirical
coverage against nominal $\tau$ and a calibrated model sits on the diagonal. A single-number summary
of whole-distribution quality is the **CRPS** (Continuous Ranked Probability Score), a *proper*
scoring rule (one that can't be gamed by lying about your uncertainty):

$$\mathrm{CRPS}(F, y) = \int_{-\infty}^{\infty}\!\big(F(x) - \mathbb{1}[y \le x]\big)^2\, dx,$$

where $F$ is the predicted cumulative distribution and $\mathbb{1}[\cdot]$ is the step function that
jumps from 0 to 1 at the true value $y$. grian estimates CRPS by averaging the pinball loss across
quantile levels (`metrics.crps`) — the discrete version of that integral.

**How we fix it: split-conformal calibration** (`lightgbm_qmean_cal`). **Conformal prediction** is a
technique that turns any model's outputs into intervals with a *guaranteed* coverage rate, using only
a held-out calibration set and no assumptions about the data's distribution. Here it's used to *widen*
the fan: hold out the most recent 28 days, fit the trees on the rest, then on the held-out set measure
how far off each quantile's coverage is and **shift each quantile** to hit its nominal rate (lower
quantiles down, upper quantiles up — a widening), re-sorting to stay monotone. It's done in dollar
space, aimed squarely at the upper quantiles' tendency to under-cover the spikes (Entry 033).

**The cautionary finding.** Calibration *did* help the risk-aware dispatch see spike risk — but the
improvement **did not survive testing on a different time window.** It had quietly fit the evaluation
window. This is a recurring danger (a knob tuned to the test set), and calibration is best understood
as a *decision-quality* fix, not an accuracy fix.

---

## Chapter 9 — Backtesting without fooling yourself

A trading strategy that looks great in a backtest and loses money in reality has almost always
**cheated** — used information it wouldn't have had at the time. This chapter is about not cheating.

**Walk-forward (rolling-origin) evaluation.** The only honest way to test a time-series strategy: train
on all history up to a day, make decisions for the *next* day, then roll forward and repeat. You never
train on the future. `runner.py` does this for the simple loop, `mpc.py` for the re-planning loop, and
`backtest.rolling_origin` is the curriculum version.

**The embargo, and how we *prove* there's no leakage.** Between the training data and the day being
tested we insert an **embargo** — a gap the length of the forecast horizon — so that the training data
can't overlap with anything the forecast is trying to predict. But claiming "no leakage" is cheap; grian
*proves* it. There is a deliberately sabotaged variant (`leak_future` in `ablations.py`) that injects a
future value into the features, and a **unit test that asserts the score gets worse when you do**
(`tests/test_backtest.py`). If leaking the future *helped*, the pipeline would be cheating somewhere;
the test guarantees it isn't.

**Refit cadence and the rolling window.** Re-training every single day is expensive, so grian refits
every `refit_days` (28 in the campaign) on a **rolling window** of the last `train_lookback_days` (548 ≈
18 months; the alternative is an *expanding* window that keeps all history). This is a compute-vs-freshness
trade-off, and it bites: too short a window once dropped a model's capture from 0.389 to 0.288.

**One common evaluation window.** Every configuration is scored over the *exact same* stretch of history
— **2025-07-01 to 2026-06-30** — against the *same* oracle, so the numbers are directly comparable. (The
`config.yaml` you'll see still lists the older curriculum window of 2023 and a 7-quantile set; the
campaign overrides both.) The dashboard can then recompute capture over any sub-window on the fly from the
saved daily revenues.

**The old sin this all fixes.** Early work scored different models on *different* windows (incomparable)
and some results were quietly voided by leakage and by a hardcoded interval length (Chapter 12's "dt
bug"). Hence the discipline: one fixed window, one frozen oracle, an embargoed walk-forward, seeds fixed
(42), versions pinned, downloads cached, and CPU/Apple-Silicon only. Reproducibility here is a
scientific requirement, not a nicety.

---

## Chapter 10 — Metrics, and why "best" depends on which one

There are two families of metric, and the whole project turns on the gap between them.

**Accuracy metrics** — how close is the forecast to the truth?
- **MAE** (Mean Absolute Error) = the average of $|y - \hat{y}|$.
- **RMSE** (Root Mean Squared Error) = $\sqrt{\text{mean}((y-\hat{y})^2)}$ — like MAE but it punishes
  big misses much harder (squaring).
- **Skill** = how much better than the naive baseline: $1 - \text{MAE}_\text{model}/\text{MAE}_\text{naive}$.
  Positive means you beat "last week, same day." (`analytics.skill_vs_naive`.)

**Decision-value metrics** — how much money did the *trades* make?
- **Capture ratio** = your revenue ÷ oracle revenue (Chapter 1).
- **Regret** = oracle revenue − your revenue (the dollars you left on the table).

**Balanced vs pooled capture — the metric choice that mattered.** If you compute one capture number for
the whole year by dividing total revenue by total oracle revenue (**pooled** capture), the result is
dominated by a *tiny* number of enormous days — the **top 10 days are about 45% of the entire year's
oracle revenue.** That makes the pooled number jumpy and easy to win by luck on two spike days. So grian
also reports **balanced** capture: compute the capture ratio *within each of the 12 months*, then
average the twelve. Each month gets equal weight, spike luck is diluted, and model comparisons become
stable. **Rule: choose balanced for ranking models, and report pooled alongside.** (`balanced_eval.py`.)

**Risk metrics.** The ledger also reports the **Sharpe ratio** (mean daily revenue ÷ its standard
deviation — reward per unit of volatility) and **peak drawdown** (the worst peak-to-trough dip).

**The headline finding, stated plainly.** Rank the models by accuracy and by capture, and the rankings
are roughly *reversed*. `lear_weather` has the **best MAE-skill (~0.63)** and the **worst capture
(~0.50)**. Why? Because **accuracy is earned in the calm 95% of intervals, but money is earned in the
spiky 5%.** MAE rewards being right on average, which favours smooth, mean-reverting forecasts — and
those are exactly the forecasts that miss the peaks a battery trades on. This is the reason the project
insists you **pick your metric before you pick your model.**

---

## Chapter 11 — Choosing a model on robustness, not luck

Given the volatility above, you should not crown the model with the single highest full-year number —
it might just have won two spike days. grian selects on **robustness**: rank models by their *average*
capture across periods **and** their *worst* period, preferring a model whose performance is a flat,
high line over one that's spiky. The curated "key models" set keeps one representative per family and
output type (linear/tree × point/quantile) plus the champion and a baseline, and drops dominated or
buggy variants. The dashboard shows this multi-objectively — capture, consistency, Sharpe, drawdown on
one radar. The enemy throughout is **overfitting the evaluation window** (as calibration and the CVaR
knob both did).

---

## Chapter 12 — The battery and the linear program

We've been forecasting; now we spend the forecast. Given a price path, how does the battery decide its
charge/discharge schedule? By solving a **linear program**.

### 12.1 What a linear program is

A **linear program** (LP) is an optimisation problem where you maximise (or minimise) a *linear*
function of some decision variables, subject to *linear* equality and inequality constraints. "Linear"
means no variable is multiplied by another and none is squared — just weighted sums. LPs are the
best-behaved optimisation problems there are: fast, and guaranteed to find the true global optimum. grian
solves them with **HiGHS**, an open-source LP solver, via `scipy.optimize.linprog`.

### 12.2 The battery LP

**Decision variables**, for each of the $T$ intervals in the horizon: charge power $c_t \ge 0$,
discharge power $d_t \ge 0$ (both in MW), and the **state of charge** $\text{SOC}_t$ — how much energy
is in the battery (MWh). Stacked together that's $3T$ variables.

**Objective — maximise arbitrage revenue.** Revenue in an interval is price × net energy sold:

$$\max_{c, d, \text{SOC}}\ \sum_{t=1}^{T} p_t\,(d_t - c_t)\,\Delta t.$$

Discharging ($d_t$) earns money, charging ($c_t$) spends it, and $\Delta t$ converts power (MW) into
energy (MWh) over the interval.

**The state-of-charge dynamics (equality constraints).** The battery isn't perfectly efficient — energy
is lost on the round trip. grian models an **85% round-trip efficiency** ($0.85$), split evenly as
$\eta = \sqrt{0.85}$ on the way in and $\eta = \sqrt{0.85}$ on the way out (so a full in-and-out cycle
keeps $\eta^2 = 0.85$). The stored energy evolves as:

$$\text{SOC}_t = \text{SOC}_{t-1} + \underbrace{\eta\,\Delta t\, c_t}_{\text{energy stored}} - \underbrace{\frac{\Delta t}{\eta}\, d_t}_{\text{energy drawn}}.$$

Note the asymmetry: to *deliver* $d_t$ to the grid you must *draw* $d_t/\eta$ from the store (you lose
some on the way out); charging $c_t$ only adds $\eta c_t$ to the store.

**Bounds.** $0 \le c_t, d_t \le P$ (power capped at $P = 100$ MW) and $0 \le \text{SOC}_t \le E$ (energy
between empty and $E = 200$ MWh).

**The cycle limit (inequality constraint).** Batteries degrade with use, so we cap how much they cycle:
**2 full cycles per calendar day.** In energy terms, total discharge in a day can't exceed
$2 \times 200 = 400$ MWh:

$$\sum_{t \in \text{one day}} d_t\, \Delta t \le 400\ \text{MWh}.$$

The solver uses a **sparse** representation of these constraints (most entries are zero), which lets it
scale from a single day ($T = 288$) all the way to the full-year oracle ($T \approx 35{,}000$) without
running out of memory.

### 12.3 The oracle, concretely

The **oracle** (`oracle.py`) is just this same LP solved against the **actual** prices with the battery
starting empty. That's what "perfect foresight" means mechanically: full knowledge of every future price.
For speed it's solved in **7-day blocks** with the battery pinned empty at each block boundary (the
full-year LP grows super-linearly in the solver, and carrying charge across a midnight is worth almost
nothing under a per-day cycle cap). This makes the oracle about 0.2% *conservative*, which is fine — it's
still an honest ceiling. The result is cached; it only changes if the battery spec changes. Over the
common window the oracle earns about **\$25.18 million**, of which the top 10 days are ~45%.

A neat internal sanity check (`_assert_closed_cycles_nonnegative`): with perfect foresight you can never
*lose* money over a complete cycle (worst case, do nothing and earn zero), so grian asserts that every
segment between the battery returning to empty is non-negative. It checks this per *closed cycle*, not per
calendar day — because a single day *can* be net-negative when the oracle optimally buys cheap overnight
to sell into the next morning's spike.

### 12.4 Two traps worth reciting

- **Trap T1 — the "dt bug."** The interval length $\Delta t$ **must** come from the config
  (`resolution_dt_hours`: 1/12 for 5-min, 0.5 for 30-min), never hardcoded. Early code hardcoded 30
  minutes while running on 5-minute data, which silently multiplied every revenue by 6 and made a dozen
  results *physically fictional.* This is the bug that motivated the whole "one common window, prove
  everything" discipline.
- **Trap T2 — clamp before you bill.** The optimiser's plan is only a *plan*; when we actually execute it
  against reality we pass it through `clamp_action`, which trims any charge/discharge that the true state
  of charge or remaining cycle budget can't support. **Revenue is only ever computed from the clamped,
  physically-real actions** — never from the raw plan.

---

## Chapter 13 — Dispatch: turning a forecast into trades

The forecast is only half the system. The **executor** is the piece that repeatedly consults the forecast,
solves the LP, and executes trades. How it does that turns out to matter as much as the forecast itself.

### 13.1 Open-loop vs MPC

- **Open-loop** (`runner.py`): at midnight, forecast the whole day, solve one LP, and commit to that plan
  regardless of what happens. Simple, but the plan goes stale the moment reality deviates.
- **MPC — Model Predictive Control** (`mpc.py`): the standard control-theory approach to exactly this
  problem. MPC is a **receding-horizon** loop — it re-solves the LP every few intervals *from the true
  current state of charge*, and re-forecasts every so often from the *latest* data ("predict-from-now"
  again). Each solve produces a plan for the whole remaining horizon, but only the first few steps are
  executed before it re-solves — the horizon keeps "receding" ahead of us. This converts the forecaster's
  much-better *short-lead* skill (it's far more accurate about the next hour than about midnight tomorrow)
  into better trades.

There's a small performance hack, **telescoping** (`_telescope`): keep the first `fine_n` steps at full
resolution and average the rest into coarse blocks, shrinking the LP. It's only used when the horizon is
large (5-minute data); at 30-minute resolution the 48-step horizon is solved in full, because coarsening
would *average away the daily peak* and the controller would sell into lesser bumps.

### 13.2 The spike gate — the fix that won

Here is one of the most instructive episodes in the project. Intuitively, re-planning constantly with the
freshest data (aggressive MPC) should beat the sluggish open-loop. It didn't — an early MPC **collapsed to
half of open-loop's capture.** That's a shocking result, and chasing it down is the kind of debugging story
worth studying.

The culprit was a feature called `observe_present`. At each decision the current price *is* known (you're
standing in it), so it seems obviously right to pin the plan's first step to the true current price. But
doing that at *every* re-solve made the LP chase the **forecast residual** — the tiny, mean-reverting gap
between the real price now and the model's smoothed forecast. The battery churned cycles reacting to noise,
and capture cratered. The insight (Entries 034/035): **intraday re-forecasting doesn't help; state-and-price
feedback does — but only for the events that matter.**

The fix is the **spike gate** (`observe_gate`): only pin the plan to the live price when that price is a
genuine spike, i.e. **≥ \$3000/MWh**. Below the gate, trust the smooth day-ahead forecast and keep doing
the clean daily arbitrage; above it, react immediately and grab the scarcity event the forecast missed. So
the winning controller is: **forecast once a day, re-solve every interval from true state, and observe the
present only above a \$3000 gate.** (Setting the gate to 0 reproduces the old buggy always-observe
behaviour.) This — a change to the *dispatch objective*, not to any forecast feature — is what carried the
system to its champion score. It is the concrete proof of the project's recurring lesson that the objective
is the lever.

### 13.3 The ladder of dispatch objectives under uncertainty

When we have a *fan* rather than a point forecast, we can dispatch in ways that account for risk. Think of
the fan as a set of price *scenarios* $s$, each a full price path $p^s$ with a probability $\pi_s$ (from the
quantile weights). Each candidate action plan $a$ earns revenue $R_s(a) = \sum_t p^s_t (d_t - c_t)\Delta t$
under scenario $s$. The question is: *which summary of the $\{R_s\}$ do we maximise?* Different answers give
a ladder from reckless to timid:

| Mode | What it maximises | Personality | How grian does it |
|---|---|---|---|
| **point** | revenue on the single mean path | ignores uncertainty | one LP on the integrated-mean price. Over-trades / "whipsaws." The failing baseline. |
| **scenario** (robust) | $\max_a \min_s R_s(a)$ | maximally cautious | solve the LP on each quantile path separately, then take the **minimum** action across them ("act only where every scenario agrees"). Under-trades. |
| **scenario_ev** | $\max_a \mathbb{E}[R_s(a)]$ | risk-neutral | per-scenario LPs, fused by **probability-weighted average** action. |
| **scenario_cvar** | a blend of EV and robust | tunable | $(1-\lambda)\cdot\text{action}_\text{EV} + \lambda\cdot\text{action}_\text{robust}$; $\lambda$ dials risk appetite. |
| **mean_cvar** | $(1-\lambda)\mathbb{E}[R] + \lambda\,\mathrm{CVaR}_\alpha(R)$ | tunable, principled | **one** joint LP across all scenarios (below). |

Why the mean and not the median drives every non-gate mode: revenue is *linear* in price, so its expected
value needs $\mathbb{E}[p]$; and the right-skew means the median under-prices spikes (§7.2).

### 13.4 The maths of CVaR

The most principled rung is **mean-CVaR**, and it's worth understanding because it's the standard way to
trade expected reward against tail risk. First, **VaR** (Value-at-Risk) at level $\alpha$ is a threshold:
the loss you won't exceed with probability $1-\alpha$. **CVaR** (Conditional Value-at-Risk, also called
expected shortfall) is the *average loss in the worst $\alpha$ fraction of scenarios* — a smarter risk
measure because it looks at *how bad* the tail is, not just where it starts.

The beautiful fact (the **Rockafellar–Uryasev** identity) is that CVaR can be written as a *linear* program.
With losses $\ell_s = -R_s$, an auxiliary variable $\eta$ (which will settle at the VaR), and slack variables
$z_s \ge 0$:

$$\mathrm{CVaR}_\alpha(\ell) = \min_{\eta}\ \Big\{\, \eta + \tfrac{1}{\alpha}\textstyle\sum_s \pi_s z_s \ \ \text{subject to}\ \ z_s \ge \ell_s - \eta,\ \ z_s \ge 0 \,\Big\}.$$

Each $z_s$ measures how far scenario $s$'s loss overshoots the threshold $\eta$; averaging those overshoots
(scaled by $1/\alpha$) and adding $\eta$ back gives the mean tail loss. Because it's all linear, grian folds
it straight into the battery LP (`solve_mean_cvar_lp`): maximise $(1-\lambda)\mathbb{E}[R] + \lambda\,\mathrm{CVaR}$
with those extra rows, producing *one* charge/discharge plan that is good on average **and** not catastrophic
in the bad tail. $\lambda = 0$ recovers pure expected value; large $\lambda$ (or small $\alpha$) approaches
worst-case. It's a bigger LP, but still an LP.

(The fully "correct" stochastic form — a two-stage program that commits one action now but plans to *adapt*
per scenario later, respecting *non-anticipativity* — is described in the maths doc but not built.)

### 13.5 Two warnings

- **Calibrate first, or all of this is garbage.** Every rung above *trusts the quantiles*. If the fan is
  over-confident (Chapter 8), CVaR under-estimates the tail and hedges too little. Calibrate the fan before
  trusting any hedge.
- **Beware knobs tuned to the test set.** The CVaR risk-shaping that looked good on the evaluation window
  **did not survive** validation on a different window — same trap as calibration. And note a clean fact the
  dashboard makes visible: the *forecast* is the same regardless of dispatch mode (robust/EV/CVaR all read
  the same fan); only the *cadence* of re-forecasting changes the forecast line. The mode changes how you
  *trade* a forecast, not the forecast itself.

---

## Chapter 14 — How the tuning was actually done

You might expect a giant automated hyperparameter search. There wasn't one, and that's a defensible choice.
grian used **principled defaults plus a small, hypothesis-driven ablation.** An **ablation** is an experiment
that toggles one design choice at a time to measure its effect — the scientific way to tune, because each
result *means* something. The centrepiece is a 16-cell factorial:

$$\{\text{LEAR}, \text{LightGBM}\} \times \{\text{point}, \text{quantile}\} \times \text{weather}\{\text{off}, \text{on}\} \times \text{Fourier}\{\text{off}, \text{on}\},$$

each run under both open-loop and spike-gated MPC. (Grid/random/Bayesian search do exist in `search.py`, but
weren't needed here.) A subtle point worth internalising: **numerical stability choices are hyperparameters
too** — winsorising features, standardising the target, picking a learning rate and epoch count that
*converge*, clamping predictions finite. "Make it converge and stay finite" is part of the recipe.

The key results, under the winning spike-gated MPC, ranked by balanced capture:

| model | balanced | pooled |
|---|---|---|
| lightgbm_qmean_weather_fourier (champion) | **0.554** | 0.585 |
| lightgbm_qmean_fourier | 0.533 | 0.554 |
| lightgbm_qmean | 0.525 | 0.569 |
| lightgbm_qmean_weather | 0.522 | 0.564 |
| lightgbm_rich_fourier (best point tree) | 0.521 | 0.563 |
| lightgbm_rich | 0.509 | 0.559 |
| lear_weather (best point LEAR) | 0.500 | 0.556 |
| lear_qmean_torch_fourier (best quantile LEAR) | 0.493 | 0.526 |
| autoregression | 0.470 | 0.500 |
| lear_fourier (the phase-reg wart) | 0.426 | 0.471 |

Reading the table: **quantile trees beat point trees beat linear models**; **Fourier helps** trees, quantile
trees, and — once the calendar is unpenalised — the quantile linear model too; **weather is conditional** (it
hurts trees, helps the point linear model, and helps the quantile tree only *with* Fourier); and the champion
uses **both** weather and Fourier on the quantile tree.

The reference recipes, all in one place:

| Family | Key settings |
|---|---|
| LightGBM point (`lightgbm_rich`) | 300 trees, lr 0.05, 31 leaves, min_child 20, subsample/colsample 0.8; `quantile@0.5` for pinball. |
| LightGBM quantile (`lightgbm_qmean`) | 150 trees, lr 0.05, same tree shape, `quantile@τ`; quantiles [0.05, 0.5, 0.9, 0.98]; optional 28-day conformal calibration; lead-dependent mean/median. |
| LEAR / Lasso (`lear`) | `LassoCV`, 3-fold CV, 20 candidate penalties, max_iter 2000; one-hot calendar by default. |
| Quantile LEAR (`lear_qmean_torch`) | one batched linear layer, 400 epochs, lr 0.005, L1 α 0.01, features clipped ±5σ, calendar exempt, predictions clamped; CPU-pinned (~13 s/refit). |
| Decision-focused weights | `magnitude` ramp (scale \$300) or `quantile` boost (q 0.9); strength 1.0; normalised to mean 1. |
| Executors | resolve every interval, reforecast daily, spike gate \$3000; CVaR λ/α for risk modes. |
| Simulation | refit every 28 days, 548-day rolling window, embargo = horizon, seed 42, asinh transform. |

---

## Chapter 15 — The engineering that makes the results trustworthy

Good numbers you can't reproduce are worthless, so grian is strict about process:

- **Determinism and caching.** Fixed seeds, pinned library versions, every download cached and never
  re-pulled, CPU/Apple-Silicon only (no CUDA-only dependencies).
- **Everything on disk, reproducible from a config.** A "trial" is a folder: its frozen config, the trade
  ledger, the metrics, optionally the model and forecasts, and the git commit hash. The dashboard is built
  entirely from these folders.
- **Checkpointing and telemetry.** Long fan builds save partial progress to a `.partial` file and only
  "commit" it on completion — so a crash 9 hours in doesn't lose everything, and only a *complete* fan counts
  as cached. Re-running a batch skips trials already done. This discipline was born from a real disaster: a
  10-hour build whose progress logging had been switched off, so it ran invisibly and unrecoverably. The
  rule now: **never silence the logs inside a build wrapper.**
- **The experiment log as a lab notebook.** Every failure and dead end gets a written-up entry in
  `outputs/experiment_log.md` (Entries 001–038) — the single best artifact for understanding *how the
  research actually went*, and adding to it is mandatory.
- **Thin notebooks, fat library, real tests.** All logic lives in `src/grian/`; the notebooks explain and
  visualise. `ruff check .` lints (docstrings enforced) and `pytest -q` runs the suite that proves the
  physics and the no-leakage guarantee.

---

## Chapter 16 — The seven lessons to walk in with

If you remember nothing else, remember these — each is backed by a concrete episode above:

1. **Decision value ≠ forecast accuracy.** The most accurate forecaster (LEAR) is the worst trader; the money
   lives in the spikes, and accuracy metrics live in the calm.
2. **"Predict-from-now" is a contract.** A model that quietly forecasts from its training-time tail goes inert
   under MPC (the AR flatline).
3. **The objective is the lever, not the features.** Spike-precursor features alone didn't help; a spike-aware
   *dispatch* rule (the \$3000 gate) did.
4. **Verify your numerics.** The Mac GPU computed wrong gradients; winsorise heavy tails; clamp before the LP.
5. **Pick the metric before the model.** Pooled vs balanced capture reorders the leaderboard; one number over
   one period is a trap.
6. **Regularisation has geometry.** L1 on a sine/cosine pair biases the phase — exempt or group-penalise the
   calendar.
7. **Beware knobs tuned to the test set.** The CVaR risk-shaping and the calibration win both evaporated on a
   fresh window.

---

## Appendix A — Check-your-understanding (say the answer out loud)

Cover the parenthetical and try to explain each in a couple of sentences.

- **What's the headline metric, and why that one?** (Capture ratio = your revenue ÷ oracle revenue; a forecast
  is only worth its downstream trading value, so we measure dollars kept, not error.)
- **Why model asinh(price) instead of price?** (Variance stabilisation — it compresses the huge spikes so they
  don't dominate the loss — and unlike log it's defined and smooth for the negative prices this market has.)
- **Why is the best forecaster the worst trader?** (Accuracy is earned in the calm 95% of intervals; profit is
  earned in the spiky 5%. L1 + smoothers flatten the peaks a battery trades on.)
- **Direct vs iterative multi-step forecasting?** (Direct = a separate model per horizon step, no error
  accumulation; iterative feeds predictions back in and compounds errors. Only AR is iterative here.)
- **What broke the aggressive MPC and how was it fixed?** (Observing the present at every re-solve made the LP
  trade the mean-reverting forecast residual and churn cycles; the fix only observes the live price above a
  \$3000 spike gate.)
- **Give the battery LP's objective and its SOC update.** ($\max \sum_t p_t(d_t - c_t)\Delta t$; $\text{SOC}_t =
  \text{SOC}_{t-1} + \eta\Delta t\, c_t - \tfrac{\Delta t}{\eta} d_t$ with $\eta = \sqrt{0.85}$.)
- **Why solve the oracle in weekly blocks?** (The LP grows super-linearly; carrying charge past midnight is
  worth ~nothing under a per-day cycle cap; the ~0.2% conservatism is acceptable.)
- **How do you *prove* there's no leakage?** (A sabotaged variant injects a future value and a unit test asserts
  the score gets *worse*, plus an embargo the length of the horizon.)
- **Why does the LP use the fan's mean, not its median?** (Revenue is linear in price, so its expectation needs
  $\mathbb{E}[p]$; the right-skew means the median under-predicts spikes.)
- **Turn CVaR into an LP.** (Rockafellar–Uryasev: with VaR level $\eta$ and slacks $z_s \ge \ell_s - \eta,\
  z_s \ge 0$, $\mathrm{CVaR}_\alpha = \min_\eta \eta + \tfrac{1}{\alpha}\sum_s \pi_s z_s$.)
- **State the pinball loss and what it converges to.** ($\rho_\tau = \max(\tau e, (\tau-1)e)$ with $e = y -
  \hat{y}$; its minimiser is the $\tau$-quantile; the gradient is bounded, so it's robust to spikes.)
- **Balanced vs pooled capture?** (Pooled = total/total, dominated by ~2 spike days; balanced = the average of
  12 monthly ratios, the fair way to rank models.)
- **What's the Fourier/LEAR "wart"?** (L1 penalises $|a| + |b|$, which isn't rotation-invariant, so it biases
  the harmonic's phase; fix by group-penalising the pair or exempting the calendar — then Fourier helps.)
- **What did the MPS episode teach?** (Verify numerics — the GPU gave wrong gradients and diverged where CPU
  converged; and the real speed-up was batching ~100 fits into one, not the hardware.)

## Appendix B — Numbers to have on the tip of your tongue

- **Battery:** 100 MW / 200 MWh (2-hour), round-trip efficiency 0.85, 2 cycles/day → 400 MWh discharge cap.
  Region SA1 (South Australia).
- **Horizon:** one day ahead = 288 five-minute steps, or 48 half-hour steps.
- **Quantiles (campaign):** 0.05, 0.5, 0.9, 0.98 — deliberately weighted to the upper tail.
- **Common evaluation window:** 2025-07-01 → 2026-06-30. Oracle revenue ≈ \$25.18M; top-10 days ≈ 45% of it.
- **Key thresholds:** spike ≈ \$300; market price cap ≈ \$16–17k; MPC observe-gate = \$3000.
- **Capture:** champion 0.554 balanced / 0.585 pooled (`lightgbm_qmean_weather_fourier` + spike gate); earlier
  champion 0.546 val / 0.562 test. LEAR: MAE-skill ~0.63 but capture ~0.50.
- **Recipes:** LightGBM 300 trees (point) / 150 (quantile), lr 0.05, 31 leaves. LEAR: LassoCV, 3-fold, 20
  penalties. Quantile-LEAR torch: 400 epochs, lr 0.005, L1 α 0.01, clip ±5σ, ~13 s/refit (~40× faster than
  sklearn's ~8 min).
- **Simulation:** refit every 28 days, 548-day (~18-month) rolling window, embargo = horizon, seed 42,
  transform asinh.

---

### Where to read the real thing

`docs/architecture.md` (how the modules connect) · `docs/dispatch-and-scoring.md` (physics and the traps) ·
`docs/dispatch-under-uncertainty-maths.md` (the full objective ladder with worked examples) ·
`docs/probabilistic-dispatch-explained.md` and `docs/executors-and-dispatch-explained.md` (dispatch, gently) ·
`docs/quantile-mpc-postmortem.md` (the failed quantile-gate MPC) · `docs/data-and-features.md` ·
`docs/running-experiments.md` · `docs/extending.md` · `docs/techniques-roadmap.md` (the terse roadmap this
document expands) · the 16-chapter `docs/learning_guides/grian_learning_guide.pdf` ·
`outputs/experiment_log.md` (Entries 001–038 — the narrative of what failed and why) ·
`outputs/reports/mpc_investigation_report.pdf` · the live `outputs/dashboard/index.html`.
