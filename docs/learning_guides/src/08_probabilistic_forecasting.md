# 8. Probabilistic Forecasting and Calibration

## Why Probabilistic Forecasts Matter

In Chapter 7, we trained models that produce both point forecasts (a single predicted price) and quantile forecasts (predictions at multiple probability levels). This chapter is about the deeper question: how do we know the probabilistic forecast is trustworthy? And what makes one probabilistic forecast better than another?

The difference between a point forecast and a probabilistic forecast is the difference between being told "it will rain tomorrow" and being told "there is a 70% chance of rain tomorrow, with a 20% chance of heavy rain." The second statement is far more useful for decision-making — it lets you assess the risk, weigh alternatives, and plan accordingly.

<div class="definition-box">
<strong>Probabilistic forecast:</strong> A forecast that provides information about the full range of possible outcomes and their likelihoods, rather than a single predicted value. It answers the question "what might happen and how likely is each outcome?" rather than just "what do we expect to happen?" For electricity prices, a probabilistic forecast might say: "the median price is $85/MWh, with a 10% chance of being below $40 and a 10% chance of being above $300."
</div>

For battery dispatch, probabilistic forecasts are not just academically interesting — they are economically essential. Consider the dispatch decision at 4 PM:

**Scenario A — Confident forecast:** The median price for 6 PM is $150/MWh, with an 80% interval of [$120, $180]. The battery should discharge now — the price is reliably high.

**Scenario B — Uncertain forecast:** The median is the same $150/MWh, but the 80% interval is [$30, $2,000]. Should the battery discharge now? Perhaps waiting is better — prices might go much higher. Or perhaps discharging now is wise — prices might collapse. The wide interval signals that the market is in an unpredictable state.

**Scenario C — Asymmetric forecast:** The median is $150, the 10th percentile is $120, but the 90th percentile is $3,000. The upside risk is enormous. A risk-neutral battery might hold its charge, betting on the possibility of a spike.

The point forecast ($150) is identical in all three scenarios. Only the probabilistic forecast distinguishes them and enables intelligent risk management.

<div class="key-point">
<strong>The fundamental argument:</strong> A battery operator who treats every $150 forecast identically — regardless of whether the uncertainty is $20 or $2,000 — is leaving money on the table. Probabilistic forecasts convert "I think the price will be $150" into "I think the price will be $150, and here is how wrong I might be." The second statement enables the dispatch optimiser (Chapter 9) to make risk-aware decisions.
</div>

### Forms of Probabilistic Forecasts

Probabilistic forecasts come in several forms, ordered from least to most informative:

**1. Prediction intervals.** A pair of values defining a range expected to contain the actual outcome with a stated probability. For example, "the 80% prediction interval for the 6 PM price is [$65, $320]" means the model believes there is an 80% chance the price will fall between $65 and $320. Prediction intervals are the simplest probabilistic forecast — easy to produce, easy to communicate, easy to use in dispatch.

<div class="definition-box">
<strong>Prediction interval:</strong> A range of values that is expected to contain the actual future observation with a specified probability (the <strong>coverage level</strong>). An 80% prediction interval should contain the actual value 80% of the time. Wider intervals have higher coverage but contain less information. The width of the interval reflects the model's <strong>uncertainty</strong> — not its accuracy. A model can be accurate (small average error) but uncertain (wide intervals) or vice versa.
</div>

**2. Quantile forecasts.** A set of predicted values at multiple probability levels (e.g., 5th, 10th, 25th, 50th, 75th, 90th, 95th percentiles). This is richer than a single prediction interval because it describes the shape of the distribution — is it symmetric or skewed? Are the tails thin or fat? Quantile forecasts are what GBT quantile regression (Chapter 7) produces directly.

**3. Density forecasts.** The full probability density function (PDF) — a curve showing the relative likelihood of every possible price. This is the most informative form but the hardest to produce accurately. In practice, density forecasts are often approximated by interpolating between quantile forecasts, fitting a parametric distribution, or using kernel density estimation.

<div class="definition-box">
<strong>Probability density function (PDF):</strong> A function that describes the relative likelihood of each possible value of a continuous random variable. The area under the PDF between two values gives the probability that the outcome falls in that range. For electricity prices, the PDF at any future time step shows the full distribution of possible prices — where the density is high, the price is more likely; where it is low, the price is unlikely.
</div>

**4. Scenario forecasts.** Multiple complete price trajectories — plausible "stories" of how prices might evolve over the forecast horizon. Each scenario is a possible future, and together they represent the range of outcomes. Scenarios are used directly in stochastic optimisation (Chapter 9), where the dispatch LP is solved across multiple scenarios simultaneously.

<div class="definition-box">
<strong>Scenario forecast:</strong> A collection of complete, plausible future price trajectories generated from the probabilistic forecast. Each scenario represents one possible future — "what might happen if wind drops and demand surges" or "what might happen on a mild, windy day." Scenarios preserve the temporal structure of the forecast (correlations across time periods), unlike quantile forecasts which describe each time period independently.
</div>

---

## Quantile Regression Averaging (QRA)

### The Forecast Combination Principle

One of the most robust findings in all of forecasting research is the **forecast combination principle**: combining forecasts from multiple models almost always improves accuracy. This result, first documented by Bates and Granger in 1969, holds across virtually every forecasting domain — weather, economics, energy, demographics, sports — and is sometimes called the "forecast combination puzzle" because it works even when one model is clearly better than the others.

<div class="definition-box">
<strong>Forecast combination principle:</strong> The empirical finding that an average (or weighted combination) of forecasts from different models is typically more accurate than any individual forecast. This works because different models make different errors — some overpredict when others underpredict — and combining them cancels out many of these idiosyncratic errors. It is one of the most reliable findings in forecasting research and forms the foundation for ensemble methods in machine learning.
</div>

Why does combination work so consistently? Consider three models for electricity price forecasting:

- **LEAR** (linear model) misses nonlinear patterns (understates spikes) but is stable and rarely produces extreme errors
- **GBT** captures nonlinear relationships but may overfit to recent market regimes, producing errors when the regime changes
- **Neural network** can learn complex temporal patterns but is noisy — its predictions vary more from run to run

These models make **different kinds of errors**. When LEAR understates a spike, GBT might predict it correctly. When GBT overfits to a recent pattern that does not persist, LEAR's simpler model may be more appropriate. Averaging the forecasts exploits these complementary strengths: the errors partially cancel, and the combined forecast is more reliable than any individual one.

<div class="example-box">
<strong>Real-world example — the power of combination:</strong> In the Global Energy Forecasting Competition 2014 (GEFCom2014), the winning entries almost universally used forecast combination. The winning team for the electricity price track combined several model types — LEAR variants, GBT, and neural networks — using QRA. No single model in their ensemble was the best individual entry, but the combination outperformed all individual entries. This pattern repeats in forecasting competitions across domains: the best ensembles beat the best individual models.
</div>

<div class="key-point">
<strong>The diversity principle:</strong> Combination works best when the component models are <em>diverse</em> — they make different types of errors. Combining three nearly identical GBT models with slightly different hyperparameters adds little value because they make similar errors. Combining a linear model, a tree model, and a neural network adds substantial value because their error patterns are structurally different.
</div>

### The QRA Method

**Quantile Regression Averaging (QRA)**, introduced by Nowotarski and Weron (2015), is a technique specifically designed for combining multiple point forecasts into a single probabilistic forecast. It has become the standard combination method in electricity price forecasting.

<div class="definition-box">
<strong>Quantile Regression Averaging (QRA):</strong> A forecast combination method that takes point forecasts from K different models as inputs and produces calibrated quantile forecasts as output. For each quantile level τ, a separate quantile regression is fitted using the point forecasts as explanatory variables. This allows different models to receive different weights at different quantile levels — one model might be best at predicting the median while another is best at predicting the tails.
</div>

The method works as follows. Suppose you have K models, each producing a point forecast: f_1, f_2, ..., f_k. For each quantile level τ (e.g., τ = 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95), fit a quantile regression:

<div class="equation">

Q_τ(y | f_1, f_2, ..., f_k) = β_0τ + β_1τ · f_1 + β_2τ · f_2 + ... + β_kτ · f_k

</div>

where the coefficients β_0τ, β_1τ, ..., β_kτ are estimated by minimising the pinball loss (Chapter 7) at level τ on a calibration dataset.

The key properties of QRA are:

**1. Different weights for different quantiles.** The coefficients β_iτ vary with the quantile level τ. A model that excels at predicting price spikes (upper tail) receives high weight at τ = 0.90 or τ = 0.95, even if it is mediocre at predicting the median. Conversely, a model that is accurate on average but poor at predicting extremes receives high weight at τ = 0.50 and low weight in the tails. This is much smarter than simple averaging, which treats all quantile levels equally.

**2. Automatic combination to probabilistic forecast.** The input is K point forecasts (single numbers). The output is a full set of quantile forecasts — a probabilistic forecast. QRA simultaneously combines the models and produces uncertainty estimates, without requiring the individual models to provide their own uncertainty.

**3. Built-in calibration.** Because QRA is fitted using the pinball loss, the resulting quantile forecasts are approximately calibrated (the stated coverage matches the actual coverage). This calibration property comes "for free" from the quantile regression fitting process.

**4. Adaptive reweighting.** When QRA is refitted periodically (e.g., daily or weekly on the most recent data), it automatically adjusts the weights as models improve or deteriorate. If one model degrades due to a market regime change, QRA will downweight it without any manual intervention.

### Why Not Simple Averaging?

Simple averaging — y\* = (f_1 + f_2 + ... + f_k) / K — is a surprisingly strong baseline for forecast combination. It has three limitations that QRA addresses:

1. **It assumes equal weights across quantiles.** Simple averaging gives the same weight to each model regardless of whether you care about the median or the tails. QRA allows quantile-specific weights.

2. **It produces only a point forecast.** The average of K point forecasts is another point forecast. It gives no uncertainty information. QRA produces a full set of quantiles.

3. **It is static.** The weights are always 1/K, regardless of whether some models have improved or degraded. QRA's weights adapt when refitted.

Despite these limitations, simple averaging should always be included as a benchmark — it is hard to beat consistently, and when QRA's improvements are small, the simpler method may be preferable for its robustness and transparency.

---

## Evaluating Probabilistic Forecasts

Evaluating a probabilistic forecast is fundamentally harder than evaluating a point forecast. For a point forecast, you compare the prediction to the actual value — the error is a single number (e.g., MAE = $25). For a probabilistic forecast, you must assess whether the *entire predicted distribution* matches the distribution of actual outcomes. This requires two concepts: **calibration** and **sharpness**.

### Calibration (Reliability)

<div class="definition-box">
<strong>Calibration (reliability):</strong> A probabilistic forecast is calibrated if the stated probabilities match the observed frequencies. If the model says "there is a 10% chance the price will be below $X," then approximately 10% of the time, the actual price should indeed be below $X. Calibration answers the question: "When the model says 90%, does it really happen 90% of the time?"
</div>

Calibration is assessed visually using the **reliability diagram** (also called the calibration plot):

1. For each nominal quantile level τ (e.g., 0.05, 0.10, 0.20, ..., 0.90, 0.95):
2. Compute the fraction of actual observations that fall below the predicted τ-quantile
3. Plot this observed fraction (y-axis) against the nominal level τ (x-axis)

A perfectly calibrated model produces points that lie exactly on the 45° diagonal — when the model says "10% chance," it happens 10% of the time; when it says "90%," it happens 90% of the time.

![Calibration (reliability) diagram](figures/08_calibration.png)

<p class="figure-caption">Figure 8.1 — A reliability diagram comparing calibration of different probabilistic forecasting models. The diagonal line represents perfect calibration. Points above the diagonal indicate overconfidence (intervals too narrow); points below indicate underconfidence (intervals too wide). The closer a model's curve follows the diagonal, the more trustworthy its probability statements.</p>

Systematic departures from the diagonal reveal specific types of miscalibration:

- **Points consistently above the diagonal (observed frequency > nominal level):** The model is **overconfident** — its prediction intervals are too narrow. For example, if the 90% interval only covers 75% of observations, the model is overstating its certainty. This is the most dangerous miscalibration for battery dispatch because the operator trusts intervals that are systematically too narrow, leading to unexpected losses during events the model did not anticipate.

- **Points consistently below the diagonal (observed frequency < nominal level):** The model is **underconfident** — its prediction intervals are too wide. The 90% interval covers 98% of observations. While less dangerous than overconfidence, this wastes information — the intervals are wider than necessary, reducing the specificity of dispatch decisions.

- **S-shaped curve:** The model is overconfident in the tails (extreme quantiles) and underconfident in the centre (near the median), or vice versa. This pattern often occurs with models that assume a symmetric distribution when the actual distribution is skewed.

- **Stepped or irregular curve:** The model has different calibration properties at different probability levels, often indicating that the training data was insufficient to learn the tail behaviour accurately.

<div class="example-box">
<strong>Real-world example — the cost of overconfidence:</strong> Suppose a battery operator uses a model whose 95% prediction interval actually covers only 80% of outcomes. The operator treats the upper bound of this interval as "almost certainly the highest the price will go." In reality, 20% of the time, the actual price exceeds this bound — potentially by thousands of dollars during spike events. The operator misses these high-price events because the battery has already discharged, thinking the price had peaked. Over a year, this overconfidence can cost hundreds of thousands of dollars in missed revenue.
</div>

### Sharpness

Calibration alone is not sufficient for a good probabilistic forecast. To see why, consider the following perfectly calibrated forecast: "For every time period, the 10th percentile is −$1,000 and the 90th percentile is $17,500." This forecast is perfectly calibrated (the actual price will almost always fall within this range, and the stated coverage is correct), but it is completely useless — it tells you nothing beyond what you already knew about the price range.

<div class="definition-box">
<strong>Sharpness:</strong> A measure of how concentrated (narrow) the prediction intervals are. A sharp forecast has narrow prediction intervals — it is making specific, informative predictions about the future. A blunt forecast has wide prediction intervals — it is hedging by covering a large range of outcomes. Sharpness is measured as the average width of prediction intervals across all time periods.
</div>

<div class="equation">

Sharpness = (1/N) · Σ_{t=1}^{N} (q_{0.9,t} − q_{0.1,t})

</div>

This formula computes the average width of the 80% prediction interval (the distance between the 90th and 10th percentile predictions) across all N time periods. A smaller value indicates sharper (more informative) predictions.

The goal of probabilistic forecasting is not simply to be calibrated — it is to be **the sharpest forecast subject to calibration**. This formulation, due to Gneiting, Balabdaoui, and Raftery (2007), captures the fundamental tradeoff: you want your prediction intervals to be as narrow as possible (maximum information) while still containing the stated proportion of actual outcomes (honest probability statements).

<div class="key-point">
<strong>The calibration-sharpness paradigm:</strong> "Maximise sharpness subject to calibration." This is the guiding principle for probabilistic forecast evaluation. A forecast that is perfectly calibrated but not sharp is useless (it says nothing specific). A forecast that is sharp but poorly calibrated is dangerous (its narrow intervals give false confidence). The ideal forecast is both calibrated and sharp — it makes specific predictions that are honestly stated.
</div>

<div class="example-box">
<strong>Analogy — weather forecasts:</strong> Saying "tomorrow's temperature will be between −40°C and 50°C" is perfectly calibrated (the actual temperature will always fall in this range) but has zero sharpness. Saying "tomorrow's temperature will be between 22°C and 24°C" is very sharp but might not be well calibrated (the actual temperature might frequently fall outside this narrow range). A good weather forecast says "between 18°C and 26°C" — sharp enough to be useful, calibrated enough to be trustworthy. The same logic applies to electricity price forecasts.
</div>

### The CRPS: A Single-Number Metric

Assessing calibration and sharpness separately is useful for diagnosis but impractical for model comparison — you need a single number that captures both. The **Continuous Ranked Probability Score (CRPS)** is that number.

<div class="definition-box">
<strong>Continuous Ranked Probability Score (CRPS):</strong> The standard single-number metric for evaluating probabilistic forecasts. CRPS generalises the Mean Absolute Error (MAE) from point forecasts to full distributions. It rewards both calibration (predicted probabilities match observed frequencies) and sharpness (prediction intervals are narrow). A lower CRPS is better. For a deterministic (point) forecast, CRPS reduces exactly to MAE. CRPS is measured in the same units as the variable being forecast (e.g., $/MWh for prices).
</div>

The CRPS is defined as:

<div class="equation">

CRPS = ∫-∞^∞ [F(x) − 1(x ≥ y)]² dx

</div>

where F(x) is the predicted cumulative distribution function (CDF), y is the actual observed value, and 1(x ≥ y) is the Heaviside step function (0 when x < y, 1 when x ≥ y).

This integral has an intuitive interpretation: it measures the area between the predicted CDF and the "perfect CDF" (a step function that jumps from 0 to 1 at the observed value). A forecast that places all its probability mass exactly on the observed value would achieve CRPS = 0.

In practice, when the forecast is given as a set of quantile predictions (as in GBT quantile regression or QRA), the CRPS can be approximated as the average pinball loss across all quantile levels:

<div class="equation">

CRPS ≈ (2/M) · Σ_{m=1}^{M} L_{τ_m}(y, q_{τ_m})

</div>

where L_τ is the pinball loss at quantile level τ, q_τ is the predicted quantile, y is the actual value, and M is the number of quantile levels. This approximation becomes exact as M increases and the quantile levels span [0, 1] uniformly.

<div class="definition-box">
<strong>Cumulative distribution function (CDF):</strong> A function F(x) that gives the probability that a random variable takes a value less than or equal to x. The CDF starts at 0 (for very small x) and increases monotonically to 1 (for very large x). If F(100) = 0.7, there is a 70% probability the value is at most 100. The CDF is related to the PDF (probability density function) — the CDF is the integral of the PDF.
</div>

<div class="key-point">
<strong>Why CRPS is the right metric:</strong> CRPS cannot be "gamed" by being maximally wide (unlike calibration alone). A forecast with trivially wide intervals ([−$1,000, $17,500]) achieves perfect calibration but terrible CRPS because the integral (area between CDFs) is large. To achieve good CRPS, the forecast must be both calibrated AND sharp. This makes CRPS the appropriate single metric for comparing probabilistic forecast models.
</div>

### Spike Coverage: The Critical Edge Case

Overall calibration can be misleading for electricity prices because spikes — the extreme high-price events that drive the majority of battery revenue — are rare. A model might have 90% overall coverage (its 90% prediction interval contains 90% of all observations) but systematically miss spikes because:

1. Spikes are rare (perhaps 2–5% of observations), so missing them has little effect on the overall coverage statistic
2. The model is overconfident specifically during volatile conditions — exactly when spikes occur
3. The tail behaviour of the model is poorly calibrated even though the centre of the distribution is well-calibrated

<div class="definition-box">
<strong>Spike coverage:</strong> The fraction of spike events (observations where the price exceeds a high threshold, e.g., $300/MWh or $1,000/MWh) that fall within the model's prediction interval. This is a conditional coverage metric — it evaluates calibration specifically during the events that matter most for battery revenue. A model with 90% overall coverage but only 60% spike coverage is dangerously overconfident about extreme events.
</div>

<div class="example-box">
<strong>Real-world example — the spike coverage trap:</strong> Consider two models, both with 90% overall coverage:

<strong>Model A:</strong> 90% overall coverage, 88% spike coverage. This model is well-calibrated even during extremes. Its prediction intervals widen appropriately when the market is volatile, capturing most spikes within their bounds.

<strong>Model B:</strong> 90% overall coverage, 55% spike coverage. This model achieves its overall coverage by being too wide during quiet periods (covering 95% of normal observations) and too narrow during volatile periods (covering only 55% of spikes). It is systematically wrong when being wrong is most expensive.

A battery operator using Model B would repeatedly be surprised by spikes — the battery would be discharged or idle when prices spike above the model's predicted range. Over a year, the revenue difference between Model A and Model B could be substantial.
</div>

<div class="key-point">
<strong>Always report spike coverage alongside overall coverage.</strong> A model with good overall calibration but poor spike coverage is more dangerous than a model with mediocre overall calibration but good spike coverage — because the battery's revenue is dominated by what happens during the 2–5% of time steps when prices spike.
</div>

---

## Conformal Prediction

### The Calibration Problem in Practice

Raw quantile forecasts from GBT or neural networks are frequently **miscalibrated** — the stated coverage does not match the actual coverage. This is not a failure of the models per se; it is an inherent difficulty of learning the tails of a distribution from limited data.

Common miscalibration patterns:

- **Overconfidence at the tails:** The 95% interval covers only 85% of observations. The model cannot accurately estimate the 2.5th and 97.5th percentiles because it has seen very few observations in those regions of the distribution.

- **Systematic bias:** All quantiles are shifted — the entire predicted distribution is too high or too low. This occurs when the recent market regime differs from the training period.

- **Conditional miscalibration:** Calibrated on average, but overconfident during volatile periods (when being right matters most) and underconfident during quiet periods (when accuracy matters least). This is the spike coverage problem described above.

**Conformal prediction** is a statistical framework that provides **finite-sample coverage guarantees** — a mathematically rigorous way to adjust prediction intervals so they achieve the desired coverage, regardless of the underlying model's idiosyncrasies.

<div class="definition-box">
<strong>Conformal prediction:</strong> A statistical framework for constructing prediction intervals (or sets) with guaranteed coverage probability. Unlike model-specific calibration methods, conformal prediction is <strong>model-agnostic</strong> — it works as a wrapper around any forecasting model, adjusting the model's outputs to achieve the desired coverage level. The key assumption is <strong>exchangeability</strong> — the calibration data and future test data come from the same distribution. Under this assumption, conformal prediction provides finite-sample (not just asymptotic) coverage guarantees.
</div>

### Split Conformal Prediction

The simplest and most practical variant is **split conformal prediction**. The idea is to measure how wrong the model has been on recent data, then widen the prediction intervals by an appropriate amount to account for this historical error.

<div class="definition-box">
<strong>Split conformal prediction:</strong> A method that divides the data into a training subset (for fitting the model) and a calibration subset (for measuring the model's errors). The errors on the calibration subset are used to compute a correction term that is added to the model's prediction intervals, guaranteeing that the widened intervals achieve the desired coverage on future data. The "split" refers to the division of data into training and calibration subsets.
</div>

The procedure, step by step:

**Step 1: Split the data.** Divide the available data into two subsets: a training set (for fitting the forecasting model) and a calibration set (for measuring the model's errors). A typical split might be 80% training, 20% calibration. The calibration set must be held out completely — the model should never have seen it during training.

**Step 2: Fit the model.** Train the quantile forecasting model (GBT, neural network, QRA) on the training subset. This produces raw quantile predictions that may or may not be well-calibrated.

**Step 3: Compute conformity scores.** On the calibration subset, compute a **conformity score** for each observation — a number measuring how "non-conforming" the observation is relative to the model's prediction.

<div class="definition-box">
<strong>Conformity score:</strong> A numerical measure of how poorly the model's prediction matches the actual outcome. For prediction intervals, a common conformity score is s_i = max(q_{low,i} − y_i, y_i − q_{high,i}) — this is positive when the observation falls outside the interval (measuring by how much) and negative when it falls inside (measuring the margin). Larger conformity scores indicate worse calibration.
</div>

For a prediction interval [q_low, q_high], two common conformity scores are:

- **Absolute residual:** s_i = |y_i − y\*_i| (how far the observation is from the point prediction)
- **Signed interval distance:** s_i = max(q_low,i − y_i, y_i − q_high,i) (how far outside the interval the observation falls; negative if inside)

**Step 4: Compute the correction.** Sort the conformity scores and find the ⌈(1−α)(1 + 1/n)⌉-th smallest score, where α is the desired miscoverage rate (e.g., α = 0.1 for 90% coverage) and n is the number of calibration observations. This score is the **conformal correction** — the amount by which to widen the prediction intervals.

**Step 5: Adjust predictions.** At prediction time, expand the model's raw prediction intervals by the conformal correction:

<div class="equation">

Adjusted interval = [q_low − correction, q_high + correction]

</div>

<div class="key-point">
<strong>The coverage guarantee:</strong> Under the assumption that the calibration observations and future test observations are <strong>exchangeable</strong> (essentially, drawn from the same distribution in any order), the adjusted prediction intervals have coverage ≥ (1 − α) in finite samples. This is a non-asymptotic guarantee — it holds for any sample size, any model, and any data distribution. The only assumption is exchangeability.
</div>

<div class="definition-box">
<strong>Exchangeability:</strong> A statistical assumption that the joint distribution of a set of random variables is invariant to permutation — reordering the observations does not change their probability. This is weaker than the assumption that observations are independent and identically distributed (i.i.d.), which also requires independence. For time series data, exchangeability is technically violated (today's price depends on yesterday's), but rolling calibration (described below) provides a practical approximation.
</div>

### Adaptive Conformal Prediction

Standard split conformal prediction has a significant limitation: it produces a **constant correction** — the same amount is added to every prediction interval, regardless of the model's inherent uncertainty. This is suboptimal:

- During **quiet market periods**, the model's raw intervals are already narrow and accurate. The conformal correction makes them unnecessarily wide — sacrificing sharpness for no real calibration benefit.
- During **volatile market periods**, the raw intervals might still be too narrow despite being wider than average. A constant correction may not add enough width where it is needed most.

**Adaptive conformal prediction** addresses this by scaling the conformity scores by the model's predicted uncertainty:

<div class="equation">

s_i = |y_i − y\*_i| / σ*_i

</div>

where σ*_i is a measure of the model's predicted uncertainty at observation i — for example, the width of the raw prediction interval (q_high,i − q_low,i) or the standard deviation of the quantile predictions.

<div class="definition-box">
<strong>Adaptive conformal prediction:</strong> A variant of conformal prediction that normalises the conformity scores by the model's predicted uncertainty. This produces prediction intervals that are wider when the model is uncertain and narrower when the model is confident — matching the intuition that intervals should adapt to the current state of the market, not be uniformly wide.
</div>

By normalising the scores, the conformal correction is applied proportionally: a large correction is applied when σ*_i is large (the model is already uncertain), and a small correction when σ*_i is small (the model is confident). The result is more informative intervals that vary in width with the model's confidence level.

<div class="example-box">
<strong>Real-world example — adaptive vs. constant correction:</strong> Consider a January day with mild weather and low demand. The GBT model predicts a narrow 80% interval of [$40, $65]. Standard conformal prediction adds a constant correction of $50 (computed from recent volatile periods), producing [$-10, $115] — unnecessarily wide for a calm market.

Adaptive conformal prediction recognises that the model's raw interval is narrow (signalling confidence) and applies a smaller correction proportional to the interval width. The result might be [$30, $75] — still properly calibrated but much sharper.

On a summer afternoon with heatwave warnings, the model's raw interval is already wide: [$50, $500]. Adaptive conformal prediction applies a larger correction, producing [$20, $700] — appropriate for the genuinely uncertain conditions.
</div>

### Practical Considerations for Conformal Prediction

**Calibration window.** The conformity scores should be computed on recent data — typically the last 14 to 60 days. Using calibration data from six months ago may not reflect the model's current accuracy, because the market structure evolves (new generators, changing demand patterns, seasonal shifts). A shorter window adapts faster to changes but has higher variance (fewer calibration observations); a longer window is more stable but may not reflect current conditions.

<div class="definition-box">
<strong>Calibration window:</strong> The period of recent data used to compute the conformity scores for conformal prediction. A rolling calibration window (e.g., the last 30 days) is updated as new data arrives, ensuring the conformal correction reflects the model's current accuracy. The choice of window length trades off adaptability (shorter windows respond faster to changes) against stability (longer windows provide more reliable estimates).
</div>

**Rolling recalibration.** Recompute the conformal correction daily or weekly as new observations arrive. This maintains approximate coverage even under non-stationarity (slowly changing market conditions). The formal coverage guarantee technically requires exchangeability (which non-stationarity violates), but rolling recalibration is robust in practice — the correction adapts to the model's recent error distribution.

**Quantile-specific correction.** Apply separate conformal corrections to each quantile level rather than a single uniform correction. The model might be well-calibrated at the median (50th percentile) but systematically overconfident at the 95th percentile — applying the same correction to both would over-correct the median and under-correct the 95th percentile. Quantile-specific corrections allow fine-grained calibration across the entire distribution.

---

## The Calibration-Sharpness Tradeoff in Practice

### Putting It All Together

The practical pipeline for producing calibrated, sharp probabilistic forecasts combines the techniques from this chapter:

1. **Train the best possible quantile model.** Use QRA to combine GBT, LEAR, and optionally neural network forecasts. This produces quantile forecasts that are reasonably sharp (each component model contributes its best quantile estimates) and approximately calibrated (QRA's pinball-loss fitting provides some calibration).

2. **Apply conformal calibration.** Use adaptive conformal prediction to adjust the quantile forecasts, ensuring the coverage guarantee. If the base model is already well-calibrated, the conformal correction will be small and sharpness is largely preserved. If the base model is miscalibrated, the correction widens the intervals to achieve proper coverage.

3. **Evaluate on a held-out test set.** Assess both calibration (reliability diagram) and sharpness (average interval width). Report CRPS as the single-number summary, and spike coverage as the critical tail metric.

4. **Iterate.** If the model is well-calibrated without conformal adjustment, the adjustment is minimal and sharpness is maximised. If the model is poorly calibrated, investigate the root cause — is it a model specification issue (wrong features, insufficient data), a regime change (the market has shifted), or a tail estimation problem (insufficient extreme observations)?

### Calibration for Battery Dispatch: Asymmetric Costs

For battery dispatch, the cost of miscalibration is **asymmetric** — different types of errors have different economic consequences:

- **Missing a spike (price above the upper quantile):** The battery fails to discharge during a high-price period. This is a direct revenue loss — potentially thousands of dollars per missed event.
- **Missing a dip (price below the lower quantile):** The battery fails to charge during a low-price period. This is an opportunity cost — the battery could have charged cheaply but instead charges at a higher price later.

The cost of missing a spike is typically much larger than missing a dip because spikes are more extreme in magnitude than dips. A $5,000 missed spike is a far larger loss than a $30 missed charging opportunity.

This asymmetry suggests using **asymmetric calibration** — targeting higher coverage in the upper tail than the lower tail. Instead of symmetric prediction intervals [q_0._1_0, q_0._9_0], you might use [q_0._1_5, q_0._9_5] to provide more headroom on the upside. The 80% coverage is maintained, but it is distributed asymmetrically to protect against the more costly error.

<div class="definition-box">
<strong>Asymmetric calibration:</strong> A calibration approach that targets different coverage levels for the upper and lower tails of the forecast distribution. When the cost of errors is asymmetric (missing a spike is more expensive than missing a dip), the prediction intervals should be asymmetric to provide more protection against the costlier error. Chapter 9's chance-constrained dispatch formalises this intuition.
</div>

<div class="example-box">
<strong>Real-world example — asymmetric risk in battery dispatch:</strong> The Mannum BESS (100 MW / 200 MWh) in SA1 considers whether to discharge at 5 PM. The symmetric 80% interval is [$80, $250]. The asymmetric 80% interval (with more upside protection) is [$100, $400]. The battery operator using symmetric intervals might discharge at $250, thinking prices are unlikely to go higher. The operator using asymmetric intervals holds back, knowing there is a meaningful chance prices could reach $400 or beyond. On days when a spike materialises, the asymmetric approach captures significantly more revenue. On days when prices stay moderate, the symmetric approach was slightly better. Over a year, the asymmetric approach wins because the spike events contribute disproportionately to annual revenue.
</div>

---

## Copula Scenario Generation

The probabilistic forecasts developed in this chapter produce **marginal** quantile predictions — for each half-hour period independently, a set of predicted quantiles. But dispatch optimisation in Chapter 9 needs complete **joint** price trajectories: 48-period paths where the correlation between adjacent hours is realistic. Simply sampling each period independently produces incoherent scenarios — paths that zigzag randomly between quantiles rather than showing the smooth, persistent price structure observed in real markets.

<div class="key-point">
<strong>Why this matters for dispatch:</strong> Independent sampling misses a critical feature of electricity prices: when 5 PM is expensive, 6 PM is usually expensive too. A scenario that has $500 at 5 PM and $30 at 6 PM is unrealistic. The stochastic and robust optimisation methods in Chapter 9 rely on realistic joint futures to make good hedging decisions — garbage scenarios produce garbage hedges.
</div>

### Marginals versus dependence

A probabilistic forecast gives the **marginal distribution** for each period — the full range of possible prices at 5 PM, independently of what happens at 6 PM. But dispatch decisions depend on the **joint distribution** — the probability that 5 PM *and* 6 PM are both high, or that prices are low all afternoon then spike in the evening.

Two sets of marginal forecasts can have identical per-period distributions but very different joint behaviour:

- **Positively correlated:** High prices at 5 PM tend to occur with high prices at 6 PM (the realistic case — price persistence)
- **Independent:** High at 5 PM tells you nothing about 6 PM (unrealistic — ignores autocorrelation)
- **Negatively correlated:** High at 5 PM makes low at 6 PM more likely (generally unrealistic for adjacent periods)

The tool that separates marginals from dependence is the **copula**.

### Sklar's theorem and the copula idea

<div class="definition-box">
<strong>Copula:</strong> A multivariate distribution function whose marginals are all uniform on [0, 1]. A copula captures the dependence structure between random variables, completely separate from their individual distributions. Any joint distribution can be decomposed into its marginals and a copula — this is Sklar's theorem.
</div>

<div class="definition-box">
<strong>Sklar's theorem:</strong> Any multivariate distribution can be written as a copula applied to the individual marginal distributions. Conversely, given any set of marginal distributions and any copula, the combination defines a valid joint distribution. This theorem is the theoretical foundation for copula-based scenario generation: we can take our calibrated marginal quantile forecasts (per period) and combine them with a copula (fitted to historical dependence patterns) to produce coherent joint scenarios.
</div>

The procedure has three steps:

1. **Fit marginals:** The quantile forecasts from QRA or conformal prediction define the marginal distribution for each period. These are already calibrated per-period.

2. **Fit the copula:** Transform historical forecast errors into uniform variables using the **probability integral transform** (PIT), then fit a copula to the resulting uniform variables. The copula captures how forecast errors are correlated across periods — when the model underpredicts at 5 PM, does it also underpredict at 6 PM?

3. **Sample scenarios:** Draw from the fitted copula (producing correlated uniform variables), then transform back through the per-period marginal quantile functions to obtain price scenarios in dollar space.

<div class="definition-box">
<strong>Probability integral transform (PIT):</strong> The transformation that converts a random variable to a uniform [0, 1] variable using its own CDF. If X has CDF F, then F(X) is uniformly distributed. Applied to forecast residuals, the PIT produces the "copula-scale" data needed to fit the dependence structure. The inverse PIT (applying the quantile function) transforms back from uniform to dollar space.
</div>

### The Gaussian copula

The simplest and most widely used copula is the **Gaussian copula**, which models dependence using a multivariate normal distribution:

1. Transform historical PIT residuals to standard normal using the inverse normal CDF (Phi^{-1})
2. Compute the correlation matrix R of the transformed residuals across all 48 periods
3. To generate a scenario: draw z from a multivariate normal N(0, R), apply the normal CDF to get uniform variables u = Phi(z), then apply the per-period quantile function to get prices

<div class="definition-box">
<strong>Gaussian copula:</strong> A copula derived from the multivariate normal distribution. It captures linear dependence (correlation) between variables. Its main advantage is simplicity — it is fully specified by a correlation matrix. Its main limitation is that it has symmetric tail dependence: it does not naturally model the tendency for extreme prices to cluster more strongly than moderate prices.
</div>

The Gaussian copula works well for electricity prices in most conditions because the dominant dependence pattern — price persistence across adjacent hours — is approximately linear. However, it can understate **tail dependence**: the tendency for extreme prices to be even more correlated than normal prices. During heatwave events, for example, all evening periods tend to spike together more strongly than the Gaussian copula predicts.

### When tails matter: vine copulas

For applications where tail dependence is critical — particularly the stochastic dispatch of batteries that earn disproportionately from spikes — more flexible copula families exist:

- **t-copula:** Like the Gaussian copula but with heavier tails, controlled by a degrees-of-freedom parameter. It captures the tendency for extremes to cluster.
- **Vine copulas:** Build a multivariate copula from a cascade of bivariate copulas, each chosen to fit the specific dependence pattern between each pair of variables. A vine copula on 48 periods decomposes into many bivariate building blocks — the copula between periods 1 and 2, between periods 2 and 3 (conditional on 1), and so on. Each building block can be a different family (Gaussian, t, Clayton, Gumbel), chosen to match the observed dependence.

<div class="definition-box">
<strong>Tail dependence:</strong> The tendency for extreme values to occur together. If two variables have high tail dependence, knowing that one is in its 99th percentile makes it more likely that the other is also in its 99th percentile — more so than a Gaussian model would predict. For electricity prices, tail dependence means that spike events tend to affect multiple consecutive periods simultaneously.
</div>

<div class="example-box">
<strong>Practical recommendation:</strong> Start with the Gaussian copula — it is easy to fit, interpret, and debug. Compare the rank correlation of generated scenarios against historical data. If the tail behaviour is inadequate (scenarios understate the clustering of extreme prices), upgrade to a t-copula with estimated degrees of freedom. Vine copulas are a further refinement but add significant complexity; reserve them for cases where you can demonstrate a material improvement in dispatch capture ratio.
</div>

### Cross-market dependence: energy and FCAS

The copula framework extends naturally to modelling the joint distribution of energy and FCAS prices (introduced in Chapter 11). Energy and FCAS prices have a complex dependence structure:

- In normal conditions, FCAS prices are low and nearly independent of energy prices
- During contingency events, FCAS prices spike while energy prices may or may not spike
- When enablement is scarce (many generators providing FCAS, reducing energy supply), energy and FCAS prices become positively correlated

Fitting a copula across both energy and FCAS price series produces scenarios where these market interactions are realistic — essential for the co-optimised dispatch developed in Chapter 11.

### Sampling and scenario count

The number of scenarios is a practical trade-off:

- **Too few (N < 20):** The scenarios do not adequately represent the distribution. The stochastic LP may overfit to the specific scenarios drawn.
- **Too many (N > 500):** The stochastic LP becomes computationally expensive (it grows linearly in N). Diminishing returns set in around N = 100–200 for a 48-period problem.
- **Practical default:** N = 50–100 copula-generated scenarios provides a good balance for the dispatch optimisation in Chapter 9.

<div class="key-point">
<strong>Copula scenarios feed directly into Chapter 9.</strong> The stochastic MPC and chance-constrained MPC both consume price scenarios as input. Replacing independently sampled scenarios with copula-generated ones typically improves the dispatch capture ratio by 1–3 percentage points — because the hedging decisions are based on realistic joint futures rather than incoherent random paths. This improvement is largest during volatile periods, which are exactly the periods that drive battery revenue.
</div>

![Copula scenarios](figures/08_copula_scenarios.png)

<p class="figure-caption">Figure 8.5 — Left: independent scenarios (each period sampled independently — note the unrealistic zigzagging). Right: copula scenarios (preserving cross-period correlation — note the smooth, persistent price paths that resemble real market behaviour).</p>

---

## Glossary

| Term | Definition |
|------|-----------|
| **Probabilistic forecast** | A forecast that describes the full range of possible outcomes and their probabilities |
| **Prediction interval** | A range expected to contain the actual value with a specified probability |
| **Quantile forecast** | Predictions at multiple probability levels characterising the forecast distribution |
| **Density forecast** | The full probability density function of the predicted variable |
| **Scenario forecast** | Multiple complete future trajectories sampled from the forecast distribution |
| **PDF (probability density function)** | A function showing the relative likelihood of each possible value |
| **CDF (cumulative distribution function)** | A function giving the probability of being at or below each value |
| **Forecast combination principle** | The finding that averaging forecasts from different models almost always improves accuracy |
| **QRA (Quantile Regression Averaging)** | A method combining multiple point forecasts into calibrated quantile forecasts |
| **Calibration (reliability)** | Whether stated probabilities match observed frequencies |
| **Reliability diagram** | A plot of observed coverage vs. nominal coverage for assessing calibration |
| **Overconfidence** | When prediction intervals are too narrow (stated coverage exceeds actual coverage) |
| **Underconfidence** | When prediction intervals are too wide (actual coverage exceeds stated coverage) |
| **Sharpness** | How narrow (concentrated) the prediction intervals are |
| **CRPS** | Continuous Ranked Probability Score — the standard metric for probabilistic forecasts |
| **Spike coverage** | Fraction of extreme price events captured within the prediction interval |
| **Conformal prediction** | A model-agnostic framework for constructing prediction intervals with coverage guarantees |
| **Split conformal prediction** | A variant using a held-out calibration set to compute interval corrections |
| **Conformity score** | A measure of how poorly a model's prediction matches the actual outcome |
| **Exchangeability** | The assumption that data ordering does not affect the joint distribution |
| **Adaptive conformal prediction** | A variant that scales corrections by the model's predicted uncertainty |
| **Calibration window** | The period of recent data used to compute conformal corrections |
| **Rolling recalibration** | Periodically recomputing conformal corrections as new data arrives |
| **Asymmetric calibration** | Targeting different coverage levels for upper and lower tails |
| **Calibration-sharpness paradigm** | "Maximise sharpness subject to calibration" — the guiding principle |
| **Copula** | A multivariate distribution capturing dependence structure, separate from marginals |
| **Sklar's theorem** | Any joint distribution decomposes into marginals and a copula |
| **Probability integral transform (PIT)** | Transformation converting a variable to uniform [0,1] via its own CDF |
| **Gaussian copula** | Copula derived from the multivariate normal; captures linear dependence |
| **Tail dependence** | Tendency for extreme values to occur together more than a Gaussian model predicts |
| **Vine copula** | Multivariate copula built from a cascade of bivariate copula building blocks |
| **Rank correlation** | Correlation measured on ranks rather than values; invariant to marginal transforms |
| **Scenario coherence** | Property that jointly sampled scenarios preserve realistic cross-period dependence |

## Summary

Probabilistic forecasts are essential for risk-aware battery dispatch because they quantify the uncertainty that point forecasts conceal. Quantile Regression Averaging (QRA) combines multiple point forecasts into a single probabilistic forecast with quantile-specific weights — different models contribute different amounts at different probability levels. Evaluation of probabilistic forecasts requires both calibration (stated coverage matches actual coverage) and sharpness (prediction intervals are as narrow as possible), captured in a single metric by the CRPS. Spike coverage deserves separate attention because overall calibration can mask dangerous overconfidence during the extreme events that drive battery revenue. Conformal prediction provides a model-agnostic safety net — finite-sample coverage guarantees achieved by adjusting prediction intervals based on the model's recent errors. Adaptive conformal prediction improves on the basic method by scaling corrections to the model's predicted uncertainty, producing intervals that are wider during volatile periods and narrower during calm periods. The calibration-sharpness paradigm — "maximise sharpness subject to calibration" — guides the entire evaluation process, and asymmetric calibration accounts for the fact that missing a price spike costs far more than missing a price dip.
