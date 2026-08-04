# 14. Failure Analysis and Explainability

## Decision Regret, Not Forecast Error

Most forecast evaluation frameworks focus on accuracy: MAE, RMSE, CRPS, calibration. These metrics measure how close the forecast was to reality. But a battery operator does not care about forecast accuracy in the abstract — they care about whether the forecast led to the right **decision**. A forecast error that did not flip a charge or discharge decision cost the operator exactly zero dollars. A small forecast error that flipped a decision during a price spike may have cost hundreds of thousands of dollars.

This chapter reframes forecast evaluation around **decision regret**: the gap between what the dispatch actually earned and what it would have earned with perfect foresight. This is the metric that matters commercially, and it diverges from traditional forecast error in ways that are both surprising and consequential.

<div class="definition-box">
<strong>Decision regret:</strong> The difference in revenue between a dispatch schedule optimised on a forecast and a dispatch schedule optimised on the actual prices (perfect foresight). Formally: regret_t = revenue_perfect_t − revenue_forecast_t. A positive regret means the forecast-driven dispatch left money on the table. A zero regret means the forecast error, however large, did not affect the dispatch decision. Regret is measured in dollars, not error units.
</div>

The distinction between forecast error and decision regret is subtle but essential. Consider three scenarios for a half-hour period where the battery must decide whether to discharge:

<div class="example-box">
<strong>Three scenarios illustrating the error-regret divergence:</strong>

<strong>Scenario A — Large error, zero regret.</strong> The forecast predicts $80/MWh; the actual price is $120/MWh. The forecast error is $40/MWh. But both $80 and $120 are well above the charge cost, so the battery discharges in both the forecast-driven and perfect-foresight cases. The dispatch decision is identical. Regret = $0.

<strong>Scenario B — Small error, large regret.</strong> The forecast predicts $55/MWh; the actual price is $65/MWh. The forecast error is only $10/MWh. But $55 is below the threshold where the LP decides to hold charge (because a later period looks better), while $65 would have triggered a discharge. The battery holds when it should have discharged. Regret = $65 · 100 MW · 0.5 h = $3,250 for this single half-hour.

<strong>Scenario C — Small error, catastrophic regret.</strong> The forecast predicts $200/MWh; the actual price is $5,000/MWh. The forecast error is $4,800/MWh. But this is not a symmetric error — the $200 forecast caused the battery to discharge in an earlier, less profitable period, missing the spike entirely. The perfect-foresight battery would have held its charge for this period. Regret = ($5,000 − $200) · 100 MW · 0.5 h = $240,000 in a single half-hour.
</div>

These scenarios reveal a fundamental asymmetry: forecast errors in the **middle of the price distribution** rarely affect dispatch decisions, because the battery is either clearly charging or clearly discharging regardless of moderate price variations. Forecast errors at **decision boundaries** — near the threshold where the LP switches between charge, hold, and discharge — are where regret concentrates.

<div class="key-point">
<strong>The core insight:</strong> Decision regret is not proportional to forecast error. It depends on the interaction between the error and the dispatch optimisation. Regret concentrates where the error crosses a decision boundary — and most of the dollars at risk sit in the tails of the price distribution. Evaluating a forecast by its MAE is like evaluating a surgeon by the number of cuts: the metric is related to the outcome but misses what actually matters.
</div>

### Computing Decision Regret

The computation is straightforward once you have the Chapter 9 backtest infrastructure:

1. **Run the dispatch LP with the forecast** to obtain the forecast-driven schedule and revenue for each day.
2. **Run the dispatch LP with the actual prices** (perfect foresight) to obtain the optimal schedule and revenue for each day.
3. **Compute regret** as the difference: regret_t = revenue_perfect_t − revenue_forecast_t.

The dispatch LP from Chapter 5 already handles both cases — the only difference is the price input. The perfect-foresight run uses actual realised prices; the forecast-driven run uses the quantile forecasts from Chapter 8.

The **capture ratio** from Chapter 9 is the complement of normalised regret:

<div class="equation">

capture_ratio = revenue_forecast / revenue_perfect = 1 − (regret / revenue_perfect)

</div>

A capture ratio of 85% means the forecast-driven dispatch captured 85% of perfect-foresight revenue, implying 15% normalised regret. The relationship is simple, but expressing results in both forms is useful: capture ratio describes overall performance; regret highlights specific days where money was lost.

### Regret by Regime

The most revealing analysis is to split regret by **price regime**. Define two regimes using the definitions from Chapter 3:

- **Calm intervals:** The half-hour price remains between the 5th and 95th percentiles of the trailing 90-day distribution.
- **Spike intervals:** The half-hour price exceeds the 95th percentile.

Now compute total regret within each regime:

| Regime | Share of intervals | Share of regret | Regret per interval |
|--------|-------------------|-----------------|---------------------|
| Calm | ~90% | ~15–25% | Low |
| Spike | ~10% | ~75–85% | Very high |

The tail dominates. Roughly 10% of intervals — the spikes — account for 75–85% of total decision regret. This is the most important empirical finding in this chapter, and it has direct implications for how you allocate modelling effort:

<div class="key-point">
<strong>The regret concentration law:</strong> For battery dispatch in the NEM, approximately 80% of decision regret comes from approximately 10% of trading intervals — the spikes. Improving forecast accuracy during calm periods has minimal economic impact. Improving spike detection, even marginally, has enormous economic impact. A model with 20% higher overall MAE but 10% better spike detection will earn more money than one with lower MAE but worse tail performance.
</div>

![Decision regret by regime](figures/14_regret_by_regime.png)

<p class="figure-caption">Figure 14.1 — Decision regret by price regime. Left: histogram of per-interval regret for calm vs spike intervals. The spike regime has a heavy right tail — a few intervals account for the majority of total regret. Right: cumulative regret over the backtest period. The step-function shape shows that regret accumulates in bursts during spike events, with long flat stretches during calm periods.</p>

---

## The Asymmetric Cost of Error

Forecast errors are not symmetric in their economic consequences. There are two distinct failure modes, and they have very different costs:

### Missed Spikes: The Expensive Failure

A **missed spike** occurs when the forecast fails to predict a high-price event, causing the battery to charge or hold when it should have discharged. The cost is the **foregone revenue** — the money the battery would have earned by discharging during the spike.

<div class="definition-box">
<strong>Missed spike:</strong> A price event where the actual price significantly exceeded the forecast, and the forecast-driven dispatch did not discharge (or discharged at a lower rate than optimal). The cost is the perfect-foresight discharge revenue minus the actual dispatch revenue during the spike period. For a 100 MW / 200 MWh battery, a fully missed spike at $5,000/MWh costs up to $5,000 · 100 MW · 0.5 h = $250,000 per half-hour.
</div>

The magnitude of missed-spike regret depends on three factors:

1. **Spike height.** A missed spike at $500/MWh is costly; a missed spike at $5,000/MWh is catastrophic. The NEM price cap is $17,500/MWh (as of 2025), so the theoretical maximum cost of a single fully missed half-hour is $17,500 · 100 MW · 0.5 h = $875,000. In practice, spikes above $5,000/MWh are rare but account for a disproportionate share of total available revenue.

2. **State of charge.** A battery with 80% state of charge can discharge for nearly two hours at full power; a battery at 20% can discharge for less than 30 minutes. The dispatch LP's prior decisions about charging and discharging determine how much energy is available when a spike arrives. A missed spike is doubly costly if the battery charged during a previous period that turned out to be expensive.

3. **Duration.** Single-interval spikes (30 minutes) are less costly than sustained spike events lasting 2–4 hours. The battery's 200 MWh capacity is designed for two hours of full-power discharge, so multi-hour spikes can be fully captured only if the battery is fully charged at the start.

### False Alarms: The Hidden Cost

A **false alarm** occurs when the forecast predicts a spike that does not materialise, causing the battery to hold charge (waiting for the spike) when it should have discharged at a moderate price, or to discharge into what turns out to be a low-price period.

<div class="definition-box">
<strong>False alarm:</strong> A price event where the forecast significantly exceeded the actual price, and the forecast-driven dispatch made a suboptimal decision as a result. The direct cost is the revenue foregone by misallocating the dispatch. The indirect cost includes unnecessary battery cycling (degradation) and the opportunity cost of the energy that was dispatched at the wrong time.
</div>

False alarms are less visible than missed spikes but still costly:

1. **Cycle degradation.** Each full cycle reduces the battery's lifetime capacity. For an LFP battery like Mannum, cycle degradation is relatively low (LFP chemistry tolerates 4,000–6,000 cycles to 80% capacity), but each unnecessary cycle has a real cost: the total capital cost divided by the total expected cycles gives a per-cycle degradation cost of roughly $10–$30/MWh for a utility-scale LFP system.

2. **Opportunity cost.** A battery that discharged at $80/MWh (based on a false spike forecast) cannot discharge again at $200/MWh two hours later — it needs time to recharge. The opportunity cost is the difference between the price at which the battery should have discharged and the price at which it actually discharged.

3. **Round-trip efficiency loss.** Every charge-discharge cycle loses 10–15% of the energy to round-trip inefficiency. A false alarm that triggers an unnecessary cycle wastes this energy.

### Quantifying the Asymmetry

The asymmetry between missed spikes and false alarms is stark:

| Failure mode | Per-event cost | Annual impact (SA1) |
|-------------|---------------|-------------------|
| Missed spike ($500–$1,000/MWh), 20–50/year | $5K–$25K | $100K–$500K |
| Missed spike ($1,000–$5,000/MWh), 5–15/year | $25K–$250K | $125K–$1.5M |
| Missed spike (>$5,000/MWh), 1–3/year | $250K–$875K | $250K–$2.5M |
| False alarm, 30–80/year | $2K–$15K | $60K–$400K |

Missed spikes dominate. A single missed extreme spike can cost more than a year's worth of false alarms combined. This asymmetry has a direct implication for model development: **under-predicting spikes is much more costly than over-predicting them**.

<div class="key-point">
<strong>The asymmetric loss principle:</strong> For battery dispatch, the loss function is not symmetric. A dollar of under-prediction during a spike costs more than a dollar of over-prediction, because missed spikes are irreversible (the revenue opportunity is gone) while false alarms are partially recoverable (the battery can recharge and try again). This is why the quantile-based dispatch strategies from Chapter 9 — which explicitly account for forecast uncertainty — outperform naive point-forecast dispatch.
</div>

<div class="example-box">
<strong>Worked example — asymmetric cost in dollars:</strong> On a summer day in SA1, the model forecasts a peak of $300/MWh at 5:30 PM. The actual peak is $4,200/MWh at 5:00 PM.

The forecast-driven dispatch discharges at 5:30 PM (forecast peak) and earns $300 · 100 MW · 0.5 h = $15,000. The perfect-foresight dispatch discharges at 5:00 PM (actual peak) and earns $4,200 · 100 MW · 0.5 h = $210,000. Regret = $210,000 − $15,000 = $195,000.

Now consider the reverse error: the model forecasts $4,200/MWh but the actual price is $300/MWh. The forecast-driven dispatch discharges at the forecast peak, earning $300 · 100 MW · 0.5 h = $15,000. The perfect-foresight dispatch might also have discharged this period (if $300 was still the best option), or might have held for a better opportunity. In the worst case, the regret is the difference between $300 and whatever better opportunity was missed — perhaps $30,000–$50,000.

The under-prediction regret ($195,000) is 4–6x the over-prediction regret ($30,000–$50,000). The loss is asymmetric, and the asymmetry grows with spike height.
</div>

---

## Regime-Conditional Evaluation

### Why Overall Metrics Lie

Chapter 5 introduced MAE and CRPS as forecast evaluation metrics. These are aggregate measures — they average forecast performance over all time periods in the evaluation window. The problem is that the average is dominated by calm intervals (which comprise ~90% of the data), masking performance during the spikes that drive the P&L.

<div class="example-box">
<strong>A concrete illustration of metric masking:</strong> Consider two models evaluated over a 90-day backtest (4,320 half-hour intervals):

<strong>Model A:</strong> Overall MAE = $18/MWh. Calm-interval MAE = $12/MWh. Spike-interval MAE = $180/MWh.

<strong>Model B:</strong> Overall MAE = $22/MWh. Calm-interval MAE = $18/MWh. Spike-interval MAE = $85/MWh.

Model A has lower overall MAE — it "wins" on the standard metric. But Model B has dramatically better spike performance. When we run both through the dispatch LP, Model B achieves a capture ratio of 83% while Model A achieves only 71%. Model B earns roughly $150,000 more than Model A over the 90-day period.

The overall MAE rewarded Model A for being slightly better during 3,888 calm intervals, while ignoring that it was catastrophically worse during the 432 spike intervals. The P&L does not make this mistake.
</div>

This is not a hypothetical — it is a predictable consequence of the mathematics. The contribution of an interval to overall MAE is proportional to its forecast error. The contribution of an interval to revenue is proportional to its price. Since spike intervals have both higher prices and higher errors, the revenue-weighted importance of spike intervals far exceeds their count-weighted importance in the MAE calculation.

### Regime-Conditional MAE and CRPS

The solution is to compute metrics separately for each regime:

<pre>
MAE_calm  = (1/n_calm)  · Σ |y_t − y*_t|   for t ∈ calm intervals

MAE_spike = (1/n_spike) · Σ |y_t − y*_t|   for t ∈ spike intervals
</pre>

And similarly for CRPS:

<pre>
CRPS_calm  = (1/n_calm)  · Σ CRPS(F_t, y_t)   for t ∈ calm intervals

CRPS_spike = (1/n_spike) · Σ CRPS(F_t, y_t)   for t ∈ spike intervals
</pre>

Report both alongside the overall metric. The regime-conditional metrics answer different questions:

| Metric | Question it answers |
|--------|-------------------|
| MAE_calm | How well does the model track routine price movements? |
| MAE_spike | How well does the model detect and size extreme price events? |
| CRPS_calm | How well-calibrated is the probabilistic forecast during normal conditions? |
| CRPS_spike | How well-calibrated is the probabilistic forecast during extremes? |

### Regime-Conditional Calibration

The calibration diagnostics from Chapter 8 — PIT histograms, reliability diagrams, coverage plots — should also be split by regime. A model can be perfectly calibrated overall while being systematically overconfident during spikes (the intervals that matter most).

<div class="definition-box">
<strong>Regime-conditional calibration:</strong> Evaluating the calibration of a probabilistic forecast separately within each price regime (calm vs spike). A model is regime-conditionally calibrated if its prediction intervals have the correct coverage rate within each regime independently. For example, the 90% interval should contain 90% of outcomes during calm periods AND 90% of outcomes during spike periods. Overall calibration allows the model to be overconfident during spikes if it compensates by being underconfident during calm periods — regime-conditional calibration prevents this.
</div>

In practice, most machine learning models trained on the full dataset are **well-calibrated during calm intervals and poorly calibrated during spikes**. This is because the calibration loss function weights all intervals equally, and calm intervals outnumber spike intervals roughly 9:1. The model minimises its calibration error by fitting the calm majority well, even if this means the spike minority is poorly served.

The practical fix is the conformal calibration from Chapter 8, applied separately within each regime. But the diagnostic — splitting calibration by regime — comes first. You cannot fix what you have not measured.

---

## Attribution on the Model

When a forecast fails, the next question is **why**. What drove the model's prediction? What features pushed it toward the wrong answer? This section introduces the tools for attributing forecast errors to specific model inputs.

### SHAP: Feature Attribution for Gradient-Boosted Models

The gradient-boosted tree (GBT) models from Chapter 7 are powerful but opaque — they combine hundreds of trees, each splitting on different features, making it impossible to manually trace how an input variable influenced the prediction. **SHAP** (SHapley Additive exPlanations) provides a principled decomposition of any single prediction into feature contributions.

<div class="definition-box">
<strong>SHAP (SHapley Additive exPlanations):</strong> A method for explaining individual predictions of machine learning models. For each prediction, SHAP computes a value for each input feature representing that feature's contribution to the prediction relative to a baseline (typically the average prediction). The SHAP values sum to the difference between the prediction and the baseline: y\*_t − baseline = sum of SHAP values for all features. SHAP values are derived from Shapley values, a concept from cooperative game theory that fairly allocates the total "payoff" (prediction minus baseline) among the "players" (features).
</div>

For tree-based models (LightGBM, XGBoost, CatBoost), the `TreeSHAP` algorithm computes exact SHAP values in polynomial time — no approximation needed. This makes SHAP practical for post-mortem analysis of every bad day in the backtest.

The interpretation is intuitive:

- A **positive SHAP value** for a feature means that feature pushed the prediction **higher** than the baseline (toward higher prices).
- A **negative SHAP value** means the feature pushed the prediction **lower** than the baseline (toward lower prices).
- The **magnitude** reflects the strength of the effect.
- The **sum** of all SHAP values equals the difference between the model's prediction and the baseline (mean) prediction.

<div class="example-box">
<strong>Reading a SHAP decomposition for a missed spike:</strong> On 18 January 2026, the model predicted $180/MWh for the 5:00 PM interval; the actual price was $3,800/MWh. The SHAP decomposition reveals:

- Baseline (average prediction): $65/MWh
- Lagged price (t−1): +$42 (recent prices were elevated, pushing the prediction up)
- Temperature forecast: +$28 (high temperature → high demand → higher prices)
- Net load forecast: +$35 (high net load forecast → higher prices)
- Wind generation forecast: +$18 (low wind → less supply → higher prices)
- Hour-of-day: +$12 (5 PM is typically a peak period)
- Day-of-week: −$5 (Saturday, typically lower demand)
- Other features: −$15 (various minor negative contributions)
- <strong>Total: $65 + $42 + $28 + $35 + $18 + $12 − $5 − $15 = $180/MWh</strong>

The SHAP values show that every major feature pushed in the right direction — the model "knew" this was going to be an expensive period. The problem was not direction but **magnitude**: the model predicted $180 when the reality was $3,800. No individual feature was "wrong" — the model simply could not extrapolate to the extreme tail. This is the fundamental limitation of tree models: they cannot predict values outside their training range, and the $3,800 price was outside the range of recent training data.
</div>

### Partial Dependence: Recovering the Hockey Stick

While SHAP explains individual predictions, **partial dependence plots** show the model's learned relationship between a single feature and the prediction, averaged over all other features. For electricity price models, the most important partial dependence plot is **price vs net load** — it should recover the hockey-stick shape from Chapter 3.

<div class="definition-box">
<strong>Partial dependence plot (PDP):</strong> A visualisation of the marginal relationship between one or two input features and the model's prediction, averaging out the effects of all other features. For a feature x_j, the partial dependence function is: PD(x_j) = (1/n) · Σ_{i=1}^{n} f(x_j, x_{-j,i}), where f is the model, x_{-j,i} are all features except x_j for observation i, and the average is taken over all observations. The PDP shows what the model "thinks" the effect of x_j is, holding all else constant.
</div>

If the partial dependence plot of price on net load shows a monotonically increasing, convex curve (flat at low net load, steep at high net load), the model has learned the merit-order relationship from Chapter 3. If the curve flattens or turns over at high net load, the model may be **truncating spike predictions** — a critical failure mode for battery dispatch.

The hockey-stick recovery test is a useful diagnostic: compare the GBT's partial dependence on net load with the isotonic regression merit-order curve from Chapter 10. They should be qualitatively similar. If the GBT's curve flattens at high net load while the isotonic curve continues to steepen, the GBT is under-predicting spikes — and the grey-box model's physics stage is compensating for a real weakness.

![SHAP attributions on spike days](figures/14_shap_spike_days.png)

<p class="figure-caption">Figure 14.2 — SHAP feature attributions for the five worst regret days. Each row is one day; each bar is one feature's SHAP value. Positive values (right) push the prediction toward higher prices; negative values (left) push toward lower prices. The dominant features vary by day: temperature drives summer spikes; wind drives shoulder-season events; lagged prices dominate post-contingency events. The "missing" attribution — the gap between the prediction and the actual price — represents the model's inability to extrapolate to the tail.</p>

### What SHAP Cannot Tell You

SHAP decomposes the prediction the model **did** make. It does not tell you about predictions the model **should** have made. If a feature that would have signalled a spike was not included in the model (e.g., a generator outage flag, a constraint equation binding status, or a bid-stack shift), SHAP cannot identify the missing feature. The absence of a feature is invisible to SHAP.

This is an important limitation. On many of the worst regret days, the post-mortem reveals that the primary driver of the spike was **information the model did not have**: an unplanned generator outage, a transmission constraint that redirected supply, or a strategic rebid by a dominant generator. SHAP correctly attributes the prediction to the features that were present, but the real story is about the features that were absent.

---

## Error Decomposition

When a day produces high regret, the natural question is: **what caused the error?** Was it a bad weather forecast? A demand surprise? An event the model could never have predicted? Error decomposition separates the total forecast error into components, each attributable to a different source.

### The Decomposition Framework

The price forecast depends on several upstream inputs, each of which has its own forecast error:

<div class="equation">

price_error = f(weather_error, demand_error, supply_error, unexplained_residual)

</div>

We can approximate this decomposition by perturbing one input at a time:

1. **Weather-forecast error component.** Replace the weather forecast with the actual weather observation (ERA5 reanalysis for the day). Re-run the price model with actual weather but forecast demand. The change in the price prediction attributable to replacing the weather forecast is the weather-forecast error component.

2. **Demand-forecast error component.** Similarly, replace the demand forecast with actual demand (from AEMO settlement data). Re-run with forecast weather but actual demand. The change attributable to replacing the demand forecast is the demand-forecast error component.

3. **Unexplained residual.** The remaining error after correcting both weather and demand forecasts. This residual captures everything the model cannot predict even with perfect weather and demand inputs: generator outages, strategic rebidding, network constraints, interconnector failures, and other supply-side events.

<div class="definition-box">
<strong>Error decomposition:</strong> The practice of separating a forecast error into components attributable to different sources. For electricity price forecasting, the standard decomposition splits the error into weather-forecast error, demand-forecast error, and an unexplained residual. The decomposition is approximate because the price model is non-linear (the effects of weather and demand errors interact), but it provides actionable insight into which upstream forecasts need improvement.
</div>

<div class="example-box">
<strong>Error decomposition for 18 January 2026 (the missed spike):</strong>

- Total price error: $3,800 − $180 = $3,620/MWh
- Replacing forecast weather with actual weather: prediction moves from $180 to $220 → weather-forecast error component = $40/MWh (1.1% of total error)
- Replacing forecast demand with actual demand: prediction moves from $180 to $260 → demand-forecast error component = $80/MWh (2.2% of total error)
- Replacing both: prediction moves from $180 to $310 → combined input error = $130/MWh (3.6% of total error)
- Unexplained residual: $3,800 − $310 = $3,490/MWh (96.4% of total error)

The finding is stark: even with perfect weather and demand forecasts, the model would have predicted $310/MWh — not $3,800/MWh. The overwhelming majority of the error is in the unexplained residual. Investigation of the AEMO incident reports reveals that a major gas generator tripped at 4:45 PM, removing 500 MW from the supply stack and triggering scarcity pricing. This information was not available to the model at forecast time, and no feature in the model represented it.
</div>

### Interpreting the Residual

The unexplained residual is not a single phenomenon — it is a catch-all for everything the model's features do not capture. In the NEM context, the major contributors to the residual are:

| Source | Mechanism |
|--------|-----------|
| **Generator outages** | Unplanned trips remove supply; unpredictable |
| **Strategic rebidding** | Generators rebid near dispatch, tightening supply |
| **Network constraints** | Transmission limits segment the market |
| **Interconnector failures** | Loss of capacity isolates a region |
| **FCAS interactions** | High FCAS prices cascade via co-optimisation |

<div class="key-point">
<strong>The uncomfortable truth about spike prediction:</strong> For the most extreme price spikes (>$5,000/MWh), the error decomposition typically shows that 80–95% of the forecast error is in the unexplained residual — driven by events that were fundamentally unpredictable at the forecast horizon. This does not mean modelling effort is wasted: the model's ability to detect elevated-risk conditions (high demand, low wind, tight supply margins) is valuable even if it cannot predict the exact timing and magnitude of extreme events. The right framing is probabilistic: instead of trying to predict "$5,000 at 5 PM", predict "there is a 15% chance of a price exceeding $1,000 between 4 PM and 7 PM."
</div>

### The Counterfactual: What If We Had Known?

A useful extension of the error decomposition is the **counterfactual analysis**: for each source of error, compute the dispatch regret that would have been avoided if that single source had been corrected.

<div class="definition-box">
<strong>Counterfactual analysis:</strong> An analytical technique that asks "what would have happened if one specific input had been different?" For error decomposition, the counterfactual replaces one forecasted input with its actual value, re-runs the price model and dispatch LP, and measures the change in revenue. This reveals the economic value of improving each upstream forecast — it answers "how much is better weather forecasting worth?"
</div>

<div class="equation">

regret_avoided(weather) = revenue(model with actual weather) − revenue(model with forecast weather)

</div>

If correcting the weather forecast would have avoided $50,000 of regret over the backtest period, but correcting demand would have avoided $30,000, and the unexplained residual accounts for $400,000, the conclusion is clear: investing in better weather or demand forecasts has limited return. The economic prize is in **reducing the unexplained residual** — better supply-side modelling, real-time outage signals, or strategic rebidding indicators.

---

## Comparing Against Mannum's Real Dispatch

### The Anchor Asset

The Mannum BESS is a 100 MW / 200 MWh lithium iron phosphate (LFP) battery located in SA1. It is owned by Epic Energy and optimised by Habitat Energy, one of the most sophisticated battery optimisers in the NEM. The battery became operational in October 2025.

Mannum's **DUID** (Dispatchable Unit Identifier) in the NEM is the unique identifier used by AEMO's dispatch engine and market data systems. Its generation and consumption data are publicly available through AEMO's NEMWeb system, accessible via NEMOSIS (the same data tool used in Chapter 1).

<div class="definition-box">
<strong>DUID (Dispatchable Unit Identifier):</strong> A unique code assigned by AEMO to each generating unit or load in the NEM. Every scheduled generator, semi-scheduled generator, and scheduled load has a DUID. Market data — dispatch targets, generation output, bid stacks — is published at the DUID level. Battery systems typically have two DUIDs: one for the generating (discharging) unit and one for the load (charging) unit.
</div>

### Pulling Mannum's Dispatch Data

Mannum's actual dispatch is available from NEMWeb via NEMOSIS, using the same `DISPATCHLOAD` table that provides generation data for all scheduled units. The key fields are:

- **INITIALMW:** The unit's actual output at the start of the dispatch interval (MW). Positive values indicate generation (discharge); negative values or the corresponding load DUID indicate consumption (charge).
- **TOTALCLEARED:** The dispatch target set by AEMO's dispatch engine (MW). This is what AEMO instructed the unit to produce; actual output may differ slightly due to ramp rates and unit dynamics.
- **AVAILABILITY:** The maximum capacity the unit offered to the market (MW). This reveals whether Habitat chose to withhold capacity.

From these fields, you can reconstruct Mannum's half-hourly dispatch profile: when it charged, when it discharged, and at what power level. Combined with the settlement price data from Chapter 1, you can compute Mannum's actual revenue for any period.

### The Comparison: Course Model vs Habitat

The comparison between the course's forecast-driven dispatch and Mannum's actual dispatch is the centrepiece of this chapter. It answers the question: **how does a well-built academic model compare to a professional optimiser operating in the real market?**

The comparison is not quite apples-to-apples, and understanding the differences is as important as the numbers:

| Factor | Course model | Mannum |
|--------|-------------|--------|
| **Forecast horizon** | Day-ahead | Intra-day, 5-min prices |
| **Price data** | 30-min spot | 5-min dispatch + FCAS |
| **Information set** | Public data only | Proprietary + real-time feeds |
| **Dispatch frequency** | Daily or 30-min | Every 5 min |
| **Revenue streams** | Energy only | Energy + FCAS |
| **Degradation model** | Simple cycle counting | Degradation-aware |
| **Risk management** | Quantile-based | Proprietary |

<div class="key-point">
<strong>Why the comparison matters despite these differences:</strong> The course model and Habitat are solving the same fundamental problem — buy low, sell high — with different tools. Where they agree, we gain confidence in the model. Where they disagree, we learn what we are missing. The days where Habitat significantly outperformed the course model are the most instructive: they reveal information advantages that no amount of model tuning can replicate.
</div>

### Divergence Analysis

For each day in the backtest, compute:

1. **Course model revenue:** From the Chapter 9 backtest.
2. **Mannum actual revenue:** From NEMWeb dispatch data multiplied by settlement prices.
3. **Perfect-foresight revenue:** Upper bound from the LP with actual prices.
4. **Revenue gap:** Mannum revenue minus course model revenue.

Days where the revenue gap is large and positive (Habitat outperformed) are the most interesting. Classify these days by the type of divergence:

| Pattern | What happened | Lesson |
|---------|---------------|--------|
| **Timing shift** | Discharged 1–2 h earlier/later | Better intra-day information |
| **Spike capture** | Discharged during missed spike | Real-time outage detection |
| **FCAS arbitrage** | Earned from FCAS markets | Energy-only model cannot match |
| **Capacity withhold** | Reduced low-price availability | Strategic vs mechanical dispatch |
| **SoC management** | Different starting SoC | Better overnight anticipation |

<div class="example-box">
<strong>A day where Habitat outperformed:</strong> On 3 February 2026, the course model earned $12,000 while Mannum earned $48,000. Perfect-foresight revenue was $62,000.

The course model dispatched based on the day-ahead forecast, which predicted a broad afternoon peak with prices around $150–$300/MWh. It discharged steadily from 2 PM to 6 PM.

Mannum's actual dispatch was different: it held charge through the afternoon, then discharged aggressively at 5:30 PM when prices spiked to $2,400/MWh following a sudden interconnector constraint. Habitat appears to have detected the tightening market conditions in real time — likely from monitoring the constraint equations in AEMO's pre-dispatch engine — and adjusted dispatch accordingly.

The course model could not have replicated this: it made its dispatch decision at midnight based on a day-ahead forecast. By the time the constraint emerged at 5:00 PM, the model had already discharged most of its energy. This is not a model failure — it is an information disadvantage. The lesson is that intra-day re-optimisation, as discussed in Chapter 10, would have partially closed this gap.
</div>

![Missed-spike timeline](figures/14_missed_spike_timeline.png)

<p class="figure-caption">Figure 14.3 — Timeline comparison for a day where Habitat outperformed the course model. Top panel: actual settlement price (black), course model forecast (blue dashed). Middle panel: course model dispatch (blue bars) vs Mannum's actual dispatch (orange bars). Bottom panel: cumulative revenue comparison. The model discharged during a moderate afternoon peak; Habitat waited and captured the evening spike. The vertical grey band marks the interconnector constraint event that triggered the spike.</p>

### What Habitat Knows That the Model Does Not

The divergence analysis typically reveals several systematic information advantages:

1. **Real-time constraint equations.** AEMO publishes constraint equation binding status and marginal values in pre-dispatch and dispatch runs. These reveal network bottlenecks that can cause regional price separation. Professional optimisers monitor these in real time; the course model does not include them.

2. **Bid-stack transparency.** AEMO publishes next-day bid stacks and rebid reason strings. Sophisticated operators analyse the bid stack to estimate the supply curve and detect strategic rebidding. The course model uses lagged prices as a proxy for supply conditions, which is a weaker signal.

3. **5-minute dispatch resolution.** The NEM dispatches every 5 minutes, and the 30-minute settlement price is an average of six 5-minute dispatch prices. Professional operators optimise on 5-minute prices, capturing intra-half-hour volatility that the course model cannot see.

4. **FCAS co-optimisation.** Habitat likely co-optimises energy and FCAS, earning revenue from frequency control ancillary services. The course model earns energy arbitrage revenue only. On some days, the FCAS revenue may exceed the energy revenue, making the comparison misleading.

5. **Proprietary meteorological and demand models.** Professional operators often use private weather services and proprietary demand models that are more accurate than the public ERA5 + BOM data used in this course.

---

## The Incident Report

### Purpose and Structure

For each of the worst days by dollar impact, write a structured **incident report** — a one-page post-mortem that documents what happened, what the model predicted, what it should have done, and what could be improved.

<div class="definition-box">
<strong>Incident report:</strong> A structured post-mortem document for a high-regret dispatch day. The report answers four questions: (1) What happened in the market? (2) What did the model predict? (3) What would perfect foresight have done? (4) What, if anything, could have prevented the regret? Incident reports are standard practice in energy trading — they build institutional knowledge and prevent the same failure mode from recurring.
</div>

The incident report template:

```
INCIDENT REPORT — [Date]
Region: [SA1]
Regret: $[amount] ([X]% of perfect-foresight revenue)

1. MARKET SUMMARY
   - What happened: [Brief description of the price event]
   - Peak price: $[X]/MWh at [time]
   - Driver: [Demand spike / generator trip / constraint / rebid / other]

2. MODEL PREDICTION
   - Forecast peak: $[X]/MWh at [time]
   - Forecast error at peak: $[X]/MWh
   - SHAP attribution: [Top 3 features and their contributions]

3. DISPATCH COMPARISON
   - Model dispatch: [Charged/discharged at time X, SoC at time of spike]
   - Perfect foresight: [Would have discharged at time Y]
   - Mannum actual: [Discharged at time Z, power level]

4. ERROR DECOMPOSITION
   - Weather-forecast error: $[X]/MWh ([Y]%)
   - Demand-forecast error: $[X]/MWh ([Y]%)
   - Unexplained residual: $[X]/MWh ([Y]%)

5. LESSONS AND MITIGATIONS
   - Could intra-day re-optimisation have helped? [Yes/No, why]
   - Could better input data have helped? [Yes/No, what data]
   - Could a model change have helped? [Yes/No, what change]
   - Recommendation: [Specific action item]
```

### Worked Example: The Five Worst Days

The following worked example takes the five worst days by regret from a Chapter 9 backtest over Q1 2026 (January–March) in SA1, and walks through the full analysis for each.

<div class="example-box">
<strong>Day 1: 18 January 2026 — Generator trip during heatwave</strong>

<strong>Regret:</strong> $287,000 (72% of perfect-foresight revenue of $398,000)

<strong>Market summary:</strong> SA1 temperatures hit 42 degrees C. Demand surged to near-record levels. At 4:45 PM, a 500 MW gas generator tripped, removing supply during peak demand. Prices spiked from ~$200/MWh to $3,800/MWh within two dispatch intervals and remained above $1,000/MWh for four half-hours.

<strong>Model prediction:</strong> The model forecast a high-price afternoon (peak forecast $280/MWh at 5:00 PM), correctly identifying the heatwave. However, it did not predict the generator trip or the resulting extreme spike. Forecast error at the peak interval: $3,520/MWh.

<strong>SHAP attribution:</strong> Temperature (+$85), net load (+$62), lagged price (+$45), low wind (+$23), hour-of-day (+$15). All features pushed in the right direction; the magnitude was simply insufficient.

<strong>Dispatch comparison:</strong> The model discharged from 3:00 PM to 6:00 PM at 80–100 MW, earning $67,000. With the price shape the model predicted, this was a reasonable strategy. Perfect foresight would have concentrated discharge into the 4:30 PM–6:00 PM window, earning $398,000. Mannum discharged 100 MW from 4:30 PM to 6:30 PM, earning $312,000 — Habitat appears to have detected the tightening conditions in real time and shifted discharge later.

<strong>Error decomposition:</strong> Weather-forecast error: $40/MWh (1.1%). Demand-forecast error: $80/MWh (2.3%). Unexplained residual: $3,400/MWh (96.6%). The residual is the generator trip.

<strong>Lessons:</strong> No model improvement could have predicted the generator trip. However, intra-day re-optimisation with 5-minute dispatch prices would have detected the price spike within 5 minutes and adjusted dispatch. Recommendation: implement intra-day re-optimisation (Chapter 10 discussion) and monitor real-time reserve margins as a spike probability indicator.
</div>

<div class="example-box">
<strong>Day 2: 7 February 2026 — Interconnector constraint during wind drought</strong>

<strong>Regret:</strong> $195,000 (65% of perfect-foresight revenue of $300,000)

<strong>Market summary:</strong> SA1 wind generation dropped to near zero across a 6-hour window. The Heywood interconnector hit its limit, preventing Victorian supply from flowing into SA. Prices rose to $2,100/MWh for three consecutive half-hours.

<strong>Model prediction:</strong> The model forecast elevated prices (peak $180/MWh) due to low wind, but did not anticipate the interconnector constraint or the sustained duration of the spike. The probabilistic forecast's 95th percentile reached $600/MWh — closer, but still far from reality.

<strong>SHAP attribution:</strong> Wind generation (−$40, i.e., low wind pushed price up), net load (+$50), interconnector flow lag (+$20). The wind signal was present but underweighted.

<strong>Dispatch comparison:</strong> The model discharged 100 MW from 5:00 PM to 7:00 PM, earning $62,000. Perfect foresight would have discharged 100 MW from 4:00 PM to 8:00 PM (the full duration of the constraint event), earning $300,000. Mannum discharged 100 MW from 4:30 PM to 7:30 PM, earning $245,000.

<strong>Error decomposition:</strong> Weather-forecast error: $20/MWh (1.0%). Demand error: $30/MWh (1.6%). Unexplained residual: $1,870/MWh (97.4%). The residual is the interconnector constraint binding.

<strong>Lessons:</strong> Including interconnector flow and constraint binding status as features would have improved detection. The model had a lagged interconnector flow feature, but did not have the constraint equation marginal value, which would have signalled the binding constraint hours in advance. Recommendation: add AEMO pre-dispatch constraint data as a feature.
</div>

<div class="example-box">
<strong>Day 3: 22 February 2026 — Strategic rebidding by dominant generator</strong>

<strong>Regret:</strong> $142,000 (58% of perfect-foresight revenue of $245,000)

<strong>Market summary:</strong> Moderate demand, adequate wind. At 4:00 PM, a dominant gas generator rebid 400 MW from $50/MWh to $12,000/MWh — moving capacity from the bottom of the supply stack to the top. The effective supply curve shifted dramatically rightward at the current demand level. Prices jumped from $80/MWh to $1,800/MWh for two half-hours, then fell back as AEMO intervention and competing generators responded.

<strong>Model prediction:</strong> The model forecast $75/MWh — entirely normal for a mild day with adequate wind. Nothing in the feature set captured strategic rebidding intentions.

<strong>SHAP attribution:</strong> All features were near their baseline contributions. The model saw no signal because there was no signal in the public data — the rebid occurred in real time and was not reflected in any lagged or forecast variable available to the model.

<strong>Dispatch comparison:</strong> The model was fully charged at 4:00 PM (it had planned to discharge later in the evening based on a forecast evening peak). It did not discharge during the rebid-driven spike. Mannum discharged 100 MW during both spiked half-hours, earning $180,000 of the $245,000 available. Habitat likely monitors the bid stack in real time and detected the rebid immediately.

<strong>Error decomposition:</strong> Weather error: ~$0 (weather forecast was accurate). Demand error: ~$5/MWh. Unexplained residual: $1,720/MWh (99.7%). This is a pure supply-side event invisible to the model.

<strong>Lessons:</strong> Strategic rebidding is the single hardest failure mode for a model that does not ingest bid-stack data. The rebid had no precursor in public data — it was a strategic decision by a single market participant. Mitigation options: (a) include bid-stack features if available, (b) include pre-dispatch price forecasts from AEMO as features (AEMO's pre-dispatch engine sees the rebids before the settlement period), (c) accept that some events are irreducible noise and focus on risk management (Chapter 9 strategies that hold reserves for unknown spikes).
</div>

<div class="example-box">
<strong>Day 4: 12 March 2026 — False alarm: model predicted spike, calm day</strong>

<strong>Regret:</strong> $38,000 (42% of perfect-foresight revenue of $90,000)

<strong>Market summary:</strong> A forecast heatwave did not materialise — temperatures peaked 6 degrees C below the forecast. Demand was moderate, wind was adequate, and prices remained between $40–$120/MWh throughout the day. A calm, unremarkable day.

<strong>Model prediction:</strong> The model forecast peak prices of $450/MWh at 5:00 PM, driven by the (incorrect) high-temperature forecast. The chance-constrained MPC from Chapter 9 held charge through the morning and early afternoon, waiting for an evening spike that never came. By 7 PM, when it became clear the spike was not arriving, the battery discharged at $85/MWh — well below the $120/MWh available at 3 PM.

<strong>SHAP attribution:</strong> Temperature forecast (+$140) was the dominant driver. Net load forecast (+$60) was elevated due to the high temperature forecast. The error cascaded from weather to demand to price.

<strong>Dispatch comparison:</strong> The model earned $32,000. Perfect foresight would have earned $90,000 by spreading discharge across the moderate-price afternoon. Mannum earned $72,000 — Habitat appears to have adjusted its dispatch intra-day as the temperature forecast was revised downward.

<strong>Error decomposition:</strong> Weather-forecast error: $250/MWh (66% of the $380 total error at the peak interval). Demand error: $80/MWh (21%). Unexplained residual: $50/MWh (13%). This day's error is dominated by the weather forecast — one of the few days where improving the weather input would have substantially reduced regret.

<strong>Lessons:</strong> This is the prototypical false alarm case. The cost ($38,000) is an order of magnitude lower than the worst missed spikes — confirming the asymmetry discussed earlier. Mitigation: use ensemble weather forecasts rather than a single deterministic forecast. If the ensemble shows disagreement about peak temperature, the probabilistic price forecast should widen its intervals, making the chance-constrained MPC more cautious. This is exactly the probabilistic approach from Chapter 8, applied upstream to the weather input.
</div>

<div class="example-box">
<strong>Day 5: 28 March 2026 — Timing error during evening ramp</strong>

<strong>Regret:</strong> $31,000 (35% of perfect-foresight revenue of $88,000)

<strong>Market summary:</strong> Solar generation dropped off 30 minutes earlier than forecast due to cloud cover. The evening price ramp started at 4:30 PM instead of the forecast 5:00 PM. Prices reached $380/MWh at 4:30 PM and $520/MWh at 5:00 PM, but the model had scheduled discharge starting at 5:00 PM.

<strong>Model prediction:</strong> The model forecast the correct magnitude (peak $480/MWh, actual $520/MWh) but the wrong timing (5:00 PM vs 4:30 PM). The 30-minute timing error caused the battery to miss the first half-hour of elevated prices.

<strong>SHAP attribution:</strong> Solar generation forecast (+$30), hour-of-day (+$25), lagged price (+$20). The features were appropriate but the solar generation forecast was 30 minutes late.

<strong>Dispatch comparison:</strong> The model earned $45,000 by discharging from 5:00 PM to 7:00 PM. Perfect foresight would have earned $88,000 by starting discharge at 4:30 PM. Mannum earned $74,000 — its 5-minute dispatch resolution captured the early ramp within a single dispatch interval.

<strong>Error decomposition:</strong> Weather error (cloud cover timing): $120/MWh (31% of peak error). Demand error: $30/MWh (8%). Unexplained residual: $240/MWh (62%).

<strong>Lessons:</strong> Timing errors are a distinct failure mode from magnitude errors. A 30-minute timing error in solar generation translates directly into a 30-minute timing error in the evening ramp. Mitigation: (a) use satellite-derived nowcasting for solar, which detects cloud cover in near-real time, (b) increase dispatch frequency to 5 minutes (aligning with NEM dispatch), so the battery can respond to the actual ramp as it occurs.
</div>

### Patterns Across Incident Reports

Reviewing the five worst days reveals recurring patterns:

1. **The unexplained residual dominates.** On four of five days, the unexplained residual (supply-side events, strategic behaviour, network constraints) accounts for more than 90% of the price forecast error. The model's inputs — weather, demand, lagged prices — are not the primary source of failure.

2. **Real-time information is the main advantage.** Habitat's systematic outperformance comes from operating on 5-minute data with real-time AEMO feeds, not from better day-ahead forecasting. Intra-day re-optimisation is the single highest-value improvement identified.

3. **The asymmetry holds.** The four missed-spike days have an average regret of $206,000; the one false-alarm day has a regret of $38,000. Missed spikes are 5x more costly on average.

4. **Generator trips and strategic rebids are unpredictable.** Two of the five worst days were caused by events that no public-data model could have predicted. Risk management (holding reserves, widening forecast intervals) is more productive than trying to predict the unpredictable.

---

## The Forecast Error vs Decision Regret Scatter

A revealing diagnostic is to scatter-plot forecast error (MAE for the day) against decision regret (dollars lost). If forecast error and regret were perfectly correlated, all points would lie on a line. In practice, they diverge substantially:

![Forecast error vs decision regret scatter](figures/14_error_vs_regret_scatter.png)

<p class="figure-caption">Figure 14.4 — Scatter plot of daily forecast MAE (x-axis) vs daily decision regret in dollars (y-axis). Each dot is one day from the backtest. The weak correlation (r approximately 0.3–0.5) shows that forecast error is a poor predictor of economic impact. Days with high MAE but low regret (upper-left cluster) had large errors that did not cross decision boundaries. Days with moderate MAE but high regret (lower-right cluster) had small errors at critical moments that flipped dispatch decisions. The five labelled days are the incident reports above.</p>

The scatter reveals four quadrants:

| Quadrant | Description |
|----------|-------------|
| **Low MAE, low regret** | Model worked well — accurate enough for good decisions |
| **High MAE, low regret** | Errors in the "safe zone" — prices far from decision boundaries |
| **Low MAE, high regret** | Small errors at critical decision boundaries during spikes |
| **High MAE, high regret** | Genuinely bad forecast during an important period |

The existence of the high-MAE, low-regret quadrant is the key insight. Traditional forecast evaluation cannot distinguish between errors that matter and errors that do not. Only decision regret, computed through the dispatch LP, captures the economic relevance of each error.

<div class="key-point">
<strong>Implications for model selection:</strong> When choosing between models, do not select the model with the lowest MAE. Select the model with the lowest regret — or equivalently, the highest capture ratio. The models may rank differently on these two criteria, and the ranking by regret is the one that determines P&L performance. Chapter 9's backtest framework computes both, but the capture ratio should be the primary selection metric.
</div>

---

## Benchmarking Regret

### Absolute and Relative Regret

Report regret in two complementary forms:

**Absolute regret ($/MWh):** Total regret divided by total MWh discharged. This normalises for battery size and utilisation, making comparisons across different batteries and time periods meaningful.

<div class="equation">

regret_per_MWh = total_regret / total_MWh_discharged

</div>

**Relative regret (% of perfect-foresight revenue):** Total regret divided by total perfect-foresight revenue, expressed as a percentage. This is one minus the capture ratio from Chapter 9.

<div class="equation">

regret_pct = (total_regret / total_perfect_foresight_revenue) · 100%

</div>

### Benchmarks for the Mannum BESS

For the Mannum BESS (100 MW / 200 MWh, SA1), the following benchmarks contextualise the results:

| Strategy | Capture ratio | Regret ($/MWh) | Regret (% PF) |
|----------|--------------|----------------|---------------|
| Naive MPC (median) | 65–75% | $25–$45 | 25–35% |
| Scenario MPC (Ch 9) | 75–85% | $12–$28 | 15–25% |
| Chance-constrained (Ch 9) | 78–88% | $10–$22 | 12–22% |
| Mannum actual (est.) | 80–90% | $8–$18 | 10–20% |

The Mannum estimates are derived from its NEMWeb dispatch data and contemporary settlement prices. The exact figures depend on the evaluation period; the ranges above reflect seasonal variation (summer is more volatile than winter).

<div class="key-point">
<strong>The gap that matters:</strong> The difference between the course model's best strategy (chance-constrained MPC with conformal calibration) and Mannum's actual performance is approximately 5–10 percentage points of capture ratio — worth roughly $400K–$1.5M per year for a 100 MW battery in SA1. This gap is primarily driven by Habitat's real-time information advantage and FCAS revenue, not by a better day-ahead price forecast. Closing it requires operational infrastructure (real-time data feeds, 5-minute dispatch) rather than better modelling.
</div>

---

## Exercises

### Exercise 1: Identify the Input Whose Forecast Error Most Often Precedes a Missed Spike

From the Chapter 9 backtest, identify all missed-spike days (regret > $20,000 and spike price > 99th percentile of training distribution). For each, compute the forecast error for each upstream input (temperature, wind speed, solar irradiance, demand).

Rank inputs by **frequency of large errors preceding missed spikes**: for each input, count days where its forecast error exceeded its own 90th percentile.

**Hint:** This asks which upstream forecast was most often badly wrong on missed-spike days — not which input had the largest SHAP value. SHAP measures model reliance; this measures upstream data quality failures.

<details>
<summary><strong>Worked solution</strong></summary>

1. Collect missed-spike days:

```python
import pandas as pd

backtest = pd.read_parquet("outputs/backtest_results.parquet")
spike_threshold = backtest["actual_price"].quantile(0.99)
missed_spikes = backtest[
    (backtest["regret_dollars"] > 20_000)
    & (backtest["actual_price_peak"] > spike_threshold)
]
```

2. Compute per-input error frequency:

```python
weather_actual = pd.read_parquet("data/processed/era5_actual.parquet")
weather_forecast = pd.read_parquet("data/processed/era5_forecast.parquet")

inputs = ["temperature", "wind_speed", "solar_irradiance", "demand"]
input_errors = {}

for inp in inputs:
    errors = abs(weather_forecast[inp] - weather_actual[inp])
    threshold_90 = errors.quantile(0.90)
    large_error_days = errors.loc[missed_spikes.index] > threshold_90
    input_errors[inp] = {
        "count_large_errors": large_error_days.sum(),
        "fraction_of_missed_spikes": large_error_days.mean(),
    }

ranking = pd.DataFrame(input_errors).T.sort_values(
    "count_large_errors", ascending=False
)
```

3. Typical result for SA1:

<pre>
Input               Large-error days / 18    Fraction
wind_speed          12                       67%
temperature          8                       44%
demand               7                       39%
solar_irradiance     4                       22%
</pre>

**Wind speed forecast error is the most frequent precursor to missed spikes in SA1.** SA1 has high wind penetration, so wind over-prediction directly under-predicts net load and price. Demand errors are also common but partially redundant with temperature errors (high temp drives high demand).

</details>

### Exercise 2: Test Whether a Simple Rule on That Input Would Have Improved Dispatch

Test this rule: on days where wind-speed forecast error exceeds its 80th percentile, shift the MPC discharge quantile from q_0.10 to q_0.05 (more conservative, reserving charge for potential spikes). Compute the change in total regret.

**Hint:** This tests whether the signal exists before investing in a full ensemble-spread solution. One input, one threshold, one dispatch adjustment.

<details>
<summary><strong>Worked solution</strong></summary>

1. Flag high-wind-error days and run both backtests:

```python
from grian.backtest import rolling_origin_backtest
from grian.dispatch import chance_constrained_mpc

wind_errors = abs(weather_forecast["wind_speed"] - weather_actual["wind_speed"])
high_wind_error_days = wind_errors > wind_errors.quantile(0.80)

results_baseline = rolling_origin_backtest(
    model=model,
    dispatch_fn=lambda forecasts: chance_constrained_mpc(
        forecasts, q_low=0.10, q_high=0.90
    ),
    data=data,
)

def adjusted_dispatch(forecasts, date):
    if high_wind_error_days.loc[date]:
        return chance_constrained_mpc(forecasts, q_low=0.05, q_high=0.95)
    return chance_constrained_mpc(forecasts, q_low=0.10, q_high=0.90)

results_adjusted = rolling_origin_backtest(
    model=model,
    dispatch_fn=adjusted_dispatch,
    data=data,
)
```

2. Compare:

```python
regret_baseline = results_baseline["regret_dollars"].sum()
regret_adjusted = results_adjusted["regret_dollars"].sum()
print(f"Baseline total regret:  ${regret_baseline:,.0f}")
print(f"Adjusted total regret:  ${regret_adjusted:,.0f}")
print(f"Relative improvement:   {(regret_baseline - regret_adjusted) / regret_baseline:.1%}")
```

<pre>
Typical improvement: 3-8% reduction in total regret.
Diebold-Mariano test: significant at 5% level on most windows.
</pre>

3. Why the improvement is modest:
   - Rule fires on only ~20% of days
   - Extra conservatism is often unnecessary (no spike materialises)
   - On some days, under-dispatch during moderate prices slightly increases regret

**The signal is real but weak in isolation.** A composite "spike risk" indicator combining wind-error, demand uncertainty, and supply-margin tightness would be more effective than this binary rule.

</details>

### Exercise 3: Hypothesise What Additional Information Habitat Used

Pick a day where Mannum earned >2x the course model's revenue. Using NEMWeb public data, reconstruct the market context:

1. Pull AEMO pre-dispatch price forecasts (NEMSEER)
2. Check constraint equation binding status (pre-dispatch PASA)
3. Examine generator rebid data (NEMWeb bidding tables)
4. Compare regional demand vs forecast (AEMO operational demand)

Write a 300-word hypothesis naming the specific data source, the timing it became available, and how it would have changed dispatch.

**Hint:** Habitat operates on 5-minute intervals with real-time data; the model operates day-ahead. The question is: which piece of information, available when, would have flipped the decision?

<details>
<summary><strong>Worked solution</strong></summary>

Example: 7 February 2026 (interconnector constraint day).

**Data reconstruction:**

```python
from nemseer import compile_data

predispatch = compile_data(
    start_time="2026-02-07 00:00",
    end_time="2026-02-08 00:00",
    table="PREDISPATCHPRICE",
    region="SA1",
)
```

<pre>
Pre-dispatch run    Forecast peak (SA1)    Signal
00:30 (midnight)    $120/MWh at 5:30 PM   No spike expected (day-ahead)
06:00               $145/MWh at 5:30 PM   Wind forecast softening
10:00               $280/MWh at 5:00 PM   Wind actuals below forecast
12:30               $650/MWh at 4:30 PM   Interconnector constraint binding
14:00               $1,200/MWh at 4:30 PM Heywood constraint confirmed
15:00               $2,800/MWh at 4:30 PM Reserve margin near zero
</pre>

**Hypothesis:**

**What they used:** AEMO pre-dispatch price forecasts and `PREDISPATCH_CONSTRAINT` marginal values, published every 30 minutes.

**When the signal appeared:** By 12:30 PM, the Heywood interconnector constraint (equation names containing "HVSC" or "HEYW") showed a non-zero marginal value, indicating SA1 was import-constrained with prices forecast above $600/MWh. This was seven hours before the event.

**How it changed dispatch:** Habitat's system ingests each pre-dispatch run and re-solves the dispatch LP. When the 12:30 PM run showed elevated prices from 4:00 PM, they revised their schedule: hold charge through the afternoon, discharge at maximum rate during the constraint period. The 14:00 and 15:00 runs confirmed and amplified the signal.

**Why the course model missed it:** Its day-ahead forecast was made 16 hours before the event, using weather and demand forecasts that did not yet reflect the wind drought or the resulting constraint.

**The minimum viable improvement:** Incorporate AEMO pre-dispatch prices as a rolling feature and re-run the dispatch LP every 30 minutes as new data arrives. This is the intra-day re-optimisation from Chapter 10 — the simplest version of what Habitat does continuously.

</details>

---

## Glossary

| Term | Definition |
|------|-----------|
| **Decision regret** | Revenue gap: forecast dispatch vs perfect foresight |
| **Asymmetric loss** | Under-predicting spikes costs more than over-predicting |
| **Regime-conditional calibration** | Calibration evaluated per regime (calm vs spike) |
| **Feature attribution** | Decomposing prediction into per-feature contributions |
| **SHAP** | Shapley-value-based feature attribution method |
| **Partial dependence** | Marginal feature-prediction relationship, averaged |
| **Counterfactual analysis** | What if one specific input had been different? |
| **Incident report** | Structured post-mortem for high-regret days |
| **DUID** | AEMO unique code for each NEM unit or load |
| **Missed spike** | Actual price far exceeded forecast; missed discharge |
| **False alarm** | Forecast far exceeded actual; suboptimal dispatch |
| **Error decomposition** | Splitting error into weather, demand, and residual |
| **Capture ratio** | Fraction of perfect-foresight revenue achieved |

## Summary

This chapter reframes forecast evaluation around decision regret — the revenue gap between forecast-driven dispatch and perfect-foresight dispatch — rather than traditional forecast error metrics like MAE and CRPS. The core finding is that forecast error and decision regret are only weakly correlated: large errors during calm periods cost nothing because they do not flip dispatch decisions, while small errors during spike periods can cost hundreds of thousands of dollars. The asymmetry is stark: missed spikes cost 5–10x more per event than false alarms, and approximately 80% of total regret comes from 10% of trading intervals. Regime-conditional evaluation — splitting metrics and calibration diagnostics into calm and spike intervals — reveals performance differences masked by overall metrics. SHAP feature attributions and partial dependence plots identify what drove the model's predictions on bad days, but the error decomposition typically shows that 80–95% of the error on the worst days comes from the unexplained residual (generator trips, strategic rebids, network constraints) rather than from weather or demand forecast errors. Comparing the course model against Mannum's actual dispatch from NEMWeb reveals that Habitat Energy's systematic advantage comes from real-time information (5-minute dispatch prices, constraint equation binding status, bid-stack monitoring) and FCAS co-optimisation, not from a better day-ahead forecast. The incident report framework — structured post-mortems for the worst days by dollar impact — builds institutional knowledge and identifies actionable improvements: intra-day re-optimisation with pre-dispatch data is consistently the highest-value recommendation. Regret should be reported in $/MWh and as a percentage of perfect-foresight revenue, alongside the capture ratio from Chapter 9, to give a complete picture of economic performance.
