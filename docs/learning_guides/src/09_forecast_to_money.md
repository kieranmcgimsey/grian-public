# 9. From Forecast to Money

## The Dispatch Decision Under Uncertainty

In Chapter 5, we introduced **Model Predictive Control (MPC)** with a point forecast: take the median price prediction for the next 48 half-hours, solve a linear program (LP) to find the optimal charge/discharge schedule, execute the first period's decision, observe the actual price, and repeat. This approach works, but it ignores a critical piece of information: the probabilistic forecast from Chapters 7 and 8.

Naive MPC treats a confident forecast and a wildly uncertain one identically. If the median predicted price for 6 PM is $150/MWh, naive MPC makes the same dispatch decision whether the 80% interval is [$120, $180] or [$20, $5,000]. This is suboptimal — the second scenario should trigger a fundamentally different strategy, one that accounts for the enormous uncertainty.

<div class="definition-box">
<strong>Model Predictive Control (MPC):</strong> A control strategy that repeatedly solves an optimisation problem over a future horizon, executes only the first decision, observes the new state, and re-solves. For battery dispatch, the optimisation maximises revenue over the next 24–48 hours, the first half-hour's charge/discharge decision is executed, the actual price is observed, and the process repeats. The "receding horizon" approach naturally corrects for forecast errors because the optimisation is re-solved with updated information at every step.
</div>

This chapter develops two methods that use the full quantile forecast from Chapters 7–8: **scenario MPC** (based on stochastic programming) and **chance-constrained MPC** (based on robust optimisation). Both are well-established in operations research and are widely used in energy trading, portfolio management, and supply chain planning.

<div class="key-point">
<strong>The central question:</strong> How do you make optimal decisions when you don't know what the future prices will be — but you do know the <em>range</em> of possible prices and their approximate likelihoods? This is the domain of decision-making under uncertainty, and it is what separates sophisticated battery operators from naive ones.
</div>

---

## Stochastic Programming: Scenario MPC

### The Idea

<div class="definition-box">
<strong>Stochastic programming:</strong> A mathematical optimisation framework for making decisions under uncertainty. Instead of optimising against a single deterministic scenario, stochastic programming considers multiple possible future outcomes (scenarios), each with an associated probability, and optimises the <strong>expected</strong> (probability-weighted average) objective across all scenarios. The decisions must be made before the uncertainty is resolved — you must commit to a charge/discharge plan before you know the actual prices.
</div>

The core idea is intuitive: instead of betting everything on a single price trajectory (the median forecast), generate many plausible price trajectories from the probabilistic forecast and find the battery schedule that performs best **on average** across all of them.

This is a classic **two-stage stochastic program**:

- **First stage (here-and-now decisions):** Choose the charge/discharge schedule before the actual prices are revealed. These decisions must be made under uncertainty — they cannot depend on which scenario actually materialises.
- **Second stage (wait-and-see outcomes):** The actual prices are observed, and the revenue is computed based on the first-stage decisions and the realised prices.

<div class="definition-box">
<strong>Two-stage stochastic program:</strong> An optimisation problem with two stages. In the first stage, decisions are made before uncertainty is resolved. In the second stage, the uncertainty is revealed and recourse actions may be taken. For battery dispatch, the first-stage decision is the charge/discharge schedule; the second stage is the realisation of actual prices and the computation of revenue. The objective is to maximise expected second-stage revenue across all scenarios.
</div>

By optimising across many scenarios, the battery **hedges** against forecast uncertainty. It avoids committing to extreme positions that would be catastrophic if the forecast is wrong. A battery that discharges all its energy at 5 PM (because the median forecast says 5 PM has the highest price) earns nothing if the spike actually occurs at 7 PM. A battery optimised across scenarios might split its discharge between 5 PM and 7 PM, earning somewhat less in the median scenario but much more on average.

<div class="example-box">
<strong>Intuitive analogy — diversifying bets:</strong> Scenario MPC is like diversifying an investment portfolio. Instead of betting your entire portfolio on one stock (the median forecast), you spread your investment across multiple stocks (scenarios). Some investments will underperform, but the portfolio as a whole has lower risk and often higher average returns than an all-in bet on a single stock. The battery "diversifies" its dispatch across time periods that are high-priced in different scenarios.
</div>

### Generating Scenarios from Quantile Forecasts

The probabilistic forecast from Chapter 8 provides quantile forecasts — for each half-hour period, we have predicted values at levels τ_1 < τ_2 < ... < τ_m (e.g., 5th, 10th, 25th, 50th, 75th, 90th, 95th percentiles). We need to convert these into complete price **scenarios** — full 48-period price trajectories.

The simplest approach:

1. **For each time period t**, draw a random quantile level u_t from a Uniform(0, 1) distribution
2. **Interpolate** between the quantile forecasts at level u_t to obtain a price. For example, if u_t = 0.35, interpolate between the 25th and 50th percentile predictions.
3. **Repeat steps 1–2 for all 48 periods** to generate one complete price trajectory (scenario)
4. **Repeat the entire process N times** to generate N scenarios

<div class="definition-box">
<strong>Scenario:</strong> In stochastic programming, a scenario is one complete realisation of the uncertain quantities — in our case, a full 48-period price trajectory. Each scenario represents one plausible future: "what might happen if the price follows this particular path." The set of all scenarios, with their associated probabilities, approximates the full distribution of possible futures.
</div>

This simple method has a limitation: the random draws u_t are independent across time periods, so the scenarios do not preserve the **temporal correlation** of prices — the tendency for high prices to cluster together and low prices to cluster together. More sophisticated scenario generation methods address this:

- **Copula-based methods:** Model the temporal dependence structure with a copula (a mathematical function that describes the correlation between random variables), generating scenarios that preserve the autocorrelation of prices.

<div class="definition-box">
<strong>Copula:</strong> A statistical function that describes the dependence structure between random variables, separate from their individual distributions. A copula captures how the variables tend to move together: if price at 5 PM is high, is price at 6 PM also likely to be high? By separating the dependence structure (copula) from the individual distributions (quantile forecasts), we can generate scenarios that have both the correct marginal distributions and the correct correlations.
</div>

- **Gaussian bridge methods:** Generate multivariate Gaussian random variables with the correct covariance structure (capturing how prices at different hours are correlated), then transform them to match the quantile forecast at each period.

- **Historical analogue methods:** Find historical days with similar forecast patterns (similar weather, demand, and predicted prices) and use their actual price trajectories as scenarios. These scenarios automatically have realistic temporal structure because they are real price paths.

For battery dispatch with a 48-period horizon, the simple independent sampling is often sufficient in practice because the LP's optimal solution depends mainly on the **ranking** of prices across periods (which periods are cheapest for charging and most expensive for discharging), not on the exact correlation structure. However, the more sophisticated methods can improve performance during volatile periods when temporal clustering matters.

### Solving the Scenario LP

Given N generated scenarios, we need to find the battery schedule that maximises expected revenue. There are two approaches:

**Approach 1: Solve and average.** Solve the standard dispatch LP (from Chapter 5) separately for each scenario, obtaining N optimal schedules. Then compute the average charge and discharge across all scenarios:

<div class="equation">

charge_t = (1/N) · Σ_{s=1}^{N} charge_{t,s}     and     discharge_t = (1/N) · Σ_{s=1}^{N} discharge_{t,s}

</div>

This approach is simple and fast (N independent LPs can be solved in parallel). The averaged schedule may occasionally violate battery constraints (e.g., the average of feasible state-of-charge trajectories may not itself be feasible), but in practice this is rarely an issue for simple battery constraints.

**Approach 2: Expected-value optimisation.** Formulate and solve a single, larger LP that maximises expected revenue across all scenarios simultaneously, with **shared decisions** (the charge and discharge variables are the same across all scenarios):

<div class="equation">

Maximise (1/N) · Σ_{s=1}^{N} Σ_{t=1}^{T} price_{t,s} · (discharge_t − charge_t) · Δt

</div>

Subject to the standard battery constraints (Chapter 5): state-of-charge limits, power limits, efficiency losses, and cycle constraints. The key difference from approach 1 is that charge_t and discharge_t are **not** subscripted by s — they are the same for all scenarios. This forces the optimiser to find a single schedule that works well across all scenarios, not the best schedule for any one scenario.

<div class="definition-box">
<strong>Expected value:</strong> The probability-weighted average of all possible outcomes. If there are N equally likely scenarios, the expected revenue is the simple average of the revenue across all scenarios. Expected-value optimisation finds the decisions that maximise this average, implicitly trading off doing well in some scenarios against doing poorly in others.
</div>

This approach is computationally more expensive (the LP has N times more revenue terms, though the decision variables remain the same) but produces a feasible schedule by construction, since the battery constraints are applied to the shared decision variables.

### How Many Scenarios?

The number of scenarios N controls the quality of the approximation to the true expected revenue. More scenarios provide a better approximation but cost more computation:

| N | Behaviour | Typical use |
|---|-----------|-------------|
| 1 | Reduces to naive MPC (single scenario = median forecast) | Baseline comparison |
| 5–10 | Captures broad uncertainty structure but may miss tail events | Quick-and-dirty applications |
| 20–50 | Usually sufficient for battery dispatch; captures the main structure | Standard practice |
| 100+ | Diminishing returns; additional scenarios refine the mean estimate marginally | Research or high-stakes applications |

<div class="definition-box">
<strong>Law of large numbers:</strong> A theorem stating that the average of a large number of independent random draws converges to the true expected value. For scenario MPC, this means that as N increases, the average revenue across scenarios converges to the true expected revenue. The convergence rate is O(1/√N) — doubling the accuracy of the expected value estimate requires quadrupling the number of scenarios.
</div>

Computation time scales linearly with N when using approach 1 (N independent LPs) and sub-linearly with approach 2 (a single LP whose size grows with N). For a 48-period battery dispatch problem, even N = 100 LPs can be solved in under a second on a modern laptop, so computational cost is rarely the binding constraint.

![Scenario MPC vs chance-constrained MPC](figures/09_scenario_vs_cc.png)

<p class="figure-caption">Figure 9.1 — Comparison of scenario MPC and chance-constrained MPC dispatch strategies. Scenario MPC hedges by considering multiple price trajectories, while chance-constrained MPC uses conservative quantile bounds. The dispatch behaviour differs: scenario MPC tends to spread discharge across multiple high-price periods, while chance-constrained MPC concentrates discharge in periods where the conservative price estimate is highest.</p>

---

## Robust Optimisation: Chance-Constrained MPC

### The Idea

Chance-constrained MPC takes a different philosophical approach to uncertainty. Instead of averaging across many scenarios (which gives the **best expected outcome**), it optimises against a **pessimistic scenario** — ensuring the dispatch schedule is profitable even when prices are worse than expected.

<div class="definition-box">
<strong>Robust optimisation:</strong> An optimisation framework that seeks decisions that perform well under the <strong>worst case</strong> (or near-worst case) within a defined set of possible outcomes (the "uncertainty set"). Unlike stochastic programming (which maximises the average outcome), robust optimisation maximises the worst-case outcome — or, in softer variants, ensures the outcome is acceptable with high probability. The result is a conservative but safe strategy.
</div>

<div class="definition-box">
<strong>Chance constraint:</strong> A constraint that must hold with at least a specified probability. For example, "the battery's discharge revenue must exceed its charge cost with at least 90% probability." Chance constraints formalise the idea of "be profitable most of the time" without requiring profitability in every single scenario.
</div>

The specific approach is elegant in its simplicity: instead of using the median (50th percentile) forecast as in naive MPC, use **different quantile forecasts for discharge revenue and charge cost**:

- **Value discharge revenue at a low quantile** (e.g., the 10th percentile): This is a conservative estimate of the revenue — "even in a pessimistic scenario, the discharge price will be at least this much."
- **Value charge cost at a high quantile** (e.g., the 90th percentile): This is a conservative estimate of the cost — "even in a pessimistic scenario, the charge price will be at most this much."

If the schedule is profitable under these conservative price estimates, it is profitable with high probability (at least 1 − 2α, where α is the quantile level).

### The Chance-Constrained LP

The LP formulation is a simple modification of the naive MPC formulation from Chapter 5. The only change is in the objective function — the battery constraints remain identical:

<div class="equation">

Maximise Σ_{t=1}^{T} [q_{low,t} · discharge_t − q_{high,t} · charge_t] · Δt

</div>

Subject to the standard battery constraints (state-of-charge limits, power limits, efficiency, cycles).

Here q_low,t is the lower quantile (e.g., 10th percentile) and q_high,t is the upper quantile (e.g., 90th percentile) for time period t.

This formulation is **conservative by construction:**

- **Discharge revenue** is valued at the low end of the price range. The battery only discharges when the revenue would be high even in a pessimistic price scenario. It will not discharge for a period where the median price is $150 but the 10th percentile is $30 — the risk of low revenue is too high.

- **Charge cost** is valued at the high end of the price range. The battery only charges when the cost would be low even in a pessimistic cost scenario. It will not charge during a period where the median price is $30 but the 90th percentile is $200 — the risk of expensive charging is too high.

<div class="example-box">
<strong>Real-world example — conservative dispatch in action:</strong> Consider two half-hour periods at 5 PM:

<strong>Period A:</strong> q_0._1_0 = $120, median = $150, q_0._9_0 = $180. The chance-constrained LP values discharge at $120. Even in a pessimistic scenario, the revenue is substantial. The battery discharges.

<strong>Period B:</strong> q_0._1_0 = $20, median = $150, q_0._9_0 = $800. The chance-constrained LP values discharge at only $20. Despite the high median, the pessimistic scenario is poor. The battery may hold back, preserving its charge for a more certain opportunity.

Naive MPC would treat both periods identically (both have median = $150). Chance-constrained MPC distinguishes them by their downside risk.
</div>

### The Conservatism Tradeoff

The choice of quantile levels controls the tradeoff between safety (avoiding losses) and aggressiveness (maximising revenue):

| Quantile pair | Behaviour | Risk profile |
|---------------|-----------|-------------|
| q_0._0_5 / q_0._9_5 | Very conservative | Battery rarely acts; only moves when price signals are extremely strong. Minimises risk but leaves significant revenue on the table. |
| q_0._1_0 / q_0._9_0 | Moderate | Default choice. Good balance of safety and activity. Captures most profitable opportunities while avoiding most false signals. |
| q_0._2_5 / q_0._7_5 | Aggressive | Battery acts frequently, assuming the forecast is well-calibrated. Higher revenue in good scenarios, higher risk in bad ones. |
| q_0._5_0 / q_0._5_0 | Reduces to naive MPC | No uncertainty information used — the dispatch is based entirely on the median forecast. |

<div class="key-point">
<strong>The optimal quantile pair depends on forecast calibration.</strong> If the 90% prediction interval is well-calibrated (it actually covers 90% of outcomes), then the q_0._1_0 / q_0._9_0 pair is naturally well-matched to the uncertainty. If the model is overconfident (the 90% interval only covers 75% of outcomes), wider quantile pairs (q_0._0_5 / q_0._9_5) are needed to compensate. This is why Chapter 8's calibration work is essential — poorly calibrated quantile forecasts lead to poorly tuned dispatch strategies.
</div>

### Comparing Scenario MPC and Chance-Constrained MPC

| Property | Scenario MPC | Chance-constrained MPC |
|----------|-------------|----------------------|
| Objective | Maximise expected revenue | Maximise worst-case (conservative) revenue |
| Uses full distribution | Yes (via sampling) | Partially (only two quantile levels) |
| Computational cost | N × LP cost (but parallelisable) | 1 × LP cost (very fast) |
| Conservatism | Implicit (from scenario diversity) | Directly controllable (via quantile pair) |
| Risk handling | Risk-neutral (optimises the average) | Risk-averse (protects against downside) |
| Handles asymmetric risk | Naturally (scenarios can be skewed) | With asymmetric quantile choices (e.g., q_0._1_5 / q_0._9_5) |
| Implementation complexity | Moderate (scenario generation + aggregation) | Simple (modified LP objective) |
| Optimality | Approaches optimal as N → ∞ | Suboptimal but safe by construction |
| When it excels | Well-calibrated forecast, risk-neutral operator | Uncertain forecast, risk-averse operator |

<div class="example-box">
<strong>Real-world industry context:</strong> In practice, many battery operators use a variant between these two extremes. A common approach is to solve the chance-constrained LP for the "base plan" and then adjust specific periods using scenario analysis — for example, increasing discharge during periods where the scenario spread suggests a spike is possible. The simplicity of chance-constrained MPC makes it the more common production approach, while scenario MPC is more common in research and trading desks with dedicated quantitative teams.
</div>

---

## Sensitivity Analysis: Battery Parameters

The **capture ratio** — the fraction of perfect-foresight revenue achieved by the dispatch strategy — depends not only on forecast quality but also on the battery's physical characteristics. Understanding these sensitivities is critical for investment decisions (which battery to buy) and operational decisions (how to configure the dispatch strategy).

<div class="definition-box">
<strong>Capture ratio (CR):</strong> The ratio of actual dispatch revenue to the maximum possible revenue (perfect foresight). CR = actual revenue / perfect foresight revenue. A CR of 0.7 means the dispatch strategy captures 70% of the theoretical maximum revenue. The capture ratio measures the combined quality of the forecast and the dispatch strategy — a higher CR means better forecasting and/or better dispatch.
</div>

### Duration (Hours of Storage)

Battery **duration** (also called "energy-to-power ratio" or "storage hours") is the number of hours the battery can sustain its maximum discharge power. A 100 MW / 2h battery can discharge at 100 MW for 2 hours (total energy = 200 MWh). A 100 MW / 4h battery can discharge at 100 MW for 4 hours (total energy = 400 MWh).

<div class="definition-box">
<strong>Battery duration:</strong> The ratio of a battery's energy capacity (MWh) to its power capacity (MW). A battery with 200 MWh of energy capacity and 100 MW of power capacity has a duration of 2 hours. Duration determines how long the battery can sustain maximum output — a 2h battery can discharge for 2 hours, while a 4h battery can discharge for 4 hours. Longer duration batteries can capture longer-duration price events but are more expensive per MW.
</div>

| Duration | Capabilities | Limitations | Typical application |
|----------|-------------|-------------|-------------------|
| **1 hour** | Captures short price spikes (1–2 intervals) | Misses multi-hour events; must time the peak precisely | FCAS and frequency response |
| **2 hours** | Standard configuration; captures most intra-day arbitrage | Cannot bridge long duration events (e.g., a 4-hour evening peak) | Day-ahead energy arbitrage (standard) |
| **4 hours** | Captures extended events; more flexible dispatch | Diminishing marginal returns for day-ahead arbitrage; most NEM price differentials resolve within 4 hours | Day-ahead arbitrage (premium); renewable energy shifting |
| **8+ hours** | Can shift energy across major parts of the day | Relevant for inter-day arbitrage or seasonal shifting; rarely justified by NEM day-ahead price patterns alone | Long-duration storage research; renewable integration |

The relationship between duration and capture ratio is **concave** — going from 1h to 2h provides a large CR improvement (capturing the main daily price differential), going from 2h to 4h provides a moderate improvement (capturing longer events), and going from 4h to 8h provides a small improvement (most NEM price events are shorter than 4 hours). This concavity means that for pure day-ahead arbitrage in the NEM, 2h batteries are the sweet spot: they capture most of the value at half the cost of a 4h battery.

### Round-Trip Efficiency

**Round-trip efficiency (RTE)** is the fraction of energy recovered from a full charge-discharge cycle. If you put 100 MWh into a battery and get 85 MWh out, the round-trip efficiency is 85%.

<div class="definition-box">
<strong>Round-trip efficiency (RTE):</strong> The ratio of energy delivered during discharge to energy consumed during charging, expressed as a percentage. An RTE of 85% means that for every 100 MWh put into the battery, 85 MWh comes back out — the other 15 MWh is lost to heat, chemical irreversibility, and power electronics inefficiency. RTE directly determines the minimum price spread needed for profitable arbitrage: the discharge price must exceed the charge price by a factor of 1/RTE.
</div>

Efficiency determines the **breakeven spread** — the minimum ratio of discharge price to charge price required for profitable operation:

<div class="equation">

Breakeven condition: P_discharge > P_charge / RTE

</div>

| Round-trip efficiency | Energy loss | Breakeven spread | Interpretation |
|----------------------|-------------|-------------------|---------------|
| 70% | 30% lost | Price must be 43% higher | Older or lower-quality batteries; only large spreads are profitable |
| 85% | 15% lost | Price must be 18% higher | Typical lithium-ion; moderate spreads are profitable |
| 90% | 10% lost | Price must be 11% higher | High-quality lithium-ion; small spreads become profitable |
| 95% | 5% lost | Price must be 5% higher | Best-in-class or theoretical; nearly all spreads are profitable |

Higher efficiency strictly increases revenue because it lowers the breakeven threshold — more price differentials become profitable. However, the **marginal value** of efficiency improvements is diminishing. Going from 70% to 85% opens up many new profitable trading opportunities (all spreads between 18% and 43%). Going from 90% to 95% opens up far fewer new opportunities (only spreads between 5% and 11%) because most profitable NEM arbitrage involves large spreads anyway.

<div class="example-box">
<strong>Real-world example — efficiency economics:</strong> Consider two batteries dispatching in SA1 on a day with charge price = $30/MWh and discharge price = $120/MWh (a 300% spread):

<strong>Battery A (RTE = 85%):</strong> Revenue per MWh discharged = $120 − ($30 / 0.85) = $120 − $35.3 = $84.7/MWh

<strong>Battery B (RTE = 90%):</strong> Revenue per MWh discharged = $120 − ($30 / 0.90) = $120 − $33.3 = $86.7/MWh

The 5-percentage-point efficiency improvement yields an extra $2/MWh per cycle. For a 100 MW / 2h battery doing 1.5 cycles per day, this is ~$600/day or ~$220K/year. This is significant but not transformative — the battery's revenue is dominated by the magnitude of the price spread, not the efficiency differential.
</div>

### Cycle Constraints and Degradation

Every charge-discharge cycle degrades the battery slightly — the lithium-ion cells lose a fraction of their capacity with each cycle due to chemical and mechanical degradation. The **cycle constraint** limits the number of full cycles per day to manage this degradation.

<div class="definition-box">
<strong>Battery cycle:</strong> One complete charge from empty to full followed by a complete discharge from full to empty. A "half cycle" is a charge-discharge that uses only half the battery's capacity. Cycle counting is typically based on the total energy throughput: if the battery has 200 MWh capacity and charges/discharges a total of 400 MWh in a day, it has done 2 full equivalent cycles.
</div>

<div class="definition-box">
<strong>Battery degradation:</strong> The gradual loss of a battery's capacity and efficiency over time, driven by charge-discharge cycling, calendar aging (degradation that occurs even when idle), temperature, depth of discharge, and charge rate. A lithium-ion battery might lose 2–3% of its capacity per year under normal cycling. Degradation is economically significant because it reduces the battery's future revenue potential — each cycle today trades a small amount of future capacity for immediate revenue.
</div>

| Cycles per day | Behaviour | Economic implication |
|---------------|-----------|---------------------|
| 1 cycle/day | Very conservative | Preserves battery life but misses secondary arbitrage opportunities. Suitable when the battery's capital cost is high relative to daily revenue. |
| 1.5–2 cycles/day | Standard | Captures the main morning-charge/evening-discharge cycle plus one or two secondary opportunities. The default for most utility batteries. |
| 3–4 cycles/day | Aggressive | Exploits smaller intra-day price swings. Suitable for batteries with low degradation costs or when daily price volatility is very high. Shortens battery life. |

In our simplified LP formulation (Chapter 5), degradation is modelled as a simple cycle-count constraint. Real battery degradation is more complex — it depends on the depth of discharge (shallow cycles are less damaging than deep cycles), the charge rate (fast charging degrades faster), temperature, and calendar aging. More sophisticated dispatch models incorporate degradation as a variable cost: each cycle has an implicit cost equal to the fraction of battery capacity consumed, valued at the battery's replacement cost. This turns the cycle limit from a hard constraint into an economic tradeoff.

---

## The Value of Information

### How Much Does Forecast Quality Matter?

This is the most commercially important question in the entire course: **what is a better forecast worth in dollars?** The answer determines whether investing in forecast improvements (better models, more data, more computation) is justified by the additional revenue.

<div class="definition-box">
<strong>Value of information:</strong> The economic benefit of having better information (a more accurate forecast) when making a decision. For battery dispatch, the value of information is the difference in revenue between a dispatch based on a perfect forecast (perfect foresight) and a dispatch based on the actual forecast. It measures how much money is "left on the table" due to forecast imperfection.
</div>

We can quantify this by **deliberately degrading** the forecast — adding noise of increasing magnitude — and measuring the capture ratio at each noise level. This traces the **value-of-information curve**:

![Value of information curve](figures/09_value_of_info.png)

<p class="figure-caption">Figure 9.2 — The value of information curve shows capture ratio (CR) as a function of forecast quality. The curve is concave: the first improvements from a poor forecast are highly valuable, while improvements from a good to an excellent forecast yield diminishing returns. This shape has profound implications for where to invest effort in the forecasting pipeline.</p>

Typical results for NEM battery dispatch:

| Forecast quality | MAE (approx.) | Capture ratio | Interpretation |
|-----------------|---------------|---------------|---------------|
| Perfect foresight | $0 | 1.00 | Theoretical maximum; achievable only with actual future prices |
| Excellent forecast | ~$15/MWh | 0.70–0.85 | Near the "knee" of the curve; further improvement has diminishing returns |
| Good forecast | ~$25/MWh | 0.55–0.70 | Achievable with a well-tuned GBT model |
| Mediocre forecast | ~$50/MWh | 0.35–0.50 | A simple model or poorly tuned model |
| Random forecast | N/A | ~0.00 | No systematic profit possible |

The relationship is **concave** — each successive improvement in forecast quality yields a smaller increase in capture ratio. This concavity has a simple explanation: the LP mainly needs to get the **price ranking** correct — which periods are cheapest (for charging) and most expensive (for discharging). Getting the ranking right is easier than getting the exact prices right. A forecast with MAE = $30 might rank the periods correctly 90% of the time, while a forecast with MAE = $15 ranks them correctly 95% of the time. The first forecast captures most of the arbitrage value because the ranking is mostly correct; the second forecast captures slightly more because of the remaining 5% of mis-ranked periods.

<div class="key-point">
<strong>The 80/20 rule of forecast value:</strong> The first 80% of forecast quality improvement (from random to decent) captures roughly 80% of the available revenue. The last 20% of quality improvement (from decent to excellent) captures only the remaining 20% of revenue. This means that beyond a certain accuracy level, investing further in forecast quality has diminishing returns — the effort is better directed at other parts of the pipeline (dispatch strategy, battery hardware, market access, ancillary services).
</div>

<div class="example-box">
<strong>Real-world example — the dollar value of forecast improvement:</strong> Consider a 100 MW / 2h battery in SA1 with a perfect-foresight revenue of $10M/year.

A good forecast (CR = 0.65) generates $6.5M/year.
An excellent forecast (CR = 0.75) generates $7.5M/year.
The improvement is worth $1M/year.

Is this $1M/year worth the cost of the better forecast? If the better forecast requires hiring two additional data scientists ($400K/year combined), upgrading weather data subscriptions ($100K/year), and more compute infrastructure ($50K/year), the answer is clearly yes — the $550K investment yields $1M in additional revenue.

But going from excellent (CR = 0.75) to near-perfect (CR = 0.85) would require $1M in additional revenue. The cost of achieving this further improvement — perhaps a dedicated ML research team, proprietary data sources, real-time infrastructure — might exceed the $1M benefit. This is where the concavity of the value curve bites: the last percentage points of capture ratio are the most expensive to achieve.
</div>

---

## The Economic Assessment

### Revenue Decomposition

Total battery revenue in the NEM comes from several distinct streams:

<div class="definition-box">
<strong>Revenue stack:</strong> The combination of revenue streams available to a battery. In the NEM, the main revenue sources are energy arbitrage, FCAS (frequency control ancillary services), network support, and capacity payments. Each stream has different characteristics — arbitrage depends on price volatility, FCAS depends on frequency deviation events, and capacity payments depend on market design and availability.
</div>

1. **Energy arbitrage:** Buy electricity at low prices, sell at high prices. This is what the dispatch LP optimises and what our capture ratio measures. It is typically the largest single revenue stream, accounting for 40–60% of total battery revenue in the NEM.

2. **FCAS (Frequency Control Ancillary Services):** Revenue from helping stabilise the grid frequency by providing rapid power injections or absorptions. Batteries are excellent at FCAS because they can respond within milliseconds. FCAS revenue is not modelled in this project but typically adds 20–40% to arbitrage revenue.

<div class="definition-box">
<strong>FCAS (Frequency Control Ancillary Services):</strong> Services provided to AEMO to maintain the grid frequency at 50 Hz. When a generator trips or demand changes suddenly, the frequency deviates from 50 Hz. FCAS providers (including batteries) respond by injecting or absorbing power to restore the frequency. There are eight FCAS markets in the NEM, each with its own price determined by a separate auction. Batteries are particularly valued for FCAS because of their extremely fast response time (milliseconds vs. seconds for gas turbines).
</div>

3. **Network support:** Revenue from reducing congestion on transmission lines. If a battery is located on the right side of a congested interconnector, it can charge when the interconnector is constrained (absorbing local excess) and discharge when it is unconstrained. This service is typically contracted with the network operator.

4. **Capacity payments:** Revenue from being available to generate electricity when needed, regardless of whether the battery actually discharges. Capacity markets pay generators (including batteries) to be available during peak demand periods, providing a revenue floor that supplements energy market earnings.

<div class="key-point">
<strong>Our capture ratio measures only the energy arbitrage component.</strong> A full economic assessment would include FCAS, network support, and capacity payments. In most NEM regions, energy arbitrage alone is currently insufficient for a positive net present value (NPV) — the other revenue streams are needed to make the investment case. However, as renewable penetration increases and price volatility grows, the arbitrage opportunity is expanding rapidly.
</div>

### From Capture Ratio to Annual Revenue

The capture ratio converts forecast quality into economic value through a simple formula:

<div class="equation">

Annual Arbitrage Revenue = CR × Perfect Foresight Revenue

</div>

<div class="definition-box">
<strong>Perfect foresight revenue:</strong> The maximum possible arbitrage revenue, achieved by a battery that knows all future prices in advance and dispatches optimally. This is computed by solving the dispatch LP with actual (historical) prices instead of forecasts. It represents the theoretical upper bound on arbitrage revenue and depends on the battery's specifications (power, duration, efficiency) and the price volatility of the region.
</div>

Perfect foresight revenue depends on the battery's physical specifications and the market's price volatility. Typical values for a 100 MW / 2h lithium-ion battery (matching the Mannum BESS specifications):

| Region | Annual perfect foresight revenue | Driver |
|--------|--------------------------------|--------|
| SA1 | $8M–$15M | Highest volatility; large renewable share |
| QLD1 | $5M–$10M | Summer spikes; growing solar |
| NSW1 | $4M–$8M | Moderate volatility; largest market |
| VIC1 | $4M–$7M | Growing wind; transitioning generation |
| TAS1 | $2M–$4M | Hydro-buffered; lowest volatility |

These figures vary significantly year to year depending on weather patterns (hot summers drive spikes), coal plant outages (remove cheap baseload, pushing up prices), and renewable output (high wind/solar reduces prices).

<div class="example-box">
<strong>Real-world example — putting it all together:</strong> The Mannum BESS (100 MW / 200 MWh, LFP) in SA1 with perfect foresight revenue of $12M/year, using a well-tuned GBT model with QRA combination and chance-constrained dispatch:

- Capture ratio: 0.70
- Annual arbitrage revenue: $12M × 0.70 = $8.4M
- Estimated FCAS revenue (additional ~30%): $2.5M
- Total annual revenue: ~$10.9M
- Battery installed cost: ~$60M (at current prices)
- Simple payback: ~5.5 years
- Operating costs (maintenance, grid fees): ~$1M/year
- Net annual revenue: ~$9.9M
- Net payback: ~6 years

With current lithium-ion costs and NEM price volatility, a well-sited, well-operated battery in SA1 can achieve payback in 5–7 years with a 15–20+ year useful life — a compelling investment case.
</div>

---

## Regional Basis and Inter-regional Trade

The NEM is not one market — it is five regional markets connected by transmission interconnectors. Most of the time, prices in adjacent regions are similar. But when an interconnector reaches its capacity limit, the regions **decouple** and prices can diverge dramatically. This has direct implications for battery revenue.

### Why regional prices separate

Each NEM region (SA1, VIC1, NSW1, QLD1, TAS1) has its own spot price, determined by the supply and demand balance *within* that region plus the flows on interconnectors to neighbouring regions.

When interconnector capacity is sufficient, cheap generation in one region flows to its neighbours, equalising prices. When an interconnector **binds** (reaches its transfer limit), the regions on either side clear independently:

<div class="example-box">
<strong>Example — the Heywood interconnector:</strong> Victoria normally exports cheap brown coal and wind generation to South Australia via the Heywood interconnector (nominal capacity ~600 MW). On a calm evening when SA wind drops and demand rises, Heywood reaches its limit. SA must rely on its own expensive gas generators, pushing SA1 prices to $300/MWh while VIC1 remains at $80/MWh. The $220 spread is the **basis** between the two regions. A battery in SA1 benefits from this spread; a battery in VIC1 does not see the same spike.
</div>

<div class="definition-box">
<strong>Basis (in electricity markets):</strong> The price difference between two related markets — typically two NEM regions, or a regional price versus a contract price. When an interconnector binds, the basis between connected regions widens. Basis risk is the risk that a financial hedge or forecast is referenced to one region while the battery settles at another.
</div>

### Basis risk for battery operators

A battery's revenue is determined by the price at its **connection point**, which is the regional reference price scaled by its marginal loss factor (discussed further in Chapter 13). Basis risk arises in several ways:

- **Regional mismatch:** A financial hedge referenced to the NSW1 price does not protect a battery in SA1 when the interconnector binds and prices separate.
- **Forecast basis:** A model trained on SA1 prices is correct for a battery in SA1, but applying it to VIC1 during a decoupled period would produce incorrect dispatch signals.
- **Inter-regional arbitrage:** Some batteries near interconnector nodes can exploit price separation — charging from the cheap region and discharging into the expensive one — but this requires the physical capability to affect flows, which most batteries do not have.

<div class="key-point">
<strong>For the Mannum BESS in SA1:</strong> SA has the highest price volatility in the NEM, partly because it frequently decouples from Victoria when the Heywood interconnector binds. This decoupling is a feature, not a bug, for an SA1 battery — it produces the extreme price spreads that drive arbitrage revenue. Understanding when decoupling is likely (high SA demand, low wind, interconnector outages) is a forecasting edge.
</div>

### Settlement residues

When power flows from a low-price region to a high-price region across an interconnector, the price difference creates a surplus — the **inter-regional settlement residue (IRSR)**. AEMO does not keep this surplus; it is auctioned quarterly as **Settlement Residue Auction (SRA) units**, which are financial instruments used by participants to hedge inter-regional basis risk.

<div class="definition-box">
<strong>Inter-regional settlement residue (IRSR):</strong> The revenue surplus that arises when electricity flows across an interconnector from a lower-priced region to a higher-priced region. The residue equals the price difference multiplied by the flow. These residues are auctioned to market participants through the Settlement Residue Auction (SRA), providing a mechanism for hedging inter-regional price risk.
</div>

For a data scientist on a battery desk, settlement residues are primarily context — the trading team uses SRA units to manage basis risk. The forecasting implication is simpler: interconnector flows and constraints are strong predictors of regional price divergence, and including interconnector data (available via NEMOSIS) as features can improve regional price forecasts, particularly for SA1 where decoupling events drive the largest price spikes.

---

## Glossary

| Term | Definition |
|------|-----------|
| **MPC (Model Predictive Control)** | A control strategy that repeatedly solves an optimisation over a rolling horizon |
| **Stochastic programming** | Optimisation under uncertainty using multiple scenarios |
| **Two-stage stochastic program** | Decisions made before uncertainty resolves (first stage); outcomes computed after (second stage) |
| **Scenario** | One complete realisation of uncertain future prices |
| **Expected value** | The probability-weighted average of outcomes across all scenarios |
| **Copula** | A function describing the dependence structure between random variables |
| **Robust optimisation** | Optimisation that protects against worst-case outcomes |
| **Chance constraint** | A constraint required to hold with at least a specified probability |
| **Capture ratio (CR)** | Actual dispatch revenue divided by perfect-foresight revenue |
| **Battery duration** | Hours the battery can sustain maximum output (energy/power ratio) |
| **Round-trip efficiency (RTE)** | Fraction of energy recovered from a charge-discharge cycle |
| **Battery cycle** | One complete charge-discharge (or equivalent energy throughput) |
| **Battery degradation** | Gradual loss of capacity due to cycling, aging, and operating conditions |
| **Value of information** | The economic benefit of having a better forecast |
| **Revenue stack** | The combination of revenue streams available to a battery |
| **FCAS** | Frequency Control Ancillary Services — grid stability services |
| **Capacity payments** | Revenue for being available to generate, regardless of actual dispatch |
| **Perfect foresight revenue** | Maximum arbitrage revenue achievable with perfect knowledge of future prices |
| **Breakeven spread** | The minimum price ratio between discharge and charge for profitable operation |
| **Law of large numbers** | Average of many random samples converges to the true expected value |
| **Concavity** | Diminishing marginal returns — each additional unit of input yields less additional output |
| **Basis** | Price difference between two related markets (e.g., two NEM regions) |
| **Basis risk** | Risk that a hedge or forecast is referenced to a different price than the battery settles at |
| **Interconnector** | Transmission link connecting two NEM regions, with a finite transfer capacity |
| **Binding constraint** | When an interconnector reaches its transfer limit, decoupling regional prices |
| **Inter-regional settlement residue (IRSR)** | Revenue surplus from price differences across interconnector flows |
| **Settlement Residue Auction (SRA)** | Quarterly auction of rights to inter-regional settlement residues |

## Summary

The journey from forecast to money requires a dispatch strategy that uses the full probabilistic forecast, not just the point prediction. Scenario MPC generates multiple plausible price trajectories and optimises expected revenue across all of them, hedging against forecast uncertainty. Chance-constrained MPC takes a simpler, more conservative approach — valuing discharge revenue at a low quantile and charge cost at a high quantile, ensuring profitability in pessimistic scenarios. Battery physical parameters — duration, round-trip efficiency, and cycle limits — interact with forecast quality to determine the capture ratio: the fraction of perfect-foresight revenue actually achieved. The value of information is concave: improving a poor forecast is extremely valuable, but improving an already good forecast yields diminishing returns because the dispatch LP mainly needs the correct price ranking, not exact price magnitudes. For a 100 MW / 2h battery in SA1, each percentage point of capture ratio is worth roughly $80K–$150K per year — a concrete economic translation of forecast quality improvements. Every link in the chain from probabilistic forecast to dispatch optimisation to battery hardware contributes to the final capture ratio, and understanding their relative contributions guides where to invest the next dollar of effort.
