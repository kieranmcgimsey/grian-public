# 3. Price Formation: How Prices Are Actually Set

## The Merit-Order Supply Stack

In Chapter 1, we described how AEMO dispatches generators from cheapest to most expensive. Now we examine this mechanism in detail — the **merit-order supply stack** — because understanding how prices form is essential for understanding what drives price changes.

![Supply stack](figures/03_supply_stack.png)

<p class="figure-caption">Figure 3.1 — The merit-order supply stack. Generators are arranged from cheapest (left) to most expensive (right). The price is set by the marginal generator — the last one dispatched to meet demand.</p>

<div class="definition-box">
<strong>Merit order:</strong> The ranking of available generators from lowest to highest marginal cost. AEMO dispatches generators in merit order — the cheapest first — until total generation meets demand. The "merit" is purely economic: the cheapest generator has the highest merit.
</div>

<div class="definition-box">
<strong>Supply stack:</strong> A graphical representation of the merit order, showing cumulative generation capacity (x-axis) against bid price (y-axis). It forms a staircase shape, stepping up as more expensive generators are added. Where the demand line intersects the supply stack determines the market price.
</div>

### The generators in the stack

Each fuel type occupies a characteristic position in the merit order:

| Fuel type | Typical marginal cost | Position in stack | Why |
|-----------|----------------------|-------------------|-----|
| **Solar** | ~$0/MWh | Bottom | No fuel cost — sunlight is free |
| **Wind** | ~$0/MWh | Bottom | No fuel cost — wind is free |
| **Hydro** | $10–30/MWh | Low-middle | Water has opportunity cost but no fuel cost |
| **Coal** | $25–50/MWh | Middle | Cheap fuel but inflexible |
| **Gas (CCGT)** | $60–100/MWh | Upper-middle | Moderate fuel cost, efficient |
| **Gas (OCGT)** | $100–300/MWh | Top | Expensive fuel, inefficient, but fast to start |

<div class="definition-box">
<strong>Marginal cost:</strong> The cost of producing one additional unit of output. For a gas plant, this is primarily the cost of the gas burned. For solar and wind, the marginal cost is essentially zero — once the farm is built, producing one more MWh costs nothing (the fuel is free). Note: marginal cost is NOT the same as total cost. A solar farm has high construction costs but zero marginal costs.
</div>

<div class="definition-box">
<strong>CCGT (Combined Cycle Gas Turbine):</strong> A gas plant that uses waste heat from the gas turbine to drive a steam turbine, achieving higher efficiency (50–60%). More efficient but slower to start than OCGT.
</div>

<div class="definition-box">
<strong>OCGT (Open Cycle Gas Turbine):</strong> A simple gas turbine that discards waste heat. Less efficient (30–40%) but can start from cold in 10–15 minutes. Used as "peaker" plants — they only run during high-demand periods when prices justify their high fuel cost.
</div>

### How demand position determines price

The critical insight: **the price depends not on the total generation mix, but on which generator is at the margin.**

<div class="example-box">
<strong>Example — low demand (1 AM):</strong> Only 1,000 MW of demand. Solar and wind supply 600 MW, coal supplies the remaining 400 MW. The marginal generator is a coal plant bidding $35/MWh. All generators — solar, wind, and coal — receive $35/MWh. Price: <strong>$35</strong>.
</div>

<div class="example-box">
<strong>Example — moderate demand (11 AM):</strong> 1,500 MW of demand. Solar and wind supply 900 MW, coal supplies 500 MW, and a CCGT gas plant supplies the final 100 MW. The marginal generator is the gas plant bidding $75/MWh. All generators receive $75/MWh — even the solar farms with zero fuel cost. Price: <strong>$75</strong>.
</div>

<div class="example-box">
<strong>Example — extreme demand (6 PM heatwave):</strong> 2,200 MW of demand. Solar is gone (sunset). Wind supplies 300 MW, coal 500 MW, CCGT 300 MW, and OCGT peakers supply the final 100 MW. The marginal OCGT bids $250/MWh. Price: <strong>$250</strong>. A 43% increase in demand (from 1,500 to 2,200 MW) caused a 233% increase in price — this is the nonlinear relationship between demand and price.
</div>

## The Hockey Stick: Net Load vs Price

When we plot net load against price across thousands of observations, a striking pattern emerges — the **hockey stick**.

![Hockey stick](figures/03_hockey_stick.png)

<p class="figure-caption">Figure 3.2 — Left: the hockey stick relationship between net load and price. Prices are roughly flat at low net load (renewables satisfy demand) and rise exponentially at high net load (expensive peakers set the price). Right: net load decomposition.</p>

<div class="definition-box">
<strong>Hockey stick (in electricity pricing):</strong> The characteristic nonlinear relationship between net load and price. At low net load, prices are flat (cheap generators have spare capacity). At high net load, prices rise exponentially as expensive peakers are dispatched. The plot resembles a hockey stick lying flat with its blade curving upward.
</div>

### Why the hockey stick is so important

The hockey stick is arguably the most important empirical relationship in electricity price forecasting:

1. **It explains why renewable energy depresses prices.** More solar/wind → lower net load → we stay in the flat part of the stick → low prices. This is called the **merit-order effect** of renewables.

2. **It explains why small demand changes cause huge price changes.** On the steep part of the stick, a 5% increase in net load can double or triple the price.

3. **It explains why price spikes are hard to predict.** The transition from the flat part to the steep part is abrupt and depends on the exact position of generators in the stack, which changes day to day.

<div class="definition-box">
<strong>Merit-order effect:</strong> The tendency of zero-marginal-cost renewables (solar, wind) to push down wholesale electricity prices by displacing expensive generators from the merit order. When wind and solar are abundant, the demand that would otherwise be met by gas peakers is met by renewables, and the marginal generator is a cheaper coal plant. The more renewables in the system, the stronger this effect.
</div>

<div class="key-point">
<strong>Key modelling insight:</strong> Net load is the single most powerful predictor of electricity prices — more powerful than demand alone, because it captures the offsetting effect of renewables. A model that includes net load can capture the hockey stick shape and the merit-order effect simultaneously.
</div>

## Anatomy of a Price Spike

Price spikes are the most dramatic feature of electricity markets. A spike can take the price from $50 to $15,000 in a single 5-minute interval. Understanding the mechanism helps us forecast them — or at least understand when they are more likely.

![Spike anatomy](figures/03_spike_anatomy.png)

<p class="figure-caption">Figure 3.3 — The three phases of a price spike: build-up (solar drops, gas ramps), the spike itself (demand exceeds available supply), and the response (batteries discharge, demand drops).</p>

### The three phases of a spike

**Phase 1 — Build-up (1–3 hours before):**
- Solar generation ramps down as the sun sets
- Demand stays high or rises (evening cooking, heating/cooling)
- Net load climbs steadily
- Gas plants start ramping up
- Prices rise from $50 to $100–200

**Phase 2 — The spike (minutes):**
- Net load exceeds comfortable supply margin
- The market clears on the most expensive generators
- Some generators bid strategically at the market price cap
- Prices jump to $1,000–$15,000+ in a single interval

**Phase 3 — Response (30–60 minutes):**
- Batteries discharge, injecting stored energy
- Price-responsive loads reduce consumption
- AEMO may direct interconnector flows or issue reliability interventions
- Prices fall back below $200

<div class="definition-box">
<strong>Price spike:</strong> An abrupt, temporary, extreme increase in the wholesale electricity price, typically lasting one to a few trading intervals. There is no universal threshold, but common definitions include prices above $300/MWh, $1,000/MWh, or the 99th percentile of historical prices. Spikes account for a tiny fraction of all intervals but contribute disproportionately to annual wholesale cost.
</div>

### What causes spikes

| Cause | Mechanism | Predictability |
|-------|-----------|---------------|
| **Heatwaves** | High A/C demand, low wind | Moderate — weather forecasts give 2–5 day warning |
| **Generator trip** | Large unit unexpectedly fails | Low — can happen any time |
| **Sunset ramp** | Solar drops faster than gas can ramp | High — happens predictably at sunset |
| **Interconnector failure** | Region isolated from cheap imports | Low — equipment failure |
| **Strategic bidding** | Generators bid high when they have market power | Low — depends on competitor behaviour |

<div class="example-box">
<strong>Real-world example — the February 2017 SA heatwave:</strong> Temperatures exceeded 40°C for three consecutive days. Air conditioning load pushed demand to record levels. Wind generation was below average. SA1 prices exceeded $1,000/MWh for 18 intervals (9 hours) over the three days, with peaks above $14,000/MWh. The total excess cost to retailers was estimated at over $50 million. A battery operator who had forecast even half of these spikes correctly could have earned several million dollars in those three days alone.
</div>

## Generator Bidding Behaviour

The supply stack is not fixed — generators can change their bids strategically, shifting the price-setting dynamics.

<div class="definition-box">
<strong>Bidding behaviour:</strong> The strategy generators use when submitting price-quantity offers to AEMO. While the merit order assumes generators bid at their marginal cost, in practice generators consider market conditions, competitor behaviour, and their portfolio position when setting bids.
</div>

### Strategic bidding

Generators with **market power** (large share of capacity in a region) can influence prices by:

1. **Withholding capacity:** Bidding some capacity at the market price cap so it won't be dispatched, reducing supply and raising the price for the remaining capacity.

2. **Rebidding:** Changing bids intra-day in response to changing conditions. A generator might bid cheaply in the morning (to be dispatched) then rebid expensive in the afternoon (when demand is higher and competition is thinner).

<div class="definition-box">
<strong>Market power:</strong> The ability of a single market participant to influence the market price. In electricity markets, market power arises when a single generator or portfolio controls enough capacity that withholding some of it would materially raise the price. Market power is more prevalent in smaller, more concentrated regions like SA1.
</div>

### Why bidding matters for forecasting

Strategic bidding introduces an element of **game-theoretic uncertainty** into price formation. Even with perfect weather forecasts and exact demand predictions, the price depends on *how generators choose to bid* — which is itself a strategic decision based on their expectations of market conditions.

This is a fundamental limit on forecasting accuracy: the price is partly determined by human decisions that respond to expectations, creating a feedback loop. No model can perfectly predict strategic behaviour.

## Supply Mix Transition

The NEM is in the middle of a generational transition from fossil fuels to renewables. This transition is reshaping the price formation dynamics:

| Trend | Effect on prices | Time scale |
|-------|-----------------|------------|
| Coal plant retirements | Higher prices when coal was the marginal unit | Years |
| More solar | Lower midday prices, deeper duck curve belly | Ongoing |
| More wind | Lower prices generally, but intermittent | Ongoing |
| More batteries | Price smoothing, faster spike response | Emerging |
| Electrification (EVs, heat pumps) | Higher demand, new load shapes | 5–15 years |

<div class="key-point">
<strong>Implication for forecasting:</strong> The supply stack is not static — it shifts over months and years as generators enter and exit. A model trained on 2020 data may not perform well on 2025 data because the generation mix has changed. This is why rolling retraining (updating the model with recent data) is essential.
</div>

---

## Market Institutions and Rules

Understanding who runs the NEM and what guardrails exist is essential vocabulary for working on a battery trading desk. This section provides the institutional context — the rules of the game that shape every price the forecasting models try to predict.

### Who runs the market

Three bodies govern the NEM:

| Institution | Role |
|-------------|------|
| **AEMO** (Australian Energy Market Operator) | Operates the market and the power system in real time — dispatches generators, publishes prices, manages system security |
| **AER** (Australian Energy Regulator) | Monitors compliance, investigates market manipulation, enforces the rules |
| **AEMC** (Australian Energy Market Commission) | Makes and amends the rules that AEMO and participants operate under |

<div class="definition-box">
<strong>AEMO (Australian Energy Market Operator):</strong> The independent operator responsible for running the NEM's dispatch engine (NEMDE) every five minutes, publishing spot prices, managing system security, and producing demand and price forecasts (pre-dispatch). AEMO does not own generators or set prices — it runs the auction that determines them.
</div>

### The market price cap and floor

The NEM spot price is bounded:

- **Market Price Cap (MPC):** The maximum price any generator can bid. As of 2024–25, the MPC is $17,500/MWh, indexed annually to CPI. This cap exists to prevent extreme exploitation of market power during supply shortages. The cap has risen over time — it was $12,500/MWh before 2010.
- **Market Floor Price (MFP):** The minimum price, set at −$1,000/MWh. Negative prices occur when generators (typically renewables with contracts that pay them regardless of the spot price) bid below zero to avoid being constrained off. A battery charging at −$200/MWh is being *paid* to take electricity.

<div class="example-box">
<strong>Why the cap matters for forecasting:</strong> The market price cap creates a hard ceiling on price spikes. A model that predicts $25,000/MWh is wrong by construction — the price cannot exceed $17,500/MWh. More subtly, the cap compresses the upper tail of the price distribution: events that would "naturally" produce different extreme prices are all capped at the same value, making the tail harder to model statistically. When reporting capture ratios, the cap also limits the upside — even perfect foresight cannot earn more than the capped price.
</div>

<div class="key-point">
<strong>Check the current MPC:</strong> The MPC is indexed annually. Before asserting a specific value, verify the current cap at AEMO's website. The value stated here ($17,500/MWh for 2024–25) is indicative.
</div>

### Administered pricing and the cumulative price threshold

When prices spike persistently, AEMO can intervene:

- **Cumulative Price Threshold (CPT):** If the rolling sum of spot prices over a seven-day window exceeds a threshold (currently $1,612,600, indexed annually), AEMO declares an **Administered Price Period** and caps the price at the **Administered Price Cap (APC)** of $600/MWh. This prevents sustained extreme pricing from causing systemic financial distress among retailers and large consumers.

<div class="definition-box">
<strong>Administered Price Period:</strong> A period triggered when cumulative spot prices over a rolling seven-day window exceed the Cumulative Price Threshold. During this period, AEMO caps the spot price at the Administered Price Cap ($600/MWh), dramatically reducing generator revenue. For a battery, an administered price period truncates the arbitrage opportunity — a forecasting model should flag when the CPT is approaching.
</div>

### Scarcity signals: lack of reserve and the reserve trader

AEMO publishes real-time indicators of system stress that affect prices:

- **Lack of Reserve (LOR):** AEMO declares LOR conditions when available generation reserves fall below specified thresholds. LOR1 is an early warning; LOR2 indicates insufficient reserves to cover the largest single contingency; LOR3 means involuntary load shedding is imminent. LOR declarations are strong signals of imminent price spikes — generators know reserves are scarce and bid accordingly.

- **Reliability and Emergency Reserve Trader (RERT):** When the market alone cannot ensure adequate reserves, AEMO can contract emergency reserves outside the normal market — paying generators or large consumers to be available. The cost of RERT is recovered from market participants. RERT activation signals extreme scarcity and near-certain elevated prices.

<div class="example-box">
<strong>LOR as a forecasting feature:</strong> Incorporating LOR declarations as features in a price forecasting model is straightforward — they are published in real time via AEMO's market notices. An LOR2 declaration during an evening peak is one of the strongest predictors of an imminent price spike, because it signals that the system is operating on the steep part of the supply stack with minimal margin.
</div>

### Beyond the spot market

The spot market is where our forecasting and dispatch operate, but battery operators also interact with adjacent markets:

- **Financial hedging:** Contracts for difference (CFDs), swaps, and caps that lock in future revenue at agreed prices, reducing exposure to spot price volatility. These are managed by the trading team, not the data science team, but they create incentives that shape dispatch strategy.
- **Capacity mechanisms:** The NEM does not currently have a formal capacity market (unlike the UK or US PJM), but capacity-like mechanisms are evolving. The Capacity Investment Scheme (CIS) provides revenue certainty for new renewable and storage investment.

These instruments sit with trading and risk management, not the forecasting role, but understanding that they exist explains why a desk sometimes dispatches differently from what the spot forecast alone would suggest — the battery may be managing a hedge position.

---

## Glossary

| Term | Definition |
|------|-----------|
| **Merit order** | Ranking of generators from cheapest to most expensive |
| **Supply stack** | Visual representation of cumulative generation vs bid price |
| **Marginal cost** | Cost of producing one additional unit of output |
| **CCGT** | Combined Cycle Gas Turbine — efficient gas plant |
| **OCGT** | Open Cycle Gas Turbine — fast-start but expensive peaker |
| **Hockey stick** | Nonlinear net-load-to-price relationship |
| **Merit-order effect** | Price depression caused by zero-cost renewables |
| **Price spike** | Abrupt, extreme, temporary price increase |
| **Market power** | Ability of one participant to influence the price |
| **Strategic bidding** | Setting bids based on market conditions rather than cost |
| **AEMO** | Australian Energy Market Operator — runs the dispatch and publishes prices |
| **AER** | Australian Energy Regulator — monitors compliance and enforces rules |
| **AEMC** | Australian Energy Market Commission — makes and amends market rules |
| **Market Price Cap (MPC)** | Maximum allowable spot price ($17,500/MWh in 2024–25, indexed to CPI) |
| **Market Floor Price** | Minimum spot price (−$1,000/MWh) |
| **Administered Price Period** | AEMO-imposed price cap triggered by sustained extreme pricing |
| **Cumulative Price Threshold** | Rolling seven-day price sum that triggers administered pricing |
| **Lack of Reserve (LOR)** | AEMO declaration indicating insufficient generation reserves |
| **RERT** | Reliability and Emergency Reserve Trader — AEMO's emergency reserve mechanism |

## Summary

Electricity prices are set by the merit-order supply stack — the cheapest generators dispatched first, with the most expensive one setting the price for all. The hockey stick relationship between net load and price is the most important pattern: prices are flat when renewables dominate and rise exponentially when expensive peakers are needed. Price spikes occur when demand exceeds comfortable supply margins, driven by heatwaves, sunset ramps, generator trips, or strategic bidding. The NEM's generation mix is transitioning rapidly from coal to renewables, continuously reshaping the supply stack and the price formation dynamics.
