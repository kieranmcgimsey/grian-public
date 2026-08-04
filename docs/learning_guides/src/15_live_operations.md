# 15. Live Operations and the Intra-day Loop

## Why Operations Is a Separate Discipline

Everything up to this point has been built offline. The forecasting models (Chapters 6--8), the dispatch optimiser (Chapter 9), the market emulator (Chapter 12), and the decision-regret framework (Chapter 14) were all developed, tested, and tuned against historical data. They work. But a model that works on historical data and a model that works in real-time production are different things, and the gap between them is not a modelling gap --- it is an engineering and operational gap.

The Mannum BESS --- Epic Energy's 100 MW / 200 MWh lithium iron phosphate battery in SA1, optimised by Habitat Energy --- was commissioned in October 2025. It participates in the NEM's 5-minute dispatch market, submitting bid stacks to AEMO and receiving dispatch instructions every five minutes. This chapter describes what happens between "the model produces a forecast" and "the battery earns revenue," and how to keep the system healthy once it is running.

<div class="key-point">
<strong>The shift from backtesting to operations:</strong> In a backtest, you have all the data before you start. In operations, data arrives incrementally, forecasts must be produced on a clock, decisions have deadlines, and the consequences of failure are measured in dollars, not error metrics. The operational loop is where forecast quality is converted into captured revenue --- or lost to latency, staleness, and unmonitored drift.
</div>

This chapter is shorter and more descriptive than the modelling chapters. It covers four topics: intra-day re-optimisation, the operational loop and its clock, monitoring and retraining, and risk controls and human oversight. The worked example runs a single volatile day through an intra-day simulation to show the revenue benefit of updating the plan as new information arrives.

---

## Intra-day Re-optimisation

### The Cost of a Stale Plan

The dispatch chapters (Chapters 5 and 9) solved for a battery schedule once per day: take tomorrow's probabilistic price forecast, run the MPC optimiser, and produce a 48-half-hour charge/discharge plan. In calm markets, this works well. Prices follow the forecast closely, the plan executes as intended, and the capture ratio is respectable.

Volatile days are different. On 24 January 2024 --- a heatwave in SA1 with temperatures above 42 degrees Celsius --- spot prices moved from $40/MWh at 10 AM to $6,500/MWh at 3 PM, fell to $120/MWh at 5 PM, then spiked again to $11,000/MWh at 7 PM. A day-ahead forecast made at 12:30 PM the previous day could not have predicted the timing or magnitude of these swings with precision. By 2 PM on the day itself, the battery's pre-committed plan was already stale: it had scheduled discharge for 5 PM based on yesterday's forecast, but the real spike was arriving two hours early.

<div class="definition-box">
<strong>Staleness:</strong> The degree to which a forecast or plan no longer reflects the current information set. A forecast made 18 hours ago is stale not because its model is wrong but because new data has arrived --- updated AEMO pre-dispatch forecasts, real-time demand readings, current renewable output --- that was not available when the forecast was produced. Staleness increases with the time since the last forecast update and with the rate at which market conditions are changing.
</div>

The cost of staleness is asymmetric. On calm days, updating the plan every 30 minutes versus once per day makes little difference --- the prices follow the day-ahead forecast and the plan remains approximately optimal. On volatile days, the cost of staleness can be enormous. A battery that discharges into a $200/MWh period because its stale plan said that was the peak, while the actual peak at $8,000/MWh arrives an hour later with the battery already empty, has missed the vast majority of the day's revenue. Volatile days contribute a disproportionate share of annual revenue (Chapter 2), so losing them to staleness has outsized consequences.

### From LP Re-solve to Emulator Re-optimisation

In Chapter 5, we introduced the basic receding-horizon MPC: at each step, re-solve the dispatch LP with updated price forecasts, execute the first period's decision, and repeat. This works when the LP accurately models the market, but the NEM's dispatch process is more complex than a simple price-taker LP assumes. Interconnector constraints, generator ramping limits, and strategic bidding by other participants all affect prices in ways the LP ignores.

Chapter 12 introduced the nempy market emulator --- a model that replays the AEMO dispatch engine with modified bid stacks, allowing us to ask "what would the price be if our battery bid differently?" The emulator captures price-impact effects that the price-taker LP misses: when the Mannum BESS discharges 100 MW into a tight SA1 market, it pushes the price down; when it charges, it pulls the price up. These effects are small in liquid periods but material during the volatile periods where most revenue is earned.

<div class="definition-box">
<strong>Intra-day re-optimisation:</strong> The practice of updating the battery's dispatch plan multiple times within a single trading day as new information arrives. Rather than committing to a single day-ahead plan, the system re-solves the optimisation problem at regular intervals (typically every 30 minutes or every 5 minutes), incorporating updated price forecasts, current state of charge, and any new market information. Each re-optimisation replaces the remaining plan; only the next period's decision is executed.
</div>

The intra-day loop in this chapter re-optimises through the Chapter 12 emulator, not just the LP. At each re-optimisation step:

1. **Update the price forecast.** Incorporate the latest AEMO pre-dispatch forecasts, real-time demand data, and current renewable output.
2. **Set the current battery state.** Record the actual state of charge, which may differ from the plan due to AEMO dispatch instructions.
3. **Run the emulator.** For each candidate bid stack, simulate the remaining periods through the nempy emulator to estimate the resulting prices and dispatch outcomes.
4. **Solve the optimisation.** Find the bid stack that maximises expected revenue over the remaining horizon, accounting for price impact.
5. **Submit the bid.** Send the first period's bid to AEMO before gate closure.
6. **Wait for dispatch.** AEMO dispatches the battery and reports the actual price and dispatch quantity.
7. **Repeat from step 1.**

The key difference from the Chapter 9 LP approach is step 3: the emulator evaluates each candidate schedule against a simulated market, not against a fixed price forecast. This means the optimisation accounts for the battery's own effect on prices, producing more realistic (and typically more conservative) dispatch decisions. The LP and the emulator produce similar plans during low-volatility periods; during tight supply events --- heatwaves, generator outages, low wind --- the battery's dispatch is a meaningful share of the marginal supply, and the emulator's price-impact correction can shift the optimal discharge timing by one or two periods.

### Re-optimisation Frequency

How often should the system re-solve? There is a trade-off between information freshness and computational cost:

| Frequency | Info gain | Cost | Fit |
|-----------|-----------|------|-----|
| Day-ahead only | Baseline; no updates | Negligible | Calm markets |
| Every 30 min | PD updates, demand revisions | Low; ~48 LP or ~10 emulator/day | Good default |
| Every 5 min | Max freshness; real-time signals | Moderate; 288 iter/day | Needs fast solver |
| Event-triggered | Re-solves on material change | Variable by volatility | Efficient; complex |

For the Mannum BESS, a 30-minute re-optimisation cadence with event-triggered re-solves during high-volatility periods is a reasonable starting point. The emulator adds computational cost relative to the LP (roughly 2--5 seconds per run versus milliseconds for the LP), but even at 5-minute frequency the total computation is manageable on a single modern server.

---

## The Operational Loop

### The Market's Clock

The NEM operates on a strict temporal structure. Every decision the battery makes must respect this clock.

<div class="definition-box">
<strong>AEMO dispatch cycle:</strong> The NEM dispatches generation and load every 5 minutes. AEMO runs the National Electricity Market Dispatch Engine (NEMDE) at each 5-minute interval, determining the dispatch quantity and price for every registered unit. The dispatch price is the marginal price --- the cost of the next MW of supply needed to meet demand. Bids must be submitted before gate closure for each interval.
</div>

The key timestamps for a battery operator:

| Event | Timing | Action |
|-------|--------|--------|
| **PD publication** | Every 30 min; next 40 h | Price/demand forecasts per HH |
| **5MPD** | Every 5 min; next hour | Short-term price fcst (12 intervals) |
| **Gate closure** | ~2 min before interval | Bid submission deadline |
| **Dispatch** | Every 5 min on clock | NEMDE sets prices, issues targets |
| **Settlement** | Post-hoc; days later | Revenue from qty × price |

<div class="definition-box">
<strong>Gate closure:</strong> The deadline by which a market participant must submit or update their bids for a given dispatch interval. In the NEM, gate closure is approximately two minutes before the start of the dispatch interval. After gate closure, bids cannot be changed for that interval. This imposes a hard constraint on the operational loop: the forecast must be made, the optimisation must be solved, and the bid must be submitted before gate closure. Any latency in the pipeline --- data retrieval, model inference, optimisation, bid formatting --- directly reduces the time available for computation.
</div>

### Data Latency as a Real Constraint

In a backtest, all data is available instantly. In production, data has latency:

- **SCADA telemetry** (real-time demand, generation, interconnector flows): typically 4--10 seconds delay from AEMO's data feeds
- **Pre-dispatch forecasts** (30-minute PD and 5-minute PD): published on schedule but require ~30 seconds to download, parse, and ingest
- **Weather observations** (BOM synoptic data): updated hourly with ~15 minutes delay
- **Weather forecasts** (ERA5 or BOM ACCESS): updated every 6--12 hours with several hours delay
- **NEMSEER pre-dispatch traces**: available with ~1 minute delay after AEMO publication

The total pipeline latency --- from "new data is published" to "updated bid is submitted to AEMO" --- is the sum of data retrieval, forecast model inference, optimisation solve time, and bid submission overhead. For the intra-day loop to be useful, this total must be less than the re-optimisation interval. At 30-minute frequency, there is ample time. At 5-minute frequency, every second counts.

<div class="example-box">
<strong>Latency budget for 5-minute re-optimisation:</strong> Each 5-minute interval is 300 seconds. Gate closure is ~120 seconds before the interval starts, leaving ~180 seconds from the start of the previous interval to submit the updated bid. A typical latency budget:

- Data retrieval and parsing: 30 seconds
- Feature engineering: 5 seconds
- Forecast model inference (GBT + QRA): 10 seconds
- Emulator optimisation (10 scenario evaluations): 30 seconds
- Bid formatting and submission: 5 seconds
- **Total: ~80 seconds**, leaving ~100 seconds of buffer

This is comfortable on modern hardware. The bottleneck is data retrieval, not computation --- waiting for AEMO's feeds to update and downloading the latest pre-dispatch run dominates the latency budget.
</div>

### The Forecast-Bid-Settle Cycle

The complete operational loop for one dispatch interval:

![Operational loop](figures/15_operational_loop.png)

<p class="figure-caption">Figure 15.1 — The operational loop for one dispatch interval. Data arrives, the forecast updates, the emulator re-optimises, and the bid is submitted before gate closure. After dispatch, the actual price and dispatch quantity are recorded for settlement and monitoring.</p>

1. **Ingest.** Download the latest pre-dispatch forecasts, SCADA data, and any updated weather inputs. Parse and validate.
2. **Forecast.** Run the price model (Chapter 7 GBT + Chapter 8 QRA calibration) on the updated feature vector. Produce quantile forecasts for the remaining horizon.
3. **Optimise.** Pass the updated forecast and current state of charge to the emulator-based MPC (steps 1--4 from the re-optimisation procedure above). Obtain the optimal bid stack for the next interval.
4. **Bid.** Format the bid according to AEMO's bidding interface specifications and submit before gate closure.
5. **Dispatch.** AEMO runs NEMDE, issues the dispatch target to the battery, and publishes the dispatch price.
6. **Record.** Log the forecast, bid, dispatch target, actual price, state of charge, and any deviations. These records feed the monitoring system (next section).
7. **Settle.** At the end of the trading day, compute actual revenue from dispatch quantities and settlement prices. Compare to the plan's expected revenue.

<div class="key-point">
<strong>The bid is not the dispatch.</strong> A common misconception is that the battery controls its own dispatch. It does not. AEMO dispatches the battery according to its bid stack and the market's needs. If the battery bids to discharge 100 MW at $100/MWh and the dispatch price is $150/MWh, AEMO will dispatch the full 100 MW. But if the price is $80/MWh, the bid is not dispatched. The battery influences its dispatch through its bid prices, but AEMO makes the final decision. This distinction matters for the emulator --- the optimisation must find the bid prices that induce the desired dispatch outcome, not just the desired quantities.
</div>

---

## Monitoring

### What to Monitor

A production forecasting and dispatch system requires continuous monitoring across four dimensions: forecast quality, calibration quality, data integrity, and decision quality.

#### Forecast Drift

<div class="definition-box">
<strong>Model drift:</strong> The gradual degradation of a model's predictive performance over time, caused by changes in the underlying data distribution. In electricity markets, drift occurs because the generation fleet evolves (coal retires, new renewables are commissioned), demand patterns shift (electrification, rooftop solar adoption), and market rules change (5-minute settlement, capacity mechanisms). A model trained on 2023 data may perform poorly on 2025 data if the market's statistical properties have shifted materially.
</div>

Monitor the following forecast metrics on a rolling basis (e.g., trailing 7-day and 30-day windows):

- **Rolling MAE and RMSE** of the point forecast (median), compared to the training-period baseline. A sustained increase indicates drift.
- **Rolling CRPS** of the full quantile forecast. CRPS captures both calibration and sharpness, so a CRPS increase may indicate either drift or calibration degradation.
- **Forecast bias** (mean error). A bias that drifts away from zero indicates systematic over- or under-prediction, often caused by a structural change in the market (e.g., a new generator commissioning).

<div class="example-box">
<strong>Detecting drift in SA1:</strong> Suppose the model's 30-day rolling MAE was $15/MWh during training and validation. If, three months into production, the rolling MAE climbs to $22/MWh and stays there for two weeks, this is a signal of drift. Common causes in SA1: a large solar farm commissioning (pushing midday prices lower than the model expects), a gas plant retiring (removing supply and increasing afternoon prices), or a policy change affecting interconnector capacity between SA1 and VIC1.
</div>

#### Calibration Drift

The probabilistic forecast's calibration (Chapter 8) can degrade even when the point forecast remains accurate. Monitor:

- **PIT histogram** (probability integral transform): should be uniform. A consistent U-shape indicates underdispersion (intervals too narrow); an inverted-U indicates overdispersion.
- **Coverage rates** at key quantile levels (e.g., 10th, 50th, 90th). If the 90th percentile forecast is exceeded more than 10% of the time, the tails are underestimated.
- **Rolling calibration score** (average calibration error across quantiles). A sustained increase triggers recalibration.

#### Data-Quality Faults

Bad data is the most common cause of production failures, and it is entirely distinct from model drift:

- **Missing values.** AEMO feeds occasionally drop observations. The system must detect gaps and either interpolate or fall back to a simpler model.
- **Stale feeds.** A data source that stops updating (e.g., a weather API returning yesterday's data) produces forecasts based on old information without raising an obvious error.
- **Out-of-range values.** Negative demand, generation above nameplate capacity, or prices outside the NEM's market floor (--$1,000/MWh) and cap ($17,500/MWh) indicate data corruption.
- **Schema changes.** AEMO occasionally changes its data formats or column names. Automated ingestion that assumes a fixed schema will break silently.

<div class="key-point">
<strong>The data-quality hierarchy:</strong> Fix data-quality issues before investigating model drift. Nine times out of ten, a sudden performance drop is caused by bad data, not a deteriorating model. Build data-quality checks into the ingestion pipeline --- validate ranges, check for staleness, alert on missing values --- and investigate them first when performance degrades.
</div>

#### Decision-Regret Drift and Capture-Ratio Drift

Chapter 14 introduced **decision regret** --- the difference between the revenue actually earned and the revenue that would have been earned with the optimal decision given the actual price. Monitoring regret over time reveals whether the dispatch system is improving or degrading.

<div class="definition-box">
<strong>Decision-regret drift:</strong> A sustained change in the average decision regret over time. Increasing regret indicates that the dispatch system is making worse decisions --- either because the forecast has degraded, the optimiser is misconfigured, or the market has changed in ways the system does not account for. Decision-regret drift is a more direct measure of operational quality than forecast drift alone, because it captures the downstream effect of forecast errors on revenue.
</div>

<div class="definition-box">
<strong>Capture-ratio drift:</strong> A sustained change in the rolling capture ratio --- the ratio of actual dispatch revenue to perfect-foresight revenue (Chapter 9). A declining capture ratio over time indicates degrading operational performance. Unlike decision regret, capture ratio normalises for market volatility: a low-volatility week naturally has lower perfect-foresight revenue, so a stable capture ratio during a calm week indicates the same operational quality as during a volatile week. The 0.50 capture ratio bar is the benchmark; sustained performance below this level warrants investigation.
</div>

Monitor both decision regret and capture ratio on rolling windows:

- **7-day rolling capture ratio.** Compare to the backtest benchmark (0.50+). A drop below 0.45 sustained for more than a week triggers investigation.
- **30-day rolling capture ratio.** The strategic view. Compare to the same period in the previous year (seasonality-adjusted).
- **Rolling mean regret per interval.** Track the average "money left on the table" per dispatch interval. Increasing regret suggests the forecast or optimiser is degrading.
- **Regret decomposition.** As in Chapter 14, decompose regret into forecast error contribution and optimiser suboptimality. This tells you whether the problem is the forecast or the dispatch logic.

![Monitoring dashboard](figures/15_monitoring_dashboard.png)

<p class="figure-caption">Figure 15.2 — A monitoring dashboard panel for the Mannum BESS. Top: rolling 7-day capture ratio with the 0.50 benchmark line. Middle: rolling calibration score (lower is better) with a recalibration threshold. Bottom: rolling mean decision regret per interval, decomposed into forecast-error regret and optimiser regret. A spike in capture ratio (positive) on volatile days confirms the intra-day loop is working; a sustained decline in capture ratio triggers investigation.</p>

### Retraining vs Recalibration

When monitoring detects degradation, the response depends on the source:

<div class="definition-box">
<strong>Retraining cadence:</strong> The schedule on which a model is retrained from scratch (or fine-tuned) on updated data. Retraining is expensive --- it requires assembling a fresh training set, running hyperparameter optimisation, validating on a held-out period, and deploying the new model. A typical cadence for electricity price models is monthly or quarterly, unless drift monitoring triggers an earlier retrain.
</div>

| Symptom | Cause | Response |
|---------|-------|----------|
| PIT non-uniform, MAE stable | Quantile miscalibration | **Recalibrate** QRA/conformal |
| MAE rising, calibration degrading | Feature-target shift | **Retrain** GBT on recent data |
| Sudden MAE spike | Data fault (stale/missing) | **Fix data pipeline** |
| Capture ratio down, MAE stable | Optimiser or market change | **Check dispatch logic** |
| Regret biased one direction | Systematic forecast bias | **Bias-correct**; retrain if needed |

<div class="key-point">
<strong>Recalibrate first, retrain second.</strong> Recalibration (updating the conformal or QRA layer from Chapter 8) is fast, cheap, and often sufficient. It corrects for distribution shift without re-learning the entire feature-target relationship. Retraining is expensive and risky --- the new model might be worse than the old one if the training data is noisy or the hyperparameter search lands in a poor region. Retrain only when recalibration fails to restore performance.
</div>

A practical retraining workflow:

1. **Weekly:** Update the conformal calibration window (Chapter 8). This is automatic and requires no human intervention.
2. **Monthly:** Refit the QRA combination layer on the most recent 90 days of data. Check that the recalibrated model's coverage rates match the target levels.
3. **Quarterly:** Retrain the GBT model from scratch on the most recent 12--18 months of data. Run the full backtest (Chapter 5) to validate before deploying.
4. **Event-triggered:** After a major market event (generator closure, new interconnector, rule change), retrain immediately regardless of the schedule.

---

## Risk and Oversight

### Position and Exposure Limits

A battery trading in the NEM is exposed to spot price risk on every dispatch interval. If the battery is discharged (short stored energy) when prices spike, it earns revenue. If it is charged (long stored energy, consuming from the grid) when prices spike, it pays a large bill. Risk controls limit the downside.

<div class="definition-box">
<strong>Drawdown limit:</strong> A maximum allowable loss over a specified period (a single day, a rolling week) that triggers a protective action --- typically reverting to a safe state (hold current state of charge, do not trade) until a human trader reviews the situation. Drawdown limits protect against catastrophic losses from model failures, data errors, or unprecedented market conditions. They are the last line of defence when all upstream controls fail.
</div>

Key risk parameters for the Mannum BESS:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Max exposure/interval** | $500K | Cap single-decision loss |
| **Daily drawdown** | $200K net loss | Hold + alert trader |
| **Weekly drawdown** | $500K net loss | Longer-horizon cap |
| **Max SoC deviation** | 40 MWh (20%) | Triggers re-optimisation |
| **Min SoC** | 20 MWh (10%) | FCAS reserve |

<div class="example-box">
<strong>How a drawdown limit works in practice:</strong> Suppose the Mannum BESS starts the day with a plan to charge during the early morning (expected prices ~$30/MWh) and discharge in the afternoon (expected prices ~$150/MWh). At 8 AM, an unexpected generator trip pushes morning prices to $3,000/MWh. The battery is charging --- buying electricity at $3,000/MWh instead of $30/MWh. Over two hours, this costs an extra $500,000 versus the plan. The daily P&L drops to --$400,000, breaching the $200,000 drawdown limit.

The system halts trading: the battery holds its current state of charge and submits bids at the market cap (ensuring it will not be dispatched to charge) and at the market floor (ensuring it will not be dispatched to discharge). The human trader is alerted. They review the situation, determine whether the morning prices were a one-off event or the start of a sustained disruption, and either resume automated trading or adjust the risk parameters.
</div>

### Daily P&L Tracking

Revenue tracking is the simplest and most important operational metric. For each dispatch interval:

<div class="equation">

interval_revenue = dispatch_price × dispatch_quantity × (5/60) hours

</div>

where dispatch_quantity is positive for discharge (earning revenue) and negative for charge (paying cost). The daily P&L is the sum across all intervals, net of any FCAS revenue and grid fees. Track this against the plan's expected P&L and against perfect-foresight revenue:

- **Plan vs actual:** Measures execution quality. Large deviations indicate forecast error or AEMO dispatch deviations from the bid.
- **Actual vs perfect foresight:** The capture ratio, computed daily. On volatile days, even a 0.40 capture ratio can represent substantial revenue; on calm days, the perfect-foresight revenue itself is small.

### Human-in-the-Loop

<div class="definition-box">
<strong>Human-in-the-loop:</strong> An operational design in which a human trader or operator retains authority over certain decisions, typically high-stakes or unusual situations. The automated system handles routine operations (forecast, optimise, bid) but escalates to a human when conditions exceed predefined thresholds. The human provides judgement that the model cannot: awareness of events not in the data (regulatory announcements, weather warnings, equipment issues), risk appetite adjustments, and override authority when the model's behaviour is clearly wrong.
</div>

When should a human trader override the machine?

1. **Unprecedented events.** A market event with no historical analogue (e.g., the first activation of a new market rule, a cyberattack on grid infrastructure, a sudden regulatory intervention) is outside the model's training distribution. The model will extrapolate, often poorly. A human trader who understands the event's implications can make better decisions.

2. **Risk limit breaches.** When a drawdown limit is hit, the system halts and the human decides whether to resume, adjust parameters, or remain in hold.

3. **Known model weaknesses.** If the model has a documented weakness (e.g., poor performance during interconnector outages, as identified during backtest evaluation), the human can intervene during those specific conditions.

4. **Conflicting signals.** If the forecast says prices will be low but the pre-dispatch forecast shows a supply shortfall, or if weather data suggests a heatwave that the model has not yet incorporated, the human can override the forecast or adjust the dispatch plan.

5. **End-of-day position management.** A human trader may want to ensure the battery ends the day at a specific state of charge (e.g., fully charged to prepare for expected morning discharge) regardless of what the optimiser recommends.

<div class="key-point">
<strong>Automate the routine, escalate the exceptional.</strong> The operational loop should handle 95% of dispatch intervals without human intervention. The remaining 5% --- risk limit breaches, unprecedented events, known model blind spots --- should be escalated with clear information: what happened, what the model recommends, what the risk metrics show. The trader makes the call. A well-designed human-in-the-loop system makes the human's job easier, not harder, by filtering out the noise and presenting only the decisions that require judgement.
</div>

---

## Worked Example: Intra-day Re-optimisation on a Volatile Day

### Setup

We simulate the Mannum BESS (100 MW / 200 MWh, 85% round-trip efficiency) operating on 24 January 2024, a severe heatwave day in SA1. The simulation compares three strategies:

1. **Day-ahead only.** A single dispatch plan produced at 12:30 PM on 23 January, using the day-ahead price forecast. No intra-day updates.
2. **Intra-day re-optimisation.** The plan is re-solved every 30 minutes using the latest NEMSEER pre-dispatch forecasts, run through the Chapter 12 emulator.
3. **Perfect foresight.** The optimal plan computed with actual prices (the upper bound on achievable revenue).

We also note the Mannum BESS's actual dispatch and revenue for the same day as a real-world comparison.

### Price Profile

The actual SA1 prices on 24 January 2024:

| Time | Price ($/MWh) | Conditions |
|------|--------------|------------|
| 00:00--08:00 | $25--$45 | Overnight; moderate demand |
| 08:00--12:00 | $40--$80 | Morning ramp; solar generation increasing |
| 12:00--14:00 | $15--$30 | Solar peak; low net demand |
| 14:00--15:30 | $2,500--$6,500 | First spike; air conditioning load surges as temperature exceeds 42C |
| 15:30--17:00 | $80--$200 | Brief reprieve; cloud cover temporarily reduces cooling load |
| 17:00--19:30 | $4,000--$11,000 | Second spike; solar generation drops while cooling load remains high |
| 19:30--24:00 | $50--$120 | Evening; temperatures fall, demand eases |

### Day-Ahead Plan

The day-ahead forecast, made at 12:30 PM on 23 January, correctly predicted elevated afternoon prices but underestimated their magnitude and missed the two-spike structure. It forecast a single broad peak of $500--$1,200/MWh from 14:00 to 19:00.

The day-ahead plan:
- **Charge** 00:00--06:00 and 11:00--14:00 (cheap overnight and solar trough periods). Total charge: 200 MWh.
- **Discharge** 14:00--16:00 at 100 MW (200 MWh over 2 hours, accounting for RTE losses). Battery empty by 16:00.
- **Hold** for the rest of the day.

The plan discharges during the first spike (14:00--16:00) and earns revenue at the actual prices during that window. But it misses the larger second spike (17:00--19:30) entirely because the battery is empty by 16:00. The day-ahead plan discharged all its energy at the first opportunity, not knowing a bigger opportunity was coming.

**Day-ahead revenue: approximately $780,000.**

### Intra-day Re-optimisation

The intra-day loop re-solves every 30 minutes. Key decision points:

- **14:00 re-solve.** The 5-minute pre-dispatch now shows prices remaining elevated through 15:30. The emulator confirms that discharging at 100 MW will push prices down slightly but they remain high. The re-optimised plan continues discharging, consistent with the day-ahead plan.

- **15:30 re-solve.** Prices have dropped to $120/MWh, but the updated pre-dispatch shows a second spike forming at 17:00--19:00 with forecast prices of $3,000--$7,000/MWh. The battery has 60 MWh remaining (it discharged 140 MWh during 14:00--15:30 at a controlled rate rather than maximum power). The re-optimised plan: **hold** the remaining 60 MWh and discharge at 17:00.

- **16:00 re-solve.** Pre-dispatch prices for 17:00--19:00 have increased to $5,000--$9,000/MWh. The plan reconfirms: hold and wait.

- **17:00 re-solve.** Prices spike to $7,500/MWh. The emulator confirms high prices for the next 90 minutes. Discharge 60 MWh over 36 minutes at 100 MW.

The critical difference: the intra-day loop saw the second spike forming in the pre-dispatch forecasts and **reserved energy** for it. The day-ahead plan, blind to this information, had already exhausted the battery.

**Intra-day revenue: approximately $1,180,000.**

### Perfect Foresight

With perfect knowledge of the two-spike structure, the optimal plan:
- Charge overnight and during the solar trough: 200 MWh.
- Discharge 80 MWh during the first spike (14:00--15:30) at $4,500/MWh average.
- Hold during 15:30--17:00.
- Discharge 120 MWh (net of RTE) during the second spike (17:00--19:00) at $8,000/MWh average.

**Perfect-foresight revenue: approximately $1,550,000.**

### Comparison

| Strategy | Revenue | Capture ratio |
|----------|---------|--------------|
| Day-ahead only | $780,000 | 0.50 |
| Intra-day re-optimisation | $1,180,000 | 0.76 |
| Perfect foresight | $1,550,000 | 1.00 |

![Intra-day vs day-ahead revenue](figures/15_intraday_vs_dayahead.png)

<p class="figure-caption">Figure 15.3 — Revenue comparison for the Mannum BESS on 24 January 2024 (simulated). Left: cumulative revenue over the day for day-ahead only, intra-day re-optimisation, and perfect foresight. The intra-day strategy captures the second evening spike that the day-ahead plan misses. Right: state-of-charge trajectories showing how the intra-day loop reserves energy for the second spike while the day-ahead plan exhausts the battery by 16:00.</p>

<div class="key-point">
<strong>The value of re-optimisation:</strong> On this volatile day, intra-day re-optimisation recovered an additional $400,000 compared to the day-ahead plan --- more than half of the revenue left on the table by the stale plan. The capture ratio improved from 0.50 (at the benchmark bar) to 0.76. On calm days, the improvement is negligible. The value of re-optimisation is concentrated in precisely the days that matter most for annual revenue.
</div>

### Real-World Comparison

The Mannum BESS's actual dispatch on this day (observable from AEMO's public dispatch data) provides a third reference point. Real-world performance is affected by factors not in the simulation: FCAS co-optimisation commitments, AEMO's dispatch conformance requirements, network constraints, and the operator's risk appetite. Actual battery dispatch typically falls between the day-ahead simulation and the intra-day simulation, reflecting the operator's use of real-time information tempered by practical constraints.

---

## Exercises

### Exercise 1: Re-optimisation Value vs Volatility

**Task:** Compare intra-day re-optimisation revenue across a calm and volatile week. Simulate the Mannum BESS with:

1. **Strategy A** --- Day-ahead only (one plan per day, no updates).
2. **Strategy B** --- 30-minute intra-day re-optimisation via the emulator.

Run both on (a) a calm SA1 winter week (June, no spikes) and (b) a volatile summer week (January, multiple spikes above $5,000/MWh). Report weekly revenue, capture ratio, and improvement.

<details><summary><strong>Worked solution</strong></summary>

**Calm week (June 2024):** Prices $20--$120/MWh, no spikes above $300. Perfect-foresight revenue: $85,000.

| Strategy | Weekly revenue | Capture ratio |
|----------|---------------|--------------|
| Day-ahead only | $52,000 | 0.61 |
| Intra-day re-optimisation | $55,000 | 0.65 |
| Improvement | $3,000 (5.8%) | +0.04 |

Marginal gains only --- the day-ahead plan is already near-optimal when prices are smooth.

**Volatile week (January 2024):** Spikes above $5,000/MWh on 3 days, above $1,000 on 5 days. Perfect-foresight revenue: $4,200,000.

| Strategy | Weekly revenue | Capture ratio |
|----------|---------------|--------------|
| Day-ahead only | $2,100,000 | 0.50 |
| Intra-day re-optimisation | $3,020,000 | 0.72 |
| Improvement | $920,000 (43.8%) | +0.22 |

<pre>
Re-optimisation benefit:
  Calm week   →  +$3,000   (+5.8%)
  Volatile week → +$920,000 (+43.8%)
</pre>

**Re-optimisation value scales super-linearly with volatility.** This motivates event-triggered re-optimisation: re-optimise frequently when volatility indicators are elevated, conserve resources during calm periods.

</details>

### Exercise 2: Drawdown Limit Design

**Task:** Implement a daily drawdown limit of $150,000 for the Mannum BESS on the volatile January week from Exercise 1. Drawdown rule: if daily cumulative P&L falls below --$150,000, halt trading for the remainder of the day.

Report:

1. Total weekly revenue with and without the limit.
2. Worst single-day P&L with and without the limit.
3. Number of days the limit was triggered.

<details><summary><strong>Worked solution</strong></summary>

**Without drawdown limit:**

| Day | P&L (no limit) |
|-----|---------------|
| Monday | +$680,000 |
| Tuesday | +$420,000 |
| Wednesday | --$280,000 |
| Thursday | +$1,050,000 |
| Friday | +$390,000 |
| Saturday | +$510,000 |
| Sunday | +$250,000 |
| **Total** | **$3,020,000** |

Wednesday's loss: unexpected price spike to $4,000/MWh during charging (transmission constraint).

**With $150,000 drawdown limit:**

Limit triggered Wednesday at ~11:00 AM. Avoids $130,000 of additional afternoon losses but also misses a $350,000 late-afternoon recovery.

| Day | P&L (with limit) | Limit triggered? |
|-----|-----------------|-----------------|
| Monday | +$680,000 | No |
| Tuesday | +$420,000 | No |
| Wednesday | --$150,000 | Yes (11:00 AM) |
| Thursday | +$1,050,000 | No |
| Friday | +$390,000 | No |
| Saturday | +$510,000 | No |
| Sunday | +$250,000 | No |
| **Total** | **$3,150,000** |

**Comparison:**

| Metric | No limit | $150K limit |
|--------|---------|-------------|
| Weekly revenue | $3,020,000 | $3,150,000 |
| Worst-day P&L | --$280,000 | --$150,000 |
| Days limit triggered | N/A | 1 |

<pre>
Net effect: +$130,000 weekly revenue, worst-day loss capped at --$150,000.
</pre>

**The drawdown limit's purpose is to bound downside, not maximise revenue.** In this case it happened to improve both. In other scenarios it may reduce total revenue by halting before a recovery --- the trade-off is tail-risk reduction.

</details>

<div class="key-point">
<strong>Drawdown limits are insurance, not optimisation.</strong> A well-set drawdown limit slightly reduces average revenue (by occasionally halting profitable trading) but substantially reduces tail risk (by capping worst-case losses). The appropriate level depends on the operator's risk appetite, the battery's annual revenue target, and the consequences of large losses (e.g., covenant breaches on project finance). For the Mannum BESS, a daily drawdown limit in the range of $150,000--$300,000 balances downside protection against opportunity cost.
</div>

---

## Glossary

| Term | Definition |
|------|-----------|
| **Intra-day re-optimisation** | Updating dispatch plan multiple times per day |
| **Staleness** | Plan no longer reflects current info |
| **Gate closure** | Bid deadline (~2 min before dispatch) |
| **Model drift** | Performance degrades as data distribution shifts |
| **Retraining cadence** | Schedule for retraining (monthly/quarterly) |
| **Drawdown limit** | Max loss before automated trading halts |
| **Human-in-the-loop** | Human authority over high-stakes decisions |
| **Capture-ratio drift** | Sustained decline in rolling capture ratio |
| **Decision-regret drift** | Sustained increase in actual-vs-optimal gap |
| **AEMO dispatch cycle** | 5-min interval: dispatch + price setting |
| **PD forecast** | AEMO price/demand fcst; 30-min updates, 40 h |
| **5MPD** | AEMO short-term fcst; 5-min updates, 1 h |
| **Price impact** | Battery's own effect on market price |
| **Recalibration** | Update calibration layer without retraining |

## Summary

Live operations convert offline models into real revenue, and the gap between backtesting and production is an engineering and risk-management challenge, not a modelling one. Intra-day re-optimisation --- re-solving the dispatch plan as new pre-dispatch and 5-minute forecasts arrive, now through the Chapter 12 market emulator rather than a simple LP --- captures revenue that a stale day-ahead plan leaves on the table, with the benefit concentrated on the volatile days that dominate annual earnings. The operational loop runs on the NEM's 5-minute clock: ingest data, update the forecast, run the emulator, submit a bid before gate closure, record the outcome. Data latency is the binding constraint, not computation. Monitoring spans four dimensions --- forecast drift, calibration drift, data-quality faults, and decision-regret and capture-ratio drift --- with recalibration as the first response and full retraining reserved for confirmed structural shifts. Risk controls (drawdown limits, exposure caps, minimum state of charge) bound the downside, and a human trader retains authority over the 5% of situations that exceed the model's training distribution. For the Mannum BESS, the worked example showed that 30-minute intra-day re-optimisation lifted the capture ratio from 0.50 to 0.76 on a volatile heatwave day, recovering $400,000 of revenue that the day-ahead plan forfeited by discharging before the largest price spike. The 0.50 capture ratio benchmark is achievable with a day-ahead plan alone; intra-day operations are what push beyond it.
