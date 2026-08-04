# 10. Capstone: From Sunlight to Revenue

## The Complete Pipeline

This final chapter assembles every piece of the course into a single end-to-end pipeline and evaluates it as a whole. The pipeline is anchored on the **Mannum BESS** — a 100 MW / 200 MWh lithium iron phosphate battery in SA1, owned by Epic Energy and optimised by Habitat Energy. By targeting a real asset, the entire course functions as a **digital twin**: the same region, the same battery specifications, and (where Mannum's short operational history allows) a direct comparison against Habitat's actual dispatch decisions.

The story we have built, chapter by chapter, is this:

**Sunlight hits a solar panel** (Chapter 4) → **irradiance is measured and processed into weather features** (Chapter 4) → **features are combined with price history, demand, and calendar variables** (Chapters 1–4) → **a price model forecasts tomorrow's prices** (Chapters 6–7) → **the forecast is calibrated** (Chapter 8) → **an LP optimises battery dispatch** (Chapters 5, 9) → **the battery charges and discharges at actual prices** → **revenue is earned**.

Every link in this chain introduces error, and the errors **compound**. The pipeline perspective asks the most commercially important question: **which link is the weakest? Where should the next unit of effort be invested?**

![End-to-end pipeline](figures/10_pipeline.png)

<p class="figure-caption">Figure 10.1 — The complete pipeline from weather data to battery revenue. Each stage transforms inputs into outputs, and each transformation introduces error. The pipeline perspective reveals where errors are largest and where improvements yield the most economic value.</p>

### Pipeline Components and Their Error Sources

| Stage | Input | Output | Primary error source |
|-------|-------|--------|---------------------|
| **Weather data** | ERA5 reanalysis | Clear-sky index, wind speed, temperature | Reanalysis is a model, not measurement; spatial averaging over grid cells |
| **Feature engineering** | Raw variables | Feature matrix | Feature selection choices; transform assumptions; potential leakage |
| **Price model** | Features | Quantile forecasts | Model misspecification; training data limitations; regime changes |
| **Calibration** | Raw quantiles | Calibrated quantiles | Calibration window choice; non-stationarity; tail estimation |
| **Dispatch LP** | Price forecast + battery specs | Charge/discharge schedule | Forecast error propagation; constraint approximations |
| **Execution** | Schedule + actual prices | Revenue | Actual prices differ from forecast; market frictions; latency |

<div class="definition-box">
<strong>Pipeline:</strong> A sequence of processing stages where each stage's output becomes the next stage's input. In machine learning, a pipeline typically includes data ingestion, feature engineering, model training, prediction, and post-processing. Each stage can introduce errors that propagate forward, potentially amplifying through subsequent stages. Evaluating a pipeline requires end-to-end assessment — optimising individual stages in isolation can produce a suboptimal whole.
</div>

<div class="key-point">
<strong>The pipeline principle:</strong> Optimising each stage independently does not guarantee an optimal pipeline. A model that produces the lowest MAE might not produce the best dispatch revenue, because the dispatch LP cares about price ranking more than magnitude accuracy. The pipeline must be evaluated end-to-end — from weather data to capture ratio — not stage by stage.
</div>

---

## Grey-Box Models

### The Philosophy: Combining Physics and Data

Throughout this course, we have used two approaches to forecasting:

- **White-box models** (also called physics-based or mechanistic models): Built from first principles about how the system works. For electricity prices, the merit-order dispatch process is a white-box model — it describes *why\* prices are what they are based on the physical relationship between supply, demand, and generator costs.

- **Black-box models** (also called data-driven or statistical models): Learn input-output relationships from historical data without incorporating domain knowledge. GBT and neural networks are black-box models — they find patterns in the data without "knowing" anything about how electricity markets work.

<div class="definition-box">
<strong>White-box model:</strong> A model built from explicit understanding of the underlying physical or economic process. The model's structure reflects the known mechanism — for electricity prices, the supply-demand balance and merit-order dispatch. White-box models are interpretable and extrapolate sensibly to unseen conditions, but they may miss patterns not captured by the assumed mechanism.
</div>

<div class="definition-box">
<strong>Black-box model:</strong> A model that learns relationships directly from data without incorporating explicit domain knowledge. The internal workings are opaque — we can see what goes in and what comes out, but the intermediate reasoning is not interpretable in physical terms. Black-box models are flexible and can capture unexpected patterns, but they may extrapolate poorly to conditions not seen in training data and provide limited physical insight.
</div>

A **grey-box model** combines both: it uses physical structure where the physics is well-understood, and data-driven learning where the physics is insufficient.

<div class="definition-box">
<strong>Grey-box model:</strong> A model that combines physical structure (white-box component) with data-driven learning (black-box component). The physical component encodes known relationships (e.g., higher net load → higher prices), while the data-driven component learns the residual — everything the physics doesn't explain. Grey-box models combine the interpretability and extrapolation properties of white-box models with the flexibility of black-box models.
</div>

For electricity prices, the natural grey-box structure is:

<div class="equation">

price = f(net_load) + g(features) + ε

</div>

where:
- **f(net_load)** is the **white-box component** — the merit-order curve mapping net load to price. This captures the dominant, physically motivated relationship.
- **g(features)** is the **black-box component** — a GBT model that captures everything else: time-of-day effects, autoregressive patterns, weather nuances, market dynamics.
- **ε** is the irreducible noise — the randomness that no model can predict (generator trips, sudden demand changes, strategic bidding).

![Grey-box model structure](figures/10_greybox.png)

<p class="figure-caption">Figure 10.2 — The grey-box model architecture. The white-box stage (isotonic regression on net load) captures the dominant merit-order relationship. The black-box stage (GBT on the residual) captures secondary patterns that the physics does not explain. Together, they outperform either stage alone.</p>

### The Merit-Order Stage: Isotonic Regression

The white-box component estimates the **merit-order curve** — the relationship between net load and price — from historical data. We use **isotonic regression** for this, because it naturally enforces the one physical constraint we are certain of: **monotonicity**.

<div class="definition-box">
<strong>Merit-order curve:</strong> The empirical relationship between net load (demand minus renewable generation) and the electricity price. As net load increases, more expensive generators are dispatched and the price rises. The merit-order curve is approximately flat at low net load (cheap baseload sets the price) and steep at high net load (expensive peaking generators set the price). This is the hockey-stick shape from Chapter 3.
</div>

<div class="definition-box">
<strong>Isotonic regression:</strong> A non-parametric regression method that finds the best-fitting function constrained to be <strong>monotonically non-decreasing</strong> (or non-increasing). Given data points (x_1, y_1), ..., (x_n, y_n), isotonic regression finds the non-decreasing step function that minimises the sum of squared errors. It produces a piecewise-constant fit that preserves the ordering — if x_1 < x_2, the fitted value at x_1 is no larger than the fitted value at x_2.
</div>

The procedure:

1. **Sort** all historical observations by net load (ascending)
2. **Fit isotonic regression:** Find the non-decreasing function f(net_load) that minimises the sum of squared errors between f(net_load) and the actual price
3. The result is a **step function** that approximates the supply stack — a staircase where each step corresponds to a different generation technology's marginal cost

Why isotonic regression instead of ordinary least squares regression or a polynomial fit?

- **Monotonicity is guaranteed.** The merit-order effect guarantees that higher net load leads to higher or equal prices (in expectation). Linear regression might fit a line with a negative slope in some regions (due to noise), and polynomial regression might produce non-monotonic oscillations. Isotonic regression cannot violate monotonicity by construction.

- **No functional form assumed.** We don't need to guess whether the relationship is linear, quadratic, exponential, or some other shape. Isotonic regression discovers the shape from the data — it could be a gentle curve, a sharp hockey stick, or a staircase. The algorithm adapts to whatever the data shows.

- **Robustness to outliers.** The step-function nature means that a few extreme prices at high net load do not distort the fit at low net load. Each segment of the curve is estimated from its own observations.

<div class="example-box">
<strong>Real-world example — the merit-order curve in SA1:</strong> The isotonic regression for SA1 might reveal:
- Net load 500–1,500 MW: price ≈ $25–$40/MWh (wind and gas baseload)
- Net load 1,500–2,500 MW: price ≈ $40–$80/MWh (more gas capacity needed)
- Net load 2,500–3,000 MW: price ≈ $80–$200/MWh (expensive peakers dispatched)
- Net load > 3,000 MW: price ≈ $200–$5,000+/MWh (scarcity pricing; all available capacity dispatched)

This curve encodes the physical structure of the SA1 generation fleet in a single function. As the fleet evolves (coal retires, new renewables enter), the curve shifts — but its monotonic shape is preserved.
</div>

### The Residual Model

After subtracting the merit-order prediction f(net_load) from the actual prices, the **residual** captures everything the physics does not explain:

<div class="equation">

residual_t = price_t − f(net_load_t)

</div>

This residual contains:

- **Time-of-day effects** beyond what net load captures — for example, generator bidding patterns differ between morning ramp-up and evening ramp-down, even at the same net load level
- **Autoregressive patterns** — yesterday's price residual predicts today's residual, because generator bidding strategies persist
- **Weather effects** not captured by net load — for instance, extreme temperature drives demand beyond what the net-load variable reflects (due to forecasting error in the demand model)
- **Market dynamics** — interconnector constraints, generator outages, strategic rebidding, and other factors that cause prices to deviate from the merit-order prediction

A GBT model trained on the residual has a **much easier job** than a GBT model trained on raw prices. The dominant nonlinearity (the hockey-stick merit-order relationship) has already been removed by the isotonic regression. The residual is more symmetric, less heavy-tailed, and more stationary than raw prices — all properties that make it easier for a machine learning model to learn accurately.

### Why Grey-Box Outperforms Either Alone

The grey-box approach has four key advantages over a pure black-box (GBT on raw prices) or pure white-box (merit-order only) model:

**1. Physical interpretability.** The merit-order stage provides a direct connection to market structure. You can inspect the fitted curve and see how the generation fleet's cost structure translates to prices. This is invaluable for understanding *why\* the model makes certain predictions and for detecting anomalies.

**2. Better extrapolation.** If net load reaches a level never seen in the training data — an unprecedented demand event or a massive renewable outage — the isotonic regression extrapolates the monotonic relationship sensibly (prices are at least as high as for the highest observed net load). A pure black-box model would extrapolate unpredictably, potentially producing nonsensical predictions.

<div class="definition-box">
<strong>Extrapolation:</strong> Making predictions for input values outside the range observed in the training data. All models must extrapolate when faced with unprecedented conditions. Models with physical structure (white-box and grey-box) tend to extrapolate sensibly because the physical relationships (monotonicity, conservation laws) hold beyond the training range. Black-box models have no such guarantee — their predictions outside the training range are essentially arbitrary.
</div>

**3. Feature efficiency.** The merit-order stage captures a large fraction of price variance with a single feature (net load). This means the residual model can focus its capacity on secondary effects — patterns that are subtler and harder to learn but still economically valuable.

**4. Structural stability.** The monotonic net-load-to-price relationship is a fundamental property of merit-order markets that persists even as the generation mix changes. Coal retires, renewables grow, and the curve shifts — but the monotonicity is preserved. A grey-box model built on this structural insight is more robust to market evolution than a pure data-driven model that must re-learn the basic supply-demand relationship every time the market changes.

---

## Ablation Studies

### Why Remove Components?

An **ablation study** is a systematic method for understanding which components of a model contribute to its performance. The term comes from neuroscience, where researchers "ablate" (remove) brain regions to understand their function. In machine learning, we remove features or model components and measure the impact on performance.

<div class="definition-box">
<strong>Ablation study:</strong> A systematic evaluation method where individual components of a model are removed one at a time, and the model's performance is measured after each removal. The component whose removal causes the largest performance degradation is the most important. Ablation studies reveal the marginal contribution of each component and identify which components are essential versus dispensable.
</div>

For a price forecasting model with features grouped into categories {price lags, calendar features, demand features, weather features}, an ablation study proceeds as follows:

1. **Full model** — train with all feature groups → baseline MAE
2. **Remove price lags** — train without any lagged price features → MAE increases by X%
3. **Remove calendar features** — train without hour, day-of-week, month → MAE increases by Y%
4. **Remove demand features** — train without demand/net load → MAE increases by Z%
5. **Remove weather features** — train without temperature, wind, irradiance → MAE increases by W%

The feature group whose removal causes the largest MAE increase is the most important — the model depends on it most heavily.

### Interpreting Ablation Results

Typical ablation results for NEM day-ahead price forecasting reveal a clear hierarchy:

| Removed feature group | MAE increase | CRPS increase | Interpretation |
|----------------------|-------------|---------------|----------------|
| **Price lags** | +30–50% | +25–40% | Dominant feature group; autoregressive information is critical |
| **Calendar features** | +10–20% | +8–15% | Diurnal and weekly patterns are significant secondary information |
| **Demand features** | +5–15% | +5–12% | Demand adds moderate but meaningful information |
| **Weather features** | +2–10% | +2–8% | Weather contributes but less than lags or calendar |
| **All except price lags** | +15–25% | +12–20% | Price lags alone capture most of the predictive power |

<div class="key-point">
<strong>The dominance of price lags:</strong> The most striking finding is that lagged prices (especially lag-48, yesterday's price at the same half-hour) dominate all other features. This is because the NEM is a <em>persistent</em> market — today's prices are strongly correlated with yesterday's prices. The persistence arises because the fundamental drivers (generator fleet composition, weather patterns, demand levels) change slowly relative to the 30-minute trading interval.
</div>

The relatively small contribution of weather features may seem surprising given Chapter 4's extensive discussion of weather's role in price formation. The explanation is subtle: **weather affects prices primarily through its effect on demand and renewable generation**, which are already partially captured by the demand feature. The residual contribution of weather features — beyond what demand already captures — is the ability to forecast *tomorrow's* conditions when they differ from today's.

<div class="example-box">
<strong>Real-world example — when weather matters most:</strong> Weather features add the most value during weather regime transitions. If today is mild (25°C) and tomorrow will be extremely hot (42°C), the temperature forecast provides critical information that lagged prices and today's demand cannot capture. Without weather features, the model would extrapolate from today's mild conditions; with weather features, it anticipates the demand surge and price spike. The ablation shows that this regime-transition effect is worth 2–10% MAE improvement — small on average but potentially very large on the specific days when it matters most (which happen to be the highest-revenue days for the battery).
</div>

### Ablation vs Feature Importance

Ablation studies and GBT feature importance (Chapter 7) measure related but distinct things:

<div class="definition-box">
<strong>Marginal contribution:</strong> The change in model performance when a component is added or removed. Ablation measures the marginal contribution of removal — how much worse the model gets without the feature. Feature importance (gain-based) measures the marginal contribution within the model — how much the feature helps the model as trained. These can differ because of <strong>redundancy</strong>: if two features contain similar information, removing one has little impact (low ablation) because the other compensates, but both show high importance within the model because the algorithm uses both.
</div>

A feature might have **high importance but low ablation impact** if other features contain similar information (redundancy). For example, "demand" and "net load" are correlated — removing one has little impact because the other compensates. But within the model, both show high importance because the algorithm splits on whichever is most convenient at each tree node.

Conversely, a feature might have **moderate importance but high ablation impact** if it provides unique information that nothing else captures. A weather feature might be used in relatively few tree splits (moderate importance) but removing it completely eliminates the model's ability to anticipate weather-driven price changes (high ablation impact).

---

## Fair Model Comparison

### The Problem of Unfair Comparisons

Comparing forecasting models is fraught with methodological pitfalls. A careless comparison can make a poor model look excellent or an excellent model look poor. Published forecasting results — even in peer-reviewed journals — frequently suffer from comparison issues that undermine their conclusions.

<div class="definition-box">
<strong>Fair comparison:</strong> A comparison where all models are evaluated under identical conditions — same data, same time period, same evaluation metric, same information available at forecast time. Any difference in measured performance can then be attributed to the models themselves, not to differences in experimental setup. Achieving fair comparison requires careful attention to data splitting, feature information, and evaluation methodology.
</div>

### Principles for Fair Comparison

**1. Same data splits.** All models must use exactly the same training and test sets. This seems obvious but is frequently violated in subtle ways: if Model A uses 3 years of training data and Model B uses 5 years, the comparison is confounded by the data size. Similarly, if Model A's hyperparameters were tuned using data that Model B's test set includes, the comparison is biased.

**2. Same evaluation period.** Results on different test periods are simply not comparable. A model tested during the volatile summer of 2019 (with multiple heatwave-driven spikes) will have a much higher MAE than the same model tested during the mild autumn of 2020, regardless of its quality. Always compare models on the same test window.

**3. Rolling-origin evaluation.** A single train/test split gives one performance estimate — a single number that might be unrepresentative of typical performance. Rolling-origin evaluation (Chapter 5) produces a distribution of performance estimates across multiple test windows, enabling robust comparison and statistical testing.

<div class="definition-box">
<strong>Rolling-origin evaluation:</strong> An evaluation method where the model is trained on a window of historical data, tested on the next period, then the window rolls forward and the process repeats. This produces multiple performance estimates (one per test window), providing a distribution rather than a single number. Rolling-origin evaluation simulates how the model would perform in production, where it is continuously retrained and evaluated on new data.
</div>

**4. Statistical testing.** The **Diebold-Mariano test** (introduced in Chapter 6) determines whether the performance difference between two models is **statistically significant** — unlikely to have arisen by chance — or merely random variation. A model that outperforms another by 2% MAE over a single test period might be genuinely better, or it might have gotten lucky. The DM test quantifies this uncertainty.

<div class="definition-box">
<strong>Diebold-Mariano (DM) test:</strong> A statistical test for comparing the predictive accuracy of two forecasting models. The null hypothesis is that both models have equal expected loss (e.g., equal MAE). The test accounts for serial correlation in the forecast errors, which is common in time series data. If the test rejects the null hypothesis (p-value < 0.05), we conclude that one model is significantly better than the other. If it does not reject, the observed difference could plausibly be due to random variation.
</div>

**5. Multiple metrics.** No single metric tells the whole story. Report at least:

- **MAE** — point forecast accuracy (how close the median prediction is to reality)
- **CRPS** — probabilistic forecast accuracy (how well the full distribution matches reality)
- **Capture ratio** — economic value (how much money the forecast-driven dispatch actually earns)

A model might win on MAE but lose on capture ratio if it is accurate on average but poor at predicting spikes — exactly the events that drive battery revenue. Conversely, a model with higher MAE might achieve better capture ratio if its spike predictions are more accurate.

### The Naive Baseline as Anchor

The **naive baseline** (similar-day forecast from Chapter 5) serves as the anchor for all comparisons. Results should always be reported as improvements relative to the naive, not as absolute numbers.

<div class="definition-box">
<strong>Forecast skill:</strong> A model's performance measured relative to a reference baseline, typically the naive forecast. Skill is computed as: skill = 1 − (model error / naive error). A skill of 0 means the model is no better than the naive; a skill of 0.3 means the model reduces the error by 30% compared to the naive. Reporting skill rather than absolute error controls for the difficulty of the forecasting problem.
</div>

Why relative reporting matters: a MAE of $25 is excellent if the naive achieves $50 (skill = 0.50) but mediocre if the naive achieves $28 (skill = 0.11). The absolute number alone tells you nothing about the forecasting problem's difficulty. Relative reporting lets you compare across regions (SA1 is harder than TAS1), across time periods (summer is harder than autumn), and across studies.

---

## The Value Chain: Where to Invest

### Marginal Value of Each Pipeline Stage

Not all improvements are equally valuable. A **back-of-the-envelope calculation** reveals where each dollar of effort yields the most revenue:

<div class="definition-box">
<strong>Marginal value:</strong> The additional value generated by one additional unit of investment or improvement. In the pipeline context, the marginal value of improving the price model by 1% MAE is the resulting increase in annual battery revenue. Marginal value varies across pipeline stages — the same investment might yield $200K in additional revenue if directed at the dispatch stage but only $50K if directed at the weather data stage.
</div>

| Improvement | MAE reduction | CR improvement | Annual value (100MW/2h, SA1) |
|-------------|-------------|----------------|------------------------------|
| Better weather data (ERA5 → NWP ensemble) | 2–5% | 1–3% | $50K–$200K |
| Better price model (LEAR → tuned GBT) | 5–15% | 3–8% | $200K–$500K |
| Model combination (GBT → QRA ensemble) | 3–8% | 2–5% | $100K–$300K |
| Better calibration (raw → conformal) | 0–3% | 1–5% | $50K–$300K |
| Better dispatch (naive MPC → scenario MPC) | 0% (same forecast) | 2–5% | $100K–$300K |
| Better battery (85% → 90% RTE) | N/A | 3–7% | $200K–$400K |

<div class="key-point">
<strong>The dispatch stage often offers the highest marginal returns.</strong> The largest revenue improvements often come from using the forecast better (the dispatch stage) rather than making the forecast more accurate. This is the concavity of the value-of-information curve (Chapter 9) in action: once the forecast is reasonably good, better <em>use</em> of the forecast matters more than a better forecast itself.
</div>

### The 80/20 Rule in Practice

<div class="definition-box">
<strong>80/20 rule (Pareto principle):</strong> A widely observed pattern where approximately 80% of the outcome is produced by 20% of the effort. In forecasting, a simple model with a few key features often captures most of the predictive value, while the remaining accuracy requires disproportionate additional complexity. This does not mean the last 20% is not worth pursuing — just that the cost per unit of improvement increases dramatically.
</div>

For a practical battery dispatch deployment, the effort-to-value breakdown follows the Pareto principle strikingly closely:

**The first 80% of value** comes from:
- Lag-48 and lag-336 price features
- Basic calendar features (hour of day, day of week)
- A well-tuned LightGBM model with early stopping
- A simple LP dispatch with point forecasts (naive MPC)
- This can be implemented in a few hundred lines of well-structured code

**The next 15% of value** comes from:
- QRA combination of GBT + LEAR forecasts
- Conformal calibration of the quantile forecasts
- Chance-constrained or scenario MPC dispatch
- Demand and weather features
- This requires a few thousand lines of code plus ongoing data pipeline management

**The last 5% of value** comes from:
- Neural network models or hybrid architectures
- Sophisticated scenario generation (copula-based, historical analogue)
- Adaptive conformal prediction with regime detection
- Continuous online retraining with concept drift detection
- Grey-box modelling with isotonic merit-order estimation
- This requires a dedicated team, significant infrastructure, and ongoing research

<div class="example-box">
<strong>Real-world industry perspective:</strong> For most battery operators, reaching the 80% level is the clear priority — it provides the bulk of the economic value with manageable engineering effort. The next 15% is worthwhile for operators with in-house data science teams and a competitive edge mandate. The last 5% is the domain of large energy trading firms and research organisations for whom every marginal percentage point of capture ratio translates to millions of dollars across their fleet of batteries.

The key strategic question is: given your battery's size, location, and competitive environment, where does the cost of the next improvement exceed its value? For a single 100 MW battery, the answer might be the 95th percentile. For a portfolio of twenty 100 MW batteries, the answer might be the 99th percentile — because the same modelling investment is amortised across a much larger revenue base.
</div>

---

## Limitations and Honest Assessment

### What This Project Does Not Address

No model is complete, and intellectual honesty requires acknowledging the simplifications we have made. Each of the following limitations represents a gap between our simplified project and a production-grade battery dispatch system:

**1. Intra-day re-optimisation.** Our MPC re-optimises once per day (or once per trading period). A production battery re-optimises every 5–30 minutes as new information arrives: updated weather forecasts, real-time demand data, generator outage announcements, and AEMO's pre-dispatch updates. The more frequently the dispatch is re-optimised, the better it can respond to unfolding events — particularly during volatile periods when prices change rapidly.

<div class="definition-box">
<strong>Intra-day re-optimisation:</strong> The process of updating the battery's dispatch schedule multiple times within a day as new information becomes available. Each re-optimisation incorporates the latest data — updated weather forecasts, actual demand observations, generator status changes — producing a schedule that reflects current conditions rather than the previous day's forecast. This is particularly valuable during volatile periods when the day-ahead forecast has become stale.
</div>

**2. FCAS co-optimisation.** Batteries earn significant revenue from Frequency Control Ancillary Services (Chapter 9 glossary), but our project optimises only energy arbitrage. A production system would **co-optimise** energy and FCAS simultaneously — determining for each half-hour whether the battery should discharge for energy revenue, provide FCAS (which requires reserving capacity), or do both (partial dispatch). The co-optimisation is more complex (it involves multiple interacting markets) but can increase total revenue by 20–40%.

**3. Degradation modelling.** We use a simple cycle-count constraint (maximum cycles per day). Real battery degradation is far more nuanced: it depends on the **depth of discharge** (shallow cycles are less damaging than deep cycles), the **charge rate** (fast charging degrades faster), **temperature** (heat accelerates chemical degradation), and **calendar aging** (batteries degrade even when idle). Sophisticated dispatch models incorporate degradation as a variable cost that depends on the specific charge/discharge profile, not just the total number of cycles.

**4. Market impact.** Our LP assumes the battery is a **price-taker** — its actions do not affect market prices. For a small battery, this is reasonable. For a large battery (100+ MW in a region like SA1 with 2,000–3,000 MW of total demand), the battery's charging adds to demand (raising prices) and its discharging adds to supply (lowering prices). A production system must account for this **market impact** — the degree to which the battery's own actions change the prices it faces.

<div class="definition-box">
<strong>Price-taker assumption:</strong> The assumption that a market participant's actions do not affect market prices. A small battery in a large market is approximately a price-taker — its 10 MW of charging has negligible impact on a 10,000 MW market. A large battery may violate this assumption: a 200 MW battery in SA1 (total demand ~2,500 MW) represents 8% of the market, and its charging/discharging could meaningfully move prices.
</div>

**5. Network constraints.** We ignore transmission losses, network charges, and locational constraints. In reality, the battery faces a net price that differs from the regional reference price due to **marginal loss factors** (MLFs), network use charges, and potential congestion at its connection point. These can reduce effective revenue by 5–15%.

**6. Operational constraints.** Real batteries have physical limitations not captured in our simplified LP: minimum up/down times (the battery cannot switch between charging and discharging instantaneously), ramp rate limits (how fast the battery can increase or decrease its power output), auxiliary power consumption (the battery's own cooling and control systems consume electricity), and state-of-health degradation that reduces capacity over time.

### Sources of Forecast Error

The gap between our capture ratio and 1.0 (perfect foresight) comes from fundamental and practical limitations:

**1. Price spikes are inherently unpredictable.** Some spikes are caused by events that no forecast can anticipate: generator mechanical failures (a 500 MW coal unit trips without warning), sudden demand surges (a temperature forecast error of 3°C can shift demand by 1,000 MW), or transmission line faults. These events set a **fundamental limit** on the achievable capture ratio — even the best possible forecast cannot predict mechanical failures.

**2. The forecast horizon is long.** Forecasting 24 hours ahead is much harder than forecasting 1 hour ahead. Forecast errors grow roughly with the square root of the horizon — a 24-hour forecast has roughly 5× the error of a 1-hour forecast. For weather-dependent periods (afternoon solar, evening wind), the day-ahead weather forecast is the binding constraint.

**3. Regime changes.** The electricity market structure evolves continuously: generators retire, new capacity is built, demand patterns shift (driven by rooftop solar, electric vehicles, and changing industrial activity), and market rules change. Models trained on historical data cannot anticipate structural breaks — they can only adapt after the change has occurred and enough new data has accumulated.

<div class="definition-box">
<strong>Regime change (structural break):</strong> A fundamental change in the statistical properties of the data — for example, a new power plant coming online changes the merit-order curve, or a rule change alters bidding behaviour. Regime changes invalidate models trained on pre-change data because the learned relationships no longer hold. Detecting and adapting to regime changes is one of the hardest problems in time series forecasting.
</div>

**4. Strategic behaviour.** Generator bidding in the NEM is partly strategic — generators adjust their bids based on expected market conditions, competitor behaviour, and fuel contract positions. This strategic layer is inherently difficult to forecast because it involves **game-theoretic reasoning** — each generator's optimal bid depends on what it expects other generators to bid, creating a complex feedback loop that is fundamentally different from the physical drivers (weather, demand) that our models capture.

### What a Production System Would Add

A production-grade forecasting and dispatch system builds on the foundations of this course but adds several critical components:

| Component | What it adds | Why it matters |
|-----------|-------------|---------------|
| Real-time data feeds | Live SCADA, demand, weather, generator status | Enables intra-day re-optimisation; reduces forecast staleness |
| Ensemble of NWP models | Multiple weather forecast sources (BoM ACCESS, ECMWF, GFS) | More robust weather prediction; captures model uncertainty |
| Continuous retraining | Models retrained daily or weekly with latest data | Adapts to evolving market conditions and regime changes |
| Intra-day update loop | Forecasts updated every 30 minutes; dispatch re-optimised | Captures within-day information; responds to unfolding events |
| Risk management | Position limits, drawdown controls, exposure monitoring | Prevents catastrophic losses during extreme market events |
| FCAS co-optimisation | Joint energy and ancillary service scheduling | Increases total revenue by 20–40% |
| Human oversight | Traders review and potentially override automated decisions | Essential during unprecedented events (major outages, policy changes) |
| Monitoring and alerting | Automated detection of model degradation, data quality issues | Prevents silent failure; maintains system reliability |

---

## The Broader Context

### Energy Storage Economics

Battery energy storage is one of the fastest-growing segments of the global electricity sector. Understanding the economic landscape provides context for why the forecasting and dispatch techniques in this course matter.

<div class="definition-box">
<strong>Levelised cost of storage (LCOS):</strong> The per-MWh cost of storing and discharging electricity, accounting for the battery's capital cost, operating costs, degradation, and financing over its lifetime. LCOS is the storage equivalent of the more familiar LCOE (Levelised Cost of Energy) for generators. For a battery to be economically viable, its annual revenue must exceed its annualised LCOS multiplied by its energy throughput.
</div>

The economics of battery storage have transformed dramatically over the past decade:

- **Lithium-ion cell costs** have fallen from approximately $1,200/kWh in 2010 to roughly $140/kWh in 2024 — an 88% reduction in 14 years. This decline, driven primarily by electric vehicle manufacturing scale, has made grid-scale batteries economically viable in ways that seemed impossible a decade ago.

- **Installed capacity** is growing at 40–60% per year globally. Australia's NEM has seen particularly rapid growth, with several hundred megawatts of new battery capacity added annually.

- **Revenue opportunities** are growing as renewable penetration increases. More wind and solar means more price volatility (negative prices when renewables are abundant, spikes when they are scarce), which directly increases arbitrage revenue.

The economic viability of batteries depends on the full **revenue stack** (energy arbitrage + FCAS + network support + capacity payments). In most NEM regions, arbitrage alone is currently sufficient for marginal economics (5–8 year payback), with the other revenue streams providing additional upside. As renewable penetration increases and price volatility grows, the arbitrage component is expected to become increasingly dominant.

<div class="example-box">
<strong>Real-world example — the Hornsdale Power Reserve:</strong> The Hornsdale Power Reserve (HPR), a 150 MW / 194 MWh battery in South Australia, was the world's largest lithium-ion battery when commissioned in 2017. It demonstrated that grid-scale batteries could be commercially viable, earning substantial revenue from both FCAS and energy arbitrage. In its first year, HPR's FCAS revenue alone exceeded expectations, and its energy arbitrage revenue grew as operators learned to forecast and dispatch more effectively. The success of HPR catalysed a wave of battery investment in the NEM and globally.
</div>

### The Forecasting Industry

Electricity price forecasting is a substantial industry in its own right, employing quantitative analysts, data scientists, and energy market experts across several sectors:

- **Energy trading companies** employ teams of "quants" who develop proprietary forecasting models, competing to predict prices more accurately than their rivals. Better forecasts translate directly to higher trading profits.

- **Battery operators** (both utility-scale systems like HPR and smaller behind-the-meter batteries at commercial sites) use forecasts for dispatch optimisation. The quality of the dispatch — and hence the battery's revenue — depends directly on forecast quality.

- **Electricity retailers** use price forecasts for contract pricing (setting the fixed price they offer to customers) and hedging (buying financial contracts to reduce their exposure to spot price volatility).

- **System operators** (AEMO in Australia) produce their own forecasts (pre-dispatch) for reliability planning — ensuring sufficient generation is available to meet demand. AEMO's pre-dispatch forecast also serves as a public benchmark against which commercial forecasters can be compared.

The methods covered in this course — LEAR, GBT, QRA, conformal prediction — represent the current mainstream of both academic research and industry practice. More exotic approaches exist (deep reinforcement learning for dispatch, attention-based Transformer architectures, physics-informed neural networks, generative adversarial networks for scenario generation) but have not consistently demonstrated practical advantages over well-tuned GBT models with good features in rigorous head-to-head comparisons.

### What Makes This Problem Interesting

Electricity price forecasting sits at the intersection of multiple disciplines — and it is this interdisciplinary nature that makes it both challenging and rewarding:

- **Power systems engineering:** Understanding how the grid works, what drives prices, and how the merit-order dispatch process creates the statistical properties we observe
- **Statistics and econometrics:** Time series modelling, probabilistic forecasting, calibration theory, model evaluation methodology
- **Machine learning:** Feature engineering, gradient boosting, neural networks, ensemble methods, hyperparameter optimisation
- **Operations research:** Linear programming, stochastic optimisation, model predictive control, decision-making under uncertainty
- **Economics:** Market design, strategic bidding, risk management, investment analysis

No single discipline provides a complete answer. The best forecasters and battery operators combine physical intuition about the electricity system (from power engineering) with statistical rigour (from econometrics) and computational efficiency (from machine learning), applied through the lens of optimal decision-making (from operations research) in a market context (from economics).

<div class="key-point">
<strong>The ultimate takeaway:</strong> The value of electricity price forecasting lies not in the forecast itself but in the decisions it enables. A forecast that sits in a spreadsheet has zero value. A forecast that drives an optimised battery dispatch — charge during the cheap periods, discharge during the expensive ones, with risk management that accounts for forecast uncertainty — can generate millions of dollars of revenue per year. Every technique in this course, from data ingestion (Chapter 1) to conformal calibration (Chapter 8), serves this single practical goal: converting information about the future into money today.
</div>

---

## Glossary

| Term | Definition |
|------|-----------|
| **Pipeline** | A sequence of processing stages where each stage's output becomes the next stage's input |
| **White-box model** | A model built from explicit physical or economic principles |
| **Black-box model** | A data-driven model that learns relationships without incorporating domain knowledge |
| **Grey-box model** | A model combining physical structure with data-driven learning |
| **Isotonic regression** | A non-parametric regression constrained to produce monotonically non-decreasing predictions |
| **Merit-order curve** | The empirical relationship between net load and electricity price |
| **Extrapolation** | Making predictions for inputs outside the range observed in training |
| **Ablation study** | Systematic removal of model components to measure their individual contributions |
| **Marginal contribution** | The change in performance when a component is added or removed |
| **Fair comparison** | Evaluating models under identical conditions to ensure performance differences reflect model quality |
| **Rolling-origin evaluation** | Repeatedly training and testing across multiple time windows |
| **Diebold-Mariano test** | A statistical test for comparing forecasting model accuracy |
| **Forecast skill** | Performance relative to a naive baseline; controls for problem difficulty |
| **Marginal value** | The additional value from one additional unit of improvement |
| **80/20 rule (Pareto principle)** | ~80% of outcomes come from ~20% of effort or causes |
| **Intra-day re-optimisation** | Updating the dispatch schedule multiple times within a day |
| **FCAS co-optimisation** | Jointly optimising energy arbitrage and ancillary service provision |
| **Price-taker assumption** | The assumption that a participant's actions do not affect market prices |
| **Regime change (structural break)** | A fundamental change in the data's statistical properties |
| **LCOS (Levelised Cost of Storage)** | The per-MWh cost of storage over the battery's lifetime |
| **Revenue stack** | The combination of all revenue streams available to a battery |

## Summary

The capstone assembles every technique from the course into a single end-to-end pipeline: weather data flows through feature engineering and price modelling into calibrated probabilistic forecasts, which drive optimised battery dispatch that generates revenue from actual prices. Grey-box models — combining a physically motivated merit-order stage (isotonic regression on net load) with a data-driven residual stage (GBT) — outperform either pure physics or pure machine learning by leveraging the strengths of both. Ablation studies confirm that price lags dominate feature contributions, with weather features adding incremental but economically meaningful value during regime transitions. Fair model comparison requires identical data splits, rolling-origin evaluation, statistical testing, and multiple metrics (MAE, CRPS, capture ratio) to avoid misleading conclusions. The marginal value of pipeline improvements is highest at the dispatch stage — using the forecast well often matters more than making the forecast better — and follows the 80/20 rule: a simple GBT model with lag features and basic MPC captures the bulk of available revenue. A production system would add real-time data, continuous retraining, intra-day re-optimisation, FCAS co-optimisation, and human oversight. The capture ratio — the fraction of perfect-foresight revenue achieved — is the single metric that ties the entire pipeline together, converting forecast quality into economic value: for a 100 MW / 2h battery in SA1, each percentage point of capture ratio is worth roughly $80K–$150K per year in arbitrage revenue alone.
