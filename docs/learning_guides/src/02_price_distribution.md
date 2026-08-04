# 2. Price Distribution and Stylised Facts

## What Are "Stylised Facts"?

Before building any forecasting model, we need to understand the patterns that electricity prices consistently exhibit. In economics and finance, these recurring patterns are called **stylised facts** — statistical properties that appear reliably across different time periods and markets.

<div class="definition-box">
<strong>Stylised facts:</strong> Empirical regularities that are observed consistently across multiple datasets, time periods, or markets. They describe "what always seems to be true" about a system, without necessarily explaining why. A good forecasting model should be consistent with these facts.
</div>

Think of stylised facts as the "personality traits" of electricity prices. Just as a person might always be early to meetings and prefer coffee over tea, electricity prices have their own persistent habits: they follow daily rhythms, cluster their volatility, and occasionally explode into extreme spikes. Any model that contradicts these traits is probably wrong.

## The Daily Price Cycle: The Duck Curve

The most visible pattern in electricity prices is the **diurnal cycle** — prices follow a predictable daily rhythm driven by when people use electricity and when the sun shines.

![Duck curve](figures/02_duck_curve.png)

<p class="figure-caption">Figure 2.1 — Left: the characteristic "duck curve" daily price profile showing the overnight trough, midday solar dip, and evening spike. Right: demand, solar generation, and net load through the day.</p>

### Why does the duck curve happen?

The shape is driven by the interaction of three forces:

1. **Demand follows human activity.** People wake up, turn on lights and appliances (morning ramp). They go to work, return home, cook dinner, run heating/cooling (evening peak). They go to sleep (overnight trough).

2. **Solar generation peaks at midday.** When the sun is strongest, solar farms flood the grid with cheap electricity, pushing the price down — even to negative values on sunny days.

3. **The evening collision.** Just as solar generation drops rapidly at sunset, demand peaks as people return home. This creates a sharp supply shortage, forcing expensive gas peakers online and causing the price spike that defines the right side of the duck curve.

<div class="definition-box">
<strong>Duck curve:</strong> The characteristic shape of net electrical load (demand minus solar generation) through the day. It resembles a duck in profile: a belly dip during midday solar, a neck ramp in the late afternoon, and a head peak in the evening. First identified by California's grid operator (CAISO), it has become a defining feature of grids with high solar penetration.
</div>

<div class="definition-box">
<strong>Net load:</strong> Total electricity demand minus variable renewable generation (wind + solar). Net load represents the amount of electricity that must be supplied by dispatchable generators (coal, gas, hydro, batteries). When net load is high, prices are high; when net load is low or negative, prices are low or negative. It is the single most important driver of electricity prices.
</div>

<div class="example-box">
<strong>Real-world example:</strong> In South Australia on a sunny spring day, solar generation might peak at 2,000 MW around noon while demand is only 1,400 MW. Net load goes <em>negative</em> — there is more solar than the state needs. The price drops below zero. Then at 6pm, the sun sets, solar drops to zero, and demand stays high. Net load jumps from −600 MW to +1,400 MW in just two hours. The price rockets from −$20 to $200+. This daily drama is the duck curve in action.
</div>

## Heavy Tails: Extreme Events Are Not Rare

In a **normal (Gaussian) distribution**, extreme events are vanishingly rare. A value more than 3 standard deviations from the mean occurs only 0.3% of the time. But electricity prices do not follow a normal distribution — they have **heavy tails**, meaning extreme events occur far more often than a normal distribution would predict.

![Heavy tails](figures/02_heavy_tails.png)

<p class="figure-caption">Figure 2.2 — Left: a normal distribution with thin tails — extreme values are extremely rare. Right: electricity prices have heavy tails, with both extreme spikes (right) and negative prices (left) occurring much more frequently than a normal distribution would predict.</p>

<div class="definition-box">
<strong>Heavy tails (fat tails):</strong> A statistical property where extreme values occur much more frequently than a normal distribution predicts. If prices were normally distributed with mean $50 and standard deviation $30, a price above $200 would be a "five-sigma event" — expected roughly once every 3.5 million observations. In reality, NEM prices exceed $200 roughly once every 100–200 observations.
</div>

<div class="definition-box">
<strong>Kurtosis:</strong> A measure of how "heavy" the tails of a distribution are. A normal distribution has kurtosis of 3. Electricity prices typically have kurtosis of 30–100+ (called <strong>leptokurtic</strong>), meaning the tails are enormously heavier than normal. This is why standard statistical methods that assume normality fail badly for electricity prices.
</div>

### Why heavy tails matter

Heavy tails have profound consequences for modelling and trading:

1. **Mean and variance are unreliable.** A few extreme spikes dominate the average, making sample statistics unstable. The average price over a month can shift dramatically depending on whether a single $10,000 spike occurred.

2. **Normal-distribution-based risk measures fail.** Value-at-Risk calculated under a normality assumption will severely underestimate the risk of extreme price events.

3. **Model training is distorted.** Standard least-squares regression will be pulled toward fitting the extreme observations, distorting predictions for the 99% of "normal" prices. This is one reason we use the arcsinh transform (Chapter 1).

<div class="example-box">
<strong>Real-world example — the Tomago smelter:</strong> In January 2019, a heatwave caused NEM prices to spike above $14,000/MWh for several hours. The Tomago aluminium smelter in NSW — one of the world's largest electricity consumers — voluntarily reduced its load to avoid the extreme prices. The cost of running at full production during those few hours would have exceeded the profit from weeks of aluminium production. These tail events are not theoretical curiosities — they drive real business decisions worth millions of dollars.
</div>

## Autocorrelation: Today Looks Like Yesterday

Electricity prices exhibit strong **autocorrelation** — the price at any given time is strongly related to the price at the same time yesterday, the day before, and even the same time last week.

![Autocorrelation and volatility](figures/02_acf_volatility.png)

<p class="figure-caption">Figure 2.3 — Top left: today's prices closely track yesterday's. Top right: scatter plot showing the strong positive correlation at lag 48 (24 hours). Bottom left: the autocorrelation function (ACF) showing spikes at 24h, 48h, and 72h lags. Bottom right: volatility clustering — calm and wild periods tend to persist.</p>

<div class="definition-box">
<strong>Autocorrelation:</strong> The correlation of a time series with a lagged copy of itself. If today's 2pm price is strongly correlated with yesterday's 2pm price, we say there is strong autocorrelation at lag 48 (48 half-hours = 24 hours). Autocorrelation tells us how "predictable" a time series is from its own past values.
</div>

<div class="definition-box">
<strong>Autocorrelation function (ACF):</strong> A plot showing the autocorrelation at every possible lag. For electricity prices, the ACF shows peaks at lags of 48 (24 hours), 96 (48 hours), and 336 (one week) — revealing that prices repeat on daily and weekly cycles.
</div>

### What the ACF tells us

The ACF for NEM prices reveals several key patterns:

- **Lag 48 (24 hours):** The strongest peak. Today's 2pm price is strongly related to yesterday's 2pm price. This is the diurnal cycle — the daily rhythm of human activity repeats.
- **Lag 96 (48 hours):** Still significant but weaker. Prices two days ago are informative but less so than yesterday.
- **Lag 336 (1 week):** A smaller but real peak. The weekly cycle matters because weekdays have higher demand than weekends.
- **Slow decay:** Autocorrelation decreases with lag but never reaches zero quickly, suggesting long-memory effects (persistent regimes of high or low prices).

<div class="key-point">
<strong>Key insight:</strong> The lag-48 autocorrelation is the single most powerful forecasting feature in electricity prices. "What was the price at this time yesterday?" is the most useful piece of information for predicting the price right now. This is why price lags dominate feature importance in every model we build.
</div>

## Volatility Clustering: Calm Breeds Calm, Chaos Breeds Chaos

Electricity prices exhibit **volatility clustering** — periods of high volatility (wild price swings) tend to be followed by more high volatility, and calm periods tend to be followed by more calm periods.

<div class="definition-box">
<strong>Volatility:</strong> The degree of variation in a time series. High volatility means prices are swinging wildly; low volatility means prices are stable. In finance, volatility is usually measured as the standard deviation of returns over a recent window.
</div>

<div class="definition-box">
<strong>Volatility clustering:</strong> The empirical observation that large price changes tend to be followed by more large price changes (of either sign), and small changes tend to be followed by small changes. In technical terms, the absolute returns are autocorrelated even when the returns themselves are not. This is a hallmark of financial and commodity time series.
</div>

### Why does volatility cluster?

Several mechanisms drive volatility clustering in the NEM:

1. **Weather persistence.** A heatwave doesn't last one afternoon — it persists for days. During the heatwave, high demand keeps prices volatile. When the heatwave breaks, prices calm down.

2. **Generator outages.** When a large coal plant trips (unexpectedly shuts down), the supply shortage persists until the plant is repaired — days or weeks. The reduced supply margin keeps prices volatile throughout.

3. **Seasonal renewables.** Spring brings consistently windy, sunny conditions that keep net load low and prices calm. Summer heatwaves bring consistently high demand that keeps prices volatile.

## The Price-Duration Curve

A powerful way to visualise the price distribution is the **price-duration curve**, which sorts all prices from highest to lowest and plots them against the percentage of time each price level is exceeded.

![Price duration curve](figures/02_price_duration.png)

<p class="figure-caption">Figure 2.4 — The price-duration curve. The top 5% of hours (spike zone) contribute a disproportionate share of total wholesale cost. Batteries aim to discharge in the red zone and charge in the green zone.</p>

<div class="definition-box">
<strong>Price-duration curve:</strong> A plot where prices are sorted from highest to lowest and plotted against the fraction of time they are exceeded. It shows at a glance what percentage of the time prices are above any given level. For batteries, it reveals the "spread" — the gap between charge prices (bottom of the curve) and discharge prices (top).
</div>

The price-duration curve reveals a fundamental fact about electricity markets: **a tiny fraction of hours accounts for a huge share of total cost**. In the NEM, the most expensive 5% of hours can account for 30–50% of total annual wholesale cost. This concentration is what makes battery arbitrage possible — buy during the cheap 50% at the bottom, sell during the expensive 5% at the top.

<div class="example-box">
<strong>Real-world example — the Mannum BESS:</strong> The Mannum BESS — a 100 MW / 200 MWh lithium iron phosphate battery in SA1, owned by Epic Energy and optimised by Habitat Energy — might earn $10 million per year in perfect-foresight arbitrage. Of that $10M, roughly $6–7M comes from the most expensive 5% of intervals — perhaps 450 half-hours out of 17,520 in a year. Getting those 450 intervals right (knowing when they will occur in advance) is the core challenge of price forecasting for battery dispatch.
</div>

## Seasonality: Annual and Weekly Rhythms

Beyond the daily cycle, electricity prices show seasonal patterns at multiple time scales:

### Weekly seasonality

- **Weekdays vs weekends:** Demand is 10–20% lower on weekends because commercial and industrial loads drop. Prices follow.
- **Monday mornings:** The transition from weekend to weekday creates a demand ramp, often with higher prices.
- **Friday evenings:** The transition to weekend relaxes demand.

### Annual seasonality

- **Summer (Dec–Feb in Australia):** Extreme heat drives air conditioning load, creating the highest demand and the most price spikes. This is the most volatile season.
- **Winter (Jun–Aug):** Heating load creates a secondary demand peak, but generally less extreme than summer.
- **Spring/Autumn (Mar–May, Sep–Nov):** Mild weather reduces demand. High solar and wind penetration can push prices consistently low.

<div class="definition-box">
<strong>Seasonality:</strong> A regular, predictable pattern that repeats at a fixed frequency. Daily seasonality repeats every 24 hours (the duck curve). Weekly seasonality repeats every 7 days (weekday vs weekend). Annual seasonality repeats every 12 months (summer vs winter). Seasonal patterns can be removed ("deseasonalised") to reveal the underlying trend and random variation.
</div>

## Summary of Stylised Facts

| Stylised fact | What it means | Modelling implication |
|--------------|--------------|----------------------|
| Diurnal cycle | Prices follow a daily rhythm | Include hour-of-day features |
| Heavy tails | Extreme events far more common than normal | Use arcsinh transform; don't assume normality |
| Autocorrelation | Today resembles yesterday | Include lagged price features (especially lag 48) |
| Volatility clustering | Volatile periods persist | Include recent volatility as a feature |
| Seasonality | Weekly and annual patterns | Include day-of-week and month features |
| Mean reversion | Spikes are short-lived, prices return to a base | Don't extrapolate spikes; model reversion |

---

## Glossary

| Term | Definition |
|------|-----------|
| **Stylised facts** | Consistent empirical patterns in a dataset |
| **Diurnal cycle** | Daily repeating pattern in prices |
| **Duck curve** | Daily net load shape: midday dip, evening spike |
| **Net load** | Demand minus wind and solar generation |
| **Heavy tails** | Extreme values occur far more often than a normal distribution predicts |
| **Kurtosis** | Measure of tail heaviness; electricity prices have very high kurtosis |
| **Autocorrelation** | Correlation of a time series with its own past values |
| **ACF** | Autocorrelation function — autocorrelation plotted at every lag |
| **Volatility** | Degree of price variation over time |
| **Volatility clustering** | Tendency for volatile periods to persist |
| **Price-duration curve** | Prices sorted from high to low vs % of time exceeded |
| **Seasonality** | Regular patterns repeating at fixed intervals |
| **Mean reversion** | Tendency of extreme prices to return to a long-run average |

## Summary

Electricity prices exhibit a consistent set of stylised facts that any forecasting model must respect. The daily duck curve — with its midday solar dip and evening spike — is the dominant pattern. Heavy tails mean extreme events are not rare; they are a defining feature of the market. Strong autocorrelation at lag 48 (24 hours) makes yesterday's price the single best predictor of today's. Volatility clusters in time, driven by persistent weather and supply conditions. And the price-duration curve reveals that a tiny fraction of extreme-price hours drives a disproportionate share of market value — making accurate spike forecasting the key to battery profitability.
