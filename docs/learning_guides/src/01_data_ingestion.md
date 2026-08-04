# 1. Data Ingestion and the NEM

## What is the NEM?

Imagine a giant marketplace where electricity is bought and sold every five minutes, stretching over 5,000 kilometres along Australia's eastern coast. This is the **National Electricity Market (NEM)** — one of the world's longest interconnected power systems, serving roughly 10 million homes and businesses.

Unlike buying petrol (where the price changes once a day), electricity is traded in a **wholesale market** where the price changes every five minutes. These prices can swing wildly — from *negative* values (generators paying you to take their electricity) to over $17,000 per megawatt-hour in extreme events. Understanding this market is the foundation of everything in this course.

<div class="definition-box">
<strong>Wholesale electricity market:</strong> A centralised marketplace where generators (power plants, wind farms, solar farms) sell electricity to retailers and large consumers. The NEM is Australia's wholesale market for the eastern states.
</div>

<div class="definition-box">
<strong>Megawatt-hour (MWh):</strong> The standard unit of electrical energy in wholesale markets. One MWh powers roughly 300–500 average Australian homes for one hour. When we say the price is "$80/MWh", it means a generator earns $80 for producing enough electricity to run ~400 homes for an hour.
</div>

## The Five Regions

The NEM is not one big grid with one price — it is divided into **five pricing regions**, each corresponding roughly to a state. Each region has its own electricity price at any given moment, determined by the generators operating in that region.

![NEM regions](figures/01_nem_regions.png)

<p class="figure-caption">Figure 1.1 — The five NEM regions and their dominant generation types. Each region has its own price, but they are connected by transmission lines (interconnectors) that allow electricity to flow between regions.</p>

| Region | State | Dominant generation | Character |
|--------|-------|-------------------|-----------|
| **NSW1** | New South Wales | Black coal, growing solar | Largest market, moderate volatility |
| **QLD1** | Queensland | Coal + gas + large-scale solar | High solar penetration, summer spikes |
| **VIC1** | Victoria | Brown coal (declining), wind | Transitioning rapidly to renewables |
| **SA1** | South Australia | Wind + solar + gas peakers | Most volatile, highest renewable share |
| **TAS1** | Tasmania | Hydro (80%+) | Structurally different, hydro-buffered |

<div class="definition-box">
<strong>Interconnector:</strong> A high-voltage transmission line that connects two regions. Interconnectors allow electricity to flow from a region with cheap power to one with expensive power. But they have limited capacity — when an interconnector "binds" (hits its transfer limit), the connected regions decouple and can have very different prices. For example, SA1 might spike to $5,000/MWh while VIC1 sits at $50, simply because the wire connecting them is full.
</div>

<div class="example-box">
<strong>Real-world example:</strong> South Australia (SA1) is the most interesting region for forecasting because it has the highest share of wind and solar in the NEM. On a windy, sunny day, SA1 can have negative prices (too much renewable energy, not enough demand). On a hot summer evening when the wind drops and everyone turns on their air conditioning, SA1 can spike to $15,000/MWh within minutes. This extreme volatility is what makes SA1 both challenging and lucrative for battery operators.
</div>

## How Electricity Is Priced

### The Gross Pool

Every single megawatt-hour of electricity in the NEM goes through a central auction, run by the **Australian Energy Market Operator (AEMO)**. This is called a **gross pool** — unlike some markets where buyers and sellers negotiate privately, all electricity must pass through the pool.

Here is how it works, step by step:

1. **Generators submit bids.** Every five minutes, each generator tells AEMO: "I can produce X megawatts at $Y per MWh." Generators can submit up to 10 price-quantity bands, from their minimum price to their maximum.

2. **AEMO stacks the bids.** AEMO arranges all bids from cheapest to most expensive. This creates the **merit-order supply stack** — a staircase of available generation, sorted by price.

3. **AEMO dispatches from the bottom.** Starting with the cheapest generator, AEMO dispatches generation until total supply equals total demand. The most expensive generator needed to balance supply and demand is the **marginal generator**.

4. **The marginal generator sets the price.** Every generator dispatched in that interval receives the marginal generator's bid price — not their own bid price. This is called **marginal pricing** (or uniform pricing).

<div class="definition-box">
<strong>Marginal pricing:</strong> A pricing rule where all sellers receive the price of the most expensive unit needed to meet demand. This means cheap generators (like solar and wind) earn far more than their own costs when an expensive gas plant sets the price. It's the same principle as an auction where all winners pay the price of the last accepted bid.
</div>

<div class="definition-box">
<strong>Dispatch price:</strong> The 5-minute price set by the marginal generator in each region. Six consecutive dispatch prices are averaged to produce the 30-minute <strong>trading price</strong>, which is the financially settled price — and our forecasting target.
</div>

<div class="key-point">
<strong>Why this matters for forecasting:</strong> Because the price is set by the <em>last</em> generator dispatched (not the average), a small change in demand at the margin can cause a huge price change. If demand rises just slightly and pushes the market from a $40/MWh coal unit to a $300/MWh gas peaker, the price jumps 7.5× even though demand only changed by 1%. This is the fundamental driver of electricity price volatility.
</div>

### The Price Range

![NEM price range](figures/01_price_range.png)

<p class="figure-caption">Figure 1.2 — The NEM price range spans from –$1,000 to $17,500/MWh, but most prices cluster in the $0–100 zone. The extreme range is what makes electricity prices so challenging to forecast.</p>

The NEM has administered price limits:

- **Market Price Cap (MPC):** $17,500/MWh. The maximum price a generator can bid. This cap exists to prevent unlimited market power exploitation.
- **Market Floor Price:** −$1,000/MWh. Generators bid negative to *avoid shutting down* (coal plants take hours to restart and cost millions to cycle) or to earn Renewable Energy Certificates.
- **Cumulative Price Threshold (CPT):** If prices stay extremely high for an extended period (roughly 7 days), an administered cap of ~$300/MWh kicks in to protect consumers.

<div class="example-box">
<strong>Real-world example — negative prices:</strong> On a sunny, windy Sunday in spring, SA1 might have 2,000 MW of wind and solar generation but only 1,200 MW of demand. The excess renewable energy pushes prices negative. Coal generators, which cannot easily shut down and restart, bid at –$1,000/MWh to stay running — they would rather <em>pay</em> $1,000 per MWh than endure the $5M+ cost of shutting down and restarting. This is a massive opportunity for batteries: charge for free (or get paid to charge) during negative prices, then discharge when prices recover.
</div>

## Timestamps: A Critical Detail

### Interval-ending vs interval-start

AEMO records data using **interval-ending** timestamps. This is a crucial but easy-to-miss convention that can silently corrupt your analysis if handled incorrectly.

![Interval timestamps](figures/01_interval_timestamps.png)

<p class="figure-caption">Figure 1.3 — Interval-ending timestamps label each period with its END time, not its START time. We shift to interval-start convention on load to make data joins safe.</p>

<div class="definition-box">
<strong>Interval-ending timestamp:</strong> A convention where each data point is labelled with the time at the END of the measurement period. The row timestamped "14:30" actually contains data measured from 14:00 to 14:30. AEMO uses this convention for all its data products.
</div>

<div class="definition-box">
<strong>Interval-start timestamp:</strong> A convention where each data point is labelled with the time at the START of the measurement period. The same data would be labelled "14:00" instead. This is more intuitive and makes joining different datasets (price, weather, demand) straightforward.
</div>

**Why this matters:** Suppose you want to join a price observation with a weather observation. The AEMO row labelled "14:30" covers the period 14:00–14:30. A weather observation at 14:30 was measured at the *end* of that period — it might contain information that was not yet available at the start. If you join on matching timestamps without shifting, you accidentally give your model future information — a form of **data leakage** that inflates performance.

<div class="key-point">
<strong>The rule:</strong> Shift AEMO timestamps back by one interval on load (14:30 → 14:00), converting to interval-start convention. Do this <strong>exactly once</strong>, at the data loading boundary. Applying it twice creates a systematic 30-minute offset. Forgetting it creates look-ahead leakage. Both errors produce no obvious error message — they silently corrupt every downstream analysis.
</div>

## The Target Transform: Taming Wild Prices

### The problem with raw prices

Electricity prices have statistical properties that break most forecasting models:

- **Enormous range:** −$1,000 to $17,500. A model trained on this range will be dominated by extreme observations.
- **Heavy tails:** Extreme prices (>$1,000) occur rarely but are far more extreme than a normal distribution would predict. Statisticians call this **leptokurtosis**.
- **Asymmetry:** The right tail (price spikes) is much heavier than the left tail (negative prices).
- **Negative values:** Prices can be negative, which rules out the common logarithmic transform (you cannot take the log of a negative number).

### The inverse hyperbolic sine (arcsinh)

The **arcsinh transform** solves all of these problems at once.

![Arcsinh transform](figures/01_arcsinh_transform.png)

<p class="figure-caption">Figure 1.4 — Left: raw prices are extremely right-skewed, making modelling nearly impossible. Centre: after arcsinh transformation, the distribution is much more symmetric. Right: the arcsinh function compresses extreme values while being nearly linear near zero.</p>

<div class="definition-box">
<strong>Inverse hyperbolic sine (arcsinh):</strong> A mathematical function defined as arcsinh(x) = ln(x + √(x^{2} + 1)). It behaves like a logarithm for large values (compressing them) but is defined for all real numbers, including negatives. It is the standard target transform for electricity price forecasting.
</div>

<div class="equation">

y = arcsinh(x) = ln(x + √(x^{2} + 1))

</div>

Why arcsinh is perfect for electricity prices:

| Property | Why it helps |
|----------|-------------|
| Defined for negatives | Handles −$1,000 floor prices without special cases |
| Compresses large values | A $15,000 spike becomes ~10 instead of dominating the model |
| Nearly linear near zero | Doesn't distort the bulk of prices ($20–$80 range) |
| Invertible | We can convert predictions back to dollars using sinh(y) |

<div class="key-point">
<strong>The golden rule:</strong> Always <em>model</em> in arcsinh space (train on arcsinh-transformed prices), but always <em>evaluate</em> in dollar space. If you measure forecast error in arcsinh space, you will hide poor performance on spikes — exactly the events that matter most for battery revenue.
</div>

<div class="example-box">
<strong>Real-world example — why the transform matters:</strong> Without arcsinh, a single $15,000 spike in the training data would dominate the mean squared error, causing the model to over-predict prices to avoid ever missing a spike. With arcsinh, that spike becomes ~10 in transformed space — still the largest value, but not 1,000× larger than a normal $50 price (which becomes ~4.6). The model can now learn patterns from both normal and extreme prices without being overwhelmed.
</div>

## Data Quality

### Missing intervals

Even AEMO's data feed is not perfect. Over a 4.5-year history at 5-minute resolution (~470,000 intervals), you will encounter gaps — missing intervals due to system outages, delayed publications, or corrupted files.

Standard approaches for handling missing data:

| Gap length | Strategy | Rationale |
|-----------|----------|-----------|
| 1–3 intervals (5–15 min) | Forward fill | Short gaps; price likely unchanged |
| 4–6 intervals (15–30 min) | Forward fill or flag | Borderline; depends on context |
| 7+ intervals (>30 min) | Flag and exclude | Too long to impute reliably |

<div class="definition-box">
<strong>Forward fill:</strong> Replace each missing value with the most recent known value. For electricity prices, this assumes the price stays constant during short data outages — a reasonable assumption since prices typically don't change by much in 5–15 minutes during normal conditions.
</div>

### The INTERVENTION flag

During **market interventions** (when AEMO directs generators to change output for grid reliability), the dispatch engine runs twice: once with the intervention, once without. Both results appear in the data. Always filter to `INTERVENTION = 0` — this is the "what would have happened under normal market operation" price, which is the correct modelling target.

### 5-minute to 30-minute aggregation

The 30-minute trading price is the arithmetic mean of six consecutive 5-minute dispatch prices. This matches AEMO's settlement calculation exactly. Using a different aggregation method (median, weighted average) would not match the actual financial settlement and would introduce systematic error.

## Data Storage: Parquet Format

<div class="definition-box">
<strong>Parquet:</strong> A columnar data storage format, standard in data science. It stores data by column rather than by row, which enables excellent compression and fast partial reads. A 200 MB CSV file of NEM price data compresses to roughly 15 MB in parquet format — with no information loss.
</div>

Advantages of parquet for this project:

- **Compression:** ~13× smaller than CSV. Saves disk space and speeds up loading.
- **Type preservation:** Dates remain dates, floats remain floats. No "is this column a string or a number?" parsing issues.
- **Partial reads:** You can load just the columns you need without reading the whole file.
- **Universal:** pandas, polars, duckdb, and spark all read parquet natively.

## Reproducibility: Why It Matters

<div class="definition-box">
<strong>Reproducibility:</strong> The ability for anyone to re-run your analysis and get exactly the same results. In scientific research and industry, this is essential for trust, debugging, and regulatory compliance.
</div>

Energy market research has a reproducibility problem. Download the same data a month later and you might get slightly different results — AEMO revises settlement data, library versions change, or different random seeds produce different train/test splits.

This project enforces reproducibility through four mechanisms:

1. **Cached downloads:** Every data download is saved locally. Once cached, the data never changes.
2. **Pinned random seeds:** A single seed (42) controls all random operations — train/test splits, cross-validation folds, model initialisation.
3. **Fixed date windows:** The train and test periods are specified in the config file, not computed from the data.
4. **Version pinning:** All Python package versions are locked in `pyproject.toml`.

<div class="example-box">
<strong>Real-world example — why reproducibility matters:</strong> Imagine you build a model that shows 15% improvement over the baseline. Your colleague re-runs it a week later and gets 8%. Did the model change? Did the data change? With full reproducibility, you can guarantee the same inputs produce the same outputs — and any difference must be due to a genuine change you made, not random variation or data drift.
</div>

---

## Glossary

| Term | Definition |
|------|-----------|
| **NEM** | National Electricity Market — Australia's wholesale electricity market for the eastern states |
| **AEMO** | Australian Energy Market Operator — runs the NEM dispatch and settlement |
| **MWh** | Megawatt-hour — standard unit of energy in wholesale markets |
| **Gross pool** | Market design where all electricity must be traded through the central exchange |
| **Marginal pricing** | All dispatched generators receive the price of the most expensive unit needed |
| **Dispatch price** | 5-minute price set by the marginal generator |
| **Trading price** | 30-minute average of six dispatch prices; the financially settled price |
| **Interconnector** | Transmission line connecting two NEM regions |
| **MPC** | Market Price Cap — maximum bid price ($17,500/MWh) |
| **Interval-ending** | Timestamp convention where rows are labelled with the period's END time |
| **arcsinh** | Inverse hyperbolic sine — the target transform for electricity prices |
| **Forward fill** | Imputation method that carries the last known value forward through gaps |
| **Parquet** | Columnar data storage format; standard for analytical workloads |

## Summary

The NEM is a marginal-pricing gross pool where the price is set by the most expensive generator dispatched every five minutes. Prices range from −$1,000 to $17,500/MWh with extreme volatility, driven by the interaction of variable renewables, inflexible coal, and weather-driven demand. Data arrives in interval-ending timestamps and must be shifted to interval-start convention exactly once on load. The arcsinh transform tames the wild price distribution for modelling, but we always evaluate in dollar space. Parquet provides efficient storage, and strict reproducibility measures ensure every result can be recreated exactly.
