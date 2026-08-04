# 11. FCAS and Co-optimised Dispatch

## Why 50 Hz Matters

The Australian power grid operates at a frequency of 50 Hz. This is not a regulation or a preference — it is a physical constraint that emerges from the synchronous rotation of all connected generators. Every turbine shaft in the National Electricity Market (NEM) rotates in lockstep: at 50 Hz, a two-pole generator spins at exactly 3,000 revolutions per minute. When supply and demand are perfectly balanced, the frequency holds steady. When they diverge, the frequency moves.

The physics is direct. If a large generator trips offline — say a 660 MW coal unit in the Latrobe Valley — the remaining generators cannot instantly increase output to compensate. The energy shortfall comes instead from the rotational kinetic energy stored in the spinning turbines themselves: they slow down slightly, converting their angular momentum into electrical energy. This deceleration manifests as a frequency drop. A loss of 660 MW on the eastern seaboard might cause the frequency to dip from 50.00 Hz to 49.80 Hz within a few seconds.

<div class="definition-box">
<strong>System frequency:</strong> The rate at which the alternating current waveform oscillates, measured in hertz (Hz). In the NEM, the nominal frequency is 50 Hz. The frequency reflects the instantaneous balance between electrical supply and demand across the entire interconnected grid. A frequency above 50 Hz means supply exceeds demand (generators are accelerating); below 50 Hz means demand exceeds supply (generators are decelerating).
</div>

Conversely, if demand drops suddenly — a large aluminium smelter shuts down, or a cloud clears and solar output surges — the generators have excess energy and speed up. The frequency rises above 50 Hz.

Why does frequency deviation matter? The consequences escalate with the magnitude of the deviation:

| Frequency | Condition | Consequence |
|-----------|-----------|-------------|
| 50.00 Hz | Nominal | Normal operation |
| 49.85–50.15 Hz | Normal operating band | Generators adjust automatically; no action required |
| 49.75–49.85 Hz | Outside normal band | AEMO activates contingency reserves |
| 49.50–49.75 Hz | Significant deviation | Under-frequency load shedding (UFLS) schemes may trigger |
| < 49.00 Hz | Emergency | Cascading trips; potential system black |
| > 50.50 Hz | Emergency | Over-frequency generator protection triggers disconnection |

The grid frequency is a single number that summarises the entire system's health. Every generator, every load, every battery connected to the grid experiences the same frequency at the same time (ignoring small local deviations due to network impedance). This makes frequency a uniquely important signal — and maintaining it at 50 Hz is the most critical task in power system operations.

<div class="key-point">
<strong>Key point:</strong> The frequency is not "set" by anyone — it <em>emerges</em> from the balance of supply and demand. AEMO cannot directly control the frequency any more than a central bank can directly control inflation. Instead, AEMO procures services that help generators and batteries respond to frequency deviations, restoring the balance. These services are collectively called Frequency Control Ancillary Services (FCAS).
</div>

### The Inertia Problem

Historically, the NEM had abundant **inertia** — the rotational kinetic energy stored in the heavy spinning masses of coal and gas turbines. A large synchronous generator might weigh hundreds of tonnes and spin at 3,000 RPM; the energy stored in that rotation acts as a buffer, slowing the rate at which frequency changes when supply and demand diverge. High inertia means the frequency changes slowly, giving other generators time to respond.

<div class="definition-box">
<strong>Inertia:</strong> The resistance of the power system to changes in frequency, determined by the total rotational kinetic energy of all synchronous generators connected to the grid. High inertia means frequency changes slowly after a disturbance (giving time for corrective action); low inertia means frequency changes rapidly (requiring faster responses). Measured in megawatt-seconds (MWs) or equivalently megajoules (MJ).
</div>

As coal plants retire and are replaced by wind and solar — which connect through power electronics and do not inherently provide inertia — the system's total inertia is declining. South Australia, with its high renewable penetration, routinely operates with inertia levels 50–70% below the levels of a decade ago. Lower inertia means faster frequency changes after disturbances, which means faster-responding reserves are needed. This is the fundamental driver behind the introduction of the very-fast 1-second FCAS markets in October 2023, and it is the reason batteries — which can respond in milliseconds — have become the dominant providers of fast frequency response.

<div class="example-box">
<strong>The SA system black — 28 September 2016:</strong> South Australia experienced a state-wide blackout after severe storms caused multiple transmission line faults. The sequence of events exposed the fragility of a low-inertia system: wind farms disconnected due to protection settings, the Heywood interconnector tripped on overload, and the remaining SA generators could not maintain frequency. The frequency collapsed from 50 Hz to below 47 Hz in under two seconds. The entire state lost power. This event was a watershed moment for FCAS — it demonstrated that the NEM's existing frequency control arrangements, designed for a system with high inertia, were inadequate for the emerging low-inertia reality. The subsequent reforms included mandatory minimum inertia requirements, new generator technical standards, and ultimately the creation of the very-fast FCAS markets.
</div>

---

## Regulation vs Contingency Services

FCAS in the NEM falls into two broad categories, defined by what they respond to:

### Regulation FCAS

Regulation services correct the small, continuous frequency deviations that occur during normal operation — the constant jitter caused by thousands of loads switching on and off, small fluctuations in wind and solar output, and the inherent imprecision of dispatching generators in five-minute intervals. The frequency never sits perfectly at 50.00 Hz; it drifts slightly around it, and regulation services nudge it back.

<div class="definition-box">
<strong>Regulation FCAS:</strong> Frequency control services that correct small, continuous deviations from 50 Hz during normal operation. Regulation is provided by generators and batteries that adjust their output in response to an Automatic Generation Control (AGC) signal sent by AEMO every four seconds. There are two regulation markets: <strong>regulation raise</strong> (increase output to push frequency up) and <strong>regulation lower</strong> (decrease output to push frequency down). Regulation providers must be able to sustain their response for at least a specified duration and respond continuously, not just to discrete events.
</div>

Regulation providers are controlled by AEMO's **Automatic Generation Control (AGC)** system, which sends a control signal every four seconds. The AGC measures the current frequency error and distributes corrective actions across all enabled regulation providers, proportional to their enabled capacity. A battery providing 10 MW of regulation raise might receive an AGC signal to increase output by 3 MW, then 5 MW, then 2 MW, adjusting every four seconds as the frequency drifts.

The regulation signal is continuous and bidirectional (though raise and lower are separate markets). Over the course of a day, a regulation provider's net energy delivery is typically close to zero — the raises and lowers roughly cancel out. This means regulation has minimal impact on the battery's state of charge (SoC), which is a significant advantage: the battery can earn FCAS revenue without using up its energy for arbitrage.

<div class="example-box">
<strong>Regulation in practice — the AGC dance:</strong> A battery enabled for 20 MW of regulation raise receives the AGC signal every four seconds. Over a 30-minute dispatch interval, it might follow a pattern like: +3 MW, +7 MW, +2 MW, -1 MW, +5 MW, +12 MW, +4 MW, +1 MW, -2 MW, +8 MW... The signal varies rapidly as AEMO chases the frequency error. The battery's average output might be +5 MW over the interval — far less than the 20 MW it has been paid to keep available. The key requirement is that the battery <em>can</em> deliver up to 20 MW within the response timeframe, not that it <em>does</em> deliver 20 MW continuously.
</div>

### Contingency FCAS

Contingency services respond to large, sudden disturbances — the loss of a major generator, a transmission line trip, or a sudden load disconnection. These are discrete events, not continuous fluctuations. When a 660 MW coal unit trips, the frequency drops sharply, and contingency FCAS providers activate automatically (not via AGC) to arrest the decline and restore the frequency.

<div class="definition-box">
<strong>Contingency FCAS:</strong> Frequency control services that respond to large, sudden frequency disturbances such as generator trips or transmission line failures. Contingency services are activated automatically based on local frequency measurements, not by AGC signals. They are categorised by their response speed: 1-second (very fast), 6-second (fast), 60-second (slow), and 5-minute (delayed). Each category has both raise and lower markets, totalling eight contingency FCAS markets.
</div>

Contingency services are categorised by how quickly they must respond:

| Service | Response | Sustain | Provider |
|---------|----------|---------|----------|
| Very fast (R1/L1) | 1 s | 1 s min | Batteries, demand response |
| Fast (R6/L6) | 6 s | 60 s min | Batteries, hydro, gas |
| Slow (R60/L60) | 60 s | 5 min | Gas, hydro, batteries |
| Delayed (R5/L5) | 5 min | 10 min | Gas, interruptible load |

The response timeframes reflect the physics of frequency recovery after a disturbance:

1. **First 1 second (very fast):** The frequency drops most rapidly immediately after the contingency. Very-fast providers must arrest this initial plunge before it triggers under-frequency load shedding.
2. **First 6 seconds (fast):** Fast providers sustain the arrest and begin stabilising the frequency.
3. **First 60 seconds (slow):** Slow providers take over from the fast providers, allowing them to return to their pre-contingency operating point.
4. **First 5 minutes (delayed):** Delayed providers relieve the slow providers, fully restoring the frequency and replenishing the reserves for the next contingency.

<div class="key-point">
<strong>Key point:</strong> A single physical unit — such as the Mannum battery — can provide multiple contingency services simultaneously, as long as it has sufficient headroom. If Mannum is generating 40 MW and can ramp up to 100 MW, it has 60 MW of headroom. It might offer 20 MW of very-fast raise, 20 MW of fast raise, and keep 20 MW as a buffer. The key constraint is that the same megawatt cannot be offered into two different services — the reserves do not overlap.
</div>

### The Key Difference

The operational distinction between regulation and contingency is important for battery economics:

- **Regulation** requires continuous, small adjustments. The battery must follow the AGC signal continuously, which causes small, frequent cycles. Over a day, the energy delivered nets close to zero, but the constant cycling adds wear.
- **Contingency** requires the battery to keep capacity in reserve and deliver it only when a contingency event occurs. Most of the time, the battery is paid to do nothing — it earns the enablement payment for being available. Only when the frequency deviates beyond the trigger threshold does the battery actually inject or absorb power. Contingency events are rare (a few per week of significant magnitude), so the energy throughput from contingency services is minimal.

For a battery operator, contingency FCAS is often more attractive than regulation because the revenue comes primarily from **availability** rather than **delivery** — the battery earns money for keeping headroom available, not for cycling. This preserves the SoC for energy arbitrage and minimises degradation.

---

## The Ten FCAS Markets

The NEM operates ten distinct FCAS markets, each cleared every five minutes alongside the energy market:

| Market | Code | Response | Added |
|--------|------|----------|-------|
| Regulation raise | RAISEREG | AGC | Original |
| Regulation lower | LOWERREG | AGC | Original |
| Very fast raise | RAISE1SEC | 1 s | Oct 2023 |
| Very fast lower | LOWER1SEC | 1 s | Oct 2023 |
| Fast raise | RAISE6SEC | 6 s | Original |
| Fast lower | LOWER6SEC | 6 s | Original |
| Slow raise | RAISE60SEC | 60 s | Original |
| Slow lower | LOWER60SEC | 60 s | Original |
| Delayed raise | RAISE5MIN | 5 min | Original |
| Delayed lower | LOWER5MIN | 5 min | Original |

The original eight markets have operated since the NEM's inception in 1998. The very-fast 1-second markets were added in October 2023 specifically to address the declining-inertia problem. These two new markets have been transformative for battery economics.

### The 1-Second Markets

The very-fast raise (RAISE1SEC) and very-fast lower (LOWER1SEC) markets require providers to respond within one second of a frequency deviation. This speed requirement effectively excludes thermal generators (which take seconds to minutes to ramp) and makes batteries and demand-response loads the only viable providers.

<div class="example-box">
<strong>Battery dominance of RAISE1SEC:</strong> In the first year of the 1-second markets (October 2023 to October 2024), batteries provided over 90% of all RAISE1SEC enablement across the NEM. No gas turbine, no hydro unit, no coal plant can respond within one second with the precision and repeatability that AEMO requires. The 1-second markets were designed with batteries in mind — they create a revenue stream that only batteries (and some demand-response loads) can access, partially compensating batteries for the declining value of arbitrage as more batteries enter the market.
</div>

The 1-second markets are particularly attractive for a battery like Mannum because:

1. **No competition from thermal generators.** The response-speed requirement creates a natural barrier to entry.
2. **Minimal energy throughput.** Like other contingency services, 1-second FCAS is paid primarily for availability. The battery keeps headroom but rarely needs to deliver it.
3. **Stacking with other services.** A battery can provide 1-second raise and 6-second raise and 60-second raise simultaneously, as long as the total reserved capacity does not exceed the available headroom.
4. **Complementary to arbitrage.** The headroom reserved for 1-second FCAS reduces the energy available for arbitrage, but the FCAS price often compensates for the foregone arbitrage revenue — especially during periods of low price volatility when arbitrage opportunities are slim.

### Price Behaviour Across FCAS Markets

FCAS prices behave very differently from energy prices. Most of the time, FCAS prices are low — typically $1–$15/MWh across most markets. The NEM usually has ample reserves, and the many batteries, hydro units, and gas turbines competing to provide FCAS keep prices near the marginal cost of availability.

But FCAS prices spike violently when reserves are scarce. A large generator trip in South Australia on a hot afternoon — when the system is already stressed, the interconnectors are near their limits, and local reserves are thin — can cause the RAISE6SEC price to spike from $5/MWh to $5,000/MWh or even $15,100/MWh (the FCAS market price cap) for one or more dispatch intervals. These spikes are brief but extreme, and they dominate the annual FCAS revenue for providers who happen to be enabled when they occur.

![FCAS price series with contingency spikes](figures/11_fcas_price_series.png)

<p class="figure-caption">Figure 11.1 — FCAS raise prices (RAISE6SEC) in SA1 over three months. Prices sit near $5/MWh for most intervals but spike to $300–$15,100/MWh during contingency events and enablement scarcity. These rare spikes dominate annual FCAS revenue. The timing and magnitude of spikes are extremely difficult to forecast.</p>

This price behaviour has two implications for battery strategy:

1. **Average FCAS revenue is modest.** A time-weighted average FCAS price of $5–$15/MWh, applied to the enabled capacity, produces a steady but unspectacular revenue stream.
2. **Spike FCAS revenue is substantial but unpredictable.** A battery enabled for 50 MW of RAISE6SEC that happens to be enabled during a $15,100/MWh spike earns $12,583 in that single five-minute interval ($15,100 · 50 MW · 5/60 hours). A few such events per year can contribute significantly to annual revenue.

<div class="key-point">
<strong>Key point:</strong> FCAS revenue has a "lottery ticket" quality — the expected value over a year is meaningful, but the realisation in any given week is highly variable. This makes FCAS revenue forecasting fundamentally different from energy price forecasting. Energy prices have strong daily and seasonal patterns that models can exploit; FCAS price spikes are driven by contingency events (generator trips, transmission faults) that are essentially random.
</div>

### FCAS Revenue as a Share of Total Battery Revenue

For a well-operated battery in the NEM, FCAS revenue typically accounts for 20–40% of total revenue, with the remainder coming from energy arbitrage:

![FCAS revenue share by service](figures/11_fcas_revenue_share.png)

<p class="figure-caption">Figure 11.2 — Breakdown of total battery revenue by source for a representative NEM battery over 2024–2025. Energy arbitrage dominates, but FCAS collectively contributes a material share. The 1-second markets, despite being the newest, already contribute meaningfully because of the limited competition from non-battery providers.</p>

| Revenue source | Typical share | Variability |
|----------------|--------------|-------------|
| Energy arbitrage | 50–70% | Driven by price volatility; higher in SA1, QLD1 |
| Regulation raise/lower | 5–10% | Relatively stable; depends on AGC enablement |
| 1-second raise/lower | 5–15% | Growing; limited competition keeps prices higher |
| 6-second raise/lower | 5–10% | Spike-dependent; high variance across months |
| 60-second raise/lower | 2–5% | Lower value; more competition from gas |
| 5-minute raise/lower | 1–3% | Lowest FCAS value; most competition |

These shares vary significantly by region (SA1 has higher FCAS value due to lower inertia and higher contingency frequency), by season (summer has more stress events), and by the number of batteries competing for FCAS enablement. As more batteries enter the NEM, FCAS prices are expected to face downward pressure, but the declining-inertia trend partially offsets this by increasing the volume of FCAS required.

---

## Co-optimisation in One Dispatch

### How NEMDE Works

Every five minutes, AEMO runs the **National Electricity Market Dispatch Engine (NEMDE)** to clear the energy market and all ten FCAS markets simultaneously. This is not a sequence of separate auctions — it is a single, integrated optimisation that finds the least-cost combination of energy dispatch and FCAS enablement across all generators, batteries, and loads in the NEM.

<div class="definition-box">
<strong>Co-optimisation:</strong> The simultaneous optimisation of energy dispatch and FCAS enablement in a single mathematical program. Co-optimisation ensures that the total cost of meeting both energy demand and FCAS requirements is minimised, accounting for the fact that a generator providing FCAS reserves cannot simultaneously use that capacity for energy production. Without co-optimisation, the energy and FCAS markets would be cleared independently, leading to higher total cost because the interdependence between energy headroom and FCAS availability would not be priced correctly.
</div>

The co-optimisation is essential because energy and FCAS compete for the same physical capacity. A battery with 100 MW of discharge capacity can offer 100 MW into the energy market, or 100 MW of FCAS raise, or any split between the two — but the total cannot exceed 100 MW. If the energy price is $200/MWh and the RAISE6SEC price is $5/MWh, the battery should clearly dispatch energy. But if the RAISE6SEC price is $300/MWh (during a scarcity event), the battery should reserve capacity for FCAS. NEMDE makes this allocation optimally for every generator and battery in the system, every five minutes.

NEMDE solves a large linear program (LP) with the following structure:

**Objective:** Minimise the total cost of energy dispatch plus the total cost of FCAS enablement across all regions and services.

**Decision variables:** For each generating unit (including batteries): the energy dispatch level, and the enablement quantity for each of the ten FCAS markets.

**Constraints:**
- Energy supply must meet demand in each region (accounting for interconnector flows and losses)
- FCAS enablement must meet the FCAS requirement in each region and service
- Each unit's energy dispatch plus FCAS enablement cannot exceed its physical capacity (this is where the joint capacity constraint — the trapezium — enters)
- Interconnector flow limits
- Ramp rate limits
- Unit-specific constraints (minimum generation, maximum capacity, etc.)

<div class="definition-box">
<strong>NEMDE (National Electricity Market Dispatch Engine):</strong> The software system operated by AEMO that clears the NEM every five minutes. NEMDE takes as input the bids and offers from all generators and loads, the demand forecasts, the FCAS requirements, the network constraints, and the interconnector limits. It solves a co-optimisation LP to find the least-cost dispatch of energy and enablement of FCAS across all regions and services. The shadow prices (dual variables) of the demand and FCAS requirement constraints become the market prices for energy and each FCAS service.
</div>

The **market prices** for energy and each FCAS service are the **shadow prices** (dual variables) of the corresponding demand and requirement constraints. This is the same mechanism as energy price formation, which was introduced in Chapter 3: the price equals the cost of supplying one additional megawatt of energy (or one additional megawatt of FCAS) at the optimal dispatch point. Co-optimisation means that the energy price and FCAS prices are determined simultaneously and influence each other — a tight FCAS market can raise the energy price (because generators that would otherwise produce cheap energy are diverted to FCAS), and a tight energy market can raise FCAS prices (because the opportunity cost of reserving capacity for FCAS — foregone energy revenue — is high).

### Using nempy to See Co-optimisation

nempy (developed by UNSW-CEEM) is an open-source Python implementation of the NEM dispatch procedure. It solves the same co-optimisation LP that NEMDE solves, using historical bid and offer data. This makes it an invaluable tool for understanding how co-optimisation works and how a battery's bids affect both energy and FCAS prices.

To see co-optimisation in action, start with a concrete example. Load the historical bid stack for SA1 at 17:30 on 15 January 2026, add the Mannum battery's energy and FCAS offers, and call nempy's dispatch solver. The output shows:

1. **The energy dispatch** for every generator in the region — how many megawatts each unit is producing
2. **The FCAS enablement** for every unit and every FCAS service — how many megawatts of each FCAS market each unit is providing
3. **The energy price** and all ten **FCAS prices** for the interval
4. **The binding constraints** — which physical limits (generator capacity, interconnector flow, FCAS requirement) are constraining the solution

<div class="example-box">
<strong>A nempy experiment — perturbing one input:</strong> Start with the base case: Mannum bids 100 MW of energy at $0/MWh (price-taking, as discussed in Chapter 3) and offers no FCAS. Note the energy price (say $185/MWh) and the RAISE6SEC price (say $8/MWh). Now modify Mannum's offers: bid only 60 MW of energy and offer 40 MW of RAISE6SEC at $0/MWh. Re-run the dispatch. The energy price rises slightly (say to $192/MWh) because 40 MW of low-cost supply has been withdrawn from the energy market, and the RAISE6SEC price drops (say to $3/MWh) because 40 MW of new supply has entered the FCAS market. Mannum's revenue changes too: in the base case, it earned $185 · 100 MW · 5/60 = $1,542 for the interval. In the co-optimised case, it earns $192 · 60 MW · 5/60 + $3 · 40 MW · 5/60 = $960 + $10 = $970. In this scenario, co-optimisation reduces Mannum's revenue — the energy price was high enough that diverting capacity to FCAS was not worthwhile. But in a different interval, where the energy price is $45/MWh and the RAISE6SEC price is $120/MWh (during an enablement scarcity event), the co-optimised strategy would dominate.
</div>

This nempy experiment illustrates the central tension of co-optimisation: **every megawatt reserved for FCAS is a megawatt that cannot earn energy arbitrage revenue, and vice versa.** The optimal allocation depends on the relative prices, which change every five minutes.

### Price Interaction Effects

Co-optimisation creates subtle price interactions that a battery operator must understand:

**FCAS constraining energy prices upward.** When FCAS requirements are tight (e.g., after a large contingency event), NEMDE must divert some generators from energy to FCAS. This reduces the energy supply, causing the energy price to rise. The increase in the energy price is precisely the opportunity cost of the FCAS — the marginal cost of "freeing up" capacity from energy to FCAS. In the extreme, a very tight FCAS market can cause energy price spikes even when there is ample generation capacity, because the generation capacity is being "locked up" for FCAS.

**Energy prices affecting FCAS prices.** When energy prices are high, the opportunity cost of reserving capacity for FCAS is also high — a generator providing FCAS raises foregos selling energy at the high energy price. This opportunity cost flows through the co-optimisation and raises the FCAS price. Conversely, when energy prices are low (e.g., during midday solar surplus), the opportunity cost of FCAS is low, and FCAS prices tend to be low as well.

**Cross-market effects.** The ten FCAS markets also interact with each other. A unit providing RAISE6SEC (fast raise) can also count toward RAISE60SEC (slow raise) and RAISE5MIN (delayed raise), because a unit that can respond within 6 seconds can certainly respond within 60 seconds. NEMDE accounts for this "cascading" — enablement in a faster service can satisfy requirements in slower services. This means that tightness in the fast raise market can push up slow raise prices (because fast providers are diverted from slow to fast) or push them down (because fast enablement cascades to satisfy slow requirements).

<div class="key-point">
<strong>Key point:</strong> Co-optimisation means you cannot analyse energy and FCAS markets independently. The prices are coupled — a change in one market ripples through all the others. This coupling is precisely why co-optimisation exists: clearing the markets independently would miss these interactions and produce a higher total cost. For a battery operator, this means the optimal bidding strategy must consider all markets simultaneously, not optimise energy and FCAS offers in isolation.
</div>

---

## The Joint Capacity Constraint: The Enablement Trapezium

### The Physical Constraint

The most important constraint in FCAS co-optimisation is the **joint capacity constraint**, which links a unit's energy dispatch to its FCAS enablement. This constraint is typically visualised as a trapezium (trapezoid) in the energy-FCAS space, and it encodes a simple physical truth: **a unit cannot simultaneously use all its capacity for energy and all its capacity for FCAS.**

<div class="definition-box">
<strong>Enablement trapezium (trapezoid constraint):</strong> The geometric region in (energy dispatch, FCAS enablement) space that defines the feasible combinations of energy output and FCAS reserve for a generating unit. The trapezium shape arises because FCAS capacity is limited by the unit's ability to increase output (for raise) or decrease output (for lower) from its current operating point. A unit running at maximum output has no headroom for raise FCAS; a unit at minimum output has no headroom for lower FCAS. The trapezium encodes these physical limits as linear constraints in the NEMDE co-optimisation.
</div>

For a battery providing contingency **raise** FCAS, the constraint is intuitive:

- The battery has a maximum discharge rate of P_max MW (100 MW for Mannum)
- If the battery is currently dispatched to produce E MW of energy, its headroom for raise FCAS is at most P_max - E MW
- Additionally, the battery must have sufficient state of charge (SoC) to sustain the raise response for the required duration

These constraints define the trapezium. For the Mannum battery (100 MW / 200 MWh), providing contingency raise:

![The enablement trapezium](figures/11_trapezium.png)

<p class="figure-caption">Figure 11.3 — The enablement trapezium for the Mannum battery (100 MW / 200 MWh) providing contingency raise FCAS. The x-axis is energy dispatch (MW), the y-axis is FCAS raise enablement (MW). The shaded region shows feasible combinations. At zero energy output, the full 100 MW is available for raise FCAS. As energy dispatch increases, the available FCAS headroom decreases linearly. At 100 MW energy dispatch, no raise FCAS is possible.</p>

The trapezium has four key points:

| Point | Energy dispatch | FCAS enablement | Description |
|-------|---------------|-----------------|-------------|
| A | 0 MW | 0 MW | Idle — no energy, no FCAS |
| B | 0 MW | 100 MW | Maximum FCAS — all capacity reserved for raise |
| C | 100 MW | 0 MW | Maximum energy — no headroom for raise |
| D | Some intermediate | Some intermediate | Co-optimised — split between energy and FCAS |

The boundary of the trapezium is defined by:

<pre>
Energy_dispatch + FCAS_raise_enablement  ≤  P_max
</pre>

For lower FCAS, the constraint works in reverse — the battery must have headroom to reduce its output (or increase its charging rate):

<pre>
FCAS_lower_enablement  ≤  Energy_dispatch - P_min          (generator)
FCAS_lower_enablement  ≤  P_max_charge - |Charging_rate|   (battery absorbing)
</pre>

When multiple FCAS services are offered simultaneously, the constraints become more complex. A battery offering very-fast raise, fast raise, slow raise, and delayed raise must satisfy:

<pre>
Energy_dispatch + RAISE1SEC + RAISE6SEC + RAISE60SEC + RAISE5MIN  ≤  P_max
</pre>

This is the **stacking** constraint — the total of energy dispatch plus all raise FCAS enablements cannot exceed the physical capacity of the unit.

### The SoC Constraint on FCAS

The power constraint (the trapezium above) is not the only limit on FCAS provision. The battery must also have sufficient **energy** (state of charge) to sustain the FCAS response for the required duration if called upon.

For contingency raise, the battery must be able to sustain the enabled capacity for the required sustain time. For RAISE6SEC, the sustain time is 60 seconds. If Mannum is enabled for 50 MW of RAISE6SEC, it must have at least:

<pre>
SoC_required = 50 MW · (60 s / 3600 s per hour) = 0.83 MWh
</pre>

This is a trivial requirement for a 200 MWh battery. But for RAISE5MIN (sustain time of 10 minutes), the SoC requirement is larger:

<pre>
SoC_required = 50 MW · (10 min / 60 min per hour) = 8.33 MWh
</pre>

Still manageable, but the SoC constraint becomes binding when the battery is nearly empty or nearly full. A battery at 5% SoC (10 MWh for Mannum) cannot credibly offer large quantities of raise FCAS requiring a 10-minute sustain, because it would run out of energy before the sustain period ends.

<div class="key-point">
<strong>Key point:</strong> The trapezium constraint links FCAS to power headroom; the SoC constraint links FCAS to energy reserves. Together, they define the feasible FCAS envelope as a function of the battery's current operating state. The dispatch optimiser (whether NEMDE or the battery's internal MPC) must respect both constraints simultaneously. This creates a three-way tradeoff: energy dispatch consumes SoC (reducing future FCAS capability), FCAS enablement reduces energy headroom (reducing current arbitrage), and the SoC trajectory over time couples current FCAS decisions to future energy decisions.
</div>

### How the Trapezium Affects Battery Strategy

The trapezium constraint has profound implications for battery dispatch strategy, particularly when integrating FCAS into the MPC framework from Chapter 9:

**1. FCAS competes directly with arbitrage throughput.** Every megawatt reserved for FCAS raise is a megawatt that cannot be used for energy discharge. If the battery is discharging at peak (100 MW) to capture a $300/MWh price spike, it has zero headroom for raise FCAS. To provide 30 MW of RAISE6SEC, it must reduce its discharge to 70 MW, giving up 30 MW · $300/MWh · (5/60) hours = $750 of energy revenue per interval. This is worthwhile only if the RAISE6SEC payment exceeds $750 for that interval — i.e., if the RAISE6SEC price exceeds $750 / (30 MW · 5/60 hours) = $300/MWh.

**2. FCAS has an opportunity cost that varies with the energy price.** The cost of providing FCAS is not the direct cost (batteries have near-zero marginal cost) but the **foregone energy revenue**. When energy prices are high, the opportunity cost of FCAS is high, and it takes a high FCAS price to justify reserving headroom. When energy prices are low, the opportunity cost is low, and even modest FCAS prices make it worthwhile.

**3. The trapezium favours partial dispatch.** A battery that always dispatches at 100% capacity (fully charging or fully discharging) can never provide raise FCAS during discharge or lower FCAS during charge. A battery that operates at 70–80% capacity retains 20–30 MW of headroom for FCAS. This partial dispatch is suboptimal for pure arbitrage but can be revenue-optimal when FCAS is included.

**4. SoC management becomes more complex.** The MPC from Chapters 5 and 9 managed SoC to maximise arbitrage — charging during cheap periods and discharging during expensive ones. With FCAS, SoC management must also consider the FCAS eligibility constraints: keeping enough energy for raise sustain, keeping enough empty capacity for lower sustain. The SoC target at any point in time is no longer purely determined by the price forecast — it also depends on the expected FCAS prices and requirements.

<div class="example-box">
<strong>The Mannum trapezium in practice:</strong> At 17:30 on a January evening, SA1 energy prices are $180/MWh and rising. Mannum has 150 MWh in its 200 MWh tank (75% SoC). The battery's MPC wants to discharge at 100 MW to capture the price spike. But RAISE6SEC is currently $25/MWh (elevated due to a nearby generator being on a forced outage), and RAISE1SEC is $40/MWh.

If Mannum discharges at 100 MW: energy revenue = $180 · 100 · (5/60) = $1,500 per interval. FCAS revenue = $0.

If Mannum discharges at 70 MW and enables 30 MW across RAISE1SEC (15 MW) and RAISE6SEC (15 MW): energy revenue = $180 · 70 · (5/60) = $1,050. FCAS revenue = $40 · 15 · (5/60) + $25 · 15 · (5/60) = $50 + $31 = $81. Total = $1,131.

In this case, full discharge ($1,500) beats the co-optimised strategy ($1,131) by $369. The energy price is too high relative to the FCAS prices. But if the energy price drops to $60/MWh and RAISE6SEC spikes to $200/MWh (a contingency event), the co-optimised strategy dominates: energy revenue = $60 · 70 · (5/60) = $350, FCAS revenue = $200 · 30 · (5/60) = $500, total = $850 vs energy-only = $60 · 100 · (5/60) = $500.
</div>

---

## Co-optimised Dispatch: Energy-Only vs Joint

### Setting Up the Comparison

To understand the value of co-optimisation, compare two dispatch strategies for the Mannum battery over a representative period — say, a week in January 2026:

**Strategy A: Energy-only dispatch.** The MPC from Chapters 5 and 9, optimising only against the energy price forecast. The battery charges at the cheapest periods and discharges at the most expensive. No FCAS is offered. All 100 MW of capacity is available for energy at all times.

**Strategy B: Co-optimised dispatch.** The MPC is extended to jointly optimise energy dispatch and FCAS enablement. At each five-minute interval, the optimiser decides how to split the battery's capacity between energy and each FCAS service, respecting the trapezium and SoC constraints. The optimiser uses forecasts for both energy prices and FCAS prices.

### The LP Extension for Co-optimisation

The energy-only LP from Chapter 5 had decision variables for charge and discharge at each time step. The co-optimised LP adds FCAS enablement variables:

For each time step t:
- charge_t: charging power (MW)
- discharge_t: discharging power (MW)
- fcas_raise1_t: enablement for very-fast raise (MW)
- fcas_raise6_t: enablement for fast raise (MW)
- fcas_raise60_t: enablement for slow raise (MW)
- fcas_raise5_t: enablement for delayed raise (MW)
- fcas_lower1_t, fcas_lower6_t, fcas_lower60_t, fcas_lower5_t: enablements for lower services (MW)
- fcas_reg_raise_t, fcas_reg_lower_t: enablements for regulation (MW)

The objective function becomes:

<pre>
Maximise  Σ_t [ p_energy(t) · (discharge(t) - charge(t)) · Δt
              + Σ_s  p_fcas(s,t) · fcas(s,t) · Δt ]
</pre>

Subject to:

<pre>
Trapezium (raise):
  discharge(t) + fcas_raise1(t) + fcas_raise6(t)
    + fcas_raise60(t) + fcas_raise5(t) + fcas_reg_raise(t)  ≤  P_max

Trapezium (lower):
  charge(t) + fcas_lower1(t) + fcas_lower6(t)
    + fcas_lower60(t) + fcas_lower5(t) + fcas_reg_lower(t)  ≤  P_max_charge

SoC dynamics:
  SoC(t+1) = SoC(t) - discharge(t) · Δt / η + charge(t) · Δt · η

SoC limits:
  SoC_min  ≤  SoC(t)  ≤  SoC_max

SoC sufficient for raise sustain:
  SoC(t)  ≥  Σ_s [ fcas_raise(s,t) · sustain_time(s) ]

SoC sufficient for lower sustain:
  (SoC_max - SoC(t))  ≥  Σ_s [ fcas_lower(s,t) · sustain_time(s) ]

Non-negativity:
  all variables ≥ 0
</pre>

This is still a linear program — no nonlinear terms — so it can be solved with the same LP solvers used in Chapters 5 and 9. The LP from Chapter 9 extends naturally; the additional variables and constraints increase the problem size but not its computational difficulty.

### Worked Example: Evening Peak with Mannum

Consider a single evening peak period — 17:00 to 21:00 on 15 January 2026 in SA1 — and compare the two strategies.

**Market conditions (representative):**
- Energy prices: rising from $80/MWh at 17:00 to $250/MWh at 18:30, falling to $60/MWh by 21:00
- RAISE6SEC: $8/MWh for most intervals, spiking to $180/MWh at 18:15 when a gas turbine trips
- RAISE1SEC: $12/MWh average, spiking to $250/MWh at 18:15
- Other FCAS services: $3–$10/MWh throughout
- Mannum starting SoC: 180 MWh (90%)

**Strategy A: Energy-only dispatch**

The energy-only MPC identifies the optimal discharge window as 17:30–19:30 (4 hours at various rates), targeting the price peak at 18:30. It discharges at 100 MW during the highest-priced intervals:

| Period | MW | Price | Revenue |
|--------|-----|-------|---------|
| 17:00–17:30 | 50 | $80 | $2,000 |
| 17:30–18:00 | 100 | $150 | $7,500 |
| 18:00–18:30 | 100 | $250 | $12,500 |
| 18:30–19:00 | 100 | $200 | $10,000 |
| 19:00–19:30 | 80 | $120 | $4,800 |
| 19:30–20:00 | 40 | $80 | $1,600 |
| **Total** | | | **$38,400** |

Energy throughput: approximately 175 MWh discharged (from 180 MWh to 5 MWh SoC).

**Strategy B: Co-optimised dispatch**

The co-optimised MPC reduces energy dispatch by 20–30 MW during high-energy-price intervals to reserve headroom for FCAS raise, and offers the reserved capacity into RAISE1SEC and RAISE6SEC:

| Period | Energy MW | FCAS MW | Energy $ | FCAS $ |
|--------|----------|---------|----------|--------|
| 17:00–17:30 | 50 | 30 | $2,000 | $150 |
| 17:30–18:00 | 75 | 25 | $5,625 | $163 |
| 18:00–18:30 | 70 | 30 | $8,750 | $3,225* |
| 18:30–19:00 | 80 | 20 | $8,000 | $130 |
| 19:00–19:30 | 70 | 20 | $4,200 | $100 |
| 19:30–20:00 | 35 | 15 | $1,400 | $45 |
| **Total** | | | **$29,975** | **$3,813** |
| **Combined** | | | | **$33,788** |

*The FCAS revenue spike at 18:00–18:30 occurs when the gas turbine trip sends RAISE6SEC to $180/MWh and RAISE1SEC to $250/MWh. This single interval contributes $250 · 15 · (5/60) + $180 · 15 · (5/60) = $313 + $225 = $538 per five-minute block, summing to approximately $3,225 over the half-hour (assuming the spike persists for about 15 minutes of the 30-minute window).

Energy throughput: approximately 142 MWh discharged (from 180 MWh to 38 MWh SoC).

**Comparison:**

| Metric | Energy-only | Co-optimised | Difference |
|--------|------------|-------------|-----------|
| Energy revenue | $38,400 | $29,975 | -$8,425 |
| FCAS revenue | $0 | $3,813 | +$3,813 |
| Total revenue | $38,400 | $33,788 | -$4,612 |
| Energy throughput | 175 MWh | 142 MWh | -33 MWh |
| Remaining SoC | 5 MWh | 38 MWh | +33 MWh |

In this particular evening, the energy-only strategy wins by $4,612 because the energy prices were consistently high and the FCAS spike, while dramatic, was too brief to compensate for the foregone energy revenue. The co-optimised battery gave up $8,425 of energy revenue to earn only $3,813 in FCAS.

<div class="key-point">
<strong>Key point:</strong> Co-optimisation does not always beat energy-only dispatch on any given day. Its value is in the <em>expected</em> return over many days, including the rare but valuable intervals when FCAS prices spike dramatically. Over a full year, the co-optimised strategy typically produces 15–30% higher total revenue than energy-only, because the FCAS revenue earned during normal operation plus the occasional windfall from FCAS spikes more than compensates for the systematically reduced energy throughput.
</div>

However, notice that the co-optimised battery retains 38 MWh of SoC at 20:00, compared to only 5 MWh for the energy-only battery. If energy prices spike again later in the evening (e.g., an unexpected demand surge at 21:00), the co-optimised battery has flexibility to respond while the energy-only battery is nearly empty. This **optionality value** of retained SoC is not captured in the single-evening comparison but is significant over time.

### The nempy View of One Interval

To understand exactly how NEMDE allocates capacity, use nempy to clear a single interval. Load the SA1 bid stack for 18:15 on 15 January 2026 — the moment the gas turbine trips and FCAS prices spike. Add Mannum's offers:

- Energy: 70 MW at $0/MWh (price-taking)
- RAISE1SEC: 15 MW at $0/MWh
- RAISE6SEC: 15 MW at $0/MWh

Call nempy's dispatch solver. The output shows Mannum dispatched for 70 MW of energy, enabled for 15 MW of RAISE1SEC, and enabled for 15 MW of RAISE6SEC — exactly matching its offers, because the FCAS requirements are binding (scarce) in this interval. The RAISE6SEC price is $180/MWh, reflecting the scarcity. The energy price is $245/MWh, slightly elevated because some generators have been diverted from energy to FCAS.

Now experiment: change Mannum's offer to 100 MW energy, 0 MW FCAS. Re-run. The energy price drops slightly (more supply), the RAISE6SEC price rises (less supply), and Mannum earns more from energy but nothing from FCAS. Compare the total revenue in each case — this is the co-optimisation tradeoff made transparent.

The nempy model lets you explore how Mannum's bidding strategy interacts with the rest of the SA1 fleet. You can observe how adding Mannum's FCAS offer displaces more expensive FCAS from gas turbines, reducing the FCAS clearing price. You can see how withdrawing Mannum's energy offer tightens the energy supply, raising the energy price. These are the general equilibrium effects that a battery operator must anticipate when designing a bidding strategy.

---

## Causer-Pays: The Cost of Regulation

### How Causer-Pays Works

Providing regulation FCAS generates revenue, but **consuming** regulation imposes a cost. AEMO allocates the cost of regulation services to participants through the **causer-pays** mechanism, which assigns costs based on how much each participant's metering deviates from their dispatch targets — i.e., how much each participant contributes to the need for regulation in the first place.

<div class="definition-box">
<strong>Causer-pays:</strong> AEMO's mechanism for allocating the cost of regulation FCAS to market participants. Each participant's contribution to system frequency deviations is measured by comparing their actual metered output to their dispatch targets over four-second intervals. Participants whose deviations consistently move frequency away from 50 Hz (anti-correlated with the system need) pay a larger share of the total regulation cost. Participants whose deviations help stabilise frequency (correlated with the system need) pay less or nothing. The mechanism creates a financial incentive for participants to follow their dispatch targets accurately.
</div>

The causer-pays calculation works in four-second intervals:

1. AEMO measures the **system frequency error** (actual frequency minus 50 Hz) every four seconds
2. AEMO measures each participant's **metering deviation** (actual output minus dispatch target) every four seconds
3. For each four-second interval, AEMO computes a **contribution factor** (CF) for each participant: the product of the metering deviation and the system frequency error. If a generator is producing more than its target when the frequency is already high (supply exceeds demand), the generator is making the problem worse — its CF is positive. If it is producing more when frequency is low, it is helping — its CF is negative (or at least not adding to cost).
4. The CFs are aggregated over a 28-day period into **causer-pays factors** (CPFs) for each participant
5. The total regulation cost for each interval is allocated proportionally to each participant's CPF

### Why Batteries Are Good (and Bad) at Causer-Pays

Batteries have two properties that interact with causer-pays in opposing directions:

**Good:** Batteries can follow dispatch targets precisely. A thermal generator that is dispatched to produce 350 MW might actually produce 345–355 MW due to the inherent imprecision of turbine control. A battery dispatched to 50 MW can produce exactly 50.0 MW, because the power electronics respond within milliseconds and with high precision. Low metering deviation means low causer-pays contributions.

**Bad:** Batteries that provide regulation FCAS are constantly adjusting their output in response to the AGC signal. If the AGC signal is noisy or if the battery's response is slightly delayed, the battery's metering can deviate from its nominal dispatch target in a way that contributes to causer-pays costs. Additionally, batteries that charge during variable renewable generation periods (when wind and solar are fluctuating rapidly) may see their metering deviate from dispatch due to the rapid changes in local grid conditions.

<div class="example-box">
<strong>Causer-pays in practice — the Mannum BESS:</strong> Suppose Mannum's CPF for a given 28-day period is 0.8% — meaning Mannum is responsible for 0.8% of the total regulation cost in SA1. If the total SA1 regulation cost for that period is $2 million, Mannum pays $16,000. This cost is deducted from Mannum's gross revenue. A well-controlled battery might achieve a CPF of 0.3–0.5%; a poorly controlled one might see 1.5–2.0%. The difference — $6,000 vs $40,000 per 28-day period — is meaningful but not dominant relative to the battery's total revenue.
</div>

For the Mannum BESS, causer-pays is a cost to manage, not a cost to fear. The key is precise dispatch-target following, which is a control-system problem rather than a market-strategy problem. The battery's optimiser and its physical control system must be well-calibrated to minimise metering deviations. This is typically handled by the battery management system (BMS) and the energy management system (EMS), not by the trading desk — but the trading desk should monitor the CPF and flag anomalies.

---

## Forecasting FCAS Prices

### The Challenge

Forecasting FCAS prices is fundamentally harder than forecasting energy prices. Energy prices have strong, exploitable patterns — as we developed through Chapters 6, 7, and 8:

- **Daily patterns:** Prices are low at night and midday, high in the morning and evening peaks
- **Seasonal patterns:** Prices are higher in summer (air conditioning) and winter (heating) than in spring and autumn
- **Weather dependence:** Prices rise with temperature extremes and fall with high wind and solar output
- **Autoregressive structure:** Today's prices are correlated with yesterday's prices at the same time

FCAS prices have some of these patterns (FCAS is generally more expensive during peak demand when generators are stressed), but the dominant feature is the **spike process** — rare, large, unpredictable price spikes driven by contingency events. A generator trip is essentially random — it cannot be forecast from weather, demand, or price history. The resulting FCAS price spike is extreme (often hitting the market price cap) and short-lived (typically 1–6 intervals before reserves are replenished).

<div class="key-point">
<strong>Key point:</strong> Energy price forecasting is primarily about predicting the level; FCAS price forecasting is primarily about predicting the probability of a spike. The base FCAS price ($3–$15/MWh) is easy to predict — it does not vary much. The spike ($300–$15,100/MWh) is nearly impossible to predict — it depends on when the next generator trips, which is essentially random. The battery operator's FCAS strategy must account for this spike risk without relying on predicting individual spikes.
</div>

### What Can Be Forecast

While individual FCAS spikes cannot be predicted, the **conditions that make spikes more likely** can be:

**1. Low reserve margin.** When the total available generation capacity is close to the total demand, a single generator trip is more likely to cause a reserve shortfall and an FCAS price spike. Reserve margin is driven by demand (which is forecastable from weather and calendar variables) and available supply (which is partially forecastable from planned outage schedules and renewable generation forecasts).

**2. Low inertia.** When the system inertia is low (high renewable penetration, few synchronous generators online), frequency deviates faster after a disturbance, requiring more fast FCAS. AEMO publishes inertia forecasts as part of the pre-dispatch process.

**3. Interconnector constraints.** When interconnectors are constrained (at their flow limits), regions become electrically isolated. A generator trip in an isolated region cannot be compensated by imports, so the local FCAS requirement increases and the FCAS price rises. Interconnector constraints are partially forecastable from demand and generation patterns.

**4. Generator outage schedules.** AEMO publishes planned outage information through the Medium-Term Projected Assessment of System Adequacy (MT PASA). Periods with many generators on planned outage are more vulnerable to FCAS scarcity if an additional unplanned trip occurs.

**5. Time of day and season.** FCAS requirements and enablement costs both have diurnal patterns — they tend to be higher during peak demand periods when the system is stressed.

### A Practical Forecasting Approach

Given the difficulty of FCAS price forecasting, a pragmatic approach for battery dispatch uses two levels:

**Level 1: Base FCAS price forecast.** Use a simple model — the average FCAS price for each half-hour period, stratified by day type (weekday/weekend) and season. This captures the predictable component (the base price, the diurnal pattern) and ignores the unpredictable component (spikes). This forecast is used for the MPC's FCAS revenue estimate: when the optimiser decides how much headroom to reserve for FCAS, it values that headroom at the base forecast.

**Level 2: Spike probability estimate.** Estimate the probability that the FCAS price exceeds a high threshold (say, $300/MWh) in each interval, based on the forecastable conditions above — reserve margin, inertia, interconnector constraints. This probability can be estimated from historical data: what fraction of intervals with similar conditions had FCAS spikes? This estimate is not used directly in the MPC (which would require a full stochastic FCAS price model) but informs the **static FCAS reservation policy** — a rule-of-thumb that sets a minimum FCAS headroom reservation based on the current spike probability.

<div class="example-box">
<strong>A simple FCAS reservation rule:</strong> The trading desk might set a policy: "If the estimated FCAS spike probability exceeds 5% for the upcoming two hours, reserve at least 20 MW of capacity for RAISE6SEC and RAISE1SEC. If it exceeds 15%, reserve at least 40 MW." This rule is not optimal in a formal sense — it does not maximise expected revenue across all possible FCAS price realisations — but it captures the most important feature of FCAS value: being enabled when the rare spike occurs. The rule can be calibrated by backtesting against historical FCAS price data, sweeping the reservation level and measuring the total (energy + FCAS) revenue.
</div>

### Integrating FCAS into the MPC

The simplest integration of FCAS into the MPC framework from Chapter 9 adds the base FCAS price forecast to the LP objective:

<pre>
Maximise  Σ_t [ p_energy(t) · (discharge(t) - charge(t)) · Δt
              + Σ_s  p_fcas_base(s,t) · fcas(s,t) · Δt ]
</pre>

This LP jointly optimises energy and FCAS, using the base FCAS forecast as the FCAS price. It naturally balances energy and FCAS — when the energy price is high relative to FCAS, the optimiser dispatches more energy; when FCAS is relatively attractive, it reserves more headroom.

The limitation of this approach is that it values FCAS at the base price, ignoring the spike premium. The true expected FCAS revenue includes the spike contribution:

<pre>
E[FCAS revenue] = base_price · (1 - p_spike) + spike_price · p_spike
</pre>

where p_spike is the spike probability and spike_price is the average price conditional on a spike. If p_spike = 3% and spike_price = $2,000/MWh, the spike contribution adds $60/MWh to the expected value — potentially doubling the FCAS value used in the LP.

A more sophisticated approach uses the Level 2 spike probability estimate to compute the expected FCAS price:

<pre>
expected_fcas_price(s,t) = base_price(s,t) · (1 - p_spike(t))
                         + E[price | spike](s,t) · p_spike(t)
</pre>

and substitutes this expected price into the LP. This gives the optimiser a more accurate valuation of FCAS headroom and leads to better allocation between energy and FCAS.

<div class="key-point">
<strong>Key point:</strong> The quality of FCAS price forecasting matters less than the quality of energy price forecasting for overall battery revenue. Energy arbitrage is the dominant revenue stream, and small improvements in energy price forecasting translate directly to higher capture ratios (as shown in Chapter 9). FCAS forecasting mainly affects the allocation of headroom between energy and FCAS — getting this allocation roughly right is sufficient. A simple base-price forecast with a spike probability overlay is adequate for most battery dispatch applications.
</div>

---

## The Capture Ratio with FCAS

### Redefining the Benchmark

In Chapters 5, 9, and 10, we defined the capture ratio (CR) as the ratio of actual arbitrage revenue to perfect-foresight arbitrage revenue. With FCAS included, the capture ratio must be extended:

<pre>
CR_total = (actual energy revenue + actual FCAS revenue)
         / (perfect-foresight energy + FCAS revenue)
</pre>

The denominator — perfect foresight revenue including FCAS — is the revenue a battery would earn if it knew all future energy and FCAS prices perfectly and optimised its energy dispatch and FCAS enablement accordingly. This is computed by solving the co-optimised LP with actual (historical) energy and FCAS prices.

<div class="definition-box">
<strong>Capture ratio with FCAS:</strong> The ratio of total actual revenue (energy arbitrage plus FCAS enablement payments) to total perfect-foresight revenue (the maximum achievable with perfect knowledge of both energy and FCAS prices). This extended capture ratio measures the battery operator's skill at both energy arbitrage and FCAS co-optimisation. A higher capture ratio indicates better forecasting and dispatch strategy across all revenue streams.
</div>

The benchmark from earlier chapters — 0.50 as the bar, 0.65–0.75 as the target — still applies, but now measured against total revenue:

| Capture ratio | Interpretation |
|--------------|----------------|
| < 0.40 | Poor — significant value left on the table in both energy and FCAS |
| 0.50 | Market average — acceptable but not competitive |
| 0.55–0.65 | Good — effective energy arbitrage with basic FCAS participation |
| 0.65–0.75 | Very good — strong energy forecasting with co-optimised FCAS |
| > 0.75 | Excellent — top-tier operator with sophisticated co-optimisation |

Including FCAS in the capture ratio has two effects:

**1. The denominator increases.** Perfect-foresight revenue with FCAS is higher than without, because a perfectly informed battery would offer FCAS during high-FCAS-price intervals and withdraw from FCAS during high-energy-price intervals. The additional FCAS revenue from perfect co-optimisation is typically 25–40% above perfect-foresight energy-only revenue.

**2. The numerator increases too, but by less.** A real battery earns FCAS revenue, but it cannot perfectly predict FCAS prices (especially spikes), so its FCAS co-optimisation is imperfect. The actual FCAS revenue is typically 10–25% of total actual revenue.

The net effect is that the FCAS-inclusive capture ratio is slightly lower than the energy-only capture ratio for the same operator, because perfect FCAS co-optimisation is harder to achieve than perfect energy arbitrage (FCAS spikes are less predictable than energy price patterns). A battery with a 0.70 energy-only CR might have a 0.62–0.68 total CR, depending on how effectively it captures FCAS spikes.

### Benchmarking Against AEMO Pre-dispatch

As established in Chapter 5, AEMO's pre-dispatch forecasts (accessed via NEMSEER) serve as the benchmark for forecast quality. With FCAS included, the benchmark extends to AEMO's pre-dispatch FCAS price forecasts, which are also available through NEMSEER.

AEMO's FCAS price forecasts have a distinctive characteristic: they are reasonably good at predicting the base level but terrible at predicting spikes. This is expected — AEMO's pre-dispatch runs NEMDE with forecast inputs but cannot predict contingency events (generator trips). As a result, the AEMO pre-dispatch FCAS prices are almost always near the base level, and the spikes that actually occur are entirely missed.

A dispatch strategy that uses AEMO pre-dispatch prices for both energy and FCAS will:

- Capture energy arbitrage reasonably well (as shown in Chapter 5)
- Undervalue FCAS headroom (because the pre-dispatch FCAS prices do not include spike risk)
- Reserve too little capacity for FCAS during periods when spikes are more likely
- Miss the FCAS revenue from the rare but valuable spike events

This is the baseline that a more sophisticated FCAS forecasting approach (the base-price plus spike-probability model described above) aims to beat. The improvement comes not from predicting individual spikes but from systematically reserving more FCAS headroom during high-risk periods, capturing more spike revenue when spikes do occur.

<div class="example-box">
<strong>Capture ratio comparison — energy-only vs co-optimised:</strong> Over a one-year backtest for the Mannum BESS in SA1:

<strong>Strategy A (energy-only):</strong>
- Perfect foresight energy revenue: $12.0M
- Actual energy revenue: $8.4M
- FCAS revenue: $0
- Energy-only CR: 0.70
- Total CR (against energy + FCAS perfect foresight of $16.2M): 0.52

<strong>Strategy B (co-optimised, base FCAS forecast):</strong>
- Actual energy revenue: $7.6M (lower due to reduced throughput)
- Actual FCAS revenue: $2.8M
- Total actual revenue: $10.4M
- Total CR (against $16.2M): 0.64

<strong>Strategy C (co-optimised, with spike probability model):</strong>
- Actual energy revenue: $7.2M
- Actual FCAS revenue: $3.4M
- Total actual revenue: $10.6M
- Total CR (against $16.2M): 0.65

The co-optimised strategies earn less from energy (reduced throughput due to FCAS reservation) but more than compensate with FCAS revenue. Strategy C's spike-aware model earns an additional $600K in FCAS revenue compared to Strategy B, primarily from being enabled during the 15–20 highest-value FCAS intervals of the year.
</div>

---

## Worked Example: Clearing One Interval with nempy

### Setup

This example walks through co-optimised clearing for a single five-minute interval: SA1 at 18:15 on 15 January 2026. A gas turbine has just tripped, making FCAS scarce.

The setup in nempy:

1. Load the historical bid stack for SA1 at this interval — all generators' energy offers and FCAS offers as submitted to AEMO
2. Load the demand forecast and FCAS requirements for this interval
3. Add the Mannum battery as a new participant with the following offers:
   - Energy: 70 MW at $0/MWh (price-taking, as introduced in Chapter 3)
   - RAISE1SEC: 15 MW at $0/MWh
   - RAISE6SEC: 15 MW at $0/MWh
   - No other FCAS services offered in this example
4. Set the trapezium constraint: energy dispatch + RAISE1SEC + RAISE6SEC <= 100 MW

Call nempy's dispatch solver. The solver outputs:

**Clearing prices:**
- Energy: $248/MWh
- RAISE1SEC: $245/MWh
- RAISE6SEC: $178/MWh
- Other FCAS services: $5–$20/MWh (not binding)

**Mannum dispatch:**
- Energy: 70 MW (fully dispatched at its offered quantity)
- RAISE1SEC: 15 MW (fully enabled)
- RAISE6SEC: 15 MW (fully enabled)

**Mannum revenue for this 5-minute interval:**
- Energy: $248 · 70 · (5/60) = $1,447
- RAISE1SEC: $245 · 15 · (5/60) = $306
- RAISE6SEC: $178 · 15 · (5/60) = $223
- **Total: $1,976**

Compare with a Mannum offering 100 MW energy only:
- Energy: $243 · 100 · (5/60) = $2,025 (the energy price is slightly lower because more supply is available)
- FCAS: $0
- **Total: $2,025**

In this single interval, the energy-only strategy wins by $49. But consider: the co-optimised strategy preserved 30 MW of headroom that could have been called upon if the frequency dropped further. And over the broader evening — including intervals where the energy price is lower — the co-optimised allocation typically wins.

### Sensitivity Analysis with nempy

The power of nempy is the ability to perturb inputs and observe how the clearing outcome changes. Run the following experiments:

**Experiment 1: Increase Mannum's FCAS offer.** Change Mannum's RAISE6SEC offer from 15 MW to 40 MW (reducing energy to 45 MW). The RAISE6SEC price drops (more supply), the energy price rises slightly (less supply). At what FCAS offer quantity does the total revenue peak? This reveals the optimal allocation for this specific interval.

**Experiment 2: Change Mannum's FCAS offer price.** Instead of bidding FCAS at $0/MWh (price-taking), bid at $100/MWh. If the clearing price is $178/MWh, Mannum is still enabled — the offer price is below the clearing price. But if Mannum bids at $200/MWh, it is not enabled (the offer price exceeds the clearing price). The FCAS market clears without Mannum, and the FCAS price may be higher (if Mannum was a marginal provider). This demonstrates the strategic bidding dimension — a battery can influence the FCAS clearing price through its offer price, not just its offer quantity.

**Experiment 3: Remove the tripped generator.** Load the bid stack with the gas turbine still online (no trip). The FCAS requirements are met easily, RAISE6SEC clears at $5/MWh, and Mannum's FCAS enablement is worth little. The energy price is lower too (more generation available). This confirms that the FCAS value in the base case was driven entirely by the contingency event.

These experiments give the reader an intuitive understanding of how co-optimisation works: **the same battery, the same capacity, in the same market, can optimally allocate itself very differently depending on the relative energy and FCAS prices, which in turn depend on the system state.**

---

## Practical FCAS Bidding for a Battery

### Offer Structure

In the NEM, a battery's FCAS offers are structured similarly to energy offers — a set of price-quantity bands. For each FCAS market, the battery specifies:

1. **Maximum availability** (MW): the most it can provide, subject to the trapezium constraint
2. **Enablement minimum and maximum** (MW): the energy dispatch range within which the FCAS is available
3. **Trapezium parameters:** the low breakpoint, high breakpoint, and enablement max that define the trapezoidal constraint region
4. **Price** ($/MWh): the minimum price at which the battery is willing to provide the service

For a battery like Mannum, the typical offer strategy is:

- **Offer FCAS at $0/MWh (price-taking)** for services where the battery wants to be enabled whenever possible. This ensures enablement — the battery will be dispatched for FCAS as long as the clearing price is positive. The battery still earns the clearing price (not the offer price), so bidding at $0/MWh does not mean earning $0.

- **Offer FCAS at a reservation price** that reflects the opportunity cost of energy. If the energy price is expected to be $150/MWh and the battery must reduce its energy dispatch by 1 MW to provide 1 MW of FCAS, the opportunity cost is $150/MWh. Offering FCAS above $150/MWh ensures the battery is only enabled when the FCAS price exceeds the foregone energy revenue. This is a more sophisticated strategy that requires forecasting the energy price at the time the FCAS offer is submitted (which is before NEMDE clears).

<div class="example-box">
<strong>Dynamic FCAS bidding — aligning offers with energy forecasts:</strong> At 17:00, Mannum's trading system forecasts SA1 energy prices for the next four hours. The forecast suggests a price peak of $200/MWh at 18:30. The system sets Mannum's FCAS offers for each upcoming dispatch interval:

- 17:00–17:30 (energy forecast: $80/MWh): Offer 40 MW RAISE6SEC at $0/MWh. The low opportunity cost means FCAS is attractive.
- 17:30–18:00 (energy forecast: $150/MWh): Offer 25 MW RAISE6SEC at $100/MWh. The higher opportunity cost means FCAS should only be enabled if it pays well.
- 18:00–18:30 (energy forecast: $200/MWh): Offer 20 MW RAISE6SEC at $180/MWh. Reserve most capacity for the energy peak; only provide FCAS if it pays nearly as much.
- 18:30–19:00 (energy forecast: $120/MWh): Offer 30 MW RAISE6SEC at $80/MWh. Moderate opportunity cost.

This dynamic bidding strategy adapts the FCAS offer quantity and price to the expected energy value, ensuring the battery is enabled for FCAS when it is most valuable relative to energy and reserved for energy when energy is most valuable. It requires the battery's bidding system to update offers frequently (every 5 minutes, in line with NEMDE clearing).
</div>

### Stacking FCAS Services

A battery can offer multiple FCAS services simultaneously, as long as the total enablement respects the trapezium constraint. The typical stacking order for raise services (from fastest to slowest):

1. **RAISE1SEC** — highest priority because of limited competition and relatively high prices
2. **RAISE6SEC** — second priority; more competition but still valuable
3. **RAISE60SEC** — lower value; offered if headroom permits after 1-second and 6-second
4. **RAISE5MIN** — lowest value among contingency services; offered with remaining headroom

The cascading property of FCAS services means that enablement in a faster service can satisfy requirements for slower services. A battery enabled for 20 MW of RAISE6SEC also counts toward 20 MW of RAISE60SEC and RAISE5MIN requirements. NEMDE handles this cascading automatically.

For lower services, the stacking works in the same direction (LOWER1SEC > LOWER6SEC > LOWER60SEC > LOWER5MIN), but the economics are different — lower FCAS requires headroom to absorb power (charge), which competes with the battery's charging strategy rather than its discharging strategy.

### The Regulation Decision

Regulation FCAS is a separate strategic decision from contingency FCAS:

- **Pros:** Steady revenue (regulation prices are less spikey than contingency); regulation helps the battery's causer-pays factor (a regulation provider is helping stabilise frequency, which can reduce its CPF)
- **Cons:** Regulation requires continuous response (higher control complexity); the AGC signal causes small, frequent SoC changes that can interfere with the MPC's planned SoC trajectory; the energy throughput from regulation is near-zero but not exactly zero, introducing SoC uncertainty

For the Mannum BESS, offering regulation is typically worthwhile during off-peak periods when energy arbitrage opportunities are limited. The regulation revenue ($5–$15/MWh on enabled capacity) provides a base revenue stream that partially offsets the low energy arbitrage value during these hours. During peak periods, the opportunity cost of reserving capacity for regulation (which requires continuous headroom, unlike contingency which is rarely called) generally favours full energy dispatch or contingency FCAS.

---

## Exercises

### Exercise 1: Sweep the Value of Reserving Headroom for FCAS

**Problem:** For the Mannum battery (100 MW / 200 MWh, 85% round-trip efficiency), compare the total revenue (energy + FCAS) across different FCAS reservation levels over a one-week period.

Assume the following conditions (representative of a summer week in SA1):
- Average energy price during discharge: $120/MWh
- Average energy price during charge: $25/MWh
- Average RAISE6SEC price: $10/MWh
- RAISE6SEC spikes to $2,000/MWh for 3 intervals during the week (15 minutes total)
- The battery does 1.5 cycles per day for energy arbitrage
- Round-trip efficiency: 85%

Compute the weekly total revenue for FCAS reservation levels of 0, 10, 20, 30, 40, and 50 MW (reducing the energy dispatch power proportionally).

<details>
<summary><strong>Worked Solution</strong></summary>

**Step 1: Compute energy-only revenue (0 MW FCAS reservation).**

<pre>
Daily discharged  = 1.5 cycles · 200 MWh = 300 MWh
Daily charged     = 300 / 0.85            = 353 MWh

Daily energy revenue = 300 · $120 - 353 · $25
                     = $36,000 - $8,825
                     = $27,175

Weekly energy revenue = 7 · $27,175 = $190,225
FCAS revenue          = $0
</pre>

**Total: $190,225.**

**Step 2: Model how reservation level R affects energy throughput.**

When R MW is reserved for FCAS, discharge power drops to (100 - R) MW. Charging is unaffected. Assume the battery discharges at reduced power for the same duration, so daily throughput scales proportionally:

<pre>
discharged = 300 · (100 - R) / 100  MWh
</pre>

**Step 3: Compute FCAS revenue as a function of R.**

Enabled hours per week: 1.5 cycles · 2 hours discharge = 3 hours/day = 21 hours/week. Assume the battery captures 2 of the 3 spike intervals.

<pre>
Base FCAS  = R · $10 · 21 hours            = $210 · R
Spike FCAS = R · $2,000 · (2 · 5/60) hours = $333 · R
Total FCAS = $543 · R
</pre>

**Step 4: Tabulate results.**

| R (MW) | Discharge power | Weekly energy revenue | Weekly FCAS revenue | Total |
|--------|----------------|----------------------|--------------------:|------:|
| 0 | 100 MW | $190,225 | $0 | $190,225 |
| 10 | 90 MW | $171,203 | $5,430 | $176,633 |
| 20 | 80 MW | $152,180 | $10,860 | $163,040 |
| 30 | 70 MW | $133,158 | $16,290 | $149,448 |
| 40 | 60 MW | $114,135 | $21,720 | $135,855 |
| 50 | 50 MW | $95,113 | $27,150 | $122,263 |

**Step 5: Interpret.**

Energy-only dispatch wins at every reservation level. The FCAS revenue ($543/MW/week) does not compensate for foregone energy revenue ($1,902/MW/week) because the average RAISE6SEC price ($10/MWh) is far below the energy spread, and spikes are too brief.

**Key insight:** If the average RAISE6SEC price were $35/MWh (high-volatility periods), or spikes were more frequent, the optimal reservation would be nonzero. The crossover is where marginal FCAS revenue per MW equals marginal energy revenue per MW (calculated in Exercise 2).

</details>

### Exercise 2: Find the Break-Even FCAS Price

**Problem:** Using the same battery parameters as Exercise 1 (100 MW / 200 MWh, 85% RTE, average discharge price $120/MWh, average charge price $25/MWh, 1.5 cycles/day), find the break-even average RAISE6SEC price at which reserving the first MW of FCAS becomes profitable.

Ignore spikes for this calculation — assume a flat FCAS price. The break-even price is the FCAS price at which the marginal revenue from FCAS exactly equals the marginal revenue from energy for the last MW of capacity.

<details>
<summary><strong>Worked Solution</strong></summary>

**Step 1: Determine discharge hours per week.**

<pre>
Discharge time per cycle = 200 MWh / 100 MW = 2 hours
Discharge hours per week = 1.5 cycles/day · 2 hours · 7 days = 21 hours
</pre>

**Step 2: Compute marginal energy revenue per MW.**

Assume the battery is power-constrained (wants to discharge all 200 MWh in the 2-hour peak). The marginal value of 1 MW of discharge capacity:

<pre>
Marginal gross revenue  = 1 MW · 2 hours · $120/MWh = $240 per cycle
Marginal charge cost    = 1 MW · 2 hours / 0.85 · $25/MWh = $58.82 per cycle
Net marginal per cycle  = $240 - $58.82 = $181.18

Weekly marginal energy  = $181.18 · 1.5 cycles/day · 7 days
                        = $1,902.4 per MW per week
</pre>

**Step 3: Compute marginal FCAS revenue per MW.**

Each reserved MW earns the FCAS price for the 21 enabled hours:

<pre>
Marginal FCAS revenue = FCAS_price · 21 hours per week
</pre>

**Step 4: Solve for break-even.**

<pre>
FCAS_price · 21 = $1,902.4
FCAS_price      = $1,902.4 / 21
               = $90.6/MWh
</pre>

**The break-even RAISE6SEC price is approximately $91/MWh.**

**Step 5: Interpret.**

This is a high bar — the average RAISE6SEC price is typically $5-$15/MWh. But this uses the *average* energy price. During low-energy-price intervals ($40-$60/MWh), the break-even drops substantially.

**Key insight:** Co-optimisation adds value not because the average FCAS price exceeds the break-even, but because there are specific intervals where the FCAS price exceeds the energy opportunity cost *at that moment*. The LP exploits these interval-level opportunities automatically.

</details>

### Exercise 3: Estimate the Annual FCAS Revenue from Spike Capture

**Problem:** The Mannum BESS is enabled for 30 MW of RAISE6SEC for approximately 40% of all dispatch intervals in a year (roughly 42,000 intervals out of 105,120). Historical data shows that RAISE6SEC in SA1 spikes above $1,000/MWh in approximately 0.3% of intervals, with an average spike price of $5,500/MWh. The average non-spike price when enabled is $8/MWh. Estimate the annual RAISE6SEC revenue split between base and spike components.

<details>
<summary><strong>Worked Solution</strong></summary>

**Step 1: Count enabled intervals.**

<pre>
Total intervals/year = 365 · 24 · 12         = 105,120
Enabled intervals    = 0.40 · 105,120         = 42,048
</pre>

**Step 2: Split into spike and non-spike intervals.**

<pre>
Spike intervals (system-wide) = 0.003 · 105,120 = 315/year
Spike intervals captured      = 0.40 · 315       = 126
Non-spike intervals enabled   = 42,048 - 126     = 41,922
</pre>

**Step 3: Compute base revenue.**

<pre>
Base revenue = 30 MW · $8/MWh · 41,922 intervals · (5/60) h/interval
             = 30 · 8 · 41,922 · 0.0833
             = $838,400
</pre>

**Step 4: Compute spike revenue.**

<pre>
Spike revenue = 30 MW · $5,500/MWh · 126 intervals · (5/60) h/interval
              = 30 · 5,500 · 126 · 0.0833
              = $1,732,050
</pre>

**Step 5: Total and split.**

<pre>
Total RAISE6SEC revenue = $838,400 + $1,732,050 = $2,570,450

Revenue split:  Base = 33%,  Spike = 67%
</pre>

**Final answer: $2,570,450 per year, with spikes contributing two-thirds despite occurring in only 0.3% of intervals.**

**Step 6: Interpret.**

Being enabled during spikes matters far more than the base FCAS price. The spike probability model (Level 2) is more valuable than the base price model (Level 1). A battery that reserves more FCAS headroom during spike-prone periods can significantly increase revenue.

Reality check: $2.57M from a single FCAS service at 30 MW is roughly 20-25% of Mannum's total annual revenue. With all FCAS services included, total FCAS revenue could reach $3.5-$5M/year.

</details>

---

## Glossary

| Term | Definition |
|------|-----------|
| **FCAS (Frequency Control Ancillary Services)** | Services procured by AEMO to maintain grid frequency at 50 Hz, including regulation and contingency services |
| **Regulation FCAS** | Continuous frequency correction via AGC signal; raise and lower markets |
| **Contingency FCAS** | Response to large, sudden frequency disturbances; categorised by response speed (1s, 6s, 60s, 5min) |
| **Enablement** | The quantity of FCAS a unit is dispatched to provide; the unit is paid the clearing price on its enabled MW |
| **Trapezium constraint (enablement trapezoid)** | The linear constraint linking energy dispatch and FCAS enablement: the sum cannot exceed physical capacity |
| **Causer-pays** | AEMO's cost-allocation mechanism for regulation FCAS, based on each participant's contribution to frequency deviations |
| **Co-optimisation** | Simultaneous clearing of energy and all FCAS markets in a single LP, minimising total cost |
| **NEMDE** | National Electricity Market Dispatch Engine — AEMO's software that clears the market every five minutes |
| **nempy** | Open-source Python implementation of the NEM dispatch procedure by UNSW-CEEM |
| **Inertia** | Rotational kinetic energy of synchronous generators; determines how quickly frequency changes after a disturbance |
| **AGC (Automatic Generation Control)** | AEMO's control system that sends regulation signals to enabled units every four seconds |
| **Causer-pays factor (CPF)** | Each participant's proportional allocation of regulation costs, based on metering deviations |
| **System frequency** | The 50 Hz oscillation rate of AC power; reflects the instantaneous supply-demand balance |
| **Very-fast FCAS (1-second)** | The newest FCAS markets (October 2023) requiring sub-second response; dominated by batteries |
| **Revenue stack** | The combination of energy arbitrage, FCAS, network support, and capacity revenue available to a battery |
| **Capture ratio (with FCAS)** | Total actual revenue (energy + FCAS) divided by total perfect-foresight revenue |
| **Opportunity cost** | The foregone revenue from the next-best use of capacity; for FCAS, the energy revenue given up by reserving headroom |
| **Shadow price (dual variable)** | The marginal cost of satisfying a constraint by one more unit; in NEMDE, shadow prices become market clearing prices |

## Summary

Frequency control is the power system's immune system — invisible when working, catastrophic when it fails. The NEM maintains 50 Hz through ten FCAS markets (regulation raise and lower, plus contingency raise and lower at four speed tiers: 1-second, 6-second, 60-second, and 5-minute), all co-optimised with energy in a single linear program (NEMDE) every five minutes. For the Mannum BESS, FCAS creates both an opportunity and a constraint: the enablement trapezium means every megawatt reserved for FCAS raise is a megawatt that cannot earn energy arbitrage revenue, and the optimal allocation between energy and FCAS shifts every five minutes with the relative prices. Using nempy to clear historical intervals with different Mannum offers makes this tradeoff concrete and inspectable. FCAS prices are dominated by rare spikes driven by contingency events — the base price is $5–$15/MWh but spike prices can reach $15,100/MWh, and these brief spikes contribute the majority of annual FCAS revenue. Forecasting FCAS prices is harder than forecasting energy prices because the spikes are essentially unpredictable, but forecasting the conditions that make spikes more likely (low reserves, low inertia, interconnector constraints) enables a spike-probability model that improves FCAS headroom allocation. The capture ratio, extended to include FCAS revenue, remains the central performance metric — 0.50 as the market-average bar, 0.65–0.75 as the target — now benchmarked against AEMO pre-dispatch via NEMSEER for both energy and FCAS prices. Co-optimised dispatch typically yields 15–30% higher total revenue than energy-only dispatch, not because the average FCAS price is high, but because the LP exploits interval-level opportunities where FCAS value exceeds the energy opportunity cost.
