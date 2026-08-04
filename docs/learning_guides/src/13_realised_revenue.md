# 13. Realised Revenue: From Idealised to Actual

## The Gap Between Model and Reality

The dispatch LP from Chapter 5, and its probabilistic extensions in Chapter 9, optimise an idealised battery. The LP assumes that one megawatt-hour discharged earns exactly the regional reference price, that the battery never degrades, that it can ramp instantaneously, and that there are no parasitic loads. None of these are true.

This chapter quantifies the gap between idealised LP revenue and the revenue that actually lands in the bank account. Each real-world effect acts as a **haircut** — a percentage reduction from the idealised number. The haircuts are multiplicative, so they compound: three independent 5% haircuts reduce revenue by about 14%, not 15%. Understanding and minimising these haircuts is what separates a competent trading desk from a naive one.

<div class="key-point">
<strong>Why this matters commercially:</strong> In Chapter 9, we established a capture ratio of approximately 0.70 for a well-tuned GBT model with chance-constrained dispatch. That number was computed against perfect-foresight revenue using the idealised LP. The <em>realised</em> capture ratio — accounting for loss factors, degradation, operational constraints, and auxiliary load — is lower. The question is: how much lower, and which haircuts are worth engineering around?
</div>

The approach is deliberately simple. Each haircut is introduced as a parameter modification to the LP from Chapter 5. We apply them cumulatively, building a **revenue waterfall** from idealised to realised. At the end, we validate the LP's simplified dispatch against nempy — the open-source reference implementation of the NEM dispatch engine — to check that our approximations are reasonable.

---

## Marginal Loss Factors

### What They Are

Electricity transmitted over long distances loses energy to resistive heating in the wires. In the NEM, these losses are not spread equally across all participants — they are allocated through **marginal loss factors (MLFs)**, which scale the regional reference price at each connection point.

<div class="definition-box">
<strong>Marginal loss factor (MLF):</strong> A multiplier applied to the regional reference price (RRP) to determine the price a generator or load actually receives or pays at its connection point. An MLF of 0.98 means the unit receives 98% of the RRP for energy exported and pays 98% of the RRP for energy imported. MLFs are set annually by AEMO and published before the start of each financial year. They reflect the marginal change in total system losses caused by an incremental increase in generation or load at that connection point.
</div>

The **connection-point price** — what the unit is actually paid — is:

<div class="equation">

connection_point_price_t = MLF × RRP_t

</div>

For a battery that both imports (charges) and exports (discharges) at the same connection point, the MLF applies symmetrically: charging costs MLF × RRP and discharging earns MLF × RRP. This means the MLF cancels out of the **spread** calculation if and only if the MLF is constant. Since MLFs are fixed for a financial year, the cancellation is exact within that year. The haircut comes from the fact that revenue is measured in absolute dollars, not spreads — a unit with MLF = 0.95 earns 5% less revenue per MWh discharged than a unit at the regional reference node.

<div class="definition-box">
<strong>Connection-point price:</strong> The effective price at a specific connection point on the transmission network, computed as the product of the regional reference price and the marginal loss factor at that point. It is the price the participant actually transacts at — generators are paid the connection-point price, and loads pay the connection-point price. Two generators in the same region but at different connection points will receive different prices for the same MWh of output if their MLFs differ.
</div>

### How AEMO Computes MLFs

AEMO sets MLFs annually using a power system model that simulates load flow across the entire NEM transmission network. The procedure, in simplified terms:

1. **Model the network** at a representative set of operating conditions (demand levels, generation patterns, interconnector flows).
2. **For each connection point**, compute the marginal change in total system losses caused by a 1 MW increase in generation (or load) at that point.
3. **Average** these marginal loss sensitivities across the operating conditions, weighting by frequency of occurrence.
4. **Publish** the resulting MLF for each connection point, effective from 1 July for one financial year.

The key subtlety: MLFs reflect the **marginal** impact on system losses, not the average. A generator located at the end of a long, thin transmission line imposes higher marginal losses than one located near a major load centre, even if the average losses on that line are modest.

### The Mannum Example

The Mannum BESS is connected to the SA1 region at its Mannum connection point, located on the eastern edge of South Australia near the SA–VIC interconnector. This is a relatively favourable location for loss factors:

- **Proximity to the interconnector** means that power exported from Mannum to the rest of the NEM travels a shorter distance through the SA transmission network than power from remote wind farms in the mid-north or far west.
- **Battery symmetry** helps: the MLF applies equally to charge and discharge, so the haircut on the spread is minimal.

Typical MLFs for SA1 connection points (illustrative, as AEMO publishes actuals annually):

| Connection point type | Typical MLF range | Revenue impact |
|---|---|---|
| Adelaide metro (load centre) | 0.99–1.01 | Negligible |
| Mannum (near interconnector) | 0.97–0.99 | 1–3% haircut |
| Mid-north wind farms | 0.90–0.95 | 5–10% haircut |
| Far-west solar/wind | 0.85–0.92 | 8–15% haircut |

<div class="example-box">
<strong>Mannum's MLF haircut:</strong> Assume Mannum's MLF is 0.98 for the current financial year. The idealised LP assumes MLF = 1.00. Revenue from the LP was $8.4M (Chapter 9's worked example: $12M perfect foresight × 0.70 capture ratio). Adjusting for the MLF: $8.4M × 0.98 = $8.23M. The MLF haircut is approximately <strong>$170K/year, or about 2%</strong>. This is modest — Mannum's location near the interconnector is one of its advantages.
</div>

### Why Remote Renewables Have Falling MLFs

Remote wind and solar farms face a structural MLF problem. As more generation is added at a remote connection point:

1. **More power flows through the same transmission lines**, increasing resistive losses (losses scale with the square of current).
2. **The marginal impact of additional generation increases**, because each new MW adds to an already-congested corridor.
3. **The MLF falls year on year**, reducing revenue for all generators at that point.

This creates a negative feedback loop: the first wind farm at a remote site gets a decent MLF (say 0.95), attracting more investment. The second and third farms drive the MLF down to 0.90 or below, reducing revenue for everyone — including the original farm. This is sometimes called the **MLF death spiral** and is a significant risk for renewable project finance.

<div class="key-point">
<strong>Siting lesson:</strong> For a battery, the MLF story is simpler than for a generator. Because the MLF applies symmetrically to charge and discharge, the absolute level matters less than the year-on-year stability. A battery at Mannum with MLF = 0.98 is in a good position: the MLF is close to unity and unlikely to decline significantly (battery operation does not structurally increase losses the way a new wind farm does). Chapter 5's LP should incorporate the MLF by scaling the price vector: price_effective_t = MLF × price_t.
</div>

### Modifying the LP for MLFs

The modification to the Chapter 5 LP is trivial. Replace the price vector with the connection-point price:

<div class="equation">

Maximise Σ_{t=1}^{T} MLF × price_t × (discharge_t − charge_t) × Δt

</div>

Since MLF is a constant (within a financial year), it factors out of the optimisation and does not change the optimal dispatch schedule. The revenue is simply scaled by MLF. This is why the MLF is a pure haircut: it does not change *what* the battery does, only *how much* it earns.

---

## Network Charges and the True Cost of Connection

The MLF adjusts the energy price, but it is not the only cost of being connected to the grid. Network charges — the fees paid to the transmission and distribution network operators for using their infrastructure — represent an additional fixed or semi-fixed cost.

### Transmission Use of System (TUOS) Charges

Generators in the NEM pay **TUOS charges** to the Transmission Network Service Provider (TNSP). These charges cover the cost of building and maintaining the transmission network. For a battery:

- **Locational component**: based on the connection point and the capacity of the battery. Reflects the cost of the transmission infrastructure used.
- **Common service component**: a shared cost across all connected generators.

For a 100 MW battery like Mannum, TUOS charges are typically in the range of $0.5M–$1.5M per year, depending on the TNSP's pricing methodology and the specifics of the connection agreement. These are largely fixed costs — they do not vary with dispatch — so they reduce net revenue without affecting optimal dispatch decisions.

### Connection Agreement Costs

Beyond TUOS, the connection agreement itself may impose:

- **Generator performance standards**: requirements for power quality, fault ride-through, and voltage control that may require additional equipment.
- **Metering costs**: NEM-compliant revenue metering at the connection point.
- **Network support obligations**: in some cases, the connection agreement requires the battery to provide specific services during network stress events.

These costs are project-specific and typically negotiated during the connection process. For modelling purposes, they are treated as fixed annual costs subtracted from gross revenue.

<div class="key-point">
<strong>For the revenue waterfall:</strong> Network charges are a fixed-cost haircut, not a variable one. They reduce annual net revenue by a roughly constant amount regardless of dispatch strategy. For Mannum, assume approximately $1M/year in combined TUOS and connection costs. This haircut does not interact with forecast quality or dispatch optimisation — it is a constant deduction.
</div>

---

## Degradation as a Variable Cost

### From Hard Constraint to Economic Cost

In Chapter 5, we modelled degradation as a hard cycle-count constraint: the LP was not allowed to exceed a fixed number of full equivalent cycles per day. Chapter 9 noted that real degradation is more nuanced. In this section, we model degradation as a **variable cost** — each MWh of throughput has an implicit cost equal to the value of the battery capacity consumed.

<div class="definition-box">
<strong>Throughput:</strong> The total energy processed by the battery, measured in MWh. One full cycle of a 200 MWh battery produces 200 MWh of throughput (discharge only — charging is implicitly counted through the round-trip efficiency). Total lifetime throughput is a useful proxy for total degradation, though the relationship between throughput and degradation depends on how the cycles are performed.
</div>

<div class="definition-box">
<strong>State of health (SoH):</strong> The current usable capacity of the battery as a fraction of its original nameplate capacity. A battery that started at 200 MWh and now has 180 MWh of usable capacity has a state of health of 90%. SoH declines over time due to cycling degradation and calendar ageing. Most battery warranties guarantee a minimum SoH (e.g., 70%) after a specified number of years or cycles.
</div>

### Degradation Mechanisms for LFP

The Mannum BESS uses **lithium iron phosphate (LFP)** cells. LFP has different degradation characteristics from the more common nickel-manganese-cobalt (NMC) chemistry used in many consumer electronics:

| Factor | LFP | NMC |
|---|---|---|
| **Cycle life** | 4,000–8,000 cycles | 1,500–3,000 cycles |
| **DoD sensitivity** | Low | High |
| **Temperature** | Moderate (>35°C risk) | High |
| **C-rate** | Moderate (>1C risk) | Similar |
| **Calendar ageing** | Low | Moderate |
| **Cost** | Lower, longer life | Higher, shorter life |

<div class="definition-box">
<strong>Depth of discharge (DoD):</strong> The fraction of the battery's capacity used in a single cycle. A full cycle (0% to 100% to 0%) has a DoD of 100%. A half cycle (50% to 100% to 50%) has a DoD of 50%. For NMC batteries, degradation per cycle increases sharply with DoD. For LFP, the relationship is flatter — deep cycles are only slightly more damaging per unit of throughput than shallow ones.
</div>

<div class="definition-box">
<strong>Calendar ageing:</strong> The degradation of battery capacity that occurs over time regardless of cycling, driven by chemical side reactions within the cells. Calendar ageing depends primarily on temperature and state of charge — batteries age faster when hot and when stored at very high or very low SoC. For LFP, calendar ageing is relatively mild compared to NMC, but it is not zero: a battery that sits idle for 20 years will still lose capacity.
</div>

For Mannum's LFP chemistry, the key degradation drivers in order of importance are:

1. **Throughput** (total energy cycled): The dominant factor. Each MWh of throughput consumes a small, roughly constant fraction of the battery's lifetime capacity.
2. **Temperature**: South Australia's climate means Mannum experiences hot summer days (35–45°C ambient), which accelerate degradation. Thermal management systems mitigate this but add to auxiliary load (see below).
3. **C-rate**: At 100 MW / 200 MWh, Mannum operates at 0.5C at full power — well within the comfortable range for LFP. Degradation from C-rate is minimal at this ratio.
4. **Calendar ageing**: LFP's strong suit. Calendar ageing contributes a modest background degradation rate of roughly 1–2% per year regardless of cycling.

### Computing the Degradation Cost per MWh

The degradation cost converts battery wear into a dollar value per MWh of throughput:

<div class="equation">

degradation_cost = (replacement_cost × capacity_fraction_per_MWh) / (1 − residual_value_fraction)

</div>

For Mannum's LFP system:

- **Replacement cost**: approximately $250/kWh for cell replacement (not the full system — balance of plant, inverters, and switchgear are reused). For 200 MWh: $50M.
- **Cycle life to 80% SoH**: approximately 6,000 full equivalent cycles.
- **Capacity consumed per cycle**: (100% − 80%) / 6,000 = 0.0033% per cycle.
- **Throughput per cycle**: 200 MWh.
- **Capacity consumed per MWh of throughput**: 0.0033% / 200 = 0.0000167% per MWh.
- **Dollar cost per MWh**: $50M × 0.0000167% = approximately **$8.3/MWh**.

<div class="example-box">
<strong>Worked example — degradation cost in context:</strong> The average arbitrage spread in SA1 (Chapter 2) is approximately $80–$120/MWh for the peak-to-trough differential. After round-trip efficiency losses (~10%), the net spread is $72–$108/MWh. The degradation cost of $8.3/MWh represents about 8–12% of the net spread. This is meaningful but not dominant — most cycles are still profitable after accounting for degradation. However, shallow cycles that capture small spreads (say $20–$30/MWh) may not be worth the wear.
</div>

### Incorporating Degradation into the LP

To include degradation as a variable cost, modify the LP objective:

<div class="equation">

Maximise Σ_{t=1}^{T} [price_t × (discharge_t − charge_t) − c_deg × discharge_t] × Δt

</div>

where c_deg is the degradation cost per MWh of discharge (approximately $8.3/MWh for Mannum). This is equivalent to reducing the effective discharge price by c_deg. The LP will now avoid shallow cycles where the price spread is insufficient to cover both efficiency losses and degradation costs.

The **breakeven spread** with degradation is:

<div class="equation">

breakeven_spread = (charge_price / η) + c_deg

</div>

where η is round-trip efficiency. For a charge price of $30/MWh, η = 0.90, and c_deg = $8.3/MWh:

breakeven_spread = ($30 / 0.90) + $8.3 = $33.3 + $8.3 = **$41.6/MWh**

Without degradation cost, the breakeven was $33.3/MWh. Degradation raises the bar by about 25%.

<div class="key-point">
<strong>LFP's advantage:</strong> Mannum's LFP chemistry has roughly half the degradation cost per MWh of an equivalent NMC system (which might see $15–$20/MWh degradation cost due to shorter cycle life). This lower degradation cost means Mannum can profitably capture smaller spreads and run more cycles per day than an NMC battery in the same location. Over a 20-year asset life, this difference compounds significantly.
</div>

### Augmentation

As the battery degrades below its nameplate capacity, the operator faces a choice: accept the reduced revenue from lower capacity, or **augment** the battery by adding new cells.

<div class="definition-box">
<strong>Augmentation:</strong> The process of adding new battery cells to an existing system to restore or maintain its nameplate capacity. Augmentation is typically planned in advance — the original system design may include space and electrical capacity for additional cell racks. The decision of when to augment depends on the tradeoff between the cost of new cells and the revenue lost from reduced capacity. LFP's longer cycle life means augmentation can be deferred further than with NMC, improving the asset's net present value.
</div>

A typical augmentation schedule for a 20-year LFP asset:

| Year | Estimated SoH | Action | Incremental cost |
|---|---|---|---|
| 0 | 100% | Initial commissioning | — |
| 8–10 | ~85% | First augmentation (~15% capacity added) | ~$7.5M |
| 14–16 | ~85% (post-augmentation SoH) | Second augmentation (~15%) | ~$5M (cell costs declining) |
| 20 | ~80% | End of warranted life | — |

The declining augmentation cost reflects expected reductions in LFP cell prices over time. In the LP framework, augmentation is modelled as a periodic step-change in E_max rather than a continuous process.

---

## Operational Constraints

### Ramp Limits

The idealised LP assumes the battery can go from full charge to full discharge instantaneously. Real batteries have **ramp limits** — the maximum rate at which power output can change between consecutive dispatch intervals.

<div class="definition-box">
<strong>Ramp limit:</strong> The maximum rate of change of power output, typically expressed in MW per minute or MW per 5-minute dispatch interval. A ramp limit of 50 MW/min means the battery cannot increase or decrease its output by more than 50 MW in one minute. Grid-scale batteries generally have fast ramp rates compared to thermal generators, but power electronics, transformer, and protection system limits impose practical constraints.
</div>

For the Mannum BESS (100 MW), the practical ramp rate is typically very fast — lithium-ion batteries can ramp from zero to full power in under a second. The binding ramp constraint is usually not the battery cells themselves but:

- **Power electronics**: Inverter ramp rates may be limited to prevent power quality issues.
- **AEMO dispatch conformance**: AEMO's dispatch engine assumes generators ramp linearly between dispatch targets over each 5-minute interval. The battery must conform to this ramp profile.
- **Connection agreement**: The network connection agreement may specify maximum ramp rates to prevent voltage disturbances.

For a 100 MW battery with a 5-minute dispatch interval, the ramp constraint is rarely binding — the battery can typically go from −100 MW (full charge) to +100 MW (full discharge) within a single interval. However, when modelling at the trading-interval level (30 minutes), ramp limits are essentially irrelevant.

To include ramp limits in the LP:

<div class="equation">

|power_t − power_{t-1}| ≤ ramp_max × Δt

where power_t = discharge_t − charge_t

</div>

### Minimum Up/Down Times

Some battery systems have **minimum up/down time** constraints — the battery must remain in a given state (charging, discharging, or idle) for at least a minimum number of consecutive intervals before switching. These constraints arise from:

- **Inverter cycling limits**: Frequent mode switching between charge and discharge can stress power electronics.
- **Thermal management**: Rapid cycling generates heat that the cooling system may not handle.

For modern LFP batteries like Mannum, minimum up/down times are typically short (one or two 5-minute intervals) and do not materially constrain the 30-minute trading-interval LP. In the LP, they are modelled as:

<div class="equation">

If the battery switches from charge to discharge (or vice versa) at time t, it must remain in the new state for at least τ_min intervals.

</div>

This constraint requires binary variables, turning the LP into a mixed-integer program (MIP). For typical NEM batteries with short minimum times, the MIP solves quickly and the revenue impact is small (less than 1%).

### Auxiliary Load

<div class="definition-box">
<strong>Auxiliary load:</strong> The electricity consumed by the battery system's own support equipment — cooling systems (HVAC, liquid cooling), battery management systems (BMS), fire suppression monitoring, site lighting, control systems, and communications. Auxiliary load is a parasitic draw that reduces net energy throughput. It is typically 1–3% of nameplate power capacity, but increases during hot weather when cooling demand rises.
</div>

For the Mannum BESS in South Australia:

- **Base auxiliary load**: approximately 0.5–1 MW (0.5–1% of 100 MW capacity) for BMS, monitoring, and standby systems.
- **Cooling load during operation**: additional 1–2 MW during sustained charge/discharge, more during summer heat.
- **Peak auxiliary load on hot days**: can reach 3–4 MW when thermal management works hardest to keep cells below degradation-accelerating temperatures.

In the LP, auxiliary load is modelled as a reduction in net power:

<div class="equation">

net_discharge_t = discharge_t − P_aux
net_charge_t = charge_t + P_aux

</div>

The auxiliary load makes both charging and discharging less efficient: the battery must charge more than the LP assumes (to power auxiliary systems while charging) and delivers less than the LP assumes when discharging. The annual cost of auxiliary load for Mannum is approximately $0.3–$0.5M (purchased at the connection-point price, which varies with time of day).

### Capacity Fade Over the Asset Life

As the battery ages, its usable capacity declines (state of health drops). This means the E_max parameter in the LP is not constant over the asset life — it decreases by roughly 2–3% per year for LFP under typical cycling patterns (before augmentation).

The revenue impact of capacity fade is straightforward: a battery with 10% less capacity earns roughly 10% less revenue (the relationship is approximately linear for small capacity reductions, because the dispatch pattern remains similar — the battery just has less energy to deploy).

In a multi-year financial model, capacity fade is modelled as a declining E_max trajectory with step increases at augmentation events.

---

## The Revenue Waterfall

### Building the Waterfall Step by Step

The revenue waterfall starts with idealised LP revenue and applies each haircut in sequence. The order of application does not matter for multiplicative haircuts, but presenting them from largest to smallest makes the chart easier to read.

<div class="example-box">
<strong>Worked example — Mannum BESS revenue waterfall:</strong>

Starting point: Chapter 9's worked example. 100 MW / 200 MWh LFP battery in SA1, GBT model with QRA combination and chance-constrained dispatch.

<strong>Step 0 — Perfect foresight revenue:</strong> $12.0M/year (from Chapter 9).

<strong>Step 1 — Forecast imperfection (capture ratio = 0.70):</strong> $12.0M × 0.70 = $8.40M. Haircut: $3.60M (30%).

<strong>Step 2 — MLF (0.98):</strong> $8.40M × 0.98 = $8.23M. Haircut: $0.17M (2%).

<strong>Step 3 — Degradation cost ($8.3/MWh, ~1.5 cycles/day):</strong> Throughput per year: 200 MWh × 1.5 cycles × 365 days = 109,500 MWh. Degradation cost: 109,500 × $8.3 = $0.91M. Net: $8.23M − $0.91M = $7.32M. Haircut: $0.91M (11%).

<strong>Step 4 — Auxiliary load (~1.5 MW average, priced at ~$60/MWh average):</strong> Annual cost: 1.5 MW × 8,760 h × $60/MWh = $0.79M. Net: $7.32M − $0.79M = $6.53M. Haircut: $0.79M (9%).

<strong>Step 5 — Network charges (TUOS + connection):</strong> $1.00M/year. Net: $6.53M − $1.00M = $5.53M. Haircut: $1.00M (12%).

<strong>Step 6 — Ramp limits and min up/down:</strong> Negligible for a 100 MW LFP battery at 30-minute resolution. Haircut: ~$0.05M (<1%).

<strong>Realised revenue: $5.48M/year.</strong>

Total haircuts from idealised LP revenue: $8.40M → $5.48M = <strong>35% reduction</strong>.

Realised capture ratio (against perfect foresight): $5.48M / $12.0M = <strong>0.46</strong>.
</div>

![Revenue waterfall](figures/13_revenue_waterfall.png)

<p class="figure-caption">Figure 13.1 — Revenue waterfall from idealised LP revenue to realised revenue for the Mannum BESS. Each bar shows the magnitude and percentage of a haircut. Forecast imperfection (the gap from perfect foresight) is the largest single factor, followed by network charges, degradation, and auxiliary load. The MLF haircut is modest due to Mannum's favourable location.</p>

<div class="key-point">
<strong>The 0.50 benchmark, revisited:</strong> In earlier chapters, we used a capture ratio of ~0.50 as the bar for a competent dispatch system (computed against idealised LP revenue). After applying real-world haircuts, the realised capture ratio against perfect foresight drops to approximately 0.46. This is the number that matters for financial modelling and investment decisions. The gap between the idealised 0.70 and the realised 0.46 represents approximately $2.9M/year in friction costs — a significant portion of the battery's gross revenue.
</div>

### Sensitivity of the Waterfall

The waterfall is not fixed — it depends on market conditions and operating parameters. Key sensitivities:

| Parameter | If it changes... | Revenue impact |
|---|---|---|
| MLF drops from 0.98 to 0.93 | 5 percentage points of additional haircut | −$0.42M/year |
| Degradation doubles (NMC instead of LFP) | Degradation cost rises to ~$16.6/MWh | −$0.91M/year |
| Average price level doubles | All haircuts remain proportional, but auxiliary load cost doubles | +$5M (approximate) |
| Battery ages to 85% SoH | Capacity fade reduces revenue by ~15% | −$0.8M/year |

---

## Using nempy to Validate the LP

### Why Validate?

The dispatch LP from Chapter 5 is a simplification. It assumes a perfectly competitive market with uniform pricing, no network constraints, and linear cost curves. The real NEM dispatch engine (NEMDE) is considerably more complex. **nempy** — the open-source Python implementation from UNSW-CEEM — replicates NEMDE's behaviour and can be used to check whether the LP's simplifications introduce systematic biases.

<div class="definition-box">
<strong>nempy:</strong> An open-source Python package developed by UNSW's Collaboration on Energy and Environmental Markets (CEEM) that implements the NEM dispatch procedure. nempy models the full dispatch engine, including regional pricing, interconnector flows, constraint equations, loss factors, and FCAS markets. It can be used to simulate dispatch outcomes for historical or hypothetical conditions, providing a validation layer for simplified models like our dispatch LP.
</div>

### What to Compare

The validation exercise runs the same price trajectory through both the simplified LP and nempy, comparing:

1. **Dispatch schedule**: Does the LP's optimal charge/discharge schedule match what nempy would produce if the battery were a price-taking participant?
2. **Revenue**: Does the LP's computed revenue match nempy's settlement calculation (which includes MLFs and the full pricing model)?
3. **Price formation effects**: For a large battery (100 MW is material in SA1), does the battery's own dispatch affect the dispatch price? The LP treats the battery as a price-taker; nempy can reveal whether this assumption holds.

### Running the Validation

The workflow:

1. **Take the LP's optimal dispatch schedule** for a representative week (e.g., a week with both calm and volatile prices).
2. **Feed the schedule as fixed bids into nempy**, along with all other generators' actual bids from the AEMO bidding data.
3. **Compare nempy's computed dispatch prices** with the actual historical prices used by the LP.
4. **Compare nempy's settlement revenue** with the LP's computed revenue.

<div class="example-box">
<strong>Typical findings from nempy validation:</strong>

<strong>Agreement:</strong> For most intervals (90%+), the LP's dispatch and nempy agree closely. The battery charges during the same low-price periods and discharges during the same high-price periods. Revenue matches within 1–2%.

<strong>Discrepancies:</strong> The LP and nempy diverge in two specific situations:

1. <strong>Network constraints:</strong> When a transmission constraint binds (e.g., the Heywood interconnector between SA and VIC reaches its limit), nempy produces a different dispatch outcome because the constraint limits the battery's ability to export. The LP, which ignores network constraints, assumes the battery can always export at its full capacity. This can cause the LP to overestimate revenue by 2–5% during constrained periods.

2. <strong>Price-maker effects:</strong> When the battery's 100 MW of discharge materially shifts the supply stack (e.g., during low-demand overnight periods when SA1 demand is 800–1,200 MW), the act of discharging pushes prices down, reducing the revenue earned. The LP assumes price-taking behaviour and misses this effect. For a 100 MW battery in SA1, this "self-cannibalisation" effect is typically 1–3% of annual revenue.
</div>

### The FCAS Trapezium

nempy also models the **FCAS trapezium** — the relationship between a generator's energy dispatch and its available FCAS capacity. A battery that is dispatched at full power for energy has no headroom to provide FCAS (raise services require the ability to increase output; lower services require the ability to decrease output). The trapezium constraint means that energy dispatch and FCAS provision are **jointly optimised** in the real dispatch engine.

The simplified LP from Chapter 5 optimises energy arbitrage alone. nempy reveals the revenue left on the table by ignoring FCAS co-optimisation — and conversely, shows how much FCAS provision might constrain the energy-optimal dispatch. Full co-optimisation of energy and FCAS is beyond the scope of this guide, but the nempy validation quantifies the gap and motivates the extension.

<div class="key-point">
<strong>The LP is good enough for strategic analysis.</strong> nempy validation typically shows that the simplified LP captures 95–98% of the realised energy arbitrage revenue. The 2–5% discrepancy comes from network constraints and price-maker effects, which are second-order for a well-sited battery like Mannum. For investment-grade financial modelling, the LP is a reasonable first pass; for real-time trading decisions, a dispatch engine that respects network constraints (or a nempy-based simulation) is preferred.
</div>

---

## Exercises

### Exercise 1: Degradation Breakeven Spread

**Question:** For the Mannum BESS (100 MW / 200 MWh, LFP, η = 0.90, c_deg = $8.3/MWh):

1. Find the minimum price spread at which a full DoD cycle covers degradation cost.
2. Compute the same breakeven for an NMC battery (c_deg = $16.6/MWh).
3. Estimate how many profitable cycling days per year the higher NMC cost excludes.

*Hint:* The breakeven spread depends on charge price — assume $30/MWh overnight.

<details><summary><strong>Worked solution</strong></summary>

Revenue from one full cycle (spread = discharge_price − charge_price):

<div class="equation">

revenue = E_max × (discharge_price × η − charge_price)

</div>

Profitable after degradation requires:

<div class="equation">

discharge_price × η − charge_price − c_deg ≥ 0

</div>

Substituting discharge_price = charge_price + spread and rearranging:

<div class="equation">

spread ≥ charge_price × (1 − η) / η + c_deg / η

</div>

With charge_price = $30/MWh:

<pre>
LFP:  spread ≥ 30 × 0.10/0.90 + 8.3/0.90  = 3.3 + 9.2  = <strong>$12.6/MWh</strong>
NMC:  spread ≥ 30 × 0.10/0.90 + 16.6/0.90 = 3.3 + 18.4 = <strong>$21.8/MWh</strong>
</pre>

NMC requires $9.2/MWh more spread. From SA1 historical data, daily max spread exceeds $12.6 on ~340 days/year but exceeds $21.8 on only ~310 days/year.

**NMC loses ~30 profitable cycling days per year** (~$0.5–$1.0M foregone revenue). Not only does each cycle cost more, but fewer cycles are worth doing.

</details>

### Exercise 2: Loss Factor and Siting

**Question:** A remote SA1 site has MLF = 0.91 but saves $2M in capital vs. Mannum (MLF = 0.98). Both are 100 MW / 200 MWh LFP with idealised LP revenue of $8.4M at MLF = 1.00. Compute:

1. Annual revenue difference due to the MLF gap.
2. Years until cumulative revenue loss exceeds the $2M capital saving.
3. Breakeven MLF for the remote site over a 15-year life (no discounting).

*Hint:* Revenue scales linearly with MLF.

<details><summary><strong>Worked solution</strong></summary>

**(a) Annual revenue difference:**

<pre>
Mannum (0.98):  $8.40M × 0.98 = $8.23M
Remote (0.91):  $8.40M × 0.91 = $7.64M
Difference:     <strong>$0.59M/year</strong>
</pre>

**(b) Payback of capital saving:**

<pre>
$2.0M / $0.59M = <strong>3.4 years</strong>
</pre>

Over 15 years the Mannum advantage totals $0.59M × 15 = $8.85M — far exceeding the $2M saving.

**(c) Breakeven MLF over 15 years:**

<div class="equation">

$8.40M × MLF_remote × 15 + $2.0M = $8.40M × 0.98 × 15

MLF_remote = 0.98 − $2.0M / ($8.40M × 15)

MLF_remote = 0.98 − 0.016 = <strong>0.964</strong>

</div>

At 0.91 the remote site falls well short. A 7-point MLF disadvantage costs ~$9M over 15 years — dwarfing any plausible land savings.

<div class="key-point">
<strong>The siting message:</strong> Battery siting is first and foremost an MLF and grid-connection story. A well-connected site near the load centre or interconnector (like Mannum) will almost always outperform a remote site with a lower MLF, even if land and civil costs are significantly cheaper. The exception is a site with a very high MLF <em>and</em> local network support payments that provide additional revenue.
</div>

</details>

---

## Glossary

| Term | Definition |
|------|-----------|
| **Marginal loss factor (MLF)** | Multiplier on RRP at a connection point, reflecting transmission losses |
| **Connection-point price** | The effective price at a connection point: MLF × regional reference price |
| **Throughput** | Total energy processed by the battery, measured in MWh of discharge |
| **Depth of discharge (DoD)** | Fraction of battery capacity used in a single cycle |
| **Calendar ageing** | Capacity loss over time regardless of cycling; driven by temperature and SoC |
| **Augmentation** | Adding new cells to restore or maintain nameplate capacity as the original cells degrade |
| **State of health (SoH)** | Current usable capacity as a fraction of original nameplate capacity |
| **Auxiliary load** | Parasitic consumption by cooling, BMS, and monitoring systems |
| **Ramp limit** | Maximum rate of change of power output between consecutive intervals |
| **Revenue waterfall** | Step-by-step decomposition from idealised revenue to realised revenue, showing each haircut |
| **Degradation cost** | The implicit cost per MWh of throughput, representing the value of battery capacity consumed |
| **Breakeven spread** | Minimum price difference between discharge and charge for a profitable cycle |
| **nempy** | Open-source Python implementation of the NEM dispatch procedure (UNSW-CEEM) |
| **FCAS trapezium** | Joint constraint between energy dispatch and FCAS capacity in the NEM dispatch engine |
| **Self-cannibalisation** | Revenue reduction caused by the battery's own dispatch affecting the market price |

## Summary

Idealised LP revenue overstates what a battery actually earns. The gap is filled by real-world haircuts: marginal loss factors (2% for well-sited Mannum, up to 15% for remote sites), degradation costs ($8.3/MWh for LFP, roughly double for NMC), auxiliary load (1–3% of capacity as parasitic draw), network charges (~$1M/year fixed), and minor operational constraints (ramp limits, minimum up/down times). Applied cumulatively to the Mannum BESS, these haircuts reduce the idealised capture ratio of 0.70 to a realised capture ratio of approximately 0.46 — a 35% reduction in revenue. The revenue waterfall makes each haircut visible and quantifiable. Mannum's LFP chemistry and favourable grid location (near the SA–VIC interconnector, MLF ~0.98) minimise two of the largest potential haircuts: degradation and loss factors. nempy validation confirms that the simplified LP captures 95–98% of realised energy arbitrage revenue; the residual discrepancy comes from network constraints and price-maker effects that the LP ignores. For investment decisions, the lesson is clear: siting (MLF), chemistry (degradation cost), and thermal management (auxiliary load and degradation acceleration) matter as much as forecast quality. Improving the capture ratio from 0.45 to 0.50 is worth roughly $0.6M/year — comparable to the entire degradation cost budget. Every parameter in the waterfall is a lever.
