# 4. Weather and Renewable Generation

## Why Weather Is the Master Variable

In the old world of electricity — grids dominated by coal and gas — weather was a secondary consideration. It affected demand (air conditioning in summer, heating in winter), but the supply side was stable. A coal plant produces the same output whether it is sunny or cloudy, windy or calm. The operator controls the throttle, not the atmosphere.

In the new world — grids where wind and solar provide 50% or more of generation — weather has become the **master variable**. It now drives *both sides* of the electricity equation simultaneously:

- **Demand side:** Temperature determines how much electricity consumers use. A 42°C day in Adelaide triggers massive air conditioning load. A 5°C morning in Melbourne increases heating demand.
- **Supply side:** Solar output depends on sunlight reaching panels. Wind output depends on wind speed at the height of turbine blades. Both are entirely determined by weather.

The combination creates a "double whammy" effect that makes renewable-heavy grids far more volatile than fossil-fuel grids. A hot, calm, hazy day is the worst case: demand is extreme (everyone running air conditioners), wind generation is zero (no wind), and solar generation is reduced (haze and high temperatures lower panel efficiency). An expensive gas peaker must fill the gap, and the price spikes.

<div class="definition-box">
<strong>Dispatchable generation:</strong> Power plants that can be turned on and off, or ramped up and down, at the operator's command. Coal, gas, and hydro plants are dispatchable. The operator decides how much electricity to produce.
</div>

<div class="definition-box">
<strong>Variable renewable energy (VRE):</strong> Generation sources whose output depends on weather conditions and cannot be directly controlled by the operator. Wind and solar are VRE — a wind farm produces whatever the wind allows, not what the operator wants. This fundamental difference from dispatchable generation is why weather forecasting has become central to electricity price forecasting.
</div>

<div class="example-box">
<strong>Real-world example:</strong> On 24 January 2019, South Australia experienced temperatures exceeding 46°C — among the highest recorded in an Australian capital city. Demand surged as every air conditioner in the state ran at full capacity. Meanwhile, wind generation dropped below 200 MW (out of ~2,000 MW installed capacity) as a heat dome brought still conditions. With solar panels also underperforming due to extreme heat (panel efficiency drops above ~25°C), the NEM price in SA1 exceeded $14,000/MWh for multiple intervals. The combination of extreme demand and collapsed renewable supply — both driven by the same weather system — created a "perfect storm" for price spikes.
</div>

<div class="key-point">
<strong>The double-edged sword:</strong> In a renewable-heavy grid, weather simultaneously creates the problem (high demand) and removes the solution (low renewable supply). This coupling means that weather forecast errors affect price forecasts twice — once through the demand channel and once through the supply channel. A model that ignores weather is flying blind.
</div>

## ERA5: A Complete Picture of the Atmosphere

### What Is a Reanalysis?

Weather stations are scattered unevenly across the landscape. Some regions have dense networks of instruments; others — oceans, deserts, remote areas — have almost none. Satellites provide global coverage but measure the top of the atmosphere, not conditions at the surface. Radiosondes (weather balloons) give vertical profiles but are launched only twice daily from a limited number of sites. No single observation system provides a complete, consistent picture of the atmosphere.

A **reanalysis** solves this problem by combining *all* available observations with a physics-based weather model to produce a gridded, gap-free reconstruction of the atmosphere's historical state. The model fills in where observations are missing, and the observations correct where the model is wrong. The result is far more reliable than either source alone.

<div class="definition-box">
<strong>Reanalysis:</strong> A systematic method of producing a comprehensive record of how the atmosphere has changed over time. A weather forecasting model is run over historical periods, continuously correcting itself by "assimilating" (incorporating) all available observations — surface stations, satellites, aircraft reports, ocean buoys, radiosondes. The output is a physically consistent, spatially complete, temporally regular dataset that represents our best estimate of past atmospheric conditions.
</div>

<div class="definition-box">
<strong>Data assimilation:</strong> The mathematical process of combining a model's prediction with real-world observations to produce an improved estimate of the atmospheric state. It uses statistical techniques (typically variants of Kalman filtering or variational methods) to weight the model and observations according to their respective uncertainties. Where observations are dense and reliable, the assimilated state closely follows them; where observations are sparse, the model's physics fills the gaps.
</div>

### ERA5: The Gold Standard

**ERA5** (ECMWF Reanalysis version 5) is produced by the **European Centre for Medium-Range Weather Forecasts (ECMWF)**, widely regarded as the world's leading weather modelling institution. ERA5 is the fifth generation of ECMWF's global reanalysis products and is considered the current gold standard for historical atmospheric data.

ERA5 provides:

| Property | Value | Significance |
|----------|-------|--------------|
| **Spatial resolution** | 0.25° × 0.25° (~31 km at equator, ~25 km at Australian latitudes) | Fine enough to capture regional weather patterns |
| **Temporal resolution** | Hourly | Matches the timescale of electricity market operations |
| **Temporal coverage** | 1940 to near-present (updated with ~5 day lag) | Covers the entire history of the modern NEM (1998–present) |
| **Vertical levels** | 137 model levels from surface to 80 km altitude | Provides wind at multiple heights, including turbine hub height |
| **Variables** | Hundreds, covering temperature, wind, radiation, precipitation, pressure, humidity, and more | All weather drivers of electricity demand and supply |

<div class="definition-box">
<strong>ECMWF (European Centre for Medium-Range Weather Forecasts):</strong> An intergovernmental organisation headquartered in Reading, England, that operates some of the world's most advanced weather forecasting models. Despite being European, ECMWF's global models and reanalysis products are used worldwide, including extensively in Australian energy research. ERA5 is freely available for research and commercial use via the Copernicus Climate Data Store.
</div>

<div class="definition-box">
<strong>Grid point:</strong> One location in a regular spatial grid. ERA5's 0.25° grid means there is a data point every 0.25 degrees of latitude and longitude. Over South Australia, this translates to a grid point roughly every 25 km — close enough to capture the broad weather patterns that drive regional generation, though not fine enough to capture microclimates at individual wind farm sites.
</div>

### Key Variables for Electricity Prices

Of ERA5's hundreds of variables, four are critical for electricity price forecasting:

| Variable | ERA5 short name | Unit | What it measures | Why it matters |
|----------|----------------|------|------------------|----------------|
| **Surface solar radiation downwards** | `ssrd` | J/m^{2} (accumulated) | Total solar energy reaching the ground | Drives solar panel output |
| **2-metre temperature** | `t2m` | Kelvin | Air temperature at 2m above the surface | Drives heating/cooling demand |
| **100m u-component of wind** | `u100` | m/s | East–west wind speed at 100m height | Used to calculate wind speed at turbine hub height |
| **100m v-component of wind** | `v100` | m/s | North–south wind speed at 100m height | Used to calculate wind speed at turbine hub height |

<div class="definition-box">
<strong>u-component and v-component of wind:</strong> Wind is a vector — it has both speed and direction. Rather than storing speed and direction directly, ERA5 decomposes wind into two perpendicular components: the <strong>u-component</strong> (east–west, with positive values meaning wind blowing from west to east) and the <strong>v-component</strong> (north–south, with positive values meaning wind blowing from south to north). The actual wind speed is calculated from these components using the Pythagorean theorem: speed = √(u^{2} + v^{2}). This decomposition is standard in meteorology because it avoids the mathematical difficulties of working with circular (directional) data.
</div>

<div class="definition-box">
<strong>Accumulated variable:</strong> Some ERA5 variables, including surface solar radiation (<code>ssrd</code>), are stored as <em>accumulated</em> values — the total energy received since the start of each forecast period, not the instantaneous rate. To convert to a rate (watts per square metre), you must compute the difference between consecutive time steps and divide by the time interval (3,600 seconds for hourly data). Forgetting this conversion and using the raw accumulated values directly is a common error that produces nonsensical results.
</div>

### Why 100m Wind, Not Surface Wind?

Modern wind turbines are enormous structures. A typical utility-scale turbine installed in the 2020s has a hub height (the height of the central axis where the blades connect) of **80–120 metres** above the ground. The rotor diameter can exceed 150 metres, meaning blade tips sweep from 30m to over 180m altitude.

Wind speed increases with height above the ground — a phenomenon called **wind shear**. Near the surface, friction from terrain (trees, buildings, hills) slows the wind. Higher up, the air moves more freely. The rate of increase depends on surface roughness: over flat, open farmland, wind speed at 100m might be 30% higher than at 10m; over forested or urban terrain, it can be 50–100% higher.

<div class="definition-box">
<strong>Wind shear:</strong> The increase in wind speed with altitude, caused by surface friction. Wind shear is typically modelled using a power law: v(h) = v(h_ref) × (h / h_ref)^α, where α is the wind shear exponent (typically 0.10–0.25 depending on terrain roughness). This means a surface wind measurement at 10m must be "extrapolated" upward to estimate conditions at turbine hub height — a process that introduces significant uncertainty.
</div>

ERA5's 100m wind data (`u100`, `v100`) is invaluable because it provides wind information at approximately the right height for turbine operation, without needing the uncertain extrapolation from surface observations. Surface (10m) wind from weather stations would need to be scaled up using wind profile models — adding a layer of approximation and error.

### From ERA5 Grid to Regional Aggregates

ERA5 provides data at every grid point, but the NEM prices are set at the *regional* level — one price for all of SA1, one for all of VIC1, and so on. We need to convert the gridded weather data into a single representative value per variable per region per time step.

The standard approach is **spatial averaging**: compute the arithmetic mean of all ERA5 grid points that fall within the region's geographic boundary. For South Australia, this might involve averaging across 50–100 grid points.

This is a simplification. In reality, solar farms and wind farms are not uniformly distributed across a region — they cluster in areas with good resources (sunny plains for solar, ridgelines and coastal areas for wind). A more sophisticated approach would weight each grid point by the installed generation capacity nearby. However, for price forecasting at the regional level, the simple spatial average captures the broad weather patterns that drive aggregate generation and is usually sufficient.

<div class="example-box">
<strong>Real-world example:</strong> South Australia's major wind farms are concentrated along the coast and in the mid-north ranges, where wind speeds are highest. The interior is calmer. A simple spatial average over the entire state understates the wind conditions at actual wind farm locations. However, the day-to-day and hour-to-hour variation — the signal that matters for forecasting — is strongly correlated across the region. When a cold front sweeps through SA, it brings strong winds to both the coast and the ranges. The spatial average captures this frontal signal even if it slightly underestimates the absolute wind speed at farm locations.
</div>

## Solar Radiation and the Clear-Sky Index

### How Sunlight Reaches the Ground

The sun delivers approximately **1,361 watts per square metre** (W/m^{2}) at the top of Earth's atmosphere — a value called the **solar constant**. But by the time this radiation reaches the Earth's surface, it has been significantly reduced by its journey through the atmosphere. Several factors determine how much arrives at any given location and time:

<div class="definition-box">
<strong>Solar constant:</strong> The total solar power per unit area received at the top of Earth's atmosphere on a surface perpendicular to the sun's rays, approximately 1,361 W/m^{2}. Despite its name, it varies slightly (~0.1%) over the 11-year solar cycle, but this variation is negligible for energy applications.
</div>

**1. Sun angle and atmospheric path length.** When the sun is directly overhead (noon in the tropics), its rays pass through the minimum amount of atmosphere. At low sun angles (morning, evening, winter at high latitudes), the same rays must traverse a much longer path through the atmosphere, increasing absorption and scattering. This path length is quantified by the **air mass (AM)**.

<div class="definition-box">
<strong>Air mass (AM):</strong> The ratio of the atmospheric path length to the shortest possible path (sun directly overhead). AM = 1 means the sun is directly overhead; AM = 2 means the path is twice as long (sun at ~60° from zenith). At sunrise and sunset, AM can exceed 10. More air mass means more absorption and scattering, reducing the radiation that reaches the ground. The standard test condition for solar panels is AM 1.5, representing typical mid-latitude conditions.
</div>

**2. Clouds.** Clouds are by far the largest source of short-term variability in solar radiation. A thick cumulonimbus cloud can block more than 80% of incoming radiation. Thin cirrus clouds may reduce it by only 10–20%. The type, thickness, altitude, and spatial extent of clouds all matter.

**3. Aerosols.** Tiny particles suspended in the atmosphere — dust, smoke, pollution, sea salt — scatter and absorb radiation. In Australia, bushfire smoke can significantly reduce solar output across entire regions for days. The "Black Summer" bushfires of 2019–2020 reduced solar irradiance across eastern Australia by 20–40% during heavy smoke events.

**4. Water vapour.** Atmospheric moisture absorbs radiation at specific infrared wavelengths. Humid tropical air absorbs more than dry desert air, even under clear skies.

### Components of Solar Radiation

The total solar radiation reaching a horizontal surface — called **Global Horizontal Irradiance (GHI)** — consists of two components:

<div class="definition-box">
<strong>Global Horizontal Irradiance (GHI):</strong> The total solar power per unit area received on a horizontal surface at the Earth's surface. It is the sum of the direct beam component and the diffuse (scattered) component. GHI is the primary driver of solar panel electricity output and is measured in watts per square metre (W/m^{2}). Typical peak values in Australia range from 800–1,100 W/m^{2} on clear days.
</div>

<div class="definition-box">
<strong>Direct Normal Irradiance (DNI):</strong> The component of solar radiation arriving in a straight line from the sun. It creates shadows and can be focused by mirrors or lenses. Under clear skies, DNI is the dominant component of GHI. Under overcast skies, DNI drops to near zero because the direct beam is blocked by clouds.
</div>

<div class="definition-box">
<strong>Diffuse Horizontal Irradiance (DHI):</strong> The component of solar radiation that has been scattered by the atmosphere (by molecules, aerosols, and clouds) and arrives at the surface from all directions across the sky, not directly from the sun. On a heavily overcast day, nearly all radiation is diffuse. Even under clear skies, 10–20% of GHI is typically diffuse due to molecular (Rayleigh) scattering.
</div>

The relationship between these components is:

<div class="equation">

GHI = DNI × cos(θ_z) + DHI

</div>

where θ_z is the **solar zenith angle** — the angle between the sun's position and directly overhead (the zenith). When the sun is directly overhead, θ_z = 0° and cos(θ_z) = 1. At sunrise/sunset, θ_z ≈ 90° and cos(θ_z) ≈ 0, so GHI consists almost entirely of diffuse radiation.

<div class="definition-box">
<strong>Solar zenith angle (θ_z):</strong> The angle between the sun's position in the sky and the point directly overhead (the zenith). θ_z = 0° means the sun is directly overhead; θ_z = 90° means the sun is on the horizon. The zenith angle depends on latitude, time of day, and day of year, and can be calculated precisely using astronomical formulae. It is the complement of the more familiar "solar elevation angle" (elevation = 90° − zenith).
</div>

### Clear-Sky Models and pvlib

A **clear-sky model** predicts the solar irradiance that *would* arrive at the surface if there were no clouds. It depends on factors that are known precisely — geographic location, time (which determines the sun's position), and atmospheric composition — but not on cloud cover, which is the main source of uncertainty.

<div class="definition-box">
<strong>Clear-sky model:</strong> A mathematical model that calculates the theoretical maximum solar irradiance at a given location and time under cloudless conditions. It accounts for the sun's position (astronomy), atmospheric path length (air mass), and atmospheric attenuation (aerosols, water vapour, ozone). Several implementations exist; the <strong>Ineichen/Perez model</strong> is widely used and accounts for altitude and atmospheric turbidity.
</div>

<div class="definition-box">
<strong>pvlib:</strong> An open-source Python library for simulating photovoltaic (solar panel) energy systems. It includes functions for calculating sun position, clear-sky irradiance, panel temperature effects, and system output. Developed by Sandia National Laboratories, it is the standard tool for solar energy research and is used in this project to compute clear-sky reference values.
</div>

<div class="definition-box">
<strong>Linke turbidity coefficient:</strong> A dimensionless number that quantifies how much the atmosphere attenuates solar radiation compared to a perfectly clean, dry atmosphere. A value of 1 means a pristine atmosphere (rare in practice); typical values range from 2 (clean, dry air) to 5+ (polluted or very humid). The Ineichen clear-sky model uses this coefficient to adjust the clear-sky irradiance for local atmospheric conditions.
</div>

The `pvlib` library implements several clear-sky models. The calculation proceeds in two steps:

1. **Solar position calculation.** Given latitude, longitude, and time, compute the sun's zenith angle, azimuth, and extraterrestrial radiation. This is pure astronomy — deterministic and precise.

2. **Atmospheric attenuation.** Given the sun's position, altitude, and atmospheric turbidity, calculate how much radiation survives its passage through the atmosphere. The Ineichen model applies empirical correction factors derived from thousands of clear-sky measurements worldwide.

The result is a smooth curve of GHI through the day — rising from zero at sunrise, peaking at solar noon, and returning to zero at sunset — that represents the maximum possible irradiance under the given atmospheric conditions.

### The Clear-Sky Index: Isolating the Weather Signal

The **clear-sky index (CSI)** is the ratio of actual measured (or reanalysis) irradiance to the clear-sky model's prediction:

<div class="equation">

CSI = GHI_actual / GHI_clearsky

</div>

![Clear-sky index](figures/04_clear_sky_index.png)

<p class="figure-caption">Figure 4.1 — The clear-sky index normalises actual irradiance against the astronomical expectation. CSI near 1.0 indicates clear skies; values below 1.0 indicate cloud cover. The CSI removes the predictable diurnal and seasonal cycles, isolating the weather-driven variability that matters for forecasting.</p>

<div class="definition-box">
<strong>Clear-sky index (CSI):</strong> The ratio of observed Global Horizontal Irradiance to the clear-sky (theoretical cloud-free) irradiance at the same location and time. CSI = 1.0 means the sky is clear and solar output matches the theoretical maximum. CSI = 0.5 means clouds are blocking half the expected radiation. CSI is only defined when GHI_clearsky > 0 (daytime); during nighttime, it is undefined and set to NaN (not a number).
</div>

The CSI is a far more useful forecasting feature than raw GHI because it **separates the weather signal from the astronomical signal**:

| Situation | Raw GHI (W/m^{2}) | CSI | Interpretation |
|-----------|----------------|-----|----------------|
| Clear noon, summer | 1,050 | 1.0 | Clear sky |
| Cloudy noon, summer | 300 | 0.29 | Very cloudy |
| Clear noon, winter | 650 | 1.0 | Clear sky |
| Clear 4pm, summer | 400 | 1.0 | Clear sky, low sun angle |
| Cloudy 4pm, summer | 150 | 0.38 | Moderately cloudy |

A raw GHI of 400 W/m^{2} could mean "clear sky at 4pm" or "moderately cloudy at noon." Without context, the number is ambiguous. A CSI of 0.38 unambiguously means "clouds are blocking 62% of the expected radiation, regardless of time of day or season." This context-independence makes CSI a much better input for machine learning models.

<div class="key-point">
<strong>Why normalise?</strong> The raw GHI signal is dominated by the predictable astronomical cycle (sunrise, solar noon, sunset, seasonal tilt). Any forecasting model could learn this cycle from calendar features alone. The <em>interesting</em> part — the part that differs between a clear day and a cloudy day, and thus affects the electricity price — is captured by the CSI. By normalising away the predictable component, we give the model a feature that contains only the information it cannot get elsewhere.
</div>

### CSI Edge Cases and Quality Control

The clear-sky index has several special cases that require careful handling:

- **CSI ≈ 1.0:** Clear sky. Actual irradiance matches the theoretical expectation.
- **CSI < 1.0:** Clouds are present, reducing irradiance below the clear-sky value. Values of 0.2–0.4 indicate heavy overcast; 0.6–0.8 indicates thin or broken cloud.
- **CSI > 1.0:** This can genuinely occur due to **cloud enhancement** — sunlight reflecting off the edges of clouds can briefly boost irradiance above the clear-sky level, sometimes by up to 40% (CSI ≈ 1.4). This happens when the sun is not blocked but nearby clouds act as additional reflectors.
- **CSI > 1.5:** Almost certainly a measurement or data error. These values should be clipped to 1.5 to prevent them from corrupting downstream analysis.
- **CSI undefined (nighttime):** When GHI_clearsky = 0 (or very close to zero, as at the low-sun-angle edges of the day), the CSI division produces infinity or numerically unstable values. These should be replaced with NaN.

<div class="definition-box">
<strong>Cloud enhancement:</strong> A phenomenon where solar irradiance at the surface briefly exceeds the clear-sky value due to reflections from the edges or undersides of nearby clouds. The direct beam from the sun is unobstructed, and additional diffuse radiation reflected from cloud surfaces augments the total. Cloud enhancement events typically last seconds to minutes and can produce GHI values up to ~1,400 W/m^{2}, exceeding the solar constant at the top of the atmosphere.
</div>

### Persistence of Cloudiness

Cloud cover — and therefore the CSI — is not random from one hour to the next. It has characteristic **persistence timescales** driven by the weather systems that produce it:

- **Seconds to minutes:** Individual cumulus clouds passing over a solar farm create rapid fluctuations. These "ramp events" matter for grid stability but are too fast for day-ahead forecasting.
- **Hours:** Synoptic weather features — fronts, troughs, cloud bands — create multi-hour periods of consistent cloudiness or clearness.
- **Days:** Large-scale circulation patterns — blocking high-pressure systems, cut-off low-pressure systems — can maintain clear or cloudy conditions for three to seven days.
- **Seasonal:** Some regions have systematically more cloud in certain seasons (e.g., winter frontal systems in southern Australia).

<div class="definition-box">
<strong>Synoptic scale:</strong> Weather features that span hundreds to thousands of kilometres and persist for days — fronts, low-pressure systems, high-pressure ridges. These are the scales captured by ERA5's ~25 km grid and are the primary drivers of day-to-day variation in cloudiness, wind, and temperature.
</div>

For day-ahead price forecasting, the hourly-to-daily persistence is the relevant timescale. Today's CSI at a given hour is a reasonable (though imperfect) predictor of tomorrow's CSI at the same hour, because the weather systems that determine cloudiness typically persist for several days. This persistence is what makes weather a useful forecasting feature even without access to a numerical weather prediction forecast.

## Wind Speed and Wind Power

### The Physics of Wind Energy

Wind turbines extract kinetic energy from moving air. The fundamental physics dictates that the power available in the wind is proportional to the **cube of the wind speed**:

<div class="equation">

P_wind = ½ × ρ × A × v^{3}

</div>

where:
- P_wind is the power available in the wind (watts)
- ρ is the air density (approximately 1.225 kg/m^{3} at sea level)
- A is the swept area of the rotor (m^{2}) — for a rotor diameter D, A = π(D/2)^{2}
- v is the wind speed (m/s)

<div class="definition-box">
<strong>Betz limit:</strong> A theoretical maximum on the fraction of wind energy that a turbine can extract, derived by German physicist Albert Betz in 1919. The limit is 16/27 ≈ 59.3%. No wind turbine, regardless of design, can convert more than 59.3% of the kinetic energy in the wind into mechanical energy. Modern turbines achieve 35–45% efficiency (called the <strong>power coefficient</strong>, C_p), which is 60–75% of the Betz limit.
</div>

The cubic relationship has profound implications for forecasting. Consider a wind speed error of just 1 m/s:

| Actual wind speed | Forecast wind speed | Power ratio (forecast/actual) | Error |
|-------------------|--------------------|-----------------------------|-------|
| 8 m/s | 7 m/s | 7^{3}/8^{3} = 0.67 | −33% |
| 8 m/s | 9 m/s | 9^{3}/8^{3} = 1.42 | +42% |
| 12 m/s | 11 m/s | 11^{3}/12^{3} = 0.77 | −23% |

A 1 m/s error in wind speed translates to a 23–42% error in power output. This amplification is why wind power forecasting is inherently harder than solar power forecasting — small atmospheric uncertainties create large generation uncertainties.

### The Wind Power Curve

The theoretical cubic relationship is modified by the engineering design of actual turbines, producing the characteristic **wind power curve**:

![Wind power curve](figures/04_wind_power_curve.png)

<p class="figure-caption">Figure 4.2 — The wind power curve showing the relationship between wind speed and electrical output. The curve has four distinct regions: below cut-in (no output), the cubic ramp, rated power (flat), and cut-out (shutdown). Small changes in wind speed near the steep part of the curve produce large changes in output.</p>

<div class="definition-box">
<strong>Wind power curve:</strong> A graph showing the relationship between wind speed and the electrical power output of a wind turbine. It is the fundamental characteristic of a turbine model and encodes the engineering limits (cut-in speed, rated speed, cut-out speed) as well as the aerodynamic performance in between. Real power curves differ slightly from the idealised version due to turbulence, air density variations, and control system behaviour.
</div>

The power curve has four distinct regions:

**Region 1 — Below cut-in speed (0 to ~3 m/s):** No power output. The wind is too weak to overcome the friction and inertia of the rotor and drivetrain. The turbine sits idle.

<div class="definition-box">
<strong>Cut-in speed:</strong> The minimum wind speed at which a turbine begins generating electricity, typically 2.5–4 m/s (~9–14 km/h). Below this speed, the aerodynamic torque on the blades is insufficient to overcome mechanical losses and start the generator.
</div>

**Region 2 — Cut-in to rated speed (~3 to ~12 m/s):** Power output increases approximately as the cube of wind speed. The turbine operates with its blades pitched to extract maximum energy from the wind. This is the steepest part of the power curve and the region where wind speed forecast errors create the largest generation forecast errors.

**Region 3 — Rated speed to cut-out speed (~12 to ~25 m/s):** Power output is constant at the turbine's **rated capacity** (the maximum sustained output it is designed for, typically 2–8 MW for modern utility-scale turbines). The blades are **pitched** (rotated along their long axis) to shed excess wind energy and prevent mechanical overload. Increasing wind speed beyond rated produces no additional power — the control system deliberately wastes the excess.

<div class="definition-box">
<strong>Rated capacity (nameplate capacity):</strong> The maximum sustained power output a turbine is designed to produce, reached at its rated wind speed. A "3 MW turbine" can produce at most 3 MW of electricity, regardless of how strong the wind blows. Rated capacity is used to calculate <strong>capacity factor</strong> — the ratio of actual output to theoretical maximum output.
</div>

<div class="definition-box">
<strong>Blade pitch control:</strong> A mechanism that rotates each turbine blade along its longitudinal axis to change the angle at which it meets the incoming wind. At low wind speeds, the blade is pitched to maximise energy capture. At high wind speeds (above rated), the blade is pitched to reduce the aerodynamic force, shedding excess energy to keep the output at rated capacity and prevent structural damage.
</div>

**Region 4 — Above cut-out speed (>~25 m/s):** The turbine shuts down entirely for safety. The blades are feathered (pitched to 90° so they present minimum area to the wind), and the rotor is braked to a stop. Output drops from full rated capacity to zero. This is rare but creates dramatic and sudden drops in wind generation across a wind farm or region.

<div class="definition-box">
<strong>Cut-out speed:</strong> The maximum wind speed at which a turbine can operate safely, typically 25 m/s (~90 km/h). Above this speed, the turbine shuts down to prevent structural damage to the blades, tower, and nacelle. Modern turbines increasingly use "storm ride-through" modes that reduce output gradually rather than cutting off abruptly, but the basic principle remains: extremely high winds produce <em>less</em> generation, not more.
</div>

<div class="example-box">
<strong>Real-world example — cut-out events:</strong> During severe storm events in South Australia, wind speeds can exceed 25 m/s across large areas. When this happens, wind farms shut down en masse, removing hundreds or thousands of megawatts of generation from the system within minutes. This is exactly when the grid is under maximum stress from storm-related outages and high demand. The September 2016 SA blackout was partly triggered by multiple wind farm shutdowns due to a severe storm — a tragic illustration of the cut-out problem.
</div>

### Calculating Wind Speed from ERA5

ERA5 provides wind as two perpendicular components at 100m height. The wind speed — which is what determines turbine output — is calculated using the Pythagorean theorem:

<div class="equation">

wind_speed = √(u100^{2} + v100^{2})

</div>

For example, if the east–west component u100 = 6 m/s and the north–south component v100 = −8 m/s, then:

wind_speed = √(6^{2} + (−8)^{2}) = √(36 + 64) = √100 = 10 m/s

The negative sign on v100 means the wind is blowing from north to south, but the speed calculation only cares about the magnitude, not the direction. This is appropriate because a turbine's power output depends on wind speed, not wind direction (turbines rotate on their yaw axis to face the wind from any direction).

<div class="definition-box">
<strong>Yaw control:</strong> A mechanism that rotates the entire nacelle (the housing containing the generator and gearbox) atop the tower so that the rotor faces directly into the wind. Modern turbines continuously adjust their yaw angle based on wind direction measurements. Because of yaw control, the power output depends on wind <em>speed</em> but not wind <em>direction</em>.
</div>

### Why Wind Is Harder to Forecast Than Solar

Wind generation is fundamentally harder to predict than solar generation:

| Factor | Solar | Wind |
|--------|-------|------|
| **Deterministic component** | Strong — sunrise, noon, sunset are precisely predictable | None — wind can be strong or weak at any time of day |
| **Dominant timescale** | Diurnal (daily cycle, always present) | Synoptic (weather systems, 2–7 days, irregular) |
| **Spatial coherence** | High — cloud systems span 100+ km | Lower — wind varies significantly over 10–50 km due to terrain |
| **Response to forecast error** | Linear — 10% GHI error ≈ 10% power error | Cubic — 10% speed error ≈ 30% power error |
| **Nighttime generation** | Zero (no forecast needed) | Non-zero (must forecast 24 hours, not just daytime) |

<div class="key-point">
<strong>The cubic amplification problem:</strong> Because wind power depends on the cube of wind speed, small errors in wind speed forecasts are amplified into large errors in power forecasts. This is the single biggest challenge in wind power forecasting. A weather model that predicts wind speed within ±1 m/s — considered a good forecast — can still produce wind power errors of 30% or more. This error propagation is why purely physical wind forecasting is insufficient for electricity price prediction and why statistical methods that learn the speed-to-power mapping from historical data are essential.
</div>

### Capacity Factor and Its Implications

<div class="definition-box">
<strong>Capacity factor:</strong> The ratio of a generator's actual output over a period to its theoretical maximum output (rated capacity × time). A wind farm with a capacity factor of 35% means it produces, on average, 35% of what it would produce if it ran at full rated output 24 hours a day, 365 days a year. Capacity factor reflects both the intermittency of the resource (wind does not always blow) and practical limitations (maintenance downtime, curtailment).
</div>

South Australian wind farms typically achieve capacity factors of **30–40%**. But this average hides a strongly **bimodal distribution** — output is frequently either near zero (low wind) or near full rated capacity (strong wind), with relatively little time spent at intermediate levels. This bimodality occurs because the steep part of the power curve (region 2) covers a relatively narrow band of wind speeds. Much of the time, the wind is either below the effective range (low output) or above rated speed (full output).

This bimodality matters for price forecasting because it means wind generation acts more like a binary switch (on/off) than a continuous dial. A weather system that brings strong winds "switches on" the wind fleet; a calm period "switches it off." This creates step changes in the supply stack that produce corresponding step changes in price.

## Temperature and Electricity Demand

### The Comfort Zone and the U-Shaped Response

The relationship between temperature and electricity demand follows a characteristic **U-shape** (or V-shape):

- **Below ~18°C:** Demand increases as temperature falls, driven by electric heating. Each degree below the comfort zone adds roughly 1–3% to demand.
- **18–22°C (the comfort zone):** Demand is at its minimum. Comfortable temperatures require neither heating nor cooling.
- **Above ~22°C:** Demand increases as temperature rises, driven by air conditioning. The response is steeper than heating because air conditioning is energy-intensive. Each degree above the comfort zone can add 2–5% to demand.

<div class="definition-box">
<strong>Temperature-demand relationship:</strong> The empirical relationship between ambient temperature and electricity demand. It is approximately U-shaped: demand is lowest at comfortable temperatures (~18–22°C) and increases for both warmer and colder temperatures. The heating branch (cold) and cooling branch (hot) have different slopes, reflecting the different energy intensities of heating and cooling systems.
</div>

<div class="definition-box">
<strong>Heating degree hours (HDH) and cooling degree hours (CDH):</strong> Metrics that quantify accumulated thermal discomfort. HDH = max(0, T_base − T_actual) summed over each hour, where T_base is typically 18°C. CDH = max(0, T_actual − T_base) summed similarly. These capture the <em>duration</em> and <em>intensity</em> of heating or cooling stress. A single hour at 40°C (CDH contribution: 22) creates more cooling demand than two hours at 29°C (CDH contribution: 2 × 11 = 22) — but the integrated metric captures both cases equally.
</div>

The temperature–demand response is **nonlinear** and **asymmetric**:

- The cooling side is steeper because air conditioners consume more electricity per degree of cooling than heaters consume per degree of heating (in the NEM, where gas heating is common, electric heating load is actually modest).
- The response is not instantaneous — buildings have thermal mass, so demand responds with a lag of 1–4 hours to temperature changes.
- The response depends on humidity. A 35°C day with 70% humidity creates more air conditioning load than a 35°C day with 20% humidity, because air conditioners must remove moisture as well as heat.

<div class="example-box">
<strong>Real-world example — temperature and demand in SA1:</strong> On a mild autumn day (22°C, low humidity), SA1 might have demand of 1,200 MW. On a 42°C summer day, demand can exceed 3,000 MW — a 150% increase driven almost entirely by air conditioning. This demand surge, combined with reduced renewable output (wind often drops during extreme heat events), creates the conditions for price spikes exceeding $10,000/MWh.
</div>

### Temperature Deviation as a Feature

Rather than using raw temperature as a forecasting feature, it is more useful to use **temperature deviation** — the difference between actual temperature and the comfort baseline:

<div class="equation">

T_dev = T_actual − T_base

</div>

where T_base is typically 18°C. Positive values indicate cooling demand; negative values indicate heating demand. This transformation captures the nonlinear demand response more naturally than raw temperature:

- T_dev = +20 (i.e., T = 38°C): Extreme cooling demand
- T_dev = 0 (i.e., T = 18°C): Minimal demand
- T_dev = −10 (i.e., T = 8°C): Moderate heating demand

Some models further split this into separate heating and cooling components to allow different slopes:

<div class="equation">

HDD = max(0, T_base − T_actual)

CDD = max(0, T_actual − T_base)

</div>

<div class="definition-box">
<strong>Heating Degree Days (HDD) and Cooling Degree Days (CDD):</strong> Cumulative measures of how cold or hot a period was relative to a base temperature (typically 18°C). HDD sums the degrees below base; CDD sums the degrees above. Originally defined on a daily basis, the concept extends to hourly resolution as heating degree hours and cooling degree hours. These are standard features in energy demand modelling because they directly proxy the thermal load on HVAC systems.
</div>

## Weather as a Price Driver: The Causal Chain

The relationship between weather and price is not direct — it flows through a chain of physical and market mechanisms:

<div class="key-point">
<strong>The causal chain:</strong> Weather → Renewable generation + Demand → Net load → Position on supply stack → Marginal generator → Price. Weather does not "cause" prices directly. It causes changes in supply (renewable generation) and demand (heating/cooling load), which together determine net load, which determines which generator sets the price. Understanding this chain is essential for building good features — we want features that are as close to the price-setting mechanism as possible.
</div>

Each link in the chain introduces uncertainty:

1. **Weather → Solar generation:** Moderated by panel efficiency, inverter capacity, curtailment
2. **Weather → Wind generation:** Amplified by the cubic power curve, moderated by turbine availability
3. **Weather → Demand:** Moderated by building thermal mass, behavioral responses, time-of-day effects
4. **Supply + Demand → Net load:** Direct subtraction
5. **Net load → Price:** The highly nonlinear supply stack mapping (the "hockey stick" from Chapter 3)

The nonlinearity at step 5 is crucial. A 100 MW change in net load might change the price by $2 in the middle of the supply stack, or by $5,000 at the steep end. This means the *same* weather forecast error can produce vastly different price forecast errors depending on where the system is operating on the supply stack.

<div class="example-box">
<strong>Real-world example — the same weather, different price impact:</strong> Suppose wind generation is forecast to be 500 MW but actually comes in at 400 MW — a 100 MW miss. On a mild day with 1,500 MW demand, this pushes net load from 1,000 MW to 1,100 MW, moving the system along the flat part of the supply stack. Price might increase from $45 to $55 — a $10 error. On a 42°C day with 3,000 MW demand, the same 100 MW miss pushes net load from 2,500 MW to 2,600 MW, hitting the steep part of the supply stack. Price might increase from $300 to $2,000 — a $1,700 error. Same weather forecast error, 170× different price impact.
</div>

## Feature Engineering: From Raw Weather to Model Inputs

### Transforming Weather Variables

Raw ERA5 variables are not directly suitable as forecasting features. They need to be transformed into quantities that have clear relationships with electricity market outcomes:

| Raw variable | Transformation | Feature | Rationale |
|-------------|---------------|---------|-----------|
| `ssrd` (accumulated J/m^{2}) | Differentiate, convert to W/m^{2}, divide by clear-sky | **Clear-sky index** | Removes astronomical cycle, isolates weather signal |
| `u100`, `v100` (m/s) | Pythagorean combination | **Wind speed** (m/s) | Directly related to turbine power output |
| `t2m` (Kelvin) | Convert to °C, subtract base | **Temperature deviation** (°C) | Captures demand response to thermal stress |
| `t2m` (Kelvin) | CDD/HDD calculation | **Cooling/heating degree hours** | Time-integrated demand proxy |

### Lagged vs. Contemporaneous Weather Features

For day-ahead forecasting — where the forecast is issued before noon for the following trading day — we face a fundamental problem: we do not know tomorrow's weather. There are three approaches:

**Approach 1: Use weather forecasts as features.** Feed the output of a numerical weather prediction (NWP) model into the price forecasting model. This is operationally correct — it uses the same weather information that would be available to a real-time forecaster. The drawback is that it requires access to historical NWP forecasts for training, which are large, expensive, and not always available.

<div class="definition-box">
<strong>Numerical Weather Prediction (NWP):</strong> The practice of forecasting weather by running physics-based computer models that simulate the atmosphere's evolution. NWP models solve the equations of fluid dynamics, thermodynamics, and radiative transfer on a three-dimensional grid, starting from an observed initial state. Modern NWP models can produce useful forecasts up to ~10 days ahead. For energy applications, the most relevant products are forecasts of temperature, wind, and solar radiation at 1–72 hour horizons.
</div>

**Approach 2: Use lagged actual weather.** Today's weather is a reasonable predictor of tomorrow's weather, because weather systems change slowly (synoptic-scale persistence). Use today's CSI as a predictor of tomorrow's CSI; use today's wind speed as a predictor of tomorrow's wind speed. This is simple and does not require NWP data, but it misses weather changes (e.g., a front arriving tomorrow that was not present today).

**Approach 3: Use reanalysis as "perfect weather" (the approach in this project).** Use ERA5 reanalysis for both historical weather and "forecast" weather. This provides an *upper bound* on how well a weather-informed model can perform — it tells us how much value weather information adds when the weather is known perfectly. In production, using actual NWP forecasts would reduce performance, and the gap between "perfect weather" and "NWP forecast" quantifies the value of improving weather forecasts.

<div class="key-point">
<strong>Upper-bound thinking:</strong> Using ERA5 reanalysis as if it were a perfect weather forecast establishes a performance ceiling. If a model using perfect weather achieves a capture ratio of 0.75, then no weather-informed model can do better than 0.75 regardless of how good the NWP forecast becomes. If a model <em>without</em> weather achieves 0.65, we know the maximum value of weather information is a 10 percentage-point improvement. This upper-bound reasoning is invaluable for deciding whether to invest in better weather data.
</div>

### AEMO's Own Forecasts: NEMSEER Pre-Dispatch

AEMO publishes its own forecasts of wind and solar generation through the **NEMSEER** (NEM System Event and Emergency Reserve) pre-dispatch system. These are updated every 30 minutes with a horizon of up to 40 hours.

<div class="definition-box">
<strong>Pre-dispatch:</strong> AEMO's process of projecting electricity dispatch outcomes (generation schedules, prices, interconnector flows) for the upcoming 24–40 hours. Pre-dispatch runs every 30 minutes and incorporates generator bids, demand forecasts, network constraints, and renewable generation forecasts. The pre-dispatch price is AEMO's best estimate of future prices — and serves as a natural benchmark for price forecasting models.
</div>

Using AEMO's generation forecasts as features (rather than raw weather) has advantages:

- They already account for **installed capacity** — a weather model does not know how many panels are in SA1, but AEMO does
- They already account for **curtailment** — AEMO knows if a wind farm is constrained by network limits
- They already account for **turbine power curves** — AEMO's forecast converts wind speed to megawatts
- They represent the **market's expectation** — other participants react to these forecasts, so they partly determine the price

The disadvantage is that if AEMO's forecast has a systematic bias (consistently over- or under-forecasting wind, for example), our price model inherits that bias.

## Persistence Forecasts as Weather Baselines

### What Is a Persistence Forecast?

The simplest weather forecast is **persistence**: assume tomorrow's weather will be the same as today's weather at the same time.

<div class="definition-box">
<strong>Persistence forecast:</strong> A forecasting method that predicts the future value of a variable will equal its current (or most recent past) value. For weather variables, persistence at the daily timescale means: tomorrow's temperature at 2pm = today's temperature at 2pm. Persistence is the simplest forecast that captures the autocorrelation (serial correlation) in weather data. It is the standard baseline — any model that cannot beat persistence has added complexity without adding skill.
</div>

- **Solar persistence:** Tomorrow's CSI at hour H = today's CSI at hour H
- **Wind persistence:** Tomorrow's wind speed at hour H = today's wind speed at hour H
- **Temperature persistence:** Tomorrow's temperature at hour H = today's temperature at hour H

Persistence works well when weather changes slowly — a multi-day high-pressure system bringing clear skies, or a sustained period of strong westerly winds. It fails when the weather changes rapidly — a cold front arriving overnight, a sea breeze onset in the afternoon, or a thunderstorm developing.

For solar forecasting, it is critical to persist the *clear-sky index*, not raw GHI. Persisting raw GHI would systematically fail because of the seasonal change in day length and solar angle — tomorrow's noon GHI in winter is lower than today's noon GHI if the season is progressing toward winter, even if the weather is identical.

<div class="key-point">
<strong>The persistence test:</strong> Persistence is the "you must be this tall to ride" test for weather and renewable forecasting models. If a sophisticated machine learning model cannot beat "tomorrow equals today," the model is either badly designed or the weather is genuinely unpredictable at the relevant horizon. Always evaluate models against persistence before drawing conclusions about their skill.
</div>

---

## Glossary

| Term | Definition |
|------|-----------|
| **VRE** | Variable Renewable Energy — generation whose output depends on weather (wind, solar) |
| **Dispatchable** | Generation that can be controlled by the operator (coal, gas, hydro) |
| **ERA5** | ECMWF's fifth-generation global reanalysis dataset |
| **Reanalysis** | Physics-model reconstruction of historical atmosphere using all available observations |
| **Data assimilation** | Mathematical fusion of model predictions and real observations |
| **ECMWF** | European Centre for Medium-Range Weather Forecasts |
| **GHI** | Global Horizontal Irradiance — total solar radiation on a horizontal surface |
| **DNI** | Direct Normal Irradiance — beam component of solar radiation |
| **DHI** | Diffuse Horizontal Irradiance — scattered component of solar radiation |
| **Solar zenith angle** | Angle between the sun and directly overhead |
| **Air mass** | Ratio of atmospheric path length to the minimum path |
| **Clear-sky model** | Model predicting irradiance under cloud-free conditions |
| **CSI** | Clear-Sky Index — ratio of actual to clear-sky irradiance |
| **pvlib** | Python library for solar energy system simulation |
| **Linke turbidity** | Coefficient quantifying atmospheric clarity |
| **Cloud enhancement** | Irradiance exceeding clear-sky due to cloud reflections |
| **Wind shear** | Increase in wind speed with altitude |
| **Cut-in speed** | Minimum wind speed for turbine generation (~3 m/s) |
| **Rated capacity** | Maximum sustained turbine output |
| **Cut-out speed** | Maximum safe operating wind speed (~25 m/s) |
| **Blade pitch control** | Rotating blades to regulate power capture |
| **Capacity factor** | Ratio of actual output to theoretical maximum |
| **Betz limit** | Theoretical maximum wind energy extraction efficiency (~59.3%) |
| **HDD / CDD** | Heating / Cooling Degree Days — thermal stress metrics |
| **NWP** | Numerical Weather Prediction — physics-based weather forecasting |
| **Pre-dispatch** | AEMO's projection of future dispatch outcomes |
| **Persistence forecast** | Predicting tomorrow = today |
| **Synoptic scale** | Weather features spanning hundreds to thousands of km |

## Summary

Weather is the master variable in renewable-dominated electricity grids, driving both supply (wind and solar generation) and demand (heating and cooling load) simultaneously. ERA5 reanalysis provides the gold-standard historical weather dataset — globally complete, hourly, physically consistent — with key variables including surface solar radiation, 100m wind components, and 2m temperature. Solar irradiance is best represented by the clear-sky index, which normalises against the predictable astronomical cycle to isolate the weather-driven variability. Wind power follows a cubic relationship with wind speed in the operating range, amplifying small forecast errors into large generation errors, and the wind power curve introduces additional nonlinearity through cut-in, rated, and cut-out thresholds. Temperature drives electricity demand through a U-shaped response centred on a comfort zone of ~18–22°C. The causal chain from weather to price passes through multiple nonlinear transformations — most critically the supply-stack mapping — meaning identical weather forecast errors can produce vastly different price forecast errors depending on system conditions. Feature engineering transforms raw weather into forecasting-ready quantities (CSI, wind speed, temperature deviation), and persistence forecasts provide the essential baseline that any model must beat.
