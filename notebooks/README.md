# grian notebooks

The teaching half of grian: a twelve-notebook walk from raw NEM data to a
forecast driving a battery. They import the library (`src/grian/`), narrate each
step, and set exercises — they don't hide logic in helpers.

## Honest status — read this first

- **Notebooks 01–10 are shipped *without* executed outputs.** They contain the
  narrative, the runnable code, and the exercises, but the cells have not been
  re-run and saved with rendered plots/tables. **Run them yourself to see the
  results** (`jupyter lab`, run top to bottom). This keeps the repo small and the
  outputs honest — nothing is a stale screenshot.
- **The committed data lets most of them run cold.** The small processed parquet
  (`data/processed/SA1_{5,30}min_sim.parquet`, price + demand) is committed, so
  01, 02, 05–10 run without any download. **Two need more:** notebook **03** pulls
  unit-level generation (NEMOSIS), and notebook **04** needs a free
  [Copernicus CDS](https://cds.climate.copernicus.eu/) key for ERA5 weather (it
  stubs cleanly and stays readable without one).
- **The exercises are left as exercises.** Each notebook ends with 2–3 analytical
  questions whose code cells are intentionally empty (`# Your analysis here`),
  with staged hints and a full worked solution hidden under a `<details>` block.
- **The headline result is *not* in these notebooks.** The champion ≈ 0.55 capture
  number comes from the simulation environment (`src/grian/sim/`), the scripts, and
  the [dashboard](../outputs/dashboard/index.html) / [experiment log](../outputs/experiment_log.md).
  Notebooks 01–10 teach the pipeline; notebooks 11–12 bridge into the campaign.

## The curriculum (01–10)

The arc: **data → market understanding → features → forecast → money.** Run in
order — each assumes data and concepts from the ones before.

| # | Notebook | What's in it | Runs on committed data? |
|---|----------|--------------|:---:|
| 01 | Get and tame NEM data | The data landscape, interval-ending timestamps and the one-interval shift, caching, no re-pull. | ✅ |
| 02 | The shape of the market | Stylised facts: the price distribution, spikes, negatives, seasonality, autocorrelation, volatility clustering, heavy tails. | ✅ |
| 03 | How price forms | Marginal pricing, the fuel mix, net load, the empirical supply stack, spike decomposition. | ⬇︎ pulls unit generation |
| 04 | Weather and generation | ERA5 weather, `pvlib` clear-sky index, wind/solar features, a first generation forecast. | 🔑 needs an ERA5 key |
| 05 | Framing, features, baselines & the dispatch harness | Day-ahead framing, the leakage-safe rolling-origin backtest, naive + AR baselines, the battery LP, a naive point-forecast MPC, and the AEMO pre-dispatch benchmark. The methodological backbone. | ✅ |
| 06 | Classical baselines (LEAR) | LASSO-per-hour forecasting, the feature-selection path, per-hour vs global, and a first capture-ratio readout. | ✅ |
| 07 | Machine learning | LightGBM quantile regression and a day-ahead quantile network (torch, CPU/MPS), point vs fan dispatch. | ✅ |
| 08 | Probabilistic forecasting | Quantile regression averaging (QRA), conformal calibration, reliability/coverage, CRPS, spike coverage. | ✅ |
| 09 | From forecast to money | Receding-horizon MPC used properly: scenario and chance-constrained variants, sensitivity, and the value-of-information curve. | ✅ |
| 10 | Capstone | The full pipeline end to end plus a grey-box structural-residual price model and the headline scorecard. | ✅ |

Every model notebook (06–08) closes by pushing its forecast through the dispatch
harness and reporting **capture ratio** alongside CRPS and relative MAE — the
forecast is a means to dispatch value, not an end.

## The investigation notebooks (11–12)

Unlike 01–10, these **do carry their executed outputs** — they are worked
investigations from the capture-ratio campaign, not part of the linear teaching arc.

| # | Notebook | What it shows |
|---|----------|---------------|
| 11 | Making the MPC work | The `observe_present` bug — why the receding-horizon MPC collapsed to ~half of open-loop, root-caused to it trading the forecast residual, and the spike-gated fix. The single best debugging story in the repo. |
| 12 | Forecast conditioning | Conditioning the forecast on the daily-shape prior. |

## Running them

```bash
pip install -e ".[dev]"     # jupyter is in the dev extra
jupyter lab                 # then open a notebook and run top to bottom
```

Notebooks 03 and 04 will fetch/require extra data as noted above; the rest use the
committed parquet. Seeds are fixed (`config.yaml`), so re-running is deterministic.

## How they're written (conventions)

If you extend or re-run them, they follow a deliberate house style:

- **Raw data first** — inspect the DataFrame (`.head()`, `.dtypes`, `.describe()`)
  before transforming it; transformations happen in cells, not hidden helpers.
- **Inline plotting** — all matplotlib lives in the cells; `viz.py` provides only
  `apply_style()` and `save_fig()`.
- **Reusable logic lives in `src/grian/`** — notebooks import and call; they don't
  define functions later notebooks depend on.
- **Exercises use staged reveals** — a question, then `<details>`-wrapped hints,
  then a hidden worked solution.
- **Determinism** — seeds set from config, downloads cached, versions pinned, CPU /
  Apple-Silicon only.
