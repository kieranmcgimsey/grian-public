# Extending the simulator

Three extension points, in order of how often you will reach for them: add a
**model**, add a **feature group**, add an **executor knob**. Each has a
worked example and the invariants you must not break. The house style is
functional and underengineered — plain functions, plain dicts, explicit state
(see [architecture.md](architecture.md#design-philosophy)). Match it.

## Adding a model

A model is a dict of four functions registered in `models.REGISTRY`. Add it in
`src/grian/sim/models.py`:

```python
def _my_fit(train_df, target_col, cfg):
    """Train and return a state dict (any shape you like)."""
    # train_df[target_col] is already in transformed space (e.g. asinh).
    ...
    return {"weights": ..., "target_col": target_col,
            "resolution": cfg.get("resolution", "5min")}

def _my_predict(state, input_df, horizon):
    """Return a pd.Series of length `horizon`, in TRANSFORMED space,
    indexed by the future timestamps after the end of input_df."""
    ...

def _my_save(state, path):
    """Serialise state into directory `path`."""
    ...

def _my_load(path):
    """Load and return the state dict."""
    ...

MY_MODEL = {
    "name": "my_model", "output": "point",
    "fit": _my_fit, "predict": _my_predict,
    "save": _my_save, "load": _my_load,
}
REGISTRY["my_model"] = MY_MODEL
```

Then `"model": "my_model"` in any config works.

### The three invariants (all learned the hard way)

1. **Predict from the end of `input_df`, not from `fit`-time state.** The runner
   and MPC loop hand `predict` all data observed up to the forecast origin. If
   you ignore it and read a `state["train_tail"]` frozen at fit time, your
   forecast goes stale between weekly refits — up to 7 days stale. This was
   Entry 014; it cost the naive model its day-of-week alignment and lightgbm its
   recency features. Use `input_df` when it is provided and long enough; fall
   back to fit-time data only otherwise.

2. **Return transformed space.** The runner inverts `cfg["transform"]` (e.g.
   asinh → sinh) before dispatch and scoring. Return what the model naturally
   predicts in the transformed target; do not invert inside `predict`. (The one
   subtlety: if you integrate quantiles to a mean, you invert *per quantile*
   first because inversion and the mean do not commute under skew — see
   `_lgbm_qmean_predict` and trap T3 in
   [dispatch-and-scoring.md](dispatch-and-scoring.md#trap-3-median-vs-mean).)

3. **Index the returned Series by the future timestamps.** Length `horizon`,
   starting one interval after `input_df`'s last timestamp.

### Verify it before trusting it

- Add a test to `tests/test_sim_mpc.py::TestPredictFromNow` asserting your
  forecast changes when you hand it more data (predict-from-now).
- Run one validation trial and check the forecast's dollar mean and max are in a
  sane range vs actuals (a common failure is systematic under-dispersion — the
  forecast is too smooth, the LP under-trades; Entry 003).

## Adding a feature group

Features live in `src/grian/sim/features.py`. Each group is a pure function
`DataFrame -> DataFrame` of **strictly backward-looking** columns, wired into
`build_features`:

```python
def build_features(df, target_col="price", resolution="5min",
                   include_demand=True) -> pd.DataFrame:
    ...
```

To add a group (say, temperature):

1. Write `def temperature_features(df, ...) -> pd.DataFrame:` returning columns
   that use **only past rows** — no look-ahead. If a column at time `t` uses any
   value from `t` or later, it is leakage.
2. Concatenate it inside `build_features`.
3. It flows automatically into `lightgbm_rich` and `lightgbm_qmean` (they call
   `build_features`).

### The leakage test is mandatory

The engineering rules require a unit test proving that injecting a future value
degrades (or that a leaky feature *improves*) the score — leakage that looks
like skill is the most dangerous bug in forecasting (trap T6). Before adding an
exogenous feature that joins on a timestamp (pre-dispatch forecasts, weather),
confirm you join on **issue time ≤ now**, not target time. A naive
join-on-target-time is silently leaky and will look brilliant.

## Adding an executor knob

Executor behaviour is configured under `cfg["mpc"]` (MPC) or `cfg["dispatch"]`
(battery spec). To add one:

1. Add the key with a safe default to `trials.DEFAULT_CONFIG` — `make_config`
   **rejects unknown keys**, so this comes first.
2. Read it in `mpc.simulate_region_mpc` (or `runner.battery_dispatch`).
3. **Prove it changes the output.** Two configs that produce byte-identical
   ledgers mean the knob is dead config (Entry 002 — "configuration theatre").
   After wiring any flag, run two trials that differ only in it and diff the
   ledgers.

Worked precedent: the `persistence_tau` / `persistence_gate` knobs were added
this way (default 0 = off), wired into the MPC planning step, and ablated on
validation. The result was negative (Entry 018) — which is a perfectly good
outcome, provided it is logged.

## The experiment-log rule

**Every modelling failure, unexpected result, or bug you find during
experimentation must be written up in
[`outputs/experiment_log.md`](../outputs/experiment_log.md) before you move on.**
This is not optional — it is the mechanism that makes failures as valuable as
successes, and it is why this repo can tell you *why* the current design is what
it is. Use the template at the bottom of the log:

```
## Entry NNN: [short descriptive title]
**Date / Component / Severity**
### What happened   — factual observation
### Why it happened — root cause in the code/config/data
### The fix (or: why there is no fix)
### Lesson          — the generalisable takeaway
```

A good entry is one a future worker can read and thereby *avoid the trap* without
rediscovering it. Entries 013–019 are the campaign; use them as models.

## Before you commit

```bash
ruff check .        # lint — docstrings are enforced (pydocstyle)
pytest -q           # full suite (198 tests)
```

Every public function needs a Google-style docstring. Then:

- Update [`outputs/plans/capture_campaign_results.md`](../outputs/plans/capture_campaign_results.md)
  if you produced a headline number (one row, with the git SHA).
- Update the status ledger in the [plan](../outputs/plans/capture_campaign.md)
  if you closed a ticket.
- Commit on a branch; **push only to `kieranmcgimsey/grian`** — verify
  `git remote -v` first. This repo must never be pushed to an org remote.

## What NOT to do

- Don't add a class hierarchy or a framework. If you find yourself writing a base
  class, stop — a dict of functions is the pattern.
- Don't hardcode `dt`, cycle counts, or window dates in new code; read them from
  config.
- Don't touch the frozen metric (oracle definition, windows, battery spec)
  without a dated justification in the plan — it invalidates every prior number.
- Don't tune on the test window. Ever.
