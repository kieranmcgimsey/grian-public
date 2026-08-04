"""Model implementations for the NEM price-forecasting **curriculum** (notebooks).

These are the teaching implementations the ten notebooks import. They are
*separate* from the simulation test bench's model registry in
``grian.sim.models`` (the ``fit``/``predict``/``save``/``load`` dicts the
walk-forward backtest and dashboard use) — same subject, different half of the
repo. Edit `grian.sim.models` for test-bench/dashboard work; edit these for the
notebook curriculum.

Submodules:
    baselines — similar-day naive and autoregression
    lear — LASSO-estimated autoregressive (LEAR) models
    gbt — LightGBM quantile regression
    nn — PyTorch day-ahead quantile network
    qra — quantile regression averaging
    conformal — conformal prediction wrapper
"""
