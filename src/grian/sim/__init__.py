"""Simulation environment for NEM energy trading experiments.

Provides a walk-forward trading simulator, experiment tracking, and
hyperparameter search — all built on plain functions and dicts.

Modules:
    trials   — Trial config, artifact save/load, reproducibility.
    models   — Fit/predict/save/load specs for each model type.
    ledger   — Append-only trade log and P&L computation.
    runner   — Day-by-day walk-forward simulation loop.
    ablations — Preconfigured "wrong on purpose" trial configs.
    search   — Pluggable hyperparameter search strategies.
    dashboard — Static HTML builder for comparing trial results.
"""
