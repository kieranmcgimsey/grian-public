"""Preconfigured ablation experiments — deliberate failures.

Each function returns a trial config that introduces one specific
mistake. The purpose is to show experimentally, through metrics, that
these mistakes matter. Run the "correct" config alongside the ablations
and compare them in the dashboard.

Ablations included:
    correct_baseline — Everything done right. The reference trial.
    wrong_loss       — MSE on raw prices (spike-dominated gradients).
    no_transform     — Model raw $/MWh without asinh (heavy-tailed).
    no_reconditioning — Train once, never refit (distribution drift).
    future_leakage   — Accidentally include future data (too good).
    no_embargo       — Train right up to forecast origin (subtle leak).
"""

from grian.sim.trials import make_config


def correct_baseline(
    model: str = "lightgbm",
    regions: list[str] | None = None,
    **overrides,
) -> dict:
    """The reference trial — everything done correctly.

    Asinh target transform, proper embargo, daily reconditioning,
    pinball loss. Every ablation should be compared against this.

    Args:
        model: Model name to use.
        regions: List of NEM regions (default: ["SA1"]).
        **overrides: Any additional config overrides.

    Returns:
        Complete trial config dict.
    """
    if regions is None:
        regions = ["SA1"]

    cfg = make_config({
        "trial_name": f"{model}_correct",
        "model": model,
        "regions": regions,
        "transform": "asinh",
        "loss": "pinball",
        "refit_days": 1,
        "ablations": {
            "scale_target": True,
            "use_transform": True,
            "use_embargo": True,
            "leak_future": False,
        },
        **overrides,
    })
    return cfg


def wrong_loss(
    model: str = "lightgbm",
    regions: list[str] | None = None,
    **overrides,
) -> dict:
    """Ablation: train with MSE on raw prices.

    MSE squares the errors, so extreme prices (spikes) dominate the
    gradient. The model over-fits to spikes and under-fits the base
    load. This should show up as higher MAE on non-spike intervals
    and worse overall revenue.

    Args:
        model: Model name to use.
        regions: List of NEM regions.
        **overrides: Any additional config overrides.

    Returns:
        Trial config with wrong loss function.
    """
    if regions is None:
        regions = ["SA1"]

    return make_config({
        "trial_name": f"{model}_wrong_loss",
        "model": model,
        "regions": regions,
        "transform": "asinh",
        "loss": "mse",
        "refit_days": 1,
        "ablations": {
            "scale_target": True,
            "use_transform": True,
            "use_embargo": True,
            "leak_future": False,
        },
        **overrides,
    })


def no_transform(
    model: str = "lightgbm",
    regions: list[str] | None = None,
    **overrides,
) -> dict:
    """Ablation: model raw $/MWh without target transform.

    NEM spot prices are heavy-tailed — they range from -$1000 to
    +$16000. Without a compressive transform like asinh, the model
    sees a distribution dominated by rare extreme values. Expect
    unstable training, poor generalisation, and large forecast errors
    in the base-load region where most revenue comes from.

    Args:
        model: Model name to use.
        regions: List of NEM regions.
        **overrides: Any additional config overrides.

    Returns:
        Trial config with no target transform.
    """
    if regions is None:
        regions = ["SA1"]

    return make_config({
        "trial_name": f"{model}_no_transform",
        "model": model,
        "regions": regions,
        "transform": "identity",
        "loss": "pinball",
        "refit_days": 1,
        "ablations": {
            "scale_target": True,
            "use_transform": False,
            "use_embargo": True,
            "leak_future": False,
        },
        **overrides,
    })


def no_reconditioning(
    model: str = "lightgbm",
    regions: list[str] | None = None,
    **overrides,
) -> dict:
    """Ablation: train once on history, never refit.

    NEM price dynamics shift over time — new generators come online,
    interconnectors change, demand patterns evolve. A model trained
    once will drift. Expect gradually worsening performance as the
    test period progresses, visible as an increasingly negative slope
    in the equity curve.

    Args:
        model: Model name to use.
        regions: List of NEM regions.
        **overrides: Any additional config overrides.

    Returns:
        Trial config with no reconditioning (refit_days very large).
    """
    if regions is None:
        regions = ["SA1"]

    return make_config({
        "trial_name": f"{model}_no_reconditioning",
        "model": model,
        "regions": regions,
        "transform": "asinh",
        "loss": "pinball",
        "refit_days": 99999,
        "ablations": {
            "scale_target": True,
            "use_transform": True,
            "use_embargo": True,
            "leak_future": False,
        },
        **overrides,
    })


def future_leakage(
    model: str = "lightgbm",
    regions: list[str] | None = None,
    **overrides,
) -> dict:
    """Ablation: accidentally include future data in training.

    The training set extends past the embargo into the forecast
    window. The model can memorise future prices, producing
    suspiciously accurate forecasts and inflated revenue. This is
    the most dangerous mistake because it looks like success.

    Args:
        model: Model name to use.
        regions: List of NEM regions.
        **overrides: Any additional config overrides.

    Returns:
        Trial config with future leakage enabled.
    """
    if regions is None:
        regions = ["SA1"]

    return make_config({
        "trial_name": f"{model}_future_leakage",
        "model": model,
        "regions": regions,
        "transform": "asinh",
        "loss": "pinball",
        "refit_days": 1,
        "ablations": {
            "scale_target": True,
            "use_transform": True,
            "use_embargo": True,
            "leak_future": True,
        },
        **overrides,
    })


def no_embargo(
    model: str = "lightgbm",
    regions: list[str] | None = None,
    **overrides,
) -> dict:
    """Ablation: zero embargo between training and forecast.

    Without an embargo gap, the last training observation is adjacent
    to the first forecast target. Autocorrelated features (lagged
    prices) can leak information about the near future. Subtler than
    full future leakage — metrics will be slightly better than they
    should be, making the model look more skilful than it is.

    Args:
        model: Model name to use.
        regions: List of NEM regions.
        **overrides: Any additional config overrides.

    Returns:
        Trial config with zero embargo.
    """
    if regions is None:
        regions = ["SA1"]

    return make_config({
        "trial_name": f"{model}_no_embargo",
        "model": model,
        "regions": regions,
        "transform": "asinh",
        "loss": "pinball",
        "refit_days": 1,
        "embargo": 0,
        "ablations": {
            "scale_target": True,
            "use_transform": True,
            "use_embargo": False,
            "leak_future": False,
        },
        **overrides,
    })


# ---------------------------------------------------------------------------
# Run all ablations as a batch
# ---------------------------------------------------------------------------

ALL_ABLATIONS = [
    correct_baseline,
    wrong_loss,
    no_transform,
    no_reconditioning,
    future_leakage,
    no_embargo,
]


def make_ablation_suite(
    model: str = "lightgbm",
    regions: list[str] | None = None,
    **overrides,
) -> list[dict]:
    """Generate configs for all ablation experiments.

    Args:
        model: Model name to use for all ablations.
        regions: List of NEM regions.
        **overrides: Passed through to each ablation factory.

    Returns:
        List of trial config dicts, one per ablation.
    """
    return [fn(model=model, regions=regions, **overrides) for fn in ALL_ABLATIONS]
