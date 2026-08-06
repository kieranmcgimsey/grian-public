"""The typed model contract.

Every forecaster is a :class:`ModelSpec` — a frozen, typed record of its name,
output kind, the four lifecycle functions (``fit``/``predict``/``save``/``load``),
its typed params class, and an optional quantile-fan predictor. This replaces the
old plain ``{"name": ..., "fit": ...}`` dict: attribute access is checked, the
record is immutable, and a missing/renamed field is an error at import time
rather than a ``KeyError`` deep in the dispatch loop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """A registered forecaster and everything the engine needs to run it.

    Attributes:
        name: The canonical model name.
        output: ``"point"`` or ``"quantile"``.
        fit: ``(train_df, target_col, cfg) -> state``.
        predict: ``(state, input_df, horizon) -> pd.Series`` (transformed space).
        save: ``(state, path) -> None``.
        load: ``(path) -> state``.
        params: The typed params class the fit validates ``model_params`` into.
        predict_fan: Optional ``(state, input_df, horizon) -> {q: array}`` for
            probabilistic dispatch; ``None`` for point-only models.
    """

    name: str
    output: str
    fit: Callable[..., dict]
    predict: Callable[..., Any]
    save: Callable[..., None]
    load: Callable[..., dict]
    params: type
    predict_fan: Callable[..., Any] | None = None
