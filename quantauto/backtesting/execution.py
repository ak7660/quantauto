"""signal-to-position and execution-cost helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def predictions_to_positions(
    predictions: pd.Series,
    *,
    task_type: str,
    threshold: float = 0.0,
    clip_exposure: float = 1.0,
) -> pd.Series:
    """map model outputs to tradable positions."""
    if clip_exposure <= 0:
        raise ValueError("clip_exposure must be > 0")
    if task_type == "regression":
        raw = np.where(predictions.astype(float).values > threshold, 1.0, -1.0)
    elif task_type == "classification":
        raw = np.where(predictions.astype(float).values >= 0.5, 1.0, -1.0)
    else:
        raise ValueError(f"unsupported task_type {task_type!r}")
    pos = np.clip(raw, -clip_exposure, clip_exposure)
    return pd.Series(pos, index=predictions.index, name="position")


def trade_cost_rate(*, fee_bps: float, slippage_bps: float) -> float:
    """convert bps costs into return-space trade cost rate."""
    return float((fee_bps + slippage_bps) / 10_000.0)

