"""label builders for forward-looking training targets."""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from quantauto.labels.timing import LabelMeta, validate_label_timing


def _validate_close_input(close: pd.Series, horizon: int) -> None:
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close index must be a DatetimeIndex")
    if horizon <= 0:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if len(close) <= horizon:
        raise ValueError(
            f"series length {len(close)} is too short for horizon={horizon}; "
            "need at least horizon + 1 rows"
        )


def make_forward_return_label(
    close: pd.Series,
    horizon: int,
    *,
    name: str = "label_forward_return",
) -> Tuple[pd.Series, LabelMeta]:
    """build forward return label: close[t+h]/close[t] - 1.

    the last ``horizon`` rows are NaN by construction.
    """
    _validate_close_input(close, horizon)
    label = (close.shift(-horizon) / close) - 1.0
    label = label.rename(name)
    ok, msg = validate_label_timing(label, close.to_frame(name="close"), horizon=horizon)
    if not ok:
        raise ValueError(f"invalid forward return label timing: {msg}")
    return label, LabelMeta(
        name=name,
        horizon_bars=horizon,
        label_type="regression",
        description=f"{horizon}-bar forward return",
    )


def make_direction_label(
    close: pd.Series,
    horizon: int,
    *,
    threshold: float = 0.0,
    name: str = "label_direction",
) -> Tuple[pd.Series, LabelMeta]:
    """build direction label from forward returns and threshold.

    outputs:
    - 1 when forward return > threshold
    - 0 when forward return <= threshold
    - NaN on trailing rows where forward return is NaN
    """
    ret, _ = make_forward_return_label(
        close, horizon, name=f"{name}_forward_return_temp"
    )
    label = (ret > threshold).astype("float64")
    label[ret.isna()] = pd.NA
    label = label.rename(name)
    ok, msg = validate_label_timing(label, close.to_frame(name="close"), horizon=horizon)
    if not ok:
        raise ValueError(f"invalid direction label timing: {msg}")
    return label, LabelMeta(
        name=name,
        horizon_bars=horizon,
        label_type="classification",
        description=f"{horizon}-bar forward direction with threshold={threshold}",
    )


def make_forward_rank_label(
    close: pd.Series,
    horizon: int,
    *,
    name: str = "label_forward_rank",
) -> Tuple[pd.Series, LabelMeta]:
    """placeholder rank-friendly label for single-asset workflow.

    for single asset series, this returns forward return values directly.
    callers can convert to cross-sectional ranks in multi-asset workflows.
    """
    label, _ = make_forward_return_label(close, horizon, name=name)
    return label, LabelMeta(
        name=name,
        horizon_bars=horizon,
        label_type="ranking",
        description=f"{horizon}-bar forward score for ranking workflows",
    )


def make_forward_cross_sectional_rank_label(
    frame: pd.DataFrame,
    *,
    close_column: str = "close",
    symbol_column: str = "symbol",
    horizon: int = 1,
    name: str = "label_forward_rank",
) -> Tuple[pd.Series, LabelMeta]:
    """cross-sectional forward-return rank per timestamp across symbols.

    expects columns [close_column, symbol_column] and a DatetimeIndex.
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("frame index must be a DatetimeIndex")
    if close_column not in frame.columns or symbol_column not in frame.columns:
        raise KeyError(
            f"frame must contain columns {close_column!r} and {symbol_column!r}"
        )
    if horizon <= 0:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    df = frame[[close_column, symbol_column]].copy()
    df["__fwd_ret__"] = (
        df.groupby(symbol_column, sort=False)[close_column].shift(-horizon) / df[close_column]
    ) - 1.0
    # rank within each timestamp among available symbols
    rank = df.groupby(level=0, sort=False)["__fwd_ret__"].rank(pct=True, method="average")
    rank = rank.rename(name)
    return rank, LabelMeta(
        name=name,
        horizon_bars=horizon,
        label_type="ranking",
        description=f"{horizon}-bar cross-sectional forward-return rank",
    )
