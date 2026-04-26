"""timing utilities: warmup masks, lookahead detection, and execution shift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass
class FeatureMeta:
    """metadata describing timing requirements of one engineered feature."""

    name: str
    # minimum bars of history consumed; the first (lookback_bars - 1) rows will be NaN
    lookback_bars: int
    description: str = ""


def get_warmup_mask(index: pd.DatetimeIndex, lookback: int) -> pd.Series:
    """returns bool series marking bars in the warmup period as True.

    the first max(0, lookback - 1) bars are warmup: not enough history has
    accumulated to produce a reliable feature value.  bars flagged True
    should be excluded from model training.
    """
    n_warmup = max(0, lookback - 1)
    mask = pd.Series(False, index=index, dtype=bool)
    if n_warmup > 0 and len(mask) > 0:
        mask.iloc[:n_warmup] = True
    return mask


def get_valid_start(lookback: int) -> int:
    """returns 0-based index of the first bar with sufficient history.

    bars before this index are in the warmup period and carry NaN feature values.
    """
    return max(0, lookback - 1)


def check_no_lookahead(
    feature: pd.Series,
    source_df: pd.DataFrame,
) -> Tuple[bool, str]:
    """structural check: feature timestamps must be a subset of source timestamps.

    verifies the feature index was not computed using rows from a different,
    future-extended frame.  this is a structural guard only — it cannot detect
    value-level lookahead introduced by negative shifts or misaligned rolling
    windows.  audit those by inspecting each indicator's window parameters.

    returns (passed: bool, message: str).
    """
    if not isinstance(feature.index, pd.DatetimeIndex):
        return False, "feature index must be a DatetimeIndex"
    if not isinstance(source_df.index, pd.DatetimeIndex):
        return False, "source_df index must be a DatetimeIndex"

    extra = feature.index.difference(source_df.index)
    if len(extra) > 0:
        return False, (
            f"feature has {len(extra)} timestamp(s) absent from source; "
            "ensure the feature was computed on the same frame as source "
            "and not on a forward-extended copy of that frame"
        )
    return True, "ok"


def apply_execution_shift(
    features: pd.DataFrame,
    periods: int = 1,
) -> pd.DataFrame:
    """shifts feature columns forward to model execution lag.

    a feature computed on bar T (using data through bar T's close) is shifted
    to bar T+periods, reflecting the delay before it is actionable.  periods=1
    is the minimal realistic assumption: the signal is available at bar T+1's
    open.  the first `periods` rows become NaN after the shift.

    use periods=0 to skip shifting (e.g. when labels already account for lag).
    """
    if periods < 0:
        raise ValueError(f"periods must be >= 0, got {periods}")
    if periods == 0:
        return features.copy()
    return features.shift(periods)
