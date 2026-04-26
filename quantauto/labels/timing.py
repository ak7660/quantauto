"""label timing: metadata, horizon validation, and leakage guards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import pandas as pd


@dataclass
class LabelMeta:
    """metadata about one label's forward-looking horizon.

    name          -- label column name
    horizon_bars  -- bars of future data used; the last horizon_bars rows are NaN
    label_type    -- 'classification', 'regression', or 'ranking'
    description   -- optional human-readable note
    """

    name: str
    horizon_bars: int
    label_type: str
    description: str = ""


def validate_label_timing(
    label: pd.Series,
    source_df: pd.DataFrame,
    *,
    horizon: int,
) -> Tuple[bool, str]:
    """checks that a label series has the expected forward-looking structure.

    verifies:
    1. label index is a DatetimeIndex that is a subset of source_df index
    2. the last `horizon` rows are NaN (as expected from shift(-horizon))
    3. at least one valid (non-NaN) value exists

    returns (passed: bool, message: str).
    """
    if not isinstance(label.index, pd.DatetimeIndex):
        return False, "label index must be a DatetimeIndex"
    if not isinstance(source_df.index, pd.DatetimeIndex):
        return False, "source_df index must be a DatetimeIndex"

    extra = label.index.difference(source_df.index)
    if len(extra) > 0:
        return False, (
            f"label has {len(extra)} timestamp(s) absent from source_df; "
            "label must be aligned to the same frame as source"
        )

    n = len(label)
    if n < horizon + 1:
        return False, (
            f"series length {n} is too short for horizon={horizon}; "
            "need at least horizon + 1 rows"
        )

    # the last horizon rows must all be NaN
    tail = label.iloc[-horizon:]
    non_nan_in_tail = int(tail.notna().sum())
    if non_nan_in_tail > 0:
        return False, (
            f"expected last {horizon} rows to be NaN (forward-looking window), "
            f"but {non_nan_in_tail} row(s) are non-NaN. "
            "ensure the label was computed with shift(-horizon)."
        )

    if label.isna().all():
        return False, "label series has no valid (non-NaN) values"

    return True, "ok"


def get_valid_label_range(
    index: pd.DatetimeIndex,
    horizon: int,
) -> pd.DatetimeIndex:
    """returns the sub-index where labels are valid (excluding trailing NaN rows).

    the last `horizon` timestamps have NaN labels because no future data is
    available.  use this to trim both features and labels to the same valid range
    before feeding them into model training.
    """
    if horizon <= 0:
        return index
    valid_n = max(0, len(index) - horizon)
    return index[:valid_n]


def check_label_feature_overlap(
    label_name: str,
    feature_names: Sequence[str],
) -> Tuple[bool, str]:
    """warns when the label name matches a feature column name exactly.

    a label column accidentally included in the feature matrix causes severe
    future leakage.  this check compares names only; auditing values is the
    responsibility of the caller.

    returns (safe: bool, message: str).  safe=False means a name collision was found.
    """
    label_lower = label_name.lower()
    for feat in feature_names:
        if feat.lower() == label_lower:
            return False, (
                f"label {label_name!r} matches feature column {feat!r} exactly; "
                "including this label in the feature matrix will cause future leakage. "
                "rename the label or the conflicting feature."
            )
    return True, "ok"
