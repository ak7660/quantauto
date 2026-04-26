"""time-ordered split generators for model training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    split_id: str
    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex


def make_holdout_split(index: pd.DatetimeIndex, *, test_split: float = 0.2) -> TimeSplit:
    if not 0.0 < test_split < 1.0:
        raise ValueError(f"test_split must be in (0,1), got {test_split}")
    n = len(index)
    if n < 5:
        raise ValueError("need at least 5 rows for holdout split")
    cut = int(n * (1.0 - test_split))
    if cut <= 0 or cut >= n:
        raise ValueError("invalid split cut produced empty train or test partition")
    return TimeSplit(
        split_id="holdout",
        train_index=index[:cut],
        test_index=index[cut:],
    )


def make_walk_forward_splits(
    index: pd.DatetimeIndex,
    *,
    n_folds: int = 3,
    test_size: int = 64,
    min_train_size: int = 128,
    purge_bars: int = 0,
    embargo_bars: int = 0,
) -> List[TimeSplit]:
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if test_size < 1:
        raise ValueError("test_size must be >= 1")
    if min_train_size < 2:
        raise ValueError("min_train_size must be >= 2")
    if purge_bars < 0 or embargo_bars < 0:
        raise ValueError("purge_bars and embargo_bars must be >= 0")
    n = len(index)
    required = min_train_size + (n_folds * (test_size + embargo_bars))
    if n < required:
        raise ValueError(
            f"not enough rows for walk-forward: have={n}, required={required}"
        )

    splits: List[TimeSplit] = []
    train_end = min_train_size
    for i in range(n_folds):
        test_start = train_end
        test_end = test_start + test_size
        train_stop = max(0, test_start - purge_bars)
        train_idx = index[:train_stop]
        test_idx = index[test_start:test_end]
        if len(train_idx) < 2 or len(test_idx) < 1:
            raise ValueError(
                f"invalid fold wf_{i+1}: train={len(train_idx)} test={len(test_idx)} "
                "(check purge/test/min_train settings)"
            )
        splits.append(
            TimeSplit(
                split_id=f"wf_{i+1}",
                train_index=train_idx,
                test_index=test_idx,
            )
        )
        train_end = test_end + embargo_bars
    return splits
