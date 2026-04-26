"""time-only splits for stacked panel data (leakage-safe)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from quantauto.models.splits import TimeSplit, make_walk_forward_splits


@dataclass(frozen=True)
class PanelTimeSplit:
    """one train/test partition on *timestamps* (all symbols at a time in one set)."""

    split_id: str
    train_times: pd.DatetimeIndex
    test_times: pd.DatetimeIndex


def list_unique_times_panel(index: pd.MultiIndex) -> pd.DatetimeIndex:
    t = index.get_level_values(0)
    return pd.DatetimeIndex(pd.Index(t).unique().sort_values())


def _holdout_time_indices(n_times: int, test_split: float) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < test_split < 1.0:
        raise ValueError(f"test_split must be in (0,1), got {test_split}")
    if n_times < 5:
        raise ValueError("need at least 5 unique times for a holdout split")
    cut = int(n_times * (1.0 - test_split))
    if cut <= 0 or cut >= n_times:
        raise ValueError("invalid time cut: empty train or test partition")
    tr = np.arange(0, cut, dtype=int)
    te = np.arange(cut, n_times, dtype=int)
    return tr, te


def make_holdout_time_split(
    times: pd.DatetimeIndex, *, test_split: float
) -> PanelTimeSplit:
    """first (1 - test_split) of unique sorted times = train, rest = test."""
    u = times.unique().sort_values()
    n = len(u)
    tr_i, te_i = _holdout_time_indices(n, test_split)
    return PanelTimeSplit(
        split_id="holdout",
        train_times=u[tr_i],
        test_times=u[te_i],
    )


def make_walk_forward_time_splits(
    times: pd.DatetimeIndex,
    *,
    n_folds: int,
    test_size: int,
    min_train_size: int,
    purge_bars: int = 0,
    embargo_bars: int = 0,
) -> List[PanelTimeSplit]:
    """expanding time splits, mirroring :func:`make_walk_forward_splits` but on a time list."""
    u = times.unique().sort_values()
    return _wf_from_unique_times(
        u,
        n_folds=n_folds,
        test_size=test_size,
        min_train_size=min_train_size,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )


def _wf_from_unique_times(
    u: pd.DatetimeIndex,
    *,
    n_folds: int,
    test_size: int,
    min_train_size: int,
    purge_bars: int,
    embargo_bars: int,
) -> List[PanelTimeSplit]:
    """
    n_folds, test_size, min_train_size are in *unique-time* count units, same
    as :func:`make_walk_forward_splits` row semantics but each row = one time.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if test_size < 1:
        raise ValueError("test_size must be >= 1")
    n = len(u)
    # delegate index math to existing helper by building a proxy index
    proxy = pd.DatetimeIndex(u)
    inner = make_walk_forward_splits(
        proxy,
        n_folds=n_folds,
        test_size=test_size,
        min_train_size=min_train_size,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )
    out: List[PanelTimeSplit] = []
    for s in inner:
        out.append(
            PanelTimeSplit(
                split_id=s.split_id,
                train_times=pd.DatetimeIndex(s.train_index).unique().sort_values(),
                test_times=pd.DatetimeIndex(s.test_index).unique().sort_values(),
            )
        )
    return out


def time_split_to_row_mask(
    index: pd.MultiIndex,
    train_times: pd.DatetimeIndex,
    test_times: pd.DatetimeIndex,
) -> Tuple[pd.Series, pd.Series]:
    """boolean masks for panel rows: level-0 in train_times / test_times."""
    t = pd.Index(index.get_level_values(0))
    tr = t.isin(train_times)
    te = t.isin(test_times)
    return (pd.Series(tr, index=index), pd.Series(te, index=index))


def resolve_panel_time_splits(
    unique_times: pd.DatetimeIndex,
    *,
    test_split: float,
    walk_forward_folds: int,
    purge_bars: int,
    embargo_bars: int,
) -> List[PanelTimeSplit]:
    """map training config to a list of :class:`PanelTimeSplit` (one or many folds)."""
    if walk_forward_folds <= 1:
        return [make_holdout_time_split(unique_times, test_split=test_split)]
    n_t = len(unique_times)
    test_size = max(8, int(n_t * test_split))
    min_train_size = max(
        2,
        n_t - (walk_forward_folds * (test_size + embargo_bars)),
    )
    return make_walk_forward_time_splits(
        unique_times,
        n_folds=walk_forward_folds,
        test_size=test_size,
        min_train_size=min_train_size,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )
