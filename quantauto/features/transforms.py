"""feature transforms: normalization, scaling, winsorization, and cross-sectional ops."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# column-wise transforms (each column treated independently)
# ---------------------------------------------------------------------------


def zscore_normalize(
    df: pd.DataFrame,
    *,
    window: Optional[int] = None,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """z-scores each feature column independently: (x - mean) / std.

    window -- if provided, uses a backward-looking rolling window (prevents
              future leakage from global mean/std; first window-1 rows become NaN).
              if None, normalises using the full-series mean and std.
    columns -- subset of columns to normalise; defaults to all columns.
    """
    target = list(columns) if columns is not None else list(df.columns)
    result = df.copy()
    subset = df[target]

    if window is not None:
        mean = subset.rolling(window=window, min_periods=window).mean()
        std = subset.rolling(window=window, min_periods=window).std()
    else:
        mean = subset.mean()
        std = subset.std()

    normalised = (subset - mean) / std.replace(0.0, np.nan)
    result[target] = normalised
    return result


def minmax_scale(
    df: pd.DataFrame,
    *,
    window: Optional[int] = None,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """scales each feature column to [0, 1] using min-max scaling.

    window -- if provided, scales relative to rolling min/max (no future leakage;
              first window-1 rows become NaN).
              if None, uses column min/max over the full series.
    columns -- subset of columns to scale; defaults to all columns.
    """
    target = list(columns) if columns is not None else list(df.columns)
    result = df.copy()
    subset = df[target]

    if window is not None:
        mn = subset.rolling(window=window, min_periods=window).min()
        mx = subset.rolling(window=window, min_periods=window).max()
    else:
        mn = subset.min()
        mx = subset.max()

    scaled = (subset - mn) / (mx - mn).replace(0.0, np.nan)
    result[target] = scaled
    return result


def winsorize(
    df: pd.DataFrame,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """clips each column at the given quantiles to reduce the impact of outliers.

    uses full-series quantiles (global).  for rolling winsorisation, compute
    rolling quantiles manually before calling this function.
    lower/upper are quantile thresholds in (0, 1), e.g. lower=0.01, upper=0.99.
    columns -- subset of columns to winsorise; defaults to all columns.
    """
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError(
            f"lower and upper must satisfy 0 <= lower < upper <= 1, got ({lower}, {upper})"
        )
    target = list(columns) if columns is not None else list(df.columns)
    result = df.copy()
    subset = df[target]
    lo = subset.quantile(lower)
    hi = subset.quantile(upper)
    result[target] = subset.clip(lower=lo, upper=hi, axis=1)
    return result


def forward_fill(
    df: pd.DataFrame,
    *,
    max_periods: Optional[int] = None,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """forward-fills NaN values within each column using the last valid observation.

    max_periods -- maximum number of consecutive NaN rows to fill; None fills all.
    columns -- subset of columns to fill; defaults to all columns.
    """
    target = list(columns) if columns is not None else list(df.columns)
    result = df.copy()
    result[target] = df[target].ffill(limit=max_periods)
    return result


def drop_warmup(df: pd.DataFrame, n_bars: int) -> pd.DataFrame:
    """drops the first n_bars rows (the warmup period with insufficient feature history).

    use EngineeredFeatures.warmup_bars to know the correct n_bars to pass.
    returns a copy; does nothing if n_bars <= 0.
    """
    if n_bars <= 0:
        return df.copy()
    return df.iloc[n_bars:].copy()


# ---------------------------------------------------------------------------
# cross-sectional transforms (operate row-wise across assets)
# ---------------------------------------------------------------------------


def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """z-scores values across columns (assets) at each timestamp.

    at each row, subtracts the row mean and divides by the row std.
    useful for ranking or normalising a single feature across multiple assets.
    rows with zero variance (all identical) produce NaN.
    df -- wide format: index = timestamps, columns = asset symbols.
    """
    row_mean = df.mean(axis=1)
    row_std = df.std(axis=1)
    safe_std = row_std.where(row_std > 0.0, np.nan)
    return df.sub(row_mean, axis=0).div(safe_std, axis=0)


def cross_sectional_rank(
    df: pd.DataFrame,
    *,
    pct: bool = True,
    ascending: bool = True,
) -> pd.DataFrame:
    """ranks values across columns (assets) at each timestamp.

    pct       -- if True (default), returns fractional rank in (0, 1]; the asset
                 with the lowest value gets rank close to 0 and the highest close to 1.
    ascending -- if False, higher values receive lower rank (useful for signals
                 where larger = better → rank 1 means top asset).
    df -- wide format: index = timestamps, columns = asset symbols.
    """
    if ascending:
        return df.rank(axis=1, pct=pct)
    # descending: negate before ranking so the highest value gets the lowest rank number
    ranked = (-df).rank(axis=1, pct=pct)
    if pct:
        # after negation, pct rank is still in (0,1]; flip so 1 = best (highest original value)
        ranked = 1.0 - ranked + ranked.min(axis=1).div(df.shape[1])
    return ranked


def align_features_to_source(
    features: pd.DataFrame,
    source_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """reindexes features to match a source DatetimeIndex, forward-filling gaps.

    use when features and source have slightly different timestamp sets
    (e.g. after resampling or when handling multi-asset aligned frames).
    missing rows at the head (before the first feature bar) are left as NaN.
    """
    return features.reindex(source_index).ffill()
