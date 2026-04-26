"""dataset assembly helpers for training."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantauto.features.engineering import EngineeredFeatures
from quantauto.labels.timing import check_label_feature_overlap, get_valid_label_range


@dataclass(frozen=True)
class TrainingDataset:
    X: pd.DataFrame
    y: pd.Series
    index: pd.DatetimeIndex
    warmup_bars: int
    horizon: int


def make_training_dataset(
    engineered: EngineeredFeatures,
    label: pd.Series,
    *,
    horizon: int,
) -> TrainingDataset:
    """align features and label into a leakage-safe training dataset."""
    features = engineered.data.copy()
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("engineered feature index must be a DatetimeIndex")
    if not isinstance(label.index, pd.DatetimeIndex):
        raise TypeError("label index must be a DatetimeIndex")

    safe, msg = check_label_feature_overlap(label.name or "label", tuple(features.columns))
    if not safe:
        raise ValueError(msg)

    valid_idx = get_valid_label_range(features.index, horizon)
    if engineered.warmup_bars > 0:
        valid_idx = valid_idx[engineered.warmup_bars :]

    X = features.reindex(valid_idx)
    y = label.reindex(valid_idx)
    combined = pd.concat([X, y.rename("__label__")], axis=1).dropna()
    if combined.empty:
        raise ValueError("no rows left after alignment/warmup/NaN filtering")

    X_clean = combined.drop(columns=["__label__"])
    y_clean = combined["__label__"]
    return TrainingDataset(
        X=X_clean,
        y=y_clean,
        index=pd.DatetimeIndex(X_clean.index),
        warmup_bars=engineered.warmup_bars,
        horizon=horizon,
    )
