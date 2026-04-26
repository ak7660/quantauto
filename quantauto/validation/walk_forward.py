"""walk-forward evaluation helpers."""

from __future__ import annotations

import copy
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from quantauto.models.base import ModelAdapter
from quantauto.models.splits import TimeSplit
from quantauto.models.types import TrainSplitResult
from quantauto.validation.metrics import score_model


def run_splits(
    model: ModelAdapter,
    X: pd.DataFrame,
    y: pd.Series,
    splits: Iterable[TimeSplit],
) -> Tuple[TrainSplitResult, ...]:
    out = []
    for split in splits:
        X_train = X.reindex(split.train_index)
        y_train = y.reindex(split.train_index)
        X_test = X.reindex(split.test_index)
        y_test = y.reindex(split.test_index)
        if X_train.empty or y_train.empty or X_test.empty or y_test.empty:
            raise ValueError(
                f"split {split.split_id} has empty partition(s): "
                f"train={len(X_train)} test={len(X_test)}"
            )
        fold_model = copy.deepcopy(model)
        fold_model.fit(X_train, y_train)
        pred = fold_model.predict(X_test).reindex(y_test.index)
        metrics = score_model(model.task_type, y_test, pred)
        out.append(
            TrainSplitResult(
                split_id=split.split_id,
                train_size=len(X_train),
                test_size=len(X_test),
                metrics=metrics,
                predictions=pred,
                actuals=y_test,
            )
        )
    return tuple(out)


def aggregate_split_metrics(split_results: Iterable[TrainSplitResult]) -> Dict[str, float]:
    rows = list(split_results)
    if not rows:
        return {}
    keys = sorted({k for r in rows for k in r.metrics.keys()})
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(r.metrics[k]) for r in rows if k in r.metrics]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            out[k] = float("nan")
        else:
            out[k] = float(np.mean(vals))
    return out
