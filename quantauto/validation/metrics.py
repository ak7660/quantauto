"""metrics helpers for regression, classification, and panel ranking tasks."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    err = y_pred - y_true
    mse = float(np.mean(np.square(err)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    denom = np.sum(np.square(y_true - np.mean(y_true)))
    r2 = float(1.0 - (np.sum(np.square(err)) / denom)) if denom > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    true = y_true.astype(int)
    pred = y_pred.astype(int)
    accuracy = float((true == pred).mean())
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def ranking_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> Dict[str, float]:
    """mean cross-sectional spearman: per timestamp rank correlation then average."""
    if not isinstance(y_true.index, pd.MultiIndex) or y_true.index.nlevels < 2:
        return regression_metrics(y_true, y_pred)
    t = y_true.index.get_level_values(0)
    corrs: list[float] = []
    for ts in t.unique().sort_values():
        m = t == ts
        yt = y_true[m].astype(float)
        yp = y_pred.reindex(yt.index).astype(float)
        al = pd.concat([yt, yp], axis=1).dropna()
        if al.shape[0] < 2:
            continue
        r = al.iloc[:, 0].corr(al.iloc[:, 1], method="spearman")
        if r is not None and np.isfinite(r):
            corrs.append(float(r))
    if not corrs:
        return {"spearman": float("nan"), "mae": float("nan"), "rmse": float("nan")}
    s = float(np.mean(corrs))
    err = (y_true.astype(float) - y_pred.reindex(y_true.index).astype(float)).dropna()
    rm = float(np.sqrt(np.mean(np.square(err))) if len(err) else float("nan"))
    ma = float(np.mean(np.abs(err)) if len(err) else float("nan"))
    return {"spearman": s, "rmse": rm, "mae": ma}


def score_model(
    task_type: str,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> Dict[str, float]:
    if task_type == "regression":
        return regression_metrics(y_true, y_pred)
    if task_type == "classification":
        return classification_metrics(y_true, y_pred)
    if task_type == "ranking":
        return ranking_metrics(y_true, y_pred)
    raise ValueError(f"unsupported task_type {task_type!r}")


def metric_preference(task_type: str) -> Tuple[str, bool]:
    """return (primary_metric, maximize_metric)."""
    if task_type == "regression":
        return "rmse", False
    if task_type == "classification":
        return "accuracy", True
    if task_type == "ranking":
        return "spearman", True
    raise ValueError(f"unsupported task_type {task_type!r}")
