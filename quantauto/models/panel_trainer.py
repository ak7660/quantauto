"""train models on stacked (time, symbol) panel with time-based splits (no row leakage)."""

from __future__ import annotations

import copy
import time
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from quantauto.models.config import DEFAULT_TRAINING_CONFIGS, TrainingConfig
from quantauto.models.advanced_models import register_default_advanced_models
from quantauto.models.panel_dataset import PanelTrainingDataset
from quantauto.models.ranking_adapters import group_counts_for_lgbm, time_ids_for_catboost
from quantauto.models.registry import create_model, has_model
from quantauto.models.sklearn_models import register_default_sklearn_models
from quantauto.models.splits_panel import (
    list_unique_times_panel,
    resolve_panel_time_splits,
    time_split_to_row_mask,
)
from quantauto.models.types import ModelResult, TrainSplitResult, TrainerResult
from quantauto.validation.metrics import metric_preference, score_model
from quantauto.validation.walk_forward import aggregate_split_metrics


def _y_within_time_rank_relevance(y: pd.Series) -> pd.Series:
    """integer-ish ranks 0..n-1 per timestamp for lgbm lambdarank."""
    s = y.groupby(level=0, group_keys=False, sort=False).rank(method="first")
    return (s - 1.0).astype(float).rename("relevance_rank")


def _fit_predict_panel_fold(
    model: object,
    model_id: str,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    y_te: pd.Series,
) -> pd.Series:
    m = copy.deepcopy(model)
    if model_id == "lgbm_lambdarank":
        y_fit = _y_within_time_rank_relevance(y_tr)
        g = group_counts_for_lgbm(X_tr.index)
        m.fit(X_tr, y_fit, group=g)  # type: ignore[attr-defined]
    elif model_id == "cat_yetirank":
        gid = time_ids_for_catboost(X_tr.index)
        m.fit(X_tr, y_tr, group_id=gid)  # type: ignore[attr-defined]
    else:
        m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    if not isinstance(pred, pd.Series):
        pred = pd.Series(pred, index=X_te.index, name="pred")
    return pred.reindex(y_te.index)


def train_panel_models(
    dataset: PanelTrainingDataset,
    *,
    task_type: str = "ranking",
    model_ids: Optional[Iterable[str]] = None,
    config: Optional[TrainingConfig] = None,
) -> TrainerResult:
    """time-only CV on unique timestamps; all symbols at t share one train/test set."""
    register_default_sklearn_models()
    register_default_advanced_models()
    cfg = config or DEFAULT_TRAINING_CONFIGS.get(task_type) or DEFAULT_TRAINING_CONFIGS["ranking"]
    if task_type != cfg.task_type:
        cfg = TrainingConfig(
            task_type=task_type,
            model_ids=cfg.model_ids,
            primary_metric=cfg.primary_metric,
            maximize_metric=cfg.maximize_metric,
            test_split=cfg.test_split,
            walk_forward_folds=cfg.walk_forward_folds,
            purge_bars=cfg.purge_bars,
            embargo_bars=cfg.embargo_bars,
            training_time_budget_minutes=cfg.training_time_budget_minutes,
            enable_layer2=False,
        )
    times = list_unique_times_panel(dataset.X.index)
    splits: List[PanelTimeSplit] = resolve_panel_time_splits(
        times,
        test_split=cfg.test_split,
        walk_forward_folds=cfg.walk_forward_folds,
        purge_bars=cfg.purge_bars,
        embargo_bars=cfg.embargo_bars,
    )
    chosen = tuple(model_ids) if model_ids is not None else cfg.model_ids
    primary, maximize = metric_preference(task_type)
    budget_s = cfg.training_time_budget_minutes * 60.0
    start = time.perf_counter()
    all_results: Dict[str, ModelResult] = {}
    trained: List[str] = []
    skipped: List[str] = []
    for mid in chosen:
        if all_results and (time.perf_counter() - start) >= budget_s:
            skipped.append(mid)
            continue
        if not has_model(mid):
            skipped.append(mid)
            continue
        try:
            model = create_model(mid)
        except (ImportError, KeyError, ValueError, RuntimeError):
            skipped.append(mid)
            continue
        split_res: List[TrainSplitResult] = []
        for s in splits:
            m_tr, m_te = time_split_to_row_mask(
                dataset.X.index, s.train_times, s.test_times
            )
            X_tr = dataset.X[m_tr]
            y_tr = dataset.y[m_tr]
            X_te = dataset.X[m_te]
            y_te = dataset.y[m_te]
            if len(X_tr) < 2 or len(X_te) < 1:
                raise ValueError(
                    f"split {s.split_id} empty: train={len(X_tr)} test={len(X_te)}"
                )
            pr = _fit_predict_panel_fold(model, mid, X_tr, y_tr, X_te, y_te)
            met = score_model(task_type, y_te, pr)
            split_res.append(
                TrainSplitResult(
                    split_id=s.split_id,
                    train_size=len(X_tr),
                    test_size=len(X_te),
                    metrics=met,
                    predictions=pr,
                    actuals=y_te,
                )
            )
        agg = aggregate_split_metrics(split_res)
        if primary not in agg or not np.isfinite(float(agg[primary])):
            skipped.append(mid)
            continue
        pv = float(agg[primary])
        all_results[mid] = ModelResult(
            model_id=mid,
            task_type=task_type,
            split_results=tuple(split_res),
            aggregate_metrics=agg,
            primary_metric=primary,
            primary_metric_value=pv,
            fitted_model=None,
        )
        trained.append(mid)

    if not all_results:
        raise RuntimeError("no successful panel models; check deps and data size")

    if maximize:
        best_id = max(
            all_results.keys(),
            key=lambda m: float(all_results[m].aggregate_metrics.get(primary, float("nan"))),
        )
    else:
        best_id = min(
            all_results.keys(),
            key=lambda m: float(all_results[m].aggregate_metrics.get(primary, float("inf"))),
        )
    return TrainerResult(
        task_type=task_type,
        leaderboard=pd.DataFrame(
            [
                {
                    "model_id": m,
                    **all_results[m].aggregate_metrics,
                    "primary_metric": all_results[m].primary_metric_value,
                }
                for m in all_results
            ]
        )
        .sort_values(
            "primary_metric",
            ascending=not maximize,
        )
        .reset_index(drop=True),
        best_model_id=best_id,
        best_result=all_results[best_id],
        all_results=all_results,
        trained_model_ids=tuple(trained),
        skipped_model_ids=tuple(skipped),
        elapsed_seconds=float(time.perf_counter() - start),
        time_budget_minutes=float(cfg.training_time_budget_minutes),
        layer2_selected=False,
    )
