"""multi-model training orchestration."""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from quantauto.models.config import DEFAULT_TRAINING_CONFIGS, TrainingConfig
from quantauto.models.advanced_models import register_default_advanced_models
from quantauto.models.dataset import TrainingDataset
from quantauto.models.registry import create_model
from quantauto.models.sklearn_models import register_default_sklearn_models
from quantauto.models.splits import TimeSplit, make_holdout_split, make_walk_forward_splits
from quantauto.models.types import ModelResult, TrainSplitResult, TrainerResult
from quantauto.validation.metrics import metric_preference, score_model
from quantauto.validation.walk_forward import aggregate_split_metrics, run_splits


def _resolve_splits(dataset: TrainingDataset, cfg: TrainingConfig) -> List[TimeSplit]:
    if cfg.walk_forward_folds <= 1:
        return [make_holdout_split(dataset.index, test_split=cfg.test_split)]
    test_size = max(8, int(len(dataset.index) * cfg.test_split))
    min_train_size = max(
        32,
        len(dataset.index) - (cfg.walk_forward_folds * (test_size + cfg.embargo_bars)),
    )
    return make_walk_forward_splits(
        dataset.index,
        n_folds=cfg.walk_forward_folds,
        test_size=test_size,
        min_train_size=min_train_size,
        purge_bars=cfg.purge_bars,
        embargo_bars=cfg.embargo_bars,
    )


def train_models(
    dataset: TrainingDataset,
    *,
    task_type: str,
    model_ids: Optional[Iterable[str]] = None,
    config: Optional[TrainingConfig] = None,
) -> TrainerResult:
    register_default_sklearn_models()
    register_default_advanced_models()
    cfg = config or DEFAULT_TRAINING_CONFIGS[task_type]
    chosen_ids = tuple(model_ids) if model_ids is not None else cfg.model_ids
    splits = _resolve_splits(dataset, cfg)
    primary_metric, maximize = metric_preference(task_type)
    start_ts = time.perf_counter()
    budget_s = cfg.training_time_budget_minutes * 60.0

    all_results: Dict[str, ModelResult] = {}
    leaderboard_rows = []
    trained_ids: List[str] = []
    skipped_ids: List[str] = []
    for model_id in chosen_ids:
        if all_results and (time.perf_counter() - start_ts) >= budget_s:
            skipped_ids.append(model_id)
            continue
        try:
            model = create_model(model_id)
        except ImportError:
            skipped_ids.append(model_id)
            continue
        split_results = run_splits(model, dataset.X, dataset.y, splits)
        agg = aggregate_split_metrics(split_results)
        if primary_metric not in agg or not np.isfinite(float(agg[primary_metric])):
            skipped_ids.append(model_id)
            continue
        primary_value = float(agg[primary_metric])
        model_result = ModelResult(
            model_id=model_id,
            task_type=task_type,
            split_results=split_results,
            aggregate_metrics=agg,
            primary_metric=primary_metric,
            primary_metric_value=primary_value,
            fitted_model=model,
        )
        all_results[model_id] = model_result
        trained_ids.append(model_id)
        leaderboard_rows.append({"model_id": model_id, **agg, "primary_metric": primary_value})

    if not all_results:
        raise RuntimeError("no trainable models available; check dependencies and model selection")

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        by="primary_metric", ascending=(not maximize)
    )

    layer2_selected = False
    if cfg.enable_layer2 and len(all_results) >= 2:
        layer2_result = _build_best_layer2(
            all_results=all_results,
            task_type=task_type,
            primary_metric=primary_metric,
            maximize=maximize,
        )
        if layer2_result is not None:
            base_best = float(leaderboard.iloc[0]["primary_metric"])
            improved = (
                layer2_result.primary_metric_value > base_best
                if maximize
                else layer2_result.primary_metric_value < base_best
            )
            if improved:
                all_results[layer2_result.model_id] = layer2_result
                leaderboard_rows.append(
                    {
                        "model_id": layer2_result.model_id,
                        **layer2_result.aggregate_metrics,
                        "primary_metric": layer2_result.primary_metric_value,
                    }
                )
                leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
                    by="primary_metric", ascending=(not maximize)
                )
                layer2_selected = True

    best_model_id = str(leaderboard.iloc[0]["model_id"])
    return TrainerResult(
        task_type=task_type,
        leaderboard=leaderboard.reset_index(drop=True),
        best_model_id=best_model_id,
        best_result=all_results[best_model_id],
        all_results=all_results,
        trained_model_ids=tuple(trained_ids),
        skipped_model_ids=tuple(skipped_ids),
        elapsed_seconds=float(time.perf_counter() - start_ts),
        time_budget_minutes=float(cfg.training_time_budget_minutes),
        layer2_selected=layer2_selected,
    )


def _build_best_layer2(
    *,
    all_results: Dict[str, ModelResult],
    task_type: str,
    primary_metric: str,
    maximize: bool,
) -> Optional[ModelResult]:
    cands: List[tuple[str, list[pd.Series]]] = []
    model_ids = list(all_results.keys())
    n_splits = len(next(iter(all_results.values())).split_results)
    for split_idx in range(n_splits):
        pred_frames = []
        actual = None
        for mid in model_ids:
            sr = all_results[mid].split_results[split_idx]
            pred_frames.append(sr.predictions.rename(mid))
            if actual is None:
                actual = sr.actuals
        assert actual is not None
        pred_df = pd.concat(pred_frames, axis=1).dropna()
        actual_aligned = actual.reindex(pred_df.index)
        if task_type == "regression":
            mean_pred = pred_df.mean(axis=1).rename("pred")
            median_pred = pred_df.median(axis=1).rename("pred")
            cands.append(("layer2_mean", [mean_pred, actual_aligned]))
            cands.append(("layer2_median", [median_pred, actual_aligned]))
            inv_rmse = []
            for mid in model_ids:
                rmse = all_results[mid].aggregate_metrics.get("rmse", 1.0)
                inv_rmse.append(1.0 / max(1e-9, float(rmse)))
            w = np.array(inv_rmse, dtype=float)
            w = w / w.sum()
            weighted = pd.Series(pred_df.values @ w, index=pred_df.index, name="pred")
            cands.append(("layer2_weighted", [weighted, actual_aligned]))
        else:
            vote = (pred_df.mean(axis=1) >= 0.5).astype(int).rename("pred")
            unanimity = (pred_df.sum(axis=1) == len(model_ids)).astype(int).rename("pred")
            cands.append(("layer2_vote", [vote, actual_aligned]))
            cands.append(("layer2_unanimity", [unanimity, actual_aligned]))

    by_name: Dict[str, list] = {}
    for name, pair in cands:
        by_name.setdefault(name, []).append(pair)

    best: Optional[ModelResult] = None
    for name, pairs in by_name.items():
        split_results = []
        for i, (pred, actual) in enumerate(pairs):
            met = score_model(task_type, actual, pred)
            split_results.append(
                TrainSplitResult(
                    split_id=f"l2_{i+1}",
                    train_size=all_results[model_ids[0]].split_results[i].train_size,
                    test_size=len(pred),
                    metrics=met,
                    predictions=pred,
                    actuals=actual,
                )
            )
        agg = aggregate_split_metrics(split_results)
        score = float(agg[primary_metric])
        candidate = ModelResult(
            model_id=name,
            task_type=task_type,
            split_results=tuple(split_results),
            aggregate_metrics=agg,
            primary_metric=primary_metric,
            primary_metric_value=score,
            fitted_model=None,
        )
        if best is None:
            best = candidate
            continue
        better = score > best.primary_metric_value if maximize else score < best.primary_metric_value
        if better:
            best = candidate
    return best
