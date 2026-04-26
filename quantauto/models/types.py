"""typed containers for model training outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import pandas as pd


TaskType = str  # "regression" | "classification" | "ranking"


@dataclass(frozen=True)
class ModelSpec:
    """registration metadata for one trainable model."""

    model_id: str
    task_type: TaskType
    family: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class TrainSplitResult:
    """model outputs for one split/fold."""

    split_id: str
    train_size: int
    test_size: int
    metrics: Dict[str, float]
    predictions: pd.Series
    actuals: pd.Series


@dataclass
class ModelResult:
    """aggregated result for a single model id."""

    model_id: str
    task_type: TaskType
    split_results: Tuple[TrainSplitResult, ...]
    aggregate_metrics: Dict[str, float]
    primary_metric: str
    primary_metric_value: float
    fitted_model: Optional[Any] = None


@dataclass
class TrainerResult:
    """full outcome of multi-model training run."""

    task_type: TaskType
    leaderboard: pd.DataFrame
    best_model_id: str
    best_result: ModelResult
    all_results: Dict[str, ModelResult]
    trained_model_ids: Tuple[str, ...]
    skipped_model_ids: Tuple[str, ...]
    elapsed_seconds: float
    time_budget_minutes: float
    layer2_selected: bool = False
