"""default model configuration presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Union

from quantauto.models.types import ModelSpec


@dataclass(frozen=True)
class TrainingConfig:
    task_type: str
    model_ids: Tuple[str, ...]
    primary_metric: str
    maximize_metric: bool
    test_split: float = 0.2
    walk_forward_folds: int = 1
    purge_bars: int = 0
    embargo_bars: int = 0
    training_time_budget_minutes: float = 30.0
    enable_layer2: bool = True


def parse_time_budget_minutes(value: Union[str, int, float]) -> float:
    """parse minutes from values like 30, '30m', '10min', '0.5h'."""
    if isinstance(value, (int, float)):
        out = float(value)
        if out <= 0:
            raise ValueError("training_time_budget must be > 0 minutes")
        return out
    raw = str(value).strip().lower()
    if raw.endswith("minutes"):
        out = float(raw[:-7].strip())
    elif raw.endswith("minute"):
        out = float(raw[:-6].strip())
    elif raw.endswith("mins"):
        out = float(raw[:-4].strip())
    elif raw.endswith("min"):
        out = float(raw[:-3].strip())
    elif raw.endswith("m"):
        out = float(raw[:-1].strip())
    elif raw.endswith("hours"):
        out = float(raw[:-5].strip()) * 60.0
    elif raw.endswith("hour"):
        out = float(raw[:-4].strip()) * 60.0
    elif raw.endswith("hrs"):
        out = float(raw[:-3].strip()) * 60.0
    elif raw.endswith("hr"):
        out = float(raw[:-2].strip()) * 60.0
    elif raw.endswith("h"):
        out = float(raw[:-1].strip()) * 60.0
    else:
        out = float(raw)
    if out <= 0:
        raise ValueError("training_time_budget must be > 0 minutes")
    return out


DEFAULT_MODEL_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="gbm_reg",
        task_type="regression",
        family="lightgbm",
        params={
            "num_leaves": 128,
            "learning_rate": 0.05,
            "n_estimators": 10000,
            "min_child_samples": 20,
            "objective": "regression",
            "metric": "rmse",
        },
    ),
    ModelSpec(
        model_id="xgb_reg",
        task_type="regression",
        family="xgboost",
        params={
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 10000,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
        },
    ),
    ModelSpec(
        model_id="cat_reg",
        task_type="regression",
        family="catboost",
        params={
            "iterations": 10000,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "verbose": False,
        },
    ),
    ModelSpec(
        model_id="nn_torch_reg",
        task_type="regression",
        family="torch",
        params={
            "num_epochs": 200,
            "learning_rate": 3e-4,
            "dropout_prob": 0.1,
            "weight_decay": 1e-6,
            "batch_size": 512,
            "hidden_dim": 128,
            "random_state": 42,
        },
    ),
    ModelSpec(
        model_id="rf_reg",
        task_type="regression",
        family="sklearn",
        params={
            "n_estimators": 300,
            "max_features": "sqrt",
            "min_samples_leaf": 5,
            "max_depth": None,
            "criterion": "squared_error",
            "n_jobs": -1,
            "bootstrap": True,
            "random_state": 42,
        },
    ),
    ModelSpec(
        model_id="linreg",
        task_type="regression",
        family="sklearn",
        params={},
    ),
    ModelSpec(
        model_id="gbm_clf",
        task_type="classification",
        family="lightgbm",
        params={
            "num_leaves": 128,
            "learning_rate": 0.05,
            "n_estimators": 10000,
            "objective": "binary",
            "metric": "binary_logloss",
            "class_weight": None,
            "is_unbalance": False,
        },
    ),
    ModelSpec(
        model_id="xgb_clf",
        task_type="classification",
        family="xgboost",
        params={
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 10000,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "tree_method": "hist",
        },
    ),
    ModelSpec(
        model_id="cat_clf",
        task_type="classification",
        family="catboost",
        params={
            "iterations": 10000,
            "depth": 6,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "verbose": False,
        },
    ),
    ModelSpec(
        model_id="nn_torch_clf",
        task_type="classification",
        family="torch",
        params={
            "num_epochs": 200,
            "learning_rate": 3e-4,
            "dropout_prob": 0.1,
            "weight_decay": 1e-6,
            "batch_size": 512,
            "hidden_dim": 128,
            "random_state": 42,
        },
    ),
    ModelSpec(
        model_id="ridge",
        task_type="regression",
        family="sklearn",
        params={"alpha": 1.0, "random_state": 42},
    ),
    ModelSpec(
        model_id="rf_reg",
        task_type="regression",
        family="sklearn",
        params={"n_estimators": 200, "max_depth": 6, "random_state": 42},
    ),
    ModelSpec(
        model_id="logreg",
        task_type="classification",
        family="sklearn",
        params={"C": 1.0, "max_iter": 400, "random_state": 42},
    ),
    ModelSpec(
        model_id="rf_clf",
        task_type="classification",
        family="sklearn",
        params={
            "n_estimators": 300,
            "criterion": "gini",
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "max_depth": None,
            "class_weight": None,
            "n_jobs": -1,
            "random_state": 42,
        },
    ),
    ModelSpec(
        model_id="rf_clf_entropy",
        task_type="classification",
        family="sklearn",
        params={
            "n_estimators": 300,
            "criterion": "entropy",
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "max_depth": None,
            "class_weight": None,
            "n_jobs": -1,
            "random_state": 42,
        },
    ),
    # pooled cross-sectional ranking: same params as regression family, different task
    ModelSpec(
        model_id="linreg_rk",
        task_type="ranking",
        family="sklearn",
        params={},
    ),
    ModelSpec(
        model_id="ridge_rk",
        task_type="ranking",
        family="sklearn",
        params={"alpha": 1.0, "random_state": 42},
    ),
    ModelSpec(
        model_id="rf_rk",
        task_type="ranking",
        family="sklearn",
        params={
            "n_estimators": 200,
            "max_depth": 6,
            "random_state": 42,
        },
    ),
    ModelSpec(
        model_id="gbm_rk",
        task_type="ranking",
        family="lightgbm",
        params={
            "num_leaves": 128,
            "learning_rate": 0.05,
            "n_estimators": 2000,
            "min_child_samples": 20,
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
        },
    ),
    ModelSpec(
        model_id="xgb_rk",
        task_type="ranking",
        family="xgboost",
        params={
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 2000,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
        },
    ),
    ModelSpec(
        model_id="cat_rk",
        task_type="ranking",
        family="catboost",
        params={
            "iterations": 2000,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "verbose": False,
        },
    ),
    ModelSpec(
        model_id="nn_torch_rk",
        task_type="ranking",
        family="torch",
        params={
            "num_epochs": 200,
            "learning_rate": 3e-4,
            "dropout_prob": 0.1,
            "weight_decay": 1e-6,
            "batch_size": 512,
            "hidden_dim": 128,
            "random_state": 42,
        },
    ),
    ModelSpec(
        model_id="lgbm_lambdarank",
        task_type="ranking",
        family="lambdarank",
        params={},
    ),
    ModelSpec(
        model_id="cat_yetirank",
        task_type="ranking",
        family="yetirank",
        params={},
    ),
)


DEFAULT_TRAINING_CONFIGS: Dict[str, TrainingConfig] = {
    "regression": TrainingConfig(
        task_type="regression",
        model_ids=("gbm_reg", "xgb_reg", "cat_reg", "nn_torch_reg", "rf_reg", "linreg", "ridge"),
        primary_metric="rmse",
        maximize_metric=False,
    ),
    "classification": TrainingConfig(
        task_type="classification",
        model_ids=(
            "gbm_clf",
            "xgb_clf",
            "cat_clf",
            "nn_torch_clf",
            "rf_clf",
            "rf_clf_entropy",
            "logreg",
        ),
        primary_metric="accuracy",
        maximize_metric=True,
    ),
    "ranking": TrainingConfig(
        task_type="ranking",
        model_ids=(
            "linreg_rk",
            "ridge_rk",
            "gbm_rk",
            "lgbm_lambdarank",
            "cat_yetirank",
            "xgb_rk",
            "cat_rk",
        ),
        primary_metric="spearman",
        maximize_metric=True,
    ),
}

