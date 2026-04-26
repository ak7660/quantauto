"""model training framework: adapters, registry, configs, trainer."""

from quantauto.models.base import ModelAdapter
from quantauto.models.advanced_models import register_default_advanced_models
from quantauto.models.config import (
    DEFAULT_MODEL_SPECS,
    DEFAULT_TRAINING_CONFIGS,
    TrainingConfig,
    parse_time_budget_minutes,
)
from quantauto.models.registry import (
    clear_registry,
    create_model,
    create_models,
    has_model,
    list_models,
    register_model,
)
from quantauto.models.sklearn_models import register_default_sklearn_models
from quantauto.models.panel_trainer import train_panel_models
from quantauto.models.trainer import train_models
from quantauto.models.types import ModelResult, ModelSpec, TrainSplitResult, TrainerResult

__all__ = [
    "ModelAdapter",
    "ModelSpec",
    "TrainSplitResult",
    "ModelResult",
    "TrainerResult",
    "TrainingConfig",
    "DEFAULT_MODEL_SPECS",
    "DEFAULT_TRAINING_CONFIGS",
    "parse_time_budget_minutes",
    "register_model",
    "clear_registry",
    "list_models",
    "has_model",
    "create_model",
    "create_models",
    "register_default_sklearn_models",
    "register_default_advanced_models",
    "train_models",
    "train_panel_models",
]
