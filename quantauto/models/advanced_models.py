"""optional advanced model adapters (LightGBM/XGBoost/CatBoost/PyTorch)."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from quantauto.models.base import ModelAdapter
from quantauto.models.config import DEFAULT_MODEL_SPECS
from quantauto.models.registry import has_model, register_model

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:  # pragma: no cover
    CatBoostClassifier = None
    CatBoostRegressor = None

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


class ExternalEstimatorAdapter(ModelAdapter):
    """Adapter for sklearn-compatible estimators from external libraries."""

    def __init__(self, model_id: str, task_type: str, params: Mapping[str, Any]):
        self._model_id = model_id
        self._task_type = task_type
        self._params = dict(params)
        self._estimator = self._build_estimator()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def task_type(self) -> str:
        return self._task_type

    @property
    def params(self) -> Mapping[str, Any]:
        return self._params

    def _build_estimator(self) -> Any:
        if self._model_id in ("gbm_reg", "gbm_rk"):
            if LGBMRegressor is None:
                raise ImportError("lightgbm is required for gbm_reg")
            return LGBMRegressor(**self._params)
        if self._model_id == "gbm_clf":
            if LGBMClassifier is None:
                raise ImportError("lightgbm is required for gbm_clf")
            return LGBMClassifier(**self._params)
        if self._model_id in ("xgb_reg", "xgb_rk"):
            if XGBRegressor is None:
                raise ImportError("xgboost is required for xgb_reg")
            p = dict(self._params)
            p.pop("use_label_encoder", None)
            return XGBRegressor(**p)
        if self._model_id == "xgb_clf":
            if XGBClassifier is None:
                raise ImportError("xgboost is required for xgb_clf")
            return XGBClassifier(**self._params)
        if self._model_id in ("cat_reg", "cat_rk"):
            if CatBoostRegressor is None:
                raise ImportError("catboost is required for cat_reg")
            return CatBoostRegressor(**self._params)
        if self._model_id == "cat_clf":
            if CatBoostClassifier is None:
                raise ImportError("catboost is required for cat_clf")
            return CatBoostClassifier(**self._params)
        raise KeyError(f"unsupported external model_id {self._model_id!r}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ExternalEstimatorAdapter":
        self._estimator.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        pred = self._estimator.predict(X)
        pred = np.asarray(pred).reshape(-1)
        return pd.Series(pred, index=X.index, name=f"pred_{self._model_id}")


class TorchTabularAdapter(ModelAdapter):
    """Small PyTorch MLP adapter for tabular data."""

    def __init__(self, model_id: str, task_type: str, params: Mapping[str, Any]):
        if torch is None or nn is None:
            raise ImportError("pytorch is required for nn_torch models")
        self._model_id = model_id
        self._task_type = task_type
        self._params = dict(params)
        self._net: nn.Module | None = None
        self._device = torch.device("cpu")

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def task_type(self) -> str:
        return self._task_type

    @property
    def params(self) -> Mapping[str, Any]:
        return self._params

    def _build_net(self, in_dim: int) -> nn.Module:
        hidden = int(self._params.get("hidden_dim", 128))
        dropout = float(self._params.get("dropout_prob", 0.1))
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        ).to(self._device)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TorchTabularAdapter":
        seed = int(self._params.get("random_state", 42))
        torch.manual_seed(seed)
        epochs = int(self._params.get("num_epochs", 200))
        lr = float(self._params.get("learning_rate", 3e-4))
        wd = float(self._params.get("weight_decay", 1e-6))
        self._net = self._build_net(X.shape[1])
        opt = torch.optim.Adam(self._net.parameters(), lr=lr, weight_decay=wd)
        if self._task_type in ("classification",):
            loss_fn: Any = nn.BCEWithLogitsLoss()
            target = y.astype(float).values.reshape(-1, 1)
        else:
            loss_fn = nn.MSELoss()
            target = y.astype(float).values.reshape(-1, 1)
        x_t = torch.tensor(X.values, dtype=torch.float32, device=self._device)
        y_t = torch.tensor(target, dtype=torch.float32, device=self._device)
        self._net.train()
        for _ in range(epochs):
            opt.zero_grad()
            out = self._net(x_t)
            loss = loss_fn(out, y_t)
            loss.backward()
            opt.step()
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        assert self._net is not None
        self._net.eval()
        with torch.no_grad():
            x_t = torch.tensor(X.values, dtype=torch.float32, device=self._device)
            out = self._net(x_t).cpu().numpy().reshape(-1)
        if self._task_type == "classification":
            pred = (1.0 / (1.0 + np.exp(-out)) >= 0.5).astype(int)
        else:
            pred = out
        return pd.Series(np.asarray(pred).ravel(), index=X.index, name=f"pred_{self._model_id}")


def register_default_advanced_models() -> None:
    for spec in DEFAULT_MODEL_SPECS:
        if spec.family not in {"lightgbm", "xgboost", "catboost", "torch"}:
            continue
        if has_model(spec.model_id):
            continue
        if spec.model_id == "nn_torch_rk":
            register_model(
                spec,
                factory=lambda task_type, params, mid=spec.model_id: TorchTabularAdapter(
                    model_id=mid, task_type=task_type, params=params
                ),
            )
            continue
        if spec.family == "torch":
            register_model(
                spec,
                factory=lambda task_type, params, mid=spec.model_id: TorchTabularAdapter(
                    model_id=mid, task_type=task_type, params=params
                ),
            )
            continue
        register_model(
            spec,
            factory=lambda task_type, params, mid=spec.model_id: ExternalEstimatorAdapter(
                model_id=mid, task_type=task_type, params=params
            ),
        )
    _register_lambdarank_yetirank()


def _register_lambdarank_yetirank() -> None:
    from quantauto.models.ranking_adapters import CatBoostYetiRankAdapter, LGBMLambdaRankAdapter

    for spec in DEFAULT_MODEL_SPECS:
        if spec.model_id == "lgbm_lambdarank" and not has_model(spec.model_id):
            register_model(
                spec,
                factory=lambda task_type, params, mid=spec.model_id: LGBMLambdaRankAdapter(
                    model_id=mid, task_type=task_type, params=params
                ),
            )
        if spec.model_id == "cat_yetirank" and not has_model(spec.model_id):
            register_model(
                spec,
                factory=lambda task_type, params, mid=spec.model_id: CatBoostYetiRankAdapter(
                    model_id=mid, task_type=task_type, params=params
                ),
            )
