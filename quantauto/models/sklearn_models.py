"""scikit-learn model adapters and registry bootstrap."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from quantauto.models.base import ModelAdapter
from quantauto.models.config import DEFAULT_MODEL_SPECS
from quantauto.models.registry import has_model, register_model

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
except Exception:  # pragma: no cover - tested through availability checks
    RandomForestClassifier = None
    RandomForestRegressor = None
    LinearRegression = None
    LogisticRegression = None
    Ridge = None


class SklearnAdapter(ModelAdapter):
    def __init__(self, model_id: str, task_type: str, params: Mapping[str, Any]):
        self._model_id = model_id
        self._task_type = task_type
        self._params = dict(params)
        self._model = self._build_estimator()

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
        if self._model_id in ("linreg", "linreg_rk"):
            if LinearRegression is None:
                raise ImportError("scikit-learn is required for linreg")
            return LinearRegression(**self._params)
        if self._model_id in ("ridge", "ridge_rk"):
            if Ridge is None:
                raise ImportError("scikit-learn is required for ridge")
            return Ridge(**self._params)
        if self._model_id in ("rf_reg", "rf_rk"):
            if RandomForestRegressor is None:
                raise ImportError("scikit-learn is required for rf_reg")
            return RandomForestRegressor(**self._params)
        if self._model_id == "logreg":
            if LogisticRegression is None:
                raise ImportError("scikit-learn is required for logreg")
            return LogisticRegression(**self._params)
        if self._model_id == "rf_clf":
            if RandomForestClassifier is None:
                raise ImportError("scikit-learn is required for rf_clf")
            return RandomForestClassifier(**self._params)
        if self._model_id == "rf_clf_entropy":
            if RandomForestClassifier is None:
                raise ImportError("scikit-learn is required for rf_clf_entropy")
            return RandomForestClassifier(**self._params)
        raise KeyError(f"unsupported sklearn model_id {self._model_id!r}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnAdapter":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        pred = self._model.predict(X)
        return pd.Series(pred, index=X.index, name=f"pred_{self._model_id}")


def sklearn_factory(task_type: str, params: Mapping[str, object], model_id: str) -> SklearnAdapter:
    return SklearnAdapter(model_id=model_id, task_type=task_type, params=params)


def register_default_sklearn_models() -> None:
    for spec in DEFAULT_MODEL_SPECS:
        if spec.family != "sklearn" or has_model(spec.model_id):
            continue
        register_model(
            spec,
            factory=lambda task_type, params, mid=spec.model_id: sklearn_factory(
                task_type, params, mid
            ),
        )
