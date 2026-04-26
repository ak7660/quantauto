"""lightgbm lambdarank and catboost yetirank for cross-sectional ranking panels."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from quantauto.models.base import ModelAdapter

try:
    from lightgbm import LGBMRanker
except Exception:  # pragma: no cover
    LGBMRanker = None

try:
    from catboost import CatBoostRanker, Pool
except Exception:  # pragma: no cover
    CatBoostRanker = None
    Pool = None

RANKER_GROUP_MODEL_IDS: frozenset[str] = frozenset({"lgbm_lambdarank", "cat_yetirank"})


def _lgbm_ranker_params(overrides: Mapping[str, Any]) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "verbosity": -1,
        "n_estimators": 200,
        "num_leaves": 64,
        "learning_rate": 0.05,
        "min_data_in_leaf": 8,
    }
    base.update(dict(overrides))
    return base


class LGBMLambdaRankAdapter(ModelAdapter):
    """lambdarank: ``group`` = row counts per query (consecutive same timestamp)."""

    def __init__(self, model_id: str, task_type: str, params: Mapping[str, Any]) -> None:
        if LGBMRanker is None:
            raise ImportError("lightgbm is required for lgbm_lambdarank")
        self._model_id = model_id
        self._task_type = task_type
        self._params = _lgbm_ranker_params(params)
        self._est: Any = LGBMRanker(**self._params)  # type: ignore[misc]

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def task_type(self) -> str:
        return self._task_type

    @property
    def params(self) -> Mapping[str, Any]:
        return self._params

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        group: Optional[np.ndarray] = None,
    ) -> "LGBMLambdaRankAdapter":
        if group is None:
            raise ValueError("lambdarank requires group (counts per query, sorted rows)")
        self._est.fit(
            X.values,
            y.values,
            group=group,
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        pred = self._est.predict(X.values)
        return pd.Series(
            np.asarray(pred).ravel(), index=X.index, name=f"pred_{self._model_id}"
        )


class CatBoostYetiRankAdapter(ModelAdapter):
    """YetiRank with ``group_id`` = integer per timestamp block."""

    def __init__(self, model_id: str, task_type: str, params: Mapping[str, Any]) -> None:
        if CatBoostRanker is None or Pool is None:
            raise ImportError("catboost is required for cat_yetirank")
        self._model_id = model_id
        self._task_type = task_type
        self._params: Dict[str, Any] = {
            "loss_function": "YetiRank",
            "iterations": 200,
            "depth": 6,
            "learning_rate": 0.1,
            "verbose": False,
        }
        self._params.update(dict(params))
        ckeys = {k: v for k, v in self._params.items()}
        self._est: Any = CatBoostRanker(**ckeys)  # type: ignore[misc]

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def task_type(self) -> str:
        return self._task_type

    @property
    def params(self) -> Mapping[str, Any]:
        return self._params

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        group_id: Optional[np.ndarray] = None,
    ) -> "CatBoostYetiRankAdapter":
        if group_id is None:
            raise ValueError("YetiRank requires group_id (int per row, same for one timestamp)")
        tr = Pool(
            data=X,
            label=y,
            group_id=group_id,
        )
        self._est.fit(tr)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        pred = self._est.predict(X)
        return pd.Series(
            np.asarray(pred).ravel(), index=X.index, name=f"pred_{self._model_id}"
        )


def time_ids_for_catboost(index: pd.MultiIndex) -> np.ndarray:
    """``group_id`` for catboost: same int for all rows sharing a timestamp (query)."""
    t = index.get_level_values(0)
    codes, _ = pd.factorize(t, sort=False)
    return codes.astype(np.int32)


def group_counts_for_lgbm(index: pd.MultiIndex) -> np.ndarray:
    """lightgbm ``group`` array of counts in row order (sorted (time, symbol))."""
    if len(index) == 0:
        return np.zeros(0, dtype=np.int32)
    times = index.get_level_values(0)
    out: List[int] = []
    c = 1
    for i in range(1, len(times)):
        if times[i] == times[i - 1]:
            c += 1
        else:
            out.append(c)
            c = 1
    out.append(c)
    return np.array(out, dtype=np.int32)
