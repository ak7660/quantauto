"""base interfaces for trainable model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import pandas as pd


class ModelAdapter(ABC):
    """minimal fit/predict interface used by the trainer."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        pass

    @property
    @abstractmethod
    def task_type(self) -> str:
        pass

    @property
    @abstractmethod
    def params(self) -> Mapping[str, Any]:
        pass

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModelAdapter":
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass
