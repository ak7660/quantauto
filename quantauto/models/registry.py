"""model registry and factory resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping

from quantauto.models.base import ModelAdapter
from quantauto.models.types import ModelSpec


ModelFactory = Callable[[str, Mapping[str, object]], ModelAdapter]


@dataclass(frozen=True)
class RegistryEntry:
    spec: ModelSpec
    factory: ModelFactory


_REGISTRY: Dict[str, RegistryEntry] = {}


def register_model(spec: ModelSpec, factory: ModelFactory) -> None:
    if spec.model_id in _REGISTRY:
        raise ValueError(f"model_id {spec.model_id!r} already registered")
    _REGISTRY[spec.model_id] = RegistryEntry(spec=spec, factory=factory)


def clear_registry() -> None:
    _REGISTRY.clear()


def list_models(task_type: str | None = None) -> List[ModelSpec]:
    specs = [entry.spec for entry in _REGISTRY.values()]
    if task_type is not None:
        specs = [s for s in specs if s.task_type == task_type]
    return sorted(specs, key=lambda s: s.model_id)


def has_model(model_id: str) -> bool:
    return model_id in _REGISTRY


def create_model(model_id: str) -> ModelAdapter:
    if model_id not in _REGISTRY:
        raise KeyError(f"unknown model_id {model_id!r}")
    entry = _REGISTRY[model_id]
    return entry.factory(entry.spec.task_type, dict(entry.spec.params))


def create_models(model_ids: Iterable[str]) -> Dict[str, ModelAdapter]:
    return {m: create_model(m) for m in model_ids}
