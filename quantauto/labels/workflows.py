"""label workflow APIs built on low-level builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence, Tuple

import pandas as pd

from quantauto.labels.builders import (
    make_direction_label,
    make_forward_rank_label,
    make_forward_return_label,
)
from quantauto.labels.timing import (
    LabelMeta,
    check_label_feature_overlap,
    validate_label_timing,
)


@dataclass(frozen=True)
class LabelSpec:
    target_type: str  # regression | classification | ranking
    horizon: int = 1
    threshold: float = 0.0
    name: str = "label"
    params: Mapping[str, object] = field(default_factory=dict)


def build_label(close: pd.Series, spec: LabelSpec) -> Tuple[pd.Series, LabelMeta]:
    if spec.target_type == "regression":
        return make_forward_return_label(close, horizon=spec.horizon, name=spec.name)
    if spec.target_type == "classification":
        return make_direction_label(
            close,
            horizon=spec.horizon,
            threshold=spec.threshold,
            name=spec.name,
        )
    if spec.target_type == "ranking":
        return make_forward_rank_label(close, horizon=spec.horizon, name=spec.name)
    raise ValueError(f"unsupported target_type {spec.target_type!r}")


def build_label_set(
    close: pd.Series,
    specs: Sequence[LabelSpec],
) -> Tuple[Dict[str, pd.Series], Dict[str, LabelMeta]]:
    labels: Dict[str, pd.Series] = {}
    meta: Dict[str, LabelMeta] = {}
    for spec in specs:
        lbl, m = build_label(close, spec)
        labels[spec.name] = lbl
        meta[spec.name] = m
    return labels, meta


def validate_label_set(
    labels: Mapping[str, pd.Series],
    source_df: pd.DataFrame,
    *,
    horizons: Mapping[str, int],
    feature_names: Sequence[str] = (),
) -> Tuple[bool, Tuple[str, ...]]:
    errors = []
    for name, series in labels.items():
        horizon = int(horizons.get(name, 1))
        ok, msg = validate_label_timing(series, source_df, horizon=horizon)
        if not ok:
            errors.append(f"{name}: {msg}")
        safe, leak_msg = check_label_feature_overlap(name, feature_names)
        if not safe:
            errors.append(leak_msg)
    return len(errors) == 0, tuple(errors)
