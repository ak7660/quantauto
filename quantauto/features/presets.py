"""preset feature bundles (``base`` / ``heavy``), column-aware filtering, and custom-feature helper."""

from __future__ import annotations

from copy import deepcopy
from typing import (
    Callable,
    Collection,
    Dict,
    FrozenSet,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pandas as pd

from quantauto.data.schema import LoadedMarketData, MultiAssetMarketData
from quantauto.features.engineering import (
    CustomFeatureSpec,
    EngineeredFeatures,
    FeatureSpec,
    build_features,
)

PresetName = Literal["base", "heavy"]

DEFAULT_PRESET: PresetName = "base"

# registered kind -> minimum canonical OHLCV columns (default params)
KIND_REQUIRED_COLUMNS: Dict[str, FrozenSet[str]] = {
    "returns": frozenset({"close"}),
    "log_returns": frozenset({"close"}),
    "sma": frozenset(),  # resolved in _roles_for_spec via params["column"]
    "ema": frozenset(),
    "rolling_std": frozenset(),
    "rolling_zscore": frozenset(),
    "rate_of_change": frozenset({"close"}),
    "rsi": frozenset({"close"}),
    "macd": frozenset({"close"}),
    "bollinger_bands": frozenset({"close"}),
    "atr": frozenset({"high", "low", "close"}),
    "obv": frozenset({"close", "volume"}),
    "volume_ratio": frozenset({"volume"}),
    "high_low_range": frozenset({"high", "low", "close"}),
    "close_position": frozenset({"open", "high", "low", "close"}),
}

# default bundle: one spec per registered family with sensible defaults
BASE_PRESET_SPECS: Tuple[FeatureSpec, ...] = (
    FeatureSpec("ret_1", "returns", {"period": 1}),
    FeatureSpec("log_ret_1", "log_returns", {"period": 1}),
    FeatureSpec("sma_20", "sma", {"window": 20}),
    FeatureSpec("ema_12", "ema", {"span": 12}),
    FeatureSpec("std_20", "rolling_std", {"window": 20}),
    FeatureSpec("zscore_20", "rolling_zscore", {"window": 20}),
    FeatureSpec("roc_10", "rate_of_change", {"period": 10}),
    FeatureSpec("rsi_14", "rsi", {"window": 14}),
    FeatureSpec("macd", "macd", {"fast": 12, "slow": 26, "signal": 9}),
    FeatureSpec("bb", "bollinger_bands", {"window": 20, "n_std": 2.0}),
    FeatureSpec("atr_14", "atr", {"window": 14}),
    FeatureSpec("obv", "obv"),
    FeatureSpec("vol_ratio", "volume_ratio", {"window": 20}),
    FeatureSpec("hl_range", "high_low_range"),
    FeatureSpec("close_pos", "close_position"),
)

# three alternative parameter sets per kind (short / mid / long). kinds omitted are not triplicated.
_HEAVY_TRIPLETS: Dict[str, Tuple[Dict[str, Union[int, float]], ...]] = {
    "returns": ({"period": 1}, {"period": 5}, {"period": 21}),
    "log_returns": ({"period": 1}, {"period": 5}, {"period": 21}),
    "sma": ({"window": 10}, {"window": 20}, {"window": 50}),
    "ema": ({"span": 8}, {"span": 12}, {"span": 26}),
    "rolling_std": ({"window": 10}, {"window": 20}, {"window": 50}),
    "rolling_zscore": ({"window": 10}, {"window": 20}, {"window": 50}),
    "rate_of_change": ({"period": 5}, {"period": 10}, {"period": 20}),
    "rsi": ({"window": 7}, {"window": 14}, {"window": 21}),
    "macd": (
        {"fast": 8, "slow": 17, "signal": 9},
        {"fast": 12, "slow": 26, "signal": 9},
        {"fast": 19, "slow": 39, "signal": 9},
    ),
    "bollinger_bands": (
        {"window": 10, "n_std": 2.0},
        {"window": 20, "n_std": 2.0},
        {"window": 30, "n_std": 2.0},
    ),
    "atr": ({"window": 7}, {"window": 14}, {"window": 21}),
    "volume_ratio": ({"window": 10}, {"window": 20}, {"window": 50}),
}

_HEAVY_NAME_SUFFIXES = ("_s", "_m", "_l")


def _roles_for_spec(spec: FeatureSpec) -> FrozenSet[str]:
    """columns required for this spec (canonical kline names)."""
    kind = spec.kind
    if kind in {"sma", "ema", "rolling_std", "rolling_zscore"}:
        col = spec.params.get("column", "close")
        return frozenset({str(col)})
    base = KIND_REQUIRED_COLUMNS.get(kind)
    if base is None:
        raise KeyError(f"unknown feature kind {kind!r} for column requirements")
    return base


def spec_fits_columns(spec: FeatureSpec, columns: Collection[str]) -> bool:
    """true if ``loaded.data`` has every column this spec needs."""
    need = _roles_for_spec(spec)
    return need <= set(columns)


def custom_fits_columns(spec: CustomFeatureSpec, columns: Collection[str]) -> bool:
    if spec.required_columns is None:
        return True
    return set(spec.required_columns) <= set(columns)


def filter_specs_to_loaded(
    specs: Sequence[FeatureSpec],
    loaded: LoadedMarketData,
) -> List[FeatureSpec]:
    """drops specs whose required OHLCV columns are missing from ``loaded.data``."""
    cols = loaded.data.columns
    return [deepcopy(s) for s in specs if spec_fits_columns(s, cols)]


def filter_custom_to_loaded(
    custom_specs: Sequence[CustomFeatureSpec],
    loaded: LoadedMarketData,
) -> List[CustomFeatureSpec]:
    cols = loaded.data.columns
    return [deepcopy(c) for c in custom_specs if custom_fits_columns(c, cols)]


def _merge_params(
    base: Mapping[str, Union[int, float]],
    override: Optional[Mapping[str, Union[int, float]]],
) -> Dict[str, Union[int, float]]:
    if not override:
        return dict(base)
    return {**base, **override}


def _apply_kind_params(
    spec: FeatureSpec,
    kind_params: Optional[Mapping[str, Mapping[str, Union[int, float]]]],
) -> FeatureSpec:
    if not kind_params or spec.kind not in kind_params:
        return spec
    new_params = _merge_params(spec.params, kind_params[spec.kind])
    return FeatureSpec(
        name=spec.name,
        kind=spec.kind,
        params=new_params,
        lookback=spec.lookback,
        description=spec.description,
    )


def expand_preset_specs(preset: PresetName) -> List[FeatureSpec]:
    """materialises feature specs for a preset (no column filtering)."""
    if preset == "base":
        return [deepcopy(s) for s in BASE_PRESET_SPECS]

    if preset != "heavy":
        raise ValueError(f"preset must be 'base' or 'heavy', got {preset!r}")

    out: List[FeatureSpec] = []
    for spec in BASE_PRESET_SPECS:
        trips = _HEAVY_TRIPLETS.get(spec.kind)
        if trips is None:
            out.append(deepcopy(spec))
            continue
        for suf, pextra in zip(_HEAVY_NAME_SUFFIXES, trips):
            merged = {**spec.params, **pextra}
            out.append(
                FeatureSpec(
                    name=spec.name + suf,
                    kind=spec.kind,
                    params=merged,
                    lookback=spec.lookback,
                    description=spec.description,
                )
            )
    return out


def make_preset_specs(
    preset: PresetName = DEFAULT_PRESET,
    *,
    include_kinds: Optional[Collection[str]] = None,
    exclude_kinds: Optional[Collection[str]] = None,
    kind_params: Optional[Mapping[str, Mapping[str, Union[int, float]]]] = None,
) -> List[FeatureSpec]:
    """builds a ``FeatureSpec`` list for ``preset`` with optional kind filters and param overrides.

    include_kinds -- if set, only these feature kinds are kept
    exclude_kinds -- kinds to remove (applied after include)
    kind_params   -- per-kind param dict merged into each matching spec
    """
    specs = expand_preset_specs(preset)
    if include_kinds is not None:
        inc = set(include_kinds)
        specs = [s for s in specs if s.kind in inc]
    if exclude_kinds is not None:
        exc = set(exclude_kinds)
        specs = [s for s in specs if s.kind not in exc]
    if kind_params:
        specs = [_apply_kind_params(s, kind_params) for s in specs]
    return specs


def make_custom_feature(
    name: str,
    compute: Callable[[pd.DataFrame], Union[pd.Series, Dict[str, pd.Series]]],
    *,
    lookback: int,
    description: str = "",
    required_columns: Optional[Sequence[str]] = None,
) -> CustomFeatureSpec:
    """factory for :class:`CustomFeatureSpec` (optional user-defined columns).

    ``compute`` receives ``loaded.data`` and should return a ``Series`` or
    ``dict`` of suffix -> ``Series`` (suffix ``""`` for a single column ``name``).
    """
    return CustomFeatureSpec(
        name=name,
        compute=compute,
        lookback=lookback,
        description=description,
        required_columns=tuple(required_columns) if required_columns is not None else None,
    )


def build_preset_features(
    loaded: LoadedMarketData,
    preset: PresetName = DEFAULT_PRESET,
    *,
    include_kinds: Optional[Collection[str]] = None,
    exclude_kinds: Optional[Collection[str]] = None,
    kind_params: Optional[Mapping[str, Mapping[str, Union[int, float]]]] = None,
    custom_specs: Sequence[CustomFeatureSpec] = (),
    filter_to_available_columns: bool = True,
    execution_shift: int = 0,
    drop_warmup: bool = False,
) -> EngineeredFeatures:
    """builds engineered features from a preset plus optional custom features.

    by default only specs whose required kline columns exist on ``loaded.data`` are used.
    if everything is filtered out and there are no custom specs, returns an empty feature frame
    (same index, zero columns) with ``warmup_bars=0``.
    """
    specs = make_preset_specs(
        preset,
        include_kinds=include_kinds,
        exclude_kinds=exclude_kinds,
        kind_params=kind_params,
    )
    if filter_to_available_columns:
        specs = filter_specs_to_loaded(specs, loaded)
        customs = filter_custom_to_loaded(custom_specs, loaded)
    else:
        customs = list(custom_specs)

    if not specs and not customs:
        return EngineeredFeatures(
            data=pd.DataFrame(index=loaded.data.index),
            meta=tuple(),
            warmup_bars=0,
        )

    return build_features(
        loaded,
        specs,
        execution_shift=execution_shift,
        drop_warmup=drop_warmup,
        custom_specs=customs,
    )


def build_multi_preset_features(
    multi: MultiAssetMarketData,
    preset: PresetName = DEFAULT_PRESET,
    *,
    include_kinds: Optional[Collection[str]] = None,
    exclude_kinds: Optional[Collection[str]] = None,
    kind_params: Optional[Mapping[str, Mapping[str, Union[int, float]]]] = None,
    custom_specs: Sequence[CustomFeatureSpec] = (),
    filter_to_available_columns: bool = True,
    execution_shift: int = 0,
    drop_warmup: bool = False,
) -> Dict[str, EngineeredFeatures]:
    """like :func:`build_preset_features` but for each symbol in ``multi``."""
    out: Dict[str, EngineeredFeatures] = {}
    for sym, loaded in multi.items():
        ef = build_preset_features(
            loaded,
            preset,
            include_kinds=include_kinds,
            exclude_kinds=exclude_kinds,
            kind_params=kind_params,
            custom_specs=custom_specs,
            filter_to_available_columns=filter_to_available_columns,
            execution_shift=execution_shift,
            drop_warmup=drop_warmup,
        )
        out[sym] = EngineeredFeatures(
            data=ef.data,
            meta=ef.meta,
            warmup_bars=ef.warmup_bars,
            source_symbol=sym,
        )
    return out
