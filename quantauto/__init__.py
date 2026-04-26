"""QuantAuto: A Python Library for ML-Driven Trading Signal Research and High-Performance Backtesting."""

from quantauto.data import (
    align_multi_asset_to_common_index,
    load_market_data,
    load_multi_market_data,
)
from quantauto.features import (
    AVAILABLE_FEATURES,
    BASE_PRESET_SPECS,
    CustomFeatureSpec,
    DEFAULT_PRESET,
    EngineeredFeatures,
    FeatureSpec,
    build_features,
    build_multi_asset_features,
    build_multi_preset_features,
    build_preset_features,
    make_custom_feature,
    make_preset_specs,
)
from quantauto.workflows import (
    MultiPipelineResults,
    PipelineResults,
    RankingPanelResults,
    run_auto,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "load_market_data",
    "load_multi_market_data",
    "align_multi_asset_to_common_index",
    "FeatureSpec",
    "CustomFeatureSpec",
    "EngineeredFeatures",
    "AVAILABLE_FEATURES",
    "DEFAULT_PRESET",
    "BASE_PRESET_SPECS",
    "build_features",
    "build_multi_asset_features",
    "make_preset_specs",
    "make_custom_feature",
    "build_preset_features",
    "build_multi_preset_features",
    "PipelineResults",
    "MultiPipelineResults",
    "RankingPanelResults",
    "run_auto",
]
