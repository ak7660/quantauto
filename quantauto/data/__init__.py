"""Data schema, loaders, and validation."""

from quantauto.data.align import AlignMultiAssetResult, align_multi_asset_to_common_index
from quantauto.data.loaders import load_market_data, load_multi_market_data
from quantauto.data.schema import KLINE_ROLES, LoadedMarketData, MultiAssetMarketData
from quantauto.data.validator import (
    ValidationResult,
    validate_loaded_market_data,
    validate_market_frame,
    validate_multi_asset_market_data,
)

__all__ = [
    "AlignMultiAssetResult",
    "KLINE_ROLES",
    "LoadedMarketData",
    "MultiAssetMarketData",
    "align_multi_asset_to_common_index",
    "ValidationResult",
    "load_market_data",
    "load_multi_market_data",
    "validate_loaded_market_data",
    "validate_market_frame",
    "validate_multi_asset_market_data",
]
