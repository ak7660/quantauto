"""inner-align multi-asset data to a common timestamp index (cross-sectional workflows)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from quantauto.data.schema import LoadedMarketData, MultiAssetMarketData


@dataclass(frozen=True)
class AlignMultiAssetResult:
    """result of :func:`align_multi_asset_to_common_index`."""

    data: MultiAssetMarketData
    common_index: pd.DatetimeIndex
    dropped_bars: Dict[str, int]
    messages: Tuple[str, ...]

    @property
    def warnings(self) -> Tuple[str, ...]:
        """user-facing note lines (dropped bar counts, etc.)."""
        return self.messages


def align_multi_asset_to_common_index(
    multi: MultiAssetMarketData,
    *,
    min_common_bars: int = 5,
) -> AlignMultiAssetResult:
    """reindex every symbol to the **intersection** of all datetime indices.

    rows outside the intersection are dropped from each symbol.  if the
    intersection is empty, or shorter than ``min_common_bars``, raises
    ``ValueError`` with a short per-symbol range summary.
    """
    if not multi.by_symbol:
        raise ValueError("multi-asset data has no symbols")
    parts: List[Tuple[str, pd.DataFrame, pd.DatetimeIndex]] = []
    for sym, loaded in multi.items():
        idx = loaded.data.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise TypeError(
                f"symbol {sym!r}: data index must be a DatetimeIndex, got {type(idx).__name__}"
            )
        parts.append((sym, loaded.data, idx))

    common = parts[0][2]
    for sym, _df, idx in parts[1:]:
        common = common.intersection(idx)

    if len(common) == 0:
        detail = _range_detail(parts)
        raise ValueError(
            "no overlapping timestamps between symbols; cannot inner-align. " + detail
        )
    if len(common) < int(min_common_bars):
        detail = _range_detail(parts)
        raise ValueError(
            f"only {len(common)} common bars (need at least {min_common_bars}); " + detail
        )

    common = common.sort_values()
    dropped: Dict[str, int] = {}
    msgs: List[str] = []
    by_symbol: Dict[str, LoadedMarketData] = {}
    for sym, orig_df, idx in parts:
        before = len(orig_df)
        new_df = orig_df.reindex(common)
        dropped_n = max(0, before - len(common))
        dropped[sym] = int(dropped_n)
        if dropped_n > 0:
            msgs.append(
                f"symbol {sym!r}: dropped {dropped_n} row(s) not in common timestamp set"
            )
        src_ld = multi.by_symbol[sym]
        by_symbol[sym] = LoadedMarketData(
            data=new_df,
            kline_source_columns=src_ld.kline_source_columns,
            timestamp_source_column=src_ld.timestamp_source_column,
            feature_columns=src_ld.feature_columns,
        )

    out = MultiAssetMarketData(
        by_symbol=by_symbol,
        symbol_column=multi.symbol_column,
        aligned=True,
    )
    result = AlignMultiAssetResult(
        data=out,
        common_index=common,
        dropped_bars=dropped,
        messages=tuple(msgs),
    )
    for m in result.messages:
        warnings.warn(m, UserWarning, stacklevel=2)
    return result


def _range_detail(parts: List[Tuple[str, pd.DataFrame, pd.DatetimeIndex]]) -> str:
    lines = []
    for sym, _df, idx in parts:
        if len(idx) == 0:
            lines.append(f"{sym!r}: empty index")
        else:
            ir = idx.sort_values()
            lines.append(
                f"{sym!r}: n={len(idx)} range=[{ir[0]!r} .. {ir[-1]!r}]"
            )
    return "; ".join(lines)
