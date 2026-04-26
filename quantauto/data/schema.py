"""market data schema: kline role names, synonyms, and load result container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterator, Optional, Tuple

import pandas as pd

# canonical kline field names used across quantauto (feature engineering expects these)
KLINE_ROLES: Tuple[str, ...] = ("open", "high", "low", "close", "volume")

# normalized synonym -> canonical role (first match wins when scanning columns)
ROLE_SYNONYMS: Tuple[Tuple[str, str], ...] = (
    ("open", "open"),
    ("o", "open"),
    ("open_price", "open"),
    ("opening", "open"),
    ("high", "high"),
    ("h", "high"),
    ("high_price", "high"),
    ("hi", "high"),
    ("low", "low"),
    ("l", "low"),
    ("low_price", "low"),
    ("lo", "low"),
    ("close", "close"),
    ("c", "close"),
    ("close_price", "close"),
    ("adj_close", "close"),
    ("adjclose", "close"),
    ("adjusted_close", "close"),
    ("last", "close"),
    ("last_price", "close"),
    ("price", "close"),
    ("px", "close"),
    ("volume", "volume"),
    ("vol", "volume"),
    ("v", "volume"),
    ("qty", "volume"),
    ("base_volume", "volume"),
    ("amount", "volume"),
)

TIMESTAMP_SYNONYMS: FrozenSet[str] = frozenset(
    {
        "timestamp",
        "time",
        "date",
        "datetime",
        "ts",
        "open_time",
        "closetime",
        "close_time",
        "t",
    }
)

# normalized column names treated as instrument / ticker identifiers (panel data)
SYMBOL_SYNONYMS: FrozenSet[str] = frozenset(
    {
        "symbol",
        "symbols",
        "sym",
        "ticker",
        "tickers",
        "ticker_symbol",
        "stock",
        "stock_symbol",
        "asset",
        "asset_id",
        "assets",
        "instrument",
        "instruments",
        "instrument_id",
        "pair",
        "pairs",
        "kline_symbol",
        "kline",
        "underlying",
        "underlying_symbol",
        "figi",
        "ric",
        "isin",
        "security",
        "security_id",
        "secid",
        "product",
        "product_id",
        "contract",
        "contract_id",
        "series",
        "series_id",
        "market",
        "market_id",
        "code",
        "tradingsymbol",
        "trading_symbol",
    }
)

# normalized column name contains one of these (substring); earlier = higher priority
SYMBOL_COLUMN_NAME_SUBSTRINGS: Tuple[str, ...] = (
    "kline_symbol",
    "kline",
    "ticker_symbol",
    "ticker",
    "trading_symbol",
    "tradingsymbol",
    "instrument",
    "underlying",
    "exchange_symbol",
    "exchange",
    "security_id",
    "security",
    "asset_id",
    "asset",
    "contract",
    "figi",
    "ric",
    "isin",
    "pair",
    "perp",
    "swap",
    "symbol",
    "sym",
    "venue",
    "listing",
    "mic",
    "sedol",
    "cusip",
)


def normalize_column_key(name: str) -> str:
    """normalizes a column label for synonym matching (lowercase, underscores)."""
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    return key


def role_synonym_map() -> Dict[str, str]:
    """maps normalized synonym to canonical kline role."""
    return {syn: role for syn, role in ROLE_SYNONYMS}


def normalized_kline_column_aliases() -> FrozenSet[str]:
    """normalized names that map to OHLCV roles (exclude from instrument-id substring picks)."""
    return frozenset(normalize_column_key(s) for s, _ in ROLE_SYNONYMS)


@dataclass
class LoadedMarketData:
    """result of loading and normalizing market time series."""

    data: pd.DataFrame
    # canonical kline role -> original source column name that was mapped
    kline_source_columns: Dict[str, str] = field(default_factory=dict)
    # name of column used as timestamp before it was moved to the index (if any)
    timestamp_source_column: Optional[str] = None
    # non-kline columns present in `data` (additional features)
    feature_columns: Tuple[str, ...] = ()

    def mapping_summary_lines(self) -> Tuple[str, ...]:
        """brief human-readable lines describing how columns were interpreted."""
        lines = []
        if self.timestamp_source_column:
            lines.append(f"timestamp: index <= {self.timestamp_source_column}")
        for role in KLINE_ROLES:
            src = self.kline_source_columns.get(role)
            if src is not None:
                lines.append(f"{role}: {src}")
        if self.feature_columns:
            lines.append("features: " + ", ".join(self.feature_columns))
        return tuple(lines)


@dataclass
class MultiAssetMarketData:
    """multiple symbols, each with its own loaded frame (no merged rows across symbols)."""

    by_symbol: Dict[str, LoadedMarketData]
    # when loaded from one panel file, the column used to split (else None)
    symbol_column: Optional[str] = None
    # true when all symbols share exactly the same datetime index
    aligned: bool = True

    @property
    def symbols(self) -> Tuple[str, ...]:
        """symbol keys in stable sorted order."""
        return tuple(sorted(self.by_symbol.keys()))

    def items(self) -> Iterator[Tuple[str, LoadedMarketData]]:
        """iterates (symbol, LoadedMarketData) in sorted key order."""
        for k in self.symbols:
            yield k, self.by_symbol[k]

    def summary_lines(self) -> Tuple[str, ...]:
        """one line per symbol with row counts."""
        lines = []
        for sym in self.symbols:
            ld = self.by_symbol[sym]
            n = len(ld.data)
            lines.append(f"{sym}: {n} rows")
        return tuple(lines)
