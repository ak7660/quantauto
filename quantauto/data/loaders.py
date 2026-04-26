"""single-call loading of csv/parquet/dataframe with kline column inference."""

from __future__ import annotations

import warnings
from collections import OrderedDict
from pathlib import Path
from collections.abc import Mapping as ABCMapping
from typing import Dict, Mapping, Optional, Sequence, Set, Union

import pandas as pd

from quantauto.data.schema import (
    KLINE_ROLES,
    LoadedMarketData,
    MultiAssetMarketData,
    ROLE_SYNONYMS,
    SYMBOL_COLUMN_NAME_SUBSTRINGS,
    SYMBOL_SYNONYMS,
    TIMESTAMP_SYNONYMS,
    normalize_column_key,
    normalized_kline_column_aliases,
)

SourceLike = Union[str, Path, pd.DataFrame]


def _is_str_mapping(x: object) -> bool:
    """true for dict-like column maps, false for plain str column names."""
    return isinstance(x, ABCMapping) and not isinstance(x, (str, bytes))


def _synonyms_by_role() -> Dict[str, tuple]:
    """ordered synonym lists per kline role (first listed = higher priority)."""
    out: OrderedDict[str, list] = OrderedDict()
    for r in KLINE_ROLES:
        out[r] = []
    for syn, role in ROLE_SYNONYMS:
        if role in out and syn not in out[role]:
            out[role].append(syn)
    return {k: tuple(v) for k, v in out.items()}


SYNONYMS_BY_ROLE = _synonyms_by_role()
_KLINE_ALIAS_KEYS = normalized_kline_column_aliases()


def _read_tabular(path: Union[str, Path], skiprows: Optional[int]) -> pd.DataFrame:
    """reads csv or parquet from path."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(p)
        except ImportError as e:
            raise ImportError(
                "reading parquet requires pyarrow or fastparquet; install pyarrow"
            ) from e
    if suf in {".csv", ".txt"} or suf == "":
        return pd.read_csv(p, skiprows=skiprows) if skiprows is not None else pd.read_csv(p)
    raise ValueError(f"unsupported file extension for {p}; use .csv or .parquet")


def _coerce_source(source: SourceLike, skiprows: Optional[int]) -> pd.DataFrame:
    """returns a dataframe copy from path or dataframe input."""
    if isinstance(source, pd.DataFrame):
        return source.copy()
    return _read_tabular(source, skiprows=skiprows)


def _parse_ts_series(
    series: pd.Series,
    *,
    dayfirst: bool = False,
    timestamp_format: Optional[str] = None,
) -> pd.Series:
    """parses a column to timezone-aware UTC datetimes (coerce errors)."""
    kwargs: Dict[str, object] = {"errors": "coerce", "utc": True}
    if timestamp_format is not None:
        kwargs["format"] = timestamp_format
    elif dayfirst:
        kwargs["dayfirst"] = True
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not infer format",
            category=UserWarning,
        )
        return pd.to_datetime(series, **kwargs)


def _infer_timestamp_column(
    df: pd.DataFrame,
    hint: Optional[str],
    min_parse_ratio: float = 0.55,
    *,
    dayfirst: bool = False,
    timestamp_format: Optional[str] = None,
) -> str:
    """picks the column that best parses as datetimes; prefers known timestamp names."""
    if hint is not None:
        if hint not in df.columns:
            raise KeyError(f"timestamp_column {hint!r} not in columns: {list(df.columns)}")
        return hint

    best_col = None
    best_score = -1.0
    name_bonus = 0.001  # tiny tie-break for canonical timestamp names

    for col in df.columns:
        nk = normalize_column_key(str(col))
        # skip obvious ohlcv numeric columns unless named like a time field
        if pd.api.types.is_numeric_dtype(df[col]) and nk not in TIMESTAMP_SYNONYMS:
            continue
        parsed = _parse_ts_series(
            df[col], dayfirst=dayfirst, timestamp_format=timestamp_format
        )
        score = float(parsed.notna().mean())
        if nk in TIMESTAMP_SYNONYMS:
            score += name_bonus
        if score > best_score:
            best_score = score
            best_col = col

    if best_col is None or best_score < min_parse_ratio:
        raise ValueError(
            "could not infer a timestamp column; set timestamp_column explicitly. "
            f"columns={list(df.columns)}"
        )
    return str(best_col)


def _best_substring_instrument_column(
    df: pd.DataFrame,
    min_unique: int,
) -> Optional[str]:
    """column whose normalized name contains a known instrument fragment; requires cardinality."""
    candidates: list[tuple[int, int, str]] = []
    for col in df.columns:
        nk = normalize_column_key(str(col))
        if nk in TIMESTAMP_SYNONYMS or nk in _KLINE_ALIAS_KEYS:
            continue
        nu = int(df[col].nunique(dropna=True))
        if nu < min_unique:
            continue
        frag_i: Optional[int] = None
        for i, frag in enumerate(SYMBOL_COLUMN_NAME_SUBSTRINGS):
            if frag in nk:
                frag_i = i
                break
        if frag_i is None:
            continue
        candidates.append((frag_i, nu, str(col)))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def _symbol_column_name_score(col: str) -> int:
    nk = normalize_column_key(col)
    hints = (
        "symbol",
        "ticker",
        "tick",
        "pair",
        "asset",
        "instru",
        "contract",
        "ric",
        "isin",
        "figi",
        "security",
        "product",
        "market",
        "code",
        "series",
        "underlying",
    )
    return sum(1 for h in hints if h in nk)


def _infer_symbol_column_heuristic(df: pd.DataFrame) -> Optional[str]:
    """last-resort panel splitter: string-like column with modest cardinality."""
    n = len(df)
    if n < 2:
        return None
    candidates: list[tuple[int, int, str]] = []
    sample_n = min(400, n)
    for col in df.columns:
        nk = normalize_column_key(str(col))
        if nk in TIMESTAMP_SYNONYMS:
            continue
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s) and nk not in SYMBOL_SYNONYMS:
            continue
        nu = int(s.nunique(dropna=True))
        if nu < 2 or nu > min(2048, max(4, n // 2)):
            continue
        head = s.head(sample_n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(head, errors="coerce")
        if float(parsed.notna().mean()) > 0.72:
            continue
        str_vals = s.dropna().astype(str).str.strip()
        if str_vals.empty:
            continue
        token_ok = str_vals.str.match(r"^[A-Za-z0-9][A-Za-z0-9_./\-]{0,47}$").mean()
        if float(token_ok) < 0.78:
            continue
        name_score = _symbol_column_name_score(str(col))
        candidates.append((-name_score, nu, str(col)))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


_QA_COALESCED_COL = "_quantauto_instrument"


def _collect_symbol_columns_for_coalesce(df: pd.DataFrame) -> list[str]:
    """ordered instrument-like columns (synonym before substring; stable within tier)."""
    tier0: list[tuple[str, str]] = []
    tier1: list[tuple[tuple[int, str], str]] = []
    for col in df.columns:
        nk = normalize_column_key(str(col))
        if nk in TIMESTAMP_SYNONYMS or nk in _KLINE_ALIAS_KEYS:
            continue
        s = df[col]
        if not s.notna().any():
            continue
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s) and nk not in SYMBOL_SYNONYMS:
            continue
        if nk in SYMBOL_SYNONYMS:
            tier0.append((nk, str(col)))
            continue
        frag_i: Optional[int] = None
        for i, frag in enumerate(SYMBOL_COLUMN_NAME_SUBSTRINGS):
            if frag in nk:
                frag_i = i
                break
        if frag_i is not None:
            tier1.append(((frag_i, nk), str(col)))
    tier0.sort(key=lambda x: x[0])
    tier1.sort(key=lambda x: x[0])
    return [c for _, c in tier0] + [c for _, c in tier1]


def _prefer_one_full_instrument_column(
    df: pd.DataFrame,
    cands: Sequence[str],
) -> Optional[str]:
    """if a single candidate already holds the id for every row that has an id anywhere, use it (no coalesce)."""
    if len(cands) < 2:
        return None
    need_mask = df[list(cands)].notna().any(axis=1)
    if not need_mask.any():
        return None
    n_need = int(need_mask.sum())
    tol = max(1, int(0.005 * n_need))
    for c in cands:
        if int(df[c].nunique(dropna=True)) < 2:
            continue
        hit = int(df.loc[need_mask, c].notna().sum())
        if hit >= n_need - tol:
            return str(c)
    return None


def _maybe_coalesce_concat_symbol_columns(
    df: pd.DataFrame,
    symbol_column: Optional[str],
) -> tuple[pd.DataFrame, Optional[str], Optional[str]]:
    """if ``pd.concat`` left mutually exclusive instrument cols (e.g. ``kline`` vs ``kline_symbol``),
    merge into ``_QA_COALESCED_COL`` and return that column name plus a display label.

    returns ``(dataframe, split_column_name_or_none, sym_col_used_label_or_none)``.
    """
    if symbol_column is not None:
        return df, None, None
    cands = _collect_symbol_columns_for_coalesce(df)
    if len(cands) < 2:
        return df, None, None
    solo = _prefer_one_full_instrument_column(df, cands)
    if solo is not None:
        return df, None, None
    sub = df[cands]
    nn = sub.notna().sum(axis=1)
    overlap = int((nn > 1).sum())
    if overlap > max(3, int(0.02 * len(df))):
        warnings.warn(
            "stacked table has several instrument-like columns with overlapping non-null values; "
            "could not merge them automatically. Pass symbol_column='...' to choose one column, "
            "or clean the data.",
            UserWarning,
            stacklevel=2,
        )
        return df, None, None
    merged = df[cands[0]]
    for c in cands[1:]:
        merged = merged.fillna(df[c])

    def _strip_id(v: object) -> object:
        if pd.isna(v):
            return pd.NA
        t = str(v).strip()
        return t if t else pd.NA

    unified = merged.map(_strip_id)
    if int(unified.nunique(dropna=True)) < 2:
        return df, None, None
    out = df.drop(columns=cands).copy()
    out[_QA_COALESCED_COL] = unified
    label = ", ".join(cands) + " (coalesced)"
    return out, _QA_COALESCED_COL, label


def _resolve_symbol_column(
    df: pd.DataFrame,
    symbol_column: Optional[str],
    infer_symbol_column: bool,
) -> Optional[str]:
    """returns column to split panel data, or None if not found."""
    if symbol_column is not None:
        if symbol_column not in df.columns:
            raise KeyError(f"symbol_column {symbol_column!r} not in columns: {list(df.columns)}")
        return str(symbol_column)
    if not infer_symbol_column:
        return None
    # multiple instruments: exact synonym match
    for col in df.columns:
        nk = normalize_column_key(str(col))
        if nk not in SYMBOL_SYNONYMS:
            continue
        nu = df[col].nunique(dropna=True)
        if nu >= 2:
            return str(col)
    # multiple instruments: name contains kline / ticker / symbol / ...
    sub_m = _best_substring_instrument_column(df, min_unique=2)
    if sub_m is not None:
        return sub_m
    # single instrument but constant label (e.g. kline_symbol all BTCUSDT)
    for col in df.columns:
        nk = normalize_column_key(str(col))
        if nk not in SYMBOL_SYNONYMS:
            continue
        nu = df[col].nunique(dropna=True)
        if nu == 1:
            return str(col)
    sub_1 = _best_substring_instrument_column(df, min_unique=1)
    if sub_1 is not None:
        return sub_1
    return _infer_symbol_column_heuristic(df)


def _instrument_split_column(
    df: pd.DataFrame,
    explicit: Optional[str],
    infer_symbol_column: bool,
) -> Optional[str]:
    """column to split one file/dataframe into several instruments (requires >=2 distinct ids)."""
    if explicit is not None:
        if explicit not in df.columns:
            raise KeyError(
                f"symbol_columns: column {explicit!r} not in columns: {list(df.columns)}"
            )
        nu = int(df[explicit].nunique(dropna=True))
        if nu >= 2:
            return str(explicit)
        return None
    if not infer_symbol_column:
        return None
    for col in df.columns:
        nk = normalize_column_key(str(col))
        if nk not in SYMBOL_SYNONYMS:
            continue
        if int(df[col].nunique(dropna=True)) >= 2:
            return str(col)
    sub = _best_substring_instrument_column(df, min_unique=2)
    if sub is not None:
        return sub
    heur = _infer_symbol_column_heuristic(df)
    if heur is not None and int(df[heur].nunique(dropna=True)) >= 2:
        return heur
    return None


def _panel_symbol_column_failure_message(df: pd.DataFrame) -> str:
    return (
        "could not infer an instrument / ticker column for stacked panel data. "
        "Pass symbol_column='your_column' to name the column that identifies each series. "
        "Separate files do not need the same column name: use "
        "by_symbol={'BTC': path_btc, 'ETH': path_eth} or paths=[...], symbols=[...] so each "
        "file is read on its own. If one file (or a concat) uses different names for the same "
        "role (e.g. kline vs kline_symbol), rename before concat or rely on automatic coalesce "
        "when the columns are mutually exclusive. For per-file instrument columns when splitting "
        "multi-asset files, pass symbol_columns={'<source_label>': '<column_in_that_file>', ...}. "
        f"columns={list(df.columns)}"
    )


def _merge_split_symbol_key(
    existing: Dict[str, LoadedMarketData],
    outer_label: str,
    inner_symbol: str,
) -> str:
    key = str(inner_symbol).strip()
    if key not in existing:
        return key
    alt = f"{outer_label}_{key}"
    if alt not in existing:
        return alt
    return f"{outer_label}__{key}"


def _maybe_warn_mismatched_instrument_columns(
    per_source_columns: Sequence[Optional[str]],
    source_labels: Sequence[str],
) -> None:
    pairs = [(lbl, c) for lbl, c in zip(source_labels, per_source_columns) if c is not None]
    if len(pairs) < 2:
        return
    names = {c for _, c in pairs}
    if len(names) <= 1:
        return
    detail = ", ".join(f"{lbl!r}→{c!r}" for lbl, c in pairs)
    warnings.warn(
        "Sources use different instrument column names (" + detail + "). "
        "That is fine while each file is loaded on its own. If you build one stacked "
        "DataFrame instead, rename those columns to a single name before concat, or pass "
        "symbol_column=... once on the combined frame. If a source fails to split into "
        "multiple assets, set symbol_columns={'<label>': '<instrument_column>'} for that path.",
        UserWarning,
        stacklevel=2,
    )


def _load_separate_sources_into_out(
    out: Dict[str, LoadedMarketData],
    labeled_sources: Sequence[tuple[str, SourceLike]],
    *,
    symbol_col_by_source: Mapping[str, str],
    skiprows: Optional[int],
    timestamp_column: Optional[str],
    timestamp_columns: Optional[Mapping[str, str]],
    kline_columns: Optional[Mapping[str, str]],
    kline_columns_by_symbol: Optional[Mapping[str, Mapping[str, str]]],
    drop_bad_timestamps: bool,
    dayfirst: bool,
    timestamp_format: Optional[str],
    timestamp_formats: Optional[Mapping[str, str]],
    dayfirst_by_symbol: Optional[Mapping[str, bool]],
    infer_symbol_column: bool,
) -> None:
    """fills ``out`` from (label, path_or_df) pairs; may split panel-like sources."""
    split_records: list[tuple[str, Optional[str]]] = []
    for label, src in labeled_sources:
        df = _coerce_source(src, skiprows=skiprows)
        explicit = symbol_col_by_source.get(label)
        df, coalesced_split_col, coalesce_lbl = _maybe_coalesce_concat_symbol_columns(
            df, explicit
        )
        split_hint = explicit if explicit is not None else coalesced_split_col
        split_col = _instrument_split_column(df, split_hint, infer_symbol_column)
        if split_col is not None:
            split_records.append((label, coalesce_lbl if coalesce_lbl else split_col))
            for inner, sub in df.groupby(split_col, sort=False):
                inner_k = str(inner).strip()
                out_k = _merge_split_symbol_key(out, label, inner_k)
                sub_ohlc = sub.drop(columns=[split_col]).reset_index(drop=True)
                ts_ov = _map_prefer_inner_then_outer(inner_k, label, timestamp_columns)
                ts_one = (
                    str(ts_ov) if ts_ov is not None else timestamp_column
                )
                k_ov = _map_prefer_inner_then_outer(inner_k, label, kline_columns_by_symbol)
                if isinstance(k_ov, ABCMapping):
                    k_one = k_ov
                else:
                    k_one = kline_columns
                fmt_ov = _map_prefer_inner_then_outer(inner_k, label, timestamp_formats)
                fmt_one = str(fmt_ov) if fmt_ov is not None else timestamp_format
                day_ov = _map_prefer_inner_then_outer(inner_k, label, dayfirst_by_symbol)
                day_one = bool(day_ov) if day_ov is not None else dayfirst
                out[out_k] = _load_market_data_core(
                    sub_ohlc,
                    timestamp_column=ts_one,
                    kline_columns=k_one,
                    drop_bad_timestamps=drop_bad_timestamps,
                    dayfirst=day_one,
                    timestamp_format=fmt_one,
                )
        else:
            split_records.append((label, None))
            if explicit is not None:
                warnings.warn(
                    f"source {label!r}: symbol_columns[{label!r}]={explicit!r} has fewer than "
                    f"two distinct instrument ids; loading the whole table as one series.",
                    UserWarning,
                    stacklevel=2,
                )
            ts_one = _str_per_symbol(label, timestamp_column, timestamp_columns)
            k_one = kline_columns
            if kline_columns_by_symbol is not None and label in kline_columns_by_symbol:
                k_one = kline_columns_by_symbol[label]
            fmt_one = _str_per_symbol(label, timestamp_format, timestamp_formats)
            day_one = dayfirst
            if dayfirst_by_symbol is not None and label in dayfirst_by_symbol:
                day_one = bool(dayfirst_by_symbol[label])
            out[label] = _load_market_data_core(
                df,
                timestamp_column=ts_one,
                kline_columns=k_one,
                drop_bad_timestamps=drop_bad_timestamps,
                dayfirst=day_one,
                timestamp_format=fmt_one,
            )

    if len(labeled_sources) >= 2:
        _maybe_warn_mismatched_instrument_columns(
            [c for _, c in split_records],
            [lbl for lbl, _ in split_records],
        )


def _default_symbol_for_source(index: int, src: SourceLike) -> str:
    """derives a symbol label from a path stem, else a generic name."""
    if isinstance(src, (str, Path)):
        return Path(str(src)).stem
    return f"series_{index}"


def _to_utc_index(
    series: pd.Series,
    *,
    dayfirst: bool = False,
    timestamp_format: Optional[str] = None,
) -> pd.DatetimeIndex:
    """parses series to timezone-aware UTC datetime index."""
    dt = _parse_ts_series(series, dayfirst=dayfirst, timestamp_format=timestamp_format)
    return pd.DatetimeIndex(dt)


def _is_numeric_kline_candidate(s: pd.Series) -> bool:
    """true if series is mostly numeric (ohlcv)."""
    num = pd.to_numeric(s, errors="coerce")
    valid = float(num.notna().mean())
    return valid >= 0.9


def _auto_assign_kline_roles(
    df: pd.DataFrame,
    used: Set[str],
    synonyms_by_role: Mapping[str, tuple],
    merged: Dict[str, str],
) -> None:
    """fills merged[role] for roles not yet set; updates used with assigned columns."""
    for role in KLINE_ROLES:
        if role in merged:
            continue
        syns = synonyms_by_role.get(role, ())
        for col in df.columns:
            if col in used:
                continue
            nk = normalize_column_key(str(col))
            if nk in syns:
                merged[role] = col
                used.add(col)
                break


def _maybe_assign_single_price_as_close(
    df: pd.DataFrame,
    used: Set[str],
    role_to_col: Dict[str, str],
) -> None:
    """if close is missing, use the sole remaining numeric column as close."""
    if "close" in role_to_col:
        return
    remaining = [c for c in df.columns if c not in used]
    numeric_remaining = [c for c in remaining if _is_numeric_kline_candidate(df[c])]
    if len(numeric_remaining) == 1:
        c = numeric_remaining[0]
        role_to_col["close"] = c
        used.add(c)


def _finalize_role_map(
    df: pd.DataFrame,
    timestamp_col: str,
    manual: Optional[Mapping[str, str]],
) -> Dict[str, str]:
    """builds role -> source column with manual first, then auto map, then close fallback."""
    used: Set[str] = {timestamp_col}
    merged: Dict[str, str] = {}
    if manual:
        for role, src in manual.items():
            if role not in KLINE_ROLES:
                raise KeyError(f"unknown kline role {role!r}; expected one of {KLINE_ROLES}")
            if src not in df.columns:
                raise KeyError(f"kline_columns[{role!r}]={src!r} not in dataframe columns")
            merged[role] = str(src)
            used.add(str(src))

    _auto_assign_kline_roles(df, used, SYNONYMS_BY_ROLE, merged)
    _maybe_assign_single_price_as_close(df, used, merged)

    if "close" not in merged:
        raise ValueError(
            "need at least one price column mapped to close; could not infer automatically. "
            "pass kline_columns={'close': '<your_price_column>'}. "
            f"columns={list(df.columns)}"
        )
    return merged


def _build_output_frame(
    df: pd.DataFrame,
    timestamp_col: str,
    role_to_col: Dict[str, str],
    *,
    dayfirst: bool = False,
    timestamp_format: Optional[str] = None,
) -> tuple:
    """sorted datetime index, standard kline columns, preserved feature columns."""
    idx = _to_utc_index(
        df[timestamp_col], dayfirst=dayfirst, timestamp_format=timestamp_format
    )
    out = pd.DataFrame(index=idx)
    kline_sources: Dict[str, str] = {}
    for role in KLINE_ROLES:
        src = role_to_col.get(role)
        if src is None:
            continue
        series = pd.to_numeric(df[src], errors="coerce")
        out[role] = series.values
        kline_sources[role] = src

    feature_cols = [
        c
        for c in df.columns
        if c != timestamp_col and c not in set(role_to_col.values())
    ]
    for c in feature_cols:
        out[c] = df[c].values

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.index.name = None
    return out, kline_sources, tuple(feature_cols)


def _load_market_data_core(
    df: pd.DataFrame,
    *,
    timestamp_column: Optional[str] = None,
    kline_columns: Optional[Mapping[str, str]] = None,
    drop_bad_timestamps: bool = True,
    dayfirst: bool = False,
    timestamp_format: Optional[str] = None,
) -> LoadedMarketData:
    """normalizes one instrument's tabular data to a time-indexed frame (no printing)."""
    ts_col = _infer_timestamp_column(
        df,
        timestamp_column,
        dayfirst=dayfirst,
        timestamp_format=timestamp_format,
    )
    if drop_bad_timestamps:
        mask = _parse_ts_series(
            df[ts_col], dayfirst=dayfirst, timestamp_format=timestamp_format
        ).notna()
        df = df.loc[mask].reset_index(drop=True)

    role_to_col = _finalize_role_map(df, ts_col, kline_columns)
    out, kline_sources, feature_cols = _build_output_frame(
        df,
        ts_col,
        role_to_col,
        dayfirst=dayfirst,
        timestamp_format=timestamp_format,
    )

    return LoadedMarketData(
        data=out,
        kline_source_columns=kline_sources,
        timestamp_source_column=ts_col,
        feature_columns=feature_cols,
    )


def load_market_data(
    source: SourceLike,
    *,
    timestamp_column: Optional[str] = None,
    kline_columns: Optional[Mapping[str, str]] = None,
    skiprows: Optional[int] = None,
    drop_bad_timestamps: bool = True,
    dayfirst: bool = False,
    timestamp_format: Optional[str] = None,
    verbose: bool = True,
) -> LoadedMarketData:
    """loads csv/parquet or accepts a dataframe; infers kline columns, sorts time index.

    automatic mapping recognizes common ohlcv names (open/high/low/close/volume, o/h/l/c/v, etc.).
    pass kline_columns to override or fix ambiguous files (role -> source column name).

    ``dayfirst=True`` helps parse day-first dates (e.g. EU ``DD/MM/YYYY``). ``timestamp_format``
    passes a :func:`pandas.to_datetime` format string when all values share one layout.

    requires at least one price series mapped to close; if only one numeric column remains
    unmatched, it is treated as close.

    prints a short summary of how columns were interpreted when verbose is true.
    """
    df = _coerce_source(source, skiprows=skiprows)
    if _is_str_mapping(timestamp_column):
        raise TypeError(
            "timestamp_column must be a single column name for load_market_data; "
            "use load_multi_market_data(timestamp_columns={...}) for per-symbol hints."
        )
    if kline_columns is not None and _is_str_mapping(kline_columns):
        first_val = next(iter(kline_columns.values()), None)
        if isinstance(first_val, ABCMapping):
            raise TypeError(
                "kline_columns must be role -> column for load_market_data; "
                "use load_multi_market_data(kline_columns_by_symbol={...})."
            )
    result = _load_market_data_core(
        df,
        timestamp_column=timestamp_column,
        kline_columns=kline_columns,
        drop_bad_timestamps=drop_bad_timestamps,
        dayfirst=dayfirst,
        timestamp_format=timestamp_format,
    )

    if verbose:
        n = len(result.data)
        t0 = result.data.index.min() if n else None
        t1 = result.data.index.max() if n else None
        print(f"loaded market data: {n} rows, index [{t0} .. {t1}]")
        for line in result.mapping_summary_lines():
            print(f"  {line}")

    return result


def _str_per_symbol(
    sym: str,
    default: Optional[str],
    overrides: Optional[Mapping[str, str]],
) -> Optional[str]:
    if overrides is not None and sym in overrides:
        return str(overrides[sym])
    return default


def _map_prefer_inner_then_outer(
    inner_key: str,
    outer_label: str,
    overrides: Optional[Mapping[str, object]],
) -> Optional[object]:
    """when a file splits into several instruments, allow overrides on inner id or file label."""
    if overrides is None:
        return None
    if inner_key in overrides:
        return overrides[inner_key]
    if outer_label in overrides:
        return overrides[outer_label]
    return None


def load_multi_market_data(
    *,
    by_symbol: Optional[Mapping[str, SourceLike]] = None,
    paths: Optional[Sequence[SourceLike]] = None,
    symbols: Optional[Sequence[str]] = None,
    source: Optional[SourceLike] = None,
    symbol_column: Optional[str] = None,
    symbol_columns: Optional[Mapping[str, str]] = None,
    infer_symbol_column: bool = True,
    skiprows: Optional[int] = None,
    timestamp_column: Optional[str] = None,
    timestamp_columns: Optional[Mapping[str, str]] = None,
    kline_columns: Optional[Mapping[str, str]] = None,
    kline_columns_by_symbol: Optional[Mapping[str, Mapping[str, str]]] = None,
    drop_bad_timestamps: bool = True,
    dayfirst: bool = False,
    timestamp_format: Optional[str] = None,
    timestamp_formats: Optional[Mapping[str, str]] = None,
    dayfirst_by_symbol: Optional[Mapping[str, bool]] = None,
    verbose: bool = True,
) -> MultiAssetMarketData:
    """loads several instruments as separate time series (no row-wise merge across symbols).

    **Workflow**

    1. **Separate files** (``by_symbol`` / ``paths``): each source is read independently; symbol
       column names **do not** need to match across files. If a source holds several instruments,
       the loader finds an id column (exact synonyms like ``kline``, ``kline_symbol``, ``ticker``,
       ``instrument``, …; then name substrings such as ``kline``, ``ticker``, ``venue``, ``mic``;
       then a cardinality heuristic). If one source is already a ``concat`` with **mutually
       exclusive** id columns (e.g. ``kline`` for some rows and ``kline_symbol`` for others), they
       are **coalesced** automatically before splitting—the same as for ``source=``.
    2. **One stacked table** (``source=``): first prefer a **single** column that already carries
       the instrument id for every row that needs one; if only complementary columns exist, merge
       them; then infer as above.
    3. If nothing works: ``UserWarning`` and ``ValueError`` suggesting ``symbol_column=...`` or
       ``symbol_columns={'<label>': '<column>', ...}``.
    4. When **two or more** separate sources each auto-split on **different** column names, a
       ``UserWarning`` explains that this is OK per file, and how to unify if you ``concat`` later.

    Substrings deliberately avoid a bare ``price`` match (would collide with ``close_price``-style
    OHLC columns). Use ``symbol_column`` for vendor-specific names.

    use exactly one of:
    - ``by_symbol``: map symbol -> csv/parquet path or dataframe (one series per key).
    - ``paths``: sequence of paths or dataframes; pass ``symbols`` aligned to each source,
      or omit ``symbols`` to use each path's filename stem as the symbol.
    - ``source``: one stacked table; split rows by ``symbol_column`` or inferred / coalesced ids.

    - ``symbol_columns``: optional map **source label** → column name (same keys as ``by_symbol``,
      or ``symbols`` / path stems for ``paths=``) when that file itself contains several
      instruments. If omitted, each file is still auto-split when a strong instrument column is
      found with two or more distinct ids.

    ``timestamp_columns`` / ``kline_columns_by_symbol`` override ``timestamp_column`` /
    ``kline_columns`` for specific symbol keys (the same keys as ``by_symbol``, or the names
    from ``symbols`` / path stems for ``paths=``). Use this when files use different date or
    OHLC column names. ``timestamp_formats`` / ``dayfirst_by_symbol`` override ``timestamp_format``
    / ``dayfirst`` per symbol when date strings differ by file.

    each value is a full ``LoadedMarketData`` suitable for per-asset feature work or aligned
    cross-sectional stacks on the shared timeframe in downstream code.
    """
    modes = sum(
        [
            by_symbol is not None,
            paths is not None,
            source is not None,
        ]
    )
    if modes != 1:
        raise ValueError(
            "pass exactly one of: by_symbol=..., paths=..., or source=... "
            "(panel file / combined dataframe)."
        )

    out: Dict[str, LoadedMarketData] = {}
    sym_col_used: Optional[str] = None
    sym_col_by_source: Dict[str, str] = dict(symbol_columns) if symbol_columns else {}

    if by_symbol is not None:
        if not by_symbol:
            raise ValueError("by_symbol is empty")
        labeled = [(str(k), v) for k, v in by_symbol.items()]
        _load_separate_sources_into_out(
            out,
            labeled,
            symbol_col_by_source=sym_col_by_source,
            skiprows=skiprows,
            timestamp_column=timestamp_column,
            timestamp_columns=timestamp_columns,
            kline_columns=kline_columns,
            kline_columns_by_symbol=kline_columns_by_symbol,
            drop_bad_timestamps=drop_bad_timestamps,
            dayfirst=dayfirst,
            timestamp_format=timestamp_format,
            timestamp_formats=timestamp_formats,
            dayfirst_by_symbol=dayfirst_by_symbol,
            infer_symbol_column=infer_symbol_column,
        )

    elif paths is not None:
        plist = list(paths)
        if not plist:
            raise ValueError("paths is empty")
        if symbols is None:
            syms = [_default_symbol_for_source(i, s) for i, s in enumerate(plist)]
        else:
            syms = [str(s) for s in symbols]
            if len(syms) != len(plist):
                raise ValueError(
                    f"symbols length {len(syms)} must match paths length {len(plist)}"
                )
        labeled = list(zip(syms, plist))
        _load_separate_sources_into_out(
            out,
            labeled,
            symbol_col_by_source=sym_col_by_source,
            skiprows=skiprows,
            timestamp_column=timestamp_column,
            timestamp_columns=timestamp_columns,
            kline_columns=kline_columns,
            kline_columns_by_symbol=kline_columns_by_symbol,
            drop_bad_timestamps=drop_bad_timestamps,
            dayfirst=dayfirst,
            timestamp_format=timestamp_format,
            timestamp_formats=timestamp_formats,
            dayfirst_by_symbol=dayfirst_by_symbol,
            infer_symbol_column=infer_symbol_column,
        )

    else:
        assert source is not None
        df = _coerce_source(source, skiprows=skiprows)
        df, coalesced_split_col, coalesce_label = _maybe_coalesce_concat_symbol_columns(
            df, symbol_column
        )
        hint = symbol_column if symbol_column is not None else coalesced_split_col
        sym_col = _resolve_symbol_column(df, hint, infer_symbol_column)
        if sym_col is None:
            msg = _panel_symbol_column_failure_message(df)
            warnings.warn(msg, UserWarning, stacklevel=2)
            raise ValueError(msg)
        sym_col_used = coalesce_label if coalesce_label is not None else sym_col
        panel = df.loc[df[sym_col].notna()].copy()
        for sym_val, sub in panel.groupby(sym_col, sort=False):
            key = str(sym_val).strip()
            sub_ohlc = sub.drop(columns=[sym_col]).reset_index(drop=True)
            fmt_one = _str_per_symbol(key, timestamp_format, timestamp_formats)
            day_one = dayfirst
            if dayfirst_by_symbol is not None and key in dayfirst_by_symbol:
                day_one = bool(dayfirst_by_symbol[key])
            out[key] = _load_market_data_core(
                sub_ohlc,
                timestamp_column=timestamp_column,
                kline_columns=kline_columns,
                drop_bad_timestamps=drop_bad_timestamps,
                dayfirst=day_one,
                timestamp_format=fmt_one,
            )

    def _utc_index_for_compare(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        t = pd.DatetimeIndex(idx)
        if t.tz is None:
            return t.tz_localize("UTC")
        return t.tz_convert("UTC")

    # check whether all symbols share the same datetime index (useful for cross-sectional work)
    aligned = True
    if out:
        indices = [ld.data.index for ld in out.values()]
        first = _utc_index_for_compare(indices[0])
        for idx in indices[1:]:
            if not first.equals(_utc_index_for_compare(idx)):
                aligned = False
                break

    result = MultiAssetMarketData(by_symbol=out, symbol_column=sym_col_used, aligned=aligned)

    if verbose:
        keys = ", ".join(result.symbols)
        print(f"loaded multi-asset market data: {len(out)} symbol(s) [{keys}]")
        if sym_col_used:
            print(f"  panel split on column: {sym_col_used}")
        if not result.aligned:
            print(
                "  warning: symbol time indexes are not fully aligned; "
                "downstream cross-sectional steps should handle missing timestamps."
            )
        for line in result.summary_lines():
            print(f"  {line}")

    return result
