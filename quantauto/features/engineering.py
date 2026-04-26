"""feature engineering: standard technical indicators and batch feature builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from quantauto.data.schema import KLINE_ROLES, LoadedMarketData, MultiAssetMarketData
from quantauto.features.timing import (
    FeatureMeta,
    apply_execution_shift,
    get_valid_start,
)


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------


@dataclass
class EngineeredFeatures:
    """features computed from a LoadedMarketData frame, aligned to the same DatetimeIndex.

    data        -- feature matrix; UTC DatetimeIndex, one column per feature
    meta        -- one ``FeatureMeta`` per registry ``FeatureSpec`` and per ``CustomFeatureSpec``
    warmup_bars -- leading rows with NaN due to the max lookback across all specs;
                   these rows should be excluded from model training
    source_symbol -- symbol label when built via build_multi_asset_features
    """

    data: pd.DataFrame
    meta: Tuple[FeatureMeta, ...]
    warmup_bars: int
    source_symbol: Optional[str] = None


# ---------------------------------------------------------------------------
# feature specification
# ---------------------------------------------------------------------------


@dataclass
class FeatureSpec:
    """declarative specification for one engineered feature (or feature group).

    name    -- column name prefix in the output DataFrame; multi-output features
               (e.g. macd) append a per-column suffix automatically
    kind    -- registered feature type; see AVAILABLE_FEATURES for valid values
    params  -- keyword arguments forwarded to the feature builder (e.g. window=14)
    lookback -- override inferred lookback; leave None to use the kind default
    description -- optional human-readable note stored in FeatureMeta
    """

    name: str
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)
    lookback: Optional[int] = None
    description: str = ""


@dataclass
class CustomFeatureSpec:
    """user-defined feature built from ``loaded.data`` (see ``build_features(..., custom_specs=)``).

    compute -- function ``(data: DataFrame) -> Series | dict[suffix, Series]``; use suffix ``""``
             for a single column named ``name``, or keys like ``"_line"`` for ``name_line``.
    lookback -- history length in bars for warmup metadata (must match your logic).
    required_columns -- if provided, preset builders skip this spec when a column is missing.
    """

    name: str
    compute: Callable[[pd.DataFrame], Union[pd.Series, Dict[str, pd.Series]]]
    lookback: int
    description: str = ""
    required_columns: Optional[Tuple[str, ...]] = None


# ---------------------------------------------------------------------------
# helper: require a canonical kline column
# ---------------------------------------------------------------------------


def _require_col(data: pd.DataFrame, col: str) -> pd.Series:
    """returns data[col] or raises a clear error if the column is missing."""
    if col not in data.columns:
        present = [c for c in data.columns if c in KLINE_ROLES]
        raise KeyError(
            f"feature requires column {col!r}; available kline columns: {present}. "
            "ensure data was loaded with load_market_data which normalizes OHLCV names."
        )
    return data[col]


# ---------------------------------------------------------------------------
# pure indicator functions — each uses only past data (no negative shifts)
# ---------------------------------------------------------------------------


def calculate_returns(close: pd.Series, *, period: int = 1) -> pd.Series:
    """percentage return over `period` bars: (close_t / close_{t-period}) - 1.

    first `period` rows are NaN (insufficient history).
    """
    return close.pct_change(periods=period)


def calculate_log_returns(close: pd.Series, *, period: int = 1) -> pd.Series:
    """log return over `period` bars: log(close_t / close_{t-period}).

    first `period` rows are NaN.
    """
    return np.log(close / close.shift(period))


def calculate_sma(series: pd.Series, *, window: int) -> pd.Series:
    """simple moving average over `window` bars.

    first `window - 1` rows are NaN (min_periods=window).
    """
    return series.rolling(window=window, min_periods=window).mean()


def calculate_ema(series: pd.Series, *, span: int) -> pd.Series:
    """exponential moving average with given `span`.

    uses adjust=False (recursive formula).  first `span - 1` rows may have
    less-stable values since EWM initialises from the first observation.
    """
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def calculate_rolling_std(series: pd.Series, *, window: int) -> pd.Series:
    """rolling standard deviation over `window` bars (ddof=1).

    first `window - 1` rows are NaN.
    """
    return series.rolling(window=window, min_periods=window).std()


def calculate_rolling_zscore(series: pd.Series, *, window: int) -> pd.Series:
    """rolling z-score: (value - rolling_mean) / rolling_std over `window` bars.

    useful for normalising a feature at the same time it is computed, with
    no future information used (window looks only backward).
    first `window - 1` rows are NaN.
    """
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    return (series - mean) / std.replace(0.0, np.nan)


def calculate_rate_of_change(close: pd.Series, *, period: int = 10) -> pd.Series:
    """rate of change as percentage: (close_t - close_{t-period}) / close_{t-period} * 100.

    first `period` rows are NaN.
    """
    shifted = close.shift(period)
    return (close - shifted) / shifted.replace(0.0, np.nan) * 100.0


def calculate_rsi(close: pd.Series, *, window: int = 14) -> pd.Series:
    """wilder's relative strength index (0–100 scale).

    uses ewm with alpha=1/window (wilder smoothing).
    first `window` rows are NaN (min_periods=window on both avg_gain/avg_loss).
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / window
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    # avg_loss == 0 → rs = inf → rsi = 100 (no losing bars); pandas handles inf correctly here
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def calculate_macd(
    close: pd.Series,
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """macd: ema_fast - ema_slow, signal line (ema of macd), and histogram.

    returns a DataFrame with columns: macd_line, signal, histogram.
    uses adjust=False ema throughout.  first meaningful values appear after
    roughly `slow + signal` bars.
    """
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=1).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=1).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd_line": macd_line, "signal": signal_line, "histogram": histogram},
        index=close.index,
    )


def calculate_bollinger_bands(
    close: pd.Series,
    *,
    window: int = 20,
    n_std: float = 2.0,
) -> pd.DataFrame:
    """bollinger bands: mid (sma), upper, lower, width, and percent_b.

    returns a DataFrame with columns: mid, upper, lower, width, pct_b.
    - width  = (upper - lower) / mid  (normalised band width)
    - pct_b  = (close - lower) / (upper - lower)  (position within bands; 0.5 = mid)
    first `window - 1` rows are NaN.
    """
    mid = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    band_range = (upper - lower).replace(0.0, np.nan)
    width = band_range / mid.replace(0.0, np.nan)
    pct_b = (close - lower) / band_range
    return pd.DataFrame(
        {"mid": mid, "upper": upper, "lower": lower, "width": width, "pct_b": pct_b},
        index=close.index,
    )


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    window: int = 14,
) -> pd.Series:
    """average true range using wilder smoothing (ewm alpha=1/window).

    true range = max(high-low, |high-prev_close|, |low-prev_close|).
    first `window` rows are NaN (min_periods=window on ewm).
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """on-balance volume: cumulative volume signed by price direction.

    direction = +1 if close > prev_close, -1 if close < prev_close, 0 if equal.
    the first row gets direction 0 (no previous close available).
    no NaN warmup; value is cumulative from the first bar.
    """
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def calculate_volume_ratio(volume: pd.Series, *, window: int = 20) -> pd.Series:
    """volume relative to its rolling mean: volume_t / sma(volume, window).

    values > 1 indicate above-average volume.
    first `window - 1` rows are NaN.
    """
    sma_vol = volume.rolling(window=window, min_periods=window).mean()
    return volume / sma_vol.replace(0.0, np.nan)


def calculate_high_low_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """normalised intrabar range: (high - low) / close.

    measures intrabar volatility as a fraction of price.  no lookback required;
    uses only the current bar's OHLC.
    """
    return (high - low) / close.replace(0.0, np.nan)


def calculate_close_position(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """close position within the bar's high-low range, scaled 0–1.

    0 = close at bar low, 1 = close at bar high.
    NaN when high == low (zero-range bar).
    """
    bar_range = (high - low).replace(0.0, np.nan)
    return (close - low) / bar_range


# ---------------------------------------------------------------------------
# registry infrastructure
# ---------------------------------------------------------------------------


class _RegistryEntry(NamedTuple):
    """builder function and lookback inferrer for one feature kind."""

    build: Callable[..., Dict[str, pd.Series]]
    infer_lookback: Callable[..., int]


def _build_returns(data: pd.DataFrame, **params: Any) -> Dict[str, pd.Series]:
    """builder: percentage returns."""
    period: int = params.get("period", 1)
    return {"": calculate_returns(_require_col(data, "close"), period=period)}


def _build_log_returns(data: pd.DataFrame, **params: Any) -> Dict[str, pd.Series]:
    """builder: log returns."""
    period: int = params.get("period", 1)
    return {"": calculate_log_returns(_require_col(data, "close"), period=period)}


def _build_sma(data: pd.DataFrame, **params: Any) -> Dict[str, pd.Series]:
    """builder: simple moving average."""
    window: int = params["window"]
    col: str = params.get("column", "close")
    return {"": calculate_sma(_require_col(data, col), window=window)}


def _build_ema(data: pd.DataFrame, **params: Any) -> Dict[str, pd.Series]:
    """builder: exponential moving average."""
    span: int = params["span"]
    col: str = params.get("column", "close")
    return {"": calculate_ema(_require_col(data, col), span=span)}


def _build_rolling_std(data: pd.DataFrame, **params: Any) -> Dict[str, pd.Series]:
    """builder: rolling standard deviation."""
    window: int = params["window"]
    col: str = params.get("column", "close")
    return {"": calculate_rolling_std(_require_col(data, col), window=window)}


def _build_rolling_zscore(data: pd.DataFrame, **params: Any) -> Dict[str, pd.Series]:
    """builder: rolling z-score."""
    window: int = params["window"]
    col: str = params.get("column", "close")
    return {"": calculate_rolling_zscore(_require_col(data, col), window=window)}


def _build_rate_of_change(data: pd.DataFrame, **params: Any) -> Dict[str, pd.Series]:
    """builder: rate of change."""
    period: int = params.get("period", 10)
    return {"": calculate_rate_of_change(_require_col(data, "close"), period=period)}


def _build_rsi(data: pd.DataFrame, **params: Any) -> Dict[str, pd.Series]:
    """builder: rsi."""
    window: int = params.get("window", 14)
    return {"": calculate_rsi(_require_col(data, "close"), window=window)}


def _build_macd(data: pd.DataFrame, **params: Any) -> Dict[str, pd.Series]:
    """builder: macd (returns three sub-columns: _line, _signal, _hist)."""
    fast: int = params.get("fast", 12)
    slow: int = params.get("slow", 26)
    signal: int = params.get("signal", 9)
    df = calculate_macd(_require_col(data, "close"), fast=fast, slow=slow, signal=signal)
    return {
        "_line": df["macd_line"],
        "_signal": df["signal"],
        "_hist": df["histogram"],
    }


def _build_bollinger_bands(data: pd.DataFrame, **params: Any) -> Dict[str, pd.Series]:
    """builder: bollinger bands (returns five sub-columns: _mid, _upper, _lower, _width, _pctb)."""
    window: int = params.get("window", 20)
    n_std: float = params.get("n_std", 2.0)
    df = calculate_bollinger_bands(
        _require_col(data, "close"), window=window, n_std=n_std
    )
    return {
        "_mid": df["mid"],
        "_upper": df["upper"],
        "_lower": df["lower"],
        "_width": df["width"],
        "_pctb": df["pct_b"],
    }


def _build_atr(data: pd.DataFrame, **params: Any) -> Dict[str, pd.Series]:
    """builder: average true range."""
    window: int = params.get("window", 14)
    return {
        "": calculate_atr(
            _require_col(data, "high"),
            _require_col(data, "low"),
            _require_col(data, "close"),
            window=window,
        )
    }


def _build_obv(data: pd.DataFrame, **params: Any) -> Dict[str, pd.Series]:
    """builder: on-balance volume."""
    return {
        "": calculate_obv(
            _require_col(data, "close"),
            _require_col(data, "volume"),
        )
    }


def _build_volume_ratio(data: pd.DataFrame, **params: Any) -> Dict[str, pd.Series]:
    """builder: volume ratio relative to rolling mean."""
    window: int = params.get("window", 20)
    return {"": calculate_volume_ratio(_require_col(data, "volume"), window=window)}


def _build_high_low_range(data: pd.DataFrame, **params: Any) -> Dict[str, pd.Series]:
    """builder: normalised intrabar high-low range."""
    return {
        "": calculate_high_low_range(
            _require_col(data, "high"),
            _require_col(data, "low"),
            _require_col(data, "close"),
        )
    }


def _build_close_position(data: pd.DataFrame, **params: Any) -> Dict[str, pd.Series]:
    """builder: close position within bar range (0=low, 1=high)."""
    return {
        "": calculate_close_position(
            _require_col(data, "open"),
            _require_col(data, "high"),
            _require_col(data, "low"),
            _require_col(data, "close"),
        )
    }


# lookback inferrers per feature kind

def _lookback_returns(**params: Any) -> int:
    """lookback for returns: period + 1."""
    return int(params.get("period", 1)) + 1


def _lookback_log_returns(**params: Any) -> int:
    """lookback for log returns: period + 1."""
    return int(params.get("period", 1)) + 1


def _lookback_sma(**params: Any) -> int:
    """lookback for sma: window."""
    return int(params["window"])


def _lookback_ema(**params: Any) -> int:
    """lookback for ema: span (ema initialises from bar 1 but needs span bars for stability)."""
    return int(params["span"])


def _lookback_rolling_std(**params: Any) -> int:
    """lookback for rolling std: window."""
    return int(params["window"])


def _lookback_rolling_zscore(**params: Any) -> int:
    """lookback for rolling zscore: window."""
    return int(params["window"])


def _lookback_rate_of_change(**params: Any) -> int:
    """lookback for roc: period + 1."""
    return int(params.get("period", 10)) + 1


def _lookback_rsi(**params: Any) -> int:
    """lookback for rsi: window + 1 (one extra bar for diff)."""
    return int(params.get("window", 14)) + 1


def _lookback_macd(**params: Any) -> int:
    """lookback for macd: slow + signal bars."""
    return int(params.get("slow", 26)) + int(params.get("signal", 9))


def _lookback_bollinger_bands(**params: Any) -> int:
    """lookback for bollinger bands: window."""
    return int(params.get("window", 20))


def _lookback_atr(**params: Any) -> int:
    """lookback for atr: window (ewm min_periods=window; tr[0] uses high-low so all tr rows valid)."""
    return int(params.get("window", 14))


def _lookback_obv(**params: Any) -> int:
    """lookback for obv: 1 (cumulative from first bar, no warmup)."""
    return 1


def _lookback_volume_ratio(**params: Any) -> int:
    """lookback for volume ratio: window."""
    return int(params.get("window", 20))


def _lookback_high_low_range(**params: Any) -> int:
    """lookback for high-low range: 1 (uses current bar only)."""
    return 1


def _lookback_close_position(**params: Any) -> int:
    """lookback for close position: 1 (uses current bar only)."""
    return 1


# central registry: kind string -> (build function, lookback inferrer)
_FEATURE_REGISTRY: Dict[str, _RegistryEntry] = {
    "returns": _RegistryEntry(_build_returns, _lookback_returns),
    "log_returns": _RegistryEntry(_build_log_returns, _lookback_log_returns),
    "sma": _RegistryEntry(_build_sma, _lookback_sma),
    "ema": _RegistryEntry(_build_ema, _lookback_ema),
    "rolling_std": _RegistryEntry(_build_rolling_std, _lookback_rolling_std),
    "rolling_zscore": _RegistryEntry(_build_rolling_zscore, _lookback_rolling_zscore),
    "rate_of_change": _RegistryEntry(_build_rate_of_change, _lookback_rate_of_change),
    "rsi": _RegistryEntry(_build_rsi, _lookback_rsi),
    "macd": _RegistryEntry(_build_macd, _lookback_macd),
    "bollinger_bands": _RegistryEntry(_build_bollinger_bands, _lookback_bollinger_bands),
    "atr": _RegistryEntry(_build_atr, _lookback_atr),
    "obv": _RegistryEntry(_build_obv, _lookback_obv),
    "volume_ratio": _RegistryEntry(_build_volume_ratio, _lookback_volume_ratio),
    "high_low_range": _RegistryEntry(_build_high_low_range, _lookback_high_low_range),
    "close_position": _RegistryEntry(_build_close_position, _lookback_close_position),
}

# public listing of registered feature kinds
AVAILABLE_FEATURES: Tuple[str, ...] = tuple(sorted(_FEATURE_REGISTRY.keys()))


def _coerce_custom_compute_output(
    raw: Union[pd.Series, Dict[str, pd.Series]],
    index: pd.Index,
    spec_name: str,
) -> Dict[str, pd.Series]:
    """normalises custom compute output to a suffix -> series map aligned to ``index``."""
    if isinstance(raw, pd.Series):
        series_dict = {"": raw.reindex(index)}
    elif isinstance(raw, dict):
        series_dict = {}
        for suffix, s in raw.items():
            if not isinstance(s, pd.Series):
                raise TypeError(
                    f"custom feature {spec_name!r}: expected Series values in dict, "
                    f"got {type(s).__name__}"
                )
            series_dict[str(suffix)] = s.reindex(index)
    else:
        raise TypeError(
            f"custom feature {spec_name!r}: compute must return Series or dict, "
            f"got {type(raw).__name__}"
        )
    return series_dict


# ---------------------------------------------------------------------------
# main batch builder
# ---------------------------------------------------------------------------


def build_features(
    loaded: LoadedMarketData,
    specs: Sequence[FeatureSpec],
    *,
    execution_shift: int = 0,
    drop_warmup: bool = False,
    custom_specs: Sequence[CustomFeatureSpec] = (),
) -> EngineeredFeatures:
    """computes all features in `specs` from one LoadedMarketData frame.

    each FeatureSpec maps a `kind` to a registered builder and gives the output
    column name(s).  multi-output features (macd, bollinger_bands) append their
    own suffix to the spec name (e.g. name='bb' produces bb_mid, bb_upper …).

    execution_shift -- shift all features forward by this many bars so a feature
                       computed on bar T is only visible at bar T+execution_shift
                       (models execution lag; 0 = no shift)
    drop_warmup     -- if True, trim the leading warmup_bars rows from the output
                       so the returned DataFrame has no NaN rows at the head

    raises ValueError for unknown feature kinds.
    raises KeyError if a required OHLCV column is missing from loaded.data.

    custom_specs -- optional user-defined features (see ``CustomFeatureSpec``); computed
                  after registry features. column names must not collide with earlier outputs.
    """
    if not specs and not custom_specs:
        raise ValueError(
            "pass at least one FeatureSpec or CustomFeatureSpec (both sequences are empty)"
        )

    data = loaded.data
    all_series: Dict[str, pd.Series] = {}
    all_meta: List[FeatureMeta] = []
    max_lookback = 0

    for spec in specs:
        entry = _FEATURE_REGISTRY.get(spec.kind)
        if entry is None:
            available = ", ".join(AVAILABLE_FEATURES)
            raise ValueError(
                f"unknown feature kind {spec.kind!r}. "
                f"available kinds: {available}"
            )

        # compute the feature(s)
        series_dict = entry.build(data, **spec.params)

        # resolve lookback
        lookback = (
            spec.lookback
            if spec.lookback is not None
            else entry.infer_lookback(**spec.params)
        )

        # store series with correct column names
        if len(series_dict) == 1 and "" in series_dict:
            all_series[spec.name] = series_dict[""]
        else:
            for suffix, s in series_dict.items():
                all_series[spec.name + suffix] = s

        all_meta.append(
            FeatureMeta(
                name=spec.name,
                lookback_bars=lookback,
                description=spec.description,
            )
        )
        max_lookback = max(max_lookback, lookback)

    for cspec in custom_specs:
        series_dict = _coerce_custom_compute_output(
            cspec.compute(data), data.index, cspec.name
        )
        lb = cspec.lookback
        if len(series_dict) == 1 and "" in series_dict:
            all_series[cspec.name] = series_dict[""]
        else:
            for suffix, s in series_dict.items():
                all_series[cspec.name + suffix] = s
        all_meta.append(
            FeatureMeta(
                name=cspec.name,
                lookback_bars=lb,
                description=cspec.description,
            )
        )
        max_lookback = max(max_lookback, lb)

    feature_df = pd.DataFrame(all_series, index=data.index)

    if execution_shift > 0:
        feature_df = apply_execution_shift(feature_df, periods=execution_shift)

    warmup_bars = get_valid_start(max_lookback)
    if drop_warmup and warmup_bars > 0:
        feature_df = feature_df.iloc[warmup_bars:].copy()
        warmup_bars = 0  # already trimmed

    return EngineeredFeatures(
        data=feature_df,
        meta=tuple(all_meta),
        warmup_bars=warmup_bars,
    )


def build_multi_asset_features(
    multi: MultiAssetMarketData,
    specs: Sequence[FeatureSpec],
    *,
    execution_shift: int = 0,
    drop_warmup: bool = False,
    custom_specs: Sequence[CustomFeatureSpec] = (),
) -> Dict[str, EngineeredFeatures]:
    """applies build_features independently to each symbol in a MultiAssetMarketData.

    returns a dict keyed by symbol with one EngineeredFeatures per asset.
    each asset's features are computed entirely from its own time series
    (no cross-asset contamination).  for cross-sectional features, use
    transforms.cross_sectional_zscore / cross_sectional_rank on the result.

    custom_specs -- forwarded to :func:`build_features` for every symbol.
    """
    out: Dict[str, EngineeredFeatures] = {}
    for sym, loaded in multi.items():
        ef = build_features(
            loaded,
            specs,
            execution_shift=execution_shift,
            drop_warmup=drop_warmup,
            custom_specs=custom_specs,
        )
        out[sym] = EngineeredFeatures(
            data=ef.data,
            meta=ef.meta,
            warmup_bars=ef.warmup_bars,
            source_symbol=sym,
        )
    return out
