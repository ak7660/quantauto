"""validation for time-indexed market frames (integrity, gaps, basic leakage hints)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from quantauto.data.schema import (
    LoadedMarketData,
    MultiAssetMarketData,
    normalize_column_key,
)


def _freq_to_timedelta(freq: str) -> pd.Timedelta:
    """converts a pandas frequency string to a step timedelta using a short anchor range."""
    idx = pd.date_range("2020-01-01", periods=2, freq=freq, tz="UTC")
    return idx[1] - idx[0]


@dataclass
class ValidationResult:
    """outcome of validating one or more market frames."""

    ok: bool
    errors: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    by_symbol: Optional[Dict[str, "ValidationResult"]] = None

    @staticmethod
    def merge_symbol_results(parts: Dict[str, "ValidationResult"]) -> "ValidationResult":
        """aggregates per-symbol results into one result."""
        errs: List[str] = []
        warns: List[str] = []
        all_ok = True
        for sym, r in sorted(parts.items()):
            if not r.ok:
                all_ok = False
            for e in r.errors:
                errs.append(f"{sym}: {e}")
            for w in r.warnings:
                warns.append(f"{sym}: {w}")
        return ValidationResult(
            ok=all_ok,
            errors=tuple(errs),
            warnings=tuple(warns),
            by_symbol=dict(parts),
        )


def _flag_forward_looking_column_names(columns: Tuple[str, ...]) -> Tuple[str, ...]:
    """returns warning lines for column names that often indicate future leakage."""
    hints = (
        "future",
        "forward",
        "fwd",
        "lead",
        "_y_next",
        "next_ret",
        "label_fwd",
    )
    lines: List[str] = []
    for c in columns:
        nk = normalize_column_key(str(c))
        if any(h in nk for h in hints):
            lines.append(
                f"column {c!r} looks forward-looking or label-like; "
                "confirm it is shifted or excluded from features at train time."
            )
    return tuple(lines)


def validate_market_frame(
    df: pd.DataFrame,
    *,
    expected_freq: Optional[str] = None,
    allow_timezone_naive: bool = False,
    strict: bool = False,
    check_forward_looking_names: bool = True,
) -> ValidationResult:
    """checks a single-instrument frame: datetime index, order, duplicates, optional gaps.

    this does not prove absence of leakage in feature values; it applies structural checks
    and lightweight name heuristics. use strict=True to treat duplicate timestamps as errors.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append(
            f"index must be a DatetimeIndex, got {type(df.index).__name__}"
        )
        return ValidationResult(ok=False, errors=tuple(errors), warnings=tuple(warnings))

    if len(df) == 0:
        warnings.append("frame is empty")
        return ValidationResult(ok=True, errors=(), warnings=tuple(warnings))

    if df.index.has_duplicates:
        msg = "index contains duplicate timestamps"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg + "; loader normally keeps last per timestamp")

    if not df.index.is_monotonic_increasing:
        errors.append("index is not monotonic increasing (sort ascending before modeling)")

    if df.index.tz is None:
        if allow_timezone_naive:
            warnings.append("index is timezone-naive; prefer UTC for consistency with loaders")
        else:
            warnings.append(
                "index is timezone-naive; load_market_data uses UTC — consider localizing"
            )

    if expected_freq is not None:
        try:
            step = _freq_to_timedelta(expected_freq)
        except (ValueError, TypeError) as e:
            warnings.append(f"could not parse expected_freq={expected_freq!r}: {e}")
        else:
            diffs = df.index.to_series().diff().dropna()
            if len(diffs):
                # allow modest jitter above nominal bar size
                tol = step * 1.5
                big = diffs > tol
                if big.any():
                    n = int(big.sum())
                    warnings.append(
                        f"detected {n} gap(s) larger than ~1.5× expected step ({step}); "
                        "series may be irregular or sessions may differ across assets"
                    )

    if check_forward_looking_names:
        for line in _flag_forward_looking_column_names(tuple(df.columns)):
            warnings.append(line)

    ok = len(errors) == 0
    return ValidationResult(ok=ok, errors=tuple(errors), warnings=tuple(warnings))


def validate_loaded_market_data(
    loaded: LoadedMarketData,
    **kwargs,
) -> ValidationResult:
    """runs validate_market_frame on ``loaded.data``."""
    return validate_market_frame(loaded.data, **kwargs)


def validate_multi_asset_market_data(
    multi: MultiAssetMarketData,
    **kwargs,
) -> ValidationResult:
    """validates each symbol and aggregates; adds a warning if ``multi.aligned`` is false."""
    parts = {
        sym: validate_loaded_market_data(ld, **kwargs)
        for sym, ld in multi.by_symbol.items()
    }
    merged = ValidationResult.merge_symbol_results(parts)
    warns = list(merged.warnings)
    if not multi.aligned:
        warns.append(
            "multi-asset indexes are not identical across symbols; "
            "cross-sectional ranking should reindex or handle missing bars explicitly"
        )
    return ValidationResult(
        ok=merged.ok,
        errors=merged.errors,
        warnings=tuple(warns),
        by_symbol=merged.by_symbol,
    )
