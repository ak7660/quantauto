"""stacked (time, symbol) panel for cross-sectional training and ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quantauto.data.schema import MultiAssetMarketData
from quantauto.features import build_multi_preset_features
from quantauto.features.engineering import EngineeredFeatures
from quantauto.labels.timing import check_label_feature_overlap, get_valid_label_range
from quantauto.workflows.config import DEFAULT_PRESET


@dataclass(frozen=True)
class PanelTrainingDataset:
    """aligned features + forward-return label, sorted by (time, symbol)."""

    X: pd.DataFrame
    y: pd.Series
    index: pd.MultiIndex
    horizon: int
    warmup_bars: int
    symbols: Tuple[str, ...]
    time_index: pd.DatetimeIndex
    # optional: same index, forward return (realized) for backtests / de-dup
    y_forward_return: Optional[pd.Series] = None

    @property
    def n_rows(self) -> int:
        return int(len(self.X))


def _forward_return_per_symbol(
    close: pd.Series,
    horizon: int,
) -> pd.Series:
    return (close.shift(-horizon) / close) - 1.0


def group_sizes_in_order(index: pd.MultiIndex) -> np.ndarray:
    """consecutive same timestamp counts for sorted (time, symbol) index."""
    if len(index) == 0:
        return np.zeros(0, dtype=np.int32)
    times = index.get_level_values(0)
    out: List[int] = []
    c = 1
    for i in range(1, len(times)):
        if times[i] == times[i - 1]:
            c += 1
        else:
            out.append(c)
            c = 1
    out.append(c)
    return np.array(out, dtype=np.int32)


def build_panel_training_dataset(
    multi: MultiAssetMarketData,
    *,
    feature_preset: str = "base",
    target_horizon: int = 1,
    execution_shift: int = 1,
    verbose: int = 0,
) -> PanelTrainingDataset:
    """build stacked panel: inner join per-symbol features with forward return label.

    uses :func:`quantauto.features.build_multi_preset_features` per symbol, then
    drops rows with missing features or label (tail ``horizon`` rows per symbol).
    """
    preset: str = str(feature_preset) if feature_preset else DEFAULT_PRESET
    if not multi.by_symbol:
        raise ValueError("empty multi-asset data")
    per_sym: Dict[str, EngineeredFeatures] = build_multi_preset_features(
        multi,
        preset,  # type: ignore[arg-type]
        execution_shift=execution_shift,
        drop_warmup=False,
    )
    syms = tuple(sorted(per_sym.keys()))
    if preset == "heavy":
        _append_heavy_cross_asset_corr_features(
            per_sym=per_sym,
            multi=multi,
            symbols=syms,
            execution_shift=execution_shift,
        )
    if verbose >= 1:
        print(
            f"[run_auto:ranking] feature engineering preset={preset} symbols={len(syms)}"
        )
        for sym in syms:
            ef = per_sym[sym]
            print(
                f"[run_auto:ranking] features[{sym}] rows={len(ef.data)} cols={ef.data.shape[1]} warmup={ef.warmup_bars}"
            )
            if verbose >= 2:
                cols = list(ef.data.columns)
                preview = ", ".join(cols[:40])
                if len(cols) > 40:
                    preview += ", ..."
                print(f"[run_auto:ranking] features[{sym}] names={preview}")
    rows: List[Tuple[pd.Timestamp, str, List[float], float]] = []
    max_wu = 0
    fcols: Optional[Tuple[str, ...]] = None
    for sym in syms:
        eng = per_sym[sym]
        max_wu = max(max_wu, int(eng.warmup_bars))
        ld = multi.by_symbol[sym]
        close = ld.data["close"]
        y_raw = _forward_return_per_symbol(close, target_horizon)
        F = eng.data
        if fcols is None:
            fcols = tuple(F.columns)
        if tuple(F.columns) != fcols:
            raise ValueError(f"symbol {sym!r}: feature columns differ across assets")
        f_ix = F.index
        if not isinstance(f_ix, pd.DatetimeIndex):
            f_ix = pd.DatetimeIndex(f_ix)
        valid_t = get_valid_label_range(f_ix, target_horizon)
        for t in valid_t:
            if t not in F.index or t not in y_raw.index:
                continue
            yv = y_raw.get(t, np.nan)
            if pd.isna(yv):
                continue
            fr = F.loc[t]
            if fr.isna().any():
                continue
            pos = F.index.get_loc(t)
            p_i: int
            if isinstance(pos, (int, np.integer)):
                p_i = int(pos)
            else:
                p_i = 0
            if eng.warmup_bars and p_i < eng.warmup_bars:
                continue
            xvec = [float(fr[c]) for c in fcols]
            rows.append((t, sym, xvec, float(yv)))

    if not rows:
        raise ValueError("no rows after panel join; check horizons and NaNs")
    idx = pd.MultiIndex.from_tuples(
        [(r[0], r[1]) for r in rows],
        names=["time", "symbol"],
    )
    X = pd.DataFrame(
        [r[2] for r in rows],
        index=idx,
        columns=fcols,
    )
    y = pd.Series([r[3] for r in rows], index=idx, name="y_forward")
    if fcols is not None:
        safe, msg = check_label_feature_overlap("y_forward", fcols)
        if not safe:
            raise ValueError(msg)
    time_u = pd.DatetimeIndex(X.index.get_level_values(0).unique().sort_values())
    return PanelTrainingDataset(
        X=X.sort_index(),
        y=y.reindex(X.index),
        index=X.index,
        horizon=target_horizon,
        warmup_bars=max_wu,
        symbols=syms,
        time_index=time_u,
        y_forward_return=y,
    )


def _append_heavy_cross_asset_corr_features(
    *,
    per_sym: Dict[str, EngineeredFeatures],
    multi: MultiAssetMarketData,
    symbols: Tuple[str, ...],
    execution_shift: int,
    corr_window: int = 24,
) -> None:
    """Add rolling return-correlation features vs every reference asset.

    leakage safety:
    - uses rolling correlation of 1-bar close returns (historical only)
    - then shifts by execution_shift (same lag convention as other engineered features)
    """
    closes: Dict[str, pd.Series] = {
        sym: multi.by_symbol[sym].data["close"].astype(float) for sym in symbols
    }
    ret_df = pd.DataFrame({sym: closes[sym].pct_change() for sym in symbols}).sort_index()

    for sym in symbols:
        ef = per_sym[sym]
        fdf = ef.data.copy()
        own = ret_df[sym]
        for ref in symbols:
            corr = own.rolling(int(corr_window), min_periods=3).corr(ret_df[ref])
            if execution_shift > 0:
                corr = corr.shift(int(execution_shift))
            col = f"xcorr_ret_to_{ref}"
            fdf[col] = corr.reindex(fdf.index)
        per_sym[sym] = EngineeredFeatures(
            data=fdf,
            meta=ef.meta,
            warmup_bars=max(int(ef.warmup_bars), int(corr_window)),
            source_symbol=ef.source_symbol,
        )
