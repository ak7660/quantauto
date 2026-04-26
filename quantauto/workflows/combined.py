"""combine per-symbol ``PipelineResults`` into a single multi-asset portfolio backtest."""

from __future__ import annotations

from typing import Dict, Iterable, Literal, Mapping

import numpy as np
import pandas as pd

from quantauto.backtesting.engine import BacktestResult
from quantauto.backtesting.performance import compute_performance_metrics
from quantauto.backtesting.visualizations import DEFAULT_PLOTS, build_backtest_plots

MultiWeighting = Literal["equal", "inverse_vol", "signal_confidence"]


def _concat_oos_predictions(trainer_result) -> pd.Series:
    sr = trainer_result.best_result.split_results
    if not sr:
        return pd.Series(dtype=float)
    p = pd.concat([s.predictions for s in sr], axis=0)
    p = p[~p.index.duplicated(keep="last")].sort_index()
    return p


def build_combined_multi_backtest(
    by_symbol: Mapping[str, object],
    *,
    weighting: MultiWeighting = "equal",
    inv_vol_lookback: int = 20,
    enable_plots: bool = True,
    default_plots: Iterable[str] = DEFAULT_PLOTS,
    optional_plots: Iterable[str] = (),
) -> BacktestResult:
    """aggregate per-symbol strategy **returns** on the datetime intersection.

    - **equal**: mean of per-symbol ``backtest_result.returns``.
    - **inverse_vol**: inverse rolling vol of each symbol's strategy return, row-normalized weights.
    - **signal_confidence**: weight by OOS ``|prediction|`` (regression) or ``|p-0.5|*2`` (classification).
    """
    syms = sorted(by_symbol.keys())
    if len(syms) < 1:
        raise ValueError("no symbols")
    strat_rets: Dict[str, pd.Series] = {}
    conf: Dict[str, pd.Series] = {}
    for sym in syms:
        pr = by_symbol[sym]
        strat_rets[sym] = pr.backtest_result.returns.copy()
        tr = pr.trainer_result
        task = tr.task_type
        pred = _concat_oos_predictions(tr)
        if task == "regression":
            conf[sym] = pred.abs()
        elif task == "classification":
            conf[sym] = (pred.astype(float) - 0.5).abs() * 2.0
        else:
            conf[sym] = pred.abs()

    idx = strat_rets[syms[0]].index
    for s in syms[1:]:
        idx = idx.intersection(strat_rets[s].index)
    if len(idx) < 2:
        raise ValueError("insufficient overlapping timestamps for combined portfolio")
    idx = idx.sort_values()

    R = pd.DataFrame({s: strat_rets[s].reindex(idx) for s in syms}).dropna(how="all")
    if R.empty:
        raise ValueError("empty strategy return matrix")

    n = R.shape[1]
    if weighting == "equal":
        wrow = pd.DataFrame(np.full(R.shape, 1.0 / n), index=R.index, columns=R.columns)
    elif weighting == "inverse_vol":
        vol = R.rolling(int(inv_vol_lookback), min_periods=2).std().shift(1)
        inv = 1.0 / vol.replace(0.0, np.nan)
        wrow = inv.div(inv.sum(axis=1), axis=0).fillna(1.0 / n)
    else:
        C = pd.DataFrame({s: conf[s].reindex(R.index) for s in syms})
        C = C.fillna(0.0).abs()
        wrow = C.div(C.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(1.0 / n)

    port = (R * wrow).sum(axis=1).dropna()
    eq = (1.0 + port).cumprod()
    dd = eq / eq.cummax() - 1.0
    wchg = wrow.diff().abs().sum(axis=1).reindex(port.index).fillna(0.0)
    full = compute_performance_metrics(
        port,
        eq,
        dd,
        turnover=wchg,
        benchmark_returns=None,
        risk_free_rate_annual=0.0,
    )
    plots = {}
    if enable_plots:
        plots = build_backtest_plots(
            equity_curve=eq,
            drawdown=dd,
            returns=port,
            positions=port,
            benchmark_equity=None,
            default_plots=tuple(default_plots),
            optional_plots=tuple(optional_plots),
            label=f"combined_{weighting}",
        )

    return BacktestResult(
        returns=port,
        returns_gross=port,
        equity_curve=eq,
        drawdown=dd,
        turnover=wchg,
        positions=port,
        summary={
            "total_return": float(full["total_return"]),
            "annualized_return": float(full.get("annualized_return", 0.0)),
            "sharpe_ratio": float(full.get("sharpe_ratio", 0.0)),
            "max_drawdown": float(full.get("max_drawdown", 0.0)),
            "hit_rate": float(full.get("hit_rate", 0.0)),
        },
        metrics_full=dict(full),
        plots=plots,
        scope="combined_multi",
    )
