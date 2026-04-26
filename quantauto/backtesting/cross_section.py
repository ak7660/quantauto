"""cross-sectional (multi-asset) portfolio: top-k and wide-matrix PnL."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

from quantauto.backtesting.engine import BacktestResult
from quantauto.backtesting.execution import trade_cost_rate
from quantauto.backtesting.performance import compute_performance_metrics
from quantauto.backtesting.visualizations import DEFAULT_PLOTS, build_backtest_plots


def _topk_dollar_neutral_w(scores: pd.Series, k: int) -> pd.Series:
    """per timestamp: +0.5/k on top k scores, -0.5/k on bottom k scores."""
    pred = scores.sort_index()
    tidx = pred.index
    tlev = tidx.get_level_values(0)
    w = np.zeros(len(pred), dtype=float)
    for ts in tlev.unique().sort_values():
        m = tlev == ts
        idx_pos = np.flatnonzero(m.to_numpy() if hasattr(m, "to_numpy") else m)
        p = pred.iloc[m]
        if len(p) < 2 * k:
            continue
        vals = p.values
        o_hi = np.argsort(-vals)
        o_lo = np.argsort(vals)
        for j in range(k):
            w[idx_pos[o_hi[j]]] = 0.5 / k
        for j in range(k):
            w[idx_pos[o_lo[j]]] = -0.5 / k
    return pd.Series(w, index=pred.index, name="w")


def predictions_to_topk_long_short_weights(
    pred: pd.Series,
    *,
    k: int,
) -> pd.Series:
    if not isinstance(pred.index, pd.MultiIndex) or pred.index.nlevels < 2:
        raise TypeError("pred must have MultiIndex (time, symbol)")
    return _topk_dollar_neutral_w(pred, k)


def forward_returns_wide(
    multi_close: List[pd.Series] | dict,
    symbols: List[str],
    horizon: int,
) -> pd.DataFrame:
    d: dict = {}
    if isinstance(multi_close, dict):
        for s in symbols:
            d[s] = (multi_close[s].shift(-horizon) / multi_close[s]) - 1.0
    else:
        for s, c in zip(symbols, multi_close):
            d[s] = (c.shift(-horizon) / c) - 1.0
    return pd.DataFrame(d).sort_index()


def run_cross_sectional_from_wide(
    weights_wide: pd.DataFrame,
    ret_wide: pd.DataFrame,
    *,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    enable_plots: bool = True,
    default_plots: Iterable[str] = DEFAULT_PLOTS,
    optional_plots: Iterable[str] = (),
    strategy_label: str = "cross_sectional_strategy",
) -> BacktestResult:
    """portfolio gross = sum_s w_{t,s} * r_{t,s}; cost on sum_s |w_{t,s} - w_{t-1,s}|."""
    c = trade_cost_rate(fee_bps=fee_bps, slippage_bps=slippage_bps)
    idx = weights_wide.index.intersection(ret_wide.index)
    w = weights_wide.reindex(idx).reindex(columns=ret_wide.columns, fill_value=0.0)
    r = ret_wide.reindex(idx)
    g = (w * r).sum(axis=1).astype(float)
    wchg = w.diff().abs().sum(axis=1)
    wchg = wchg.fillna(w.abs().sum(axis=1))
    net = g - c * wchg
    net = net.replace([np.inf, -np.inf], np.nan).dropna()
    if net.empty:
        raise ValueError("no portfolio returns after alignment")
    eq = (1.0 + net).cumprod()
    dd = eq / eq.cummax() - 1.0
    full = compute_performance_metrics(
        net, eq, dd, turnover=wchg.reindex(net.index).fillna(0), benchmark_returns=None, risk_free_rate_annual=0.0
    )
    plots = {}
    if enable_plots:
        bench = None
        plots = build_backtest_plots(
            equity_curve=eq,
            drawdown=dd,
            returns=net,
            positions=weights_wide.sum(axis=1).reindex(net.index).rename("position"),
            benchmark_equity=bench,
            default_plots=tuple(default_plots),
            optional_plots=tuple(optional_plots),
            label=strategy_label,
        )
    return BacktestResult(
        returns=net,
        returns_gross=g.reindex(net.index),
        equity_curve=eq,
        drawdown=dd,
        turnover=wchg.reindex(net.index).fillna(0.0),
        positions=net,
        summary={
            "total_return": float(full["total_return"]),
            "annualized_return": float(full.get("annualized_return", 0.0)),
            "sharpe_ratio": float(full.get("sharpe_ratio", 0.0)),
            "max_drawdown": float(full.get("max_drawdown", 0.0)),
            "hit_rate": float(full.get("hit_rate", 0.0)),
        },
        metrics_full=dict(full),
        plots=plots,
        scope="cross_sectional_wide",
    )
