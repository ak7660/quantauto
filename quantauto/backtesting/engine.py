"""backtesting orchestration entrypoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from quantauto.backtesting.execution import (
    predictions_to_positions as _predictions_to_positions,
    trade_cost_rate,
)
from quantauto.backtesting.performance import compute_performance_metrics
from quantauto.backtesting.portfolio import PortfolioPath, build_portfolio_path
from quantauto.backtesting.visualizations import DEFAULT_PLOTS, build_backtest_plots


@dataclass
class BacktestResult:
    returns: pd.Series  # net returns
    returns_gross: pd.Series
    equity_curve: pd.Series
    drawdown: pd.Series
    turnover: pd.Series
    positions: pd.Series
    summary: Dict[str, float]
    metrics_full: Dict[str, float] = field(default_factory=dict)
    plots: Dict[str, object] = field(default_factory=dict)
    scope: str = "all_test_folds"


def predictions_to_positions(
    predictions: pd.Series,
    *,
    task_type: str,
    threshold: float = 0.0,
) -> pd.Series:
    return _predictions_to_positions(
        predictions,
        task_type=task_type,
        threshold=threshold,
    )


def run_backtest(
    predictions: pd.Series,
    realized_returns: pd.Series,
    *,
    task_type: str,
    threshold: float = 0.0,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate_annual: float = 0.0,
    enable_plots: bool = True,
    default_plots: Iterable[str] = DEFAULT_PLOTS,
    optional_plots: Iterable[str] = (),
    use_numba: bool = True,
    scope: str = "all_test_folds",
    strategy_label: str = "strategy",
    benchmark_label: str = "benchmark",
) -> BacktestResult:
    aligned = pd.concat(
        [predictions.rename("pred"), realized_returns.rename("ret")], axis=1
    ).dropna()
    if aligned.empty:
        raise ValueError("no overlapping prediction/return rows to backtest")
    positions = predictions_to_positions(
        aligned["pred"], task_type=task_type, threshold=threshold
    )
    path: PortfolioPath = build_portfolio_path(
        positions=positions,
        realized_returns=aligned["ret"],
        trade_cost_rate=trade_cost_rate(fee_bps=fee_bps, slippage_bps=slippage_bps),
        use_numba=use_numba,
    )
    bench_equity = None
    if benchmark_returns is not None:
        b = benchmark_returns.reindex(path.returns_net.index).dropna()
        if len(b):
            bench_equity = (1.0 + b).cumprod().rename("benchmark_equity")
    full = compute_performance_metrics(
        path.returns_net,
        path.equity_curve,
        path.drawdown,
        turnover=path.turnover,
        benchmark_returns=benchmark_returns,
        risk_free_rate_annual=risk_free_rate_annual,
    )
    summary = {
        "total_return": float(full["total_return"]),
        "annualized_return": float(full["annualized_return"]),
        "sharpe_ratio": float(full["sharpe_ratio"]),
        "max_drawdown": float(full["max_drawdown"]),
        "hit_rate": float(full["hit_rate"]),
    }
    plots = {}
    if enable_plots:
        plots = build_backtest_plots(
            equity_curve=path.equity_curve,
            drawdown=path.drawdown,
            returns=path.returns_net,
            positions=path.positions,
            benchmark_equity=bench_equity,
            default_plots=tuple(default_plots),
            optional_plots=tuple(optional_plots),
            label=strategy_label,
            benchmark_label=benchmark_label,
        )
    return BacktestResult(
        returns=path.returns_net,
        returns_gross=path.returns_gross,
        equity_curve=path.equity_curve,
        drawdown=path.drawdown,
        turnover=path.turnover,
        positions=path.positions,
        summary=summary,
        metrics_full=full,
        plots=plots,
        scope=scope,
    )
