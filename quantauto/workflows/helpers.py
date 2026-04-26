"""helpers for formatting workflow outputs."""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

from quantauto.backtesting.engine import BacktestResult
from quantauto.models.types import TrainerResult

if TYPE_CHECKING:
    from quantauto.workflows.pipeline import MultiPipelineResults


def elaborate_ml_results(trainer_result: TrainerResult) -> Dict[str, object]:
    return {
        "best_model_id": trainer_result.best_model_id,
        "leaderboard": trainer_result.leaderboard.copy(),
        "best_metrics": dict(trainer_result.best_result.aggregate_metrics),
    }


def elaborate_backtest_results(backtest_result: BacktestResult) -> Dict[str, object]:
    return {
        "summary": dict(backtest_result.summary),
        "metrics_full": dict(backtest_result.metrics_full),
        "equity_curve": backtest_result.equity_curve.copy(),
        "returns": backtest_result.returns.copy(),
        "returns_gross": backtest_result.returns_gross.copy(),
        "drawdown": backtest_result.drawdown.copy(),
        "turnover": backtest_result.turnover.copy(),
        "positions": backtest_result.positions.copy(),
        "available_plots": tuple(backtest_result.plots.keys()),
        "scope": backtest_result.scope,
    }


def elaborate_multi_results(multi_result: "MultiPipelineResults") -> Dict[str, object]:
    return {
        "aggregate_summary": multi_result.aggregate_summary.copy(),
        "symbols": tuple(sorted(multi_result.by_symbol.keys())),
        "errors": dict(multi_result.errors),
        "combined_backtest_results": (
            dict(multi_result.combined_backtest_results)
            if multi_result.combined_backtest_results is not None
            else None
        ),
        "per_symbol_best_models": {
            sym: res.trainer_result.best_model_id for sym, res in multi_result.by_symbol.items()
        },
    }
