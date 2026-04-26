"""backtesting engine exports."""

from quantauto.backtesting.engine import BacktestResult, predictions_to_positions, run_backtest
from quantauto.backtesting.performance import compute_performance_metrics
from quantauto.backtesting.visualizations import build_backtest_plots, iter_unique_plot_figures

__all__ = [
    "BacktestResult",
    "predictions_to_positions",
    "run_backtest",
    "compute_performance_metrics",
    "build_backtest_plots",
    "iter_unique_plot_figures",
]
