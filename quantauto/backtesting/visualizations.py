"""matplotlib visualization helpers for backtesting."""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, Mapping, Optional, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_PLOTS = ("equity_curve", "returns_distribution")
OPTIONAL_PLOTS = (
    "rolling_sharpe",
    "monthly_heatmap",
    "drawdown_timeline",
    "positions",
)


def build_backtest_plots(
    *,
    equity_curve: pd.Series,
    drawdown: pd.Series,
    returns: pd.Series,
    positions: pd.Series,
    benchmark_equity: Optional[pd.Series] = None,
    default_plots: Iterable[str] = DEFAULT_PLOTS,
    optional_plots: Iterable[str] = (),
    label: str = "strategy",
    benchmark_label: str = "benchmark",
) -> Dict[str, plt.Figure]:
    """build selected matplotlib figures and return a keyed dict."""
    plt.style.use("seaborn-v0_8-whitegrid")
    selected = set(_normalize_plot_name(p) for p in (set(default_plots) | set(optional_plots)))
    figs: Dict[str, plt.Figure] = {}

    if "equity_curve" in selected:
        fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
        equity_curve.plot(ax=axes[0], label=label, linewidth=1.9, color="#1f77b4")
        if benchmark_equity is not None:
            benchmark_equity.plot(
                ax=axes[0], label=benchmark_label, linewidth=1.5, alpha=0.9, color="#ff7f0e"
            )
        axes[0].set_title("Equity Curve", fontsize=12, pad=8)
        axes[0].legend(loc="best", frameon=True)
        axes[0].grid(alpha=0.25)
        drawdown.plot(ax=axes[1], color="#d62728", linewidth=1.4)
        axes[1].fill_between(drawdown.index, drawdown.values, 0.0, color="#d62728", alpha=0.15)
        axes[1].set_title("Drawdown", fontsize=11, pad=6)
        axes[1].grid(alpha=0.25)
        fig.tight_layout()
        figs["equity_curve"] = fig

    if "returns_distribution" in selected:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        returns.hist(ax=ax, bins=60, color="#4c78a8", alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.axvline(float(returns.mean()), color="#d62728", linestyle="--", linewidth=1.4, label="mean")
        ax.set_title("Return Distribution", fontsize=12, pad=8)
        ax.legend(loc="best", frameon=True)
        ax.grid(alpha=0.22)
        fig.tight_layout()
        figs["returns_distribution"] = fig

    if "rolling_sharpe" in selected:
        fig, ax = plt.subplots(figsize=(9, 4))
        roll = returns.rolling(63).mean() / returns.rolling(63).std(ddof=1)
        roll.plot(ax=ax, linewidth=1.6, color="#2ca02c")
        ax.set_title("Rolling Sharpe (63 bars)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        figs["rolling_sharpe"] = fig

    if "monthly_heatmap" in selected:
        fig, ax = plt.subplots(figsize=(9, 4))
        monthly = (1.0 + returns).resample("M").prod() - 1.0
        monthly.plot(kind="bar", ax=ax)
        ax.set_title("Monthly Returns")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        figs["monthly_heatmap"] = fig

    if "drawdown_timeline" in selected:
        fig, ax = plt.subplots(figsize=(9, 4))
        drawdown.plot(ax=ax, color="#d62728", linewidth=1.5)
        ax.set_title("Drawdown Timeline")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        figs["drawdown_timeline"] = fig

    if "positions" in selected:
        fig, ax = plt.subplots(figsize=(9, 3))
        positions.plot(ax=ax, linewidth=1.2, color="#9467bd")
        ax.set_title("Position Timeline")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        figs["positions"] = fig

    return figs


def iter_unique_plot_figures(plots: Mapping[str, plt.Figure]) -> Iterator[Tuple[str, plt.Figure]]:
    """yield plot figures once, even if multiple keys point to same figure."""
    seen: set[int] = set()
    for name, fig in plots.items():
        key = id(fig)
        if key in seen:
            continue
        seen.add(key)
        yield name, fig


def _normalize_plot_name(name: str) -> str:
    aliases = {
        "equity_drawdown": "equity_curve",
        "returns_hist": "returns_distribution",
    }
    return aliases.get(str(name), str(name))

