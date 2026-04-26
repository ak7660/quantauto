"""portfolio path construction from positions and realized returns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantauto.backtesting.numba_utils import compute_paths


@dataclass
class PortfolioPath:
    turnover: pd.Series
    returns_gross: pd.Series
    returns_net: pd.Series
    equity_curve: pd.Series
    drawdown: pd.Series
    positions: pd.Series


def build_portfolio_path(
    positions: pd.Series,
    realized_returns: pd.Series,
    *,
    trade_cost_rate: float,
    use_numba: bool = True,
) -> PortfolioPath:
    """construct gross/net return and equity/drawdown series."""
    idx = positions.index
    pos_np = positions.astype(float).to_numpy()
    ret_np = realized_returns.astype(float).to_numpy()
    turnover_np, net_np, eq_np, dd_np = compute_paths(
        pos_np, ret_np, trade_cost_rate, use_numba=use_numba
    )
    gross_np = pos_np * ret_np
    return PortfolioPath(
        turnover=pd.Series(turnover_np, index=idx, name="turnover"),
        returns_gross=pd.Series(gross_np, index=idx, name="gross_return"),
        returns_net=pd.Series(net_np, index=idx, name="strategy_return"),
        equity_curve=pd.Series(eq_np, index=idx, name="equity"),
        drawdown=pd.Series(dd_np, index=idx, name="drawdown"),
        positions=positions.rename("position"),
    )

