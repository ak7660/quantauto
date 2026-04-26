"""numba-accelerated numeric kernels for backtesting."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit

    HAS_NUMBA = True
except Exception:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def _wrap(fn):
            return fn

        return _wrap


@njit(cache=True)
def _compute_paths_numba(
    positions: np.ndarray,
    realized_returns: np.ndarray,
    trade_cost_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(positions)
    turnover = np.zeros(n, dtype=np.float64)
    net_returns = np.zeros(n, dtype=np.float64)
    equity = np.zeros(n, dtype=np.float64)
    drawdown = np.zeros(n, dtype=np.float64)
    eq = 1.0
    peak = 1.0
    prev_pos = positions[0]
    for i in range(n):
        pos = positions[i]
        if i == 0:
            turnover[i] = 0.0
        else:
            turnover[i] = abs(pos - prev_pos)
        gross = pos * realized_returns[i]
        cost = turnover[i] * trade_cost_rate
        net = gross - cost
        net_returns[i] = net
        eq = eq * (1.0 + net)
        equity[i] = eq
        if eq > peak:
            peak = eq
        drawdown[i] = eq / peak - 1.0
        prev_pos = pos
    return turnover, net_returns, equity, drawdown


def _compute_paths_numpy(
    positions: np.ndarray,
    realized_returns: np.ndarray,
    trade_cost_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    turnover = np.abs(np.diff(positions, prepend=positions[:1]))
    gross = positions * realized_returns
    net = gross - turnover * trade_cost_rate
    equity = np.cumprod(1.0 + net)
    peaks = np.maximum.accumulate(equity)
    drawdown = equity / peaks - 1.0
    return turnover, net, equity, drawdown


def compute_paths(
    positions: np.ndarray,
    realized_returns: np.ndarray,
    trade_cost_rate: float,
    *,
    use_numba: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """compute turnover, net returns, equity, and drawdown arrays."""
    if use_numba and HAS_NUMBA:
        return _compute_paths_numba(positions, realized_returns, float(trade_cost_rate))
    return _compute_paths_numpy(positions, realized_returns, float(trade_cost_rate))

