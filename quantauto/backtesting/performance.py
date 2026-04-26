"""performance metrics for backtesting outputs."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _annual_factor(index: pd.DatetimeIndex, fallback: float = 252.0) -> float:
    if len(index) < 3:
        return fallback
    days = np.diff(index.view("i8")) / (24 * 3600 * 1e9)
    d = float(np.median(days)) if len(days) else 1.0
    if d <= 0:
        return fallback
    return 365.25 / d


def compute_performance_metrics(
    returns: pd.Series,
    equity: pd.Series,
    drawdown: pd.Series,
    *,
    turnover: Optional[pd.Series] = None,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate_annual: float = 0.0,
) -> Dict[str, float]:
    """compute broad return/risk/trade distribution metrics."""
    r = returns.dropna().astype(float)
    if r.empty:
        raise ValueError("returns are empty after dropna")
    ann = _annual_factor(r.index)
    mean = float(r.mean())
    std = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    downside = r[r < 0.0]
    down_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    total_return = float(equity.iloc[-1] - 1.0)
    ann_return = float((1.0 + total_return) ** (ann / len(r)) - 1.0)
    ann_vol = float(std * np.sqrt(ann))
    rf_period = risk_free_rate_annual / ann
    excess = r - rf_period
    sharpe = float(excess.mean() / std * np.sqrt(ann)) if std > 0 else float("nan")
    sortino = float(excess.mean() / down_std * np.sqrt(ann)) if down_std > 0 else float("nan")
    max_dd = float(drawdown.min())
    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else float("nan")
    wins = r[r > 0.0]
    losses = r[r < 0.0]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else float("inf")
    q95 = float(np.quantile(r, 0.05))
    cvar = float(r[r <= q95].mean()) if (r <= q95).any() else q95
    p95 = float(np.quantile(r, 0.95))
    tail_ratio = float(p95 / abs(q95)) if q95 < 0 else float("nan")
    out: Dict[str, float] = {
        "total_return": total_return,
        "annualized_return": ann_return,
        "mean_return": mean,
        "volatility": std,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
        "hit_rate": float((r > 0.0).mean()),
        "trade_count": float(int((r != 0.0).sum())),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "skewness": float(r.skew()),
        "kurtosis": float(r.kurtosis()),
        "var_95": q95,
        "cvar_95": cvar,
        "tail_ratio": tail_ratio,
    }
    if turnover is not None:
        out["avg_turnover"] = float(turnover.mean())
    if benchmark_returns is not None:
        b = benchmark_returns.reindex(r.index).dropna()
        both = pd.concat([r.rename("s"), b.rename("b")], axis=1).dropna()
        if len(both) > 1:
            cov = np.cov(both["s"], both["b"], ddof=1)
            beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else float("nan")
            alpha = float((both["s"] - beta * both["b"]).mean() * ann) if np.isfinite(beta) else float("nan")
            tracking = float((both["s"] - both["b"]).std(ddof=1))
            info = float((both["s"] - both["b"]).mean() / tracking * np.sqrt(ann)) if tracking > 0 else float("nan")
            out["beta"] = beta
            out["alpha"] = alpha
            out["information_ratio"] = info
            out["excess_return"] = float((1.0 + both["s"]).prod() - (1.0 + both["b"]).prod())
    return out

