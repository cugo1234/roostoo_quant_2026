from __future__ import annotations

import math
import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    return float((equity / equity.cummax() - 1.0).min())


def performance_metrics(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    exposure: pd.Series | None = None,
    transaction_costs: pd.Series | None = None,
    annual_periods: int = 2190,
    trades: int | None = None,
) -> dict[str, float]:
    r = returns.dropna().astype(float)
    if r.empty:
        return {}
    equity = (1.0 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    n = len(r)
    ann_return = float(equity.iloc[-1] ** (annual_periods / max(n, 1)) - 1.0) if equity.iloc[-1] > 0 else -1.0
    vol = float(r.std(ddof=0))
    ann_vol = vol * math.sqrt(annual_periods)
    sharpe = float(r.mean() / vol * math.sqrt(annual_periods)) if vol > 1e-15 else float("nan")
    downside = r[r < 0]
    downside_dev = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
    sortino = float(r.mean() / downside_dev * math.sqrt(annual_periods)) if downside_dev > 1e-15 else float("nan")
    mdd = max_drawdown(equity)
    calmar = float(ann_return / abs(mdd)) if mdd < -1e-12 else float("nan")
    active = r[r != 0]
    win_rate = float((active > 0).mean()) if len(active) else float("nan")
    out = {
        "periods": float(n),
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "win_rate_active_bars": win_rate,
        "mean_bar_return": float(r.mean()),
        "competition_score": float(0.4 * sortino + 0.3 * sharpe + 0.3 * calmar)
        if np.isfinite(sortino) and np.isfinite(sharpe) and np.isfinite(calmar)
        else float("nan"),
    }
    if turnover is not None:
        t = turnover.reindex(r.index).fillna(0.0)
        out["total_turnover"] = float(t.sum())
        out["avg_turnover_per_bar"] = float(t.mean())
    if exposure is not None:
        e = exposure.reindex(r.index).fillna(0.0)
        out["avg_gross_exposure"] = float(e.mean())
        out["max_gross_exposure"] = float(e.max())
    if transaction_costs is not None:
        c = transaction_costs.reindex(r.index).fillna(0.0)
        out["transaction_cost_drag"] = float(c.sum())
    if trades is not None:
        out["order_count"] = float(trades)
        out["return_per_order"] = float(total_return / trades) if trades else float("nan")
    return out
