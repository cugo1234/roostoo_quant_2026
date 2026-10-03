from __future__ import annotations

import pandas as pd


def drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def risk_multiplier(
    current_drawdown: float,
    daily_return: float,
    daily_loss_cut: float = 0.05,
    portfolio_drawdown_cut: float = 0.08,
) -> float:
    """Simple fail-safe overlay, not an alpha source.

    A severe portfolio drawdown moves the book mostly to cash; a large intraday
    loss cuts exposure in half. The thresholds are deliberately coarse and
    should be stress-tested rather than optimized to a single backtest.
    """
    if current_drawdown <= -abs(portfolio_drawdown_cut):
        return 0.20
    if daily_return <= -abs(daily_loss_cut):
        return 0.50
    return 1.0
