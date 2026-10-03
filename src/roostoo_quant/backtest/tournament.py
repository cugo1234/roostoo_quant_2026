from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import performance_metrics


def rolling_tournament_windows(detail: pd.DataFrame, bars_per_day: int, days: int = 14, annual_periods: int = 2190) -> pd.DataFrame:
    bars = int(bars_per_day * days)
    rows = []
    if len(detail) < bars:
        return pd.DataFrame()
    for end in range(bars, len(detail) + 1):
        win = detail.iloc[end-bars:end]
        m = performance_metrics(
            win["net_return"],
            turnover=win.get("turnover"),
            exposure=win.get("gross_exposure"),
            transaction_costs=win.get("trading_cost"),
            annual_periods=annual_periods,
        )
        m["start"] = win.index[0]
        m["end"] = win.index[-1]
        rows.append(m)
    return pd.DataFrame(rows).set_index("start") if rows else pd.DataFrame()


def tournament_summary(windows: pd.DataFrame) -> dict[str, float]:
    if windows.empty:
        return {}
    r = windows["total_return"]
    dd = windows["max_drawdown"]
    score = windows["competition_score"].replace([np.inf, -np.inf], np.nan)
    return {
        "windows": float(len(windows)),
        "mean_14d_return": float(r.mean()),
        "median_14d_return": float(r.median()),
        "p05_return": float(r.quantile(0.05)),
        "p25_return": float(r.quantile(0.25)),
        "p75_return": float(r.quantile(0.75)),
        "p95_return": float(r.quantile(0.95)),
        "prob_positive_return": float((r > 0).mean()),
        "prob_loss_gt_5pct": float((r < -0.05).mean()),
        "worst_14d_return": float(r.min()),
        "best_14d_return": float(r.max()),
        "median_max_drawdown": float(dd.median()),
        "median_composite_score": float(score.median()) if score.notna().any() else float("nan"),
    }
