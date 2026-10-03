from __future__ import annotations

import numpy as np
import pandas as pd

from roostoo_quant.signals.volatility import inverse_vol_weights


def long_budget_from_regime(btc_mom: float, breadth: float, min_long: float = 0.35, max_long: float = 0.65) -> float:
    """Map broad crypto regime into a modest long/short budget tilt.

    The portfolio stays approximately market balanced; this is deliberately
    bounded because the contest is short and raw direction can dominate risk.
    """
    if pd.isna(btc_mom) or pd.isna(breadth):
        return 0.50
    # Both inputs are continuous. btc_mom is squashed so a single outlier does
    # not force a full directional bet.
    btc_component = float(np.tanh(8.0 * btc_mom))
    regime = 0.55 * btc_component + 0.45 * float(np.clip(breadth, -1.0, 1.0))
    return float(np.clip(0.50 + 0.15 * regime, min_long, max_long))


def gross_from_stress(stress: float, normal: float = 1.0, stressed: float = 0.60, severe: float = 0.35) -> float:
    if pd.isna(stress):
        return stressed
    if stress >= 0.80:
        return severe
    if stress >= 0.50:
        return stressed
    return normal


def _allocate_side(symbols: list[str], vol_row: pd.Series, budget: float, cap: float) -> pd.Series:
    if not symbols or budget <= 0:
        return pd.Series(dtype=float)
    v = vol_row.reindex(symbols)
    w = inverse_vol_weights(v, cap=1.0)  # cap applied after side budget scaling
    if w.empty:
        w = pd.Series(1.0 / len(symbols), index=symbols)
    w = w * budget
    # Enforce global absolute cap and redistribute within the side.
    for _ in range(10):
        over = w > cap
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w.loc[over] = cap
        under = ~over
        if under.any() and w[under].sum() > 0:
            w.loc[under] += excess * w[under] / w[under].sum()
    return w


def target_weights_from_score(
    score_row: pd.Series,
    vol_row: pd.Series,
    gross: float,
    long_budget_fraction: float,
    top_k: int = 2,
    bottom_k: int = 2,
    max_asset_weight: float = 0.25,
    previous: pd.Series | None = None,
    rank_buffer: int = 0,
) -> pd.Series:
    """Construct a long/short target from a cross-sectional score.

    Gross exposure is <= 1.0 by design. Long and short are both funded from the
    same 1x gross budget to remain compatible with the competition's no-leverage
    requirement.
    """
    valid = score_row.dropna().sort_values()
    if len(valid) < 2:
        return pd.Series(0.0, index=score_row.index)

    k_long = min(top_k, max(1, len(valid) // 2))
    k_short = min(bottom_k, max(1, len(valid) // 2))
    long_rank_pool = list(valid.index[-min(len(valid), k_long + max(0, rank_buffer)):])
    short_rank_pool = list(valid.index[:min(len(valid), k_short + max(0, rank_buffer))])

    prev = previous.reindex(valid.index).fillna(0.0) if previous is not None else pd.Series(0.0, index=valid.index)
    # Keep incumbents while their rank remains inside a wider buffer, then fill
    # remaining slots by current score. This is a standard churn-control device.
    longs = [x for x in prev[prev > 0].index if x in long_rank_pool]
    for x in reversed(list(valid.index)):
        if x not in longs:
            longs.append(x)
        if len(longs) >= k_long:
            break
    shorts = [x for x in prev[prev < 0].index if x in short_rank_pool and x not in longs]
    for x in list(valid.index):
        if x not in shorts and x not in longs:
            shorts.append(x)
        if len(shorts) >= k_short:
            break

    long_budget = gross * float(np.clip(long_budget_fraction, 0.0, 1.0))
    short_budget = gross - long_budget
    lw = _allocate_side(longs, vol_row, long_budget, max_asset_weight)
    sw = _allocate_side(shorts, vol_row, short_budget, max_asset_weight)

    out = pd.Series(0.0, index=score_row.index, dtype=float)
    out.loc[lw.index] = lw.values
    out.loc[sw.index] = -sw.values

    # Numerical guardrail: gross may become slightly > requested due to
    # redistribution/capping edge cases.
    actual_gross = float(out.abs().sum())
    if actual_gross > gross + 1e-12 and actual_gross > 0:
        out *= gross / actual_gross
    return out


def apply_deadband(target: pd.Series, previous: pd.Series, min_change: float = 0.03) -> pd.Series:
    """Keep prior weights when the desired change is too small to justify fees."""
    prev = previous.reindex(target.index).fillna(0.0)
    out = target.copy()
    small = (out - prev).abs() < min_change
    out.loc[small] = prev.loc[small]
    # If deadband pushed gross above one, scale back proportionally.
    gross = float(out.abs().sum())
    if gross > 1.0:
        out /= gross
    return out
