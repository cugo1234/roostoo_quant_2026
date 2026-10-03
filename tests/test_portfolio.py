import pandas as pd
from roostoo_quant.portfolio.construct import target_weights_from_score, apply_deadband


def test_target_gross_is_bounded():
    score = pd.Series({"A": 1.0, "B": 0.7, "C": -0.6, "D": -1.0})
    vol = pd.Series({"A": .02, "B": .03, "C": .04, "D": .02})
    w = target_weights_from_score(score, vol, gross=1.0, long_budget_fraction=.55, top_k=2, bottom_k=2, max_asset_weight=.30)
    assert w.abs().sum() <= 1.0000001
    assert w.sum() <= .30
    assert (w > 0).any() and (w < 0).any()


def test_deadband_keeps_small_changes():
    prev = pd.Series({"A": .20, "B": -.20})
    target = pd.Series({"A": .21, "B": -.40})
    out = apply_deadband(target, prev, min_change=.03)
    assert out["A"] == prev["A"]
    assert out["B"] == target["B"]
