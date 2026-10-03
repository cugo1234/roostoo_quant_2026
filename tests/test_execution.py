import hashlib, hmac
from roostoo_quant.roostoo.client import RoostooClient
from roostoo_quant.execution.planner import plan_rebalance
import pandas as pd


def test_signature_matches_returned_query():
    c = RoostooClient("key", "secret")
    headers, payload, query = c._signed({"pair": "BTC/USD", "side": "BUY", "quantity": 1, "type": "MARKET"})
    expected = hmac.new(b"secret", query.encode(), hashlib.sha256).hexdigest()
    assert headers["MSG-SIGNATURE"] == expected
    assert "timestamp=" in query


def test_rebalance_planner_close_first():
    target = pd.Series({"BTC/USD": -.20})
    current = pd.Series({"BTC/USD": .20})
    tickers = {"Data": {"BTC/USD": {"LastPrice": 100.0}}}
    rules = {"BTC/USD": {"AmountPrecision": 4, "MiniOrder": 1.0, "CanTrade": True}}
    orders = plan_rebalance(target, current, 10000, tickers, rules, min_weight_change=.01)
    assert orders[0].action == "SELL"
    assert any(o.action == "SHORT_OPEN" for o in orders)
