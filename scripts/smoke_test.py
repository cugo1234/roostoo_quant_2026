#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd

from roostoo_quant.config import load_yaml
from roostoo_quant.backtest.engine import run_backtest
from roostoo_quant.backtest.robustness import sensitivity_grid
from roostoo_quant.backtest.tournament import rolling_tournament_windows, tournament_summary


def synthetic_ohlcv(n=600):
    rng = np.random.default_rng(20261003)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    syms = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","TRXUSDT","DOGEUSDT","ADAUSDT"]
    # Deterministic regime-switching synthetic market: useful for software tests,
    # absolutely not empirical evidence about the real competition.
    regimes = np.where((np.arange(n)//180)%3==0, .0010, np.where((np.arange(n)//180)%3==1, -.0007, .00005))
    common = regimes + rng.normal(0, .010, n)
    close = {}
    for j,s in enumerate(syms):
        rel = (j - (len(syms)-1)/2) * 0.00004
        slow = 0.0007*np.sin(np.arange(n)/(28+j*2)+j)
        r = (0.65 + 0.03*j)*common + rel + slow + rng.normal(0,.007+.0006*j,n)
        close[s] = 100*np.exp(np.cumsum(r))
    close = pd.DataFrame(close,index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    u = pd.DataFrame(rng.uniform(.001,.012,close.shape),index=idx,columns=close.columns)
    return {
        "open": open_,
        "high": np.maximum(open_,close)*(1+u),
        "low": np.minimum(open_,close)*(1-u),
        "close": close,
        "volume": pd.DataFrame(rng.lognormal(10,.7,close.shape),index=idx,columns=close.columns),
    }


def main():
    cfg=load_yaml("config/default.yaml")
    data=synthetic_ohlcv()
    out=Path("results/synthetic_smoke"); out.mkdir(parents=True,exist_ok=True)
    base=run_backtest(data,cfg)
    base.returns.to_csv(out/"detail.csv")
    base.positions.to_csv(out/"positions.csv")
    base.trades.to_csv(out/"trades.csv",index=False)
    (out/"metrics.json").write_text(json.dumps(base.metrics,indent=2,default=str))
    tw=rolling_tournament_windows(base.returns.iloc[:-1],bars_per_day=6,days=14,annual_periods=cfg["research"]["annual_periods"])
    tw.to_csv(out/"tournament_windows.csv")
    ts=tournament_summary(tw)
    (out/"tournament_summary.json").write_text(json.dumps(ts,indent=2,default=str))
    grid=[]
    for rb, fee in [(1,.001),(2,.001),(3,.001),(2,.002)]:
        grid.append({"strategy.rebalance_every_bars":rb,"competition.taker_fee":fee})
    sens=sensitivity_grid(data,cfg,grid)
    sens.to_csv(out/"sensitivity.csv",index=False)
    status={
        "purpose":"software/invariant smoke test only; synthetic results are not strategy evidence",
        "base_metrics":base.metrics,
        "tournament_summary":ts,
        "invariants":{
            "max_gross_exposure":float(base.positions.abs().sum(axis=1).max()),
            "lookahead_execution":"signals at close t; target shifted one bar before open-to-open PnL",
            "fees_charged":True,
        }
    }
    (out/"STATUS.json").write_text(json.dumps(status,indent=2,default=str))
    print(json.dumps(status,indent=2,default=str))

if __name__=="__main__":
    main()
