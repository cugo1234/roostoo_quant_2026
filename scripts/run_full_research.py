#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from roostoo_quant.config import load_yaml
from roostoo_quant.data.binance import load_cached_symbol, panel_from_frames
from roostoo_quant.research.candidates import candidate_scores
from roostoo_quant.research.evaluate import candidate_config, evaluate_score
from roostoo_quant.backtest.engine import run_backtest
from roostoo_quant.backtest.tournament import rolling_tournament_windows, tournament_summary
from roostoo_quant.backtest.robustness import walk_forward_splits, slice_ohlcv, anchored_walk_forward_windows, block_bootstrap_return_distribution
from roostoo_quant.signals.utils import cross_sectional_rank_score
from roostoo_quant.signals.volatility import realized_vol
from roostoo_quant.backtest.stress import inject_market_shock, inject_stale_asset


def load_panels(cfg, universe_path="config/universe.yaml", cache="data/binance"):
    u = yaml.safe_load(Path(universe_path).read_text())
    frames = {}
    for sym in u["symbols"]:
        try:
            frames[sym] = load_cached_symbol(cache, sym, cfg["research"]["interval"], cfg["research"]["start"], cfg["research"]["end"])
        except FileNotFoundError:
            continue
    if len(frames) < 2:
        raise SystemExit("Need >=2 cached symbols. Run scripts/download_data.py first.")
    panels = {f: panel_from_frames(frames, f) for f in ["open","high","low","close","volume"]}
    keep = panels["close"].notna().sum(axis=1) >= 2
    return {k:v.loc[keep] for k,v in panels.items()}


def candidate_score_map(ohlcv, cfg):
    m = candidate_scores(ohlcv, cfg)
    # Literature-motivated higher-moment candidate: cross-sectional realized
    # volatility rank over one day (6 x 4h bars).
    m["realized_volatility_rank_24h"] = cross_sectional_rank_score(realized_vol(ohlcv["close"], 6))
    return m


def evaluate_period(ohlcv, cfg, label, out_dir):
    rows=[]
    scores=candidate_score_map(ohlcv,cfg)
    for name,score in scores.items():
        cc=candidate_config(cfg,name)
        res=evaluate_score(ohlcv,score,cc)
        row={"period":label,"candidate":name,**res.metrics}
        rows.append(row)
        cdir=out_dir/label/name; cdir.mkdir(parents=True,exist_ok=True)
        res.returns.to_csv(cdir/"detail.csv")
        res.trades.to_csv(cdir/"trades.csv",index=False)
        tw=rolling_tournament_windows(res.returns.iloc[:-1],6,14,cc["research"]["annual_periods"])
        tw.to_csv(cdir/"tournament_windows.csv")
        (cdir/"tournament_summary.json").write_text(json.dumps(tournament_summary(tw),indent=2,default=str))
    return pd.DataFrame(rows)


def default_sensitivity(trainval, cfg, out_dir):
    variants=[]
    # Broad, pre-declared plateaus around the default; final holdout is not used.
    for rb in [3,6,9]:
        c=deepcopy(cfg); c["strategy"]["rebalance_every_bars"]=rb
        r=run_backtest(trainval,c)
        variants.append({"variation":f"rebalance_{rb}bars",**r.metrics})
    for fee in [.0005,.001,.0015,.002]:
        c=deepcopy(cfg); c["competition"]["taker_fee"]=fee
        r=run_backtest(trainval,c)
        variants.append({"variation":f"taker_fee_{fee:.4f}",**r.metrics})
    for slip in [0,2,5,10]:
        c=deepcopy(cfg); c["research"]["slippage_bps"]=slip
        r=run_backtest(trainval,c)
        variants.append({"variation":f"slippage_{slip}bps",**r.metrics})
    for buffer in [0,1,2,3]:
        c=deepcopy(cfg); c["strategy"]["hysteresis_rank_buffer"]=buffer
        r=run_backtest(trainval,c)
        variants.append({"variation":f"rank_buffer_{buffer}",**r.metrics})
    df=pd.DataFrame(variants); df.to_csv(out_dir/"default_sensitivity_train_validation.csv",index=False)
    return df


def ablations(holdout, cfg, out_dir):
    variants=[]
    cases={
        "full_default": {},
        "no_regime_scaling": {"strategy.normal_gross":1.0,"strategy.stressed_gross":1.0,"strategy.severe_gross":1.0},
        "no_hysteresis": {"strategy.hysteresis_rank_buffer":0},
        "no_deadband": {"strategy.min_trade_weight":0.0},
        "no_breakout": {"strategy.score_weights":{"cs_mom":0.67,"ts_mom":0.33,"breakout":0.0}},
        "no_ts_momentum": {"strategy.score_weights":{"cs_mom":0.67,"ts_mom":0.0,"breakout":0.33}},
    }
    for name,mods in cases.items():
        c=deepcopy(cfg)
        for k,v in mods.items():
            sec,field=k.split(".",1)
            c[sec][field]=v
        r=run_backtest(holdout,c)
        variants.append({"ablation":name,**r.metrics})
    df=pd.DataFrame(variants); df.to_csv(out_dir/"holdout_ablations.csv",index=False)
    return df


def regime_report(holdout, cfg, detail, out_dir):
    btc=holdout["close"]["BTCUSDT"]
    mom=btc/btc.shift(18)-1
    vol=btc.pct_change().rolling(42).std()
    med=vol.median()
    labels=pd.Series("sideways",index=btc.index)
    labels[mom>0.03]="bull"
    labels[mom<-0.03]="bear"
    labels[vol>med*1.5]="high_vol"
    rows=[]
    for label in labels.dropna().unique():
        idx=labels[labels==label].index.intersection(detail.index)
        r=detail.loc[idx,"net_return"]
        if len(r):
            rows.append({"regime":label,"bars":len(r),"mean_bar_return":r.mean(),"cumulative_return":(1+r).prod()-1,"win_rate":(r>0).mean()})
    df=pd.DataFrame(rows); df.to_csv(out_dir/"holdout_regimes.csv",index=False)
    return df



def walk_forward_default(ohlcv, cfg, out_dir):
    idx=ohlcv["close"].index
    # Roughly 2 years initial training context, then non-overlapping 90-day OOS folds.
    initial=min(max(2190*2, 500), max(500, len(idx)//2))
    test_bars=6*90
    rows=[]
    for fold,(train_idx,test_idx) in enumerate(anchored_walk_forward_windows(idx,initial,test_bars,test_bars),1):
        # Include 200 warmup bars immediately before test; metrics are measured only on test.
        start_pos=max(0,idx.get_loc(test_idx[0])-220)
        combo_idx=idx[start_pos:idx.get_loc(test_idx[-1])+1]
        combo=slice_ohlcv(ohlcv,combo_idx)
        r=run_backtest(combo,cfg)
        detail=r.returns.loc[test_idx.intersection(r.returns.index)]
        from roostoo_quant.backtest.metrics import performance_metrics
        m=performance_metrics(detail["net_return"],detail["turnover"],detail["gross_exposure"],detail["trading_cost"],cfg["research"]["annual_periods"])
        rows.append({"fold":fold,"train_end":str(train_idx[-1]),"test_start":str(test_idx[0]),"test_end":str(test_idx[-1]),**m})
    df=pd.DataFrame(rows); df.to_csv(out_dir/"walk_forward_default.csv",index=False)
    return df


def extra_robustness(trainval, test, cfg, out_dir):
    rows=[]
    # Delayed execution sensitivity on train+validation only.
    for delay in [1,2,3]:
        c=deepcopy(cfg); c["research"]["signal_delay_bars"]=delay
        r=run_backtest(trainval,c)
        rows.append({"test":"execution_delay","value":delay,**r.metrics})
    # Universe sensitivity. The ordering uses the predeclared proxy universe, not performance.
    cols=list(trainval["close"].columns)
    for n in [2,5,len(cols)]:
        subset=cols[:min(n,len(cols))]
        if "BTCUSDT" not in subset and "BTCUSDT" in cols:
            subset=["BTCUSDT"]+subset[:-1]
        d={k:v[subset] for k,v in trainval.items()}
        if len(subset)>=2:
            r=run_backtest(d,cfg)
            rows.append({"test":"universe_size","value":len(subset),**r.metrics})
    pd.DataFrame(rows).to_csv(out_dir/"extra_robustness_train_validation.csv",index=False)

    # Stress the untouched test using fixed default parameters; these are scenario
    # diagnostics, not parameter tuning.
    stress_rows=[]
    if len(test["close"])>100:
        mid=test["close"].index[len(test["close"])//2]
        for shock in [-.10,-.20,-.35]:
            r=run_backtest(inject_market_shock(test,mid,shock),cfg)
            stress_rows.append({"scenario":f"common_market_gap_{shock:.0%}",**r.metrics})
        sym=test["close"].columns[-1]
        r=run_backtest(inject_stale_asset(test,sym,mid,6),cfg)
        stress_rows.append({"scenario":f"stale_{sym}_6bars",**r.metrics})
    pd.DataFrame(stress_rows).to_csv(out_dir/"holdout_stress_scenarios.csv",index=False)
    return rows,stress_rows

def main():
    cfg=load_yaml("config/default.yaml")
    ohlcv=load_panels(cfg)
    out=Path("results/full_research"); out.mkdir(parents=True,exist_ok=True)
    tr_idx,va_idx,te_idx=walk_forward_splits(ohlcv["close"].index,.60,.20)
    train=slice_ohlcv(ohlcv,tr_idx); val=slice_ohlcv(ohlcv,va_idx); test=slice_ohlcv(ohlcv,te_idx)
    trainval=slice_ohlcv(ohlcv,tr_idx.append(va_idx))
    all_rows=pd.concat([evaluate_period(train,cfg,"train",out),evaluate_period(val,cfg,"validation",out)],ignore_index=True)
    all_rows.to_csv(out/"candidate_train_validation.csv",index=False)
    sensitivity=default_sensitivity(trainval,cfg,out)
    walk_forward_default(trainval,cfg,out)

    # Parameters are frozen at config/default.yaml before touching the holdout.
    final=run_backtest(test,cfg)
    final.returns.to_csv(out/"FINAL_HOLDOUT_detail.csv")
    final.positions.to_csv(out/"FINAL_HOLDOUT_positions.csv")
    final.trades.to_csv(out/"FINAL_HOLDOUT_trades.csv",index=False)
    (out/"FINAL_HOLDOUT_metrics.json").write_text(json.dumps(final.metrics,indent=2,default=str))
    tw=rolling_tournament_windows(final.returns.iloc[:-1],6,14,cfg["research"]["annual_periods"])
    tw.to_csv(out/"FINAL_HOLDOUT_tournament_windows.csv")
    (out/"FINAL_HOLDOUT_tournament_summary.json").write_text(json.dumps(tournament_summary(tw),indent=2,default=str))
    ablations(test,cfg,out)
    regime_report(test,cfg,final.returns,out)
    extra_robustness(trainval,test,cfg,out)
    boot=block_bootstrap_return_distribution(final.returns["net_return"].iloc[:-1], horizon_bars=6*14, block_bars=6, simulations=1000, seed=7)
    boot.to_csv(out/"FINAL_HOLDOUT_block_bootstrap_14d.csv",index=False)
    if not boot.empty:
        (out/"FINAL_HOLDOUT_bootstrap_summary.json").write_text(json.dumps({
            "mean_return":float(boot.total_return.mean()),"median_return":float(boot.total_return.median()),
            "p05_return":float(boot.total_return.quantile(.05)),"p95_return":float(boot.total_return.quantile(.95)),
            "prob_positive":float((boot.total_return>0).mean()),"median_max_drawdown":float(boot.max_drawdown.median())
        },indent=2))
    manifest={
        "data_start":str(ohlcv["close"].index.min()),"data_end":str(ohlcv["close"].index.max()),
        "symbols":list(ohlcv["close"].columns),"bars":len(ohlcv["close"]),
        "train_bars":len(tr_idx),"validation_bars":len(va_idx),"holdout_bars":len(te_idx),
        "holdout_run_policy":"default parameters frozen before holdout; do not iterate after reading holdout without creating a new future holdout",
    }
    (out/"research_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps({"manifest":manifest,"final_holdout":final.metrics,"tournament":tournament_summary(tw)},indent=2,default=str))

if __name__=="__main__":
    main()
