# Roostoo Quant Trading Hackathon - Autonomous Trading Agent

An autonomous quantitative trading bot for the **Roostoo Crypto Exchange** competition framework, complete with automated data downloading, factor research, backtesting, robustness diagnostics, and live execution scripts.

```
python scripts/preflight_competition.py             # verify keys/clock/venue (read-only, no orders)
python scripts/discover_universe.py                 # discover tradable pairs from the exchange
python scripts/download_data.py                     # fetch historical binance data cache
python scripts/run_backtest.py                      # run in-sample / research backtest
python scripts/run_full_research.py                 # run full research, walk-forward, and robustness suite
python scripts/run_live.py --once                   # start live trading / single-cycle check

```

> **Important Architecture Note:** API keys and credentials must **never** be hardcoded into the bot logic. They must be stored securely in your `.env` file (copied from `.env.example`).

## Submission materials & Code Layout

Everything submitted is in **Markdown**:

| Path / File | Contents | 
| ----- | ----- | 
| `README.md` | this document - portfolio, strategy, engine, backtest, fees, risk | 
| `scripts/run_live.py` | The main start button / entry point for running live trading or checks | 
| `src/roostoo_quant/live/` | Core bot logic, execution loops, and runtime management | 
| `src/roostoo_quant/roostoo/` | API communication layer that talks to Roostoo (REST client, HMAC signing) | 
| `config/universe.yaml` | asset universe configuration | 
| `config/default.yaml` | core strategy and research configuration | 

## 1. How the portfolio is curated

Five core steps power the portfolio management flow:

| \# | Decision | Where | Section | 
| ----- | ----- | ----- | ----- | 
| 1 | **Which coins may we hold?** Filtered via exchange exchangeInfo and volume criteria | `discover_universe.py`, `config/universe.yaml` | 6, 7 | 
| 2 | **When do we buy?** Multi-stage signal generation combining cross-sectional and time-series momentum | `roostoo_quant/research/candidates.py` | 2 | 
| 3 | **How much?** Equal-weight and risk-adjusted sizing respecting position caps and gross exposure | `roostoo_quant/backtest/engine.py` | 6 | 
| 4 | **When do we sell?** Dynamic rebalancing intervals with hysteresis rank buffers | `config/default.yaml` | 6 | 
| 5 | **What does it cost?** Explicit maker/taker transaction fee modeling and slippage basis points | `config/default.yaml`, `backtest` | 5 | 

The portfolio is structured to maintain robust diversification while capping total risk exposure, keeping cash buffers available to limit overall portfolio drawdowns.

## 2. Strategy: what it does, and how we arrived at it

### 2.1 The Signal Pipeline

The trading strategy evaluates multiple candidate signals:

* **Cross-Sectional Momentum (`cs_mom`)**: Ranks assets based on relative performance over historical windows.

* **Time-Series Momentum (`ts_mom`)**: Measures absolute trend strength per asset.

* **Breakout & Volatility Filters**: Integrates realized volatility rankings (`realized_volatility_rank_24h`) and cross-sectional rank scoring to filter out noisy regimes.

### 2.2 Regime Scaling & Execution

The strategy scales gross exposure dynamically depending on market volatility and trend conditions (bull, bear, high volatility, or sideways regimes). This ensures that capital allocation contracts during high-uncertainty periods.

## 3. The trading engine & API Communication

The execution engine is structured around clean separation of concerns:

* **Bot Logic (`src/roostoo_quant/live/`)**: Manages the core operational loops, state tracking, and decision-making logic.

* **Roostoo Interface (`src/roostoo_quant/roostoo/`)**: Handles secure communication and authenticated request signing with the Roostoo exchange endpoints.

* **Start Button (`scripts/run_live.py`)**: Acts as the primary entry point to launch single-cycle evaluations or continuous trading operations.

* **State Reconciliation**: Reconciles local positions against exchange balances to maintain synchronization.

## 4. Backtesting & Research Framework

```
python scripts/run_backtest.py
python scripts/run_full_research.py

```

* **Sample Splitting**: Walk-forward splits separate training, validation, and holdout datasets to avoid overfitting.

* **Look-Ahead Bias Prevention**: Signals calculated on bar close values are shifted appropriately before execution to prevent look-ahead bias.

* **Robustness & Stress Testing**: Includes block bootstrap return distributions, market gap shocks, and asset staleness injection scenarios.

## 5. Transaction costs: maker and taker

The backtest and live configurations explicitly account for trading friction:

* **Taker/Maker Fees**: Configurable fee structures applied directly on trade notionals.

* **Slippage Modeling**: Basis point slippage parameters simulate realistic market execution impact.

* **Cost-Aware Execution**: Rebalancing logic incorporates hysteresis and rank buffers (`hysteresis_rank_buffer`) to minimize unnecessary turnover and fee drag.

## 6. Risk management

Risk controls are built directly into the engine architecture:

* **Exposure Caps**: Strict limits on maximum gross exposure, individual pair weights, and open positions.

* **Turnover Control**: Rank buffers prevent asset thrashing during choppy sideways markets.

* **Stress Diagnostics**: Evaluates performance against common market shocks and liquidity dry-ups.

## 7. The data problem, and how it is solved

To overcome limitations in real-time public API snapshots:

* **Historical Archives**: Leverages cached Binance historical archives (`data/binance`) to supply deep OHLCV training panels.

* **Live Ticker Sampling**: Interacts with Roostoo's `/v3/exchangeInfo` and ticker endpoints via `src/roostoo_quant/roostoo/` to build up-to-date live features.

## 8. Operations & Credential Configuration

API keys **must** reside in your `.env` file and never be hardcoded into the codebase:

```
ROOSTOO_API_KEY=your_api_key_here
ROOSTOO_SECRET_KEY=your_secret_key_here
ROOSTOO_BASE_URL=https://mock-api.roostoo.com

```

To run operations:

```
# 1. Verify environment credentials and connection (read-only)
python scripts/preflight_competition.py

# 2. Run a single live loop iteration using the start button script
python scripts/run_live.py --once

# 3. Run live trading continuously
python scripts/run_live.py

```

Ensure host clocks are synchronized to prevent signature validation rejections.

## 9. Reproducing everything

```
# Discover universe pairs
python scripts/discover_universe.py

# Download cached data
python scripts/download_data.py

# Run backtest suite
python scripts/run_backtest.py

# Run full research diagnostics and walk-forward splits
python scripts/run_full_research.py

```

## 10. Limitations and disclosures

* **Mock Environment**: Designed specifically for the Roostoo mock exchange competition.

* **No Guarantee**: Past backtest performance does not guarantee live market results. Use caution when modifying default parameters.