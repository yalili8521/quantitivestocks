# Quantitative Stocks

An ML-driven quantitative trading system for ETFs. Combines traditional technical indicators with LSTM neural networks to generate buy/sell signals, backtest strategies, and execute paper trades via Alpaca Markets. Features options spread trading with put/call spreads.

## 🏆 Performance Highlights

Recent backtest results (Jan 2024 - Feb 2026):
- **Average Total Return**: +68.7%
- **Best Performer**: SLV +210.1% (3.36 Sharpe)
- **Win Rate**: 66% across 94 trades
- **Profit Factor**: 2.44

## Project Structure

```
quantitivestocks/
├── main.py                     # Unified CLI entry point
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── streamlit_app.py            # Streamlit dashboard
│
├── models/                     # Trained ML models & scalers
│   ├── SPY_lstm.pt            # PyTorch model weights
│   ├── SPY_scaler.json        # Feature normalization
│   ├── GLD_lstm.pt
│   └── ...
│
├── data/                       # Raw data (empty - fetched on demand)
│
├── outputs/                    # All results & analysis tools
│   ├── backtest_SPY.csv       # Equity curves
│   ├── trades_SPY.csv         # Trade histories
│   ├── backtest_SPY_chart.html # Interactive charts
│   ├── dashboard.py           # Streamlit dashboard
│   ├── analyze_results.py     # Command-line analysis
│   └── signals.json           # Signal generation output
│
├── settings/                   # Configuration & secrets
│   ├── settings.py            # System configuration
│   └── alpaca.env             # API credentials
│
└── Core Python Files (root):
    ├── signals_engine.py       # Technical indicators & signals
    ├── ml_model.py             # LSTM training & prediction
    ├── backtester.py           # Walk-forward backtesting
    ├── paper_trader.py         # Stock paper trading
    └── options_trader.py       # Options spread trading
```

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt
```

Dependencies include: pandas, numpy, torch, alpaca-trade-api, yfinance, streamlit, plotly

### 2. Set Environment Variables

```bash
export FRED_API_KEY="your_fred_api_key"          # Free at https://fred.stlouisfed.org/docs/api/api_key.html
export ALPACA_API_KEY="your_alpaca_api_key"       # Free at https://app.alpaca.markets/signup
export ALPACA_API_SECRET="your_alpaca_api_secret"
```

On Windows (PowerShell):
```powershell
$env:FRED_API_KEY = "your_fred_api_key"
$env:ALPACA_API_KEY = "your_alpaca_api_key"
$env:ALPACA_API_SECRET = "your_alpaca_api_secret"
```

FRED key is used for VIX data. Alpaca keys are only required for the `alpaca`/`hybrid` data providers and for paper trading. The `yahoo` provider works without any keys.

### 3. Run Signals

```bash
python main.py signals --provider yahoo
```

This outputs a table of sentiment signals for the default universe (SPY, QQQ, IWM, IGV, SLV) and saves results to `data/output/signals.json`.

### 4. Train a Model

```bash
python main.py train --symbol SPY --epochs 50
```

Fetches ~1000 days of history, builds features, and trains the LSTM. Saves weights to `data/models/SPY_lstm.pt` and scaler to `data/models/SPY_scaler.json`.

### 5. Run a Prediction

```bash
python main.py predict --symbol SPY
```

Loads the trained model and outputs the predicted direction and confidence.

### 6. Backtest

```bash
python main.py backtest --symbol SPY --start 2024-01-01
```

Walk-forward backtest using ML predictions. Outputs a performance report with trade history and saves equity curves to `data/output/`.

### View Results in a Browser (Streamlit Dashboard)

Instead of opening CSVs in Excel, you can use the included Streamlit dashboard to explore equity curves and trades interactively:

```bash
python -m streamlit run streamlit_app.py
```

Then open the local link it prints (typically `http://localhost:8501`). If port `8501` is busy, Streamlit will use another port (e.g., `http://localhost:8502`).

Main features:
- Select a symbol (auto-discovers `backtest_*.csv` in `data/output/`)
- Equity curve + drawdown charts
- Quick metrics: initial/final, total return, CAGR (approx), max drawdown
- Trade blotter table (from `trades_<SYMBOL>.csv` when present)
- Optional per-trade chart with entry/exit markers (loads price history via `yfinance`)

The dashboard reads your existing outputs in `data/output/` and runs locally on your machine (no email sending / no uploads).

### 7. Paper Trade

```bash
python main.py trade --interval 5 --confidence 0.2
```

Starts a continuous paper trading loop on Alpaca. Checks ML signals every N minutes during market hours and executes trades automatically. `paper=True` is hardcoded for safety.

If you run paper trading via the Task Scheduler wrapper script (`scripts/run_paper_trade.ps1`), logs are written to:

`C:\ProgramData\QuantitativeStocks\logs`

## CLI Reference

| Command | Description |
|---------|-------------|
| `signals` | Run the sentiment signal engine |
| `train` | Train LSTM model for a symbol |
| `predict` | Get ML prediction for a symbol |
| `backtest` | Walk-forward backtest with ML signals |
| `trade` | Start Alpaca paper trading loop |

Run `python main.py <command> --help` for all options.

### Common Options

```
--provider {yahoo,alpaca,hybrid}    Data source (default: yahoo)
--symbols SPY,QQQ                   Comma-separated symbol list
--ml                                Include ML predictions (signals command)
--confidence 0.2                    Min confidence threshold to trade
--stop-loss 0.03                    Stop loss percentage (3%)
```

## How It Works

### Signal Engine

The signal engine computes two composite scores for each symbol:

**Panic Score (0-1)** — Higher means more fear/selling pressure (potential buy opportunity):
- 5-day drawdown intensity (30%)
- RSI below 50 (25%)
- VIX spike (25%)
- Underperformance vs SPY (20%)

**Overheat Score (0-1)** — Higher means overbought conditions:
- 5-day gain (25%)
- 10-day gain (25%)
- RSI above 50 (30%)
- Low VIX / complacency (20%)

Additional indicators include dollar volume, volume imbalance, VWAP, dollar volume acceleration, and spread proxy.

### ML Model (LSTM)

A 2-layer LSTM neural network predicts whether the next trading day will close higher or lower.

**Architecture:**
- Input: 20-day sliding window of 13 features
- LSTM: 2 layers, 64 hidden units, dropout 0.2
- Output: Sigmoid probability (>0.5 = UP, <0.5 = DOWN)
- ~45,000 parameters per model (~227 KB on disk)

**13 Features:**
| # | Feature | Description |
|---|---------|-------------|
| 1 | rsi14 | 14-day Relative Strength Index (0-1) |
| 2 | ret5 | 5-day return |
| 3 | ret10 | 10-day return |
| 4 | wk_ret | Weekly return |
| 5 | mo_ret | Monthly return (21 trading days) |
| 6 | vol20 | 20-day realized volatility (annualized) |
| 7 | log_dollar_vol | Log10 of daily dollar volume |
| 8 | vix | CBOE VIX index level |
| 9 | vix_chg | VIX daily percent change |
| 10 | vol_imbalance | Buy/sell volume imbalance estimate |
| 11 | vwap_ratio | Close / 5-day rolling VWAP |
| 12 | dv_accel | Dollar volume acceleration |
| 13 | spread_proxy | (High-Low)/Close as bid-ask spread proxy |

**Training:** Adam optimizer, binary cross-entropy loss, early stopping with patience of 7 epochs. Data is split 80/20 temporally (no random shuffle — respects time ordering).

**Confidence:** The raw sigmoid output is converted to a confidence score: `confidence = |probability - 0.5| * 2`. A probability of 0.70 gives confidence 0.40. Trades are only taken when confidence exceeds the threshold (default: 0.20).

### Backtester

Walk-forward backtesting with ML predictions:
- **Entry:** Buy when ML predicts UP with confidence above threshold
- **Exit:** Stop-loss (default 3%) or signal flip (ML predicts DOWN with confidence)
- **Sizing:** Fixed percentage of equity per position (default 95%)
- **Direction:** LONG only

Outputs: total return, annualized return, Sharpe ratio, max drawdown, win rate, profit factor, and full trade history.

### Paper Trader

Continuous trading loop for Alpaca paper trading:
- Runs during market hours (9:30 AM - 4:00 PM ET, weekdays)
- Checks ML signals at configurable intervals (default: 5 minutes)
- Equal-weight allocation across all symbols
- Same entry/exit logic as the backtester
- Graceful shutdown with Ctrl+C
- `paper=True` is hardcoded — cannot accidentally trade live

## Data Providers

| Provider | Daily | Intraday | API Key Required |
|----------|-------|----------|------------------|
| `yahoo` | Yes | Yes | No |
| `alpaca` | Yes | Yes | Yes (free) |
| `hybrid` | Yahoo | Alpaca | Yes (free) |

The `hybrid` provider uses Yahoo for reliable daily history and Alpaca for real-time intraday bars.

## Default Universe

| Symbol | Name |
|--------|------|
| SPY | S&P 500 ETF |
| QQQ | Nasdaq-100 ETF |
| IWM | Russell 2000 ETF |
| IGV | Software ETF |
| SLV | Silver ETF |

Override with `--symbols SPY,AAPL,MSFT`.

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. Paper trading does not guarantee future results. Always do your own research before trading with real money.
