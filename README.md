# Quantitative Stocks

An ML-driven quantitative trading system for ETFs. Combines 21 technical indicators
with a 2-layer LSTM + temporal attention network to generate directional signals,
backtest strategies, and execute paper trades automatically via Alpaca Markets.

> **Paper trading only.** `paper=True` is hardcoded throughout. No real capital is at risk.

Live dashboard: **[quantitative-stocks.vercel.app](https://quantitative-stocks.vercel.app)**

---

## Trading Universe (15 ETFs — each with its own independent LSTM model)

Each symbol has two independent models: daily (swing) and intraday (5-min).

| Status | Symbol | Name | Backtest Return | Extended Hours |
|--------|--------|------|----------------|----------------|
| Active | SPY | S&P 500 | +35.0% | Yes |
| Active | QQQ | Nasdaq 100 | +29.6% | Yes |
| Active | IWM | Russell 2000 Small-Cap | +15.3% | Yes |
| Active | IGV | iShares Expanded Tech-Software | 0% (0 trades) | Yes |
| Active | XLE | Energy Select Sector | +7.6% | Yes |
| Active | SOXX | iShares Semiconductor | +38.4% | Yes |
| Active | GLD | SPDR Gold Shares | +147.7% | Yes |
| Active | SLV | iShares Silver Trust | +129.4% | Yes |
| Active | EWJ | iShares MSCI Japan | +14.1% | No |
| Active | EWT | iShares MSCI Taiwan | +23.8% | No |
| Active | EEM | iShares MSCI Emerging Markets | +14.5% | No |
| Benched | XLF | Financial Select Sector | 0 trades — macro/rate driven | Yes |
| Benched | XLV | Health Care Select Sector | 0 trades — event/earnings driven | Yes |
| Benched | INDA | iShares MSCI India | 0 trades — low signal confidence | No |
| Benched | FXI | iShares China Large-Cap | 0 trades — regulatory jumps | No |

**Benched symbols** have trained models (kept in `models/`) but are excluded from the live trading run script because backtests produced 0 trades — their ML signals never reached the confidence threshold. Root causes: XLF/XLV move on macro/earnings events not captured by technical indicators; INDA/FXI overfit quickly with poor validation loss. They remain in `DEFAULT_UNIVERSE` for signal monitoring and can be re-activated after retraining with more data.

**TLT (20yr Treasuries) fully excluded:** Bond ETFs move on Fed policy, CPI, and yield-curve dynamics — none of which appear in the 21 technical/microstructure features. Val_loss stalled at ≈0.69 (essentially random). Excluded from both trading and `DEFAULT_UNIVERSE`.

**Asian ETFs are regular-hours only:** During US extended hours the underlying markets are closed — spreads widen to 0.5–2% with near-zero volume. Extended-hours trading is limited to the 8 liquid US ETFs marked "Yes" above.

## Backtest Performance — Original 7 Symbols (Jan 2024 – Feb 2026)

| Symbol | Return | Sharpe | Max DD | Trades | Win Rate |
|--------|--------|--------|--------|--------|----------|
| GLD | +147.7% | 2.41 | −12.2% | 14 | 86% |
| SLV | +129.4% | 1.24 | −35.6% | 32 | 56% |
| SPY | +35.0% | 1.20 | −14.6% | 6 | 67% |
| QQQ | +29.6% | 0.83 | −24.2% | 9 | 56% |
| IWM | +15.3% | 0.88 | −7.3% | 2 | 100% |
| XLE | +7.6% | 0.36 | −7.5% | 5 | 60% |
| IGV | 0.0% | — | — | 0 | — |

*Backtest results for new symbols (SOXX, XLF, XLV, EWJ, EWT, INDA, FXI, EEM) pending — run `python main.py backtest --symbol SOXX --start 2024-01-01` after training.*

---

## Project Structure

```
quantitivestocks/
│
│── Core Python
├── main.py                  # Unified CLI — train / predict / backtest / trade
├── signals_engine.py        # Data adapters, all technical indicators, Hurst exponent
├── ml_model.py              # LSTM model, triple-barrier training, Predictor class
├── paper_trader.py          # Live intraday paper trading loop (Alpaca)
├── options_trader.py        # Long ATM straddle options trader (VIX-spike entry)
├── backtester.py            # Walk-forward backtesting engine
├── streamlit_app.py         # Local web dashboard (train / backtest / compare)
│
│── Web Dashboard (Vercel)
├── index.html               # Public dashboard — live signals, P&L, equity curves
├── favicon.svg
├── vercel.json              # Vercel routing config
├── api/
│   ├── signals.py           # GET /api/signals  — live VIX data
│   ├── positions.py         # GET /api/positions — Alpaca account + positions
│   ├── history.py           # GET /api/history  — portfolio equity + filled orders
│   └── requirements.txt
│
│── Models & Outputs (git-ignored)
├── models/
│   ├── {SYMBOL}_lstm.pt         # Daily LSTM weights
│   ├── {SYMBOL}_lstm_5min.pt    # Intraday LSTM weights
│   ├── {SYMBOL}_scaler.json     # Daily feature scaler
│   └── {SYMBOL}_scaler_5min.json
├── outputs/
│   ├── backtest_{SYMBOL}.csv        # Equity curve per backtest run
│   ├── trades_{SYMBOL}.csv          # Trade-by-trade log
│   ├── run_paper_trade.ps1          # Launch script for Windows Task Scheduler
│   └── run_options_trade.ps1
├── logs/
│   ├── paper_trader_{DATE}.log      # Daily paper trader output
│   └── options_trader_{DATE}.log    # Daily options trader output
│
│── Automation (Windows Task Scheduler)
├── run_paper_trade.cmd       # Double-click launcher → calls outputs/run_paper_trade.ps1
├── run_options_trade.cmd     # Double-click launcher → calls outputs/run_options_trade.ps1
├── setup_paper_task.ps1      # Register paper trader as a scheduled task (run once)
├── setup_options_task.ps1    # Register options trader as a scheduled task (run once)
├── setup_both_tasks.ps1      # Register both tasks at once
│
│── Configuration
├── settings/
│   ├── alpaca.env            # API keys (git-ignored)
│   └── settings.py           # Central config (symbols, thresholds, paths)
├── requirements.txt          # Python dependencies
├── .gitignore
│
│── Deployment
├── deploy/                   # Mirror of root for Vercel (auto-synced by pre-commit hook)
└── scripts/                  # Debug / task inspection scripts (dev use only)
```

---

## ML Architecture

```
Input (batch, 20 bars, 21 features)
    ↓  LayerNorm
2-layer LSTM  (hidden=96, dropout=0.25)
    ↓  Temporal Attention  (soft attention over all 20 time steps)
Fully Connected  96 → 48 → ReLU → Dropout → 1 → Sigmoid
    ↓
Probability (0–1) → Direction (UP/DOWN) + Confidence (|p − 0.5| × 2)
```

### 21 Input Features

| Category | Features |
|----------|----------|
| Returns | RSI-14, 5d return, 10d return, weekly return, monthly return |
| Volatility | 20d realized vol, vol regime (short/long ratio), ATR % |
| Volume/Flow | Log dollar vol, vol imbalance (buy fraction), DV acceleration |
| Microstructure | VWAP ratio, spread proxy |
| Macro | VIX level, VIX daily change |
| Trend | MACD histogram, Bollinger %B, BB bandwidth, ADX, trend strength (EMA cross), momentum quality |

### Training Method

| Parameter | Value | Source |
|-----------|-------|--------|
| Labels | **Triple-barrier** (PT +1.5% / SL −1.0% / 5-bar horizon) | De Prado *AFML* Ch.3 |
| Train/Val split | 80% train → **20-bar embargo** → val | De Prado *AFML* Ch.7 |
| Loss | BCE with label smoothing (0.05) | — |
| Optimizer | AdamW + weight decay 1e-4 | — |
| LR schedule | Cosine annealing with warm restarts | — |
| Early stopping | Patience = 10 epochs | — |
| Gradient clipping | 1.0 | — |
| Parameters | ~130,000 per model (~513 KB) | — |

---

## Trading Rules

### Trading Sessions (Alpaca)

| Session | Hours (ET) | Order Type |
|---------|-----------|------------|
| Pre-market | Mon–Fri 4:00 AM – 9:30 AM | Limit + `extended_hours=True` |
| Regular | Mon–Fri 9:30 AM – 4:00 PM | Market order |
| After-hours | Mon–Fri 4:00 PM – 8:00 PM | Limit + `extended_hours=True` |
| Closed | 8:00 PM – 4:00 AM + weekends | Sleep / no trading |

During extended hours the system automatically submits limit orders at last\_price ± 0.1% to ensure fills. Alpaca rejects market orders outside regular hours.

### Entry — all conditions must pass in order

| # | Rule | Source |
|---|------|--------|
| 1 | Session is `regular` or `extended` (not closed overnight / weekend) | — |
| 2 | ADX ≥ 20 (trend strong enough) | — |
| 3 | VIX daily move ≤ 2σ of 20-day rolling σ | Aldridge *HFT* |
| 4 | Hurst exponent H > 0.55 (trending regime) | Chan *Algorithmic Trading* Ch.2 |
| 5 | ML confidence ≥ 0.20 (LONG) or ≥ 0.15 (SHORT) | — |
| 6 | +DI/−DI confirms ML direction | — |
| 7 | Sufficient allocation | — |

### Position Sizing

```
size = base(90%) × vol_scalar × confidence_scalar × kelly(0.5)

vol_scalar        = min(2.0, max(0.3, target_vol / realized_vol))
confidence_scalar = 0.5 + ML_confidence
kelly             = 0.5  ← half-Kelly, reduces drawdown ~75% vs full Kelly
```

*Per Ernest Chan, half-Kelly achieves ~75% of optimal long-run growth with much lower variance.*

### Exit — first match wins

| Priority | Rule | Trigger |
|----------|------|---------|
| 1 | Trailing take-profit | P&L ≥ +4% activates; exit if P&L pulls back 2% from peak |
| 2 | ATR trailing stop | Price moves 2.5 × ATR_at_entry against the peak |
| 3 | Signal flip | ML flips direction with confidence ≥ 0.10 |

---

## Options Strategy (Long ATM Straddle)

**Entry:** VIX daily change ≥ threshold (dynamic 8–20% based on IV rank)
**Position:** Buy ATM call + ATM put with ~30 DTE (delta-neutral, profits from big moves in either direction)

**Exit rules:**
| Rule | Trigger |
|------|---------|
| Profit target | Either leg reaches 1.8× entry cost (+80%) |
| Stop loss | Total position value ≤ 0.6× total cost (−40%) |
| ML signal exit | ML confidence ≥ 0.30 in one direction → close losing leg, ride winner |
| Expiry guard | DTE ≤ 5 days → close everything |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

Create `settings/alpaca.env`:
```
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here
FRED_API_KEY=your_fred_key_here
```

### 3. Train models

```bash
# Daily model (for overnight swing trading)
python main.py train --symbol SPY --mode daily --epochs 50

# Intraday model (for 5-min paper trading)
python main.py train --symbol SPY --mode intraday --interval 5min

# Train all 7 symbols (daily + intraday)
foreach symbol in SPY QQQ IWM IGV SLV GLD XLE:
    python main.py train --symbol {symbol} --mode daily
    python main.py train --symbol {symbol} --mode intraday --interval 5min
```

### 4. Run a backtest

```bash
python main.py backtest --symbol SPY --start 2024-01-01 --end 2026-01-01
```

### 5. Start paper trading

```bash
# Intraday stock trading (checks every 1 min during market hours)
python main.py trade --symbols SPY,QQQ,IWM,IGV,SLV,GLD,XLE --mode intraday

# Options straddle trading (checks every 15 min)
python main.py trade-options --symbols SPY,QQQ,IWM --vix-spike-threshold 15
```

### 6. HTML report (backtest + live log viewer)

```bash
python main.py report        # writes outputs/report.html
python main.py report --open # writes + opens in browser
```

The report has four tabs:
- **Summary** — color-coded performance table (Return, Sharpe, Max DD, Win %, Profit Factor)
- **Equity Curves** — all symbols overlaid + per-symbol drawdown chart
- **Trades** — P&L bar chart + full trade-by-trade table (newest first)
- **Live Log** — latest paper-trader session parsed with pandas: account equity chart, current position status, action counts

### 7. Local Streamlit dashboard (train / backtest controls)

```bash
streamlit run streamlit_app.py
# Open http://localhost:8501
```

---

## Automated Scheduling (Windows)

Both traders run automatically at **6:25 AM Mon–Fri** via Windows Task Scheduler as SYSTEM (no login required).

```powershell
# One-time setup (run as Administrator)
powershell -ExecutionPolicy Bypass -File setup_both_tasks.ps1
```

The tasks:
- Run as **SYSTEM** — no user login needed
- Run on **battery or AC power**
- **Start if missed** — catches up after sleep
- Load API keys from `settings/alpaca.env`
- Write logs to `logs/` with daily timestamps

---

## Data Sources

| Source | Used For |
|--------|----------|
| Yahoo Finance | Historical daily OHLCV, VIX (^VIX fallback) |
| Alpaca Markets | Real-time intraday bars, order execution |
| FRED API | VIX history (primary source) |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| LSTM + attention over plain LSTM | Attention lets the model focus on the most informative bars in the 20-bar window |
| Triple-barrier labels (De Prado) | Economically meaningful labels — ignores 1-bar noise, requires ±1.5%/1.0% moves |
| Purge + embargo in train/val (De Prado) | Prevents feature-window overlap from inflating validation accuracy |
| Half-Kelly sizing (Chan) | ~75% of optimal growth rate with significantly lower drawdown vs full Kelly |
| Hurst exponent filter (Chan) | Only trade momentum when H > 0.55 — avoids entering when market is mean-reverting |
| VIX halt rule (Aldridge) | Stops new entries during structural regime breaks (VIX spike > 2σ) |
| ADX + DI confirmation | ADX confirms trend strength; +DI/−DI confirms direction agrees with ML signal |
| ATR adaptive stops | Stop distance widens in volatile markets — avoids being stopped out by normal noise |

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `ALPACA_API_KEY` | Yes | Alpaca paper trading API key |
| `ALPACA_API_SECRET` | Yes | Alpaca paper trading API secret |
| `FRED_API_KEY` | Recommended | FRED VIX history (falls back to yfinance if missing) |

---

*For educational and research purposes only. Not financial advice.
All performance metrics are from backtests and Alpaca paper trading.
Past performance does not indicate future results.*
