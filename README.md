# Quantitative Stocks

An ML-driven quantitative trading system for ETFs and crypto. Uses a **return-regression
LSTM** (predicts 10-day forward expected return, not direction probability), XGBoost swing
models, and LightGBM intraday models to generate trading signals, backtest strategies,
and execute paper trades automatically via Alpaca Markets.

> **Paper trading only.** `paper=True` is hardcoded throughout. No real capital is at risk.

Live dashboard: **[quantitative-stocks.vercel.app](https://quantitative-stocks.vercel.app)**

---

## Trading Groups (3 separate Alpaca paper accounts)

| Group | Model | Symbols | Schedule |
|-------|-------|---------|----------|
| **Intraday** | LightGBM 5-min | SMH, IWM, IGV, QQQ, EWT, SOXX | Market hours only |
| **Swing** | XGBoost + LSTM daily | GDX, SLV, IGV, QQQ, GLD, SMH, XLK, IBIT | 24/7 |
| **Crypto** | XGBoost daily | 14 crypto pairs (BTC, ETH, SOL, …) | 24/7 |

**Asian ETFs are regular-hours only:** During US extended hours the underlying markets are closed — spreads widen to 0.5–2% with near-zero volume.

---

## Project Structure

```
quantitivestocks/
│
│── Core Python
├── main.py                  # Unified CLI — train / predict / backtest / trade
├── src/
│   ├── signals_engine.py    # Data adapters, all technical indicators, Hurst exponent
│   ├── ml_model.py          # ReturnLSTM model (regression), Predictor class
│   ├── backtester.py        # Walk-forward backtesting engine
│   ├── paper_trader.py      # Live paper trading loop (Alpaca, 3 groups)
│   ├── swing_model.py       # XGBoost swing model (commodities, EM ETFs)
│   ├── intraday_model.py    # LightGBM intraday momentum model
│   ├── alerts.py            # Slack webhook alerts on position entry
│   └── utils.py             # Shared constants, helpers
│
│── Web Dashboard (Vercel)
├── web/
│   └── index.html           # Public dashboard — live signals, P&L, equity curves
├── vercel.json              # Vercel routing config
├── api/
│   ├── signals.py           # GET /api/signals  — live VIX data
│   ├── positions.py         # GET /api/positions — Alpaca account + positions
│   ├── history.py           # GET /api/history  — portfolio equity + filled orders
│   └── requirements.txt
│
│── Models & Outputs
├── models/                      # Trained weights (git-ignored)
│   ├── {SYMBOL}_lstm.pt         # Daily LSTM weights
│   ├── {SYMBOL}_lstm_5min.pt    # Intraday LSTM weights
│   ├── {SYMBOL}_xgb_swing.*     # XGBoost swing model + config + scaler
│   ├── {SYMBOL}_lgb_intraday.*  # LightGBM intraday model + config
│   └── {SYMBOL}_scaler*.json    # Feature scalers
├── outputs/
│   ├── run_paper_trade.ps1      # Watchdog launcher for Windows Task Scheduler
│   ├── backtest_{SYMBOL}.csv    # Equity curve per backtest run
│   ├── trades_{SYMBOL}.csv      # Trade-by-trade log
│   └── signals.json             # Latest signal output
├── logs/
│   └── paper_trader_{DATE}.log  # Daily paper trader output
│
│── Automation (Windows Task Scheduler)
├── run_paper_trade.cmd       # Double-click launcher → calls outputs/run_paper_trade.ps1
│
│── Configuration
├── config/
│   └── trading.json          # Production trading parameters
├── secrets/
│   └── alpaca.env            # API keys (git-ignored)
├── requirements.txt          # Python dependencies
└── .gitignore
```

---

## ML Architecture

### ReturnLSTM (regression — predicts expected return, not direction probability)

```
Input (batch, 20 bars, 12–13 features)
    ↓  LayerNorm
2-layer LSTM  (hidden=96, dropout=0.25)
    ↓  Temporal Attention  (soft attention over all 20 time steps)
Fully Connected  96 → 48 → ReLU → Dropout → 1 (linear, no sigmoid)
    ↓
Expected Return (continuous, e.g. +0.015 = +1.5%)
    ↓
Direction: UP if E[r] > +COST_THRESHOLD, DOWN if < −COST_THRESHOLD, else FLAT
Confidence: min(1.0, |E[r]| / TARGET_RETURN)
```

`COST_THRESHOLD = 0.001` (0.1% — minimum expected return to justify a trade)
`TARGET_RETURN  = 0.02`  (2% — maps to full position size; confidence = 1.0)

> **Note:** The old `abs(probability − 0.5) × 2` confidence formula from the classification era is no longer used. The model now directly outputs expected return, and confidence scales linearly with `|E[r]|`.

### Meta-labeling (deprecated)

RandomForest meta-gating (`{symbol}_meta_rf.joblib`) was trained in v1 to filter low-quality signals. In the v2 regression model it is **not used** — both `backtester.py` and `paper_trader.py` gate purely on `expected_return` vs `cost_threshold`. The `train-meta` CLI command still exists for backward compatibility but has no effect on live trading.

### Per-Group Models

| Group | Model | Features |
|-------|-------|----------|
| **Intraday** | LightGBM (`{SYMBOL}_lgb_intraday.joblib`) | 13 features (trend_strength, adx, log_dollar_vol, …) |
| **Swing** | XGBoost (`{SYMBOL}_xgb_swing.joblib`) | 40–50 features + per-symbol supplements |
| **Daily LSTM** | ReturnLSTM (`{SYMBOL}_lstm.pt`) | 12 features (bb_bandwidth, vol20, ret5, …) |

### Training Method

| Parameter | Value |
|-----------|-------|
| Labels | **10-day forward return** (continuous regression) |
| Train/Val split | 80% train → **20-bar embargo** → val |
| Loss | MSE (regression) |
| Optimizer | AdamW + weight decay 1e-4 |
| LR schedule | Cosine annealing with warm restarts |
| Early stopping | Patience = 10 epochs |
| Gradient clipping | 1.0 |
| Parameters | ~130,000 per LSTM model (~513 KB) |

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

| # | Rule | Notes |
|---|------|-------|
| 1 | Session is `regular` or `extended` (not closed overnight / weekend) | — |
| 2 | VIX < 30 (swing/crypto only; intraday skips — LightGBM handles vol) | Regime filter |
| 3 | SPY above 200-day SMA (bull market check) | Regime filter |
| 4 | Rolling 20-trade win rate ≥ 50% (else 7-day cooldown) | Drawdown guard |
| 5 | `abs(expected_return) > COST_THRESHOLD` (0.1%) | Signal strength |
| 6 | Sufficient allocation per Kelly sizing | — |

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

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

Create `secrets/alpaca.env`:
```
ALPACA_INTRADAY_KEY=your_key_here
ALPACA_INTRADAY_SECRET=your_secret_here
ALPACA_SWING_KEY=your_key_here
ALPACA_SWING_SECRET=your_secret_here
ALPACA_CRYPTO_KEY=your_key_here
ALPACA_CRYPTO_SECRET=your_secret_here
FRED_API_KEY=your_fred_key_here
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/...  # optional
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
# Intraday group (5-min bars, market hours only)
python main.py trade --group intraday --mode intraday --interval 5min

# Swing group (daily bars, 24/7)
python main.py trade --group swing

# Or launch all 3 groups via the watchdog script
run_paper_trade.cmd
```

### 6. HTML report (backtest + live log viewer)

```bash
python main.py report        # writes outputs/report.html
python main.py report --open # writes + opens in browser
```

---

## Automated Scheduling (Windows)

The watchdog launcher runs all 3 trading groups via a single Windows Task Scheduler task.

**Task name:** `QuantStocks-PaperTrader`

| Setting | Value |
|---------|-------|
| Triggers | Boot (30 s delay), Daily 6:15 AM, Logon (10 s delay) |
| Logon type | S4U (runs whether user is logged in or not, no password) |
| Multiple instances | Ignore new (prevents duplicates) |
| Action | `powershell.exe -ExecutionPolicy Bypass -File outputs\run_paper_trade.ps1` |

The task can also be started manually:
```cmd
run_paper_trade.cmd
```

To export the current task definition for version control:
```powershell
schtasks /Query /TN "QuantStocks-PaperTrader" /XML > outputs/QuantStocks-PaperTrader.xml
```

The watchdog (`outputs/run_paper_trade.ps1`):
- Loads API keys from `secrets/alpaca.env` (per-group: `ALPACA_INTRADAY_KEY`, `ALPACA_SWING_KEY`, `ALPACA_CRYPTO_KEY`)
- Kills stale trader processes before restart
- Checks every 120 s; auto-restarts crashed groups
- Intraday group sleeps outside market hours; swing and crypto run 24/7
- Rotates daily logs to `logs/`

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
| Return regression over classification | Direct expected-return output enables natural position sizing (confidence ∝ |E[r]|) |
| LSTM + attention over plain LSTM | Attention lets the model focus on the most informative bars in the 20-bar window |
| Purge + embargo in train/val (De Prado) | Prevents feature-window overlap from inflating validation accuracy |
| Dynamic Kelly sizing (Chan) | Rolling 60-trade window, min 20 trades; adapts bet size to recent edge |
| VIX < 30 regime gate (swing only) | Blocks entries during extreme volatility; intraday LightGBM handles vol internally |
| SPY SMA(200) bull-market filter | Avoids long entries in bear markets |
| ATR adaptive stops | Stop distance widens in volatile markets — avoids being stopped out by normal noise |
| Meta-labeling deprecated | RF meta-gate added no lift in v2 regression model; kept for backward compat only |

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `ALPACA_INTRADAY_KEY` | Yes | Alpaca paper trading API key (intraday account) |
| `ALPACA_INTRADAY_SECRET` | Yes | Alpaca paper trading API secret (intraday account) |
| `ALPACA_SWING_KEY` | Yes | Alpaca paper trading API key (swing account) |
| `ALPACA_SWING_SECRET` | Yes | Alpaca paper trading API secret (swing account) |
| `ALPACA_CRYPTO_KEY` | Yes | Alpaca paper trading API key (crypto account) |
| `ALPACA_CRYPTO_SECRET` | Yes | Alpaca paper trading API secret (crypto account) |
| `FRED_API_KEY` | Recommended | FRED VIX history (falls back to yfinance if missing) |
| `ALERT_WEBHOOK_URL` | Optional | Slack webhook for position-entry alerts |

---

*For educational and research purposes only. Not financial advice.
All performance metrics are from backtests and Alpaca paper trading.
Past performance does not indicate future results.*
