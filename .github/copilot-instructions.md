# Copilot instructions for QuantitativeStocks

## Big picture architecture
- Use `main.py` as the only entrypoint for workflows (`signals`, `train`, `predict`, `backtest`, `trade`). Most modules are designed to be invoked through it.
- Core pipeline is: `signals_engine.py` (feature + sentiment signals) -> `ml_model.py` (ReturnLSTM train/predict) -> `backtester.py` (walk-forward simulation) -> `paper_trader.py` (live paper execution, 3 groups: intraday/swing/crypto).
- `signals_engine.py` defines the provider abstraction (`DataAdapter`) and concrete adapters (`YahooFinanceAdapter`, `AlpacaAdapter`, `HybridAdapter`). Reuse this pattern for new data sources.
- `ml_model.py` and `backtester.py` share feature contracts through `FEATURE_COLS`, `FeatureEngine`, and `SEQ_LEN=20`; keep these synchronized when changing features.
- `api/*.py` are lightweight Vercel serverless endpoints for dashboard/live status; they intentionally avoid heavy ML dependencies (for example, no torch inference in `api/signals.py`).

## Source-of-truth paths and outputs
- Treat `outputs/` as the canonical runtime output directory (`signals.json`, `backtest_<SYMBOL>.csv`, `trades_<SYMBOL>.csv`, charts, dashboards).
- Model artifacts are loaded from `models/` (for example `SPY_lstm.pt`, `SPY_scaler.json`) via `DEFAULT_MODEL_DIR` in `ml_model.py`.
- API credentials live in `secrets/alpaca.env` (per-group keys: `ALPACA_INTRADAY_KEY`, `ALPACA_SWING_KEY`, `ALPACA_CRYPTO_KEY`).

## Developer workflows (Windows-first)
- Create env and install deps: `python -m venv .venv` then `.venv\Scripts\activate` then `pip install -r requirements.txt`.
- Typical local loop:
  - `python main.py signals --provider yahoo --ml`
  - `python main.py train --symbol SPY --provider yahoo --epochs 50`
  - `python main.py backtest --symbol SPY --start 2024-01-01`
  - `python main.py trade --provider alpaca --mode intraday --interval 5min`
- Dashboard workflows:
  - Streamlit: `python -m streamlit run streamlit_app.py`
  - Static/API dashboard: `index.html` + `api/*.py` (deployed via `vercel.json`).
- Scheduled-task entrypoint is operationally important: `outputs/run_paper_trade.ps1` (loads env files, rotates logs, kills stale loops before restart, watchdog for 3 trading groups).

## Project-specific conventions
- The LSTM model (ReturnLSTM) predicts 10-day forward expected return (continuous, not probability). Confidence is derived as `min(1.0, abs(expected_return) / TARGET_RETURN)` where `TARGET_RETURN=0.02`. Entry gate requires `abs(expected_return) > COST_THRESHOLD` (0.001).
- Meta-labeling (RandomForest gate) is deprecated in v2 — not used in backtester or paper_trader.
- Both `backtester.py` and `paper_trader.py` currently disable take-profit exits in code; exits are primarily trailing-stop + signal flip.
- Trading modules are explicitly paper-only (`TradingClient(..., paper=True)` hardcoded). Do not remove this safety default unless explicitly requested.
- VIX regime gate: VIX < 30 blocks swing/crypto entries; intraday skips this gate (LightGBM handles vol internally).

## Integrations and boundaries
- External services: Alpaca (trading + market data, 3 separate paper accounts), Yahoo Finance (`yfinance`), FRED VIX API (+ yfinance `^VIX` fallback in ML/trading path).
- Env vars expected in real runs: `ALPACA_INTRADAY_KEY`, `ALPACA_INTRADAY_SECRET`, `ALPACA_SWING_KEY`, `ALPACA_SWING_SECRET`, `ALPACA_CRYPTO_KEY`, `ALPACA_CRYPTO_SECRET`, optional `FRED_API_KEY`, optional `ALERT_WEBHOOK_URL` (Slack).
- Serverless endpoints use direct REST with `requests` and short cache headers; avoid adding large dependencies in `api/`.

## Change guidance for agents
- Prefer minimal, surgical edits in core files (`main.py`, `signals_engine.py`, `ml_model.py`, `backtester.py`, `paper_trader.py`) because logic is tightly coupled by shared feature/threshold assumptions.
- If changing output schemas (CSV/JSON), update both producers and consumers (`web/index.html`, `api/history.py`).
- When adding new runtime commands, route them through `main.py` so automation scripts and user workflows remain consistent.