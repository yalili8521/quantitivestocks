# Copilot instructions for QuantitativeStocks

## Big picture architecture
- Use `main.py` as the only entrypoint for workflows (`signals`, `train`, `predict`, `backtest`, `trade`, `trade-options`). Most modules are designed to be invoked through it.
- Core pipeline is: `signals_engine.py` (feature + sentiment signals) -> `ml_model.py` (LSTM train/predict) -> `backtester.py` (walk-forward simulation) -> `paper_trader.py` / `options_trader.py` (live paper execution).
- `signals_engine.py` defines the provider abstraction (`DataAdapter`) and concrete adapters (`YahooFinanceAdapter`, `AlpacaAdapter`, `HybridAdapter`). Reuse this pattern for new data sources.
- `ml_model.py` and `backtester.py` share feature contracts through `FEATURE_COLS`, `FeatureEngine`, and `SEQ_LEN=20`; keep these synchronized when changing features.
- `api/*.py` are lightweight Vercel serverless endpoints for dashboard/live status; they intentionally avoid heavy ML dependencies (for example, no torch inference in `api/signals.py`).

## Source-of-truth paths and outputs
- Treat `outputs/` as the canonical runtime output directory (`signals.json`, `backtest_<SYMBOL>.csv`, `trades_<SYMBOL>.csv`, charts, dashboards).
- Model artifacts are loaded from `models/` (for example `SPY_lstm.pt`, `SPY_scaler.json`) via `DEFAULT_MODEL_DIR` in `ml_model.py`.
- Be careful with stale path constants: `settings/settings.py` references `results/` and `src/`, but active modules write to/read from root-level files and `outputs/`.
- Keep compatibility with existing generated assets under `outputs/` and deployed API copies under `deploy/api/`.

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
- Scheduled-task entrypoints are operationally important: `outputs/run_paper_trade.ps1` and `outputs/run_options_trade.ps1` (these load env files, rotate logs, and kill stale loops before restart).

## Project-specific conventions
- Confidence is derived as `abs(probability - 0.5) * 2`; thresholds differ for long/short/exit. Preserve this contract across backtest and live trading.
- Both `backtester.py` and `paper_trader.py` currently disable take-profit exits in code; exits are primarily trailing-stop + signal flip.
- Trading modules are explicitly paper-only (`TradingClient(..., paper=True)` hardcoded). Do not remove this safety default unless explicitly requested.
- Keep CLI flags aligned across modules (`--mode`, `--interval`, `--confidence`, `--short-confidence`, `--exit-confidence`) to avoid workflow drift.

## Integrations and boundaries
- External services: Alpaca (trading + market data), Yahoo Finance (`yfinance`), FRED VIX API (+ yfinance `^VIX` fallback in ML/trading path).
- Env vars expected in real runs: `ALPACA_API_KEY`, `ALPACA_API_SECRET`, optional `FRED_API_KEY`.
- Serverless endpoints use direct REST with `requests` and short cache headers; avoid adding large dependencies in `api/`.

## Change guidance for agents
- Prefer minimal, surgical edits in core files (`main.py`, `signals_engine.py`, `ml_model.py`, `backtester.py`, `paper_trader.py`, `options_trader.py`) because logic is tightly coupled by shared feature/threshold assumptions.
- If changing output schemas (CSV/JSON), update both producers and consumers (`streamlit_app.py`, `index.html`, `api/history.py`, and `outputs/create_comprehensive_dashboard.py`).
- When adding new runtime commands, route them through `main.py` so automation scripts and user workflows remain consistent.