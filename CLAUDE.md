# Quantitative Stocks — Claude Code Instructions

## Environment
- **Python**: always use `.venv/Scripts/python.exe` (uv-managed, Python 3.12.12)
- **NEVER use Anaconda** (`D:\anaconda\...`)
- `uv` binary: `C:\Users\yalil\.local\bin\uv.exe`
- OS: Windows 11; use Unix shell syntax in Bash (forward slashes, /dev/null)

## Project Structure
```
quantitivestocks/
    main.py                     ← unified CLI entry point
    swing_model.py              ← XGBoost swing trading model
    src/
        __init__.py
        signals_engine.py       ← signal engine, adapters, indicators
        ml_model.py             ← ReturnLSTM (regression), training, prediction
        backtester.py           ← walk-forward backtester (with cost model)
        paper_trader.py         ← Alpaca paper trading loop (5 groups, dynamic selection)
        risk_config.py          ← centralized risk params, portfolio constraints
        risk_overlay.py         ← cross-group correlation-based scaling (refreshed every 30min)
        cost_model.py           ← per-symbol spread/slippage/fee model
        model_monitor.py        ← rolling IC, calibration, auto-pause
        oos_feedback.py         ← composite scoring (James-Stein + fee-aware + correlation penalty)
        etf_selector.py         ← ETFSelectorML + ETFIntradaySelectorML (LambdaRank)
        etf_screener.py         ← ETF universe discovery (Alpaca API + yfinance validation)
        coin_selector.py        ← CoinSelector (LambdaRank for crypto)
        alerts.py               ← Slack webhook alerts on position entry
    config/
        trading.json            ← production trading parameters (v4.0)
    tests/
        test_risk_and_costs.py  ← 49 tests for risk/cost/monitoring modules
    scripts/
        retrain_all.py          ← batch retraining (synced to paper_trader groups)
        weekly_pipeline.py      ← automated weekly retrain + backtest + registry update
    data/
        models/                 ← trained weights (.pt, .joblib, .json)
        output/                 ← signals.json, backtest CSVs, trade CSVs
```

## CLI Usage
```bash
# Signals & prediction
python main.py signals  --provider yahoo --ml
python main.py predict  --symbol SPY

# Training (walk-forward split is default for all)
python main.py train    --symbol SPY --epochs 50
python main.py train    --symbol SPY --mode intraday --interval 5min
python main.py train-intraday --symbols XLV,XLF,XLE,USO,SPY,PDBC,XLY --provider yahoo --walk-forward
python main.py train-swing    --symbols FBTC,EWH,SMH,IAU,IBIT,EWW,EWU,GDXJ,MCHI,SLV --provider yahoo --train-recent
python main.py train-crypto
python main.py train-crypto-intraday --walk-forward

# Selectors (LambdaRank)
python main.py train-intraday-etf-selector   # trains ETFIntradaySelectorML
python main.py screen-etf-universe            # refreshes ETF universe (Alpaca + yfinance)

# Backtesting (VIX leakage fixed — include_live=False)
python main.py backtest --symbol SPY --start 2024-01-01 --trailing-stop 0.05 --take-profit 0.08
python main.py backtest --symbol SPY --start 2025-01-01 --mode intraday
python main.py backtest --symbol SPY --start 2024-01-01 --stress-cost-mult 2.0

# Trading
python main.py trade    --group intraday --mode intraday --interval 5min
python main.py trade    --group swing
python main.py trade    --group crypto
python main.py trade    --group crypto_intraday

# Monitoring
python main.py model-health
python main.py validate-risk
```

## Architecture
- `PROJECT_ROOT` defined in `signals_engine.py` (one level up from src/)
- All imports use `from src.xxx import ...` pattern
- `DEFAULT_MODEL_DIR` = `PROJECT_ROOT/data/models/`

## Risk & Portfolio Management (v3)
- **`src/risk_config.py`**: Centralized risk parameters, horizon validation, portfolio constraints
  - Swing: position_pct=45%, max_position=12%, max_sector=35%, max_exposure=75% (from trading.json)
  - Intraday: position_pct=10%, max_position=10%, max_sector=30%, max_exposure=30% (from trading.json)
  - Crypto: position_pct=15%, max_position=6%, max_exposure=40%, kelly_cap=10%
  - Crypto intraday: position_pct=12%, max_position=4%, max_exposure=12%
  - **IMPORTANT**: risk_config.py hardcoded defaults diverge from trading.json — JSON overrides win at runtime
  - `validate_model_mode()`: fails fast if LSTM used with intraday or LightGBM with daily
  - `check_position_allowed()`: per-position, sector, total exposure, BTC beta caps
- **`src/cost_model.py`**: Per-symbol spread + slippage + fee model
  - Liquid ETFs (SPY): ~3bps RT, Mid ETFs (SMH): ~7bps, Low (EWT): ~12bps
  - Crypto major: ~40bps RT, Crypto alt: ~80bps RT (Kraken taker = 26bps/side); extended hours 3x multiplier
  - `simulate_fill()`: realistic fill price for backtester
  - `validate_cost_threshold()`: ensures cost_threshold > estimated_costs × 1.5x
- **`src/model_monitor.py`**: Rolling IC, hit rate, calibration mapping
  - `ModelMonitor`: tracks predicted vs realized returns, auto-pause on degradation
  - `CalibrationMap`: quantile-based mapping from raw predictions to calibrated confidence
  - Thresholds: IC < 0 → warning, IC < -0.10 → pause; hit_rate < 45% → warning

## ML Model Details
- **ReturnLSTM** (was DirectionLSTM): hidden=96, attention, SEQ_LEN=20, **linear output** (no sigmoid)
- Predicts **10-day forward expected return** (continuous); NOT direction probability
- **Horizon metadata**: each model saves `horizon: "10d"` in metrics JSON; validated at load time
- **Label Winsorization**: 1st/99th percentile clipping (was fixed ±10%)
- **Calibration**: after training, builds quantile-based calibration map (`{symbol}_calibration.json`)
- Confidence: calibrated via CalibrationMap if available, else `min(1.0, abs(E[r]) / TARGET_RETURN)`
- Entry gate: `abs(expected_return) > COST_THRESHOLD` (validated against cost model)
- `FEATURE_COLS_DAILY` (12): bb_bandwidth, vol20, ret5, bb_pct_b, wk_ret (3-day momentum), dv_accel, rsi14, ret10, adx, macd_hist_norm, vwap_ratio, vol_regime
- `FEATURE_COLS_INTRADAY` (13): trend_strength, adx, log_dollar_vol, spread_proxy, vol20, atr_pct, vix_chg, vix, ret5, bb_pct_b, wk_ret (3-day), mo_ret, momentum_quality
- NOTE: `wk_ret` was changed from `pct_change(5)` to `pct_change(3)` on 2026-03-25 (was duplicate of ret5)
- `SYMBOL_FEATURE_OVERRIDES`: IGV and FXI use 17 features (12 daily + vix, vix_chg, trend_strength, momentum_quality, mo_ret)
- Helper: `get_feature_cols(mode, symbol=None)` — always pass `symbol=` arg

## Meta-labeling (deprecated in v2)
- RandomForest meta-model: `{symbol}_meta_rf[_5min].joblib`
- **Not used** in current regression model — both backtester and paper_trader gate purely on `expected_return` vs `cost_threshold`
- `train-meta` CLI command exists for backward compatibility but has no effect on trading

## Dynamic Symbol Selection (paper_trader.py) — Unified Architecture (v4, 2026-03-26)
- **All 4 ML groups use the SAME composite scoring pipeline** (previously only crypto had it)
- **SYMBOL_GROUPS are broad universe pools** — NOT hardcoded trading lists
  - ETF pool: ~59 symbols (from `screen-etf-universe` + Alpaca API validation)
  - Crypto pool: ~73 symbols (from Layer 0 universe screener: CMC top-250 × Kraken pairs)
- **3-layer selection pipeline**:
  1. **Layer 0: Universe screening** — ETF screener (Alpaca API + yfinance) or crypto screener (CMC + Kraken)
  2. **Layer 1: LambdaRank selector** — cross-sectional ranking by ML features
     - `ETFSelectorML` (swing ETFs, 10 features)
     - `ETFIntradaySelectorML` (intraday ETFs, 10 features including intraday_activity)
     - `CoinSelector` (crypto, 11 features)
  3. **Layer 2: Composite scoring** (`oos_feedback.py:compute_composite_scores`)
     - `composite = w_selector × rank_norm(selector_score) + w_sharpe × rank_norm(blended_sharpe)`
     - James-Stein shrinkage: cold symbols 30/70 prior/data → live symbols 70/30
     - SPY correlation penalty for ETFs (mirrors BTC penalty for crypto)
     - Fee-aware adjustment via `cost_model` — expensive symbols penalized
- **Top-K selection per group**:
  - `swing`: top-10 from composite ranking
  - `intraday`: top-8 from composite ranking
  - `crypto`: top-6 from composite ranking
  - `crypto_intraday`: top-7 from composite ranking
- **Background auto-training**: untrained top-ranked symbols get trained via `_spawn_background_train`
- **OOS Sharpe registries**:
  - ETF swing: `data/models/swing/promoted_symbols.json`
  - ETF intraday: `data/models/intraday/promoted_symbols.json`
  - Crypto swing: `config/trading.json:oos_sharpe_registry`
  - Crypto intraday: `config/trading.json:oos_sharpe_registry_intraday`
- **Re-ranking frequency**: every 30 minutes in paper_trader run_loop

## Position Sizing — Alpha-Weighted Risk Budget (v4, 2026-03-27)
- **Unified system**: composite score IS the position weight. Selector + sizer are one integrated pipeline.
- **Formula**: `sizing_pct = weight × total_risk_budget / realized_vol`
  - `weight` = `composite_score ^ concentration / sum(all_scores ^ concentration)` (normalized to 1.0)
  - `total_risk_budget`: swing=10%, intraday=5%, crypto=8%, crypto_intraday=4%
  - `concentration=1.5`: rank #1 gets ~1.8x the capital of rank #5
- **Vol-targeted**: high-vol assets get fewer dollars, low-vol get more — equal risk contribution
- **E[r] only gates direction**: LONG if E[r] > cost_threshold, SHORT if < -cost_threshold, SKIP otherwise
- **Safety stages still apply**: VIX scaling, drawdown throttle, auto de-risk, post-loss reduction, max_position_pct cap
- **Legacy fallback**: if `total_risk_budget=0`, falls through to old 12-stage sizing pipeline
- **BTC/SPY correlation penalty**: embedded in composite score, NOT double-counted in sizing
- **Config**: `total_risk_budget` and `concentration` per group in `config/trading.json`

## Regime Filter
- SPY SMA(200) + VIX < 30 (swing/crypto only; intraday skips VIX gate) + rolling 20-trade win-rate cooldown (7d)
- Applied in both backtester.py and paper_trader.py

## Scheduled Task
- Task name: `QuantStocks-PaperTrader`
- Triggers: boot (30s delay), daily 6:15 AM, logon (10s delay)
- Logon type: S4U (runs whether user is logged in or not)
- Action: `powershell.exe -ExecutionPolicy Bypass -File outputs\run_paper_trade.ps1`
- Export: `schtasks /Query /TN "QuantStocks-PaperTrader" /XML > outputs/QuantStocks-PaperTrader.xml`

## Key Fixes (do not regress)
- Backtester meta feature: `iloc[idx_pos-1]` (not `idx_pos` — 1-bar lookahead bug)
- alpaca-py TimeFrame: `TimeFrame(5, TimeFrameUnit.Minute)` not `TimeFrame(5, "Min")`
- Retry with backoff: always `raise RuntimeError(...) from last_exc`
- Dynamic Kelly: rolling 60-trade window, min 20 trades (in both paper_trader + backtester)
- Bar counter restoration: group-aware intervals (300s for ETF 5min, 60s for crypto_intraday 1min)
- Watchdog: PID-based tree kill via `Stop-Process` (not `taskkill /IM python.exe` which kills all Python)
- Gold scalper: staleness check on price data (rejects bars >5min/30min old)
- Risk overlay: `update_overlay()` called every 30min in paper_trader run_loop
- `wk_ret` = `pct_change(3)` NOT `pct_change(5)` (was duplicate of ret5)

### Audit Fixes (2026-03-26)
- **Backtester VIX leakage**: `_fetch_vix_for_training(include_live=False)` in backtester (was True, leaking future VIX)
- **LSTM window off-by-one**: `features_norm.iloc[window_start+1:idx_pos+1]` (was `[window_start:idx_pos]`, missing current bar)
- **Kraken fees**: `_KRAKEN_FEE_BPS = 26.0` (was 13.0 — Kraken charges 26bps/side, not 13bps)
- **Regime cooldown re-arm**: reset `_regime_cooldown_until = None` after expiry (was permanently disabled after first use)
- **Atomic state writes**: all JSON state files use tmpfile + `os.replace()` (was non-atomic truncate-then-write)
- **post_loss_size_mult**: wired into sizing logic (was defined but never applied)
- **initial_capital fallback**: set to default on first-cycle API failure (was undefined → crash)
- **Model monitor NaN guard**: Pearson fallback returns 0.0 on NaN (was propagating NaN, disabling auto-pause)
- **Log rotation**: RotatingFileHandler (10MB, 5 backups) for all paper_trader groups + gold_scalper
- **Session filter overflow**: safe time comparison for minute >= 56 (was `time(hour, minute+4)` crash)

## API Keys
- **All keys live in `secrets/alpaca.env`** — load with `source secrets/alpaca.env` or read programmatically
- **3 separate Alpaca paper accounts** (each group has its own key pair):
  - `ALPACA_INTRADAY_KEY/SECRET` — intraday ETF trading
  - `ALPACA_SWING_KEY/SECRET` — swing ETF trading (different account!)
  - Crypto uses Kraken paper (local state in `outputs/paper_state/`)
- **IMPORTANT**: When checking positions, use the correct account's key for each group. The intraday key does NOT show swing positions and vice versa.
- Other keys in env file: `FRED_API_KEY`, `CMC_API_KEY`, `ALERT_WEBHOOK_URL`, `KRAKEN_API_KEY/SECRET`

## Vercel Site
- Live: `https://quantitative-stocks.vercel.app`
- API routes: `/api/signals`, `/api/positions`, `/api/history` (Python serverless)
- Static: `/models_info.json` — pre-generated model registry (32 models)
- `.vercelignore` excludes `models/*.pt`
- `api/pyproject.toml` declares `requests` as dependency

## Weekly Retraining Pipeline (automated, Sunday 2 AM)
- **Script**: `scripts/weekly_pipeline.py` (orchestrator), `outputs/weekly_pipeline.ps1` (PowerShell wrapper)
- **Task Scheduler**: `QuantStocks-WeeklyRetrain` (S4U, Sunday 2:00 AM)
- **Pipeline steps**:
  1. Screen universes (ETF: Alpaca+yfinance, Crypto: CMC+Kraken)
  2. Define OOS cutoff (today - 30 days)
  3. Retrain ALL models on data before cutoff (258 models: 59 swing + 59 intraday + 65 crypto swing + 65 crypto intraday + 10 selectors)
  4. Backtest ALL models on OOS data (cutoff → today)
  5. Update OOS Sharpe registries from backtest results
  6. Refresh LambdaRank selectors
  7. Slack summary of what improved/degraded
- **OOS cutoff stored in**: `config/trading.json` under `_retrain_cadence._oos_cutoff`
- **CRITICAL**: Training data BEFORE cutoff, backtest data AFTER cutoff. No model sees test data.
- **Per-symbol failure handling**: If one symbol fails, skip it and continue with the rest

## Cross-Group Exposure Coordination (v4, 2026-03-26)
- **Theme caps enforced cross-group**: `check_theme_cap()` merges positions from all groups via `risk_overlay.publish_positions()` / `get_all_group_positions()`
- **Theme caps**: tech=30%, commodities=25%, crypto=15%, emerging=10%, broad=20%
- **Sleeve budgets enforced**: `check_sleeve_budget()` caps each group's exposure as % of total portfolio
- **Sleeve budgets**: swing=60%, intraday=15%, crypto=25%, crypto_intraday=12%
- **SYMBOL_SECTOR**: Expanded to cover all 59 ETFs + 73 crypto pairs (previously only ~25 symbols mapped)
- **Shared state file**: `outputs/cross_group_positions.json` (atomic writes, 15-min staleness filter)

## Audit Fixes (2026-03-26, session 2)
- **Kraken fees**: `_KRAKEN_FEE_BPS = 26.0` per side (was 13.0 — RT is now 52bps, not 26bps)
- **Regime cooldown re-arm**: reset `_regime_cooldown_until = None` after expiry (was permanently disabled)
- **Crypto intraday target_return**: 4% → 1% (was crushing position sizes — E[r]/target_return ratio was ~0.1)
- **wk_ret consistency**: `WEEK_DAYS = 3` in signals_engine.py (matches ml_model.py `pct_change(3)`)
- **Gold scalper TP restore**: Recalculates remaining contracts on restart based on pips passed (was [0,0,0,0])
- **Gold scalper staleness**: Reverted — Yahoo futures have normal ~10min delay, not a bug
- **Watchdog $pid**: Renamed to `$procId` to avoid PowerShell read-only variable conflict
- **Session filter minute overflow**: Clamped to 59 (was `minute + 4` crash when minute >= 56)
- **Dashboard win rate**: Live Win Rate, PF, Avg Win/Loss, Net P&L per group tab
- **Retrain script**: Symbol lists synced to active paper_trader groups + crypto_intraday step added
- **FRED API key**: Removed hardcoded fallback from retrain_all.py

## User Goals
- End goal: **intraday trading** with real-time data
- Alpaca for both data and execution
- MGC gold futures: TradingView webhook signals → paper executor
