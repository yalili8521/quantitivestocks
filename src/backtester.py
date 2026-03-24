#!/usr/bin/env python3
"""
ML-Prediction-Driven Backtester (v2 Regression)
=================================
Walk-forward backtest using ML regression predictions.
Architecture: trend-following base layer (SMA) + signal-proportional sizing
+ signal-decay exits + disaster stop (ATR safety net).

Usage (via main.py):
    python main.py backtest --symbol SPY --start 2024-01-01
    python main.py backtest --symbol SPY --start 2024-01-01 --model swing
    python main.py backtest --symbol XLE --start 2024-01-01 --model options

Requires: trained model (run: python main.py train --symbol SPY first).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import torch

from signals_engine import (
    build_adapter, PROJECT_ROOT,
    compute_atr,
)
from ml_model import FeatureEngine, DirectionLSTM, get_feature_cols, SEQ_LEN  # LSTM path (deprecated)
from utils import (
    DEFAULT_MODEL_DIR, SWING_MODEL_DIR, INTRADAY_MODEL_DIR, CRYPTO_MODEL_DIR,
    CRYPTO_INTRADAY_MODEL_DIR,
    BACKTEST_DIR, TRADES_DIR, OUTPUT_DIR as _OUTPUT_DIR,
    _fetch_vix_for_training, COST_THRESHOLD, TARGET_RETURN, get_model_dir,
)
from risk_config import (
    validate_model_mode,
    get_risk_config,
    get_symbol_cap,
    get_effective_min_hold,
    DeRiskState,
    evaluate_derisk,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("backtester")

OUTPUT_DIR = _OUTPUT_DIR  # backward compat


# ===================================================================
# Trade & Portfolio tracking
# ===================================================================
@dataclass
class Trade:
    entry_date: object          # date or datetime
    entry_price: float
    direction: str              # "LONG" or "SHORT"
    size: float                 # number of shares
    peak_price: float = 0.0     # best price since entry (highest for LONG, lowest for SHORT)
    atr_at_entry: float = 0.0   # ATR at entry for adaptive stop
    tp_activated: bool = False   # trailing take-profit activated
    tp_trail_peak: float = 0.0   # highest P&L since TP activation
    breakeven_armed: bool = False  # breakeven ratchet activated
    signal_flip_count: int = 0     # consecutive signal flips for decay exit
    bars_held: int = 0             # bars since entry for min-hold enforcement
    # MFE/MAE tracking (updated bar-by-bar)
    mfe_pct: float = 0.0          # max favorable excursion (best unrealized PnL %)
    mae_pct: float = 0.0          # max adverse excursion (worst unrealized PnL %, negative)
    exit_layer: str = ""          # which exit layer fired (safety/breakeven/profit_lock/signal_decay/time)
    exit_date: object = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: str = ""


@dataclass
class Portfolio:
    initial_capital: float = 100_000.0
    cash: float = 100_000.0
    position: Optional[Trade] = None
    closed_trades: List[Trade] = field(default_factory=list)
    equity_curve: List[dict] = field(default_factory=list)


# ===================================================================
# Backtest result
# ===================================================================
@dataclass
class BacktestResult:
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    avg_trade_duration_days: float
    equity_curve: pd.DataFrame
    price_series: pd.DataFrame = field(default_factory=pd.DataFrame)
    drawdown_series: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: List[Trade] = field(default_factory=list)
    mode: str = "daily"
    intraday_interval: str = "5min"
    vol_tier: str = ""
    exit_variant: str = ""
    bt_group: str = ""


# ===================================================================
# Backtester
# ===================================================================
class Backtester:
    """Walk-forward backtester driven by ML regression predictions (v2).

    Architecture:
    - Trend-following base layer: SMA(50) determines allowed direction
    - ML signal: expected return magnitude determines position size
    - Signal-decay exit: re-run model each bar, exit when signal reverses
    - Disaster stop: 3x ATR safety net (not primary exit mechanism)
    """

    def __init__(
        self,
        symbol: str,
        adapter,
        fred_key: Optional[str] = None,
        model_dir: str = DEFAULT_MODEL_DIR,
        initial_capital: float = 100_000.0,
        position_pct: float = 0.50,          # was 0.95, now conservative
        mode: str = "daily",
        intraday_interval: str = "5min",
        model_type: str = "lstm",
        # v2 regression parameters
        trend_sma_period: int = 50,
        cost_threshold: float = COST_THRESHOLD,
        target_return: float = TARGET_RETURN,
        disaster_stop_atr_mult: float = 3.0,
        disaster_stop_max_pct: float = 0.20,
        profit_lock_atr_mult: float = 2.0,
        profit_lock_trail_atr_mult: float = 1.5,
        max_underwater_days: int = 90,
        # Cost model / stress test
        stress_cost_mult: float = 1.0,
        use_cost_model: bool = True,
        # A/B exit variant
        exit_variant: Optional[str] = None,
    ):
        self.symbol = symbol
        self.adapter = adapter
        self.fred_key = fred_key
        self.model_dir = model_dir
        self.initial_capital = initial_capital
        self.position_pct = position_pct
        self.mode = mode
        self.intraday_interval = intraday_interval
        self.model_type = model_type
        self.trend_sma_period = trend_sma_period
        self.cost_threshold = cost_threshold
        self.target_return = target_return
        self.disaster_stop_atr_mult = disaster_stop_atr_mult
        self.disaster_stop_max_pct = disaster_stop_max_pct
        self.profit_lock_atr_mult = profit_lock_atr_mult
        self.profit_lock_trail_atr_mult = profit_lock_trail_atr_mult
        self.max_underwater_days = max_underwater_days
        self.stress_cost_mult = stress_cost_mult
        self.use_cost_model = use_cost_model
        self.exit_variant = exit_variant
        if self.model_type == "crypto_intraday":
            self._risk_group = "crypto_intraday"
        elif self.model_type == "intraday":
            self._risk_group = "intraday"
        elif self.symbol.endswith("-USD") or "/" in self.symbol:
            self._risk_group = "crypto"
        else:
            self._risk_group = "swing"
        self._derisk_state = DeRiskState()
        self._max_loss_exit = None

        if self.model_type == "intraday":
            validate_model_mode("lgb_intraday", mode)
        elif self.model_type == "crypto_intraday":
            validate_model_mode("lgb_intraday_crypto", mode)
        elif self.model_type == "swing":
            validate_model_mode("tft_swing", mode)
        else:
            validate_model_mode(self.model_type, mode)

        # Validate cost threshold against estimated costs
        if use_cost_model:
            try:
                from cost_model import validate_cost_threshold
                ok, msg, rt_cost = validate_cost_threshold(
                    symbol, cost_threshold, safety_margin=1.5
                )
                if not ok:
                    log.warning("Cost threshold validation: %s", msg)
                    # Auto-adjust to safe level
                    self.cost_threshold = rt_cost * 1.5
                    log.info("Auto-adjusted cost_threshold to %.4f for %s",
                             self.cost_threshold, symbol)
            except ImportError:
                pass

    # ------------------------------------------------------------------
    def _compute_trend_signal(self, close_series: pd.Series) -> pd.Series:
        """Trend-following base layer: +1 when close > SMA, -1 when below."""
        sma = close_series.rolling(self.trend_sma_period).mean()
        return (close_series > sma).astype(float) * 2 - 1  # +1 or -1

    # ------------------------------------------------------------------
    def _run_intraday_lgb(self, start_date: str,
                          end_date: Optional[str] = None) -> BacktestResult:
        """Day-level intraday backtest using LightGBM momentum filter.

        Logic per trading day:
          1. Build first-30m features (IntradayFeatureEngine)
          2. LightGBM predicts P(following first-30m direction works today)
          3. If P >= threshold: enter at first-30m close (~10:00), exit at EOD (~15:30)
          4. Direction = sign of first-30m return
        Alpaca data fetched for full lookback; Yahoo caps at 60 days.
        """
        from intraday_model import IntradayFeatureEngine, FEATURE_NAMES
        import joblib

        start_dt = pd.to_datetime(start_date).date()
        end_dt = pd.to_datetime(end_date).date() if end_date else datetime.now(timezone.utc).date()
        # Must reach back to start_dt from today, not just span the window
        from datetime import date as _date
        days_needed = (_date.today() - start_dt).days + 60

        log.info("Fetching %d days of 5-min intraday data for %s (Alpaca)...",
                 days_needed, self.symbol)
        bars = self.adapter.fetch_intraday(self.symbol, "5min", lookback_days=days_needed)
        log.info("Got %d bars.", len(bars))

        vix_df = _fetch_vix_for_training(self.fred_key, lookback_days=max(days_needed, 500))
        log.info("Got %d VIX rows.", len(vix_df))

        # Build per-day feature dataset
        engine = IntradayFeatureEngine()
        data = engine.build_training_data(bars, vix_df)
        if data.empty:
            log.error("No training data built for %s intraday backtest.", self.symbol)
            return self._compute_results(Portfolio(
                initial_capital=self.initial_capital,
                cash=self.initial_capital,
            ))

        log.info("Built %d daily samples for %s.", len(data), self.symbol)

        # Load LightGBM model + threshold
        model_path = os.path.join(self.model_dir, f"{self.symbol}_lgb_intraday.joblib")
        config_path = os.path.join(self.model_dir, f"{self.symbol}_lgb_intraday_config.json")
        if not os.path.exists(model_path):
            log.error("No trained intraday model for %s. Run: "
                      "python main.py train-intraday --symbols %s --provider alpaca",
                      self.symbol, self.symbol)
            sys.exit(1)

        lgb_model = joblib.load(model_path)
        threshold = 0.55
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            threshold = float(cfg.get("threshold", 0.55))
        log.info("Loaded LightGBM for %s (threshold=%.3f).", self.symbol, threshold)

        # Filter to backtest window
        data["date"] = pd.to_datetime(data["date"]).dt.date
        data = data[(data["date"] >= start_dt) & (data["date"] <= end_dt)].reset_index(drop=True)
        log.info("Backtest window: %d trading days (%s → %s).",
                 len(data), start_dt, end_dt)

        # --- Regime filter: hardcoded SMA gates removed (model features handle regime) ---
        # Rolling 20-trade win rate cooldown remains active.
        log.info("Hardcoded regime gates disabled — model features drive entry decisions.")

        # --- Rolling 20-trade win rate cooldown ---
        from collections import deque
        from datetime import timedelta as _td
        recent_wins: deque = deque(maxlen=20)
        cooldown_until = None

        # Simulate day-by-day
        portfolio = Portfolio(initial_capital=self.initial_capital, cash=self.initial_capital)
        price_rows = []
        cooldown_skipped = 0

        for _, row in data.iterrows():
            day = row["date"]
            entry_price = float(row["entry_price"])
            eod_close = float(row["eod_close"])
            first_30m_ret = float(row["first_30m_ret_raw"])

            if entry_price <= 0 or eod_close <= 0:
                continue

            price_rows.append({"date": day, "close": eod_close})

            # Rolling win rate cooldown
            if cooldown_until is not None:
                if day < cooldown_until:
                    self._record_equity(portfolio, day, eod_close)
                    cooldown_skipped += 1
                    continue
                else:
                    cooldown_until = None  # cooldown expired

            # LightGBM prediction
            x = np.array([[row[f] for f in FEATURE_NAMES]], dtype=np.float32)
            prob = float(lgb_model.predict_proba(x)[0][1])

            # Entry: require LightGBM confidence + nonzero first-30m signal
            if prob < threshold or abs(first_30m_ret) < 1e-6:
                self._record_equity(portfolio, day, eod_close)
                continue

            direction = "LONG" if first_30m_ret > 0 else "SHORT"

            # Open position at 10:00 (~first_30m close)
            equity = self._current_equity(portfolio, entry_price)
            invest = equity * min(self.position_pct, 0.98)
            shares = invest / entry_price
            portfolio.cash -= invest

            trade = Trade(
                entry_date=day, entry_price=entry_price,
                direction=direction, size=shares,
                peak_price=entry_price, atr_at_entry=0.0,
            )

            # Close at EOD (~15:30)
            trade.exit_date = day
            trade.exit_price = eod_close
            trade.exit_reason = "eod"
            if direction == "LONG":
                trade.pnl = shares * (eod_close - entry_price)
                proceeds = shares * eod_close
            else:
                trade.pnl = shares * (entry_price - eod_close)
                proceeds = shares * entry_price + trade.pnl
            portfolio.cash += proceeds
            portfolio.closed_trades.append(trade)

            # Update rolling win rate; trigger cooldown if < 50% over last 20 trades
            recent_wins.append(1 if trade.pnl > 0 else 0)
            if len(recent_wins) == 20 and cooldown_until is None:
                wr = sum(recent_wins) / 20
                if wr < 0.5:
                    cooldown_until = day + _td(days=7)
                    log.info("Rolling WR %.0f%% < 50%% → cooldown until %s", wr * 100, cooldown_until)

            self._record_equity(portfolio, day, eod_close)

        log.info("Cooldown skipped %d days.", cooldown_skipped)

        price_df = pd.DataFrame(price_rows)
        return self._compute_results(portfolio, price_df)

    # ------------------------------------------------------------------
    def _run_crypto_intraday(self, start_date: str,
                              end_date: Optional[str] = None) -> BacktestResult:
        """Bar-level backtest for crypto intraday LGB model (5-min bars).

        Logic per bar:
          1. Build features via CryptoIntradayFeatureEngine
          2. LGB predicts expected 1-hour (12-bar) forward return
          3. If |E[r]| > cost_threshold and no position: enter
          4. Exit on signal decay, max-hold (48 bars), or disaster stop
        Data fetched from BinanceUS via CryptoIntradayData.
        """
        from crypto_intraday_model import (
            CryptoIntradayFeatureEngine, get_intraday_feature_cols,
            FORWARD_BARS, MAX_HOLD_BARS, LOOKBACK_BARS,
            COST_THRESHOLD as CRYPTO_INTRADAY_CT,
            TARGET_RETURN as CRYPTO_INTRADAY_TR,
        )
        from crypto_intraday_data import CryptoIntradayData

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp.now(tz="UTC")
        if start_dt.tzinfo is None:
            start_dt = start_dt.tz_localize("UTC")
        if end_dt.tzinfo is None:
            end_dt = end_dt.tz_localize("UTC")

        days_needed = (end_dt - start_dt).days + 30  # extra for feature warmup
        log.info("Fetching %d days of 5-min bars for %s from BinanceUS...",
                 days_needed, self.symbol)

        data_source = CryptoIntradayData()
        # Normalize symbol: BTC-USD → BTC/USD for ccxt
        ccxt_sym = self.symbol.replace("-", "/")
        bars = data_source.fetch_training_bars(ccxt_sym, days=days_needed)
        if len(bars) < 1000:
            log.error("%s: only %d bars, need >= 1000", self.symbol, len(bars))
            return self._compute_results(Portfolio(
                initial_capital=self.initial_capital,
                cash=self.initial_capital,
            ))
        log.info("Got %d bars for %s.", len(bars), self.symbol)

        # Fetch BTC bars for cross-market features (altcoins only)
        is_btc = "BTC" in self.symbol.upper()
        btc_bars = None
        if not is_btc:
            btc_bars = data_source.fetch_training_bars("BTC/USD", days=days_needed)
            log.info("Got %d BTC reference bars.", len(btc_bars))

        # Build features
        engine = CryptoIntradayFeatureEngine()
        features = engine.build_features(bars, btc_bars, symbol=self.symbol)
        if features.empty:
            log.error("No features built for %s.", self.symbol)
            return self._compute_results(Portfolio(
                initial_capital=self.initial_capital,
                cash=self.initial_capital,
            ))
        log.info("Built %d feature rows for %s.", len(features), self.symbol)

        # Load LightGBM model
        sym_clean = self.symbol.replace("/", "-")
        model_path = os.path.join(self.model_dir,
                                  f"{sym_clean}_lgb_intraday_crypto.joblib")
        config_path = os.path.join(self.model_dir,
                                   f"{sym_clean}_lgb_intraday_crypto_config.json")
        if not os.path.exists(model_path):
            log.error("No crypto intraday model for %s at %s. "
                      "Run: python main.py train-crypto-intraday",
                      self.symbol, model_path)
            sys.exit(1)

        lgb_model = joblib.load(model_path)
        feature_cols = get_intraday_feature_cols(self.symbol)

        # Load GRU model for ensemble (parity with CryptoIntradayPredictor)
        gru_path = os.path.join(self.model_dir,
                                f"{sym_clean}_gru_intraday_crypto.pt")
        gru_model = None
        gru_mean = None
        gru_std = None
        gru_seq_len = 12  # default GRU_SEQ_LEN

        # Load config for cost/target overrides
        crypto_ct = self.cost_threshold
        crypto_tr = self.target_return
        crypto_pred_scale = 1.0
        gru_active = False
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            crypto_ct = max(
                cfg.get("cost_threshold", CRYPTO_INTRADAY_CT),
                self.cost_threshold,  # respect auto-adjusted safety threshold
            )
            crypto_tr = cfg.get("target_return", CRYPTO_INTRADAY_TR)
            crypto_pred_scale = cfg.get("pred_scale", 1.0)
            gru_active = cfg.get("gru_active", False)
            saved_features = cfg.get("feature_names")
            if saved_features:
                feature_cols = saved_features
            # OOS contamination check — hard block to prevent polluted backtests
            val_end = cfg.get("val_end")
            if val_end:
                val_end_dt = pd.to_datetime(val_end)
                if val_end_dt.tzinfo is None:
                    val_end_dt = val_end_dt.tz_localize("UTC")
                if start_dt < val_end_dt:
                    raise ValueError(
                        f"OOS CONTAMINATION: backtest start {start_dt.date()} is before "
                        f"model val_end {val_end_dt.date()}. Backtest data would overlap "
                        f"training/validation data. Use --start with a date after "
                        f"{val_end_dt.date()} for true out-of-sample results."
                    )

        # Load GRU if active (matching CryptoIntradayPredictor._load)
        if gru_active and os.path.exists(gru_path):
            try:
                import torch
                checkpoint = torch.load(gru_path, map_location="cpu", weights_only=False)
                from attention import GRUWithAttention as GRUReturnModel
                gru_model = GRUReturnModel(
                    n_features=checkpoint["n_features"],
                    hidden=checkpoint["hidden"],
                    n_layers=checkpoint["n_layers"],
                )
                gru_model.load_state_dict(checkpoint["model_state"])
                gru_model.eval()
                gru_mean = np.array(checkpoint["scaler_mean"], dtype=np.float32)
                gru_std = np.array(checkpoint["scaler_std"], dtype=np.float32)
                gru_seq_len = checkpoint.get("seq_len", 12)
                log.info("Loaded GRU ensemble for %s (seq_len=%d)", self.symbol, gru_seq_len)
            except Exception as e:
                log.warning("Failed to load GRU for %s: %s — using LGB-only", self.symbol, e)
                gru_model = None

        ensemble_str = "LGB+GRU" if gru_model else "LGB-only"
        log.info("Loaded %s for %s (%d features, ct=%.4f, tr=%.4f, pred_scale=%.2f).",
                 ensemble_str, self.symbol, len(feature_cols), crypto_ct, crypto_tr, crypto_pred_scale)

        # --- Regime filter: hardcoded BTC SMA(200) gate removed ---
        # Model features (btc_sma200_flag, btc_trend_strength, etc.) handle regime.
        bars_ts = pd.to_datetime(bars["ts"])
        if bars_ts.dt.tz is None:
            bars_ts = bars_ts.dt.tz_localize("UTC")
        log.info("Hardcoded BTC regime gate disabled — model features drive entry decisions.")

        # --- Walk-forward bar-by-bar ---
        from collections import deque
        from datetime import timedelta as _td

        feature_ts = pd.to_datetime(features["ts"])
        if feature_ts.dt.tz is None:
            feature_ts = feature_ts.dt.tz_localize("UTC")

        # Align bars to feature timestamps for close prices
        bars_indexed = bars.set_index("ts")
        if bars_indexed.index.tz is None:
            bars_indexed.index = bars_indexed.index.tz_localize("UTC")

        portfolio = Portfolio(
            initial_capital=self.initial_capital,
            cash=self.initial_capital,
        )
        recent_wins: deque = deque(maxlen=20)
        cooldown_until = None
        cooldown_skipped = 0
        bars_in_position = 0
        reentry_cooldown_remaining = 0  # bars to wait after signal_decay exit
        price_rows = []

        # Match paper_trader protections (from risk_config)
        from risk_config import (
            CRYPTO_INTRADAY_MIN_HOLD_BARS,
            CRYPTO_INTRADAY_SIGNAL_REVERSAL_COOLDOWN_BARS,
        )
        min_hold_bars = CRYPTO_INTRADAY_MIN_HOLD_BARS          # 12 bars (1 hour)
        reentry_cooldown_bars = CRYPTO_INTRADAY_SIGNAL_REVERSAL_COOLDOWN_BARS  # 3 bars (15 min)

        for i in range(len(features)):
            ts = feature_ts.iloc[i]
            bar_date = ts.date()

            if ts < start_dt:
                continue
            if ts > end_dt:
                break

            # Get close price for this bar
            close_val = float(features.iloc[i].get("close_position", 0))
            # Actually we need the raw close — look it up from bars
            closest_idx = bars_indexed.index.searchsorted(ts)
            if closest_idx >= len(bars_indexed):
                closest_idx = len(bars_indexed) - 1
            bar_close = float(bars_indexed.iloc[closest_idx]["close"])
            bar_high = float(bars_indexed.iloc[closest_idx]["high"])
            bar_low = float(bars_indexed.iloc[closest_idx]["low"])

            price_rows.append({"date": bar_date, "close": bar_close})

            is_last = (i == len(features) - 1)

            # --- LGB + GRU ensemble prediction (parity with CryptoIntradayPredictor) ---
            try:
                x_row = features[feature_cols].iloc[i:i + 1].values.astype(np.float32)
                lgb_pred = float(lgb_model.predict(x_row)[0])

                gru_pred = None
                if gru_model is not None and i >= gru_seq_len:
                    import torch
                    x_seq = features[feature_cols].iloc[i - gru_seq_len:i].values.astype(np.float32)
                    x_seq_n = (x_seq - gru_mean) / np.clip(gru_std, 1e-8, None)
                    x_tensor = torch.from_numpy(x_seq_n[np.newaxis])
                    with torch.no_grad():
                        gru_pred = float(gru_model(x_tensor).item())

                # Ensemble: default 70/30 LGB/GRU (matching predictor defaults)
                if gru_pred is not None:
                    w_lgb = 0.70
                    expected_return = (w_lgb * lgb_pred + (1 - w_lgb) * gru_pred) * crypto_pred_scale
                else:
                    expected_return = lgb_pred * crypto_pred_scale
            except Exception:
                expected_return = 0.0

            # --- Exit logic ---
            if portfolio.position is not None:
                pos = portfolio.position
                bars_in_position += 1

                if pos.direction == "LONG":
                    unrealized_pct = (bar_close - pos.entry_price) / pos.entry_price
                else:
                    unrealized_pct = (pos.entry_price - bar_close) / pos.entry_price

                # Disaster stop: 2% max loss
                if unrealized_pct <= -0.02:
                    self._close_position(portfolio, bar_date, bar_close,
                                         "disaster_stop")
                    recent_wins.append(0)
                    bars_in_position = 0
                    self._record_equity(portfolio, bar_date, bar_close)
                    continue

                # Max-hold exit: after MAX_HOLD_BARS (48 bars = 4 hours)
                if bars_in_position >= MAX_HOLD_BARS:
                    self._close_position(portfolio, bar_date, bar_close,
                                         "max_hold")
                    recent_wins.append(1 if portfolio.closed_trades[-1].pnl > 0 else 0)
                    bars_in_position = 0
                    self._record_equity(portfolio, bar_date, bar_close)
                    continue

                # Signal-decay: exit when prediction reverses direction
                # Only after minimum hold period (parity with paper_trader)
                if bars_in_position >= min_hold_bars:
                    if pos.direction == "LONG" and expected_return <= 0:
                        self._close_position(portfolio, bar_date, bar_close,
                                             "signal_decay")
                        recent_wins.append(1 if portfolio.closed_trades[-1].pnl > 0 else 0)
                        bars_in_position = 0
                        reentry_cooldown_remaining = reentry_cooldown_bars
                    elif pos.direction == "SHORT" and expected_return >= 0:
                        self._close_position(portfolio, bar_date, bar_close,
                                             "signal_decay")
                        recent_wins.append(1 if portfolio.closed_trades[-1].pnl > 0 else 0)
                        bars_in_position = 0
                        reentry_cooldown_remaining = reentry_cooldown_bars

                # Cooldown check after close
                if len(recent_wins) == 20 and cooldown_until is None:
                    wr = sum(recent_wins) / 20
                    if wr < 0.5:
                        cooldown_until = ts + _td(days=7)
                        log.info("Rolling WR %.0f%% < 50%% at %s → cooldown 7d",
                                 wr * 100, ts)

                self._record_equity(portfolio, bar_date, bar_close)
                continue

            # --- Entry logic (no position) ---
            if is_last:
                self._record_equity(portfolio, bar_date, bar_close)
                continue

            # Re-entry cooldown after signal_decay (parity with paper_trader)
            if reentry_cooldown_remaining > 0:
                reentry_cooldown_remaining -= 1
                self._record_equity(portfolio, bar_date, bar_close)
                continue

            # Cooldown gate
            if cooldown_until is not None:
                if ts < cooldown_until:
                    cooldown_skipped += 1
                    self._record_equity(portfolio, bar_date, bar_close)
                    continue
                else:
                    cooldown_until = None

            # Entry signal
            enter_dir = None
            if expected_return > crypto_ct:
                enter_dir = "LONG"
            elif expected_return < -crypto_ct:
                enter_dir = "SHORT"

            if enter_dir is not None:
                # Cost-adjusted entry: only trade if expected return > cost
                from cost_model import get_symbol_costs, simulate_fill
                sym_costs = get_symbol_costs(self.symbol)
                rt_cost_bps = (sym_costs.half_spread_bps + sym_costs.slippage_bps) * 2
                rt_cost_pct = rt_cost_bps / 10000.0
                cost_adjusted_return = abs(expected_return) - rt_cost_pct
                if cost_adjusted_return <= 0:
                    self._record_equity(portfolio, bar_date, bar_close)
                    continue

                # Position sizing: Kelly-capped, max 25% of equity per trade
                equity = self._current_equity(portfolio, bar_close)
                signal_pct = min(1.0, max(0.1,
                                          abs(expected_return) / crypto_tr))
                invest_pct = min(self.position_pct * signal_pct, 0.25)  # cap at 25%
                invest = equity * invest_pct

                # Simulate realistic fill with spread + slippage
                fill_side = "buy" if enter_dir == "LONG" else "sell"
                fill_price, _ = simulate_fill(self.symbol, bar_close, fill_side,
                                              stress_mult=self.stress_cost_mult)
                shares = invest / fill_price
                portfolio.cash -= invest
                portfolio.position = Trade(
                    entry_date=bar_date, entry_price=fill_price,
                    direction=enter_dir, size=shares,
                    peak_price=fill_price, atr_at_entry=0.0,
                )
                bars_in_position = 0

            self._record_equity(portfolio, bar_date, bar_close)

        # Close any remaining position
        if portfolio.position is not None and price_rows:
            last_close = price_rows[-1]["close"]
            last_date = price_rows[-1]["date"]
            self._close_position(portfolio, last_date, last_close,
                                 "end_of_backtest")
            self._record_equity(portfolio, last_date, last_close)

        log.info("Crypto intraday: %d trades, %d cooldown-skipped bars.",
                 len(portfolio.closed_trades), cooldown_skipped)

        price_df = pd.DataFrame(price_rows)
        # Deduplicate prices (multiple 5-min bars per day)
        if not price_df.empty:
            price_df = price_df.groupby("date").last().reset_index()

        return self._compute_results(portfolio, price_df)

    def _run_etf_intraday(self, start_date: str,
                           end_date: Optional[str] = None) -> BacktestResult:
        """Bar-level backtest for ETF intraday LGB+GRU model (5-min bars).

        Logic per bar:
          1. Build features via EtfIntradayFeatureEngine
          2. LGB predicts expected 1-hour (12-bar) forward return
          3. If |E[r]| > cost_threshold and no position: enter
          4. Exit on signal decay, max-hold (78 bars), or disaster stop
        Data fetched from Alpaca.
        """
        from etf_intraday_model import (
            EtfIntradayFeatureEngine, FEATURE_NAMES as ETF_INTRADAY_FEATURES,
            FORWARD_BARS, MAX_HOLD_BARS, LOOKBACK_BARS,
            COST_THRESHOLD as ETF_INTRADAY_CT,
            TARGET_RETURN as ETF_INTRADAY_TR,
        )

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp.now(tz="UTC")
        if start_dt.tzinfo is None:
            start_dt = start_dt.tz_localize("UTC")
        if end_dt.tzinfo is None:
            end_dt = end_dt.tz_localize("UTC")

        days_needed = (end_dt - start_dt).days + 30  # extra for feature warmup
        log.info("Fetching %d days of 5-min bars for %s from Alpaca...",
                 days_needed, self.symbol)

        bars = self.adapter.fetch_intraday(self.symbol, "5min",
                                           lookback_days=days_needed)
        if len(bars) < 500:
            log.error("%s: only %d bars, need >= 500", self.symbol, len(bars))
            return self._compute_results(Portfolio(
                initial_capital=self.initial_capital,
                cash=self.initial_capital,
            ))
        log.info("Got %d bars for %s.", len(bars), self.symbol)

        # Fetch SPY bars for sector_rel_strength (skip if symbol IS SPY)
        is_spy = self.symbol.upper() == "SPY"
        spy_bars = None
        if not is_spy:
            spy_bars = self.adapter.fetch_intraday("SPY", "5min",
                                                   lookback_days=days_needed)
            log.info("Got %d SPY reference bars.", len(spy_bars))

        # Build features
        engine = EtfIntradayFeatureEngine()
        features = engine.build_features(bars, spy_bars, symbol=self.symbol)
        if features.empty:
            log.error("No features built for %s.", self.symbol)
            return self._compute_results(Portfolio(
                initial_capital=self.initial_capital,
                cash=self.initial_capital,
            ))
        log.info("Built %d feature rows for %s.", len(features), self.symbol)

        # Load LightGBM model
        model_path = os.path.join(self.model_dir,
                                  f"{self.symbol}_lgb_intraday_etf.joblib")
        config_path = os.path.join(self.model_dir,
                                   f"{self.symbol}_lgb_intraday_etf_config.json")
        if not os.path.exists(model_path):
            log.error("No ETF intraday model for %s at %s. "
                      "Run: python main.py train-intraday --symbols %s",
                      self.symbol, model_path, self.symbol)
            sys.exit(1)

        lgb_model = joblib.load(model_path)
        feature_cols = list(ETF_INTRADAY_FEATURES)

        # Load config for cost/target overrides
        etf_ct = self.cost_threshold
        etf_tr = self.target_return
        etf_pred_scale = 1.0
        etf_vol_normalized = False
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            etf_ct = cfg.get("cost_threshold", ETF_INTRADAY_CT)
            etf_tr = cfg.get("target_return", ETF_INTRADAY_TR)
            etf_pred_scale = cfg.get("pred_scale", 1.0)
            etf_vol_normalized = cfg.get("label_vol_normalized", False)
            saved_features = cfg.get("feature_names")
            if saved_features:
                feature_cols = saved_features
            # OOS contamination check — hard block to prevent polluted backtests
            val_end = cfg.get("val_end")
            if val_end:
                val_end_dt = pd.to_datetime(val_end)
                if val_end_dt.tzinfo is None:
                    val_end_dt = val_end_dt.tz_localize("US/Eastern")
                if start_dt < val_end_dt:
                    raise ValueError(
                        f"OOS CONTAMINATION: backtest start {start_dt.date()} is before "
                        f"model val_end {val_end_dt.date()}. Backtest data would overlap "
                        f"training/validation data. Use --start with a date after "
                        f"{val_end_dt.date()} for true out-of-sample results."
                    )

        # Load multi-horizon models (6-bar, 24-bar)
        lgb_6bar = None
        lgb_24bar = None
        lgb_6_path = os.path.join(self.model_dir, f"{self.symbol}_lgb_intraday_etf_6bar.joblib")
        lgb_24_path = os.path.join(self.model_dir, f"{self.symbol}_lgb_intraday_etf_24bar.joblib")
        if os.path.exists(lgb_6_path):
            lgb_6bar = joblib.load(lgb_6_path)
        if os.path.exists(lgb_24_path):
            lgb_24bar = joblib.load(lgb_24_path)

        # Load pooled model
        from etf_intraday_model import (
            INTRADAY_SYMBOL_TO_POOL, INTRADAY_CLUSTER_SYMBOL_IDS,
            INTRADAY_VOL_LOOKBACK, INTRADAY_VOL_FLOOR,
        )
        etf_pooled_model = None
        etf_pooled_features = None
        pool_name = INTRADAY_SYMBOL_TO_POOL.get(self.symbol.upper())
        if pool_name:
            pool_path = os.path.join(self.model_dir, f"{pool_name}_pool_lgb_intraday.joblib")
            pool_cfg_path = os.path.join(self.model_dir, f"{pool_name}_pool_lgb_intraday_config.json")
            if os.path.exists(pool_path):
                etf_pooled_model = joblib.load(pool_path)
                if os.path.exists(pool_cfg_path):
                    with open(pool_cfg_path) as pcf:
                        etf_pooled_features = json.load(pcf).get("feature_names")

        log.info("Loaded LGB for %s (%d features, ct=%.4f, tr=%.4f, pred_scale=%.2f, "
                 "vol_norm=%s, horizons=%s, pooled=%s).",
                 self.symbol, len(feature_cols), etf_ct, etf_tr, etf_pred_scale,
                 etf_vol_normalized,
                 "+".join(["12"] + (["6"] if lgb_6bar else []) + (["24"] if lgb_24bar else [])),
                 "yes" if etf_pooled_model else "no")

        # Timestamps
        feature_ts = pd.to_datetime(features["ts"])
        if feature_ts.dt.tz is None:
            feature_ts = feature_ts.dt.tz_localize("US/Eastern")

        bars_indexed = bars.set_index("ts")
        if bars_indexed.index.tz is None:
            bars_indexed.index = bars_indexed.index.tz_localize("US/Eastern")

        # --- Walk-forward bar-by-bar ---
        from collections import deque
        from datetime import timedelta as _td

        portfolio = Portfolio(
            initial_capital=self.initial_capital,
            cash=self.initial_capital,
        )
        recent_wins: deque = deque(maxlen=20)
        cooldown_until = None
        cooldown_skipped = 0
        bars_in_position = 0
        reentry_cooldown_remaining = 0
        price_rows = []

        # Use same min-hold/cooldown as crypto intraday (both are 5-min bar models)
        from risk_config import (
            CRYPTO_INTRADAY_MIN_HOLD_BARS,
            CRYPTO_INTRADAY_SIGNAL_REVERSAL_COOLDOWN_BARS,
        )
        min_hold_bars = CRYPTO_INTRADAY_MIN_HOLD_BARS
        reentry_cooldown_bars = CRYPTO_INTRADAY_SIGNAL_REVERSAL_COOLDOWN_BARS

        for i in range(len(features)):
            ts = feature_ts.iloc[i]
            bar_date = ts.date()

            if ts < start_dt:
                continue
            if ts > end_dt:
                break

            # Get close price from bars
            closest_idx = bars_indexed.index.searchsorted(ts)
            if closest_idx >= len(bars_indexed):
                closest_idx = len(bars_indexed) - 1
            bar_close = float(bars_indexed.iloc[closest_idx]["close"])

            price_rows.append({"date": bar_date, "close": bar_close})

            is_last = (i == len(features) - 1)

            # --- LGB prediction (scaled to match real return magnitude) ---
            try:
                avail_cols = [c for c in feature_cols if c in features.columns]
                x_row = features[avail_cols].iloc[i:i + 1].values.astype(np.float32)
                expected_return = float(lgb_model.predict(x_row)[0]) * etf_pred_scale

                # Multi-horizon consensus
                if lgb_6bar is not None or lgb_24bar is not None:
                    pred_6 = float(lgb_6bar.predict(x_row)[0]) if lgb_6bar else None
                    pred_24 = float(lgb_24bar.predict(x_row)[0]) if lgb_24bar else None
                    if pred_6 is not None and pred_24 is not None:
                        signs = [np.sign(pred_6), np.sign(expected_return), np.sign(pred_24)]
                        agreement = sum(1 for s in signs if s == signs[0])
                        if agreement == 3:
                            expected_return = 0.25 * pred_6 + 0.50 * expected_return + 0.25 * pred_24
                        elif agreement >= 2:
                            expected_return *= 0.5
                        else:
                            expected_return *= 0.3
                    elif pred_6 is not None and np.sign(pred_6) == np.sign(expected_return):
                        expected_return = 0.40 * pred_6 + 0.60 * expected_return

                # Pooled model blend (70/30)
                if etf_pooled_model is not None and etf_pooled_features:
                    sym_id = INTRADAY_CLUSTER_SYMBOL_IDS.get(
                        pool_name or "", {}).get(self.symbol.upper(), 0)
                    pool_row = features[avail_cols].iloc[i:i + 1].copy()
                    pool_row["symbol_id"] = sym_id
                    pool_avail = [c for c in etf_pooled_features if c in pool_row.columns]
                    x_pool = pool_row[pool_avail].values.astype(np.float32)
                    pooled_pred = float(etf_pooled_model.predict(x_pool)[0])
                    expected_return = 0.70 * expected_return + 0.30 * pooled_pred

                # Vol-denormalization
                if etf_vol_normalized:
                    close_s = bars_indexed["close"].astype(float)
                    log_ret_s = np.log(close_s / close_s.shift(1))
                    cur_vol = log_ret_s.rolling(INTRADAY_VOL_LOOKBACK).std().iloc[
                        min(closest_idx, len(close_s) - 1)]
                    if np.isnan(cur_vol) or cur_vol < INTRADAY_VOL_FLOOR:
                        cur_vol = INTRADAY_VOL_FLOOR
                    expected_return = expected_return * cur_vol

            except Exception:
                expected_return = 0.0

            # --- Exit logic ---
            if portfolio.position is not None:
                pos = portfolio.position
                bars_in_position += 1

                if pos.direction == "LONG":
                    unrealized_pct = (bar_close - pos.entry_price) / pos.entry_price
                else:
                    unrealized_pct = (pos.entry_price - bar_close) / pos.entry_price

                # Disaster stop: 1.5% for ETFs (tighter than crypto 2%)
                if unrealized_pct <= -0.015:
                    self._close_position(portfolio, bar_date, bar_close,
                                         "disaster_stop")
                    recent_wins.append(0)
                    bars_in_position = 0
                    self._record_equity(portfolio, bar_date, bar_close)
                    continue

                # Max-hold exit: after MAX_HOLD_BARS (78 bars = full trading day)
                if bars_in_position >= MAX_HOLD_BARS:
                    self._close_position(portfolio, bar_date, bar_close,
                                         "max_hold")
                    recent_wins.append(1 if portfolio.closed_trades[-1].pnl > 0 else 0)
                    bars_in_position = 0
                    self._record_equity(portfolio, bar_date, bar_close)
                    continue

                # Signal-decay: exit when prediction reverses direction
                # Only after minimum hold period (parity with paper_trader)
                if bars_in_position >= min_hold_bars:
                    if pos.direction == "LONG" and expected_return <= 0:
                        self._close_position(portfolio, bar_date, bar_close,
                                             "signal_decay")
                        recent_wins.append(1 if portfolio.closed_trades[-1].pnl > 0 else 0)
                        bars_in_position = 0
                        reentry_cooldown_remaining = reentry_cooldown_bars
                    elif pos.direction == "SHORT" and expected_return >= 0:
                        self._close_position(portfolio, bar_date, bar_close,
                                             "signal_decay")
                        recent_wins.append(1 if portfolio.closed_trades[-1].pnl > 0 else 0)
                        bars_in_position = 0
                        reentry_cooldown_remaining = reentry_cooldown_bars

                # Cooldown check after close
                if len(recent_wins) == 20 and cooldown_until is None:
                    wr = sum(recent_wins) / 20
                    if wr < 0.5:
                        cooldown_until = ts + _td(days=7)
                        log.info("Rolling WR %.0f%% < 50%% at %s -> cooldown 7d",
                                 wr * 100, ts)

                self._record_equity(portfolio, bar_date, bar_close)
                continue

            # --- Entry logic (no position) ---
            if is_last:
                self._record_equity(portfolio, bar_date, bar_close)
                continue

            # Re-entry cooldown after signal_decay (parity with paper_trader)
            if reentry_cooldown_remaining > 0:
                reentry_cooldown_remaining -= 1
                self._record_equity(portfolio, bar_date, bar_close)
                continue

            # Cooldown gate
            if cooldown_until is not None:
                if ts < cooldown_until:
                    cooldown_skipped += 1
                    self._record_equity(portfolio, bar_date, bar_close)
                    continue
                else:
                    cooldown_until = None

            # Entry signal
            enter_dir = None
            if expected_return > etf_ct:
                enter_dir = "LONG"
            elif expected_return < -etf_ct:
                enter_dir = "SHORT"

            if enter_dir is not None:
                equity = self._current_equity(portfolio, bar_close)
                signal_pct = min(1.0, max(0.1,
                                          abs(expected_return) / etf_tr))
                invest_pct = min(self.position_pct * signal_pct, 0.98)
                invest = equity * invest_pct
                shares = invest / bar_close
                portfolio.cash -= invest
                portfolio.position = Trade(
                    entry_date=bar_date, entry_price=bar_close,
                    direction=enter_dir, size=shares,
                    peak_price=bar_close, atr_at_entry=0.0,
                )
                bars_in_position = 0

            self._record_equity(portfolio, bar_date, bar_close)

        # Close any remaining position
        if portfolio.position is not None and price_rows:
            last_close = price_rows[-1]["close"]
            last_date = price_rows[-1]["date"]
            self._close_position(portfolio, last_date, last_close,
                                 "end_of_backtest")
            self._record_equity(portfolio, last_date, last_close)

        log.info("ETF intraday: %d trades, %d cooldown-skipped bars.",
                 len(portfolio.closed_trades), cooldown_skipped)

        price_df = pd.DataFrame(price_rows)
        if not price_df.empty:
            price_df = price_df.groupby("date").last().reset_index()

        return self._compute_results(portfolio, price_df)

    def run(self, start_date: str, end_date: Optional[str] = None,
            seq_len: int = SEQ_LEN) -> BacktestResult:
        """Run the walk-forward backtest with optimized strategy."""
        # Intraday LightGBM uses a separate day-level simulation
        if self.model_type == "intraday":
            return self._run_intraday_lgb(start_date, end_date)
        if self.model_type == "crypto_intraday":
            return self._run_crypto_intraday(start_date, end_date)
        if self.model_type == "etf_intraday":
            return self._run_etf_intraday(start_date, end_date)

        # Fetch data
        if self.mode == "daily":
            lookback = 1000
            log.info("Fetching daily data for %s...", self.symbol)
            bars = self.adapter.fetch_daily(self.symbol, lookback)
        else:
            start_dt_temp = pd.to_datetime(start_date).date()
            end_dt_temp = (pd.to_datetime(end_date).date() if end_date
                          else datetime.now(timezone.utc).date())
            days_needed = (end_dt_temp - start_dt_temp).days + 60
            lookback = days_needed
            log.info("Fetching %d days of %s intraday data for %s...",
                     days_needed, self.intraday_interval, self.symbol)
            bars = self.adapter.fetch_intraday(
                self.symbol, self.intraday_interval, lookback_days=days_needed)

        log.info("Got %d bars.", len(bars))

        vix_df = _fetch_vix_for_training(self.fred_key, lookback_days=max(lookback, 500))
        log.info("Got %d VIX rows.", len(vix_df))

        # Build features and load model — varies by model_type
        model = None       # LSTM/PatchTST nn.Module (None for XGBoost)
        xgb_model = None   # XGBoost regressor (None for LSTM/PatchTST)
        spy_bars = None    # SPY bars for feature engineering (fetched by swing, None for LSTM)
        effective_seq_len = seq_len  # may change for PatchTST

        if self.model_type == "swing":
            from swing_model import SwingFeatureEngine, get_swing_feature_cols
            engine = SwingFeatureEngine()
            scaler_path = os.path.join(self.model_dir, f"{self.symbol}_xgb_swing_scaler.json")
            if not os.path.exists(scaler_path):
                log.error("No swing XGBoost scaler for %s. Train first: "
                          "python main.py train-swing --symbols %s", self.symbol, self.symbol)
                sys.exit(1)
            engine.load_scaler(scaler_path)
            spy_bars = self.adapter.fetch_daily("SPY", lookback) if self.mode == "daily" else None
            features = engine.build_features(bars, vix_df, spy_bars=spy_bars, symbol=self.symbol)
            features_norm = engine.transform(features)
            log.info("Built %d swing feature rows (XGBoost).", len(features_norm))

            model_path = os.path.join(self.model_dir, f"{self.symbol}_xgb_swing.joblib")
            if not os.path.exists(model_path):
                log.error("No swing XGBoost model for %s. Train first: "
                          "python main.py train-swing --symbols %s", self.symbol, self.symbol)
                sys.exit(1)
            xgb_model = joblib.load(model_path)
            effective_seq_len = 1  # XGBoost point-in-time, no sequence
            log.info("Loaded XGBoost swing for %s (%d features).",
                     self.symbol, len(get_swing_feature_cols(self.symbol)))

            # Check if model was trained with vol-normalized labels
            swing_vol_normalized = False
            swing_feature_names = None  # feature names from config (for correct column selection)
            cfg_path = os.path.join(self.model_dir, f"{self.symbol}_xgb_swing_config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as _cf:
                    _swing_cfg = json.load(_cf)
                swing_vol_normalized = _swing_cfg.get("label_vol_normalized", False)
                swing_feature_names = _swing_cfg.get("feature_names")
                if swing_vol_normalized:
                    log.info("Vol-normalized labels detected — will denormalize predictions.")

            # Load pooled cluster model if available (blend 70/30 with per-symbol)
            # Load 5d short-horizon model if available
            xgb_5d_model = None
            xgb_5d_path = os.path.join(self.model_dir, f"{self.symbol}_xgb_swing_5d.joblib")
            if os.path.exists(xgb_5d_path):
                xgb_5d_model = joblib.load(xgb_5d_path)
                log.info("Loaded 5d short-horizon XGBoost for multi-horizon blend.")

            pooled_xgb = None
            pooled_feature_names = None
            from swing_model import SYMBOL_TO_POOL
            if self.symbol in SYMBOL_TO_POOL:
                pool_name = SYMBOL_TO_POOL[self.symbol]
                pooled_path = os.path.join(self.model_dir, f"{pool_name}_pool_xgb_10d.joblib")
                pooled_cfg_path = os.path.join(self.model_dir, f"{pool_name}_pool_xgb_10d_config.json")
                if os.path.exists(pooled_path):
                    pooled_xgb = joblib.load(pooled_path)
                    if os.path.exists(pooled_cfg_path):
                        with open(pooled_cfg_path) as _pcf:
                            pooled_feature_names = json.load(_pcf).get("feature_names")
                    log.info("Loaded pooled %s model for ensemble.", pool_name)

        else:  # "lstm" (default)
            engine = FeatureEngine()
            suffix = "" if self.mode == "daily" else f"_{self.intraday_interval}"
            scaler_path = os.path.join(self.model_dir, f"{self.symbol}_scaler{suffix}.json")
            if not os.path.exists(scaler_path):
                log.error("No scaler for %s (%s mode). Train first: "
                          "python main.py train --symbol %s%s",
                          self.symbol, self.mode, self.symbol,
                          f" --mode {self.mode}" if self.mode != "daily" else "")
                sys.exit(1)
            engine.load_scaler(scaler_path)
            features = engine.build_features(bars, vix_df, mode=self.mode, symbol=self.symbol)
            features_norm = engine.transform(features)
            log.info("Built %d feature rows.", len(features_norm))

            # Load primary LSTM model
            weights_path = os.path.join(self.model_dir, f"{self.symbol}_lstm{suffix}.pt")
            if not os.path.exists(weights_path):
                log.error("No LSTM weights for %s (%s mode). Train first: "
                          "python main.py train --symbol %s%s",
                          self.symbol, self.mode, self.symbol,
                          f" --mode {self.mode}" if self.mode != "daily" else "")
                sys.exit(1)
            model = DirectionLSTM(n_features=len(get_feature_cols(self.mode, self.symbol)))
            model.load_state_dict(
                torch.load(weights_path, map_location="cpu", weights_only=True))
            model.eval()

        # Pre-compute indicators for v2 strategy
        close_s = bars["close"].astype(float)
        high_s = bars["high"].astype(float)
        low_s = bars["low"].astype(float)
        atr_series = compute_atr(high_s, low_s, close_s, period=14)
        trend_signal = self._compute_trend_signal(close_s)

        # Date filtering
        bar_timestamps = pd.to_datetime(bars["ts"])
        bar_dates = bar_timestamps.dt.date
        start_dt = pd.to_datetime(start_date).date()
        end_dt = pd.to_datetime(end_date).date() if end_date else datetime.now(timezone.utc).date()

        # --- Regime filter: hardcoded SMA/VIX gates removed ---
        # Model features (spy_above_sma200, spy_trend_strength, vix_regime, etc.)
        # handle regime-dependent behavior. Rolling win rate cooldown stays active.
        log.info("Hardcoded regime gates disabled for swing — model features drive entry decisions.")

        # Rolling 20-trade win rate cooldown
        from collections import deque
        from datetime import timedelta as _td
        swing_recent_wins: deque = deque(maxlen=20)
        swing_cooldown_until = None
        swing_cooldown_skipped = 0

        portfolio = Portfolio(
            initial_capital=self.initial_capital,
            cash=self.initial_capital,
        )

        # Classify vol tier for this symbol using the price data we already have
        from risk_config import (classify_vol_tier, compute_vol_metrics,
                                 get_exit_params, VolTier, ExitVariant)
        try:
            _ohlc_df = pd.DataFrame({
                "High": bars["high"] if "high" in bars.columns else bars["High"],
                "Low": bars["low"] if "low" in bars.columns else bars["Low"],
                "Close": bars["close"] if "close" in bars.columns else bars["Close"],
            })
            _atr_ratio, _vol20 = compute_vol_metrics(_ohlc_df)
            _vol_tier = classify_vol_tier(_atr_ratio, _vol20)
        except Exception:
            _vol_tier = VolTier.HIGH
            _atr_ratio, _vol20 = 0.0, 0.0

        # Determine group for exit params: crypto if symbol ends with -USD, else swing
        _bt_group = "crypto" if self.symbol.endswith("-USD") else "swing"
        _exit_variant = ExitVariant(self.exit_variant) if self.exit_variant else None
        _exit_params = get_exit_params(_bt_group, _vol_tier, variant=_exit_variant)
        self._result_vol_tier = _vol_tier.value
        self._result_bt_group = _bt_group
        log.info("Backtest vol tier: %s (ATR/P=%.2f%%, Vol20=%.1f%%, group=%s, variant=%s)",
                 _vol_tier.value, _atr_ratio * 100, _vol20 * 100, _bt_group,
                 self.exit_variant or "base")

        feature_indices = features_norm.index.tolist()
        log.info("Walking %d bars from %s to %s...", len(feature_indices), start_dt, end_dt)

        last_bar_idx = len(feature_indices) - 1
        for idx_pos in range(effective_seq_len, len(feature_indices)):
            feat_idx = feature_indices[idx_pos]
            bar_date = bar_dates.iloc[feat_idx]
            bar_time = bar_timestamps.iloc[feat_idx] if self.mode != "daily" else None

            if bar_date < start_dt:
                continue
            if bar_date > end_dt:
                break

            is_last_bar = (idx_pos == last_bar_idx or
                           (idx_pos + 1 < len(feature_indices) and
                            bar_dates.iloc[feature_indices[idx_pos + 1]] > end_dt))

            bar_close = float(close_s.iloc[feat_idx])
            bar_atr = float(atr_series.iloc[feat_idx]) if not np.isnan(atr_series.iloc[feat_idx]) else bar_close * 0.02
            bar_trend = float(trend_signal.iloc[feat_idx]) if not np.isnan(trend_signal.iloc[feat_idx]) else 0.0

            # --- Get prediction (v2 regression: expected return) ---
            if xgb_model is not None:
                if swing_feature_names:
                    avail = [c for c in swing_feature_names if c in features_norm.columns]
                    x_row = features_norm[avail].iloc[idx_pos:idx_pos + 1].values.astype(np.float32)
                else:
                    x_row = features_norm.iloc[idx_pos:idx_pos + 1].values.astype(np.float32)
                expected_return = float(xgb_model.predict(x_row)[0])
                # Multi-horizon blend: 5d + 10d (60/40 when agreeing)
                if xgb_5d_model is not None:
                    try:
                        ret_5d = float(xgb_5d_model.predict(x_row)[0])
                        if np.sign(ret_5d) == np.sign(expected_return):
                            expected_return = 0.60 * expected_return + 0.40 * ret_5d
                    except Exception:
                        pass
                # Blend with pooled cluster model (70/30)
                if pooled_xgb is not None:
                    try:
                        if pooled_feature_names:
                            pcols = [c for c in pooled_feature_names if c in features_norm.columns]
                            px = features_norm[pcols].iloc[idx_pos:idx_pos + 1].values.astype(np.float32)
                        else:
                            px = x_row
                        pooled_ret = float(pooled_xgb.predict(px)[0])
                        expected_return = 0.70 * expected_return + 0.30 * pooled_ret
                    except Exception:
                        pass  # fall back to per-symbol only
                # Vol-denormalize: multiply by current trailing 20-day realized vol
                if swing_vol_normalized:
                    vol_floor = 0.005 / np.sqrt(252)
                    lookback_start = max(0, feat_idx - 20)
                    trail_rets = close_s.iloc[lookback_start:feat_idx].pct_change().dropna()
                    cur_vol = trail_rets.std() if len(trail_rets) >= 5 else vol_floor
                    if np.isnan(cur_vol) or cur_vol < vol_floor:
                        cur_vol = vol_floor
                    expected_return = expected_return * cur_vol
            else:
                window_start = idx_pos - effective_seq_len
                window = features_norm.iloc[window_start:idx_pos].values
                x = torch.FloatTensor(window).unsqueeze(0)
                with torch.no_grad():
                    expected_return = model(x).item()

            # --- Exit logic (tier-based: safety > breakeven > profit-lock > signal decay > time) ---
            if portfolio.position is not None:
                pos = portfolio.position
                ep = _exit_params

                if pos.direction == "LONG":
                    unrealized_pct = (bar_close - pos.entry_price) / pos.entry_price
                else:
                    unrealized_pct = (pos.entry_price - bar_close) / pos.entry_price

                # MFE/MAE tracking using High/Low for intrabar extremes
                bar_high = float(high_s.iloc[feat_idx])
                bar_low = float(low_s.iloc[feat_idx])
                if pos.direction == "LONG":
                    bar_best = (bar_high - pos.entry_price) / pos.entry_price
                    bar_worst = (bar_low - pos.entry_price) / pos.entry_price
                else:
                    bar_best = (pos.entry_price - bar_low) / pos.entry_price
                    bar_worst = (pos.entry_price - bar_high) / pos.entry_price
                pos.mfe_pct = max(pos.mfe_pct, bar_best)
                pos.mae_pct = min(pos.mae_pct, bar_worst)

                # Track peak price
                if pos.direction == "LONG":
                    pos.peak_price = max(pos.peak_price, bar_close)
                    drawdown_from_peak = (pos.peak_price - bar_close) / pos.peak_price if pos.peak_price > 0 else 0
                else:
                    pos.peak_price = min(pos.peak_price, bar_close) if pos.peak_price > 0 else bar_close
                    drawdown_from_peak = (bar_close - pos.peak_price) / pos.peak_price if pos.peak_price > 0 else 0

                # LAYER 1: Disaster stop (fixed % or ATR, whichever is tighter)
                if ep.use_atr and pos.atr_at_entry > 0:
                    disaster_pct = min(ep.disaster_stop_pct,
                                       ep.disaster_atr_mult * pos.atr_at_entry / pos.entry_price)
                else:
                    disaster_pct = ep.disaster_stop_pct
                if unrealized_pct <= -disaster_pct:
                    pos.exit_layer = "safety"
                    self._close_position(portfolio, bar_date, bar_close, "disaster_stop", bar_time=bar_time)
                    self._record_equity(portfolio, bar_date, bar_close)
                    continue

                # Breakeven ratchet: arm when profitable past threshold
                if unrealized_pct >= ep.breakeven_ratchet_pct:
                    pos.breakeven_armed = True
                if pos.breakeven_armed and unrealized_pct <= 0:
                    pos.exit_layer = "breakeven"
                    self._close_position(portfolio, bar_date, bar_close, "breakeven_ratchet", bar_time=bar_time)
                    self._record_equity(portfolio, bar_date, bar_close)
                    continue

                # LAYER 1.5: Strong opposing signal override
                # If model predicts >2x cost_threshold in opposing direction, exit immediately
                strong_opposing = ((pos.direction == "LONG" and expected_return < -2 * self.cost_threshold)
                                   or (pos.direction == "SHORT" and expected_return > 2 * self.cost_threshold))
                if strong_opposing and pos.bars_held >= 2:
                    pos.exit_layer = "signal_override"
                    self._close_position(portfolio, bar_date, bar_close,
                                         "strong_signal_override", bar_time=bar_time)
                    if portfolio.closed_trades:
                        swing_recent_wins.append(1 if portfolio.closed_trades[-1].pnl > 0 else 0)
                    self._record_equity(portfolio, bar_date, bar_close)
                    continue

                # LAYER 2: Profit-lock (arm + trail)
                if ep.use_atr and pos.atr_at_entry > 0:
                    arm_pct = min(ep.profit_lock_arm_pct,
                                  ep.arm_atr_mult * pos.atr_at_entry / pos.entry_price)
                    trail_pct = min(ep.profit_lock_trail_pct,
                                    ep.trail_atr_mult * pos.atr_at_entry / pos.entry_price)
                else:
                    arm_pct = ep.profit_lock_arm_pct
                    trail_pct = ep.profit_lock_trail_pct

                if unrealized_pct >= arm_pct:
                    pos.tp_activated = True

                min_hold = get_effective_min_hold(
                    ep.min_hold_bars,
                    pos.atr_at_entry,
                    bar_atr,
                )
                next_bars_held = pos.bars_held + 1

                if pos.tp_activated:
                    # 2a: Armed + model flips + still profitable -> immediate exit
                    signal_flipped = ((pos.direction == "LONG" and expected_return <= 0)
                                      or (pos.direction == "SHORT" and expected_return >= 0))
                    if signal_flipped and unrealized_pct > 0 and next_bars_held >= min_hold:
                        pos.exit_layer = "profit_lock"
                        self._close_position(portfolio, bar_date, bar_close,
                                             "profit_lock+signal_decay", bar_time=bar_time)
                        self._record_equity(portfolio, bar_date, bar_close)
                        continue
                    # 2b: Trail from peak
                    if drawdown_from_peak >= trail_pct:
                        pos.exit_layer = "profit_lock"
                        self._close_position(portfolio, bar_date, bar_close,
                                             "profit_lock_trail", bar_time=bar_time)
                        self._record_equity(portfolio, bar_date, bar_close)
                        continue

                # LAYER 3: Signal decay (consecutive flip requirement)
                pos.bars_held = next_bars_held

                signal_flipped = ((pos.direction == "LONG" and expected_return <= 0)
                                  or (pos.direction == "SHORT" and expected_return >= 0))
                if signal_flipped:
                    pos.signal_flip_count += 1
                else:
                    pos.signal_flip_count = 0

                if pos.bars_held >= min_hold and pos.signal_flip_count >= ep.signal_flip_consecutive:
                    pos.exit_layer = "signal_decay"
                    self._close_position(portfolio, bar_date, bar_close, "signal_decay", bar_time=bar_time)
                    if portfolio.closed_trades:
                        swing_recent_wins.append(1 if portfolio.closed_trades[-1].pnl > 0 else 0)
                        if len(swing_recent_wins) == 20 and swing_cooldown_until is None:
                            wr = sum(swing_recent_wins) / 20
                            if wr < 0.5:
                                swing_cooldown_until = bar_date + _td(days=7)
                    self._record_equity(portfolio, bar_date, bar_close)
                    continue

                # LAYER 4: Max underwater duration
                if portfolio.position is not None and ep.max_underwater_days > 0:
                    pos = portfolio.position
                    if pos.direction == "LONG":
                        uw_pct = (bar_close - pos.entry_price) / pos.entry_price
                    else:
                        uw_pct = (pos.entry_price - bar_close) / pos.entry_price
                    if uw_pct < 0:
                        try:
                            days_held = (bar_date - pos.entry_date).days
                        except TypeError:
                            days_held = 0
                        if days_held >= ep.max_underwater_days:
                            pos.exit_layer = "time"
                            self._close_position(portfolio, bar_date, bar_close,
                                                 f"max_underwater_{ep.max_underwater_days}d",
                                                 bar_time=bar_time)
                            self._record_equity(portfolio, bar_date, bar_close)
                            continue

            # --- Entry logic (v2: trend filter + signal-proportional sizing) ---
            if portfolio.position is None and not is_last_bar:
                # Rolling win rate cooldown
                if swing_cooldown_until is not None:
                    if bar_date < swing_cooldown_until:
                        self._record_equity(portfolio, bar_date, bar_close)
                        swing_cooldown_skipped += 1
                        continue
                    else:
                        swing_cooldown_until = None

                enter_dir = None
                if expected_return > self.cost_threshold:
                    enter_dir = "LONG"
                elif expected_return < -self.cost_threshold:
                    enter_dir = "SHORT"

                if enter_dir is not None:
                    self._open_position(
                        portfolio, bar_date, bar_close, enter_dir,
                        expected_return=expected_return, bar_atr=bar_atr,
                        bar_time=bar_time,
                    )

            self._record_equity(portfolio, bar_date, bar_close)

        log.info("Swing cooldown skipped %d days.", swing_cooldown_skipped)

        # Close any open position at end
        if portfolio.position is not None:
            last_idx = feature_indices[-1]
            last_close = float(close_s.iloc[last_idx])
            last_date = bar_dates.iloc[last_idx]
            last_time = bar_timestamps.iloc[last_idx] if self.mode != "daily" else None
            portfolio.position.exit_layer = "time"
            self._close_position(portfolio, last_date, last_close, "end_of_backtest",
                                 bar_time=last_time)
            self._record_equity(portfolio, last_date, last_close)

        # Build price series for charts
        price_rows = []
        for idx_pos in range(seq_len, len(feature_indices)):
            feat_idx = feature_indices[idx_pos]
            bd = bar_dates.iloc[feat_idx]
            if bd < start_dt:
                continue
            if bd > end_dt:
                break
            price_rows.append({
                "date": bd,
                "close": float(close_s.iloc[feat_idx]),
            })
        price_df = pd.DataFrame(price_rows)

        return self._compute_results(portfolio, price_df)

    def _determine_session(self, bar_time) -> str:
        """Determine trading session from bar timestamp.

        Returns 'regular' for 9:30-16:00 ET, 'extended' for 4:00-9:30 and
        16:00-20:00 ET.  Daily-mode bars always return 'regular'.
        """
        if self.mode == "daily" or bar_time is None:
            return "regular"
        try:
            from zoneinfo import ZoneInfo
            et = ZoneInfo("America/New_York")
            if hasattr(bar_time, 'tzinfo') and bar_time.tzinfo is not None:
                bar_et = bar_time.astimezone(et)
            else:
                bar_et = bar_time.replace(tzinfo=et)
            t = bar_et.time()
            from datetime import time as _time
            if _time(9, 30) <= t < _time(16, 0):
                return "regular"
            return "extended"
        except Exception:
            return "regular"

    def _apply_fill_cost(self, price: float, side: str,
                         bar_time=None) -> tuple[float, bool]:
        """Apply spread + slippage cost model to a price.

        Returns (fill_price, filled).  When the cost model is disabled or
        unavailable, filled is always True.
        """
        if not self.use_cost_model:
            return price, True
        try:
            from cost_model import simulate_fill
            session = self._determine_session(bar_time)
            fill_price, filled = simulate_fill(
                self.symbol, price, side,
                session=session,
                stress_mult=self.stress_cost_mult,
            )
            return fill_price, filled
        except ImportError:
            return price, True

    @staticmethod
    def _hours_since(then, now) -> float:
        try:
            return max(0.0, (pd.Timestamp(now) - pd.Timestamp(then)).total_seconds() / 3600.0)
        except Exception:
            return float("inf")

    def _open_position(self, portfolio: Portfolio, date, price: float,
                       direction: str = "LONG", expected_return: float = 0.0,
                       bar_atr: float = 0.0, bar_time=None) -> bool:
        """Signal-proportional position sizing (v2).

        Size = clip(abs(E[r]) / target_return, 0.1, 1.0) * base_allocation
        Applies cost model for realistic fill prices.
        Returns True if position was opened, False if fill failed.
        """
        equity = self._current_equity(portfolio, price)

        # Signal-proportional sizing (v3: size from total equity, cap by max_position_pct)
        risk = get_risk_config(self._risk_group)
        if self._max_loss_exit is not None:
            loss_age_h = self._hours_since(self._max_loss_exit["time"], date)
            if loss_age_h < risk.max_loss_cooldown_hours:
                return False
            if loss_age_h < 24 and direction == self._max_loss_exit["direction"]:
                effective_threshold = self.cost_threshold * risk.same_dir_confidence_mult
                if abs(expected_return) < effective_threshold:
                    return False
        signal_pct = min(1.0, max(risk.min_signal_scale, abs(expected_return) / self.target_return))
        kelly_f = self._derisk_state.half_kelly(
            window=risk.kelly_window,
            min_trades=risk.kelly_min_trades,
        )
        if kelly_f is not None:
            effective_cap = risk.kelly_cap * risk.cross_group_kelly_discount
            base_frac = min(kelly_f * risk.cross_group_kelly_discount, effective_cap)
        else:
            base_frac = min(self.position_pct, risk.position_pct)

        base_pct = base_frac * signal_pct

        # Inverse-vol sizing: scale inversely with realized vol
        # Target vol = 15% annualized. Higher vol → smaller position.
        if bar_atr > 0 and price > 0:
            realized_vol_ann = (bar_atr / price) * np.sqrt(252)
            target_vol = 0.15
            inv_vol_mult = min(2.0, max(0.3, target_vol / max(realized_vol_ann, 0.05)))
            base_pct *= inv_vol_mult

        base_pct = min(base_pct, risk.max_position_pct)
        base_pct = min(base_pct, get_symbol_cap(self.symbol, self._risk_group))
        # conf_mult and rank_scalar removed — sym_cap is the single OOS-based
        # sizing factor (avoids redundant stacking of OOS Sharpe multipliers)
        derisk_action, _ = evaluate_derisk(self._derisk_state, window=50)
        if derisk_action == "disable":
            return False
        if derisk_action == "halfsize":
            base_pct *= 0.5
        if base_pct <= 0:
            return False

        invest = equity * base_pct
        # Apply cost model to entry price
        fill_price, filled = self._apply_fill_cost(price, direction,
                                                   bar_time=bar_time)
        if not filled or fill_price <= 0:
            log.debug("Fill failed for %s %s at %s — skipping entry (fill_price=%.6f)",
                      direction, self.symbol, date, fill_price)
            return False
        shares = invest / fill_price
        portfolio.cash -= invest
        portfolio.position = Trade(
            entry_date=date, entry_price=fill_price,
            direction=direction, size=shares,
            peak_price=fill_price, atr_at_entry=bar_atr,
        )
        return True

    def _close_position(self, portfolio: Portfolio, date, price: float,
                        reason: str, bar_time=None) -> None:
        trade = portfolio.position
        trade.exit_date = date
        # Apply cost model to exit price
        exit_side = "SELL" if trade.direction == "LONG" else "BUY"
        fill_price, _filled = self._apply_fill_cost(price, exit_side,
                                                    bar_time=bar_time)
        trade.exit_price = fill_price
        trade.exit_reason = reason
        if trade.direction == "LONG":
            trade.pnl = trade.size * (fill_price - trade.entry_price)
            proceeds = trade.size * fill_price
        else:  # SHORT
            trade.pnl = trade.size * (trade.entry_price - fill_price)
            proceeds = trade.size * trade.entry_price + trade.pnl
        portfolio.cash += proceeds
        portfolio.closed_trades.append(trade)
        if trade.entry_price > 0 and trade.size > 0:
            pnl_pct = trade.pnl / (trade.size * trade.entry_price)
            self._derisk_state.record_trade(float(pnl_pct))
        if reason == "disaster_stop":
            self._max_loss_exit = {"time": date, "direction": trade.direction}
        portfolio.position = None

    def _current_equity(self, portfolio: Portfolio, current_price: float) -> float:
        equity = portfolio.cash
        if portfolio.position is not None:
            pos = portfolio.position
            if pos.direction == "LONG":
                equity += pos.size * current_price
            else:  # SHORT
                equity += pos.size * pos.entry_price + pos.size * (pos.entry_price - current_price)
        return equity

    def _record_equity(self, portfolio: Portfolio, date, price: float) -> None:
        portfolio.equity_curve.append({
            "date": date,
            "equity": self._current_equity(portfolio, price),
        })

    def _compute_results(self, portfolio: Portfolio,
                         price_df: pd.DataFrame = None) -> BacktestResult:
        eq_df = pd.DataFrame(portfolio.equity_curve)

        if price_df is None:
            price_df = pd.DataFrame(columns=["date", "close"])

        if eq_df.empty:
            return BacktestResult(
                symbol=self.symbol, start_date="N/A", end_date="N/A",
                initial_capital=self.initial_capital, final_equity=self.initial_capital,
                total_return_pct=0.0, annualized_return_pct=0.0, sharpe_ratio=0.0,
                max_drawdown_pct=0.0, total_trades=0, win_rate=0.0,
                avg_win_pct=0.0, avg_loss_pct=0.0, profit_factor=0.0,
                avg_trade_duration_days=0.0, equity_curve=eq_df,
                price_series=price_df,
            )

        final_equity = eq_df["equity"].iloc[-1]
        total_return = (final_equity / self.initial_capital) - 1

        # Daily returns for Sharpe
        eq_df["daily_return"] = eq_df["equity"].pct_change().fillna(0)
        n_rows = len(eq_df)
        # Use actual trading days count; for crypto (365d/yr) vs equity (252d/yr)
        try:
            first_date = pd.Timestamp(eq_df["date"].iloc[0])
            last_date = pd.Timestamp(eq_df["date"].iloc[-1])
            calendar_days = (last_date - first_date).days
            n_years = max(calendar_days / 365.25, 0.01)
        except Exception:
            n_years = max(n_rows / 252, 0.01)
        # Annualization factor: use n_rows for Sharpe scaling
        ann_factor = np.sqrt(n_rows / max(n_years, 0.01)) if n_years > 0 else np.sqrt(252)

        annualized_return = (
            (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
            if total_return > -1 else -1.0
        )

        daily_std = eq_df["daily_return"].std()
        sharpe = (eq_df["daily_return"].mean() / daily_std * ann_factor
                  if daily_std > 0 else 0.0)

        # Max drawdown
        eq_df["peak"] = eq_df["equity"].cummax()
        eq_df["drawdown"] = (eq_df["equity"] - eq_df["peak"]) / eq_df["peak"]
        max_dd = eq_df["drawdown"].min()

        # Trade-level stats
        trades = portfolio.closed_trades
        n_trades = len(trades)
        wins = [t for t in trades if t.pnl is not None and t.pnl > 0]
        losses = [t for t in trades if t.pnl is not None and t.pnl <= 0]
        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0

        avg_win = (np.mean([t.pnl / (t.size * t.entry_price) for t in wins])
                   if wins else 0.0)
        avg_loss = (np.mean([t.pnl / (t.size * t.entry_price) for t in losses])
                    if losses else 0.0)

        gross_profit = sum(t.pnl for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        def _trade_days(t: Trade) -> int:
            try:
                d1 = t.entry_date if not hasattr(t.entry_date, "date") else t.entry_date
                d2 = t.exit_date if not hasattr(t.exit_date, "date") else t.exit_date
                return (pd.Timestamp(d2) - pd.Timestamp(d1)).days
            except Exception:
                return 0

        avg_duration = (np.mean([_trade_days(t) for t in trades])
                        if trades else 0.0)

        return BacktestResult(
            symbol=self.symbol,
            start_date=str(eq_df["date"].iloc[0]),
            end_date=str(eq_df["date"].iloc[-1]),
            initial_capital=self.initial_capital,
            final_equity=round(final_equity, 2),
            total_return_pct=round(total_return * 100, 2),
            annualized_return_pct=round(annualized_return * 100, 2),
            sharpe_ratio=round(sharpe, 3),
            max_drawdown_pct=round(max_dd * 100, 2),
            total_trades=n_trades,
            win_rate=round(win_rate, 3),
            avg_win_pct=round(avg_win * 100, 2),
            avg_loss_pct=round(avg_loss * 100, 2),
            profit_factor=round(profit_factor, 3) if profit_factor != float("inf") else 999.0,
            avg_trade_duration_days=round(avg_duration, 1),
            equity_curve=eq_df[["date", "equity"]],
            price_series=price_df,
            drawdown_series=eq_df[["date", "drawdown"]],
            trades=portfolio.closed_trades,
            mode=self.mode,
            intraday_interval=self.intraday_interval,
            vol_tier=getattr(self, '_result_vol_tier', ''),
            exit_variant=self.exit_variant or '',
            bt_group=getattr(self, '_result_bt_group', ''),
        )


# ===================================================================
# Interactive Charts
# ===================================================================
def generate_charts(result: BacktestResult) -> Optional[str]:
    """Generate an interactive Plotly HTML report with 3 charts."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        log.warning("plotly not installed; skipping charts. Run: pip install plotly")
        return None

    if result.price_series.empty or result.equity_curve.empty:
        log.warning("No data to chart.")
        return None

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.45, 0.30, 0.25],
        subplot_titles=(
            f"{result.symbol} Price & Trades",
            "Portfolio Equity",
            "Drawdown",
        ),
    )

    # --- Chart 1: Price with trade markers ---
    dates = result.price_series["date"]
    closes = result.price_series["close"]

    fig.add_trace(
        go.Scatter(
            x=dates, y=closes,
            mode="lines", name="Close Price",
            line=dict(color="#636EFA", width=1.5),
        ),
        row=1, col=1,
    )

    # Trade shading
    for t in result.trades:
        pnl = t.pnl or 0.0
        color = "rgba(0,200,83,0.12)" if pnl > 0 else "rgba(255,59,48,0.12)"
        fig.add_vrect(
            x0=t.entry_date, x1=t.exit_date,
            fillcolor=color, layer="below", line_width=0,
            row=1, col=1,
        )

    # Entry markers — color by direction
    long_entries = [t for t in result.trades if t.direction == "LONG"]
    short_entries = [t for t in result.trades if t.direction == "SHORT"]

    if long_entries:
        fig.add_trace(
            go.Scatter(
                x=[t.entry_date for t in long_entries],
                y=[t.entry_price for t in long_entries],
                mode="markers", name="Buy (LONG)",
                marker=dict(symbol="triangle-up", size=12, color="#00C853",
                            line=dict(width=1, color="white")),
                text=[f"BUY {t.size:.0f} @ ${t.entry_price:,.2f}" for t in long_entries],
                hoverinfo="text+x",
            ),
            row=1, col=1,
        )

    if short_entries:
        fig.add_trace(
            go.Scatter(
                x=[t.entry_date for t in short_entries],
                y=[t.entry_price for t in short_entries],
                mode="markers", name="Short (SHORT)",
                marker=dict(symbol="triangle-down", size=12, color="#FF9500",
                            line=dict(width=1, color="white")),
                text=[f"SHORT {t.size:.0f} @ ${t.entry_price:,.2f}" for t in short_entries],
                hoverinfo="text+x",
            ),
            row=1, col=1,
        )

    # Exit markers
    exits = [t for t in result.trades if t.exit_price]
    long_exits = [t for t in exits if t.direction == "LONG"]
    short_exits = [t for t in exits if t.direction == "SHORT"]

    if long_exits:
        fig.add_trace(
            go.Scatter(
                x=[t.exit_date for t in long_exits],
                y=[t.exit_price for t in long_exits],
                mode="markers", name="Sell (LONG exit)",
                marker=dict(symbol="triangle-down", size=10, color="#FF3B30",
                            line=dict(width=1, color="white")),
                text=[f"SELL @ ${t.exit_price:,.2f}<br>P&L: ${t.pnl:+,.2f}<br>{t.exit_reason}"
                      for t in long_exits],
                hoverinfo="text+x",
            ),
            row=1, col=1,
        )

    if short_exits:
        fig.add_trace(
            go.Scatter(
                x=[t.exit_date for t in short_exits],
                y=[t.exit_price for t in short_exits],
                mode="markers", name="Cover (SHORT exit)",
                marker=dict(symbol="triangle-up", size=10, color="#34C759",
                            line=dict(width=1, color="white")),
                text=[f"COVER @ ${t.exit_price:,.2f}<br>P&L: ${t.pnl:+,.2f}<br>{t.exit_reason}"
                      for t in short_exits],
                hoverinfo="text+x",
            ),
            row=1, col=1,
        )

    # --- Chart 2: Equity curve ---
    eq_dates = result.equity_curve["date"]
    eq_values = result.equity_curve["equity"]

    fig.add_trace(
        go.Scatter(
            x=eq_dates, y=eq_values,
            mode="lines", name="Equity",
            line=dict(color="#00C853", width=2),
        ),
        row=2, col=1,
    )

    fig.add_hline(
        y=result.initial_capital, line_dash="dash",
        line_color="gray", line_width=1,
        annotation_text=f"${result.initial_capital:,.0f}",
        annotation_position="top left",
        row=2, col=1,
    )

    # Set Y-axis range to show equity changes clearly (not starting from 0)
    eq_min = float(eq_values.min())
    eq_max = float(eq_values.max())
    eq_pad = (eq_max - eq_min) * 0.1
    fig.update_yaxes(
        range=[eq_min - eq_pad, eq_max + eq_pad],
        row=2, col=1,
    )

    # --- Chart 3: Drawdown ---
    if not result.drawdown_series.empty:
        dd_dates = result.drawdown_series["date"]
        dd_values = result.drawdown_series["drawdown"] * 100

        fig.add_trace(
            go.Scatter(
                x=dd_dates, y=dd_values,
                mode="lines", name="Drawdown %",
                line=dict(color="#FF3B30", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(255,59,48,0.15)",
            ),
            row=3, col=1,
        )

    # --- Layout ---
    long_trades = [t for t in result.trades if t.direction == "LONG"]
    short_trades = [t for t in result.trades if t.direction == "SHORT"]
    stats_text = (
        f"Return: {result.total_return_pct:+.2f}%  |  "
        f"Sharpe: {result.sharpe_ratio:.3f}  |  "
        f"Max DD: {result.max_drawdown_pct:.2f}%  |  "
        f"Win Rate: {result.win_rate:.0%}  |  "
        f"Trades: {result.total_trades} ({len(long_trades)}L / {len(short_trades)}S)  |  "
        f"PF: {result.profit_factor:.2f}"
    )

    fig.update_layout(
        title=dict(
            text=(f"{result.symbol} Backtest: {result.start_date} to {result.end_date}"
                  f"<br><sup>{stats_text}</sup>"),
            x=0.5,
        ),
        template="plotly_dark",
        height=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    # Save and open
    os.makedirs(BACKTEST_DIR, exist_ok=True)
    chart_path = os.path.join(BACKTEST_DIR, f"backtest_{result.symbol}_chart.html")
    fig.write_html(chart_path)
    log.info("Chart saved to %s", chart_path)

    try:
        webbrowser.open(f"file:///{chart_path.replace(os.sep, '/')}")
    except Exception:
        pass

    return chart_path


# ===================================================================
# Report
# ===================================================================
def _upsert_promoted_symbols(result: BacktestResult, summary: dict) -> None:
    """Upsert backtest result into promoted_symbols.json for the ETF selector.

    Every backtest updates the selector's knowledge of model performance.
    The promoted file uses a merge strategy: newer results overwrite older ones
    for the same (symbol, mode) pair.
    """
    # Determine target model dir based on model type / mode
    mode = getattr(result, "mode", "daily")
    if mode == "intraday":
        target_dir = INTRADAY_MODEL_DIR
    else:
        # Swing/daily models: check if it's a crypto symbol
        sym = result.symbol
        if "/" in sym or sym.endswith("-USD"):
            target_dir = CRYPTO_MODEL_DIR
        else:
            target_dir = SWING_MODEL_DIR

    promoted_path = os.path.join(target_dir, "promoted_symbols.json")

    # Extract the fields the selector needs
    entry = {
        "symbol": result.symbol,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "total_return_pct": round(float(result.total_return_pct), 2),
        "annualized_return_pct": round(float(result.annualized_return_pct), 2),
        "sharpe_ratio": round(float(result.sharpe_ratio), 3),
        "max_drawdown_pct": round(float(result.max_drawdown_pct), 2),
        "total_trades": int(result.total_trades),
        "win_rate": round(float(result.win_rate), 3),
        "profit_factor": round(float(result.profit_factor), 3)
                         if result.profit_factor != float("inf") else 999.0,
        "avg_trade_duration_days": round(float(result.avg_trade_duration_days), 1),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Load existing promoted file
    existing = {"details": [], "symbols": [], "count": 0}
    if os.path.exists(promoted_path):
        try:
            with open(promoted_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass

    # Merge: replace existing entry for same symbol, or append
    details = existing.get("details", [])
    replaced = False
    for i, d in enumerate(details):
        if d.get("symbol") == result.symbol:
            details[i] = entry
            replaced = True
            break
    if not replaced:
        details.append(entry)

    # Re-sort by Sharpe descending
    details.sort(key=lambda x: -x.get("sharpe_ratio", 0))

    existing["details"] = details
    existing["symbols"] = [d["symbol"] for d in details]
    existing["count"] = len(details)
    existing["last_updated"] = datetime.now(timezone.utc).isoformat()

    os.makedirs(target_dir, exist_ok=True)
    with open(promoted_path, "w") as f:
        json.dump(existing, f, indent=2)


def print_report(result: BacktestResult) -> None:
    print("\n" + "=" * 60)
    print("  BACKTEST REPORT")
    print("=" * 60)
    print(f"  Symbol             : {result.symbol}")
    print(f"  Period             : {result.start_date} to {result.end_date}")
    print(f"  Initial Capital    : ${result.initial_capital:,.2f}")
    print(f"  Final Equity       : ${result.final_equity:,.2f}")
    print("-" * 60)
    print(f"  Total Return       : {result.total_return_pct:+.2f}%")
    print(f"  Annualized Return  : {result.annualized_return_pct:+.2f}%")
    print(f"  Sharpe Ratio       : {result.sharpe_ratio:.3f}")
    print(f"  Max Drawdown       : {result.max_drawdown_pct:.2f}%")
    print("-" * 60)
    print(f"  Total Trades       : {result.total_trades}")
    print(f"  Win Rate           : {result.win_rate:.1%}")
    print(f"  Avg Win            : {result.avg_win_pct:+.2f}%")
    print(f"  Avg Loss           : {result.avg_loss_pct:+.2f}%")
    print(f"  Profit Factor      : {result.profit_factor:.3f}")
    print(f"  Avg Hold Duration  : {result.avg_trade_duration_days:.1f} days")

    # Per-direction breakdown
    long_trades = [t for t in result.trades if t.direction == "LONG"]
    short_trades = [t for t in result.trades if t.direction == "SHORT"]
    if long_trades or short_trades:
        print("-" * 60)
        if long_trades:
            long_wins = sum(1 for t in long_trades if t.pnl and t.pnl > 0)
            long_wr = long_wins / len(long_trades) if long_trades else 0
            print(f"  LONG  trades: {len(long_trades):>3}   win rate: {long_wr:.0%}")
        if short_trades:
            short_wins = sum(1 for t in short_trades if t.pnl and t.pnl > 0)
            short_wr = short_wins / len(short_trades) if short_trades else 0
            print(f"  SHORT trades: {len(short_trades):>3}   win rate: {short_wr:.0%}")
    print("=" * 60)

    # --- Trade history table ---
    if result.trades:
        print(f"\n  TRADE HISTORY ({len(result.trades)} trades)")
        print("-" * 110)
        print(f"  {'#':>3}  {'Dir':>5}  {'Entry Date':>12}  {'Entry $':>10}  {'Exit Date':>12}  "
              f"{'Exit $':>10}  {'Shares':>8}  {'P&L':>12}  {'Return':>8}  {'Reason'}")
        print("-" * 110)
        for i, t in enumerate(result.trades, 1):
            pnl = t.pnl or 0.0
            ret_pct = (pnl / (t.size * t.entry_price) * 100) if t.size and t.entry_price else 0.0
            print(f"  {i:>3}  {t.direction:>5}  {str(t.entry_date):>12}  {t.entry_price:>10,.2f}  "
                  f"{str(t.exit_date):>12}  {t.exit_price:>10,.2f}  {t.size:>8,.1f}  "
                  f"{'${:>+,.2f}'.format(pnl):>12}  {ret_pct:>+7.2f}%  {t.exit_reason}")
        print("-" * 110)

    # Save equity curve and summary (for training tables)
    os.makedirs(BACKTEST_DIR, exist_ok=True)
    mode = getattr(result, "mode", "daily")
    mode_suffix = f"_{mode}" if mode != "daily" else ""
    csv_path = os.path.join(BACKTEST_DIR, f"backtest_{result.symbol}{mode_suffix}.csv")
    result.equity_curve.to_csv(csv_path, index=False)
    print(f"\n  Equity curve saved to {csv_path}")

    # Variant-specific backtest dir
    variant_tag = result.exit_variant
    if variant_tag:
        bt_dir = os.path.join(os.path.dirname(BACKTEST_DIR), f"backtests_{variant_tag}")
        os.makedirs(bt_dir, exist_ok=True)
    else:
        bt_dir = BACKTEST_DIR
    summary_path = os.path.join(bt_dir, f"backtest_{result.symbol}{mode_suffix}_summary.json")
    # Also write the non-suffixed version for backwards compat
    summary_path_compat = os.path.join(bt_dir, f"backtest_{result.symbol}_summary.json")
    summary = {
        "symbol": result.symbol,
        "mode": getattr(result, "mode", "daily"),
        "intraday_interval": getattr(result, "intraday_interval", "5min"),
        "start_date": result.start_date,
        "end_date": result.end_date,
        "initial_capital": float(result.initial_capital),
        "final_equity": float(result.final_equity),
        "total_return_pct": float(result.total_return_pct),
        "annualized_return_pct": float(result.annualized_return_pct),
        "sharpe_ratio": float(result.sharpe_ratio),
        "max_drawdown_pct": float(result.max_drawdown_pct),
        "total_trades": int(result.total_trades),
        "win_rate": float(result.win_rate),
        "avg_win_pct": float(result.avg_win_pct),
        "avg_loss_pct": float(result.avg_loss_pct),
        "profit_factor": float(result.profit_factor) if result.profit_factor != float("inf") else None,
        "avg_trade_duration_days": float(result.avg_trade_duration_days),
        "vol_tier": result.vol_tier,
        "exit_variant": variant_tag or "base",
        "bt_group": result.bt_group,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(summary_path_compat, "w") as f:
        json.dump(summary, f, indent=2)

    # Upsert into promoted_symbols.json so the ETF selector always has fresh metrics
    _upsert_promoted_symbols(result, summary)

    # Save trade history
    if result.trades:
        # Variant-specific output dir
        variant_tag = result.exit_variant
        if variant_tag:
            trades_dir = os.path.join(os.path.dirname(TRADES_DIR), f"trades_{variant_tag}")
        else:
            trades_dir = TRADES_DIR
        os.makedirs(trades_dir, exist_ok=True)
        trades_csv = os.path.join(trades_dir, f"trades_{result.symbol}{mode_suffix}.csv")
        rows = []
        for t in result.trades:
            pnl = t.pnl or 0.0
            ret_pct = (pnl / (t.size * t.entry_price)) if t.size and t.entry_price else 0.0
            rows.append({
                "symbol": result.symbol,
                "group": result.bt_group,
                "tier": result.vol_tier,
                "variant": variant_tag or "base",
                "direction": t.direction,
                "entry_date": t.entry_date, "entry_price": round(t.entry_price, 4),
                "exit_date": t.exit_date, "exit_price": round(t.exit_price, 4),
                "shares": round(t.size, 4), "pnl": round(pnl, 2),
                "return_pct": round(ret_pct * 100, 4),
                "mfe_pct": round(t.mfe_pct * 100, 4),
                "mae_pct": round(t.mae_pct * 100, 4),
                "exit_reason": t.exit_reason,
                "exit_layer": t.exit_layer,
            })
        pd.DataFrame(rows).to_csv(trades_csv, index=False)
        print(f"  Trade history saved to {trades_csv}\n")
    else:
        print()

    # Intraday diagnostics (day-of-week, month breakdown)
    if result.trades and getattr(result, "mode", "daily") in ("intraday",) or (
        result.trades and all(
            t.entry_date == t.exit_date for t in result.trades if t.exit_date
        )
    ):
        _print_intraday_diagnostics(result)

    # Generate interactive chart
    chart_path = generate_charts(result)
    if chart_path:
        print(f"  Interactive chart opened in browser: {chart_path}\n")


def _print_intraday_diagnostics(result: BacktestResult) -> None:
    """Print per-day-of-week and per-month P&L breakdown for intraday trades."""
    if not result.trades:
        return

    trades_data = []
    for t in result.trades:
        pnl = t.pnl or 0.0
        ret_pct = (pnl / (t.size * t.entry_price)) if t.size and t.entry_price else 0.0
        entry = pd.Timestamp(t.entry_date)
        trades_data.append({
            "date": entry,
            "dow": entry.day_name(),
            "dow_num": entry.dayofweek,
            "month": entry.strftime("%Y-%m"),
            "pnl": pnl,
            "return_pct": ret_pct * 100,
            "win": 1 if pnl > 0 else 0,
        })

    df = pd.DataFrame(trades_data)

    # --- Day of week breakdown ---
    print("\n  INTRADAY DIAGNOSTICS: Day-of-Week")
    print("-" * 75)
    print(f"  {'Day':>10}  {'Trades':>7}  {'WinRate':>8}  {'AvgRet':>8}  {'TotalP&L':>12}  {'Sharpe':>7}")
    print("-" * 75)

    for dow_num in range(5):
        day_df = df[df["dow_num"] == dow_num]
        if day_df.empty:
            continue
        n = len(day_df)
        wr = day_df["win"].mean()
        avg_ret = day_df["return_pct"].mean()
        total_pnl = day_df["pnl"].sum()
        std = day_df["return_pct"].std()
        sharpe = avg_ret / std if std > 0 else 0.0
        day_name = day_df["dow"].iloc[0]
        flag = " !" if wr < 0.45 or avg_ret < 0 else ""
        print(f"  {day_name:>10}  {n:>7}  {wr:>7.0%}  {avg_ret:>+7.2f}%  "
              f"{'${:>+,.0f}'.format(total_pnl):>12}  {sharpe:>+6.2f}{flag}")
    print("-" * 75)

    # --- Monthly breakdown ---
    months = sorted(df["month"].unique())
    if len(months) > 1:
        print("\n  INTRADAY DIAGNOSTICS: Monthly")
        print("-" * 75)
        print(f"  {'Month':>10}  {'Trades':>7}  {'WinRate':>8}  {'AvgRet':>8}  {'TotalP&L':>12}  {'Sharpe':>7}")
        print("-" * 75)

        for month in months:
            mo_df = df[df["month"] == month]
            n = len(mo_df)
            wr = mo_df["win"].mean()
            avg_ret = mo_df["return_pct"].mean()
            total_pnl = mo_df["pnl"].sum()
            std = mo_df["return_pct"].std()
            sharpe = avg_ret / std if std > 0 else 0.0
            flag = " !" if wr < 0.45 or avg_ret < 0 else ""
            print(f"  {month:>10}  {n:>7}  {wr:>7.0%}  {avg_ret:>+7.2f}%  "
                  f"{'${:>+,.0f}'.format(total_pnl):>12}  {sharpe:>+6.2f}{flag}")
        print("-" * 75)

    # --- Worst/best days ---
    if len(df) >= 5:
        print("\n  Top 5 best trades:")
        best = df.nlargest(5, "pnl")
        for _, row in best.iterrows():
            print(f"    {str(row['date'].date()):>12}  {row['dow']:>9}  "
                  f"P&L: {'${:>+,.0f}'.format(row['pnl'])}  ({row['return_pct']:>+.2f}%)")

        print("  Top 5 worst trades:")
        worst = df.nsmallest(5, "pnl")
        for _, row in worst.iterrows():
            print(f"    {str(row['date'].date()):>12}  {row['dow']:>9}  "
                  f"P&L: {'${:>+,.0f}'.format(row['pnl'])}  ({row['return_pct']:>+.2f}%)")
    print()


# ===================================================================
# CLI
# ===================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ML-driven backtester for ETF signals (v2 regression).",
    )
    parser.add_argument("--symbol", required=True, help="Symbol to backtest (e.g. SPY)")
    parser.add_argument("--provider", default="yahoo",
                        choices=["yahoo", "alpaca", "hybrid"])
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--capital", type=float, default=100_000,
                        help="Initial capital (default: 100000)")
    parser.add_argument("--mode", default="daily", choices=["daily", "intraday"],
                        help="Backtest mode (default: daily)")
    parser.add_argument("--interval", default="5min", choices=["1min", "5min"],
                        help="Intraday bar interval (default: 5min)")
    parser.add_argument("--model", default="lstm",
                        choices=["lstm", "swing", "options", "intraday", "crypto_intraday", "etf_intraday"],
                        help="Model type: lstm (default), swing, options, intraday, crypto_intraday, etf_intraday")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory for model files (default: models/ or models/crypto/ for crypto)")
    # v2 regression parameters
    parser.add_argument("--trend-sma", type=int, default=50,
                        help="SMA period for trend filter (default: 50)")
    parser.add_argument("--cost-threshold", type=float, default=COST_THRESHOLD,
                        help="Min expected return to trade (default: 0.001 = 0.1%%)")
    parser.add_argument("--target-return", type=float, default=TARGET_RETURN,
                        help="Expected return for full position size (default: 0.02 = 2%%)")
    parser.add_argument("--disaster-stop-mult", type=float, default=3.0,
                        help="ATR multiplier for disaster stop (default: 3.0)")
    parser.add_argument("--disaster-stop-max-pct", type=float, default=0.20,
                        help="Hard cap on disaster stop percentage (default: 0.20 = 20%%)")
    parser.add_argument("--profit-lock-atr-mult", type=float, default=2.0,
                        help="ATR multiplier for profit-lock activation (default: 2.0)")
    parser.add_argument("--profit-lock-trail-atr-mult", type=float, default=1.5,
                        help="ATR multiplier for profit-lock trailing stop (default: 1.5)")
    parser.add_argument("--max-underwater-days", type=int, default=90,
                        help="Max days to hold an underwater position (default: 90; 0=off)")
    parser.add_argument("--position-pct", type=float, default=0.50,
                        help="Base capital fraction for position sizing (default: 0.50, was 0.95)")
    # Cost model / stress test
    parser.add_argument("--stress-cost-mult", type=float, default=1.0,
                        help="Cost multiplier for stress testing (default: 1.0; use 2.0 or 3.0 for stress)")
    parser.add_argument("--no-cost-model", action="store_true",
                        help="Disable cost model (use raw prices, no slippage)")
    parser.add_argument("--exit-variant", default=None, choices=["A", "B"],
                        help="A/B exit variant: A=tighter, B=looser (default: base params)")

    args = parser.parse_args()

    # Auto-infer mode from model type — intraday models must use intraday mode
    _INTRADAY_MODELS = {"intraday", "crypto_intraday", "etf_intraday"}
    if args.model in _INTRADAY_MODELS and args.mode == "daily":
        log.info("Auto-setting --mode intraday (required by --model %s)", args.model)
        args.mode = "intraday"

    adapter = build_adapter(args.provider)
    fred_key = os.environ.get("FRED_API_KEY")

    # Determine model_dir: check group-specific dirs, then fall back
    if args.model_dir:
        model_dir = args.model_dir
    else:
        sym_upper = args.symbol.upper()
        if args.model == "swing":
            # Check swing dir, then crypto dir, then legacy root
            for candidate in [SWING_MODEL_DIR, CRYPTO_MODEL_DIR, DEFAULT_MODEL_DIR]:
                if os.path.exists(os.path.join(candidate, f"{sym_upper}_xgb_swing.joblib")):
                    model_dir = candidate
                    break
            else:
                model_dir = SWING_MODEL_DIR
        elif args.model in ("intraday", "etf_intraday"):
            model_dir = INTRADAY_MODEL_DIR
        elif args.model == "crypto_intraday":
            model_dir = CRYPTO_INTRADAY_MODEL_DIR
        else:
            model_dir = DEFAULT_MODEL_DIR

    bt = Backtester(
        symbol=args.symbol.upper(),
        adapter=adapter,
        fred_key=fred_key,
        model_dir=model_dir,
        initial_capital=args.capital,
        position_pct=args.position_pct,
        mode=args.mode,
        intraday_interval=args.interval,
        model_type=args.model,
        trend_sma_period=args.trend_sma,
        cost_threshold=args.cost_threshold,
        target_return=args.target_return,
        disaster_stop_atr_mult=args.disaster_stop_mult,
        disaster_stop_max_pct=args.disaster_stop_max_pct,
        profit_lock_atr_mult=args.profit_lock_atr_mult,
        profit_lock_trail_atr_mult=args.profit_lock_trail_atr_mult,
        max_underwater_days=args.max_underwater_days,
        stress_cost_mult=args.stress_cost_mult,
        use_cost_model=not args.no_cost_model,
        exit_variant=args.exit_variant,
    )

    result = bt.run(start_date=args.start, end_date=args.end)
    print_report(result)


if __name__ == "__main__":
    main()
