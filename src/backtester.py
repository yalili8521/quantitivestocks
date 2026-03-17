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

        # --- Regime filter layer 1: SPY SMA(200) ---
        # Only trade when SPY close > SMA(200) (bull regime).
        spy_sma_lookup: dict = {}  # date -> bool (True = bull regime)
        try:
            from signals_engine import build_adapter as _build_yahoo
            yahoo_adp = _build_yahoo("yahoo")
            spy_df = yahoo_adp.fetch_daily("SPY", days_needed + 250)
            if not spy_df.empty:
                sma200 = spy_df["close"].rolling(200).mean()
                for d, c, s in zip(spy_df.index, spy_df["close"], sma200):
                    if not pd.isna(s):
                        dk = d.date() if hasattr(d, "date") else d
                        spy_sma_lookup[dk] = bool(c > s)
            log.info("SPY SMA(200) regime lookup built (%d days).", len(spy_sma_lookup))
        except Exception as exc:
            log.warning("SPY SMA(200) regime filter unavailable: %s", exc)

        # --- Regime filter layer 2: rolling 20-trade win rate cooldown ---
        from collections import deque
        from datetime import timedelta as _td
        recent_wins: deque = deque(maxlen=20)
        cooldown_until = None

        # Simulate day-by-day
        portfolio = Portfolio(initial_capital=self.initial_capital, cash=self.initial_capital)
        price_rows = []
        regime_skipped = 0
        cooldown_skipped = 0

        for _, row in data.iterrows():
            day = row["date"]
            entry_price = float(row["entry_price"])
            eod_close = float(row["eod_close"])
            first_30m_ret = float(row["first_30m_ret_raw"])

            if entry_price <= 0 or eod_close <= 0:
                continue

            price_rows.append({"date": day, "close": eod_close})

            # Layer 1: regime gate — SPY SMA(200) (VIX gate skipped for intraday)
            spy_bull = spy_sma_lookup.get(day, True)  # default open if data missing
            if not spy_bull:
                self._record_equity(portfolio, day, eod_close)
                regime_skipped += 1
                continue

            # Layer 2: rolling win rate cooldown
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

        log.info("Regime filter skipped %d days; cooldown skipped %d days.",
                 regime_skipped, cooldown_skipped)

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

        # Load config for cost/target overrides
        crypto_ct = self.cost_threshold
        crypto_tr = self.target_return
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            crypto_ct = cfg.get("cost_threshold", CRYPTO_INTRADAY_CT)
            crypto_tr = cfg.get("target_return", CRYPTO_INTRADAY_TR)
            saved_features = cfg.get("feature_names")
            if saved_features:
                feature_cols = saved_features
        log.info("Loaded LGB for %s (%d features, ct=%.4f, tr=%.4f).",
                 self.symbol, len(feature_cols), crypto_ct, crypto_tr)

        # --- BTC SMA(200) regime filter (daily) ---
        # Resample 5-min bars to daily close for SMA calculation
        bars_ts = pd.to_datetime(bars["ts"])
        if bars_ts.dt.tz is None:
            bars_ts = bars_ts.dt.tz_localize("UTC")
        btc_regime_bars = btc_bars if btc_bars is not None else bars
        btc_regime_ts = pd.to_datetime(btc_regime_bars["ts"])
        if btc_regime_ts.dt.tz is None:
            btc_regime_ts = btc_regime_ts.dt.tz_localize("UTC")
        btc_daily = btc_regime_bars.copy()
        btc_daily["date"] = btc_regime_ts.dt.date
        btc_daily_close = btc_daily.groupby("date")["close"].last()
        btc_sma200 = btc_daily_close.rolling(200).mean()

        regime_lookup = {}
        for d, c, s in zip(btc_daily_close.index, btc_daily_close.values, btc_sma200.values):
            if not pd.isna(s):
                regime_lookup[d] = bool(float(c) > float(s))
        log.info("BTC SMA(200) regime lookup: %d days (%d bull).",
                 len(regime_lookup),
                 sum(1 for v in regime_lookup.values() if v))

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
        regime_skipped = 0
        cooldown_skipped = 0
        bars_in_position = 0
        price_rows = []

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

            # --- LGB prediction ---
            try:
                x_row = features[feature_cols].iloc[i:i + 1].values.astype(np.float32)
                expected_return = float(lgb_model.predict(x_row)[0])
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
                if pos.direction == "LONG" and expected_return <= 0:
                    self._close_position(portfolio, bar_date, bar_close,
                                         "signal_decay")
                    recent_wins.append(1 if portfolio.closed_trades[-1].pnl > 0 else 0)
                    bars_in_position = 0
                elif pos.direction == "SHORT" and expected_return >= 0:
                    self._close_position(portfolio, bar_date, bar_close,
                                         "signal_decay")
                    recent_wins.append(1 if portfolio.closed_trades[-1].pnl > 0 else 0)
                    bars_in_position = 0

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

            # Regime gate: BTC above SMA(200)
            regime_ok = regime_lookup.get(bar_date, True)
            if not regime_ok:
                regime_skipped += 1
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
                # Use crypto-specific position sizing
                equity = self._current_equity(portfolio, bar_close)
                signal_pct = min(1.0, max(0.1,
                                          abs(expected_return) / crypto_tr))
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

        log.info("Crypto intraday: %d trades, %d regime-skipped bars, "
                 "%d cooldown-skipped bars.",
                 len(portfolio.closed_trades), regime_skipped, cooldown_skipped)

        price_df = pd.DataFrame(price_rows)
        # Deduplicate prices (multiple 5-min bars per day)
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
        spy_bars = None    # SPY bars for regime filter (fetched by swing, None for LSTM)
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

        # --- Regime filter: SPY SMA(200) + VIX < 30 + rolling win rate ---
        # Layer 1: SPY SMA(200) — use already-fetched spy_bars if available
        _spy_for_regime = spy_bars if (spy_bars is not None and not spy_bars.empty) else bars
        _spy_close = _spy_for_regime["close"].astype(float)
        _spy_sma200 = _spy_close.rolling(200).mean()
        _spy_ts = pd.to_datetime(_spy_for_regime["ts"]).dt.date
        swing_spy_sma_lookup: dict = {}
        for _d, _c, _s in zip(_spy_ts, _spy_close, _spy_sma200):
            if not pd.isna(_s):
                swing_spy_sma_lookup[_d] = bool(_c > _s)
        log.info("SPY SMA(200) regime lookup built for swing (%d days).", len(swing_spy_sma_lookup))

        # Layer 2: VIX lookup from vix_df
        swing_vix_lookup: dict = {}
        if not vix_df.empty:
            _vcol = vix_df.columns[0]
            for _idx, _row in vix_df.iterrows():
                _dk = _idx.date() if hasattr(_idx, "date") else _idx
                try:
                    swing_vix_lookup[_dk] = float(_row[_vcol])
                except Exception:
                    pass

        # Layer 3: rolling 20-trade win rate cooldown
        from collections import deque
        from datetime import timedelta as _td
        swing_recent_wins: deque = deque(maxlen=20)
        swing_cooldown_until = None
        swing_regime_skipped = 0
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
                x_row = features_norm.iloc[idx_pos:idx_pos + 1].values.astype(np.float32)
                expected_return = float(xgb_model.predict(x_row)[0])
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

                if pos.tp_activated:
                    # 2a: Armed + model flips + still profitable -> immediate exit
                    signal_flipped = ((pos.direction == "LONG" and expected_return <= 0)
                                      or (pos.direction == "SHORT" and expected_return >= 0))
                    if signal_flipped and unrealized_pct > 0:
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
                signal_flipped = ((pos.direction == "LONG" and expected_return <= 0)
                                  or (pos.direction == "SHORT" and expected_return >= 0))
                if signal_flipped:
                    pos.signal_flip_count += 1
                else:
                    pos.signal_flip_count = 0

                if pos.signal_flip_count >= ep.signal_flip_consecutive:
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

            # --- Entry logic (v2: trend filter + regime gate + signal-proportional sizing) ---
            if portfolio.position is None and not is_last_bar:
                # Regime gate: SPY SMA(200) + VIX < 30
                _spy_ok = swing_spy_sma_lookup.get(bar_date, True)
                _vix_val = swing_vix_lookup.get(bar_date, 20.0)
                if not _spy_ok or _vix_val >= 30.0:
                    self._record_equity(portfolio, bar_date, bar_close)
                    swing_regime_skipped += 1
                    continue
                # Rolling win rate cooldown
                if swing_cooldown_until is not None:
                    if bar_date < swing_cooldown_until:
                        self._record_equity(portfolio, bar_date, bar_close)
                        swing_cooldown_skipped += 1
                        continue
                    else:
                        swing_cooldown_until = None

                enter_dir = None
                if expected_return > self.cost_threshold and bar_trend > 0:
                    enter_dir = "LONG"
                elif expected_return < -self.cost_threshold and bar_trend < 0:
                    enter_dir = "SHORT"

                if enter_dir is not None:
                    self._open_position(
                        portfolio, bar_date, bar_close, enter_dir,
                        expected_return=expected_return, bar_atr=bar_atr,
                        bar_time=bar_time,
                    )

            self._record_equity(portfolio, bar_date, bar_close)

        log.info("Swing regime filter skipped %d days; cooldown skipped %d days.",
                 swing_regime_skipped, swing_cooldown_skipped)

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
        signal_pct = min(1.0, max(0.1, abs(expected_return) / self.target_return))
        base_pct = self.position_pct * signal_pct

        # Apply per-position cap directly as fraction of equity
        try:
            from risk_config import get_risk_config
            group = "intraday" if self.mode == "intraday" else "swing"
            risk = get_risk_config(group)
            base_pct = min(base_pct, risk.max_position_pct)
        except ImportError:
            base_pct = min(base_pct, 0.15)

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
                        choices=["lstm", "swing", "options", "intraday", "crypto_intraday"],
                        help="Model type: lstm (default), swing, options, intraday, crypto_intraday")
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
        elif args.model == "intraday":
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
