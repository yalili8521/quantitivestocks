#!/usr/bin/env python3
"""
Range Trader -- Intraday Mean Reversion (Alpaca Paper)
======================================================
Live paper-trading counterpart to range_backtester.py.
Runs on the same Alpaca account as the intraday group (ALPACA_INTRADAY_KEY).
Budget: 30% of total account equity.

Strategy:
  - Regime gate:  H < 0.62, ADX < 28, VIX < 30
  - Mean price:   running VWAP (prior bars only) adjusted by volume drift + morning trend
  - Range bounds: predicted_mean +/-2.0 x intra_ATR (VIX-scaled)
  - LONG  entry:  bar_low <= lower_bound AND close > lower_bound AND bar_buy_frac >= 0.50
  - SHORT entry:  bar_high >= upper_bound AND close < upper_bound AND bar_buy_frac <= 0.50
  - Stop:         entry +/- 1.0 x intra_ATR (fixed distance, R:R ~2)
  - Exits:        profit target (VWAP at entry), breakeven stop, hard stop, time stop 15:15 ET

Usage:
    python main.py range-trade --symbols SPY,QQQ,IWM,SOXX

Required env vars (any one of these key sets):
    ALPACA_INTRADAY_KEY / ALPACA_INTRADAY_SECRET  (preferred -- shares intraday account)
    ALPACA_API_KEY / ALPACA_API_SECRET             (fallback)
    FRED_API_KEY                                   (optional, for live VIX)

PAPER TRADING ONLY -- paper=True is hardcoded.
"""

from __future__ import annotations

import argparse
import collections
import logging
import math
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from signals_engine import (
    build_adapter,
    compute_atr,
    compute_adx_full,
    compute_hurst_exponent,
    compute_vwap,
    compute_volume_imbalance,
    DAILY_LOOKBACK,
)
from ml_model import _fetch_vix_for_training
from paper_trader import _get_session, _time_until_next_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("range_trader")

# ---------------------------------------------------------------------------
# Constants (synced with range_backtester.py)
# ---------------------------------------------------------------------------
DEFAULT_SYMBOLS = ["SPY", "QQQ", "IWM", "SOXX"]

BUDGET_PCT        = 0.30        # total range-trading budget = 30% of account equity
POSITION_PCT      = 0.10        # max 10% of budget per symbol
TARGET_VOL        = 0.15        # annualised vol target for position scaling

# Regime gates
HURST_THRESHOLD   = 0.62        # H < 0.62 = mean-reverting
ADX_THRESHOLD     = 28.0        # ADX < 28  = no strong trend
VIX_MAX           = 30.0        # VIX >= 30 = extreme panic, skip
VOL_RATIO_MIN     = 0.40        # bar volume >= 40% of rolling avg

# Range geometry
RANGE_MULT        = 2.00        # bounds at +/-2.0 x intra_ATR
STOP_MULT         = 1.00        # fixed stop = 1.0 x intra_ATR from entry (R:R ~2)
VOL_DRIFT_SCALE   = 0.30        # volume imbalance adjusts mean by <=30% of intra_ATR
MORNING_DRIFT_CAP = 0.20        # morning trend capped at +/-20% of intra_ATR

# Intraday ATR (5-min bars)
INTRA_ATR_PERIOD   = 10
INTRA_ATR_MIN_BARS = 15
INTRA_TO_DAILY     = 0.06       # fallback: intra_ATR ~6% of daily ATR

# Exit rules
BREAKEVEN_PCT     = 0.003       # +0.3% triggers breakeven protection
BREAKEVEN_BUFFER  = 0.002       # exit at -0.2% after breakeven
TIME_STOP_HOUR    = 15
TIME_STOP_MIN     = 15

# Loop
CHECK_INTERVAL    = 5 * 60      # seconds between cycles (5 minutes)
MARKET_OPEN_HOUR  = 10          # skip 9:30-10:00 open noise

_ET = "America/New_York"


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def _et_now():
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo(_ET))

def _et_hm():
    t = _et_now()
    return t.hour, t.minute

def _in_trade_window() -> bool:
    h, m = _et_hm()
    return (h, m) >= (MARKET_OPEN_HOUR, 0) and (h, m) < (TIME_STOP_HOUR, TIME_STOP_MIN)

def _time_stop_imminent() -> bool:
    h, m = _et_hm()
    return (h, m) >= (TIME_STOP_HOUR, TIME_STOP_MIN)

def _today_et() -> str:
    return _et_now().strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# RangeTrader
# ---------------------------------------------------------------------------
class RangeTrader:
    """
    Intraday mean-reversion paper trader.
    Shares the intraday group Alpaca account; uses at most BUDGET_PCT of equity.
    Checks every 5 minutes during regular hours.
    """

    def __init__(
        self,
        symbols:        List[str],
        api_key:        str,
        api_secret:     str,
        provider:       str  = "alpaca",
        fred_key:       Optional[str] = None,
        position_pct:   float = POSITION_PCT,
        check_interval: int   = CHECK_INTERVAL,
    ):
        self.symbols        = symbols
        self.position_pct   = position_pct
        self.check_interval = check_interval
        self.fred_key       = fred_key or os.environ.get("FRED_API_KEY")

        self.trading_client = TradingClient(api_key, api_secret, paper=True)
        self.adapter        = build_adapter(provider)

        # Per-symbol state (reset each time a position closes)
        self._entry_price:    Dict[str, float] = {}
        self._entry_stop:     Dict[str, float] = {}   # fixed stop level set at entry
        self._entry_bounds:   Dict[str, dict]  = {}   # {lower, predicted_mean, upper}
        self._breakeven:      Dict[str, bool]  = {}
        self._open_trade_day: Dict[str, str]   = {}

        # Per-cycle cached state
        self._cycle_vix:       float = 20.0
        self._cycle_hurst:     Dict[str, float] = {}
        self._cycle_adx:       Dict[str, float] = {}
        self._cycle_atr:       Dict[str, float] = {}   # daily ATR (sizing only)
        self._cycle_vol20:     Dict[str, float] = {}
        self._cycle_avg5vol:   Dict[str, float] = {}

        # Closed-trade history for dynamic Kelly
        self._closed_trades: collections.deque = collections.deque(maxlen=60)

        self._running = True
        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, *_):
        log.info("Shutdown signal received.")
        self._running = False

    # ------------------------------------------------------------------
    # Account / positions
    # ------------------------------------------------------------------
    def _account(self) -> dict:
        acc = self.trading_client.get_account()
        return {"equity": float(acc.equity), "cash": float(acc.cash)}

    def _positions(self) -> Dict[str, dict]:
        result = {}
        for pos in self.trading_client.get_all_positions():
            qty  = float(pos.qty)
            side = "SHORT" if qty < 0 else "LONG"
            result[pos.symbol] = {
                "qty":             abs(qty),
                "side":            side,
                "entry_price":     float(pos.avg_entry_price),
                "current_price":   float(pos.current_price),
                "unrealized_pnl":  float(pos.unrealized_pl),
                "pnl_pct":         float(pos.unrealized_plpc),
            }
        return result

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def _submit(self, symbol: str, qty: int, side: OrderSide,
                limit_price: Optional[float] = None) -> Optional[str]:
        session = _get_session()
        try:
            if session == "extended" and limit_price is not None:
                lp = round(limit_price * (1.001 if side == OrderSide.BUY else 0.999), 2)
                req = LimitOrderRequest(
                    symbol=symbol, qty=qty, side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=lp, extended_hours=True,
                )
            else:
                req = MarketOrderRequest(
                    symbol=symbol, qty=qty, side=side,
                    time_in_force=TimeInForce.DAY,
                )
            order = self.trading_client.submit_order(req)
            log.info("%s %s x%d -- order %s", side.value, symbol, qty, order.id)
            return str(order.id)
        except Exception as exc:
            log.error("Order failed %s %s x%d: %s", side.value, symbol, qty, exc)
            return None

    def _buy(self, symbol, qty, price=None):
        return self._submit(symbol, qty, OrderSide.BUY, price)

    def _sell(self, symbol, qty, reason="", price=None):
        oid = self._submit(symbol, qty, OrderSide.SELL, price)
        if oid:
            log.info("SELL %s reason=%s", symbol, reason)
        return oid

    def _short(self, symbol, qty, price=None):
        return self._submit(symbol, qty, OrderSide.SELL, price)

    def _cover(self, symbol, qty, reason="", price=None):
        oid = self._submit(symbol, qty, OrderSide.BUY, price)
        if oid:
            log.info("COVER %s reason=%s", symbol, reason)
        return oid

    def _clear(self, symbol: str):
        for d in (self._entry_price, self._entry_stop, self._entry_bounds,
                  self._breakeven, self._open_trade_day):
            d.pop(symbol, None)

    def _record(self, pnl_pct: float):
        self._closed_trades.append(pnl_pct)

    # ------------------------------------------------------------------
    # Dynamic Kelly
    # ------------------------------------------------------------------
    def _kelly(self) -> float:
        trades = list(self._closed_trades)
        if len(trades) < 20:
            return 0.5
        wins   = [p for p in trades if p > 0]
        losses = [abs(p) for p in trades if p <= 0]
        if not wins or not losses:
            return 0.5
        W = len(wins) / len(trades)
        B = np.mean(wins) / max(np.mean(losses), 1e-9)
        f = (W * (B + 1) - 1) / B
        return float(np.clip(f * 0.5, 0.1, 1.0))

    # ------------------------------------------------------------------
    # Per-cycle daily context (regime + sizing)
    # ------------------------------------------------------------------
    def _refresh_daily_context(self, symbol: str) -> bool:
        try:
            bars = self.adapter.fetch_daily(symbol, lookback=60)
        except Exception as exc:
            log.warning("[%s] Daily context fetch failed: %s", symbol, exc)
            return False

        if len(bars) < 30:
            return False

        hi = bars["high"].astype(float)
        lo = bars["low"].astype(float)
        cl = bars["close"].astype(float)

        atr_s = compute_atr(hi, lo, cl, period=14)
        atr   = float(atr_s.dropna().iloc[-1]) if not atr_s.dropna().empty else np.nan
        if pd.isna(atr) or atr <= 0:
            return False

        adx_s, _, _ = compute_adx_full(hi, lo, cl, period=14)
        adx   = float(adx_s.dropna().iloc[-1]) if not adx_s.dropna().empty else np.nan

        hurst = compute_hurst_exponent(cl)

        rets  = cl.pct_change().dropna()
        vol20 = float(rets.tail(20).std() * math.sqrt(252)) if len(rets) >= 5 else 0.15

        avg5v = float(bars["volume"].tail(20).mean()) / 78.0 if "volume" in bars.columns else 1.0

        self._cycle_atr[symbol]    = atr
        self._cycle_adx[symbol]    = adx
        self._cycle_hurst[symbol]  = hurst
        self._cycle_vol20[symbol]  = vol20
        self._cycle_avg5vol[symbol] = max(avg5v, 1.0)
        return True

    def _refresh_vix(self):
        try:
            vix_df = _fetch_vix_for_training(self.fred_key, lookback_days=5)
            if vix_df is not None and not vix_df.empty:
                self._cycle_vix = float(vix_df["vix"].iloc[-1])
        except Exception as exc:
            log.warning("VIX refresh failed: %s", exc)

    # ------------------------------------------------------------------
    # Per-cycle intraday context (VWAP, intra_ATR, bar data, vol imbalance)
    # ------------------------------------------------------------------
    def _intraday_context(self, symbol: str) -> Optional[dict]:
        """
        Fetch today's 5min bars.
        Returns VWAP from prior bars (backtester-style), intraday ATR,
        last bar OHLCV, and volume ratio.
        """
        try:
            bars = self.adapter.fetch_intraday(symbol, "5min", lookback_days=2)
        except Exception as exc:
            log.warning("[%s] Intraday fetch failed: %s", symbol, exc)
            return None

        if bars.empty:
            return None

        bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
        today_str  = _today_et()
        bars["et_date"] = bars["ts"].dt.tz_convert(_ET).dt.strftime("%Y-%m-%d")
        today_bars  = bars[bars["et_date"] == today_str].reset_index(drop=True)

        # Keep bars from 9:30 onwards
        def _after_930(ts):
            t = pd.Timestamp(ts).tz_convert(_ET)
            return (t.hour, t.minute) >= (9, 30)

        today_bars = today_bars[today_bars["ts"].apply(_after_930)].reset_index(drop=True)
        if len(today_bars) < 2:
            return None

        # Last bar values
        last        = today_bars.iloc[-1]
        price       = float(last["close"])
        bar_high    = float(last.get("high", price))
        bar_low     = float(last.get("low",  price))
        vol_last    = float(last["volume"]) if "volume" in today_bars else 0.0
        open_price  = float(today_bars.iloc[0]["open"])
        n_bars      = len(today_bars)

        # VWAP from PRIOR bars only (not including last bar)
        bars_prior = today_bars.iloc[:-1]
        vwap = compute_vwap(bars_prior)
        if pd.isna(vwap) or vwap <= 0:
            vwap = open_price

        # Volume imbalance from prior bars
        vi_raw        = compute_volume_imbalance(bars_prior) if not bars_prior.empty else 0.0
        vol_imbalance = (vi_raw + 1) / 2.0

        # Intraday 5-min ATR
        daily_atr = self._cycle_atr.get(symbol, price * 0.01)
        if n_bars >= INTRA_ATR_MIN_BARS:
            intra_atr_s = compute_atr(
                today_bars["high"].astype(float),
                today_bars["low"].astype(float),
                today_bars["close"].astype(float),
                period=INTRA_ATR_PERIOD,
            )
            valid = intra_atr_s.dropna()
            intra_atr = float(valid.iloc[-1]) if not valid.empty else daily_atr * INTRA_TO_DAILY
        else:
            intra_atr = daily_atr * INTRA_TO_DAILY
        intra_atr = max(intra_atr, 1e-4)

        # Morning trend bias (first 60 min = 12 bars)
        atr_pct = daily_atr / price if price > 0 else 0.01
        if n_bars <= 12 and price > 0:
            raw_b = (price - open_price) / open_price / max(atr_pct, 1e-6)
            morning_bias = float(np.clip(raw_b, -MORNING_DRIFT_CAP, MORNING_DRIFT_CAP))
        else:
            morning_bias = 0.0

        avg5v     = self._cycle_avg5vol.get(symbol, 1.0)
        vol_ratio = vol_last / avg5v if avg5v > 0 else 1.0

        return {
            "price":          price,
            "bar_high":       bar_high,
            "bar_low":        bar_low,
            "open_price":     open_price,
            "vwap":           vwap,
            "vol_imbalance":  vol_imbalance,
            "morning_bias":   morning_bias,
            "vol_ratio":      vol_ratio,
            "intra_atr":      intra_atr,
            "n_bars":         n_bars,
        }

    # ------------------------------------------------------------------
    # Range bounds computation (uses intraday ATR)
    # ------------------------------------------------------------------
    def _range_bounds(self, intra: dict) -> dict:
        intra_atr = intra["intra_atr"]
        vix       = self._cycle_vix

        vol_drift      = (intra["vol_imbalance"] - 0.5) * VOL_DRIFT_SCALE * intra_atr
        predicted_mean = intra["vwap"] + vol_drift + intra["morning_bias"] * intra_atr

        vix_mult    = float(np.clip(vix / 20.0, 0.6, 1.6))
        range_width = intra_atr * vix_mult
        lower = predicted_mean - RANGE_MULT * range_width
        upper = predicted_mean + RANGE_MULT * range_width

        return {"lower": lower, "predicted_mean": predicted_mean, "upper": upper}

    # ------------------------------------------------------------------
    # Main per-symbol logic
    # ------------------------------------------------------------------
    def check_and_trade(
        self,
        symbol:     str,
        positions:  Dict[str, dict],
        allocation: float,
    ) -> str:
        pos = positions.get(symbol)

        adx   = self._cycle_adx.get(symbol, 99.0)
        hurst = self._cycle_hurst.get(symbol, 0.6)
        vix   = self._cycle_vix

        # Time stop: close any position approaching end of session
        if _time_stop_imminent() and pos is not None:
            qty  = int(pos["qty"])
            side = pos["side"]
            cp   = pos["current_price"]
            if side == "LONG":
                self._sell(symbol, qty, reason="time_stop", price=cp)
            else:
                self._cover(symbol, qty, reason="time_stop", price=cp)
            pnl_pct = pos["pnl_pct"]
            self._record(pnl_pct)
            self._clear(symbol)
            return f"EXIT  (time stop {pnl_pct:+.2%})  H={hurst:.2f} ADX={adx:.0f}"

        # Fetch intraday context
        intra = self._intraday_context(symbol)
        if intra is None:
            return f"SKIP  (intraday data unavailable)  H={hurst:.2f}"

        price     = intra["price"]
        bar_high  = intra["bar_high"]
        bar_low   = intra["bar_low"]
        vol_ratio = intra["vol_ratio"]
        intra_atr = intra["intra_atr"]
        bounds    = self._range_bounds(intra)
        lower     = bounds["lower"]
        p_mean    = bounds["predicted_mean"]
        upper     = bounds["upper"]

        # Bar-level reversal confirmation (buyers won / sellers won this bar)
        bar_range    = max(bar_high - bar_low, 1e-6)
        bar_buy_frac = (price - bar_low) / bar_range

        # ============================================================
        # MANAGE OPEN POSITION
        # ============================================================
        if pos is not None:
            qty     = int(pos["qty"])
            side    = pos["side"]
            ep      = pos["entry_price"]
            pnl_pct = pos["pnl_pct"]
            cp      = pos["current_price"]
            stop_px = self._entry_stop.get(symbol, 0.0)

            # Activate breakeven once sufficiently profitable
            if pnl_pct >= BREAKEVEN_PCT and not self._breakeven.get(symbol):
                self._breakeven[symbol] = True
                log.info("Breakeven activated for %s at %+.2f%%", symbol, pnl_pct * 100)

            # Breakeven stop
            if self._breakeven.get(symbol) and pnl_pct <= -BREAKEVEN_BUFFER:
                if side == "LONG":
                    self._sell(symbol, qty, reason="breakeven_stop", price=cp)
                else:
                    self._cover(symbol, qty, reason="breakeven_stop", price=cp)
                self._record(pnl_pct)
                self._clear(symbol)
                return f"EXIT  (breakeven stop {pnl_pct:+.2%})  RANGE {side}"

            # Profit target: price returned to predicted mean at entry
            if side == "LONG" and cp >= p_mean:
                self._sell(symbol, qty, reason="profit_target", price=cp)
                self._record(pnl_pct)
                self._clear(symbol)
                return f"EXIT  (profit target {pnl_pct:+.2%}, mean=${p_mean:.2f})"

            if side == "SHORT" and cp <= p_mean:
                self._cover(symbol, qty, reason="profit_target", price=cp)
                self._record(pnl_pct)
                self._clear(symbol)
                return f"EXIT  (profit target {pnl_pct:+.2%}, mean=${p_mean:.2f})"

            # Hard stop: fixed distance from entry
            if side == "LONG" and cp < stop_px:
                self._sell(symbol, qty, reason="stop_loss", price=cp)
                self._record(pnl_pct)
                self._clear(symbol)
                return f"EXIT  (stop loss {pnl_pct:+.2%}, broke ${stop_px:.2f})"
            if side == "SHORT" and cp > stop_px:
                self._cover(symbol, qty, reason="stop_loss", price=cp)
                self._record(pnl_pct)
                self._clear(symbol)
                return f"EXIT  (stop loss {pnl_pct:+.2%}, broke ${stop_px:.2f})"

            return (f"HOLD  ({side} {qty}sh @ ${ep:.2f}, P&L:{pnl_pct:+.2%})  "
                    f"stop=${stop_px:.2f}  target=${p_mean:.2f}  "
                    f"H={hurst:.2f} ADX={adx:.0f} VIX={vix:.1f}")

        # ============================================================
        # ENTRY LOGIC -- no position
        # ============================================================
        if not _in_trade_window():
            return f"TIME BLOCK  (outside trade window)  H={hurst:.2f}"

        # Regime gate
        if hurst >= HURST_THRESHOLD:
            return f"SKIP  (trending H={hurst:.2f}>={HURST_THRESHOLD})  ADX={adx:.0f}"
        if adx >= ADX_THRESHOLD:
            return f"SKIP  (trending ADX={adx:.0f}>={ADX_THRESHOLD})  H={hurst:.2f}"
        if vix >= VIX_MAX:
            return f"SKIP  (panic VIX={vix:.1f}>={VIX_MAX})"
        if vol_ratio < VOL_RATIO_MIN:
            return f"SKIP  (thin vol ratio={vol_ratio:.2f}<{VOL_RATIO_MIN})"

        # Sizing
        daily_atr  = self._cycle_atr.get(symbol, price * 0.01)
        vol20      = self._cycle_vol20.get(symbol, 0.15)
        vol_scalar = min(2.0, max(0.3, TARGET_VOL / vol20)) if vol20 > 0 else 1.0
        h_scalar   = 1.0 - max(0.0, (hurst - 0.45) * 4.0)
        kelly      = self._kelly()
        sizing_pct = min(self.position_pct * vol_scalar * h_scalar * kelly, 0.25)
        invest     = allocation * sizing_pct
        qty        = int(invest / price)
        if qty <= 0:
            return f"SKIP  (insufficient allocation ${allocation:,.0f})  H={hurst:.2f}"

        # LONG: reversal bar at lower bound (buyers won this bar)
        if (bar_low <= lower
                and price > lower
                and bar_buy_frac >= 0.50):
            stop_px = price - STOP_MULT * intra_atr
            self._buy(symbol, qty, price=price)
            self._entry_price[symbol]  = price
            self._entry_stop[symbol]   = stop_px
            self._entry_bounds[symbol] = bounds
            self._breakeven[symbol]    = False
            self._open_trade_day[symbol] = _today_et()
            return (f"BUY  ({qty}sh @ ~${price:.2f}, ${qty*price:,.0f}, "
                    f"size={sizing_pct:.0%})  "
                    f"target=${p_mean:.2f}  stop=${stop_px:.2f}  "
                    f"H={hurst:.2f} ADX={adx:.0f} VIX={vix:.1f}")

        # SHORT: reversal bar at upper bound (sellers won this bar)
        if (bar_high >= upper
                and price < upper
                and bar_buy_frac <= 0.50):
            stop_px = price + STOP_MULT * intra_atr
            self._short(symbol, qty, price=price)
            self._entry_price[symbol]  = price
            self._entry_stop[symbol]   = stop_px
            self._entry_bounds[symbol] = bounds
            self._breakeven[symbol]    = False
            self._open_trade_day[symbol] = _today_et()
            return (f"SHORT  ({qty}sh @ ~${price:.2f}, ${qty*price:,.0f}, "
                    f"size={sizing_pct:.0%})  "
                    f"target=${p_mean:.2f}  stop=${stop_px:.2f}  "
                    f"H={hurst:.2f} ADX={adx:.0f} VIX={vix:.1f}")

        return (f"WAIT  (price ${price:.2f} not at bounds "
                f"[{lower:.2f} / {upper:.2f}])  "
                f"mean=${p_mean:.2f}  frac={bar_buy_frac:.2f}  "
                f"H={hurst:.2f} ADX={adx:.0f} VIX={vix:.1f}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run_loop(self) -> None:
        log.info("Starting Range Trader (PAPER) -- %d symbols", len(self.symbols))
        log.info("Symbols:  %s", ", ".join(self.symbols))
        log.info("Regime gate:  H < %.2f  ADX < %.0f  VIX < %.0f",
                 HURST_THRESHOLD, ADX_THRESHOLD, VIX_MAX)
        log.info("Range: +/-%.2f x intra_ATR | Stop: %.2f x intra_ATR from entry | "
                 "Time stop: %d:%02d ET",
                 RANGE_MULT, STOP_MULT, TIME_STOP_HOUR, TIME_STOP_MIN)
        log.info("Budget: %.0f%% of equity | Sizing: %.0f%% per symbol x vol x Hurst x Kelly",
                 BUDGET_PCT * 100, self.position_pct * 100)
        print("\n  Press Ctrl+C to stop.\n")

        cycle = 0
        while self._running:
            session = _get_session()
            if session == "closed":
                wait = _time_until_next_session()
                print(f"  Market closed -- next session in {wait}. Sleeping 10 min.")
                time.sleep(600)
                continue

            cycle += 1
            try:
                account   = self._account()
                positions = self._positions()

                # 30% budget cap: total range allocation split per symbol
                total_budget = account["equity"] * BUDGET_PCT
                alloc_per    = total_budget / max(len(self.symbols), 1)

                print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"=== Range Trading Cycle #{cycle} ===")
                print(f"  Account: ${account['equity']:,.2f} equity | "
                      f"Budget (30%): ${total_budget:,.2f} | "
                      f"Per symbol: ${alloc_per:,.2f}")
                print()

                # Refresh VIX once per cycle
                self._refresh_vix()
                print(f"  VIX: {self._cycle_vix:.1f}")
                print()

                # Refresh daily context once per cycle per symbol
                for sym in self.symbols:
                    self._refresh_daily_context(sym)

                # Evaluate each symbol
                for sym in self.symbols:
                    if sym not in self._cycle_atr:
                        print(f"    {sym:6s}:  SKIP  (daily context unavailable)")
                        continue
                    action = self.check_and_trade(sym, positions, alloc_per)
                    print(f"    {sym:6s}:  {action}")

            except Exception as exc:
                log.error("Cycle error: %s", exc, exc_info=True)

            time.sleep(self.check_interval)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Range Trader -- intraday mean reversion (Alpaca paper)"
    )
    parser.add_argument("--symbols",
                        default=",".join(DEFAULT_SYMBOLS),
                        help="Comma-separated symbols (default: SPY,QQQ,IWM,SOXX)")
    parser.add_argument("--provider",
                        default="alpaca", choices=["alpaca", "yahoo", "hybrid"],
                        help="Data provider (default: alpaca)")
    parser.add_argument("--position-pct",
                        type=float, default=POSITION_PCT,
                        help=f"Max budget fraction per symbol (default: {POSITION_PCT})")
    parser.add_argument("--check-interval",
                        type=int, default=CHECK_INTERVAL // 60,
                        help="Check interval in minutes (default: 5)")
    args = parser.parse_args()

    # API keys: prefer intraday group keys, fall back to generic
    api_key = (
        os.environ.get("ALPACA_INTRADAY_KEY")
        or os.environ.get("ALPACA_API_KEY", "")
    )
    api_secret = (
        os.environ.get("ALPACA_INTRADAY_SECRET")
        or os.environ.get("ALPACA_API_SECRET", "")
    )

    if not api_key or not api_secret:
        print("ERROR: Set ALPACA_INTRADAY_KEY + ALPACA_INTRADAY_SECRET "
              "(or ALPACA_API_KEY + ALPACA_API_SECRET).")
        sys.exit(1)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    trader = RangeTrader(
        symbols        = symbols,
        api_key        = api_key,
        api_secret     = api_secret,
        provider       = args.provider,
        fred_key       = os.environ.get("FRED_API_KEY"),
        position_pct   = args.position_pct,
        check_interval = args.check_interval * 60,
    )
    trader.run_loop()


if __name__ == "__main__":
    main()
