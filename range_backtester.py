#!/usr/bin/env python3
"""
Range Backtester — Intraday Mean Reversion Strategy
=====================================================
Validates the range trading strategy on historical 5-min bars before live trading.

Strategy logic:
  - Regime gate:  H < 0.52 (mean-reverting), ADX < 20 (no strong trend), VIX < 25
  - Mean price:   VWAP adjusted by volume imbalance and morning trend
  - Range bounds: predicted_mean ± 0.45 × ATR (VIX-scaled)
  - LONG entry:   price ≤ lower_bound, sellers not overwhelming
  - SHORT entry:  price ≥ upper_bound, buyers not overwhelming
  - Exits:        profit target (return to mean), stop loss (hard ATR floor), time stop 15:15 ET

Usage:
    python main.py range-backtest --symbols SPY,QQQ --start 2024-01-01
    python main.py range-backtest --symbols SPY,QQQ,IWM,SOXX,EWT,GLD,EEM,SLV,EWJ,EWS,XLE,INDA \\
        --start 2024-01-01 --provider alpaca
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from signals_engine import (
    build_adapter,
    compute_atr,
    compute_adx_full,
    compute_hurst_exponent,
    compute_vwap,
    compute_volume_imbalance,
)
from ml_model import _fetch_vix_for_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("range_backtester")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SYMBOLS = [
    "SPY", "QQQ", "IWM", "SOXX",
    "EWT", "GLD", "EEM", "SLV",
    "EWJ", "EWS", "XLE", "INDA",
]

INITIAL_CAPITAL   = 100_000.0
POSITION_PCT      = 0.10        # max 10% of equity per symbol
TARGET_VOL        = 0.15        # 15% annualised vol target for scaling

# Regime gates (loosened — prior thresholds eliminated 95% of days)
HURST_THRESHOLD   = 0.62        # H < 0.62 = mean-reverting (was 0.52)
ADX_THRESHOLD     = 28.0        # ADX < 28 = no strong trend (was 20)
VIX_MAX           = 30.0        # VIX >= 30 = extreme panic, skip (was 25)
VOL_RATIO_MIN     = 0.40        # bar volume >= 40% of avg 5-min volume (was 0.50)

# Range geometry — now uses intraday 5-min ATR, not daily ATR
RANGE_MULT        = 2.00        # bounds at +/-2.0 x intra_ATR (wider = fewer but cleaner signals)
STOP_MULT         = 1.00        # stop = 1.0 x intra_ATR below entry price (fixed-distance, R:R ~ 2)
INTRA_ATR_PERIOD  = 10          # ATR look-back on 5-min bars
INTRA_ATR_MIN_BARS = 15         # need ≥15 intraday bars before computing intra_ATR
INTRA_TO_DAILY    = 0.06        # fallback ratio: intra_ATR ≈ 6% of daily ATR

# Range geometry adjustments
VOL_DRIFT_SCALE   = 0.30        # volume imbalance adjusts mean by ≤ 30% of intra_ATR
MORNING_DRIFT_CAP = 0.20        # morning trend bias capped at ±20% of intra_ATR

# Exit rules
BREAKEVEN_PCT     = 0.003       # once +0.3%, activate breakeven protection
BREAKEVEN_BUFFER  = 0.002       # exit if falls to -0.2% after breakeven
TIME_STOP_HOUR    = 15          # force-exit at 15:15 ET
TIME_STOP_MIN     = 15

# Data requirements
DAILY_BARS_MIN    = 30          # need >= 30 daily bars for reliable ADX / Hurst
MARKET_OPEN_HOUR  = 10          # skip 9:30-10:00 opening noise


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------
@dataclass
class RangeTrade:
    symbol:         str
    entry_date:     str
    entry_time:     str
    entry_price:    float
    direction:      str     # "LONG" | "SHORT"
    qty:            int
    lower_bound:    float
    predicted_mean: float
    upper_bound:    float
    atr:            float
    hurst:          float
    adx:            float
    vix:            float
    exit_date:      str   = ""
    exit_time:      str   = ""
    exit_price:     float = 0.0
    exit_reason:    str   = ""
    pnl:            float = 0.0
    pnl_pct:        float = 0.0


# ---------------------------------------------------------------------------
# Timezone helpers
# ---------------------------------------------------------------------------
_ET = "America/New_York"

def _to_et(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(_ET)

def _et_hm(ts) -> Tuple[int, int]:
    t = _to_et(ts)
    return t.hour, t.minute

def _after_open(ts) -> bool:
    h, m = _et_hm(ts)
    return (h, m) >= (MARKET_OPEN_HOUR, 0)

def _before_close(ts) -> bool:
    h, m = _et_hm(ts)
    return (h, m) < (TIME_STOP_HOUR, TIME_STOP_MIN)


# ---------------------------------------------------------------------------
# Main backtester
# ---------------------------------------------------------------------------
class RangeBacktester:
    """
    Walk-forward intraday mean-reversion backtester.

    For each trading day:
      1. Compute regime (Hurst, ADX, VIX) from daily bars up to that date.
      2. If ranging regime: compute range bounds from ATR + predicted VWAP.
      3. Simulate bar-by-bar: enter at bounds, exit at mean / stop / time.
    """

    def __init__(self, provider: str = "alpaca", fred_key: Optional[str] = None):
        self.adapter  = build_adapter(provider)
        self.fred_key = fred_key or os.environ.get("FRED_API_KEY")

    # ------------------------------------------------------------------
    # VIX helpers
    # ------------------------------------------------------------------
    def _fetch_vix(self, lookback_days: int = 800) -> pd.DataFrame:
        """Return DataFrame with columns ['date', 'vix']; date is tz-naive."""
        try:
            vix_df = _fetch_vix_for_training(self.fred_key, lookback_days=lookback_days)
            if vix_df is None or vix_df.empty:
                return pd.DataFrame(columns=["date", "vix"])
            vix_df["date"] = pd.to_datetime(vix_df["date"]).dt.normalize()
            return vix_df.sort_values("date").reset_index(drop=True)
        except Exception as exc:
            log.warning("VIX fetch failed: %s", exc)
            return pd.DataFrame(columns=["date", "vix"])

    @staticmethod
    def _vix_for_date(vix_df: pd.DataFrame, date) -> float:
        """Return VIX value for a date using forward-fill for holidays/weekends."""
        if vix_df.empty:
            return 20.0
        d = pd.Timestamp(date).normalize()
        prior = vix_df[vix_df["date"] <= d]
        return float(prior["vix"].iloc[-1]) if not prior.empty else 20.0

    # ------------------------------------------------------------------
    # Daily-bar context (regime + sizing params)
    # ------------------------------------------------------------------
    def _daily_context(self, daily: pd.DataFrame, as_of: pd.Timestamp) -> Optional[dict]:
        """
        Compute ATR, ADX, Hurst, vol20 from daily bars strictly before `as_of`.
        Returns None if insufficient history.
        """
        mask = daily["ts"] < as_of
        bars = daily[mask].tail(60)
        if len(bars) < DAILY_BARS_MIN:
            return None

        hi = bars["high"].astype(float)
        lo = bars["low"].astype(float)
        cl = bars["close"].astype(float)

        atr_s  = compute_atr(hi, lo, cl, period=14)
        atr    = float(atr_s.iloc[-1]) if not atr_s.dropna().empty else np.nan
        if pd.isna(atr) or atr <= 0:
            return None

        adx_s, _, _ = compute_adx_full(hi, lo, cl, period=14)
        adx = float(adx_s.dropna().iloc[-1]) if not adx_s.dropna().empty else np.nan

        hurst = compute_hurst_exponent(cl)

        rets  = cl.pct_change().dropna()
        vol20 = float(rets.tail(20).std() * math.sqrt(252)) if len(rets) >= 5 else 0.15

        # Average 5-min bar volume estimate: daily vol ÷ 78 bars/day
        avg_daily_vol = float(bars["volume"].tail(20).mean()) if "volume" in bars else 0.0
        avg_5min_vol  = max(avg_daily_vol / 78.0, 1.0)

        return {
            "atr":          atr,
            "adx":          adx,
            "hurst":        hurst,
            "vol20":        vol20,
            "avg_5min_vol": avg_5min_vol,
        }

    # ------------------------------------------------------------------
    # Single-day simulation
    # ------------------------------------------------------------------
    def _simulate_day(
        self,
        symbol:    str,
        day_bars:  pd.DataFrame,
        ctx:       dict,
        vix:       float,
        equity:    float,
    ) -> List[RangeTrade]:
        """Simulate one trading day bar by bar. Returns completed trades."""
        trades: List[RangeTrade] = []

        atr          = ctx["atr"]        # daily ATR — used for position sizing
        adx          = ctx["adx"]
        hurst        = ctx["hurst"]
        vol20        = ctx["vol20"]
        avg_5min_vol = ctx["avg_5min_vol"]

        # --- Regime gate (daily context) ---
        if hurst >= HURST_THRESHOLD:
            return trades   # trending — momentum system's job
        if adx >= ADX_THRESHOLD:
            return trades   # directional — momentum system's job
        if vix >= VIX_MAX:
            return trades   # extreme panic — don't range-trade

        # Filter to regular trading window
        bars = day_bars[
            day_bars["ts"].apply(_after_open) &
            day_bars["ts"].apply(_before_close)
        ].reset_index(drop=True)

        if len(bars) < 5:
            return trades

        open_price = float(bars.iloc[0]["open"])

        # --- Position sizing (uses daily metrics for scale) ---
        vol_scalar = min(2.0, max(0.3, TARGET_VOL / vol20)) if vol20 > 0 else 1.0
        h_scalar   = 1.0 - max(0.0, (hurst - 0.45) * 4.0)
        sizing_pct = min(POSITION_PCT * vol_scalar * h_scalar, 0.25)
        invest     = equity * sizing_pct

        position = None  # dict or None

        for idx in range(len(bars)):
            row   = bars.iloc[idx]
            ts    = row["ts"]
            price = float(row["close"])
            vol   = float(row["volume"]) if not pd.isna(row.get("volume", np.nan)) else 0.0

            # --- Intraday 5-min ATR (switches from fallback once we have enough bars) ---
            if idx >= INTRA_ATR_MIN_BARS:
                i_bars = bars.iloc[:idx + 1]
                intra_atr_s = compute_atr(
                    i_bars["high"].astype(float),
                    i_bars["low"].astype(float),
                    i_bars["close"].astype(float),
                    period=INTRA_ATR_PERIOD,
                )
                valid = intra_atr_s.dropna()
                intra_atr = float(valid.iloc[-1]) if not valid.empty else atr * INTRA_TO_DAILY
            else:
                intra_atr = atr * INTRA_TO_DAILY   # fallback: ~6% of daily ATR
            intra_atr = max(intra_atr, 1e-4)

            bar_low   = float(row.get("low",  price))
            bar_high  = float(row.get("high", price))

            # --- VWAP and volume imbalance from PRIOR bars only ---
            # Using strictly prior bars keeps VWAP as a stable equilibrium anchor;
            # if we include the current bar, the bound tracks the current price
            # and bar_low can never reach the lower_bound.
            bars_prior = bars.iloc[:idx]
            if bars_prior.empty:
                vwap = open_price
            else:
                vwap = compute_vwap(bars_prior)
                if pd.isna(vwap) or vwap <= 0:
                    vwap = open_price

            vi_raw        = compute_volume_imbalance(bars_prior) if not bars_prior.empty else 0.0
            vol_imbalance = (vi_raw + 1) / 2.0                      # remap to [0, 1]

            # Morning trend bias (first 60 min = 12 bars), scaled by intra_atr
            if idx < 12 and price > 0:
                atr_pct      = atr / price
                raw_bias     = (price - open_price) / open_price / max(atr_pct, 1e-6)
                morning_bias = float(np.clip(raw_bias, -MORNING_DRIFT_CAP, MORNING_DRIFT_CAP))
            else:
                morning_bias = 0.0

            # --- Predicted mean (vol drift and bias now scale with intra_atr) ---
            vol_drift      = (vol_imbalance - 0.5) * VOL_DRIFT_SCALE * intra_atr
            predicted_mean = vwap + vol_drift + morning_bias * intra_atr

            # --- Range bounds (intraday ATR × VIX scale) ---
            vix_mult    = float(np.clip(vix / 20.0, 0.6, 1.6))
            range_width = intra_atr * vix_mult
            lower_bound = predicted_mean - RANGE_MULT * range_width
            upper_bound = predicted_mean + RANGE_MULT * range_width

            vol_ratio = vol / avg_5min_vol if avg_5min_vol > 0 else 1.0

            # ============================================================
            # MANAGE OPEN POSITION
            # ============================================================
            if position is not None:
                side   = position["side"]
                ep     = position["entry_price"]
                qty    = position["qty"]
                p_mean = position["predicted_mean"]
                # Stop anchored below/above the bar low/high at entry
                stop_long  = position["stop_price"]
                stop_short = position["stop_price"]

                pnl_pct = (price - ep) / ep if side == "LONG" else (ep - price) / ep

                # Activate breakeven once up BREAKEVEN_PCT
                if pnl_pct >= BREAKEVEN_PCT and not position["breakeven"]:
                    position["breakeven"] = True

                # Breakeven stop: fell back after being profitable
                if position["breakeven"] and pnl_pct <= -BREAKEVEN_BUFFER:
                    trades.append(self._close(position, price, ts, "breakeven_stop"))
                    position = None
                    continue

                # Profit target: price returns to predicted mean at entry
                if side == "LONG" and price >= p_mean:
                    trades.append(self._close(position, price, ts, "profit_target"))
                    position = None
                    continue
                if side == "SHORT" and price <= p_mean:
                    trades.append(self._close(position, price, ts, "profit_target"))
                    position = None
                    continue

                # Hard stop: anchored at bar extreme at entry ± STOP_MULT × intra_atr
                if side == "LONG" and bar_low < stop_long:
                    trades.append(self._close(position, stop_long, ts, "stop_loss"))
                    position = None
                    continue
                if side == "SHORT" and bar_high > stop_short:
                    trades.append(self._close(position, stop_short, ts, "stop_loss"))
                    position = None
                    continue

                # Time stop: approaching close
                h, m = _et_hm(ts)
                if (h, m) >= (TIME_STOP_HOUR, TIME_STOP_MIN):
                    trades.append(self._close(position, price, ts, "time_stop"))
                    position = None
                    continue

                continue  # hold

            # ============================================================
            # ENTRY LOGIC — reversal-bar confirmation
            # Bar low/high touches the bound but close recovers inside:
            #   LONG:  bar_low  <= lower_bound AND close > lower_bound  (level held)
            #   SHORT: bar_high >= upper_bound AND close < upper_bound  (level held)
            # This filters "falling knives" without needing RSI momentum.
            # ============================================================
            if vol_ratio < VOL_RATIO_MIN:
                continue  # thin market — skip

            qty = int(invest / price)
            if qty <= 0:
                continue

            # Bar-level reversal confirmation: close in upper half of bar range
            bar_range    = max(bar_high - bar_low, 1e-6)
            bar_buy_frac = (price - bar_low) / bar_range   # 1.0 = closed at high (strong buy)

            # LONG reversal bar: touched lower bound, close in upper half of bar
            if (bar_low <= lower_bound
                    and price > lower_bound          # close recovered above bound
                    and bar_buy_frac >= 0.50):       # buyers won this bar
                stop_px = price - STOP_MULT * intra_atr  # fixed stop below entry close
                pos_dict = self._new_pos(
                    symbol, "LONG", price, qty, ts,
                    lower_bound, predicted_mean, upper_bound, atr, hurst, adx, vix,
                )
                pos_dict["intra_atr"]  = intra_atr
                pos_dict["stop_price"] = stop_px
                position = pos_dict

            # SHORT reversal bar: touched upper bound, close in lower half of bar
            elif (bar_high >= upper_bound
                    and price < upper_bound          # close recovered below bound
                    and bar_buy_frac <= 0.50):       # sellers won this bar
                stop_px = price + STOP_MULT * intra_atr  # fixed stop above entry close
                pos_dict = self._new_pos(
                    symbol, "SHORT", price, qty, ts,
                    lower_bound, predicted_mean, upper_bound, atr, hurst, adx, vix,
                )
                pos_dict["intra_atr"]  = intra_atr
                pos_dict["stop_price"] = stop_px
                position = pos_dict

        # End of day: close any leftover position
        if position is not None and len(bars) > 0:
            last = bars.iloc[-1]
            trades.append(self._close(position, float(last["close"]), last["ts"], "eod"))

        return trades

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _new_pos(symbol, side, price, qty, ts,
                 lower_bound, predicted_mean, upper_bound, atr, hurst, adx, vix) -> dict:
        return {
            "symbol":         symbol,
            "side":           side,
            "entry_price":    price,
            "qty":            qty,
            "entry_ts":       ts,
            "lower_bound":    lower_bound,
            "predicted_mean": predicted_mean,
            "upper_bound":    upper_bound,
            "atr":            atr,
            "hurst":          hurst,
            "adx":            adx,
            "vix":            vix,
            "breakeven":      False,
        }

    @staticmethod
    def _close(position: dict, exit_price: float, exit_ts, reason: str) -> RangeTrade:
        side = position["side"]
        ep   = position["entry_price"]
        qty  = position["qty"]
        pnl  = (exit_price - ep) * qty if side == "LONG" else (ep - exit_price) * qty
        pnl_pct = pnl / (ep * qty) if ep * qty > 0 else 0.0
        return RangeTrade(
            symbol         = position["symbol"],
            entry_date     = str(_to_et(position["entry_ts"]).date()),
            entry_time     = str(_to_et(position["entry_ts"]).time()),
            entry_price    = ep,
            direction      = side,
            qty            = qty,
            lower_bound    = position["lower_bound"],
            predicted_mean = position["predicted_mean"],
            upper_bound    = position["upper_bound"],
            atr            = position["atr"],
            hurst          = position["hurst"],
            adx            = position["adx"],
            vix            = position["vix"],
            exit_date      = str(_to_et(exit_ts).date()),
            exit_time      = str(_to_et(exit_ts).time()),
            exit_price     = exit_price,
            exit_reason    = reason,
            pnl            = round(pnl, 2),
            pnl_pct        = round(pnl_pct, 6),
        )

    # ------------------------------------------------------------------
    # Full run for one symbol
    # ------------------------------------------------------------------
    def run(
        self,
        symbol:   str,
        start:    str,
        end:      Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Backtest `symbol` from `start` to `end`.
        Returns (trades_df, equity_curve_df).
        """
        end_dt   = pd.Timestamp(end or datetime.now().strftime("%Y-%m-%d"))
        start_dt = pd.Timestamp(start)
        # Calendar days needed (backtest window + warmup for daily indicators)
        lookback = (datetime.now() - start_dt.to_pydatetime()).days + 90

        log.info("[%s] Fetching %d cal-days of 5min + daily data ...", symbol, lookback)

        try:
            intra = self.adapter.fetch_intraday(symbol, "5min", lookback_days=lookback)
        except Exception as exc:
            log.error("[%s] Intraday fetch failed: %s", symbol, exc)
            return pd.DataFrame(), pd.DataFrame()

        try:
            daily = self.adapter.fetch_daily(symbol, lookback=min(lookback, 504))
        except Exception as exc:
            log.error("[%s] Daily fetch failed: %s", symbol, exc)
            return pd.DataFrame(), pd.DataFrame()

        vix_df = self._fetch_vix(lookback_days=lookback + 30)

        # Normalise timestamps
        intra["ts"] = pd.to_datetime(intra["ts"], utc=True)
        daily["ts"] = pd.to_datetime(daily["ts"], utc=True)

        # Filter to backtest window
        s_utc = start_dt.tz_localize("UTC") if start_dt.tzinfo is None else start_dt
        e_utc = end_dt.tz_localize("UTC")   if end_dt.tzinfo   is None else end_dt
        intra = intra[(intra["ts"] >= s_utc) & (intra["ts"] <= e_utc)]
        if intra.empty:
            log.warning("[%s] No intraday data in range %s → %s", symbol, start, end)
            return pd.DataFrame(), pd.DataFrame()

        # Group by ET trading date
        intra["et_date"] = intra["ts"].dt.tz_convert(_ET).dt.date
        trading_days     = sorted(intra["et_date"].unique())

        equity    = INITIAL_CAPITAL
        eq_curve: List[dict] = [{"date": str(trading_days[0]), "equity": equity}]
        all_trades: List[RangeTrade] = []

        skipped_regime = 0
        for day in trading_days:
            day_ts  = pd.Timestamp(day).tz_localize(_ET).tz_convert("UTC")
            day_bars = intra[intra["et_date"] == day].reset_index(drop=True)
            if day_bars.empty:
                continue

            ctx = self._daily_context(daily, day_ts)
            if ctx is None:
                continue  # insufficient history

            vix = self._vix_for_date(vix_df, day)

            # Log regime skip for visibility
            if ctx["hurst"] >= HURST_THRESHOLD or ctx["adx"] >= ADX_THRESHOLD or vix >= VIX_MAX:
                skipped_regime += 1
                eq_curve.append({"date": str(day), "equity": equity})
                continue

            day_trades = self._simulate_day(symbol, day_bars, ctx, vix, equity)
            for t in day_trades:
                equity += t.pnl
                all_trades.append(t)
            eq_curve.append({"date": str(day), "equity": equity})

        log.info(
            "[%s] Done: %d trades on %d ranging days (%d days skipped — trending regime)",
            symbol, len(all_trades), len(trading_days) - skipped_regime, skipped_regime,
        )

        trades_df = (
            pd.DataFrame([t.__dict__ for t in all_trades])
            if all_trades else pd.DataFrame()
        )
        equity_df = pd.DataFrame(eq_curve)
        return trades_df, equity_df

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    @staticmethod
    def compute_stats(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> dict:
        if trades_df.empty or len(trades_df) == 0:
            return {
                "trades": 0, "win_rate": 0, "profit_factor": 0,
                "total_return_pct": 0, "sharpe": 0, "max_dd_pct": 0,
                "avg_win": 0, "avg_loss": 0,
            }

        n        = len(trades_df)
        wins_s   = trades_df[trades_df["pnl"] > 0]
        losses_s = trades_df[trades_df["pnl"] <= 0]
        win_rate = len(wins_s) / n

        avg_win  = float(wins_s["pnl"].mean())   if len(wins_s)   > 0 else 0.0
        avg_loss = float(losses_s["pnl"].abs().mean()) if len(losses_s) > 0 else 1.0
        pf       = (len(wins_s) * avg_win) / max(len(losses_s) * avg_loss, 1e-9)

        total_return = (
            (equity_df["equity"].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
            if not equity_df.empty else 0.0
        )

        sharpe = 0.0
        if len(equity_df) > 2:
            dr = equity_df["equity"].pct_change().dropna()
            sharpe = float(dr.mean() / dr.std() * math.sqrt(252)) if dr.std() > 0 else 0.0

        max_dd = 0.0
        if not equity_df.empty:
            eq   = equity_df["equity"].values
            peak = np.maximum.accumulate(eq)
            dd   = (eq - peak) / np.where(peak > 0, peak, 1)
            max_dd = float(dd.min())

        by_reason = (
            trades_df["exit_reason"].value_counts().to_dict()
            if "exit_reason" in trades_df.columns else {}
        )

        return {
            "trades":           n,
            "win_rate":         round(win_rate * 100, 1),
            "profit_factor":    round(pf, 2),
            "total_return_pct": round(total_return * 100, 2),
            "sharpe":           round(sharpe, 3),
            "max_dd_pct":       round(max_dd * 100, 2),
            "avg_win":          round(avg_win, 2),
            "avg_loss":         round(avg_loss, 2),
            "exit_reasons":     by_reason,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Range backtester — intraday mean reversion")
    parser.add_argument("--symbols",  default=",".join(DEFAULT_SYMBOLS),
                        help="Comma-separated symbols (default: all 12)")
    parser.add_argument("--start",    default="2024-01-01",
                        help="Backtest start date YYYY-MM-DD")
    parser.add_argument("--end",      default=None,
                        help="Backtest end date YYYY-MM-DD (default: today)")
    parser.add_argument("--provider", default="alpaca",
                        choices=["alpaca", "yahoo", "hybrid"],
                        help="Data provider (default: alpaca — needed for 2yr history)")
    parser.add_argument("--output-dir", default="outputs",
                        help="Directory to write CSV results (default: outputs/)")
    args = parser.parse_args()

    symbols  = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    fred_key = os.environ.get("FRED_API_KEY")
    bt       = RangeBacktester(provider=args.provider, fred_key=fred_key)

    os.makedirs(args.output_dir, exist_ok=True)

    all_stats: List[dict] = []
    print(f"\n  Range Backtest  {args.start} -> {args.end or 'today'}")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Regime gate: H < {HURST_THRESHOLD}, ADX < {ADX_THRESHOLD}, VIX < {VIX_MAX}")
    print(f"  Range: ±{RANGE_MULT}×ATR | Stop: {STOP_MULT}×ATR beyond bound | "
          f"Time stop: {TIME_STOP_HOUR}:{TIME_STOP_MIN:02d} ET\n")

    for sym in symbols:
        trades_df, equity_df = bt.run(sym, start=args.start, end=args.end)
        stats = bt.compute_stats(trades_df, equity_df)
        stats["symbol"] = sym
        all_stats.append(stats)

        # Save per-symbol CSVs
        if not trades_df.empty:
            tpath = os.path.join(args.output_dir, f"range_trades_{sym}.csv")
            trades_df.to_csv(tpath, index=False)
        if not equity_df.empty:
            epath = os.path.join(args.output_dir, f"range_equity_{sym}.csv")
            equity_df.to_csv(epath, index=False)

        reason_str = "  ".join(f"{k}={v}" for k, v in stats.get("exit_reasons", {}).items())
        print(
            f"  {sym:6s}  trades={stats['trades']:3d}  "
            f"WR={stats['win_rate']:5.1f}%  PF={stats['profit_factor']:5.2f}  "
            f"ret={stats['total_return_pct']:+6.1f}%  "
            f"Sharpe={stats['sharpe']:5.3f}  "
            f"maxDD={stats['max_dd_pct']:+5.1f}%  "
            f"[{reason_str}]"
        )

    # Summary CSV
    summary_df = pd.DataFrame(all_stats)
    summary_path = os.path.join(args.output_dir, "range_backtest_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
