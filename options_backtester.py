#!/usr/bin/env python3
"""
Options Backtester — Black-Scholes synthetic options backtest.
Validates two options strategies using existing LSTM direction signals + VolPredictor.

Strategy A — Directional (ITM calls/puts):
    Entry: tradeable=True + confidence >= threshold + IV rank <= 55%
    Strike: delta ~0.68 (ITM) via BS d1 approximation
    Expiry: 28 DTE
    Exit: +50% profit, -25% stop, 7 DTE forced, direction flip

Strategy B — Vol Expansion Straddle:
    Entry: confidence < 0.45 + IV rank <= 25% + vol_expanding=True
    Strike: ATM (round(current_price))
    Expiry: 30 DTE
    Exit: +80% on either leg, -40% total, 7 DTE forced

No historical options chain required — uses Black-Scholes with VIX as IV proxy.

Usage (via main.py):
    python main.py backtest-options --symbol SPY --start 2024-01-01
    python main.py backtest-options --symbol QQQ --start 2024-01-01 --strategy directional
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from signals_engine import build_adapter, DAILY_LOOKBACK
from ml_model import Predictor, _fetch_vix_for_training, DEFAULT_MODEL_DIR
from options_ml import VolPredictor
from options_trader import GreeksEstimator, _norm_cdf, _norm_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("options_backtester")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Per-symbol IV scaling: effective_iv = VIX/100 * IV_SCALE[symbol]
# SPY options trade at roughly 1:1 with VIX; tech ETFs are higher vol.
IV_SCALE: Dict[str, float] = {
    "SPY":  1.00, "QQQ":  1.10, "IWM":  1.15, "SOXX": 1.20,
    "GLD":  0.75, "SLV":  1.20, "XLE":  1.25,
    "EWT":  1.10, "EWS":  1.10, "EEM":  1.15, "EWJ":  1.00, "INDA": 1.20,
}
IV_FLOOR = 0.10          # minimum sigma even when VIX is low
BUY_SLIPPAGE  = 1.02     # 2% wider on entry
SELL_SLIPPAGE = 0.98     # 2% wider on exit
R = 0.05                 # risk-free rate assumption


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class OptionTrade:
    symbol: str
    strategy: str           # "DIRECTIONAL" or "STRADDLE"
    option_type: str        # "CALL", "PUT", or "STRADDLE"
    direction: str          # "UP", "DOWN", or "ANY"
    entry_date: object
    expiry_date: object     # date object
    strike: float
    entry_cost: float       # total premium × 100 contracts (incl. slippage)
    underlying_at_entry: float
    entry_vix: float
    # For straddle: track legs separately
    call_cost: float = 0.0
    put_cost: float = 0.0
    call_open: bool = True
    put_open: bool = True
    # Exit tracking
    exit_date: object = None
    exit_value: float = 0.0
    pnl: float = 0.0
    exit_reason: str = ""


@dataclass
class OptionsPortfolio:
    initial_capital: float = 100_000.0
    cash: float = 100_000.0
    dir_trade: Optional[OptionTrade] = None      # one directional trade at a time
    straddle_trade: Optional[OptionTrade] = None  # one straddle at a time
    closed_trades: List[OptionTrade] = field(default_factory=list)
    equity_curve: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# BS helpers
# ---------------------------------------------------------------------------
def _effective_sigma(symbol: str, vix: float) -> float:
    scale = IV_SCALE.get(symbol, 1.0)
    return max(IV_FLOOR, (vix / 100.0) * scale)


def _call_delta(S: float, K: float, T: float, sigma: float) -> float:
    """N(d1) — Black-Scholes call delta approximation."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.5
    d1 = (math.log(S / K) + (R + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)


def _find_target_delta_strike(
    is_call: bool, current_price: float, T: float, sigma: float,
    target_delta: float = 0.68,
) -> float:
    """Find strike where |delta| ≈ target_delta.

    For calls: delta = N(d1) ≈ target_delta → strike slightly below current_price (ITM)
    For puts:  |delta| = 1 - N(d1) ≈ target_delta → N(d1) ≈ 1-target_delta → ITM put

    Returns the strike rounded to nearest $5 for underlyings ≥ $100.
    """
    step = 5.0 if current_price >= 100 else 1.0
    # For puts, target call-side N(d1) = 1 - target_delta
    call_side_target = target_delta if is_call else (1.0 - target_delta)

    # Search in the direction of ITM
    best_strike = round(current_price / step) * step
    best_diff = float("inf")

    # Calls ITM → strike below price; Puts ITM → strike above price
    lo = current_price - 50 if is_call else current_price
    hi = current_price       if is_call else current_price + 50

    k = lo
    while k <= hi:
        delta = _call_delta(current_price, k, T, sigma)
        diff = abs(delta - call_side_target)
        if diff < best_diff:
            best_diff = diff
            best_strike = k
        k += step

    return best_strike


def _option_bs_value(
    is_call: bool, S: float, K: float, T: float, sigma: float,
) -> float:
    """Black-Scholes option price per share (×100 = per contract)."""
    if T <= 0:
        # Intrinsic value at expiry
        return max(0.0, S - K) if is_call else max(0.0, K - S)
    if is_call:
        return GreeksEstimator.call_price(S, K, T, R, sigma)
    return GreeksEstimator.put_price(S, K, T, R, sigma)


# ---------------------------------------------------------------------------
# Natenberg / Sinclair / Davey helper functions
# ---------------------------------------------------------------------------
def _option_theta(
    is_call: bool, S: float, K: float, T: float, sigma: float,
) -> float:
    """Black-Scholes daily theta per share (always negative for long options).

    Natenberg (*Option Volatility & Pricing*, Ch.7): theta is the daily rent
    paid to hold a long option.  Checking theta at entry ensures the trade has
    enough expected move to overcome time decay.

    Returns theta per calendar day (divide annualized theta by 365).
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (R + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    theta_ann = (
        -(S * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
        - R * K * math.exp(-R * T) * (_norm_cdf(d2) if is_call else _norm_cdf(-d2))
    )
    return theta_ann / 365.0


def _kelly_fraction(
    confidence: float, profit_target: float, stop_loss: float,
) -> float:
    """Fractional Kelly (¼ Kelly) for position sizing.

    Sinclair (*Volatility Trading*, Ch.8) and Davey (*Building Winning
    Algorithmic Trading Systems*): use ¼ Kelly to hedge against model
    mis-estimation and to smooth the equity curve.

        Kelly f = p − (1−p) / b
        where p = win probability, b = win/loss ratio

    Returns a scale factor in [0, 1] applied to max_risk.
    Used for STRADDLE sizing where vol_prob is in [0.40, 1.0] — a valid
    win-probability directly (no remapping needed).
    """
    p = min(max(float(confidence), 0.10), 0.95)
    b = profit_target / stop_loss if stop_loss > 0 else 2.0
    raw_kelly = p - (1.0 - p) / b
    raw_kelly = max(0.0, min(raw_kelly, 1.0))
    return raw_kelly * 0.25          # quarter-Kelly per Davey


def _kelly_fraction_directional(
    confidence: float, profit_target: float, stop_loss: float,
    conf_lo: float = 0.03, conf_hi: float = 0.12,
) -> float:
    """Fractional Kelly (¼ Kelly) for DIRECTIONAL options sizing.

    The direction LSTM outputs compressed probabilities in [0.03, 0.12]
    due to sigmoid saturation at the extremes. Feeding these directly into
    Kelly gives negative fractions (p=0.10, b=2.0 → f=-0.35 → 0), which
    blocks ALL directional entries.

    Fix: remap [conf_lo, conf_hi] → [0.55, 0.75] before Kelly.
    Calibration basis: the meta-RF-gated direction models achieve
    71–80% realized win rates in daily backtests (see MEMORY.md).
    - At conf=conf_lo (0.03): p_mapped=0.55 — minimal edge
    - At conf=conf_hi (0.12): p_mapped=0.75 — strong directional edge

    Returns a scale factor in [0, 1] applied to max_risk.
    """
    t = (float(confidence) - conf_lo) / max(conf_hi - conf_lo, 1e-6)
    t = max(0.0, min(1.0, t))
    p = 0.55 + 0.20 * t  # linear remap: [0.55, 0.75]
    b = profit_target / stop_loss if stop_loss > 0 else 2.0
    raw_kelly = p - (1.0 - p) / b
    raw_kelly = max(0.0, min(raw_kelly, 1.0))
    return raw_kelly * 0.25          # quarter-Kelly per Davey


def _iv_rv_ratio(symbol: str, vix: float, rv: float) -> float:
    """IV / RV ratio: how expensive options are relative to recent realized vol.

    Sinclair (*Volatility Trading*, Ch.2): the core edge metric.
    - ratio < 1.0  → IV below RV → buying edge (cheap premium)
    - ratio > 1.25 → IV meaningfully above RV → options over-priced for longs

    Uses the same per-symbol IV scaling as the backtester constants.
    """
    scale = IV_SCALE.get(symbol, 1.0)
    iv = max(IV_FLOOR, (vix / 100.0) * scale)
    rv = max(IV_FLOOR, rv)
    return iv / rv


# ---------------------------------------------------------------------------
# Rolling IV rank helper
# ---------------------------------------------------------------------------
def _compute_iv_rank_series(vix_df: pd.DataFrame, window: int = 252) -> pd.Series:
    """Return rolling IV rank (0-100) as a Series indexed by date."""
    if vix_df.empty:
        return pd.Series(dtype=float)
    vix_series = vix_df.set_index("date")["vix"]
    lo = vix_series.rolling(window, min_periods=20).min()
    hi = vix_series.rolling(window, min_periods=20).max()
    rank = (vix_series - lo) / (hi - lo + 1e-9) * 100
    return rank.clip(0, 100)


# ---------------------------------------------------------------------------
# Options Backtester
# ---------------------------------------------------------------------------
class OptionsBacktester:
    """Walk-forward options strategy backtester using Black-Scholes pricing.

    Strategy A (directional): buy ITM call/put when LSTM has high confidence.
    Strategy B (straddle):    buy ATM straddle when vol model expects expansion at low IV.
    """

    def __init__(
        self,
        symbol: str,
        adapter,
        fred_key: Optional[str] = None,
        model_dir: str = DEFAULT_MODEL_DIR,
        initial_capital: float = 100_000.0,
        confidence_threshold: float = 0.10,
        iv_rank_max_directional: float = 55.0,
        iv_rank_max_straddle: float = 25.0,
        target_delta: float = 0.68,
        expiry_days_directional: int = 28,
        expiry_days_straddle: int = 30,
        max_risk_pct: float = 0.05,          # directional ITM options (need $1,500–$3,000 budget)
        max_risk_pct_straddle: float = 0.03,  # straddle budget kept tight to preserve filter quality
        profit_target_directional: float = 0.50,
        stop_loss_directional: float = 0.25,
        profit_target_straddle: float = 0.80,
        stop_loss_straddle: float = 0.40,
        vol_prob_min: float = 0.40,
        strategy: str = "both",   # "directional" | "straddle" | "both"
        mode: str = "daily",
    ):
        self.symbol = symbol
        self.adapter = adapter
        self.fred_key = fred_key
        self.model_dir = model_dir
        self.initial_capital = initial_capital
        self.confidence_threshold = confidence_threshold
        self.iv_rank_max_directional = iv_rank_max_directional
        self.iv_rank_max_straddle = iv_rank_max_straddle
        self.target_delta = target_delta
        self.expiry_days_directional = expiry_days_directional
        self.expiry_days_straddle = expiry_days_straddle
        self.max_risk_pct = max_risk_pct
        self.max_risk_pct_straddle = max_risk_pct_straddle
        self.profit_target_directional = profit_target_directional
        self.stop_loss_directional = stop_loss_directional
        self.profit_target_straddle = profit_target_straddle
        self.stop_loss_straddle = stop_loss_straddle
        self.vol_prob_min = vol_prob_min
        self.strategy = strategy
        self.mode = mode

    def run(self, start_date: str, end_date: Optional[str] = None) -> dict:
        """Run the backtest and return a results dict."""
        # ── 1. Fetch data ──
        lookback = 1200
        log.info("Fetching daily bars for %s (lookback=%d)...", self.symbol, lookback)
        bars = self.adapter.fetch_daily(self.symbol, lookback)
        log.info("Got %d bars.", len(bars))

        # VIX needs to cover the full bars window PLUS the tail(400) history used per bar.
        # bars = 1200 trading days ≈ 1750 calendar days; add 400 trading-day buffer ≈ 580 days.
        vix_lookback = lookback * 2   # generous calendar-day cover (1200*2=2400 days ≈ 6.5 yrs)
        vix_df = _fetch_vix_for_training(self.fred_key, lookback_days=vix_lookback)
        log.info("Got %d VIX rows.", len(vix_df))

        # ── 2. Load LSTM predictor ──
        predictor = Predictor(self.symbol, model_dir=self.model_dir, mode=self.mode)
        vol_predictor = VolPredictor(self.symbol, model_dir=self.model_dir)

        # ── 3. Build rolling IV rank series and realized vol lookup ──
        iv_rank_series = _compute_iv_rank_series(vix_df, window=252)
        vix_by_date: Dict = {}
        if not vix_df.empty:
            for _, row in vix_df.iterrows():
                d = row["date"]
                if hasattr(d, "date"):
                    d = d.date()
                vix_by_date[d] = row["vix"]

        # Rolling 20-bar realized vol for IV/RV ratio (Sinclair Ch.2)
        close_series = pd.to_numeric(bars["close"], errors="coerce")
        # Normalize to tz-naive to avoid UTC vs. naive comparison errors
        bar_ts_raw = pd.to_datetime(bars["ts"])
        if hasattr(bar_ts_raw, "dt") and bar_ts_raw.dt.tz is not None:
            bar_ts_raw = bar_ts_raw.dt.tz_localize(None)
        bar_ts_series = bar_ts_raw
        rv_series = close_series.pct_change().rolling(20).std() * np.sqrt(252)
        rv_by_date: Dict = {}
        for ts_val, rv_val in zip(bar_ts_series, rv_series):
            if pd.isna(rv_val):
                continue
            d = ts_val.date() if hasattr(ts_val, "date") else ts_val
            rv_by_date[d] = float(rv_val)

        # ── 4. Walk-forward loop ──
        start = pd.Timestamp(start_date)
        end   = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()

        # Only consider bars within the backtest window (tz-naive comparison)
        bar_dates = bar_ts_series
        valid_mask = (bar_dates >= start) & (bar_dates <= end)
        # Keep original index so backtest_bars.index[idx] gives the true position in
        # the full bars DataFrame, which is needed to build the correct rolling history window.
        backtest_bars = bars[valid_mask]

        if len(backtest_bars) < 40:
            log.warning("Only %d bars in backtest window — too few.", len(backtest_bars))
            return {}

        portfolio = OptionsPortfolio(
            initial_capital=self.initial_capital,
            cash=self.initial_capital,
        )

        for idx in range(40, len(backtest_bars)):   # need 40 bars of history for features
            bar = backtest_bars.iloc[idx]
            bar_date = pd.Timestamp(bar["ts"]).date()
            current_price = float(bar["close"])

            # Get VIX at this date
            vix_today = vix_by_date.get(bar_date, 20.0)
            if vix_today == 0 or np.isnan(vix_today):
                vix_today = 20.0

            # Compute IV rank at this date
            iv_rank = 50.0
            rank_val = iv_rank_series.get(bar_date, None)
            if rank_val is None:
                # Try by timestamp
                for k in iv_rank_series.index:
                    if hasattr(k, "date") and k.date() == bar_date:
                        rank_val = iv_rank_series[k]
                        break
            if rank_val is not None and not np.isnan(rank_val):
                iv_rank = float(rank_val)

            # Get LSTM prediction using bars up to current idx.
            # Need tail(400) min: frac_diff warmup=282, vol EWMA=~50, MACD=~34 bars.
            history_bars = bars.iloc[:bars.index.get_loc(backtest_bars.index[idx]) + 1
                                    if hasattr(bars.index, 'get_loc')
                                    else max(0, len(bars) - len(backtest_bars) + idx + 1)].tail(400)
            try:
                pred = predictor.predict(history_bars, vix_df)
                direction   = pred["direction"]
                confidence  = pred["confidence"]
                tradeable   = pred.get("tradeable", True)
            except Exception:
                continue

            # Get vol prediction
            vol_pred: dict = {}
            try:
                vol_pred = vol_predictor.predict(history_bars, vix_df)
                vol_expanding = vol_pred.get("vol_expanding", False)
            except Exception:
                vol_expanding = False

            sigma = _effective_sigma(self.symbol, vix_today)

            # IV/RV ratio for Sinclair edge gating (Ch.2)
            rv_today = rv_by_date.get(bar_date, 0.15)
            iv_rv = _iv_rv_ratio(self.symbol, vix_today, rv_today)

            # ─── Manage open positions ───
            if portfolio.dir_trade is not None:
                portfolio.dir_trade = self._manage_directional(
                    portfolio, bar_date, current_price, sigma, vix_today,
                    direction, confidence, pred,
                )

            if portfolio.straddle_trade is not None:
                portfolio.straddle_trade = self._manage_straddle(
                    portfolio, bar_date, current_price, sigma,
                )

            # ─── Entry logic ───
            equity = self._equity(portfolio, bar_date, current_price, sigma)
            max_risk = equity * self.max_risk_pct

            # Strategy A: Directional ITM call/put
            # tradeable: meta-RF gate (precision-calibrated) is now a hard entry condition.
            # IV/RV <= 1.75: filter extreme fear spikes where options are hugely over-priced
            # (Sinclair Ch.2 / Natenberg Ch.4 — don't overpay for directional premium)
            if (
                self.strategy in ("directional", "both")
                and portfolio.dir_trade is None
                and tradeable                              # meta-RF hard gate (was soft Kelly boost)
                and confidence >= self.confidence_threshold
                and iv_rank <= self.iv_rank_max_directional
                and iv_rv <= 1.75
                and direction in ("UP", "DOWN")
            ):
                T = self.expiry_days_directional / 365.0
                is_call = (direction == "UP")
                strike = _find_target_delta_strike(is_call, current_price, T, sigma, self.target_delta)
                price_per_share = _option_bs_value(is_call, current_price, strike, T, sigma)
                entry_cost = price_per_share * 100 * BUY_SLIPPAGE

                # Theta budget check: daily theta < 1.5% of entry cost (Natenberg Ch.7)
                theta_daily = abs(_option_theta(is_call, current_price, strike, T, sigma)) * 100
                theta_pct = theta_daily / entry_cost if entry_cost > 0 else 1.0

                # Kelly-proportional sizing using remapped LSTM confidence (Sinclair Ch.8 / Davey)
                # Uses _kelly_fraction_directional: remaps [0.03, 0.12] → [0.55, 0.75] before Kelly
                # so negative-Kelly collapse (p=0.10, b=2.0 → f=-0.35) does not block entries.
                kelly = _kelly_fraction_directional(
                    confidence, self.profit_target_directional, self.stop_loss_directional,
                )
                kelly_scale = 0.5 + 0.5 * kelly        # range [0.50, 0.625] of max_risk
                risk_budget = max_risk * kelly_scale

                if entry_cost > 0 and entry_cost <= risk_budget and theta_pct <= 0.015:
                    expiry_dt = date.fromisoformat(str(bar_date)) + timedelta(days=self.expiry_days_directional)
                    trade = OptionTrade(
                        symbol=self.symbol, strategy="DIRECTIONAL",
                        option_type="CALL" if is_call else "PUT",
                        direction=direction, entry_date=bar_date, expiry_date=expiry_dt,
                        strike=strike, entry_cost=entry_cost,
                        underlying_at_entry=current_price, entry_vix=vix_today,
                    )
                    portfolio.cash -= entry_cost
                    portfolio.dir_trade = trade
                    log.info(
                        "%s  OPEN %s %s  strike=%.1f  cost=$%.0f  IVR=%.0f  "
                        "IV/RV=%.2f  conf=%.3f  kelly=%.2f  θ/cost=%.2f%%",
                        bar_date, direction, "CALL" if is_call else "PUT",
                        strike, entry_cost, iv_rank,
                        iv_rv, confidence, kelly, theta_pct * 100,
                    )

            # Strategy B: Vol Expansion Straddle
            # IV/RV <= 1.50: buy straddles only when premium is not severely elevated
            # Historical equity VRP 1.2–1.5×; 1.50 blocks high-fear spikes (Sinclair Ch.2)
            # vol_prob >= 0.40: higher bar than 0.30 to reduce false-positive straddle entries.
            # Default 0.0 means no signal if vol predictor failed or returned empty.
            vol_prob = vol_pred.get("vol_expansion_prob", 0.0)
            if (
                self.strategy in ("straddle", "both")
                and portfolio.straddle_trade is None
                and confidence < 0.45
                and iv_rank <= self.iv_rank_max_straddle
                and iv_rv <= 1.50
                and vol_prob >= self.vol_prob_min
            ):
                T = self.expiry_days_straddle / 365.0
                atm_strike = round(current_price / 5) * 5 if current_price >= 100 else round(current_price)
                call_cost = _option_bs_value(True,  current_price, atm_strike, T, sigma) * 100 * BUY_SLIPPAGE
                put_cost  = _option_bs_value(False, current_price, atm_strike, T, sigma) * 100 * BUY_SLIPPAGE
                total_cost = call_cost + put_cost

                # Kelly sizing using vol expansion probability (Sinclair Ch.8)
                # Uses separate straddle risk budget (3%) to preserve filter tightness.
                # The higher directional max_risk_pct (5%) must not bleed into straddle sizing.
                kelly_s = _kelly_fraction(
                    vol_prob, self.profit_target_straddle, self.stop_loss_straddle,
                )
                kelly_scale_s = 0.5 + 0.5 * kelly_s
                max_risk_s = equity * self.max_risk_pct_straddle
                risk_budget_s = max_risk_s * kelly_scale_s

                if total_cost > 0 and total_cost <= risk_budget_s:
                    expiry_dt = date.fromisoformat(str(bar_date)) + timedelta(days=self.expiry_days_straddle)
                    trade = OptionTrade(
                        symbol=self.symbol, strategy="STRADDLE",
                        option_type="STRADDLE", direction="ANY",
                        entry_date=bar_date, expiry_date=expiry_dt,
                        strike=atm_strike, entry_cost=total_cost,
                        underlying_at_entry=current_price, entry_vix=vix_today,
                        call_cost=call_cost, put_cost=put_cost,
                    )
                    portfolio.cash -= total_cost
                    portfolio.straddle_trade = trade
                    log.info(
                        "%s  OPEN STRADDLE  strike=%.1f  cost=$%.0f  IVR=%.0f  "
                        "IV/RV=%.2f  vol_exp=%.2f  kelly=%.2f",
                        bar_date, atm_strike, total_cost, iv_rank,
                        iv_rv, vol_prob, kelly_s,
                    )

            # Record equity
            eq = self._equity(portfolio, bar_date, current_price, sigma)
            portfolio.equity_curve.append({"date": bar_date, "equity": eq})

        # Close open positions at end
        if portfolio.dir_trade is not None:
            self._force_close_directional(portfolio, bar_date, current_price, sigma, "end_of_backtest")
        if portfolio.straddle_trade is not None:
            self._force_close_straddle(portfolio, bar_date, current_price, sigma, "end_of_backtest")

        return self._compute_results(portfolio, start_date, bar_date)

    # ────────────────────────────────────────────────────
    # Position management
    # ────────────────────────────────────────────────────
    def _manage_directional(
        self, portfolio, bar_date, current_price, sigma, vix_today,
        direction, confidence, pred,
    ) -> Optional[OptionTrade]:
        trade = portfolio.dir_trade
        dte = max(0, (trade.expiry_date - bar_date).days)
        T = dte / 365.0
        is_call = (trade.option_type == "CALL")

        current_value = _option_bs_value(is_call, current_price, trade.strike, T, sigma) * 100

        # Force close at 7 DTE
        if dte <= 7:
            self._force_close_directional(portfolio, bar_date, current_price, sigma, "7_dte")
            return None

        pnl_mult = current_value / trade.entry_cost if trade.entry_cost > 0 else 1.0

        # Profit target: +50%
        if pnl_mult >= (1.0 + self.profit_target_directional):
            self._force_close_directional(portfolio, bar_date, current_price, sigma, "profit_target")
            return None

        # Stop loss: -25%
        if pnl_mult <= (1.0 - self.stop_loss_directional):
            self._force_close_directional(portfolio, bar_date, current_price, sigma, "stop_loss")
            return None

        # Direction flip: LSTM confidence >= 0.50 in OPPOSITE direction
        if pred.get("tradeable", True) and confidence >= 0.50:
            if (is_call and direction == "DOWN") or (not is_call and direction == "UP"):
                self._force_close_directional(portfolio, bar_date, current_price, sigma, "direction_flip")
                return None

        return trade

    def _force_close_directional(self, portfolio, bar_date, current_price, sigma, reason):
        trade = portfolio.dir_trade
        if trade is None:
            return
        dte = max(0, (trade.expiry_date - bar_date).days)
        T = dte / 365.0
        is_call = (trade.option_type == "CALL")
        exit_value = _option_bs_value(is_call, current_price, trade.strike, T, sigma) * 100 * SELL_SLIPPAGE
        trade.exit_date = bar_date
        trade.exit_value = exit_value
        trade.pnl = exit_value - trade.entry_cost
        trade.exit_reason = reason
        portfolio.cash += exit_value
        portfolio.closed_trades.append(trade)
        ret_pct = (trade.pnl / trade.entry_cost * 100) if trade.entry_cost > 0 else 0
        log.info("%s  CLOSE %s  reason=%s  pnl=$%.0f  (%.1f%%)",
                 bar_date, trade.option_type, reason, trade.pnl, ret_pct)
        portfolio.dir_trade = None

    def _manage_straddle(self, portfolio, bar_date, current_price, sigma) -> Optional[OptionTrade]:
        trade = portfolio.straddle_trade
        dte = max(0, (trade.expiry_date - bar_date).days)
        T = dte / 365.0

        if dte <= 7:
            self._force_close_straddle(portfolio, bar_date, current_price, sigma, "7_dte")
            return None

        call_val = _option_bs_value(True,  current_price, trade.strike, T, sigma) * 100 if trade.call_open else 0.0
        put_val  = _option_bs_value(False, current_price, trade.strike, T, sigma) * 100 if trade.put_open  else 0.0
        total_val = call_val + put_val

        call_mult = (call_val / trade.call_cost) if trade.call_cost > 0 else 1.0
        put_mult  = (put_val  / trade.put_cost)  if trade.put_cost  > 0 else 1.0
        total_mult = (total_val / trade.entry_cost) if trade.entry_cost > 0 else 1.0

        # Profit target: either leg hits +80%
        if (trade.call_open and call_mult >= 1.0 + self.profit_target_straddle) or \
           (trade.put_open  and put_mult  >= 1.0 + self.profit_target_straddle):
            self._force_close_straddle(portfolio, bar_date, current_price, sigma, "profit_target")
            return None

        # Stop loss: total -40%
        if total_mult <= (1.0 - self.stop_loss_straddle):
            self._force_close_straddle(portfolio, bar_date, current_price, sigma, "stop_loss")
            return None

        # McMillan rule — close losing leg when winning leg is ≥ 3× the loser.
        # (*Options as a Strategic Investment*, straddle management):
        # Lock in the winner's gains by closing the dead weight and letting
        # the profitable leg run unhedged.
        if trade.call_open and trade.put_open and call_val > 0 and put_val > 0:
            if call_val >= 3.0 * put_val:
                put_close = _option_bs_value(False, current_price, trade.strike, T, sigma) * 100 * SELL_SLIPPAGE
                portfolio.cash += put_close
                trade.put_open = False
                trade.entry_cost = max(trade.entry_cost - trade.put_cost, trade.call_cost)
                trade.put_cost = 0.0
                log.info("%s  CLOSE PUT leg (call≥3× put)  put_val=$%.0f  call_val=$%.0f",
                         bar_date, put_val, call_val)
            elif put_val >= 3.0 * call_val:
                call_close = _option_bs_value(True, current_price, trade.strike, T, sigma) * 100 * SELL_SLIPPAGE
                portfolio.cash += call_close
                trade.call_open = False
                trade.entry_cost = max(trade.entry_cost - trade.call_cost, trade.put_cost)
                trade.call_cost = 0.0
                log.info("%s  CLOSE CALL leg (put≥3× call)  call_val=$%.0f  put_val=$%.0f",
                         bar_date, call_val, put_val)

        return trade

    def _force_close_straddle(self, portfolio, bar_date, current_price, sigma, reason):
        trade = portfolio.straddle_trade
        if trade is None:
            return
        dte = max(0, (trade.expiry_date - bar_date).days)
        T = dte / 365.0
        call_val = _option_bs_value(True,  current_price, trade.strike, T, sigma) * 100 * SELL_SLIPPAGE if trade.call_open else 0.0
        put_val  = _option_bs_value(False, current_price, trade.strike, T, sigma) * 100 * SELL_SLIPPAGE if trade.put_open  else 0.0
        exit_value = call_val + put_val
        trade.exit_date = bar_date
        trade.exit_value = exit_value
        trade.pnl = exit_value - trade.entry_cost
        trade.exit_reason = reason
        portfolio.cash += exit_value
        portfolio.closed_trades.append(trade)
        ret_pct = (trade.pnl / trade.entry_cost * 100) if trade.entry_cost > 0 else 0
        log.info("%s  CLOSE STRADDLE  reason=%s  pnl=$%.0f  (%.1f%%)",
                 bar_date, reason, trade.pnl, ret_pct)
        portfolio.straddle_trade = None

    def _equity(self, portfolio, bar_date, current_price, sigma) -> float:
        eq = portfolio.cash
        if portfolio.dir_trade is not None:
            t = portfolio.dir_trade
            dte = max(0, (t.expiry_date - bar_date).days)
            T = dte / 365.0
            is_call = (t.option_type == "CALL")
            eq += _option_bs_value(is_call, current_price, t.strike, T, sigma) * 100
        if portfolio.straddle_trade is not None:
            t = portfolio.straddle_trade
            dte = max(0, (t.expiry_date - bar_date).days)
            T = dte / 365.0
            cv = _option_bs_value(True,  current_price, t.strike, T, sigma) * 100 if t.call_open else 0.0
            pv = _option_bs_value(False, current_price, t.strike, T, sigma) * 100 if t.put_open  else 0.0
            eq += cv + pv
        return eq

    # ────────────────────────────────────────────────────
    # Results computation and reporting
    # ────────────────────────────────────────────────────
    def _compute_results(self, portfolio, start_date_str, end_date_val) -> dict:
        closed = portfolio.closed_trades

        def _strategy_stats(trades):
            if not trades:
                return {}
            pnls = [t.pnl for t in trades]
            costs = [t.entry_cost for t in trades]
            ret_pcts = [(p / c * 100) if c > 0 else 0 for p, c in zip(pnls, costs)]
            wins = [r for r in ret_pcts if r > 0]
            losses = [r for r in ret_pcts if r <= 0]
            gross_profit = sum(p for p in pnls if p > 0)
            gross_loss   = abs(sum(p for p in pnls if p < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
            win_rate = len(wins) / len(trades) if trades else 0
            avg_win  = float(np.mean(wins))   if wins   else 0.0
            avg_loss = float(np.mean(losses)) if losses else 0.0
            total_pnl = sum(pnls)
            return {
                "n_trades": len(trades), "win_rate": win_rate,
                "avg_win_pct": avg_win, "avg_loss_pct": avg_loss,
                "profit_factor": profit_factor, "total_pnl": total_pnl,
            }

        dir_trades     = [t for t in closed if t.strategy == "DIRECTIONAL"]
        straddle_trades = [t for t in closed if t.strategy == "STRADDLE"]
        dir_stats      = _strategy_stats(dir_trades)
        straddle_stats = _strategy_stats(straddle_trades)

        # Overall equity curve metrics
        if portfolio.equity_curve:
            eq_vals = [e["equity"] for e in portfolio.equity_curve]
            final_eq = eq_vals[-1]
            total_ret = (final_eq - self.initial_capital) / self.initial_capital * 100

            # Sharpe ratio
            eq_series = pd.Series(eq_vals)
            daily_rets = eq_series.pct_change().dropna()
            sharpe = float(daily_rets.mean() / daily_rets.std() * np.sqrt(252)) if len(daily_rets) > 1 and daily_rets.std() > 0 else 0.0

            # Max drawdown
            roll_max = eq_series.cummax()
            drawdown = (eq_series - roll_max) / roll_max * 100
            max_dd = float(drawdown.min())
        else:
            final_eq = self.initial_capital
            total_ret = 0.0
            sharpe = 0.0
            max_dd = 0.0

        return {
            "symbol": self.symbol,
            "start_date": start_date_str,
            "end_date": str(end_date_val),
            "initial_capital": self.initial_capital,
            "final_equity": round(final_eq, 2),
            "total_return_pct": round(total_ret, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "strategy": self.strategy,
            "directional": dir_stats,
            "straddle": straddle_stats,
            "equity_curve": portfolio.equity_curve,
            "closed_trades": closed,
        }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------
def _print_report(results: dict) -> None:
    if not results:
        print("  No results to report.")
        return

    bar = "=" * 62
    dash = "-" * 62

    print()
    print(f"  OPTIONS BACKTEST REPORT")
    print(f"  {bar}")
    print(f"  Symbol             : {results['symbol']}")
    print(f"  Period             : {results['start_date']} to {results['end_date']}")
    print(f"  Initial Capital    : ${results['initial_capital']:,.2f}")
    print(f"  Final Equity       : ${results['final_equity']:,.2f}")
    print(f"  {dash}")
    print(f"  Total Return       : {results['total_return_pct']:+.2f}%")
    print(f"  Sharpe Ratio       : {results['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown       : {results['max_drawdown_pct']:.2f}%")
    print(f"  {bar}")

    def _section(name, stats, entry_info):
        if not stats or stats.get("n_trades", 0) == 0:
            print(f"\n  {name}")
            print(f"  {dash}")
            print(f"  No trades executed.")
            return
        print(f"\n  {name}")
        # encode-safe print for Windows cp1252 consoles
        safe_info = entry_info.encode("ascii", errors="replace").decode("ascii")
        print(f"  {safe_info}")
        print(f"  {dash}")
        print(f"  Total Trades       : {stats['n_trades']}")
        print(f"  Win Rate           : {stats['win_rate']:.1%}")
        print(f"  Avg Win            : {stats['avg_win_pct']:+.1f}%")
        print(f"  Avg Loss           : {stats['avg_loss_pct']:+.1f}%")
        print(f"  Profit Factor      : {stats['profit_factor']:.3f}")
        print(f"  Total P&L          : ${stats['total_pnl']:+,.0f}")

    _section(
        "STRATEGY A — Directional (ITM calls/puts)",
        results.get("directional"),
        "  Entry: tradeable + confidence ≥ threshold + IV rank ≤ 55%",
    )
    _section(
        "STRATEGY B — Vol Expansion Straddle",
        results.get("straddle"),
        "  Entry: confidence < 0.45 + IV rank ≤ 25% + vol_expanding=True",
    )
    print(f"\n  {bar}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Options Backtester — BS-synthetic options strategy validation.",
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=None,  help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--provider", default="yahoo", choices=["yahoo", "alpaca", "hybrid"])
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--confidence", type=float, default=0.10,
                        help="Min LSTM confidence for directional entry (default: 0.10; LSTM range is 0.03-0.12)")
    parser.add_argument("--iv-rank-max", type=float, default=55.0,
                        help="Max IV rank for directional entry (default: 55)")
    parser.add_argument("--iv-rank-straddle", type=float, default=25.0,
                        help="Max IV rank for straddle entry (default: 25)")
    parser.add_argument("--target-delta", type=float, default=0.68,
                        help="Target call/put delta for directional strike (default: 0.68)")
    parser.add_argument("--max-risk-pct", type=float, default=0.05,
                        help="Max risk per trade as fraction of equity (default: 0.05 = 5%%)")
    parser.add_argument("--vol-prob-min", type=float, default=0.40,
                        help="Min vol LSTM probability for straddle entry (default: 0.40)")
    parser.add_argument("--strategy", default="both",
                        choices=["directional", "straddle", "both"])
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--mode", default="daily", choices=["daily"])

    args = parser.parse_args()

    adapter  = build_adapter(args.provider)
    fred_key = os.environ.get("FRED_API_KEY")

    bt = OptionsBacktester(
        symbol=args.symbol.upper(),
        adapter=adapter,
        fred_key=fred_key,
        model_dir=args.model_dir,
        initial_capital=args.initial_capital,
        confidence_threshold=args.confidence,
        iv_rank_max_directional=args.iv_rank_max,
        iv_rank_max_straddle=args.iv_rank_straddle,
        target_delta=args.target_delta,
        max_risk_pct=args.max_risk_pct,
        vol_prob_min=args.vol_prob_min,
        strategy=args.strategy,
        mode=args.mode,
    )

    results = bt.run(start_date=args.start, end_date=args.end)
    _print_report(results)


if __name__ == "__main__":
    main()
