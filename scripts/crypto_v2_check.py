#!/usr/bin/env python3
"""Quick v2 system verification dry run."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.environ.setdefault("FRED_API_KEY", "5e06c25a712146a59c69804dc0cdec4c")

from paper_trader import (SYMBOL_GROUPS, _crypto_to_yfinance,
                          CRYPTO_VOL_THRESHOLD_K, CRYPTO_TARGET_VOL_ANN,
                          CRYPTO_MAX_HOLD_DAYS, CRYPTO_TRAILING_TRIGGER_ATR,
                          CRYPTO_TRAILING_WIDTH_ATR, CRYPTO_FUNDING_THRESHOLD)
from utils import CRYPTO_MODEL_DIR, _fetch_vix_for_training
from swing_model import SwingPredictor, FORWARD_DAYS
from signals_engine import YahooFinanceAdapter, compute_atr
from crypto_sentiment import get_funding_rate
import numpy as np

adapter = YahooFinanceAdapter()
fred_key = os.environ.get("FRED_API_KEY")
vix_df = _fetch_vix_for_training(fred_key, lookback_days=400)

print("=" * 70)
print("  CRYPTO STRATEGY V2 - COMPLETE SYSTEM VERIFICATION")
print("=" * 70)

# Regime
btc_bars = adapter.fetch_daily("BTC-USD", 250)
btc_price = float(btc_bars["close"].iloc[-1])
btc_sma200 = float(btc_bars["close"].rolling(200).mean().iloc[-1])
regime_ok = btc_price > btc_sma200
regime_str = "BULL" if regime_ok else "BEAR"
print(f"\n  REGIME: {regime_str} (BTC ${btc_price:,.0f} vs SMA200 ${btc_sma200:,.0f})")

# Per-symbol
total_equity = 384535.0
vols = {}
for sym in SYMBOL_GROUPS["crypto"]:
    yf_sym = _crypto_to_yfinance(sym)
    bars = adapter.fetch_daily(yf_sym, 400)
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    price = float(close.iloc[-1])
    atr = float(compute_atr(high, low, close, period=14).iloc[-1])
    rv = float(close.pct_change().rolling(30).std().iloc[-1]) * np.sqrt(365)
    vols[sym] = rv
    sma50 = float(close.rolling(50).mean().iloc[-1])
    trend = "+" if price > sma50 else "-"
    vol_thr = CRYPTO_VOL_THRESHOLD_K * atr / price
    fr = get_funding_rate(sym)
    fr_str = f"{fr:+.6f}" if fr is not None else "N/A"

    predictor = SwingPredictor(yf_sym, model_dir=CRYPTO_MODEL_DIR)
    result = predictor.predict(bars, vix_df)
    er = result["expected_return"]
    xgb_r = result.get("xgb_return", 0)
    tft_r = result.get("tft_return")
    tft_w = result.get("tft_weight", 0.6)

    signal_str = min(1.0, max(0.1, abs(er) / 0.02))
    vol_scale = min(2.0, CRYPTO_TARGET_VOL_ANN / rv) if rv > 0 else 1.0
    sizing = min(1.0, signal_str * vol_scale)

    tft_str = f"{tft_r:+.4f}" if tft_r is not None else "   N/A"

    print(f"\n  {sym}:")
    print(f"    Price=${price:,.2f}  ATR=${atr:,.2f}  RV={rv:.0%}  Trend={trend}  Funding={fr_str}")
    print(f"    XGB={xgb_r:+.4f}  TFT={tft_str}  w_tft={tft_w:.2f}  E[r]={er:+.4f}")
    print(f"    VolThreshold={vol_thr:.4f}  Pass={abs(er) > vol_thr}  Sizing={sizing:.0%}")

    if not regime_ok:
        print(f"    --> BLOCKED (bearish regime)")
    elif abs(er) <= vol_thr:
        print(f"    --> SKIP (signal below vol threshold)")
    elif (trend == "+" and er > 0) or (trend == "-" and er < 0):
        direction = "LONG" if er > 0 else "SHORT"
        print(f"    --> WOULD {direction}")
    else:
        print(f"    --> SKIP (trend disagrees)")

# Allocation
inv_vols = {s: 1.0 / v for s, v in vols.items()}
total_inv = sum(inv_vols.values())
weights = {s: iv / total_inv for s, iv in inv_vols.items()}
print(f"\n  ALLOCATION (inv-vol):")
for s in SYMBOL_GROUPS["crypto"]:
    print(f"    {s}: {weights[s]:.1%} = ${total_equity * weights[s]:,.0f}")

print(f"\n  SYSTEM CONFIG:")
print(f"    Prediction horizon: {FORWARD_DAYS}d | Vol threshold k={CRYPTO_VOL_THRESHOLD_K}")
print(f"    Target vol: {CRYPTO_TARGET_VOL_ANN:.0%} | Max hold: {CRYPTO_MAX_HOLD_DAYS}d")
print(f"    Trail trigger: {CRYPTO_TRAILING_TRIGGER_ATR}xATR | Trail width: {CRYPTO_TRAILING_WIDTH_ATR}xATR")
print(f"    Funding threshold: +/-{CRYPTO_FUNDING_THRESHOLD}")
print(f"    Ensemble: dynamic MAE (window=30)")

status = "BULL" if regime_ok else "BEAR (waiting)"
print(f"\n  STATUS: All systems operational. Regime={status}")
print("=" * 70)
