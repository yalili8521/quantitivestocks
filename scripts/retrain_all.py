#!/usr/bin/env python3
"""
Retrain all models with mode-specific feature sets (FEATURE_COLS_DAILY / FEATURE_COLS_INTRADAY).

Usage:
    python scripts/retrain_all.py
    python scripts/retrain_all.py --step lstm       # LSTM only
    python scripts/retrain_all.py --step meta       # meta RF only (requires LSTM done)
    python scripts/retrain_all.py --step vol        # Vol Expansion LSTM + meta RF
    python scripts/retrain_all.py --step all        # lstm + meta + vol
    python scripts/retrain_all.py --mode daily      # daily symbols only
    python scripts/retrain_all.py --mode intraday   # intraday symbols only
"""

from __future__ import annotations

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ml_model import train_model, train_meta_model, DEFAULT_MODEL_DIR
from signals_engine import build_adapter
from options_ml import train_vol_model, train_vol_meta_model, ACTIVE_SYMBOLS as VOL_SYMBOLS

# ---------------------------------------------------------------------------
# Symbol universe
# ---------------------------------------------------------------------------
DAILY_SYMBOLS = [
    # Account 1 (Intraday account — daily models also needed for meta RF timing)
    "SPY", "QQQ", "IWM", "SOXX",
    # Account 2 (Swing — daily mode)
    "EWT", "GLD", "EEM", "SLV", "TLT",
    # Account 3 (Expansion — daily mode)
    "EWJ", "EWS", "IGV", "XLE", "INDA", "FXI",
]

INTRADAY_SYMBOLS = [
    "SPY", "QQQ", "IWM", "SOXX",
]

EPOCHS = 60
LOOKBACK = 1000

# ---------------------------------------------------------------------------

def train_all_lstm(mode: str, adapter, fred_key: str | None) -> None:
    symbols = INTRADAY_SYMBOLS if mode == "intraday" else DAILY_SYMBOLS
    interval = "5min"
    print(f"\n{'='*65}")
    print(f"  LSTM Training — {mode.upper()} ({len(symbols)} symbols)")
    print(f"  Features: {12 if mode == 'daily' else 13}")
    print(f"{'='*65}\n")

    failed = []
    for sym in symbols:
        print(f"\n--- Training {sym} ({mode}) ---")
        t0 = time.time()
        try:
            train_model(
                symbol=sym,
                adapter=adapter,
                fred_key=fred_key,
                epochs=EPOCHS,
                lookback=LOOKBACK,
                save_dir=DEFAULT_MODEL_DIR,
                mode=mode,
                intraday_interval=interval,
            )
            elapsed = time.time() - t0
            print(f"  [OK] {sym} ({mode}) — {elapsed:.0f}s")
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  [FAIL] {sym} ({mode}) after {elapsed:.0f}s: {exc}")
            failed.append(sym)

    print(f"\n{'='*65}")
    print(f"  LSTM {mode.upper()} done. Failed: {failed or 'none'}")
    print(f"{'='*65}\n")


def train_all_meta(mode: str, adapter, fred_key: str | None) -> None:
    symbols = INTRADAY_SYMBOLS if mode == "intraday" else DAILY_SYMBOLS
    interval = "5min"
    print(f"\n{'='*65}")
    print(f"  Meta RF Training — {mode.upper()} ({len(symbols)} symbols)")
    print(f"{'='*65}\n")

    failed = []
    for sym in symbols:
        print(f"\n--- Meta RF for {sym} ({mode}) ---")
        t0 = time.time()
        try:
            train_meta_model(
                symbol=sym,
                adapter=adapter,
                fred_key=fred_key,
                lookback=LOOKBACK,
                save_dir=DEFAULT_MODEL_DIR,
                mode=mode,
                intraday_interval=interval,
            )
            elapsed = time.time() - t0
            print(f"  [OK] {sym} ({mode}) meta RF — {elapsed:.0f}s")
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  [FAIL] {sym} ({mode}) meta RF after {elapsed:.0f}s: {exc}")
            failed.append(sym)

    print(f"\n{'='*65}")
    print(f"  Meta RF {mode.upper()} done. Failed: {failed or 'none'}")
    print(f"{'='*65}\n")


def train_all_vol(adapter, fred_key: str | None, with_meta: bool = True) -> None:
    """Train Vol Expansion LSTM (+ optional meta RF) for all active options symbols."""
    print(f"\n{'='*65}")
    print(f"  Vol Expansion LSTM Training ({len(VOL_SYMBOLS)} symbols)")
    print(f"{'='*65}\n")

    failed_lstm, failed_meta = [], []
    for sym in VOL_SYMBOLS:
        print(f"\n--- Vol LSTM for {sym} ---")
        t0 = time.time()
        try:
            train_vol_model(
                symbol=sym,
                adapter=adapter,
                fred_key=fred_key,
                save_dir=DEFAULT_MODEL_DIR,
            )
            print(f"  [OK] {sym} vol LSTM — {time.time() - t0:.0f}s")
        except Exception as exc:
            print(f"  [FAIL] {sym} vol LSTM after {time.time() - t0:.0f}s: {exc}")
            failed_lstm.append(sym)
            continue  # skip meta if LSTM failed

        if with_meta:
            print(f"--- Vol meta RF for {sym} ---")
            t0 = time.time()
            try:
                train_vol_meta_model(
                    symbol=sym,
                    adapter=adapter,
                    fred_key=fred_key,
                    save_dir=DEFAULT_MODEL_DIR,
                )
                print(f"  [OK] {sym} vol meta RF — {time.time() - t0:.0f}s")
            except Exception as exc:
                print(f"  [FAIL] {sym} vol meta RF after {time.time() - t0:.0f}s: {exc}")
                failed_meta.append(sym)

    print(f"\n{'='*65}")
    print(f"  Vol models done.  LSTM failed: {failed_lstm or 'none'}  Meta failed: {failed_meta or 'none'}")
    print(f"{'='*65}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step", choices=["lstm", "meta", "vol", "both", "all"], default="both",
        help="both=lstm+meta (default), vol=vol expansion only, all=lstm+meta+vol",
    )
    parser.add_argument("--mode", choices=["daily", "intraday", "both"], default="both")
    args = parser.parse_args()

    fred_key = os.environ.get("FRED_API_KEY", "5e06c25a712146a59c69804dc0cdec4c")
    adapter = build_adapter("yahoo")

    modes = ["daily", "intraday"] if args.mode == "both" else [args.mode]

    t_start = time.time()
    for mode in modes:
        if args.step in ("lstm", "both", "all"):
            train_all_lstm(mode, adapter, fred_key)
        if args.step in ("meta", "both", "all"):
            train_all_meta(mode, adapter, fred_key)

    if args.step in ("vol", "all"):
        train_all_vol(adapter, fred_key, with_meta=True)

    total = time.time() - t_start
    print(f"\nAll done in {total/60:.1f} min.")


if __name__ == "__main__":
    main()
