"""
Shared utilities for the quantitative trading system.

Extracted from ml_model.py so that intraday_model and swing_model can import
lightweight helpers without pulling in PyTorch / the full LSTM training pipeline.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

from signals_engine import FREDVixFetcher, PROJECT_ROOT

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CRYPTO_MODEL_DIR  = os.path.join(PROJECT_ROOT, "models", "crypto")

COST_THRESHOLD = 0.001   # 0.1 % minimum expected return to trade
TARGET_RETURN  = 0.02    # 2 % return = full position size


# ---------------------------------------------------------------------------
# VIX history helper (shared across swing, intraday, range)
# ---------------------------------------------------------------------------

def _fetch_vix_for_training(fred_key: Optional[str], lookback_days: int) -> pd.DataFrame:
    """Fetch VIX history. Try FRED first, fall back to yfinance ^VIX.

    After getting daily history, always try to append today's live intraday
    VIX price so vix_change_pct() reflects the real intraday spike, not just
    yesterday's close vs the day before.
    """
    fetcher = FREDVixFetcher(api_key=fred_key)
    vix_df = fetcher.fetch(lookback_days=lookback_days)
    if len(vix_df) < 20:
        # FRED free tier only returns ~10 recent observations; fall back to yfinance
        log.info("FRED VIX data sparse (%d rows); falling back to yfinance ^VIX.", len(vix_df))
        try:
            import yfinance as yf
            ticker = yf.Ticker("^VIX")
            cal_days = int(lookback_days * 1.5) + 10
            hist = ticker.history(period=f"{cal_days}d", interval="1d")
            if hist is not None and not hist.empty:
                vdf = pd.DataFrame({
                    "date": hist.index,
                    "vix": hist["Close"].values,
                })
                if vdf["date"].dt.tz is not None:
                    vdf["date"] = vdf["date"].dt.tz_localize(None)
                vdf = vdf.sort_values("date").reset_index(drop=True)
                log.info("Fetched %d VIX rows from yfinance.", len(vdf))
                vix_df = vdf
        except Exception as exc:
            log.warning("yfinance ^VIX fallback failed: %s", exc)

    # Always append today's live VIX price so intraday spikes are captured.
    # FRED only updates after market close, so during market hours the last
    # row is yesterday's close — this override fixes that.
    try:
        import yfinance as yf
        _vix_ticker = yf.Ticker("^VIX")
        live_vix = float(_vix_ticker.fast_info.last_price)
        if live_vix > 0:
            today = pd.Timestamp.now().normalize()
            last_date = vix_df["date"].iloc[-1].normalize() if not vix_df.empty else pd.Timestamp("1900-01-01")
            if last_date < today:
                today_row = pd.DataFrame({"date": [today], "vix": [live_vix]})
                vix_df = pd.concat([vix_df, today_row], ignore_index=True)
                log.info("Appended live intraday VIX=%.1f for today.", live_vix)
            else:
                vix_df.loc[vix_df.index[-1], "vix"] = live_vix
                log.info("Updated today's VIX row to live value=%.1f.", live_vix)
    except Exception as exc:
        log.debug("Live VIX fetch failed (using daily close): %s", exc)

    return vix_df
