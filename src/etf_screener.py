#!/usr/bin/env python3
"""
ETF Universe Screener (Layer 0 for Equity Groups)
===================================================
Discovers tradeable ETFs from the broad ETF market by cross-referencing
data sources with yfinance history validation and correlation dedup.

Mirrors src/universe_screener.py in structure and interface — same JSON
schema, same load_universe() contract, same CLI pattern.

Data source fallback chain (never fails):
  Tier 1: yfinance .info loop over ETF_SEED_UNIVERSE
  Tier 2: Cached etf_universe.json (reuse if < 90 days old)

Pipeline:
  1. Fetch AUM, expense ratio, avg volume for seed universe via yfinance
  2. Hard filters: min AUM, min avg daily dollar vol, max expense ratio,
     no leveraged/inverse ETFs
  3. History validation: 504+ days, stability checks (reuse validate_coin_history)
  4. Correlation dedup: remove pairs with >0.85 corr (keep higher volume)
  5. Classify liquidity tier + compute dynamic cost thresholds
  6. Output: etf_universe.json for Layer 1 ETF selector

Usage (via main.py):
    python main.py screen-etf-universe
    python main.py screen-etf-universe --skip-history
    python main.py screen-etf-universe --min-aum 1e9
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from utils import SWING_MODEL_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("etf_screener")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MIN_AUM = 500_000_000           # $500M minimum AUM
DEFAULT_MIN_AVG_DOLLAR_VOL = 50_000_000 # $50M avg daily dollar volume
DEFAULT_MAX_EXPENSE_RATIO = 0.0075      # 0.75% max expense ratio
DEFAULT_MIN_HISTORY_DAYS = 504          # 2 years (252*2)
DEFAULT_MAX_FLASH_CRASH_DAYS = 3
UNIVERSE_FILE = "etf_universe.json"
MAX_CORRELATION = 0.92                  # drops SOXX/SMH, IAU/GLD, FBTC/IBIT but keeps QQQ/IWM, EEM/EWT
CACHE_MAX_AGE_DAYS = 90                 # reuse cache if < 90 days old

# Volatility sanity bounds (annualized, equity)
VOL_MIN_ANNUALIZED = 0.03   # 3% — too stable = no edge
VOL_MAX_ANNUALIZED = 1.50   # 150% — too wild = unmodellable

# Leveraged/inverse name patterns to exclude
LEVERAGED_PATTERNS = {
    "2x", "3x", "ultra", "inverse", "short", "bear", "bull",
    "leveraged", "direxion", "proshares ultra", "proshares short",
}

# ---------------------------------------------------------------------------
# Seed universe — comprehensive ETF pool
# ---------------------------------------------------------------------------
ETF_SEED_UNIVERSE = {
    "us_sector": [
        "XLK", "XLE", "XLF", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE",
        "SMH", "SOXX", "IGV", "BOTZ", "CIBR", "SKYY", "ARKK", "DRIV",
    ],
    "us_factor": [
        "SPY", "QQQ", "IWM", "IWB", "VTV", "QUAL", "MTUM", "USMV", "VBR", "IWF",
    ],
    "commodities": [
        "GLD", "SLV", "GDX", "GDXJ", "CPER", "USO", "DBA", "PDBC", "IAU", "URNM",
    ],
    "intl_developed": [
        "EWJ", "EWG", "VGK", "EWA", "EWU", "EWL", "EWQ", "EWC", "EWS", "EWH",
    ],
    "intl_em": [
        "EEM", "EWT", "EWZ", "MCHI", "VWO", "INDA", "EWY", "EWW",
    ],
    "fixed_income": [
        "TLT", "IEF", "HYG", "LQD", "EMB", "SHY", "BND", "TIP",
    ],
    "crypto_etf": [
        "IBIT", "FBTC", "GBTC", "ETHA",
    ],
}

# Flat list of all seed symbols
ALL_SEED_SYMBOLS = sorted(set(
    sym for group in ETF_SEED_UNIVERSE.values() for sym in group
))

# Map symbol to category
SYMBOL_CATEGORY: Dict[str, str] = {}
for _cat, _syms in ETF_SEED_UNIVERSE.items():
    for _s in _syms:
        SYMBOL_CATEGORY[_s] = _cat

# ---------------------------------------------------------------------------
# Liquidity tiers for ETFs
# ---------------------------------------------------------------------------
ETF_TIER_THRESHOLDS = {
    "mega":   {"min_vol": 500_000_000, "spread_bps": 1,  "slippage_bps": 1,  "fee_bps": 0},
    "large":  {"min_vol": 100_000_000, "spread_bps": 2,  "slippage_bps": 2,  "fee_bps": 0},
    "mid":    {"min_vol": 50_000_000,  "spread_bps": 3,  "slippage_bps": 3,  "fee_bps": 0},
    "small":  {"min_vol": 10_000_000,  "spread_bps": 5,  "slippage_bps": 5,  "fee_bps": 0},
}


def classify_etf_liquidity_tier(avg_dollar_vol: float) -> str:
    for tier, cfg in ETF_TIER_THRESHOLDS.items():
        if avg_dollar_vol >= cfg["min_vol"]:
            return tier
    return "small"


def get_etf_cost_threshold(tier: str) -> float:
    cfg = ETF_TIER_THRESHOLDS.get(tier, ETF_TIER_THRESHOLDS["small"])
    one_way_bps = cfg["spread_bps"] + cfg["slippage_bps"] + cfg["fee_bps"]
    rt_pct = 2 * one_way_bps / 10_000
    return round(rt_pct * 1.5, 5)


# ---------------------------------------------------------------------------
# ScreenedETF dataclass
# ---------------------------------------------------------------------------
@dataclass
class ScreenedETF:
    """An ETF that passed the universe screen."""
    symbol: str               # plain ticker: GLD, SMH (NOT GLD-USD)
    name: str
    category: str             # us_sector, commodities, etc.
    aum: float                # total assets under management (USD)
    expense_ratio: float      # annual expense ratio (e.g. 0.0020)
    avg_dollar_volume_30d: float  # 30-day avg daily dollar volume
    price: float              # current price
    history_days: int = 0
    volume_active_ratio: float = 0.0
    realized_vol_ann: float = 0.0
    liquidity_tier: str = ""
    cost_threshold: float = 0.0
    stability_ok: bool = True
    reject_reason: str = ""


# ---------------------------------------------------------------------------
# Data fetching: yfinance .info for all seed ETFs
# ---------------------------------------------------------------------------
def fetch_etf_info(symbols: List[str]) -> List[dict]:
    """Fetch AUM, expense ratio, and volume for ETFs via yfinance.

    Returns list of dicts with: symbol, name, aum, expense_ratio,
    avg_volume, price, category.
    """
    import yfinance as yf

    log.info("Fetching info for %d ETFs via yfinance...", len(symbols))
    results = []
    for i, sym in enumerate(symbols):
        if (i + 1) % 20 == 0:
            log.info("  Fetched %d/%d...", i + 1, len(symbols))
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info or {}

            aum = info.get("totalAssets") or 0
            expense_raw = info.get("netExpenseRatio") or info.get("annualReportExpenseRatio") or 0.0
            if not isinstance(expense_raw, (int, float)):
                expense_raw = 0.0
            # yfinance returns expense ratio in % form (0.0945 = 0.0945%)
            # Convert to fraction: 0.0945% → 0.000945
            expense = expense_raw / 100.0
            avg_vol = info.get("averageVolume") or 0
            price = info.get("regularMarketPrice") or info.get("previousClose") or 0
            name = info.get("longName") or info.get("shortName") or sym

            results.append({
                "symbol": sym,
                "name": name,
                "aum": float(aum),
                "expense_ratio": float(expense),
                "avg_volume": float(avg_vol),
                "price": float(price),
                "category": SYMBOL_CATEGORY.get(sym, "unknown"),
            })
        except Exception as exc:
            log.warning("  %s: yfinance info failed: %s", sym, exc)
            results.append({
                "symbol": sym,
                "name": sym,
                "aum": 0,
                "expense_ratio": 0,
                "avg_volume": 0,
                "price": 0,
                "category": SYMBOL_CATEGORY.get(sym, "unknown"),
            })
        time.sleep(0.2)  # gentle on yfinance

    log.info("Fetched info for %d ETFs", len(results))
    return results


# ---------------------------------------------------------------------------
# History validation (reuses same logic as universe_screener)
# ---------------------------------------------------------------------------
def validate_etf_history(
    symbol: str,
    min_days: int = DEFAULT_MIN_HISTORY_DAYS,
    max_flash_crashes: int = DEFAULT_MAX_FLASH_CRASH_DAYS,
) -> dict:
    """Validate ETF history length + stability via yfinance.

    Same logic as universe_screener.validate_coin_history but with
    sqrt(252) annualization for equities.
    """
    import yfinance as yf

    result = {
        "history_days": 0,
        "avg_dollar_volume_30d": 0.0,
        "volume_active_ratio": 0.0,
        "realized_vol_ann": 0.0,
        "stability_ok": True,
        "reject_reason": "",
    }

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="3y", interval="1d")
        if hist is None or hist.empty:
            result["stability_ok"] = False
            result["reject_reason"] = "no yfinance data"
            return result

        result["history_days"] = len(hist)

        if len(hist) < min_days:
            result["stability_ok"] = False
            result["reject_reason"] = f"only {len(hist)} days (need {min_days})"
            return result

        close = hist["Close"].astype(float)
        volume = hist["Volume"].astype(float)
        dollar_vol = close * volume

        # 30-day average dollar volume
        result["avg_dollar_volume_30d"] = float(dollar_vol.tail(30).mean()) if len(dollar_vol) >= 30 else 0.0

        # Flash crash check
        daily_ret = close.pct_change()
        vol_20d_avg = volume.rolling(20).mean()
        flash_crashes = (
            (daily_ret < -0.30) &
            (volume < vol_20d_avg * 0.5)
        ).sum()

        if flash_crashes > max_flash_crashes:
            result["stability_ok"] = False
            result["reject_reason"] = f"{flash_crashes} flash crashes (max {max_flash_crashes})"
            return result

        # Volume stability: 85% of recent 90 days with >$1M dollar vol (higher bar for ETFs)
        lookback = min(90, len(dollar_vol))
        if lookback >= 30:
            recent_dv = dollar_vol.tail(lookback)
            active_days = int((recent_dv >= 1_000_000).sum())
            active_ratio = active_days / lookback
            result["volume_active_ratio"] = round(active_ratio, 3)

            if active_ratio < 0.80:
                result["stability_ok"] = False
                result["reject_reason"] = (
                    f"volume stability {active_ratio:.0%} < 80% "
                    f"({active_days}/{lookback} days with >$1M vol)"
                )
                return result

        # Volatility sanity: annualized realized vol (60d) — sqrt(252) for equities
        vol_window = min(60, len(daily_ret) - 1)
        if vol_window >= 20:
            recent_ret = daily_ret.tail(vol_window).dropna()
            daily_std = float(recent_ret.std())
            ann_vol = daily_std * np.sqrt(252)  # equity: 252 trading days
            result["realized_vol_ann"] = round(ann_vol, 4)

            if ann_vol < VOL_MIN_ANNUALIZED:
                result["stability_ok"] = False
                result["reject_reason"] = f"vol too low: {ann_vol:.1%} ann (min {VOL_MIN_ANNUALIZED:.0%})"
                return result
            if ann_vol > VOL_MAX_ANNUALIZED:
                result["stability_ok"] = False
                result["reject_reason"] = f"vol too high: {ann_vol:.1%} ann (max {VOL_MAX_ANNUALIZED:.0%})"
                return result

    except Exception as exc:
        result["stability_ok"] = False
        result["reject_reason"] = f"yfinance error: {exc}"

    return result


# ---------------------------------------------------------------------------
# Correlation dedup
# ---------------------------------------------------------------------------
def deduplicate_by_correlation(
    etfs: List[ScreenedETF],
    max_corr: float = MAX_CORRELATION,
) -> List[ScreenedETF]:
    """Remove near-duplicate ETFs based on 252-day return correlation.

    For each pair with corr > max_corr, drop the lower avg_dollar_volume symbol.
    """
    import yfinance as yf

    if len(etfs) < 2:
        return etfs

    symbols = [e.symbol for e in etfs]
    log.info("Correlation dedup: computing pairwise corr for %d ETFs...", len(symbols))

    # Download 252 days of close prices
    try:
        data = yf.download(symbols, period="1y", progress=False, auto_adjust=True)
        if data.empty:
            log.warning("Correlation dedup: no data, skipping")
            return etfs
        close = data["Close"] if "Close" in data.columns else data
        if hasattr(close, "columns"):
            # Multi-symbol case
            returns = close.pct_change().dropna()
        else:
            log.warning("Correlation dedup: single column, skipping")
            return etfs
    except Exception as exc:
        log.warning("Correlation dedup failed: %s — skipping", exc)
        return etfs

    # Build correlation matrix
    corr_matrix = returns.corr()

    # Build volume lookup for tie-breaking
    vol_lookup = {e.symbol: e.avg_dollar_volume_30d for e in etfs}

    # Find pairs to drop
    to_drop = set()
    dropped_pairs = []
    for i, sym_a in enumerate(symbols):
        if sym_a in to_drop:
            continue
        for sym_b in symbols[i + 1:]:
            if sym_b in to_drop:
                continue
            try:
                corr_val = corr_matrix.loc[sym_a, sym_b]
            except KeyError:
                continue
            if np.isnan(corr_val):
                continue
            if abs(corr_val) > max_corr:
                # Drop lower volume
                if vol_lookup.get(sym_a, 0) >= vol_lookup.get(sym_b, 0):
                    to_drop.add(sym_b)
                    dropped_pairs.append((sym_a, sym_b, corr_val))
                else:
                    to_drop.add(sym_a)
                    dropped_pairs.append((sym_b, sym_a, corr_val))

    for kept, dropped, corr in dropped_pairs:
        log.info("  Dedup: dropped %s (corr %.3f with %s, lower volume)", dropped, corr, kept)

    result = [e for e in etfs if e.symbol not in to_drop]
    log.info("Correlation dedup: %d → %d ETFs (%d dropped)", len(etfs), len(result), len(to_drop))
    return result


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _cache_age_days(save_dir: str) -> float:
    """Return age of etf_universe.json in days, or inf if missing."""
    path = os.path.join(save_dir, UNIVERSE_FILE)
    if not os.path.exists(path):
        return float("inf")
    try:
        with open(path) as f:
            data = json.load(f)
        screened_at = data.get("screened_at", "")
        if not screened_at:
            return float("inf")
        dt = datetime.fromisoformat(screened_at.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).total_seconds() / 86400
    except Exception:
        return float("inf")


def _is_leveraged(name: str) -> bool:
    """Check if ETF name indicates leveraged/inverse product."""
    name_lower = name.lower()
    return any(pat in name_lower for pat in LEVERAGED_PATTERNS)


# ---------------------------------------------------------------------------
# Main screening pipeline
# ---------------------------------------------------------------------------
def screen_etf_universe(
    min_aum: float = DEFAULT_MIN_AUM,
    min_avg_dollar_vol: float = DEFAULT_MIN_AVG_DOLLAR_VOL,
    max_expense: float = DEFAULT_MAX_EXPENSE_RATIO,
    save_dir: Optional[str] = None,
    skip_history: bool = False,
) -> List[ScreenedETF]:
    """Run the full ETF universe screen.

    Steps:
      1. Fetch yfinance info for seed universe
      2. Hard filters: AUM, volume, expense ratio, no leveraged/inverse
      3. History validation: min 504 days, stability checks
      4. Correlation dedup: remove pairs with >0.85 corr
      5. Classify liquidity tier + compute cost thresholds
      6. Save to etf_universe.json
    """
    save_dir = save_dir or SWING_MODEL_DIR

    # Tier 1: yfinance .info for all seed ETFs
    raw_data = None
    try:
        raw_data = fetch_etf_info(ALL_SEED_SYMBOLS)
    except Exception as exc:
        log.warning("[Tier 1] yfinance info fetch failed: %s", exc)

    # Tier 2: cached universe
    if not raw_data:
        cache_age = _cache_age_days(save_dir)
        if cache_age < CACHE_MAX_AGE_DAYS:
            log.warning("[Tier 2] Using cached ETF universe (%.0f days old)", cache_age)
            cached = load_etf_universe_detail(save_dir)
            return [ScreenedETF(**c) for c in cached] if cached else []
        log.error("All data sources failed (cache %.0f days old) — returning empty", cache_age)
        return []

    # Step 2: Hard filters
    candidates = []
    rejected = {"aum": 0, "volume": 0, "expense": 0, "leveraged": 0}
    for item in raw_data:
        sym = item["symbol"]
        name = item["name"]
        aum = item["aum"]
        expense = item["expense_ratio"]
        avg_vol = item["avg_volume"]
        price = item["price"]
        category = item["category"]

        # No leveraged/inverse
        if _is_leveraged(name):
            rejected["leveraged"] += 1
            log.debug("Rejected %s: leveraged/inverse (%s)", sym, name)
            continue

        # Min AUM
        if aum > 0 and aum < min_aum:
            rejected["aum"] += 1
            continue

        # Avg daily dollar volume (estimate from avg_volume * price)
        est_dollar_vol = avg_vol * price if price > 0 else 0
        if est_dollar_vol > 0 and est_dollar_vol < min_avg_dollar_vol:
            rejected["volume"] += 1
            continue

        # Max expense ratio (only filter if we got a real value)
        if expense > 0 and expense > max_expense:
            rejected["expense"] += 1
            continue

        tier = classify_etf_liquidity_tier(est_dollar_vol)
        cost_thresh = get_etf_cost_threshold(tier)

        candidates.append(ScreenedETF(
            symbol=sym,
            name=name,
            category=category,
            aum=aum,
            expense_ratio=expense,
            avg_dollar_volume_30d=est_dollar_vol,
            price=price,
            liquidity_tier=tier,
            cost_threshold=cost_thresh,
        ))

    log.info("Hard filters: %d candidates passed (rejected: %s)", len(candidates), rejected)

    # Step 3: History validation
    if not skip_history:
        log.info("Validating price history for %d candidates...", len(candidates))
        validated = []
        for i, etf in enumerate(candidates):
            if (i + 1) % 10 == 0:
                log.info("  History check: %d/%d...", i + 1, len(candidates))
            hv = validate_etf_history(etf.symbol)
            etf.history_days = hv["history_days"]
            etf.volume_active_ratio = hv.get("volume_active_ratio", 0.0)
            etf.realized_vol_ann = hv.get("realized_vol_ann", 0.0)
            etf.stability_ok = hv["stability_ok"]
            etf.reject_reason = hv["reject_reason"]

            if etf.stability_ok:
                # Update dollar volume from history (more reliable)
                if hv["avg_dollar_volume_30d"] > 0:
                    etf.avg_dollar_volume_30d = hv["avg_dollar_volume_30d"]
                    etf.liquidity_tier = classify_etf_liquidity_tier(etf.avg_dollar_volume_30d)
                    etf.cost_threshold = get_etf_cost_threshold(etf.liquidity_tier)
                validated.append(etf)
            else:
                log.info("  Rejected %s: %s", etf.symbol, etf.reject_reason)

            time.sleep(0.3)

        log.info("History validation: %d passed, %d rejected",
                 len(validated), len(candidates) - len(validated))
        candidates = validated

    # Step 4: Correlation dedup
    if not skip_history and len(candidates) >= 2:
        candidates = deduplicate_by_correlation(candidates, MAX_CORRELATION)

    # Sort by avg dollar volume descending
    candidates.sort(key=lambda c: c.avg_dollar_volume_30d, reverse=True)

    log.info("Final ETF universe: %d ETFs", len(candidates))

    # Step 5: Save
    _save_etf_universe(candidates, save_dir)

    return candidates


def _save_etf_universe(
    etfs: List[ScreenedETF], save_dir: str,
) -> None:
    """Save screened ETF universe to JSON."""
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, UNIVERSE_FILE)

    tier_counts = {}
    cat_counts = {}
    for e in etfs:
        tier_counts[e.liquidity_tier] = tier_counts.get(e.liquidity_tier, 0) + 1
        cat_counts[e.category] = cat_counts.get(e.category, 0) + 1

    payload = {
        "screened_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "yfinance",
        "count": len(etfs),
        "filters": {
            "min_aum": DEFAULT_MIN_AUM,
            "min_avg_dollar_vol": DEFAULT_MIN_AVG_DOLLAR_VOL,
            "max_expense_ratio": DEFAULT_MAX_EXPENSE_RATIO,
            "min_history_days": DEFAULT_MIN_HISTORY_DAYS,
            "max_correlation": MAX_CORRELATION,
        },
        "tier_counts": tier_counts,
        "category_counts": cat_counts,
        "coins": [asdict(e) for e in etfs],  # "coins" key for compat with load_universe()
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Saved ETF universe (%d ETFs) to %s", len(etfs), out_path)
    for cat, count in sorted(cat_counts.items()):
        log.info("  %s: %d ETFs", cat, count)


# ---------------------------------------------------------------------------
# Momentum / relative-strength scoring
# ---------------------------------------------------------------------------

def compute_etf_momentum_scores(symbols: List[str]) -> Dict[str, float]:
    """Compute cross-sectional momentum scores for a list of ETFs.

    For each ETF, computes a momentum_score in [0, 1] based on:
      - 6-month absolute return (ret_126)
      - 3-month absolute return (ret_63)
      - 3-month excess return vs SPY (ret_63_vs_spy)
      - 1-month return with negative weight (mild mean-reversion)

    All returns are rank-percentiled across the universe, then combined:
      momentum_score = 0.40 * rank(ret_126)
                     + 0.40 * rank(ret_63)
                     + 0.20 * rank(ret_63_vs_spy)
                     - 0.10 * rank(ret_21)

    The result is clipped to [0, 1] so it combines cleanly with base_score.

    Returns:
        {symbol: momentum_score} for each symbol with sufficient data.
        Symbols with insufficient history get score 0.5 (neutral).
    """
    import yfinance as yf
    import pandas as pd

    # Need SPY as benchmark — ensure it's in the download
    all_syms = list(set(symbols) | {"SPY"})
    log.info("Computing momentum scores for %d ETFs...", len(symbols))

    try:
        data = yf.download(all_syms, period="9mo", progress=False, auto_adjust=True)
        if data.empty:
            log.warning("Momentum: no data returned from yfinance")
            return {s: 0.5 for s in symbols}
        close = data["Close"] if "Close" in data.columns else data
    except Exception as exc:
        log.warning("Momentum: yfinance download failed: %s", exc)
        return {s: 0.5 for s in symbols}

    # Compute returns for each lookback
    results: Dict[str, Dict[str, float]] = {}
    spy_close = close["SPY"] if "SPY" in close.columns else None

    for sym in symbols:
        if sym not in close.columns:
            results[sym] = {"ret_126": np.nan, "ret_63": np.nan,
                            "ret_21": np.nan, "ret_63_vs_spy": np.nan}
            continue

        c = close[sym].dropna()
        if len(c) < 130:  # need ~6 months
            results[sym] = {"ret_126": np.nan, "ret_63": np.nan,
                            "ret_21": np.nan, "ret_63_vs_spy": np.nan}
            continue

        ret_126 = float((c.iloc[-1] / c.iloc[-126]) - 1) if len(c) >= 126 else np.nan
        ret_63 = float((c.iloc[-1] / c.iloc[-63]) - 1) if len(c) >= 63 else np.nan
        ret_21 = float((c.iloc[-1] / c.iloc[-21]) - 1) if len(c) >= 21 else np.nan

        # Excess return vs SPY (3-month)
        ret_63_vs_spy = np.nan
        if spy_close is not None and len(c) >= 63:
            spy_c = spy_close.dropna()
            if len(spy_c) >= 63:
                spy_ret_63 = float((spy_c.iloc[-1] / spy_c.iloc[-63]) - 1)
                ret_63_vs_spy = ret_63 - spy_ret_63

        results[sym] = {
            "ret_126": ret_126,
            "ret_63": ret_63,
            "ret_21": ret_21,
            "ret_63_vs_spy": ret_63_vs_spy,
        }

    # Cross-sectional rank percentiles (0 = worst, 1 = best)
    df = pd.DataFrame(results).T
    ranked = pd.DataFrame(index=df.index)
    for col in ["ret_126", "ret_63", "ret_21", "ret_63_vs_spy"]:
        valid = df[col].dropna()
        if len(valid) >= 2:
            ranked[col] = valid.rank(pct=True)
        else:
            ranked[col] = 0.5

    # Fill NaN ranks with 0.5 (neutral)
    ranked = ranked.fillna(0.5)

    # Composite momentum score
    momentum = (
        0.40 * ranked["ret_126"]
        + 0.40 * ranked["ret_63"]
        + 0.20 * ranked["ret_63_vs_spy"]
        - 0.10 * ranked["ret_21"]   # mild 1-month mean-reversion
    )

    # Clip to [0, 1]
    momentum = momentum.clip(0.0, 1.0)

    scores = {sym: round(float(momentum.loc[sym]), 4) if sym in momentum.index else 0.5
              for sym in symbols}

    # Log top/bottom 5
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    top5 = ", ".join(f"{s}={v:.3f}" for s, v in sorted_scores[:5])
    bot5 = ", ".join(f"{s}={v:.3f}" for s, v in sorted_scores[-5:])
    log.info("Momentum scores: top5=[%s] bot5=[%s]", top5, bot5)

    return scores


# ---------------------------------------------------------------------------
# Intraday activity scoring
# ---------------------------------------------------------------------------

def compute_etf_intraday_activity_scores(symbols: List[str]) -> Dict[str, float]:
    """Compute intraday-specific activity scores for ETF ranking.

    For each ETF, computes an intraday_activity_score in [0, 1] based on:
      - ATR% (14-day Average True Range as % of price) — captures daily range
      - Opening-range volatility proxy (high-low range of recent days)
      - Overnight gap frequency (abs gap > 0.5% in last 60 days)

    All metrics are rank-percentiled across the universe, then combined:
      intraday_activity = 0.50 * rank(atr_pct)
                        + 0.30 * rank(avg_range_pct)
                        + 0.20 * rank(gap_frequency)

    Returns:
        {symbol: intraday_activity_score} for each symbol.
        Symbols with insufficient data get 0.5 (neutral).
    """
    import yfinance as yf
    import pandas as pd

    log.info("Computing intraday activity scores for %d ETFs...", len(symbols))

    try:
        data = yf.download(symbols, period="4mo", progress=False, auto_adjust=True)
        if data.empty:
            log.warning("Intraday activity: no data returned")
            return {s: 0.5 for s in symbols}
    except Exception as exc:
        log.warning("Intraday activity: yfinance failed: %s", exc)
        return {s: 0.5 for s in symbols}

    results: Dict[str, Dict[str, float]] = {}

    for sym in symbols:
        try:
            if len(symbols) == 1:
                high = data["High"].astype(float)
                low = data["Low"].astype(float)
                close = data["Close"].astype(float)
                opn = data["Open"].astype(float)
            else:
                high = data["High"][sym].astype(float)
                low = data["Low"][sym].astype(float)
                close = data["Close"][sym].astype(float)
                opn = data["Open"][sym].astype(float)
        except (KeyError, TypeError):
            results[sym] = {"atr_pct": np.nan, "avg_range_pct": np.nan,
                            "gap_freq": np.nan}
            continue

        high = high.dropna()
        low = low.dropna()
        close = close.dropna()
        opn = opn.dropna()

        if len(close) < 30:
            results[sym] = {"atr_pct": np.nan, "avg_range_pct": np.nan,
                            "gap_freq": np.nan}
            continue

        # ATR% (14-day)
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean().iloc[-1]
        atr_pct = float(atr_14 / close.iloc[-1]) if close.iloc[-1] > 0 else 0.0

        # Average daily range % (last 30 days)
        range_pct = ((high - low) / close).tail(30)
        avg_range_pct = float(range_pct.mean()) if len(range_pct) > 0 else 0.0

        # Overnight gap frequency (abs(open/prev_close - 1) > 0.5%)
        lookback = min(60, len(close) - 1)
        if lookback > 0:
            # Align open and previous close
            gaps = (opn.iloc[-lookback:].values / close.iloc[-lookback-1:-1].values) - 1
            gap_freq = float(np.sum(np.abs(gaps) > 0.005) / lookback)
        else:
            gap_freq = 0.0

        results[sym] = {
            "atr_pct": atr_pct,
            "avg_range_pct": avg_range_pct,
            "gap_freq": gap_freq,
        }

    # Cross-sectional rank percentiles
    df = pd.DataFrame(results).T
    ranked = pd.DataFrame(index=df.index)
    for col in ["atr_pct", "avg_range_pct", "gap_freq"]:
        valid = df[col].dropna()
        if len(valid) >= 2:
            ranked[col] = valid.rank(pct=True)
        else:
            ranked[col] = 0.5
    ranked = ranked.fillna(0.5)

    # Composite
    activity = (
        0.50 * ranked["atr_pct"]
        + 0.30 * ranked["avg_range_pct"]
        + 0.20 * ranked["gap_freq"]
    ).clip(0.0, 1.0)

    scores = {sym: round(float(activity.loc[sym]), 4) if sym in activity.index else 0.5
              for sym in symbols}

    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    top5 = ", ".join(f"{s}={v:.3f}" for s, v in sorted_scores[:5])
    bot5 = ", ".join(f"{s}={v:.3f}" for s, v in sorted_scores[-5:])
    log.info("Intraday activity scores: top5=[%s] bot5=[%s]", top5, bot5)

    return scores


# ---------------------------------------------------------------------------
# Load functions (mirror universe_screener.py interface)
# ---------------------------------------------------------------------------
def load_etf_universe(save_dir: Optional[str] = None) -> List[str]:
    """Load the screened ETF universe from disk.

    Returns list of plain ticker symbols (e.g. ['GLD', 'SMH', 'QQQ']).
    Falls back to empty list if no universe file exists.
    """
    save_dir = save_dir or SWING_MODEL_DIR
    path = os.path.join(save_dir, UNIVERSE_FILE)
    if not os.path.exists(path):
        log.warning("No ETF universe file at %s", path)
        return []

    try:
        with open(path) as f:
            data = json.load(f)
        symbols = [c["symbol"] for c in data.get("coins", [])]
        screened_at = data.get("screened_at", "unknown")
        log.info("Loaded ETF universe: %d symbols (screened %s)", len(symbols), screened_at)
        return symbols
    except (json.JSONDecodeError, KeyError) as exc:
        log.warning("Corrupt ETF universe file: %s", exc)
        return []


def load_etf_universe_detail(save_dir: Optional[str] = None) -> List[dict]:
    """Load full ETF universe detail (all fields)."""
    save_dir = save_dir or SWING_MODEL_DIR
    path = os.path.join(save_dir, UNIVERSE_FILE)
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("coins", [])
    except (json.JSONDecodeError, KeyError):
        return []


def load_promoted_symbols(save_dir: Optional[str] = None) -> List[str]:
    """Load OOS-validated promoted symbols from promoted_symbols.json.

    Returns list of symbols or empty list if file doesn't exist.
    """
    save_dir = save_dir or SWING_MODEL_DIR
    path = os.path.join(save_dir, "promoted_symbols.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("symbols", [])
    except (json.JSONDecodeError, KeyError):
        return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def print_etf_table(etfs: List[ScreenedETF]) -> None:
    """Pretty-print the screened ETF universe."""
    print(f"\n{'='*110}")
    print(f"  ETF Universe Screen -- {len(etfs)} ETFs qualified")
    print(f"{'='*110}")
    print(f"  {'#':<4} {'Symbol':<8} {'Name':<28} {'Cat':<16} {'Tier':<6} {'AUM':>12} "
          f"{'AvgDolVol':>12} {'Days':>5} {'AnnVol':>7} {'Exp%':>6} {'Cost%':>6}")
    print(f"  {'-'*4} {'-'*8} {'-'*28} {'-'*16} {'-'*6} {'-'*12} "
          f"{'-'*12} {'-'*5} {'-'*7} {'-'*6} {'-'*6}")
    for i, e in enumerate(etfs, 1):
        aum_str = f"${e.aum/1e9:.1f}B" if e.aum >= 1e9 else f"${e.aum/1e6:.0f}M"
        vol_str = f"${e.avg_dollar_volume_30d/1e6:.0f}M" if e.avg_dollar_volume_30d >= 1e6 else f"${e.avg_dollar_volume_30d/1e3:.0f}K"
        days_str = str(e.history_days) if e.history_days > 0 else "?"
        av = f"{e.realized_vol_ann:.0%}" if e.realized_vol_ann > 0 else "?"
        exp_str = f"{e.expense_ratio*100:.2f}" if e.expense_ratio > 0 else "?"
        cost_str = f"{e.cost_threshold*100:.3f}" if e.cost_threshold > 0 else "?"
        safe_name = e.name.encode("ascii", errors="replace").decode("ascii")[:28]
        print(f"  {i:<4} {e.symbol:<8} {safe_name:<28} {e.category:<16} {e.liquidity_tier:<6} "
              f"{aum_str:>12} {vol_str:>12} {days_str:>5} {av:>7} {exp_str:>6} {cost_str:>6}")
    print(f"{'='*110}")

    # Category summary
    cat_counts = {}
    for e in etfs:
        cat_counts[e.category] = cat_counts.get(e.category, 0) + 1
    print(f"\n  Categories: ", end="")
    for cat, count in sorted(cat_counts.items()):
        print(f"{cat}={count}  ", end="")
    print()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="ETF Universe Screener (Layer 0)")
    parser.add_argument("--min-aum", type=float, default=DEFAULT_MIN_AUM,
                        help=f"Minimum AUM in USD (default: {DEFAULT_MIN_AUM:.0f})")
    parser.add_argument("--min-volume", type=float, default=DEFAULT_MIN_AVG_DOLLAR_VOL,
                        help=f"Minimum avg daily dollar volume (default: {DEFAULT_MIN_AVG_DOLLAR_VOL:.0f})")
    parser.add_argument("--save-dir", type=str, default=SWING_MODEL_DIR,
                        help="Directory to save etf_universe.json")
    parser.add_argument("--skip-history", action="store_true",
                        help="Skip yfinance history validation (faster)")
    args = parser.parse_args()

    etfs = screen_etf_universe(
        min_aum=args.min_aum,
        min_avg_dollar_vol=args.min_volume,
        save_dir=args.save_dir,
        skip_history=args.skip_history,
    )
    print_etf_table(etfs)


if __name__ == "__main__":
    main()
