#!/usr/bin/env python3
"""
Crypto Universe Screener (Layer 0)
===================================
Discovers tradeable crypto coins from the full market by cross-referencing
market data APIs with Kraken available trading pairs, then validates
each candidate with yfinance price history and stability checks.

Data source fallback chain (never fails):
  1. CoinMarketCap  (primary — best free tier, 1 call = top 200 coins)
  2. CoinGecko      (secondary — no API key needed)
  3. Kraken Ticker   (tertiary — volume/price only, no market cap)
  4. Cached universe.json (last resort — reuse if < 30 days old)

Pipeline:
  1. Fetch all USD trading pairs from Kraken (public API, no auth)
  2. Fetch top coins from data source (CMC → CoinGecko → Kraken → cache)
  3. Cross-reference: keep only coins tradeable on Kraken
  4. Hard filters: min market cap, min 24h volume, vol/mcap ratio
  5. History validation: fetch yfinance data, require 365+ days, stability check
  6. Compute liquidity tier + dynamic cost threshold per coin
  7. Output: universe.json for Layer 1 coin selector

Usage (via main.py):
    python main.py screen-universe                    # default filters
    python main.py screen-universe --min-mcap 50e6    # $50M min market cap
    python main.py screen-universe --top-n 300        # scan top 300 by mcap
    python main.py screen-universe --skip-history     # fast mode (no yfinance)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import requests

from utils import CRYPTO_MODEL_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("universe_screener")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MIN_MCAP = 100_000_000       # $100M minimum market cap
DEFAULT_MIN_VOLUME_24H = 5_000_000   # $5M minimum 24h trading volume
DEFAULT_MIN_VOL_MCAP_RATIO = 0.005   # 0.5% daily volume / market cap (fallback)
DEFAULT_TOP_N = 250                  # scan top-N coins from CoinGecko
DEFAULT_MIN_HISTORY_DAYS = 365       # need 1 year+ of daily history for swing
DEFAULT_MAX_FLASH_CRASH_DAYS = 3     # max days with >30% drop on low volume
UNIVERSE_FILE = "universe.json"

# Turnover tier thresholds by market cap band — larger coins need lower turnover
TURNOVER_TIERS = [
    # (min_mcap, required_turnover_ratio)
    (3_000_000_000, 0.002),   # >$3B: ≥0.2% turnover
    (500_000_000,   0.005),   # $500M–$3B: ≥0.5% turnover
    (100_000_000,   0.010),   # $100M–$500M: ≥1.0% turnover
]

# Volume stability: fraction of recent 90 days with >$100K daily dollar volume
VOLUME_STABILITY_MIN_RATIO = 0.85   # 85% of days must have meaningful volume
VOLUME_STABILITY_LOOKBACK = 90      # days
VOLUME_STABILITY_FLOOR = 100_000    # $100K minimum to count as "active"

# Volatility sanity bounds (annualized)
VOL_MIN_ANNUALIZED = 0.05   # 5% — too stable = no edge
VOL_MAX_ANNUALIZED = 3.00   # 300% — too wild = unmodellable

# ---------------------------------------------------------------------------
# Blacklist: loaded from config/trading.json → crypto_blacklist
# ---------------------------------------------------------------------------
def _load_blacklist() -> set:
    """Load blacklisted symbols from config/trading.json.

    Returns a set of base symbols (e.g. 'DOGE', 'XMR') to exclude.
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "trading.json")
    if not os.path.exists(config_path):
        return set()
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        blacklist_cfg = cfg.get("crypto_blacklist", {})
        symbols = set()
        for category, info in blacklist_cfg.items():
            if category.startswith("_"):
                continue
            if isinstance(info, dict):
                for sym in info.get("symbols", []):
                    symbols.add(sym.upper())
        return symbols
    except Exception:
        return set()


BLACKLISTED_SYMBOLS = _load_blacklist()

# Coins to always exclude (stablecoins, wrapped, etc.)
EXCLUDE_SYMBOLS = {
    # Stablecoins
    "USDT", "USDC", "DAI", "BUSD", "TUSD", "USDP", "FDUSD", "PYUSD",
    "USDS", "USDE", "USD1", "USDG", "RLUSD", "USDD", "GUSD", "AUSD",
    "EURC", "FRAX", "LUSD", "CRVUSD", "GHO", "SUSD",
    # Gold/fiat-backed tokens
    "XAUT", "PAXG",
    # Wrapped/staked
    "WBTC", "WETH", "STETH", "CBETH", "RETH", "WSTETH", "METH",
    # Exchange tokens
    "LEO", "OKB", "BGB", "GT", "KCS", "HT", "WBT", "CRO", "FTT",
}
# Also exclude by name patterns
EXCLUDE_NAME_PATTERNS = {
    "usd", "stable", "tether", "pax gold", "tether gold",
    "wrapped", "bridged", "staked",
}

# ---------------------------------------------------------------------------
# Liquidity tiers — determines cost model and max position size
# ---------------------------------------------------------------------------
# Tier thresholds based on 24h volume
TIER_THRESHOLDS = {
    "mega":   {"min_vol": 500_000_000, "spread_bps": 3,  "slippage_bps": 2,  "fee_bps": 10},
    "large":  {"min_vol": 50_000_000,  "spread_bps": 5,  "slippage_bps": 5,  "fee_bps": 10},
    "mid":    {"min_vol": 10_000_000,  "spread_bps": 10, "slippage_bps": 10, "fee_bps": 10},
    "small":  {"min_vol": 5_000_000,   "spread_bps": 15, "slippage_bps": 15, "fee_bps": 10},
}


def classify_liquidity_tier(volume_24h: float) -> str:
    """Classify a coin into a liquidity tier based on 24h volume."""
    for tier, cfg in TIER_THRESHOLDS.items():
        if volume_24h >= cfg["min_vol"]:
            return tier
    return "small"


def get_dynamic_cost_threshold(tier: str) -> float:
    """Get minimum expected return (cost threshold) for a liquidity tier.

    cost_threshold = round_trip_cost * 1.5x safety margin
    round_trip_cost = 2 * (half_spread + slippage + fee)
    """
    cfg = TIER_THRESHOLDS.get(tier, TIER_THRESHOLDS["small"])
    one_way_bps = cfg["spread_bps"] + cfg["slippage_bps"] + cfg["fee_bps"]
    rt_pct = 2 * one_way_bps / 10_000
    return round(rt_pct * 1.5, 4)


def get_max_order_usd(volume_24h: float) -> float:
    """Max single order size = daily volume / 10 (rule of thumb)."""
    return volume_24h / 10


def check_turnover(mcap: float, volume_24h: float) -> bool:
    """Check if volume/mcap ratio meets tiered turnover requirement.

    Larger coins get a lower bar (BTC only needs 0.2% turnover),
    smaller coins need higher turnover to prove liquidity.
    """
    if mcap <= 0:
        return False
    ratio = volume_24h / mcap
    for min_mcap, required_ratio in TURNOVER_TIERS:
        if mcap >= min_mcap:
            return ratio >= required_ratio
    # Below all tiers — shouldn't happen with $100M min mcap, but use strictest
    return ratio >= TURNOVER_TIERS[-1][1]


def get_required_turnover(mcap: float) -> float:
    """Return the required turnover ratio for a given market cap."""
    for min_mcap, required_ratio in TURNOVER_TIERS:
        if mcap >= min_mcap:
            return required_ratio
    return TURNOVER_TIERS[-1][1]


@dataclass
class ScreenedCoin:
    """A coin that passed the universe screen."""
    symbol: str              # yfinance format: BTC-USD
    kraken_pair: str         # Kraken pair name: XXBTZUSD
    coingecko_id: str        # CoinGecko ID: bitcoin
    name: str                # Full name: Bitcoin
    market_cap: float        # USD market cap
    volume_24h: float        # USD 24h volume (CoinGecko snapshot)
    price: float             # Current USD price
    mcap_rank: int           # CoinGecko market cap rank
    vol_mcap_ratio: float = 0.0    # daily volume / market cap
    history_days: int = 0          # days of yfinance price history
    avg_dollar_volume_30d: float = 0.0  # 30-day average daily dollar volume
    liquidity_tier: str = ""       # mega/large/mid/small
    cost_threshold: float = 0.0    # dynamic cost threshold (pct)
    max_order_usd: float = 0.0     # max single order in USD
    volume_active_ratio: float = 0.0   # fraction of recent days with >$100K volume
    realized_vol_ann: float = 0.0      # annualized realized volatility (60d)
    stability_ok: bool = True      # passed stability checks
    reject_reason: str = ""        # why rejected (if any)


# ---------------------------------------------------------------------------
# Kraken: get all tradeable USD pairs
# ---------------------------------------------------------------------------
def fetch_kraken_usd_pairs() -> Dict[str, str]:
    """Fetch all USD trading pairs from Kraken.

    Returns dict of {normalized_base_symbol: kraken_pair_name}.
    """
    log.info("Fetching Kraken tradeable pairs...")
    resp = requests.get(
        "https://api.kraken.com/0/public/AssetPairs",
        timeout=15,
    )
    resp.raise_for_status()
    result = resp.json().get("result", {})

    KRAKEN_TO_STANDARD = {
        "XBT": "BTC", "XXBT": "BTC",
        "XDG": "DOGE", "XDOGE": "DOGE",
        "XETH": "ETH", "XLTC": "LTC", "XXRP": "XRP",
        "XMLN": "MLN", "XREP": "REP", "XXMR": "XMR",
        "XZEC": "ZEC",
    }

    pairs = {}
    for pair_name, info in result.items():
        quote = info.get("quote", "")
        if quote not in ("ZUSD", "USD"):
            continue
        if pair_name.endswith(".d"):
            continue

        base = info.get("base", "")
        normalized = KRAKEN_TO_STANDARD.get(base, base)
        if len(normalized) >= 4 and normalized[0] in ("X", "Z"):
            alt = normalized[1:]
            if alt.isalpha() and len(alt) >= 3:
                normalized = alt

        wsname = info.get("wsname", "")
        if wsname:
            ws_base = wsname.split("/")[0]
            normalized = KRAKEN_TO_STANDARD.get(ws_base, ws_base)

        pairs[normalized.upper()] = pair_name

    log.info("Found %d USD pairs on Kraken", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# CoinMarketCap: primary data source (free tier: 10K credits/month)
# ---------------------------------------------------------------------------
def fetch_coinmarketcap_top_coins(
    top_n: int = DEFAULT_TOP_N,
) -> List[dict]:
    """Fetch top coins from CoinMarketCap /v1/cryptocurrency/listings/latest.

    Free tier: 10K credits/month, 30 calls/min.
    A single call with limit=200 uses 1 credit and returns all we need.
    Returns list of dicts matching CoinGecko schema for compatibility.
    """
    api_key = os.environ.get("CMC_API_KEY", "")
    if not api_key:
        raise ValueError("CMC_API_KEY not set -- skipping CoinMarketCap")

    log.info("Fetching top %d coins from CoinMarketCap...", top_n)
    headers = {
        "X-CMC_PRO_API_KEY": api_key,
        "Accept": "application/json",
    }
    params = {
        "start": 1,
        "limit": min(top_n, 200),  # free tier max per call
        "convert": "USD",
        "sort": "market_cap",
        "sort_dir": "desc",
    }

    coins = []
    # Paginate if top_n > 200
    fetched = 0
    while fetched < top_n:
        params["start"] = fetched + 1
        params["limit"] = min(200, top_n - fetched)

        resp = requests.get(
            "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest",
            headers=headers,
            params=params,
            timeout=15,
        )
        if resp.status_code == 429:
            log.warning("CoinMarketCap rate limited")
            raise RuntimeError("CMC rate limited")
        resp.raise_for_status()

        data = resp.json()
        status = data.get("status", {})
        if status.get("error_code", 0) != 0:
            raise RuntimeError(f"CMC error: {status.get('error_message', 'unknown')}")

        page_data = data.get("data", [])
        if not page_data:
            break

        for item in page_data:
            quote = item.get("quote", {}).get("USD", {})
            coins.append({
                "symbol": item.get("symbol", "").upper(),
                "id": item.get("slug", ""),
                "name": item.get("name", ""),
                "market_cap": quote.get("market_cap") or 0,
                "total_volume": quote.get("volume_24h") or 0,
                "current_price": quote.get("price") or 0,
                "market_cap_rank": item.get("cmc_rank") or 9999,
            })

        fetched += len(page_data)
        if len(page_data) < params["limit"]:
            break
        if fetched < top_n:
            time.sleep(2)  # be gentle

    log.info("Fetched %d coins from CoinMarketCap", len(coins))
    return coins


# ---------------------------------------------------------------------------
# CoinGecko: secondary data source (no API key needed)
# ---------------------------------------------------------------------------
def fetch_coingecko_top_coins(
    top_n: int = DEFAULT_TOP_N,
) -> List[dict]:
    """Fetch top coins from CoinGecko /coins/markets endpoint."""
    log.info("Fetching top %d coins from CoinGecko...", top_n)
    coins = []
    per_page = min(250, top_n)
    pages = (top_n + per_page - 1) // per_page

    for page in range(1, pages + 1):
        log.info("  CoinGecko page %d/%d...", page, pages)
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": per_page,
            "page": page,
            "sparkline": "false",
            "locale": "en",
        }
        for attempt in range(3):
            resp = requests.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params=params, timeout=15,
            )
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                log.warning("CoinGecko rate limited -- waiting %ds...", wait)
                time.sleep(wait)
                continue
            break
        resp.raise_for_status()
        page_data = resp.json()
        if not page_data:
            break
        coins.extend(page_data)
        if page < pages:
            time.sleep(7)  # rate limit: ~10 calls/min public API

    log.info("Fetched %d coins from CoinGecko", len(coins))
    return coins


def fetch_kraken_ticker_fallback(
    kraken_pairs: Dict[str, str],
) -> List[dict]:
    """Fallback: use Kraken's Ticker API for volume data when CoinGecko is down.

    Returns list of dicts matching the CoinGecko schema (symbol, total_volume,
    current_price) but with market_cap=0 (Kraken doesn't provide it).
    """
    log.info("CoinGecko unavailable -- using Kraken ticker fallback...")
    # Kraken Ticker endpoint accepts up to ~50 pairs per call
    all_pair_names = list(kraken_pairs.values())
    coins = []

    # Batch requests (Kraken allows comma-separated pairs)
    batch_size = 40
    for i in range(0, len(all_pair_names), batch_size):
        batch = all_pair_names[i : i + batch_size]
        pair_str = ",".join(batch)
        try:
            resp = requests.get(
                "https://api.kraken.com/0/public/Ticker",
                params={"pair": pair_str},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("error"):
                log.warning("Kraken ticker errors: %s", data["error"])
            result = data.get("result", {})

            # Reverse lookup: kraken_pair_name -> normalized symbol
            pair_to_sym = {v: k for k, v in kraken_pairs.items()}

            for pair_name, ticker in result.items():
                sym = pair_to_sym.get(pair_name, "")
                if not sym:
                    continue
                # ticker fields: c=last trade, v=volume(24h), p=vwap
                price = float(ticker["c"][0]) if "c" in ticker else 0
                vol_24h_units = float(ticker["v"][1]) if "v" in ticker else 0
                vol_24h_usd = vol_24h_units * price

                coins.append({
                    "symbol": sym.lower(),
                    "id": sym.lower(),
                    "name": sym,
                    "market_cap": 0,  # Kraken doesn't provide mcap
                    "total_volume": vol_24h_usd,
                    "current_price": price,
                    "market_cap_rank": 9999,
                })
        except Exception as exc:
            log.warning("Kraken ticker batch %d failed: %s", i // batch_size + 1, exc)

        if i + batch_size < len(all_pair_names):
            time.sleep(1)  # Kraken rate limit: ~1 req/sec public

    log.info("Kraken ticker fallback: got data for %d pairs", len(coins))
    return coins


def _universe_cache_age_days(save_dir: Optional[str] = None) -> float:
    """Return age of universe.json in days, or inf if missing."""
    save_dir = save_dir or CRYPTO_MODEL_DIR
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
        age = (datetime.now(timezone.utc) - dt).total_seconds() / 86400
        return age
    except Exception:
        return float("inf")


# ---------------------------------------------------------------------------
# yfinance: validate history length + stability
# ---------------------------------------------------------------------------
def validate_coin_history(
    symbol: str,
    min_days: int = DEFAULT_MIN_HISTORY_DAYS,
    max_flash_crashes: int = DEFAULT_MAX_FLASH_CRASH_DAYS,
) -> dict:
    """Fetch yfinance daily data and validate history length + stability.

    Returns dict with: history_days, avg_dollar_volume_30d,
    volume_active_ratio, realized_vol_ann, stability_ok, reject_reason.
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
        hist = ticker.history(period="2y", interval="1d")
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

        # Stability check: count days with >30% single-day drop on below-average volume
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

        # Check for zombie coins: >50% of recent 30 days have 0 volume or <$10k dollar vol
        if len(dollar_vol) >= 30:
            recent_dv = dollar_vol.tail(30)
            dead_days = (recent_dv < 10_000).sum()
            if dead_days > 15:
                result["stability_ok"] = False
                result["reject_reason"] = f"{dead_days}/30 recent days with near-zero volume"
                return result

        # --- Volume stability: % of recent N days with >$100K dollar volume ---
        lookback = min(VOLUME_STABILITY_LOOKBACK, len(dollar_vol))
        if lookback >= 30:
            recent_dv_window = dollar_vol.tail(lookback)
            active_days = int((recent_dv_window >= VOLUME_STABILITY_FLOOR).sum())
            active_ratio = active_days / lookback
            result["volume_active_ratio"] = round(active_ratio, 3)

            if active_ratio < VOLUME_STABILITY_MIN_RATIO:
                result["stability_ok"] = False
                result["reject_reason"] = (
                    f"volume stability {active_ratio:.0%} < {VOLUME_STABILITY_MIN_RATIO:.0%} "
                    f"({active_days}/{lookback} days with >${VOLUME_STABILITY_FLOOR/1e3:.0f}K vol)"
                )
                return result

        # --- Volatility sanity: annualized realized vol (60d) ---
        vol_window = min(60, len(daily_ret) - 1)
        if vol_window >= 20:
            recent_ret = daily_ret.tail(vol_window).dropna()
            daily_std = float(recent_ret.std())
            ann_vol = daily_std * np.sqrt(365)  # crypto trades 365 days
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
# Main screening pipeline
# ---------------------------------------------------------------------------
def screen_universe(
    top_n: int = DEFAULT_TOP_N,
    min_mcap: float = DEFAULT_MIN_MCAP,
    min_volume: float = DEFAULT_MIN_VOLUME_24H,
    min_vol_mcap_ratio: float = DEFAULT_MIN_VOL_MCAP_RATIO,
    save_dir: Optional[str] = None,
    skip_history: bool = False,
) -> List[ScreenedCoin]:
    """Run the full universe screen.

    Steps:
      1. Fetch Kraken USD pairs
      2. Fetch CoinGecko top coins
      3. Hard filters: symbol exclusions, min mcap, min volume, vol/mcap ratio
      4. (Optional) yfinance history validation: min days, stability checks
      5. Classify liquidity tier + compute dynamic cost thresholds
      6. Save to universe.json
    """
    save_dir = save_dir or CRYPTO_MODEL_DIR

    # Step 1: Kraken tradeable pairs
    kraken_pairs = fetch_kraken_usd_pairs()

    # Step 2: Fetch top coins — 4-tier fallback chain (never fails)
    #   1. CoinMarketCap (primary)  — best free tier, needs CMC_API_KEY
    #   2. CoinGecko     (secondary) — no key needed, rate-limited
    #   3. Kraken Ticker  (tertiary)  — volume/price only, no market cap
    #   4. Cached universe.json       — reuse if < 30 days old
    market_coins = None
    data_source = "unknown"

    # --- Tier 1: CoinMarketCap ---
    try:
        market_coins = fetch_coinmarketcap_top_coins(top_n)
        data_source = "coinmarketcap"
    except Exception as exc:
        log.warning("[Tier 1] CoinMarketCap failed: %s", exc)

    # --- Tier 2: CoinGecko ---
    if not market_coins:
        try:
            market_coins = fetch_coingecko_top_coins(top_n)
            data_source = "coingecko"
        except Exception as exc:
            log.warning("[Tier 2] CoinGecko failed: %s", exc)

    # --- Tier 3: Kraken Ticker ---
    if not market_coins:
        try:
            market_coins = fetch_kraken_ticker_fallback(kraken_pairs)
            data_source = "kraken_ticker"
            log.info("[Tier 3] Using Kraken ticker fallback (no market cap filtering)")
        except Exception as exc:
            log.warning("[Tier 3] Kraken ticker fallback failed: %s", exc)

    # --- Tier 4: Cached universe ---
    if not market_coins:
        cache_age = _universe_cache_age_days(save_dir)
        if cache_age < 30:
            log.warning("[Tier 4] All APIs failed -- reusing cached universe (%.0f days old)", cache_age)
            cached = load_universe_detail(save_dir)
            return [ScreenedCoin(**c) for c in cached] if cached else []
        else:
            log.error("All 4 data sources exhausted (cache %.0f days old) -- returning empty", cache_age)
            return []

    log.info("Data source: %s (%d coins fetched)", data_source, len(market_coins))

    # Step 3: Hard filters
    # When using Kraken fallback, skip market cap filter (not available)
    skip_mcap_filter = (data_source == "kraken_ticker")
    if skip_mcap_filter:
        log.info("Kraken fallback mode: skipping market cap and vol/mcap ratio filters")

    candidates = []
    rejected_hard = 0
    rejected_blacklist = 0
    for coin in market_coins:
        symbol = coin.get("symbol", "").upper()
        mcap = coin.get("market_cap") or 0
        volume = coin.get("total_volume") or 0
        price = coin.get("current_price") or 0
        rank = coin.get("market_cap_rank") or 9999
        name = coin.get("name", "")
        cg_id = coin.get("id", "")

        # Exclusion lists
        if symbol in EXCLUDE_SYMBOLS:
            continue
        if symbol in BLACKLISTED_SYMBOLS:
            rejected_blacklist += 1
            log.debug("Blacklisted %s", symbol)
            continue
        name_lower = name.lower()
        if any(pat in name_lower for pat in EXCLUDE_NAME_PATTERNS):
            continue

        # Min market cap (skip when using Kraken fallback — no mcap data)
        if not skip_mcap_filter and mcap < min_mcap:
            continue

        # Min 24h volume
        if volume < min_volume:
            continue

        # Must be on Kraken
        if symbol not in kraken_pairs:
            continue

        # Tiered turnover check (skip when using Kraken fallback — no mcap)
        vol_mcap = volume / mcap if mcap > 0 else 0
        if not skip_mcap_filter and not check_turnover(mcap, volume):
            rejected_hard += 1
            required = get_required_turnover(mcap)
            log.debug("Rejected %s: turnover=%.3f%% < required %.3f%% (mcap=$%.0fM)",
                       symbol, vol_mcap * 100, required * 100, mcap / 1e6)
            continue

        yf_symbol = f"{symbol}-USD"
        tier = classify_liquidity_tier(volume)
        cost_thresh = get_dynamic_cost_threshold(tier)
        max_order = get_max_order_usd(volume)

        candidates.append(ScreenedCoin(
            symbol=yf_symbol,
            kraken_pair=kraken_pairs[symbol],
            coingecko_id=cg_id,
            name=name,
            market_cap=mcap,
            volume_24h=volume,
            price=price,
            mcap_rank=rank,
            vol_mcap_ratio=round(vol_mcap, 4),
            liquidity_tier=tier,
            cost_threshold=cost_thresh,
            max_order_usd=round(max_order, 0),
        ))

    log.info("Hard filters: %d candidates passed, %d rejected (turnover), %d blacklisted",
             len(candidates), rejected_hard, rejected_blacklist)
    if BLACKLISTED_SYMBOLS:
        log.info("Blacklist active: %d symbols (%s)",
                 len(BLACKLISTED_SYMBOLS), ", ".join(sorted(BLACKLISTED_SYMBOLS)))

    # Step 4: yfinance history validation
    if not skip_history:
        log.info("Validating price history for %d candidates (yfinance)...", len(candidates))
        validated = []
        for i, coin in enumerate(candidates):
            if (i + 1) % 10 == 0:
                log.info("  History check: %d/%d...", i + 1, len(candidates))
            hv = validate_coin_history(coin.symbol)
            coin.history_days = hv["history_days"]
            coin.avg_dollar_volume_30d = hv["avg_dollar_volume_30d"]
            coin.volume_active_ratio = hv.get("volume_active_ratio", 0.0)
            coin.realized_vol_ann = hv.get("realized_vol_ann", 0.0)
            coin.stability_ok = hv["stability_ok"]
            coin.reject_reason = hv["reject_reason"]

            if coin.stability_ok:
                # Update liquidity tier with 30d avg volume (more reliable than 24h snapshot)
                if coin.avg_dollar_volume_30d > 0:
                    coin.liquidity_tier = classify_liquidity_tier(coin.avg_dollar_volume_30d)
                    coin.cost_threshold = get_dynamic_cost_threshold(coin.liquidity_tier)
                    coin.max_order_usd = get_max_order_usd(coin.avg_dollar_volume_30d)
                validated.append(coin)
            else:
                log.info("  Rejected %s: %s", coin.symbol, coin.reject_reason)

            # Be gentle on yfinance
            time.sleep(0.3)

        log.info("History validation: %d passed, %d rejected",
                 len(validated), len(candidates) - len(validated))
        candidates = validated

    # Sort by market cap descending
    candidates.sort(key=lambda c: c.market_cap, reverse=True)

    log.info("Final universe: %d coins (source: %s)", len(candidates), data_source)

    # Step 5: Save — but NEVER overwrite a good universe with an empty one.
    # An empty result likely means an API or network failure, not that all
    # coins genuinely disappeared.  Reuse the cached file instead.
    if not candidates:
        cache_age = _universe_cache_age_days(save_dir)
        if cache_age < 30:
            log.warning("Screen produced 0 coins — keeping cached universe (%.0f days old)", cache_age)
            cached = load_universe_detail(save_dir)
            return [ScreenedCoin(**c) for c in cached] if cached else []
        log.error("Screen produced 0 coins and cache is stale (%.0f days) — saving empty", cache_age)

    _save_universe(candidates, save_dir, data_source=data_source)

    return candidates


def _save_universe(
    coins: List[ScreenedCoin], save_dir: str, data_source: str = "unknown",
) -> None:
    """Save screened universe to JSON."""
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, UNIVERSE_FILE)

    # Tier summary
    tier_counts = {}
    for c in coins:
        tier_counts[c.liquidity_tier] = tier_counts.get(c.liquidity_tier, 0) + 1

    payload = {
        "screened_at": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "count": len(coins),
        "filters": {
            "min_mcap": DEFAULT_MIN_MCAP,
            "min_volume_24h": DEFAULT_MIN_VOLUME_24H,
            "min_vol_mcap_ratio": DEFAULT_MIN_VOL_MCAP_RATIO,
            "min_history_days": DEFAULT_MIN_HISTORY_DAYS,
        },
        "tier_counts": tier_counts,
        "coins": [asdict(c) for c in coins],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Saved universe (%d coins) to %s", len(coins), out_path)
    for tier, count in sorted(tier_counts.items()):
        log.info("  %s: %d coins", tier, count)


def load_universe(save_dir: Optional[str] = None) -> List[str]:
    """Load the screened universe from disk.

    Returns list of yfinance-format symbols (e.g. ['BTC-USD', 'ETH-USD', ...]).
    Falls back to empty list if no universe file exists.
    """
    save_dir = save_dir or CRYPTO_MODEL_DIR
    path = os.path.join(save_dir, UNIVERSE_FILE)
    if not os.path.exists(path):
        log.warning("No universe file at %s -- falling back to hardcoded universe", path)
        return []

    try:
        with open(path) as f:
            data = json.load(f)
        symbols = [c["symbol"] for c in data.get("coins", [])]
        screened_at = data.get("screened_at", "unknown")
        log.info("Loaded universe: %d coins (screened %s)", len(symbols), screened_at)
        return symbols
    except (json.JSONDecodeError, KeyError) as exc:
        log.warning("Corrupt universe file: %s", exc)
        return []


def load_universe_detail(save_dir: Optional[str] = None) -> List[dict]:
    """Load full universe detail (all fields, not just symbols)."""
    save_dir = save_dir or CRYPTO_MODEL_DIR
    path = os.path.join(save_dir, UNIVERSE_FILE)
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("coins", [])
    except (json.JSONDecodeError, KeyError):
        return []


# ---------------------------------------------------------------------------
# Dynamic cost model integration
# ---------------------------------------------------------------------------
def get_coin_cost_config(symbol: str, save_dir: Optional[str] = None) -> dict:
    """Get cost config for a coin from the screened universe.

    Returns dict with: liquidity_tier, cost_threshold, max_order_usd,
    spread_bps, slippage_bps, fee_bps.
    Falls back to conservative defaults if coin not in universe.
    """
    coins = load_universe_detail(save_dir)
    for c in coins:
        if c["symbol"] == symbol or c["symbol"].replace("-", "/") == symbol:
            tier = c.get("liquidity_tier", "small")
            cfg = TIER_THRESHOLDS.get(tier, TIER_THRESHOLDS["small"])
            return {
                "liquidity_tier": tier,
                "cost_threshold": c.get("cost_threshold", 0.012),
                "max_order_usd": c.get("max_order_usd", 5000),
                "spread_bps": cfg["spread_bps"],
                "slippage_bps": cfg["slippage_bps"],
                "fee_bps": cfg["fee_bps"],
                "avg_dollar_volume_30d": c.get("avg_dollar_volume_30d", 0),
            }

    # Conservative fallback for unknown coins
    return {
        "liquidity_tier": "small",
        "cost_threshold": 0.012,
        "max_order_usd": 5000,
        "spread_bps": 15,
        "slippage_bps": 15,
        "fee_bps": 10,
        "avg_dollar_volume_30d": 0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def print_universe_table(coins: List[ScreenedCoin]) -> None:
    """Pretty-print the screened universe."""
    print(f"\n{'='*100}")
    print(f"  Crypto Universe Screen -- {len(coins)} coins qualified")
    print(f"{'='*100}")
    print(f"  {'#':<4} {'Symbol':<12} {'Name':<18} {'Tier':<6} {'MCap':>12} {'24h Vol':>12} "
          f"{'V/MC%':>6} {'Days':>5} {'VolStb':>6} {'AnnVol':>7} {'Cost%':>6} {'MaxOrd':>10}")
    print(f"  {'-'*4} {'-'*12} {'-'*18} {'-'*6} {'-'*12} {'-'*12} "
          f"{'-'*6} {'-'*5} {'-'*6} {'-'*7} {'-'*6} {'-'*10}")
    for i, c in enumerate(coins, 1):
        mcap_str = f"${c.market_cap/1e9:.1f}B" if c.market_cap >= 1e9 else f"${c.market_cap/1e6:.0f}M"
        vol_str = f"${c.volume_24h/1e6:.0f}M" if c.volume_24h >= 1e6 else f"${c.volume_24h/1e3:.0f}K"
        vmr = f"{c.vol_mcap_ratio*100:.1f}" if c.vol_mcap_ratio > 0 else "?"
        days_str = str(c.history_days) if c.history_days > 0 else "?"
        vs = f"{c.volume_active_ratio:.0%}" if c.volume_active_ratio > 0 else "?"
        av = f"{c.realized_vol_ann:.0%}" if c.realized_vol_ann > 0 else "?"
        cost_str = f"{c.cost_threshold*100:.2f}" if c.cost_threshold > 0 else "?"
        order_str = f"${c.max_order_usd/1e3:.0f}K" if c.max_order_usd >= 1000 else f"${c.max_order_usd:.0f}"
        safe_name = c.name.encode("ascii", errors="replace").decode("ascii")[:18]
        print(f"  {i:<4} {c.symbol:<12} {safe_name:<18} {c.liquidity_tier:<6} {mcap_str:>12} {vol_str:>12} "
              f"{vmr:>6} {days_str:>5} {vs:>6} {av:>7} {cost_str:>6} {order_str:>10}")
    print(f"{'='*100}")

    # Tier summary
    tier_counts = {}
    for c in coins:
        tier_counts[c.liquidity_tier] = tier_counts.get(c.liquidity_tier, 0) + 1
    print(f"\n  Tiers: ", end="")
    for tier in ("mega", "large", "mid", "small"):
        if tier in tier_counts:
            cfg = TIER_THRESHOLDS[tier]
            cost_pct = get_dynamic_cost_threshold(tier) * 100
            print(f"{tier}={tier_counts[tier]} (cost>{cost_pct:.2f}%)  ", end="")
    print(f"\n")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Crypto Universe Screener (Layer 0)")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help=f"Scan top N coins by market cap (default: {DEFAULT_TOP_N})")
    parser.add_argument("--min-mcap", type=float, default=DEFAULT_MIN_MCAP,
                        help=f"Minimum market cap in USD (default: {DEFAULT_MIN_MCAP:.0f})")
    parser.add_argument("--min-volume", type=float, default=DEFAULT_MIN_VOLUME_24H,
                        help=f"Minimum 24h volume in USD (default: {DEFAULT_MIN_VOLUME_24H:.0f})")
    parser.add_argument("--save-dir", type=str, default=CRYPTO_MODEL_DIR,
                        help="Directory to save universe.json")
    parser.add_argument("--skip-history", action="store_true",
                        help="Skip yfinance history validation (faster but less thorough)")
    args = parser.parse_args()

    coins = screen_universe(
        top_n=args.top_n,
        min_mcap=args.min_mcap,
        min_volume=args.min_volume,
        save_dir=args.save_dir,
        skip_history=args.skip_history,
    )
    print_universe_table(coins)


if __name__ == "__main__":
    main()
