#!/usr/bin/env python3
"""
HN Topic → Asset Price Correlation Backtest
============================================
Pulls daily HN mention counts via Algolia API, fetches matching asset prices
via yfinance, and computes lead/lag correlations + simple long-only backtest.

Usage:
    python scripts/hn_correlation.py
    python scripts/hn_correlation.py --days 365 --lags 5
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Topic → Asset mapping
# ---------------------------------------------------------------------------
TOPIC_ASSETS = {
    "bitcoin":      {"assets": ["BTC-USD"],          "keywords": ["bitcoin", "btc"]},
    "ethereum":     {"assets": ["ETH-USD"],           "keywords": ["ethereum", "eth"]},
    "solana":       {"assets": ["SOL-USD"],           "keywords": ["solana", "sol"]},
    "crypto":       {"assets": ["BTC-USD", "ETH-USD", "SOL-USD"], "keywords": ["crypto", "cryptocurrency"]},
    "AI":           {"assets": ["SMH", "QQQ", "SOXX"], "keywords": ["artificial intelligence", "openai", "chatgpt", "llm", "gpt"]},
    "semiconductor": {"assets": ["SMH", "SOXX"],      "keywords": ["semiconductor", "chip", "nvidia", "tsmc"]},
    "gold":         {"assets": ["GLD", "GDX"],         "keywords": ["gold", "precious metal"]},
}


# ---------------------------------------------------------------------------
# HN Algolia API — daily mention counts
# ---------------------------------------------------------------------------
def fetch_hn_mentions(keywords: list[str], start_date: str, end_date: str) -> pd.Series:
    """Fetch daily HN story counts matching any keyword via Algolia search API.

    Returns a Series indexed by date with daily mention counts.
    """
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    query = " OR ".join(keywords)
    url = "https://hn.algolia.com/api/v1/search"

    daily_counts: dict[str, int] = {}
    page = 0
    max_pages = 50  # safety limit

    while page < max_pages:
        params = {
            "query": query,
            "tags": "story",
            "numericFilters": f"created_at_i>={start_ts},created_at_i<{end_ts}",
            "hitsPerPage": 1000,
            "page": page,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        hits = data.get("hits", [])
        if not hits:
            break

        for hit in hits:
            ts = hit.get("created_at_i", 0)
            day = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            daily_counts[day] = daily_counts.get(day, 0) + 1

        if page >= data.get("nbPages", 1) - 1:
            break
        page += 1
        time.sleep(0.5)  # rate limit courtesy

    if not daily_counts:
        return pd.Series(dtype=float, name="mentions")

    idx = pd.date_range(start_date, end_date, freq="D")
    series = pd.Series(daily_counts, name="mentions", dtype=float)
    series.index = pd.to_datetime(series.index)
    series = series.reindex(idx, fill_value=0)
    return series


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------
def fetch_prices(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV from yfinance."""
    # Add buffer for forward returns
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=15)
    df = yf.download(symbol, start=start_date, end=end_dt.strftime("%Y-%m-%d"),
                     progress=False, auto_adjust=True)
    if df.empty:
        return df
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df["return_1d"] = df["Close"].pct_change()
    df["return_3d"] = df["Close"].pct_change(3)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)
    return df


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------
def compute_correlations(mentions: pd.Series, prices: pd.DataFrame,
                         max_lag: int = 5) -> pd.DataFrame:
    """Compute lead/lag correlations between HN mentions and forward returns.

    Lag = 0: same-day mentions vs same-day return
    Lag = 1: today's mentions vs tomorrow's return (predictive)
    Lag = -1: yesterday's mentions vs today's return
    """
    # Align to trading days
    merged = pd.DataFrame({
        "mentions": mentions,
        "mentions_ma3": mentions.rolling(3, min_periods=1).mean(),
        "mentions_ma7": mentions.rolling(7, min_periods=1).mean(),
    })
    merged = merged.join(prices[["return_1d", "return_3d", "return_5d", "return_10d"]], how="inner")
    merged = merged.dropna()

    if len(merged) < 30:
        return pd.DataFrame()

    results = []
    for mention_col in ["mentions", "mentions_ma3", "mentions_ma7"]:
        for ret_col in ["return_1d", "return_3d", "return_5d", "return_10d"]:
            for lag in range(-max_lag, max_lag + 1):
                shifted = merged[mention_col].shift(lag)
                valid = pd.concat([shifted, merged[ret_col]], axis=1).dropna()
                if len(valid) < 20:
                    continue
                corr = valid.iloc[:, 0].corr(valid.iloc[:, 1])
                results.append({
                    "mention_type": mention_col,
                    "return_horizon": ret_col,
                    "lag_days": lag,
                    "correlation": corr,
                    "n_samples": len(valid),
                })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Simple backtest: go long when mentions spike
# ---------------------------------------------------------------------------
def backtest_mention_signal(mentions: pd.Series, prices: pd.DataFrame,
                            threshold_pctile: int = 75,
                            hold_days: int = 5) -> dict:
    """Simple long-only backtest: buy when daily mentions exceed the Nth percentile,
    hold for `hold_days`, measure forward return."""
    merged = pd.DataFrame({"mentions": mentions})
    merged = merged.join(prices[["Close", f"return_{hold_days}d"]], how="inner").dropna()

    if len(merged) < 30:
        return {"n_trades": 0}

    threshold = merged["mentions"].quantile(threshold_pctile / 100)
    if threshold <= 0:
        threshold = 1  # at least 1 mention

    signals = merged[merged["mentions"] >= threshold]
    if len(signals) == 0:
        return {"n_trades": 0}

    ret_col = f"return_{hold_days}d"
    trades = signals[ret_col].dropna()

    # Baseline: buy-and-hold over same period
    all_rets = merged[ret_col].dropna()

    return {
        "n_trades": len(trades),
        "mean_return": trades.mean(),
        "median_return": trades.median(),
        "win_rate": (trades > 0).mean(),
        "total_return": (1 + trades).prod() - 1,
        "baseline_mean": all_rets.mean(),
        "baseline_win_rate": (all_rets > 0).mean(),
        "edge_vs_baseline": trades.mean() - all_rets.mean(),
        "mention_threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="HN Topic → Asset Price Correlation")
    parser.add_argument("--days", type=int, default=365, help="Lookback days (default: 365)")
    parser.add_argument("--lags", type=int, default=5, help="Max lag days for correlation (default: 5)")
    parser.add_argument("--topics", default=None, help="Comma-separated topics (default: all)")
    args = parser.parse_args()

    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    topics = list(TOPIC_ASSETS.keys())
    if args.topics:
        topics = [t.strip() for t in args.topics.split(",")]

    all_corr_results = []
    all_backtest_results = []

    for topic in topics:
        cfg = TOPIC_ASSETS.get(topic)
        if not cfg:
            print(f"  Unknown topic: {topic}")
            continue

        print(f"\n{'='*65}")
        print(f"  Topic: {topic.upper()}  |  Keywords: {cfg['keywords']}")
        print(f"  Period: {start_date} -> {end_date} ({args.days} days)")
        print(f"{'='*65}")

        # Fetch HN mentions
        mentions = fetch_hn_mentions(cfg["keywords"], start_date, end_date)
        total_mentions = int(mentions.sum())
        days_with = int((mentions > 0).sum())
        print(f"  HN mentions: {total_mentions} total, {days_with} days with activity")

        if total_mentions < 10:
            print(f"  Too few mentions, skipping.")
            continue

        # For each asset
        for symbol in cfg["assets"]:
            print(f"\n  --- {symbol} ---")
            prices = fetch_prices(symbol, start_date, end_date)
            if prices.empty:
                print(f"    No price data for {symbol}")
                continue

            # Correlations
            corr_df = compute_correlations(mentions, prices, max_lag=args.lags)
            if not corr_df.empty:
                corr_df["topic"] = topic
                corr_df["symbol"] = symbol
                all_corr_results.append(corr_df)

                # Show best predictive correlations (lag > 0 = mentions lead price)
                predictive = corr_df[corr_df["lag_days"] > 0].copy()
                predictive["abs_corr"] = predictive["correlation"].abs()
                best = predictive.nlargest(3, "abs_corr")
                if not best.empty:
                    print(f"    Top predictive correlations (mentions → future returns):")
                    for _, row in best.iterrows():
                        direction = "+" if row["correlation"] > 0 else "-"
                        print(f"      {row['mention_type']} lag={row['lag_days']}d → {row['return_horizon']}: "
                              f"r={row['correlation']:+.3f} ({direction}) n={row['n_samples']}")

            # Backtest
            for hold in [1, 3, 5, 10]:
                ret_col = f"return_{hold}d"
                if ret_col not in prices.columns:
                    continue
                bt = backtest_mention_signal(mentions, prices,
                                             threshold_pctile=75, hold_days=hold)
                if bt["n_trades"] > 0:
                    bt["topic"] = topic
                    bt["symbol"] = symbol
                    bt["hold_days"] = hold
                    all_backtest_results.append(bt)

        time.sleep(1)  # rate limit between topics

    # ---------------------------------------------------------------------------
    # Summary tables
    # ---------------------------------------------------------------------------
    print(f"\n\n{'='*65}")
    print(f"  CORRELATION SUMMARY")
    print(f"{'='*65}\n")

    if all_corr_results:
        corr_all = pd.concat(all_corr_results, ignore_index=True)
        # Best predictive signal per topic-asset pair
        predictive = corr_all[corr_all["lag_days"] > 0].copy()
        predictive["abs_corr"] = predictive["correlation"].abs()
        summary = (predictive
                    .sort_values("abs_corr", ascending=False)
                    .groupby(["topic", "symbol"])
                    .first()
                    .reset_index()
                    [["topic", "symbol", "mention_type", "return_horizon",
                      "lag_days", "correlation", "n_samples"]])
        print(summary.to_string(index=False))
    else:
        print("  No correlation data.")

    print(f"\n\n{'='*65}")
    print(f"  BACKTEST SUMMARY (long when mentions > 75th pctile)")
    print(f"{'='*65}\n")

    if all_backtest_results:
        bt_df = pd.DataFrame(all_backtest_results)
        cols = ["topic", "symbol", "hold_days", "n_trades", "mean_return",
                "win_rate", "baseline_mean", "edge_vs_baseline"]
        bt_display = bt_df[cols].copy()
        for c in ["mean_return", "win_rate", "baseline_mean", "edge_vs_baseline"]:
            bt_display[c] = bt_display[c].map(lambda x: f"{x:+.4f}" if isinstance(x, float) else x)
        print(bt_display.to_string(index=False))

        # Highlight positive edge
        winners = bt_df[bt_df["edge_vs_baseline"] > 0]
        if not winners.empty:
            print(f"\n  Positive edge signals ({len(winners)}):")
            for _, row in winners.iterrows():
                print(f"    {row['topic']} → {row['symbol']} ({row['hold_days']}d hold): "
                      f"edge={row['edge_vs_baseline']:+.4f}, "
                      f"win={row['win_rate']:.0%}, "
                      f"n={row['n_trades']}")
    else:
        print("  No backtest data.")

    print()


if __name__ == "__main__":
    main()
