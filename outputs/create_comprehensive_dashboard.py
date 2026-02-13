#!/usr/bin/env python3
"""
AI-Native Quant Startup Dashboard Generator
=============================================
Reads all output files and creates a single-page website/dashboard hybrid
with an institutional dark theme (Vercel + Two Sigma aesthetic).
"""

import os
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import yfinance as yf


def parse_recent_log_entries(project_root, max_entries=20):
    """Parse the most recent paper trading log for display."""
    logs_dir = os.path.join(project_root, "logs")
    if not os.path.isdir(logs_dir):
        return []
    log_files = sorted(glob.glob(os.path.join(logs_dir, "paper_trader_*.log")))
    if not log_files:
        return []
    latest_log = log_files[-1]
    entries = []
    with open(latest_log, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    cycle_pat = re.compile(r"\[(\d{2}:\d{2}:\d{2})\] === Paper Trading Cycle #(\d+)")
    acct_pat = re.compile(r"Account: \$([\d,.]+) equity")
    i = len(lines) - 1
    while i >= 0 and len(entries) < max_entries:
        m = cycle_pat.search(lines[i])
        if m:
            time_str, cycle_num = m.group(1), int(m.group(2))
            equity_val = None
            symbols_status = []
            for j in range(i + 1, min(i + 15, len(lines))):
                am = acct_pat.search(lines[j])
                if am:
                    equity_val = am.group(1)
                stripped = lines[j].strip()
                if ":  " in stripped and any(
                    kw in stripped for kw in ("HOLD", "SKIP", "BUY", "SELL", "EXIT")
                ):
                    symbols_status.append(stripped)
            entries.append(
                {
                    "time": time_str,
                    "cycle": cycle_num,
                    "equity": equity_val,
                    "symbols": symbols_status,
                }
            )
        i -= 1
    entries.reverse()
    return entries


def consolidate_all_data():
    """Consolidate all individual output files into comprehensive dashboard data."""
    print("Consolidating all output files into comprehensive dashboard...")

    output_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(output_dir)
    legacy_output_dir = os.path.join(project_root, "data", "output")
    models_dirs = [
        os.path.join(project_root, "models"),
        os.path.join(project_root, "data", "models"),
    ]
    source_output_dirs = [output_dir, legacy_output_dir]

    # Read signals data
    signals_file = os.path.join(output_dir, "signals.json")
    signals_data = {}
    if os.path.exists(signals_file):
        with open(signals_file, "r") as f:
            signals_data = json.load(f)

    # Find all symbols from backtests, models, and live signals
    backtest_by_symbol = {}
    trades_by_symbol = {}

    for source_dir in source_output_dirs:
        if not os.path.isdir(source_dir):
            continue
        for backtest_file in glob.glob(os.path.join(source_dir, "backtest_*.csv")):
            symbol = (
                os.path.basename(backtest_file)
                .replace("backtest_", "")
                .replace(".csv", "")
            )
            if symbol not in backtest_by_symbol:
                backtest_by_symbol[symbol] = backtest_file
        for trade_file in glob.glob(os.path.join(source_dir, "trades_*.csv")):
            symbol = (
                os.path.basename(trade_file)
                .replace("trades_", "")
                .replace(".csv", "")
            )
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = trade_file

    model_symbols = set()
    for models_dir in models_dirs:
        if not os.path.isdir(models_dir):
            continue
        for model_file in glob.glob(os.path.join(models_dir, "*_lstm*.pt")):
            model_name = os.path.basename(model_file)
            symbol = model_name.split("_lstm")[0]
            if symbol:
                model_symbols.add(symbol)

    signal_symbols = set(signals_data.get("signals", {}).keys())
    all_symbols = sorted(
        set(backtest_by_symbol.keys()) | model_symbols | signal_symbols
    )

    consolidated_data = {
        "signals": signals_data,
        "symbols": {},
        "portfolio_summary": {},
    }

    total_trades = 0
    winning_trades = 0
    total_pnl = 0
    returns = []
    sharpes = []
    drawdowns = []
    gross_profit = 0
    gross_loss = 0

    symbols_with_backtests = 0
    for symbol in all_symbols:
        backtest_file = backtest_by_symbol.get(symbol)
        print(f"  Processing {symbol}...")

        equity_records = []
        metrics = {
            "has_backtest": False,
            "total_return": None,
            "sharpe_ratio": None,
            "max_drawdown": None,
            "initial_equity": None,
            "final_equity": None,
        }

        if backtest_file and os.path.exists(backtest_file):
            symbols_with_backtests += 1
            equity_df = pd.read_csv(backtest_file)
            equity_df["date"] = pd.to_datetime(equity_df["date"])

            initial_equity = equity_df["equity"].iloc[0]
            final_equity = equity_df["equity"].iloc[-1]
            total_return = (final_equity / initial_equity - 1) * 100

            equity_df["daily_return"] = equity_df["equity"].pct_change()
            mean_return = equity_df["daily_return"].mean()
            std_return = equity_df["daily_return"].std()
            sharpe = (
                (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0
            )

            peak = equity_df["equity"].expanding(min_periods=1).max()
            drawdown = (equity_df["equity"] - peak) / peak * 100
            max_drawdown = drawdown.min()

            equity_records = equity_df.to_dict("records")
            metrics = {
                "has_backtest": True,
                "total_return": round(total_return, 2),
                "sharpe_ratio": round(sharpe, 2),
                "max_drawdown": round(max_drawdown, 2),
                "initial_equity": initial_equity,
                "final_equity": final_equity,
            }
            returns.append(total_return)
            sharpes.append(sharpe)
            drawdowns.append(max_drawdown)

        # Load trades data
        trade_file = trades_by_symbol.get(symbol)
        trades_data = {
            "trades": [],
            "summary": {
                "total_trades": 0,
                "winning_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_trade": 0,
            },
        }

        if trade_file and os.path.exists(trade_file):
            trades_df = pd.read_csv(trade_file)
            trades_data["trades"] = trades_df.to_dict("records")

            symbol_trades = len(trades_df)
            symbol_wins = (
                len(trades_df[trades_df["pnl"] > 0]) if "pnl" in trades_df else 0
            )
            symbol_win_rate = (
                (symbol_wins / symbol_trades * 100) if symbol_trades > 0 else 0
            )
            symbol_pnl = trades_df["pnl"].sum() if "pnl" in trades_df else 0

            # Profit factor components
            if "pnl" in trades_df:
                gross_profit += trades_df[trades_df["pnl"] > 0]["pnl"].sum()
                gross_loss += abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())

            trades_data["summary"] = {
                "total_trades": symbol_trades,
                "winning_trades": symbol_wins,
                "win_rate": symbol_win_rate,
                "total_pnl": symbol_pnl,
                "avg_trade": symbol_pnl / symbol_trades if symbol_trades > 0 else 0,
            }

            total_trades += symbol_trades
            winning_trades += symbol_wins
            total_pnl += symbol_pnl

        # Fetch price history for context
        price_records = []
        try:
             # Fetch for all symbols to ensure we have context for live signals too
             # We assume 2024-01-01 start for simplicity as per dashboard context. 
             # Using current date as end to ensure we get latest data.
             ticker = yf.Ticker(symbol)
             hist = ticker.history(start="2024-01-01", interval="1d")
             hist = hist.reset_index()
             for _, row in hist.iterrows():
                 dt = row["Date"]
                 # Handle different datetime formats from yfinance
                 dt_str = dt.strftime("%Y-%m-%d") if pd.notnull(dt) else ""
                 if dt_str:
                     price_records.append({
                         "date": dt_str,
                         "close": round(row["Close"], 2)
                     })
        except Exception as e:
            print(f"    Warning: Could not fetch price data for {symbol}: {e}")

        consolidated_data["symbols"][symbol] = {
            "equity_curve": equity_records,
            "trades": trades_data,
            "metrics": metrics,
            "prices": price_records
        }

    # Portfolio summary with new fields
    consolidated_data["portfolio_summary"] = {
        "total_symbols": len(all_symbols),
        "symbols_with_backtests": symbols_with_backtests,
        "total_trades": total_trades,
        "overall_win_rate": round(
            (winning_trades / total_trades * 100) if total_trades > 0 else 0, 1
        ),
        "total_pnl": round(total_pnl, 2),
        "avg_return": round(np.mean(returns) if returns else 0, 2),
        "total_return_range": (
            f"{min(returns):.1f}% to {max(returns):.1f}%" if returns else "N/A"
        ),
        "avg_sharpe": round(np.mean(sharpes) if sharpes else 0, 2),
        "profit_factor": round(
            gross_profit / gross_loss if gross_loss > 0 else 0, 2
        ),
        "worst_drawdown": round(min(drawdowns) if drawdowns else 0, 2),
    }

    # Parse log entries for the Logs section
    consolidated_data["log_entries"] = parse_recent_log_entries(project_root)

    # Parse live paper trading equity history
    consolidated_data["live_pnl"] = parse_live_equity_history(project_root)

    # Fetch live VIX from yfinance (FRED data is delayed by 1+ day)
    try:
        vix_ticker = yf.Ticker("^VIX")
        vix_hist = vix_ticker.history(period="2d")
        if len(vix_hist) >= 1:
            live_vix = round(float(vix_hist["Close"].iloc[-1]), 2)
            # Override stale FRED VIX in signals data
            if "market" not in consolidated_data["signals"]:
                consolidated_data["signals"]["market"] = {}
            prev_vix = float(vix_hist["Close"].iloc[-2]) if len(vix_hist) >= 2 else live_vix
            vix_chg = round((live_vix - prev_vix) / prev_vix * 100, 3) if prev_vix > 0 else 0
            consolidated_data["signals"]["market"]["vix"] = live_vix
            consolidated_data["signals"]["market"]["vix_change_1d_pct"] = vix_chg
            print(f"  Live VIX: {live_vix} ({vix_chg:+.2f}%)")
    except Exception as e:
        print(f"  Warning: Could not fetch live VIX: {e}")

    print(
        f"  Consolidated {len(all_symbols)} symbols "
        f"({symbols_with_backtests} with backtests), {total_trades} total trades"
    )
    return consolidated_data


# ---------------------------------------------------------------------------
# Section builders — each returns an HTML string
# ---------------------------------------------------------------------------


def _html_head():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quantitative Stocks &mdash; AI-Native Quant</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
/* ---- Reset & Base ---- */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth;-webkit-font-smoothing:antialiased}
body{
  font-family:'Inter',system-ui,-apple-system,sans-serif;
  background:#0b0f19;
  color:#94a3b8;
  font-size:15px;
  line-height:1.6;
}

/* ---- Tokens ---- */
:root{
  --bg:#0b0f19;
  --bg-card:rgba(255,255,255,0.03);
  --bg-card-hover:rgba(255,255,255,0.055);
  --bg-elevated:rgba(255,255,255,0.06);
  --border:rgba(255,255,255,0.06);
  --border-muted:rgba(255,255,255,0.04);
  --accent:#3b82f6;
  --accent-glow:rgba(59,130,246,0.15);
  --accent-muted:rgba(59,130,246,0.10);
  --text:#f8fafc;
  --text-sec:#94a3b8;
  --text-muted:#64748b;
  --text-dim:#475569;
  --positive:#22c55e;
  --positive-bg:rgba(34,197,94,0.08);
  --negative:#ef4444;
  --negative-bg:rgba(239,68,68,0.08);
  --warning:#f59e0b;
  --mono:'JetBrains Mono','SF Mono','Fira Code',monospace;
  --radius:12px;
  --radius-sm:8px;
  --radius-xs:4px;
}
.mono{font-family:var(--mono)}

/* ---- Trade History Table ---- */
.trade-history-details { margin-top: 16px; border-top: 1px solid var(--border-muted); padding-top: 12px; }
.trade-history-summary { cursor: pointer; color: var(--accent); font-size: 11px; font-weight: 600; text-transform: uppercase; list-style: none; display: flex; align-items: center; gap: 4px; }
.trade-history-summary::-webkit-details-marker { display: none; }
.trade-history-summary::after { content: '▾'; font-size: 14px; }
details[open] .trade-history-summary::after { content: '▴'; }

.trade-table-wrap { max-height: 200px; overflow-y: auto; margin-top: 12px; border: 1px solid var(--border-muted); border-radius: 4px; }
.trade-table-wrap::-webkit-scrollbar { width: 4px; }
.trade-table-wrap::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.trade-table { width: 100%; border-collapse: collapse; font-size: 11px; font-family: var(--mono); }
.trade-table th { position: sticky; top: 0; background: #0b0f19; color: var(--text-muted); font-weight: 600; text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border-muted); }
.trade-table td { padding: 4px 8px; border-bottom: 1px solid var(--border-muted); color: var(--text-sec); }
.trade-table tr:last-child td { border-bottom: none; }

/* ---- Layout ---- */
.container{max-width:1440px;margin:0 auto;padding:0 40px}
@media(max-width:768px){.container{padding:0 20px}}
section,.section-wrap{padding:100px 0}
@media(max-width:768px){section,.section-wrap{padding:60px 0}}

/* ---- Glass Card ---- */
.glass{
  background:var(--bg-card);
  border:1px solid var(--border);
  border-radius:var(--radius);
  backdrop-filter:blur(12px);
  -webkit-backdrop-filter:blur(12px);
  transition:background .2s,border-color .2s;
}
.glass:hover{background:var(--bg-card-hover);border-color:rgba(255,255,255,0.09)}

/* ---- Section Labels ---- */
.sec-label{
  font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
  color:var(--accent);margin-bottom:8px;
}
.sec-title{
  font-size:1.5rem;font-weight:700;color:var(--text);letter-spacing:-.02em;
  margin-bottom:40px;
}

/* ---- Reveal animation ---- */
.reveal{opacity:0;transform:translateY(24px);transition:opacity .6s ease,transform .6s ease}
.reveal.visible{opacity:1;transform:translateY(0)}

/* ---- Positive / Negative ---- */
.pos{color:var(--positive)}.neg{color:var(--negative)}
.pos-bg{background:var(--positive-bg);color:var(--positive);padding:2px 8px;border-radius:var(--radius-xs);font-size:13px}
.neg-bg{background:var(--negative-bg);color:var(--negative);padding:2px 8px;border-radius:var(--radius-xs);font-size:13px}

/* ============ HERO ============ */
#hero{
  position:relative;min-height:80vh;display:flex;flex-direction:column;align-items:center;
  justify-content:center;text-align:center;overflow:hidden;padding-top:80px;
}
#hero::before{
  content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse 80% 60% at 50% 40%,rgba(59,130,246,.12),transparent 70%),
             radial-gradient(ellipse 60% 50% at 30% 70%,rgba(99,102,241,.08),transparent 60%);
  animation:gradShift 15s ease infinite alternate;
  pointer-events:none;
}
@keyframes gradShift{
  0%{opacity:.6;transform:scale(1)}
  100%{opacity:1;transform:scale(1.08)}
}
.hero-eyebrow{
  font-size:11px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);margin-bottom:20px;
}
.hero-title{
  font-size:clamp(3rem,7vw,5rem);font-weight:800;color:var(--text);
  letter-spacing:-.04em;line-height:1.1;margin-bottom:16px;
  background: linear-gradient(to right, #fff, #94a3b8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.hero-tagline{
  font-size:clamp(1.1rem,2vw,1.3rem);color:var(--text-sec);max-width:600px;
  margin:0 auto 40px;
}
.hero-stats{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 24px;
    margin-top: 40px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 32px;
    max-width: 900px;
    width: 100%;
}
@media(max-width: 768px) {
    .hero-stats { grid-template-columns: repeat(2, 1fr); }
}
.h-stat { text-align: center; }
.h-stat-label { 
    font-size: 11px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; 
    color: var(--text-muted); margin-bottom: 8px; 
}
.h-stat-val { 
    font-family: var(--mono); font-size: 2.5rem; font-weight: 700; color: var(--text); 
    line-height: 1.1;
}
.h-stat-sub { font-size: 12px; color: var(--text-dim); margin-top: 4px; }

/* ============ LIVE PANEL ============ */
.market-bar{display:flex;align-items:center;gap:32px;padding:16px 24px;margin-bottom:24px;flex-wrap:wrap}
.market-stat{display:flex;align-items:center;gap:8px}
.market-label{font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--text-muted)}
.market-val{font-family:var(--mono);font-size:15px;font-weight:600;color:var(--text)}
.live-dot{width:6px;height:6px;border-radius:50%;background:var(--positive);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

.signals-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}
.sig-card{padding:20px}
.sig-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
.sig-symbol{font-size:15px;font-weight:700;color:var(--text)}
.sig-dir{font-size:11px;font-weight:700;letter-spacing:.06em;padding:3px 10px;border-radius:var(--radius-xs)}
.sig-dir.up{color:var(--positive);background:var(--positive-bg)}
.sig-dir.down{color:var(--negative);background:var(--negative-bg)}
.conf-bar-wrap{height:4px;background:rgba(255,255,255,0.04);border-radius:2px;margin-bottom:16px;overflow:hidden}
.conf-bar{height:100%;border-radius:2px;background:var(--accent);transition:width 1s ease}
.sig-metrics{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.sig-m{display:flex;justify-content:space-between;font-size:12px}
.sig-m-label{color:var(--text-muted)}
.sig-m-val{font-family:var(--mono);color:var(--text-sec)}

/* ============ LIVE P&L ============ */
.live-pos{padding:16px;margin-bottom:12px}
.live-pos:last-child{margin-bottom:0}
.live-pos-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.live-pos-label{font-size:10px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--text-muted);margin-bottom:4px}
.live-pos-val{font-family:var(--mono);font-size:13px;font-weight:600;color:var(--text-sec)}
@media(max-width:768px){
  #live-pnl .container > div:first-child{grid-template-columns:1fr !important}
}

/* ============ KPI ============ */
.kpi-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:16px;margin-bottom:16px}
@media(max-width:1024px){.kpi-grid{grid-template-columns:repeat(3,1fr)}}
@media(max-width:768px){.kpi-grid{grid-template-columns:repeat(2,1fr)}}
.kpi{padding:24px;text-align:center}
.kpi-label{font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--text-muted);margin-bottom:8px}
.kpi-val{font-family:var(--mono);font-size:2rem;font-weight:700;color:var(--text);line-height:1.2}
@media(max-width:768px){.kpi-val{font-size:1.5rem}}
.kpi-sub{font-size:11px;color:var(--text-dim);margin-top:6px}
.kpi-disclaimer{font-size:12px;color:var(--text-dim);text-align:center;margin-top:16px}

.metrics-table-wrap{padding:0 0 16px;overflow:hidden}
.metrics-table{width:100%;border-collapse:collapse;font-size:12px;font-family:var(--mono)}
.metrics-table th{text-align:left;padding:10px 16px;font-size:10px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--text-muted);border-bottom:1px solid var(--border-muted);white-space:nowrap}
.metrics-table td{padding:10px 16px;border-bottom:1px solid rgba(255,255,255,0.03);color:var(--text-sec);white-space:nowrap}
.metrics-table tbody tr:hover{background:rgba(255,255,255,0.02)}
.metrics-table tbody tr:last-child td{border-bottom:none}

/* ============ EQUITY CHART ============ */
.chart-wrap{padding:24px;margin-bottom:16px}
.chart-wrap .js-plotly-plot{border-radius:8px;overflow:hidden}

/* ============ STRATEGY CARDS ============ */
.strat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;align-items:start}
@media(max-width:1024px){.strat-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:768px){.strat-grid{grid-template-columns:1fr}}
.strat{padding:24px}
.strat-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
.strat-sym{font-size:18px;font-weight:700;color:var(--text)}
.strat-badge{font-size:10px;font-weight:700;letter-spacing:.06em;padding:3px 10px;border-radius:var(--radius-xs);text-transform:uppercase}
.strat-badge.long{color:var(--positive);background:var(--positive-bg)}
.strat-badge.short{color:var(--negative);background:var(--negative-bg)}
.strat-badge.both{color:var(--accent);background:var(--accent-muted)}
.strat-spark{height:50px;margin-bottom:12px}
.strat-stats{display:grid;grid-template-columns:1fr 1fr;gap:8px 16px;margin-bottom:16px}
.strat-s{display:flex;justify-content:space-between;font-size:13px}
.strat-s-label{color:var(--text-muted)}
.strat-s-val{font-family:var(--mono);font-weight:600;color:var(--text)}
.strat-recent-label{font-size:10px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--text-muted);margin-bottom:8px}
.strat-trade{display:flex;align-items:center;gap:8px;font-size:12px;padding:4px 0;border-bottom:1px solid var(--border-muted)}
.strat-trade:last-child{border-bottom:none}
.strat-trade-dir{font-weight:700;width:40px;font-size:10px;letter-spacing:.04em}
.strat-trade-date{font-family:var(--mono);color:var(--text-muted);flex:1}
.strat-trade-pnl{font-family:var(--mono);font-weight:600}

/* ============ RESEARCH ============ */
.research-grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}
@media(max-width:768px){.research-grid{grid-template-columns:1fr}}
.lstm-diagram{display:flex;flex-direction:column;align-items:center;gap:0;padding:32px}
.lstm-layer{
  width:180px;text-align:center;padding:14px;border-radius:var(--radius-sm);
  border:1px solid var(--border);background:var(--bg-card);
}
.lstm-layer .layer-label{font-size:12px;font-weight:600;color:var(--text);display:block}
.lstm-layer .layer-detail{font-family:var(--mono);font-size:11px;color:var(--text-muted);display:block;margin-top:2px}
.lstm-layer.input-layer{border-color:var(--accent);background:var(--accent-muted)}
.lstm-layer.output-layer{border-color:var(--positive);background:var(--positive-bg)}
.lstm-arrow{width:2px;height:20px;background:var(--border);margin:0 auto;position:relative}
.lstm-arrow::after{content:'';position:absolute;bottom:-3px;left:-3px;width:8px;height:8px;border-right:2px solid var(--text-muted);border-bottom:2px solid var(--text-muted);transform:rotate(45deg)}
.research-details{display:flex;flex-direction:column;gap:16px}
.research-card{padding:20px}
.research-card h4{font-size:13px;font-weight:700;color:var(--text);margin-bottom:12px}
.features-grid{display:flex;flex-wrap:wrap;gap:6px}
.feat-tag{font-family:var(--mono);font-size:11px;padding:4px 10px;border-radius:var(--radius-xs);background:var(--bg-elevated);color:var(--text-sec);border:1px solid var(--border-muted)}
.research-list{list-style:none;padding:0}
.research-list li{font-size:13px;color:var(--text-sec);padding:4px 0;padding-left:16px;position:relative}
.research-list li::before{content:'';position:absolute;left:0;top:11px;width:4px;height:4px;border-radius:50%;background:var(--accent)}

/* ============ PIPELINE ============ */
.pipeline-flow{display:flex;align-items:center;gap:0;overflow-x:auto;padding:16px 0}
@media(max-width:768px){.pipeline-flow{flex-direction:column}}
.pipe-stage{flex:1;min-width:160px;padding:24px 16px;text-align:center}
.pipe-icon{font-size:24px;margin-bottom:8px;color:var(--accent)}
.pipe-stage h4{font-size:13px;font-weight:700;color:var(--text);margin-bottom:4px}
.pipe-stage p{font-size:11px;color:var(--text-muted)}
.pipe-conn{display:flex;align-items:center;width:40px;justify-content:center;flex-shrink:0}
@media(max-width:768px){.pipe-conn{width:auto;height:24px;transform:rotate(90deg)}}
.pipe-conn-line{width:40px;height:2px;background:linear-gradient(90deg,var(--border),var(--accent),var(--border))}

/* ============ LOGS ============ */
.logs-box{padding:0;max-height:400px;overflow-y:auto}
.logs-box::-webkit-scrollbar{width:4px}
.logs-box::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.log-entry{display:flex;align-items:flex-start;gap:12px;padding:10px 20px;border-bottom:1px solid var(--border-muted);font-size:12px}
.log-entry:last-child{border-bottom:none}
.log-time{font-family:var(--mono);color:var(--text-muted);white-space:nowrap;min-width:60px}
.log-cycle{font-family:var(--mono);color:var(--accent);white-space:nowrap;min-width:60px}
.log-equity{font-family:var(--mono);color:var(--text);white-space:nowrap;min-width:100px}
.log-detail{color:var(--text-sec);flex:1;word-break:break-word}
.logs-empty{padding:40px;text-align:center;color:var(--text-dim)}

/* ============ FAQ ============ */
.faq-list{max-width:800px;margin:0 auto;display:flex;flex-direction:column;gap:8px}
.faq-item{border-radius:var(--radius);overflow:hidden}
.faq-item summary{
  cursor:pointer;padding:16px 20px;font-size:14px;font-weight:600;color:var(--text);
  list-style:none;display:flex;justify-content:space-between;align-items:center;
}
.faq-item summary::-webkit-details-marker{display:none}
.faq-item summary::after{content:'+';font-size:18px;color:var(--text-muted);transition:transform .2s}
.faq-item[open] summary::after{content:'-'}
.faq-answer{padding:0 20px 16px;font-size:13px;color:var(--text-sec);line-height:1.7}

/* ============ FOOTER ============ */
footer{border-top:1px solid var(--border);padding:40px 0}
.footer-content{text-align:center}
.footer-brand{display:flex;align-items:center;justify-content:center;gap:8px;margin-bottom:12px}
.footer-logo{font-family:var(--mono);font-size:14px;font-weight:700;color:var(--accent);background:var(--accent-muted);padding:4px 10px;border-radius:var(--radius-xs)}
.footer-name{font-size:14px;font-weight:600;color:var(--text)}
.footer-disclaimer{font-size:11px;color:var(--text-dim);max-width:600px;margin:0 auto 16px;line-height:1.6}
.footer-meta{font-size:11px;color:var(--text-dim);display:flex;justify-content:center;gap:16px;flex-wrap:wrap}
.footer-meta .mono{color:var(--text-muted)}

/* ---- Nav (minimal top bar) ---- */
.topnav{
  position:fixed;top:0;left:0;right:0;z-index:100;
  background:rgba(11,15,25,.85);backdrop-filter:blur(12px);
  border-bottom:1px solid var(--border-muted);
  padding:0 40px;display:flex;align-items:center;height:48px;
}
.topnav-brand{font-family:var(--mono);font-size:13px;font-weight:700;color:var(--accent)}
.topnav-links{display:flex;gap:24px;margin-left:auto}
.topnav-links a{font-size:12px;font-weight:500;color:var(--text-muted);text-decoration:none;transition:color .2s}
.topnav-links a:hover{color:var(--text)}
@media(max-width:768px){.topnav-links{display:none}}
</style>
</head>
<body>

<!-- Top Nav -->
<nav class="topnav">
  <span class="topnav-brand">QS</span>
  <div class="topnav-links">
    <a href="#live-panel">Signals</a>
    <a href="#live-pnl">Live P&L</a>
    <a href="#metrics">Metrics</a>
    <a href="#equity">Equity</a>
    <a href="#strategies">Strategies</a>
    <a href="#research">Research</a>
    <a href="#pipeline">Pipeline</a>
    <a href="#logs">Logs</a>
    <a href="#faq">FAQ</a>
  </div>
</nav>
'''


def _section_hero(_data):
    return '''
<!-- HERO -->
<section id="hero" style="position: relative; overflow: hidden; background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);">
  <canvas id="hero-canvas" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; opacity: 0.6; pointer-events: none; mix-blend-mode: screen;"></canvas>
  <div class="hero-content" style="position:relative;z-index:1; width: 100%; display: flex; flex-direction: column; align-items: center; text-shadow: 0 4px 20px rgba(0,0,0,0.8);">
    <p class="hero-eyebrow" style="background: rgba(34, 197, 94, 0.1); color: #4ade80; padding: 4px 12px; border-radius: 20px; border: 1px solid rgba(34, 197, 94, 0.2); backdrop-filter: blur(4px);">AI-NATIVE QUANTITATIVE TRADING</p>
    <h1 class="hero-title" style="margin-top: 16px;">Quantitative Stocks</h1>
    <p class="hero-tagline">Result-driven systematic trading across equity and commodity ETFs, powered by LSTM neural networks.</p>
  </div>
  <script>
    (function() {
        const canvas = document.getElementById('hero-canvas');
        const ctx = canvas.getContext('2d');
        let width, height;
        
        // Math Themes
        const mathSymbols = ['∑', '∫', '∂', 'λ', 'σ', 'μ', 'Δ', '∇', 'π', 'θ', 'Ω', 'β'];
        
        let nodes = [];
        let floaters = [];
        
        // Configuration
        const nodeCount = 70; // Dense network
        const floaterCount = 15; // Floating symbols
        const connectionDist = 160;
        const mouseDist = 200;
        
        let mouse = { x: null, y: null };
        
        window.addEventListener('mousemove', function(e) {
            const rect = canvas.getBoundingClientRect();
            mouse.x = e.clientX - rect.left;
            mouse.y = e.clientY - rect.top;
        });
        
        window.addEventListener('mouseleave', function() {
            mouse.x = null;
            mouse.y = null;
        });

        function resize() {
            width = canvas.width = canvas.parentElement.offsetWidth;
            height = canvas.height = canvas.parentElement.offsetHeight;
        }
        
        class Node {
            constructor() {
                this.x = Math.random() * width;
                this.y = Math.random() * height;
                this.vx = (Math.random() - 0.5) * 0.8;
                this.vy = (Math.random() - 0.5) * 0.8;
                this.size = Math.random() * 2 + 1.5;
            }
            
            update() {
                this.x += this.vx;
                this.y += this.vy;
                
                // Mouse interaction - gentle repulsion
                if (mouse.x != null) {
                    let dx = mouse.x - this.x;
                    let dy = mouse.y - this.y;
                    let distance = Math.sqrt(dx*dx + dy*dy);
                    if (distance < mouseDist) {
                        const forceDirectionX = dx / distance;
                        const forceDirectionY = dy / distance;
                        const force = (mouseDist - distance) / mouseDist;
                        const directionX = forceDirectionX * force * 2;
                        const directionY = forceDirectionY * force * 2;
                        this.vx -= directionX * 0.05;
                        this.vy -= directionY * 0.05;
                    }
                }

                if (this.x < 0 || this.x > width) this.vx *= -1;
                if (this.y < 0 || this.y > height) this.vy *= -1;
            }
            
            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                ctx.fillStyle = '#60a5fa'; // Blue-400
                ctx.fill();
            }
        }
        
        class Floater {
            constructor() {
                this.reset();
                this.y = Math.random() * height; // Start anywhere vertically
            }
            
            reset() {
                this.x = Math.random() * width;
                this.y = height + 50;
                this.speed = Math.random() * 0.5 + 0.2;
                this.symbol = mathSymbols[Math.floor(Math.random() * mathSymbols.length)];
                this.opacity = 0;
                this.size = Math.random() * 14 + 10;
                this.maxOpacity = Math.random() * 0.4 + 0.1;
                this.life = 0;
            }
            
            update() {
                this.y -= this.speed;
                this.life += 0.01;
                
                // Fade in/out
                if (this.life < 1) this.opacity = this.life * this.maxOpacity;
                if (this.y < height * 0.2) this.opacity -= 0.005;
                
                if (this.y < -50 || this.opacity <= 0 && this.life > 2) {
                    this.reset();
                }
            }
            
            draw() {
                ctx.font = `lighter ${this.size}px "JetBrains Mono", monospace`;
                ctx.fillStyle = `rgba(148, 163, 184, ${this.opacity})`; // Slate-400
                ctx.fillText(this.symbol, this.x, this.y);
            }
        }
        
        function init() {
            resize();
            nodes = [];
            floaters = [];
            for (let i = 0; i < nodeCount; i++) nodes.push(new Node());
            for (let i = 0; i < floaterCount; i++) floaters.push(new Floater());
        }
        
        function animate() {
            ctx.clearRect(0, 0, width, height);
            
            // Draw floating math symbols first (background)
            floaters.forEach(f => {
                f.update();
                f.draw();
            });

            // Update interactions
            for (let i = 0; i < nodes.length; i++) {
                nodes[i].update();
                nodes[i].draw();
                
                // Draw Connections
                for (let j = i + 1; j < nodes.length; j++) {
                    const dx = nodes[i].x - nodes[j].x;
                    const dy = nodes[i].y - nodes[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    
                    if (dist < connectionDist) {
                        ctx.beginPath();
                        const alpha = 1 - (dist / connectionDist);
                        ctx.strokeStyle = `rgba(56, 189, 248, ${alpha * 0.4})`; // Sky-400 lines
                        ctx.lineWidth = 1;
                        ctx.moveTo(nodes[i].x, nodes[i].y);
                        ctx.lineTo(nodes[j].x, nodes[j].y);
                        ctx.stroke();
                        
                        // Occasionally draw a triangle for "mesh" look
                        for (let k = j + 1; k < nodes.length; k++) {
                            const dx2 = nodes[j].x - nodes[k].x;
                            const dy2 = nodes[j].y - nodes[k].y;
                            const dist2 = Math.sqrt(dx2 * dx2 + dy2 * dy2);
                             if (dist2 < connectionDist) {
                                ctx.beginPath();
                                ctx.fillStyle = `rgba(56, 189, 248, ${alpha * 0.05})`; // Very faint fill
                                ctx.moveTo(nodes[i].x, nodes[i].y);
                                ctx.lineTo(nodes[j].x, nodes[j].y);
                                ctx.lineTo(nodes[k].x, nodes[k].y);
                                ctx.closePath();
                                ctx.fill();
                             }
                        }
                    }
                }
            }
            
            requestAnimationFrame(animate);
        }
        
        window.addEventListener('resize', resize);
        init();
        animate();
    })();
  </script>
</section>
'''


def _section_live_panel(data):
    sig = data.get("signals", {})
    market = sig.get("market", {})
    vix = market.get("vix", "N/A")
    vix_chg = market.get("vix_change_1d_pct", 0)
    asof = sig.get("asof", "")
    if asof:
        try:
            from zoneinfo import ZoneInfo
            dt = datetime.fromisoformat(asof)
            dt_est = dt.astimezone(ZoneInfo("America/New_York"))
            asof_fmt = dt_est.strftime("%b %d, %Y %I:%M %p EST")
            # Show how stale the data is
            now_utc = datetime.now(ZoneInfo("UTC"))
            age = now_utc - dt
            age_mins = int(age.total_seconds() / 60)
            if age_mins < 2:
                age_str = "just now"
            elif age_mins < 60:
                age_str = f"{age_mins}m ago"
            elif age_mins < 1440:
                age_str = f"{age_mins // 60}h {age_mins % 60}m ago"
            else:
                age_str = f"{age_mins // 1440}d ago"
            asof_fmt += f" ({age_str})"
        except Exception:
            asof_fmt = asof
    else:
        asof_fmt = "N/A"

    vix_cls = "pos" if vix_chg < 0 else "neg" if vix_chg > 0 else ""
    vix_sign = "+" if vix_chg > 0 else ""

    signals = sig.get("signals", {})
    
    # Get open positions from logs to display ACTUAL sizing
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    open_positions = parse_open_positions_from_logs(project_root)
    # Get latest account equity from logs if possible, else use default $100k for estimation
    # Or calculate percentage based on position value / total equity 
    # The parse_open_positions_from_logs doesn't return total equity currently.
    # Let's enhance it quickly or just use the log parsing we already have in _section_logs but better.
    # For now, we'll try to extract equity from the latest log line that has it.
    
    latest_equity = 100000.0
    try:
        logs_dir = os.path.join(project_root, "logs")
        if os.path.isdir(logs_dir):
            log_files = sorted(glob.glob(os.path.join(logs_dir, "paper_trader_*.log")))
            if log_files:
                with open(log_files[-1], "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                    # Find last occurrence of "Account: $..."
                    matches = re.findall(r"Account: \$([\d,.]+) equity", content)
                    if matches:
                        latest_equity = float(matches[-1].replace(",", ""))
    except:
        pass

    cards_html = ""
    for sym, s in sorted(signals.items()):
        direction = s.get("ml_direction", "UNKNOWN")
        conf = s.get("ml_confidence", 0)
        conf_pct = conf * 100 if conf <= 1 else conf
        
        # Determine actionable signal
        action = "HOLD"
        act_cls = "text-muted"
        color_var = "accent"
        
        if direction == "UP":
            action = "BUY" if conf_pct > 60 else "BULLISH"
            act_cls = "pos"
            color_var = "positive"
        elif direction == "DOWN":
            action = "SELL" if conf_pct > 60 else "BEARISH"
            act_cls = "neg"
            color_var = "negative"
            
        rsi = s.get("rsi14", 0)
        ret5 = s.get("ret5", 0)
        ret5_cls = "pos" if ret5 >= 0 else "neg"
        ret5_sign = "+" if ret5 >= 0 else ""

        # Logic for Sizing Display
        # 1. If we have an open position, show THAT size.
        # 2. If no position, show recommended size based on signal.
        
        sizing_label = "Rec. Size" # Default
        sizing_val = "0.0%"
        sizing_color = "var(--text-muted)"
        
        if sym in open_positions:
            pos = open_positions[sym]
            # Calculate % of equity
            # value = qty * price
            # We have entry price in pos, current price in s? s doesn't have price directly usually.
            # We can use entry price for approximation or fetch from history
            current_val = pos['qty'] * pos['entry_price'] 
            # ideally use current price but entry is fine for a rough %
            
            pct_equity = (current_val / latest_equity) * 100
            sizing_label = "Current Pos"
            sizing_val = f"{pct_equity:.1f}% ({pos['side']})"
            sizing_color = "var(--accent)"
            
        else:
            # No position, show recommendation
            base_alloc = 0.0
            if conf_pct > 60:
                base_alloc = 2.0 + ((conf_pct - 60) / 40.0) * 8.0  # Scales from 2% to 10%
            elif conf_pct > 50:
                base_alloc = 1.0  # Watch position
            
            sizing_val = f"{base_alloc:.1f}%"
            if base_alloc > 0:
                sizing_color = "var(--positive)"

        cards_html += f'''
      <div class="sig-card glass" style="border-top: 2px solid var(--{color_var})">
        <div class="sig-header">
          <span class="sig-symbol" style="font-size: 18px;">{sym}</span>
          <span class="sig-dir" style="font-weight: 800; font-size: 14px; color: var(--{color_var})">{action}</span>
        </div>
        <div class="conf-bar-wrap"><div class="conf-bar" style="width:{conf_pct:.1f}%; background: var(--{color_var})"></div></div>
        
        <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
            <div style="text-align: left;">
                <div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase;">Confidence</div>
                <div style="font-family: var(--mono); font-weight: 700; color: var(--text);">{conf_pct:.1f}%</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase;">Est. Move</div>
                <div style="font-family: var(--mono); font-weight: 700; color: var(--text-sec);">Dynamic</div>
            </div>
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-bottom: 12px; background: rgba(255,255,255,0.02); padding: 8px; border-radius: 4px;">
            <div style="text-align: left;">
                <div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase;">{sizing_label}</div>
                <div style="font-family: var(--mono); font-weight: 700; color: {sizing_color};">{sizing_val}</div>
            </div>
            <div style="text-align: right;">
                 <div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase;">Action</div>
                 <div style="font-family: var(--mono); font-weight: 700; color: var(--{color_var});">{action}</div>
            </div>
        </div>

        <div class="sig-metrics" style="border-top: 1px solid var(--border-muted); padding-top: 12px; margin-bottom: 16px;">
          <div class="sig-m"><span class="sig-m-label">RSI (14)</span><span class="sig-m-val">{rsi:.1f}</span></div>
          <div class="sig-m"><span class="sig-m-label">5d Ret</span><span class="sig-m-val {ret5_cls}">{ret5_sign}{ret5*100:.2f}%</span></div>
          <div class="sig-m"><span class="sig-m-label">Vol</span><span class="sig-m-val">{s.get("vol20",0):.2f}</span></div>
        </div>

        <details class="trade-history-details" data-symbol="{sym}">
            <summary class="trade-history-summary">View Live Price Action</summary>
            <div id="trade_curve_live_{sym}" style="height:200px; margin-top:16px;"></div>
        </details>
      </div>'''

    if not signals:
        cards_html = '<div style="grid-column:1/-1;text-align:center;padding:40px;color:var(--text-dim)">No live signals available. Run: python main.py signals --ml</div>'

    return f'''
<!-- LIVE PANEL -->
<section id="live-panel" class="section-wrap reveal">
  <div class="container">
    <div class="sec-label">ACTIONABLE INTELLIGENCE</div>
    <h2 class="sec-title">Live Trading Signals</h2>
    <div class="market-bar glass">
      <div class="market-stat">
        <span class="market-label">VIX</span>
        <span class="market-val mono">{vix}</span>
        <span class="mono {vix_cls}" style="font-size:13px">{vix_sign}{vix_chg:+.2f}%</span>
        <span class="live-dot"></span>
      </div>
      <div class="market-stat">
        <span class="market-label">LAST UPDATE</span>
        <span class="market-val mono" style="font-size:13px">{asof_fmt}</span>
      </div>
    </div>
    <div class="signals-grid">
      {cards_html}
    </div>
  </div>
</section>
'''


def _section_live_pnl(data):
    lp = data.get("live_pnl", {})
    timeline = lp.get("equity_timeline", [])
    positions = lp.get("positions", {})
    start_eq = lp.get("start_equity", 100000)
    latest_eq = lp.get("latest_equity", 100000)
    cash = lp.get("cash", 0)
    in_pos = lp.get("in_positions", 0)
    log_date = lp.get("log_date", "N/A")

    pnl = latest_eq - start_eq
    pnl_pct = (pnl / start_eq) * 100 if start_eq else 0
    pnl_cls = "pos" if pnl >= 0 else "neg"
    pnl_sign = "+" if pnl >= 0 else ""

    total_cycles = len(timeline)

    if timeline:
        eq_high = max(t["equity"] for t in timeline)
        eq_low = min(t["equity"] for t in timeline)
        eq_dd = ((eq_low - eq_high) / eq_high) * 100 if eq_high else 0
    else:
        eq_high = latest_eq
        eq_low = latest_eq
        eq_dd = 0

    # Equity composition bar widths
    cash_pct = (cash / latest_eq * 100) if latest_eq else 0
    pos_pct = (in_pos / latest_eq * 100) if latest_eq else 0

    # Build position table rows
    total_unrealized = 0
    pos_rows = ""
    for sym, p in sorted(positions.items()):
        p_cls = "pos" if p["pnl_pct"] >= 0 else "neg"
        p_sign = "+" if p["pnl_pct"] >= 0 else ""
        pos_val = p["qty"] * p["entry_price"]
        unrealized = pos_val * (p["pnl_pct"] / 100)
        u_sign = "+" if unrealized >= 0 else ""
        total_unrealized += unrealized
        alloc_pct = (pos_val / latest_eq * 100) if latest_eq else 0
        side_cls = "pos" if p["side"] == "LONG" else "neg"

        pos_rows += f'''
              <tr>
                <td style="font-weight:700;color:var(--text)">{sym}</td>
                <td class="{side_cls}">{p["side"]}</td>
                <td>{p["qty"]:.0f}</td>
                <td>${p["entry_price"]:.2f}</td>
                <td>${pos_val:,.0f}</td>
                <td>{alloc_pct:.1f}%</td>
                <td class="{p_cls}">{p_sign}{p["pnl_pct"]:.2f}%</td>
                <td class="{p_cls}">{u_sign}${abs(unrealized):,.0f}</td>
              </tr>'''

    tu_cls = "pos" if total_unrealized >= 0 else "neg"
    tu_sign = "+" if total_unrealized >= 0 else ""

    if not positions:
        pos_table = '<div style="text-align:center;padding:24px;color:var(--text-dim);font-size:12px">No open positions</div>'
    else:
        pos_table = f'''
    <div style="overflow-x:auto">
      <table class="metrics-table">
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Side</th>
            <th>Qty</th>
            <th>Entry</th>
            <th>Mkt Value</th>
            <th>Alloc %</th>
            <th>P&amp;L %</th>
            <th>Unrealized</th>
          </tr>
        </thead>
        <tbody>
          {pos_rows}
          <tr style="border-top:1px solid var(--border);font-weight:700">
            <td colspan="4" style="color:var(--text)">Total</td>
            <td style="color:var(--text)">${in_pos:,.0f}</td>
            <td style="color:var(--text)">{pos_pct:.1f}%</td>
            <td></td>
            <td class="{tu_cls}">{tu_sign}${abs(total_unrealized):,.0f}</td>
          </tr>
        </tbody>
      </table>
    </div>'''

    return f'''
<!-- LIVE P&L -->
<section id="live-pnl" class="section-wrap reveal">
  <div class="container">
    <div class="sec-label">PAPER TRADING</div>
    <h2 class="sec-title" style="margin-bottom:24px">Live Performance</h2>

    <!-- Full Equity Breakdown -->
    <div class="glass" style="padding:28px;margin-bottom:24px">
      <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:32px;align-items:center;margin-bottom:24px">
        <div>
          <div class="live-pos-label">Total Equity</div>
          <div style="font-family:var(--mono);font-size:2.4rem;font-weight:700;color:var(--text);line-height:1">${latest_eq:,.2f}</div>
        </div>
        <div style="width:1px;height:48px;background:var(--border-muted)"></div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:24px">
          <div>
            <div class="live-pos-label">Cash</div>
            <div style="font-family:var(--mono);font-size:1.1rem;font-weight:600;color:var(--text)">${cash:,.2f}</div>
            <div style="font-size:11px;color:var(--text-dim)">{cash_pct:.1f}%</div>
          </div>
          <div>
            <div class="live-pos-label">In Positions</div>
            <div style="font-family:var(--mono);font-size:1.1rem;font-weight:600;color:var(--text)">${in_pos:,.2f}</div>
            <div style="font-size:11px;color:var(--text-dim)">{pos_pct:.1f}%</div>
          </div>
          <div>
            <div class="live-pos-label">Session P&amp;L</div>
            <div class="{pnl_cls}" style="font-family:var(--mono);font-size:1.1rem;font-weight:700">{pnl_sign}${abs(pnl):,.2f}</div>
            <div class="{pnl_cls}" style="font-size:11px;font-weight:600">{pnl_sign}{pnl_pct:.3f}%</div>
          </div>
        </div>
      </div>

      <!-- Equity Composition Bar -->
      <div style="margin-bottom:20px">
        <div style="display:flex;height:8px;border-radius:4px;overflow:hidden;background:rgba(255,255,255,0.03)">
          <div style="width:{cash_pct}%;background:var(--accent);border-radius:4px 0 0 4px" title="Cash {cash_pct:.1f}%"></div>
          <div style="width:{pos_pct}%;background:var(--positive)" title="Positions {pos_pct:.1f}%"></div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:6px;font-size:10px;color:var(--text-dim)">
          <span><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:var(--accent);margin-right:4px;vertical-align:middle"></span>Cash ({cash_pct:.1f}%)</span>
          <span><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:var(--positive);margin-right:4px;vertical-align:middle"></span>Positions ({pos_pct:.1f}%)</span>
          <span>Starting: ${start_eq:,.0f}</span>
          <span>High: ${eq_high:,.2f}</span>
          <span>Low: ${eq_low:,.2f}</span>
          <span>DD: <span class="neg">{eq_dd:.3f}%</span></span>
        </div>
      </div>

      <!-- Equity Chart -->
      <div id="live-equity-chart" style="height:160px"></div>
    </div>

    <!-- Current Positions Table -->
    <div class="glass" style="padding:0;overflow:hidden;margin-bottom:16px">
      <div style="padding:20px 24px 12px;display:flex;justify-content:space-between;align-items:center">
        <div style="font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--text-muted)">Current Positions</div>
        <div style="font-size:11px;color:var(--text-dim)">{len(positions)} open</div>
      </div>
      {pos_table}
    </div>

    <div style="font-size:10px;color:var(--text-dim);text-align:center">
      {total_cycles} cycles &bull; {log_date} &bull; Alpaca Paper Trading
    </div>
  </div>
</section>
'''


def _section_metrics_kpi(data):
    ps = data["portfolio_summary"]
    avg_ret = ps.get("avg_return", 0)
    avg_sharpe = ps.get("avg_sharpe", 0)
    worst_dd = ps.get("worst_drawdown", 0)
    win_rate = ps.get("overall_win_rate", 0)
    total_trades = ps.get("total_trades", 0)
    pf = ps.get("profit_factor", 0)
    total_pnl = ps.get("total_pnl", 0)

    ret_cls = "pos" if avg_ret >= 0 else "neg"
    ret_sign = "+" if avg_ret >= 0 else ""
    pnl_cls = "pos" if total_pnl >= 0 else "neg"
    pnl_sign = "+" if total_pnl >= 0 else ""

    # Build per-symbol comparison rows
    sym_rows = ""
    for symbol, sym_data in sorted(data["symbols"].items()):
        m = sym_data["metrics"]
        if not m.get("has_backtest"):
            continue
        ts = sym_data["trades"]["summary"]
        s_ret = m.get("total_return", 0)
        s_sharpe = m.get("sharpe_ratio", 0)
        s_dd = m.get("max_drawdown", 0)
        s_trades = ts.get("total_trades", 0)
        s_wr = ts.get("win_rate", 0)
        s_pnl = ts.get("total_pnl", 0)
        s_avg = ts.get("avg_trade", 0)

        s_ret_cls = "pos" if s_ret >= 0 else "neg"
        s_ret_sign = "+" if s_ret >= 0 else ""
        s_pnl_cls = "pos" if s_pnl >= 0 else "neg"
        s_pnl_sign = "+" if s_pnl >= 0 else ""
        s_avg_cls = "pos" if s_avg >= 0 else "neg"
        s_avg_sign = "+" if s_avg >= 0 else ""

        sym_rows += f'''
              <tr>
                <td style="font-weight:700;color:var(--text)">{symbol}</td>
                <td class="{s_ret_cls}">{s_ret_sign}{s_ret:.2f}%</td>
                <td>{s_sharpe:.2f}</td>
                <td class="neg">{s_dd:.2f}%</td>
                <td>{s_trades}</td>
                <td>{s_wr:.0f}%</td>
                <td class="{s_pnl_cls}">{s_pnl_sign}${abs(s_pnl):,.0f}</td>
                <td class="{s_avg_cls}">{s_avg_sign}${abs(s_avg):,.0f}</td>
              </tr>'''

    return f'''
<!-- METRICS KPI -->
<section id="metrics" class="section-wrap reveal">
  <div class="container">
    <div class="sec-label">DETAILED ANALYTICS</div>
    <h2 class="sec-title">Portfolio Metrics</h2>
    <div class="kpi-grid">
      <div class="kpi glass">
        <div class="kpi-label">Avg Return</div>
        <div class="kpi-val {ret_cls}">{ret_sign}{avg_ret}%</div>
        <div class="kpi-sub">{ps.get("total_return_range", "N/A")}</div>
      </div>
      <div class="kpi glass">
        <div class="kpi-label">Sharpe Ratio</div>
        <div class="kpi-val">{avg_sharpe}</div>
        <div class="kpi-sub">Risk-Adjusted</div>
      </div>
      <div class="kpi glass">
        <div class="kpi-label">Max Drawdown</div>
        <div class="kpi-val neg">{worst_dd}%</div>
        <div class="kpi-sub">Worst Symbol Peak-to-Trough</div>
      </div>
      <div class="kpi glass">
        <div class="kpi-label">Win Rate</div>
        <div class="kpi-val">{win_rate}%</div>
        <div class="kpi-sub">{total_trades} Total Trades</div>
      </div>
      <div class="kpi glass">
        <div class="kpi-label">Total P&amp;L</div>
        <div class="kpi-val {pnl_cls}">{pnl_sign}${abs(total_pnl):,.0f}</div>
        <div class="kpi-sub">Across All Symbols</div>
      </div>
      <div class="kpi glass">
        <div class="kpi-label">Profit Factor</div>
        <div class="kpi-val">{pf}</div>
        <div class="kpi-sub">Gross Win / Gross Loss</div>
      </div>
    </div>

    <div class="metrics-table-wrap glass" style="margin-top:32px;">
      <div style="padding:20px 24px 12px;font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--text-muted)">Per-Symbol Breakdown</div>
      <div style="overflow-x:auto">
        <table class="metrics-table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Return</th>
              <th>Sharpe</th>
              <th>Max DD</th>
              <th>Trades</th>
              <th>Win Rate</th>
              <th>Total P&amp;L</th>
              <th>Avg Trade</th>
            </tr>
          </thead>
          <tbody>
            {sym_rows}
          </tbody>
        </table>
      </div>
    </div>

    <p class="kpi-disclaimer">All metrics derived from historical backtests (Jan 2024 &ndash; present). Past performance does not guarantee future results.</p>
  </div>
</section>
'''


def _section_equity_curve(data):
    return '''
<!-- EQUITY CURVE -->
<section id="equity" class="section-wrap reveal">
  <div class="container">
    <div class="sec-label">PERFORMANCE</div>
    <h2 class="sec-title">Normalized Equity Curves</h2>
    <div class="chart-wrap glass">
      <div id="equity-chart" style="height:500px"></div>
    </div>
  </div>
</section>
'''


def _section_strategy_cards(data):
    cards = ""
    for symbol, sym_data in sorted(data["symbols"].items()):
        m = sym_data["metrics"]
        t = sym_data["trades"]
        ts = t["summary"]
        trades_list = t["trades"]

        if not m.get("has_backtest"):
            continue

        # Direction badge
        directions = set(tr.get("direction", "") for tr in trades_list)
        if "LONG" in directions and "SHORT" in directions:
            badge_cls, badge_txt = "both", "L+S"
        elif "SHORT" in directions:
            badge_cls, badge_txt = "short", "SHORT"
        else:
            badge_cls, badge_txt = "long", "LONG"

        ret_cls = "pos" if m["total_return"] >= 0 else "neg"
        ret_sign = "+" if m["total_return"] >= 0 else ""

        # Recent trades (last 3)
        recent = trades_list[-3:] if len(trades_list) > 3 else trades_list
        recent_html = ""
        for tr in reversed(recent):
            pnl = tr.get("pnl", 0)
            pnl_cls = "pos" if pnl >= 0 else "neg"
            pnl_sign = "+" if pnl >= 0 else ""
            d = tr.get("direction", "?")
            d_cls = "pos" if d == "LONG" else "neg"
            exit_dt = str(tr.get("exit_date", ""))[:10]
            recent_html += f'''
            <div class="strat-trade">
              <span class="strat-trade-dir {d_cls}">{d[:1]}</span>
              <span class="strat-trade-date">{exit_dt}</span>
              <span class="strat-trade-pnl {pnl_cls}">{pnl_sign}${abs(pnl):,.0f}</span>
            </div>'''
            
        # Full trade history table
        history_rows = ""
        for tr in reversed(trades_list):
            pnl = tr.get("pnl", 0)
            pnl_cls = "pos" if pnl >= 0 else "neg"
            pnl_sign = "+" if pnl >= 0 else ""
            d = tr.get("direction", "?")
            d_cls = "pos" if d == "LONG" else "neg"
            entry_dt = str(tr.get("entry_date", ""))[:10]
            entry_px = tr.get("entry_price", 0)
            exit_dt = str(tr.get("exit_date", ""))[:10]
            exit_px = tr.get("exit_price", 0)
            
            history_rows += f'''
            <tr>
                <td class="{d_cls}">{d[:1]}</td>
                <td>{entry_dt}</td>
                <td>{entry_px:.2f}</td>
                <td>{exit_dt}</td>
                <td>{exit_px:.2f}</td>
                <td class="{pnl_cls}">{pnl_sign}{abs(pnl):.0f}</td>
            </tr>'''

        trades_table_html = f'''
        <details class="trade-history-details" data-symbol="{symbol}">
            <summary class="trade-history-summary">View Full Trade History</summary>
            <div id="trade_curve_{symbol}" style="height:250px; margin-top:16px; margin-bottom:16px;"></div>
            <div class="trade-table-wrap">
                <table class="trade-table">
                    <thead>
                        <tr>
                            <th>D</th>
                            <th>Entry</th>
                            <th>Px</th>
                            <th>Exit</th>
                            <th>Px</th>
                            <th>PnL</th>
                        </tr>
                    </thead>
                    <tbody>
                        {history_rows}
                    </tbody>
                </table>
            </div>
        </details>''' if history_rows else ""

        cards += f'''
      <div class="strat glass">
        <div class="strat-head">
          <span class="strat-sym">{symbol}</span>
          <span class="strat-badge {badge_cls}">{badge_txt}</span>
        </div>
        <div class="strat-spark" id="spark_{symbol}"></div>
        <div class="strat-stats">
          <div class="strat-s"><span class="strat-s-label">Return</span><span class="strat-s-val {ret_cls}">{ret_sign}{m["total_return"]}%</span></div>
          <div class="strat-s"><span class="strat-s-label">Sharpe</span><span class="strat-s-val">{m["sharpe_ratio"]}</span></div>
          <div class="strat-s"><span class="strat-s-label">Win Rate</span><span class="strat-s-val">{ts["win_rate"]:.0f}%</span></div>
          <div class="strat-s"><span class="strat-s-label">Trades</span><span class="strat-s-val">{ts["total_trades"]}</span></div>
        </div>
        <div class="strat-recent-label">RECENT TRADES</div>
        {recent_html if recent_html else '<div style="font-size:12px;color:var(--text-dim)">No trades</div>'}
        {trades_table_html}
      </div>'''

    return f'''
<!-- STRATEGY CARDS -->
<section id="strategies" class="section-wrap reveal">
  <div class="container">
    <div class="sec-label">STRATEGIES</div>
    <h2 class="sec-title">Per-Symbol Analysis</h2>
    <div class="strat-grid">
      {cards}
    </div>
  </div>
</section>
'''


def _section_research():
    return '''
<!-- RESEARCH -->
<section id="research" class="section-wrap reveal">
  <div class="container">
    <div class="sec-label">RESEARCH</div>
    <h2 class="sec-title">ML Architecture</h2>
    <div class="research-grid">

      <div class="glass" style="padding:32px;display:flex;flex-direction:column;align-items:center">
        <div class="lstm-diagram">
          <div class="lstm-layer input-layer">
            <span class="layer-label">Input</span>
            <span class="layer-detail mono">20 &times; 13 features</span>
          </div>
          <div class="lstm-arrow"></div>
          <div class="lstm-layer">
            <span class="layer-label">LSTM Layer 1</span>
            <span class="layer-detail mono">hidden = 64</span>
          </div>
          <div class="lstm-arrow"></div>
          <div class="lstm-layer">
            <span class="layer-label">LSTM Layer 2</span>
            <span class="layer-detail mono">hidden = 64, dropout 0.2</span>
          </div>
          <div class="lstm-arrow"></div>
          <div class="lstm-layer">
            <span class="layer-label">Fully Connected</span>
            <span class="layer-detail mono">64 &rarr; 32 &rarr; ReLU</span>
          </div>
          <div class="lstm-arrow"></div>
          <div class="lstm-layer output-layer">
            <span class="layer-label">Output</span>
            <span class="layer-detail mono">Sigmoid &rarr; UP / DOWN</span>
          </div>
        </div>
      </div>

      <div class="research-details">
        <div class="research-card glass">
          <h4>13 Input Features</h4>
          <div class="features-grid">
            <span class="feat-tag">RSI-14</span>
            <span class="feat-tag">5d Return</span>
            <span class="feat-tag">10d Return</span>
            <span class="feat-tag">Weekly Return</span>
            <span class="feat-tag">Monthly Return</span>
            <span class="feat-tag">20d Volatility</span>
            <span class="feat-tag">Log Dollar Vol</span>
            <span class="feat-tag">VIX Level</span>
            <span class="feat-tag">VIX Change</span>
            <span class="feat-tag">Vol Imbalance</span>
            <span class="feat-tag">VWAP Ratio</span>
            <span class="feat-tag">DV Acceleration</span>
            <span class="feat-tag">Spread Proxy</span>
          </div>
        </div>
        <div class="research-card glass">
          <h4>Training Methodology</h4>
          <ul class="research-list">
            <li>Binary cross-entropy loss, Adam optimizer (lr=1e-3)</li>
            <li>Early stopping with patience = 7 epochs</li>
            <li>Temporal 80/20 train-validation split (no lookahead bias)</li>
            <li>~45,000 parameters per model (~227 KB)</li>
            <li>Separate model trained per symbol</li>
          </ul>
        </div>
        <div class="research-card glass">
          <h4>Prediction Pipeline</h4>
          <ul class="research-list">
            <li>Confidence = |sigmoid(output) &minus; 0.5| &times; 2</li>
            <li>Entry threshold: confidence &gt; 0.20</li>
            <li>Supports daily and 5-minute intraday modes</li>
            <li>Z-score normalization via per-symbol scaler</li>
          </ul>
        </div>
      </div>

    </div>
  </div>
</section>
'''


def _section_pipeline():
    # Simple inline SVG icons
    icon_db = '<svg width="24" height="24" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v6c0 1.66 3.58 3 8 3s8-1.34 8-3V6"/><path d="M4 12v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6"/></svg>'
    icon_gear = '<svg width="24" height="24" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9c.38.17.63.54.68.94V10a2 2 0 0 1 0 4v.09c-.05.4-.3.77-.68.94Z"/></svg>'
    icon_brain = '<svg width="24" height="24" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M12 2a7 7 0 0 0-7 7c0 3 2 5.5 4 7l1 1.5V20h4v-2.5L15 16c2-1.5 4-4 4-7a7 7 0 0 0-7-7Z"/><path d="M9 20h6M10 23h4"/></svg>'
    icon_signal = '<svg width="24" height="24" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M2 20h.01M7 20v-4M12 20v-8M17 20V8M22 20V4"/></svg>'
    icon_play = '<svg width="24" height="24" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><polygon points="5,3 19,12 5,21"/></svg>'

    stages = [
        (icon_db, "Data Sources", "Yahoo Finance, Alpaca Markets, FRED VIX"),
        (icon_gear, "Feature Engineering", "13 technical indicators per symbol"),
        (icon_brain, "ML Prediction", "LSTM neural network, SEQ_LEN=20"),
        (icon_signal, "Signal Generation", "Direction + confidence score"),
        (icon_play, "Execution", "Alpaca paper trading, trailing stops"),
    ]

    html = ""
    for i, (icon, title, desc) in enumerate(stages):
        if i > 0:
            html += '<div class="pipe-conn"><div class="pipe-conn-line"></div></div>'
        html += f'''
      <div class="pipe-stage glass">
        <div class="pipe-icon">{icon}</div>
        <h4>{title}</h4>
        <p>{desc}</p>
      </div>'''

    return f'''
<!-- PIPELINE -->
<section id="pipeline" class="section-wrap reveal">
  <div class="container">
    <div class="sec-label">INFRASTRUCTURE</div>
    <h2 class="sec-title">Data Pipeline</h2>
    <div class="pipeline-flow">
      {html}
    </div>
  </div>
</section>
'''


def _section_logs(data):
    entries = data.get("log_entries", [])
    if not entries:
        rows = '<div class="logs-empty">No paper trading activity recorded yet. Run: python main.py trade</div>'
    else:
        rows = ""
        for e in entries:
            equity_str = f"${e['equity']}" if e.get("equity") else ""
            syms = " | ".join(e.get("symbols", [])[:5])
            rows += f'''
        <div class="log-entry">
          <span class="log-time">{e["time"]}</span>
          <span class="log-cycle">#{e["cycle"]}</span>
          <span class="log-equity">{equity_str}</span>
          <span class="log-detail">{syms}</span>
        </div>'''

    return f'''
<!-- LOGS -->
<section id="logs" class="section-wrap reveal">
  <div class="container">
    <div class="sec-label">ACTIVITY</div>
    <h2 class="sec-title">Paper Trading Log</h2>
    <div class="logs-box glass">
      {rows}
    </div>
  </div>
</section>
'''


def _section_faq():
    faqs = [
        (
            "How does the ML model generate predictions?",
            "A 2-layer LSTM neural network processes 20-bar sliding windows of 13 technical "
            "features (RSI, returns, volatility, VIX, volume metrics). The sigmoid output "
            "represents the probability of the next bar closing higher. Confidence is computed as "
            "|probability - 0.5| &times; 2, ranging from 0 (uncertain) to 1 (very confident).",
        ),
        (
            "What risk management is in place?",
            "Each position uses a 5% trailing stop from peak price. Take-profit targets (8%) are "
            "available but currently disabled. Position sizing allocates capital equally across the "
            "symbol universe. The system can immediately flip from LONG to SHORT on strong reversal signals.",
        ),
        (
            "What data sources are used?",
            "Historical daily OHLCV data from Yahoo Finance, real-time intraday bars from Alpaca Markets, "
            "and the CBOE Volatility Index (VIX) via the FRED API. All data is fetched fresh for each "
            "prediction cycle.",
        ),
        (
            "Are these real trading results?",
            "No. All performance metrics shown are derived from <strong>historical backtests</strong> "
            "and <strong>Alpaca paper trading</strong> (simulated execution). No real capital is at risk. "
            "Past performance does not indicate future results.",
        ),
        (
            "How often are signals updated?",
            "Signals can be regenerated on-demand via the CLI. In paper trading mode, the system checks "
            "for new signals every 1-5 minutes during market hours, depending on the configured interval.",
        ),
        (
            "What is the confidence threshold for trading?",
            "A minimum confidence of 0.20 (20%) is required for LONG entries. SHORT entries use a lower "
            "threshold of 0.15 (15%) since short signals are less frequent. Exit signals use an even lower "
            "threshold of 0.10 to allow quicker position exits.",
        ),
        (
            "Can this system trade with real money?",
            "The paper trading module has <code>paper=True</code> hardcoded. Switching to live trading would "
            "require changing this flag and thorough additional validation. This system is intended for research "
            "and educational purposes only.",
        ),
        (
            "What symbols are covered?",
            "The default universe includes SPY, QQQ, IWM, IGV, GLD, SLV, and XLE &mdash; a mix of equity "
            "index ETFs, sector ETFs, and commodity ETFs. Each symbol has its own independently trained model.",
        ),
    ]
    items = ""
    for q, a in faqs:
        items += f'''
      <details class="faq-item glass">
        <summary>{q}</summary>
        <div class="faq-answer"><p>{a}</p></div>
      </details>'''

    return f'''
<!-- FAQ -->
<section id="faq" class="section-wrap reveal">
  <div class="container">
    <div class="sec-label">FAQ</div>
    <h2 class="sec-title">Frequently Asked Questions</h2>
    <div class="faq-list">
      {items}
    </div>
  </div>
</section>
'''


def _section_footer(data):
    ps = data["portfolio_summary"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f'''
<!-- FOOTER -->
<footer>
  <div class="container">
    <div class="footer-content">
      <div class="footer-brand">
        <span class="footer-logo">QS</span>
        <span class="footer-name">Quantitative Stocks</span>
      </div>
      <div class="footer-disclaimer">
        For educational and research purposes only. Not financial advice.
        All performance metrics are derived from backtests and paper trading.
        Past performance does not indicate future results.
      </div>
      <div class="footer-meta">
        <span class="mono">{now}</span>
        <span>{ps.get("total_symbols", 0)} symbols</span>
        <span>{ps.get("total_trades", 0)} backtest trades</span>
      </div>
    </div>
  </div>
</footer>
'''


def parse_live_equity_history(project_root):
    """Parse equity timeline + open positions from the latest paper trading log."""
    logs_dir = os.path.join(project_root, "logs")
    if not os.path.isdir(logs_dir):
        return {"equity_timeline": [], "positions": {}, "start_equity": 100000,
                "latest_equity": 100000, "cash": 0, "in_positions": 0}

    log_files = sorted(glob.glob(os.path.join(logs_dir, "paper_trader_*.log")))
    if not log_files:
        return {"equity_timeline": [], "positions": {}, "start_equity": 100000,
                "latest_equity": 100000, "cash": 0, "in_positions": 0}

    latest_log = log_files[-1]
    with open(latest_log, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Parse starting equity from first "Account equity: $..." line
    start_match = re.search(r"Account equity: \$([\d,.]+)", content)
    start_equity = float(start_match.group(1).replace(",", "")) if start_match else 100000

    # Parse all cycle equity snapshots: time + equity + cash + positions value
    cycle_pat = re.compile(
        r"\[(\d{2}:\d{2}:\d{2})\] === Paper Trading Cycle #(\d+)")
    acct_pat = re.compile(
        r"Account: \$([\d,.]+) equity \| \$([\d,.]+) cash \| \$([\d,.]+) in positions")

    timeline = []
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        cm = cycle_pat.search(lines[i])
        if cm:
            time_str = cm.group(1)
            cycle_num = int(cm.group(2))
            # Look at next few lines for account data
            for j in range(i + 1, min(i + 4, len(lines))):
                am = acct_pat.search(lines[j])
                if am:
                    eq = float(am.group(1).replace(",", ""))
                    cash = float(am.group(2).replace(",", ""))
                    pos_val = float(am.group(3).replace(",", ""))
                    timeline.append({
                        "time": time_str, "cycle": cycle_num,
                        "equity": eq, "cash": cash, "positions_val": pos_val
                    })
                    break
        i += 1

    # Parse open positions (latest state)
    positions = {}
    pos_pat = re.compile(
        r"^\s*([A-Z]+):\s+(?:HOLD|BUY|SELL)\s+\((LONG|SHORT)\s+([\d.]+)\s+sh\s+@\s+\$([\d.]+),\s+P&L:\s+([+-]?[\d.]+)%\)",
        re.MULTILINE)
    for m in pos_pat.finditer(content):
        positions[m.group(1)] = {
            "symbol": m.group(1), "side": m.group(2),
            "qty": float(m.group(3)), "entry_price": float(m.group(4)),
            "pnl_pct": float(m.group(5))
        }

    latest_equity = timeline[-1]["equity"] if timeline else start_equity
    cash = timeline[-1]["cash"] if timeline else 0
    in_positions = timeline[-1]["positions_val"] if timeline else 0

    return {
        "equity_timeline": timeline,
        "positions": positions,
        "start_equity": start_equity,
        "latest_equity": latest_equity,
        "cash": cash,
        "in_positions": in_positions,
        "log_date": datetime.fromtimestamp(
            os.path.getmtime(latest_log)).strftime("%Y-%m-%d"),
    }


def parse_open_positions_from_logs(project_root):
    """Parse open positions from the latest log file."""
    logs_dir = os.path.join(project_root, "logs")
    if not os.path.isdir(logs_dir):
        return {}
        
    log_files = sorted(glob.glob(os.path.join(logs_dir, "paper_trader_*.log")))
    if not log_files:
        return {}
        
    latest_log = log_files[-1]
    positions = {}
    
    # Regex to capture: SYMBOL: HOLD (SIDE QTY sh @ PRICE, P&L: VAL%)
    # e.g. SPY:  HOLD  (LONG 10 sh @ $692.27, P&L: +0.33%)
    pos_pat = re.compile(r"^\s*([A-Z]+):\s+(?:HOLD|BUY|SELL)\s+\((LONG|SHORT)\s+([\d.]+)\s+sh\s+@\s+\$([\d.]+),\s+P&L:\s+([+-]?[\d.]+)%\)")
    
    with open(latest_log, "r", encoding="utf-8", errors="replace") as f:
        # Read from end to find latest state
        lines = f.readlines()
        
    # Scan backwards to find the last complete cycle
    # Or just scan the whole file and keep updating the dict? 
    # Scanning whole file ensures we get the *latest* status for each symbol.
    for line in lines:
        m = pos_pat.search(line)
        if m:
            symbol = m.group(1)
            side = m.group(2)
            qty = float(m.group(3))
            entry_price = float(m.group(4))
            pnl_pct = float(m.group(5))
            
            positions[symbol] = {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "entry_price": entry_price,
                "pnl_pct": pnl_pct,
                # Use file modification time or today's date if not strict
                "date": datetime.fromtimestamp(os.path.getmtime(latest_log)).strftime("%Y-%m-%d")
            }
            
    return positions


def _embedded_js(data):
    # Prepare chart data as a single JSON blob
    chart_data = {}
    
    # Get current signals to overlay
    current_signals = data.get("signals", {}).get("signals", {})
    
    # Get open positions from logs
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    open_positions = parse_open_positions_from_logs(project_root)
    
    for symbol, sym_data in data["symbols"].items():
        ec = sym_data["equity_curve"]
        # Include chart data even if no equity curve, as long as we have prices
        if not ec and not sym_data.get("prices"):
            continue

        sig_data = None
        if symbol in current_signals:
            curr = current_signals[symbol]
            conf_pct = curr.get("ml_confidence", 0)
            if conf_pct <= 1: conf_pct *= 100
            
            # Simple heuristic action mapping
            act = "HOLD"
            if curr.get("ml_direction") == "UP" and conf_pct > 60: act = "BUY"
            elif curr.get("ml_direction") == "DOWN" and conf_pct > 60: act = "SELL"
            
            if act in ["BUY", "SELL"]:
                sig_data = {
                    "action": act,
                    "confidence": conf_pct,
                    "direction": curr.get("ml_direction"),
                    # Use last known price from history or 0 if missing
                    "price": sym_data["prices"][-1]["close"] if sym_data.get("prices") else 0,
                    "date": sym_data["prices"][-1]["date"] if sym_data.get("prices") else "Now"
                }

        chart_data[symbol] = {
            "dates": [str(p["date"])[:10] for p in ec] if ec else [],
            "equity": [p["equity"] for p in ec] if ec else [],
            "prices": sym_data.get("prices", []),
            "current_signal": sig_data,
            "open_position": open_positions.get(symbol),
            "trades": [
                {
                    "direction": t.get("direction", ""),
                    "entry_date": str(t.get("entry_date", ""))[:10],
                    "exit_date": str(t.get("exit_date", ""))[:10],
                    "entry_price": t.get("entry_price", 0),
                    "exit_price": t.get("exit_price", 0),
                    "pnl": t.get("pnl", 0),
                    "return_pct": t.get("return_pct", 0),
                }
                for t in sym_data["trades"]["trades"]
            ]
        }

    # Add live equity timeline for the Live P&L chart
    lp = data.get("live_pnl", {})
    timeline = lp.get("equity_timeline", [])
    chart_data["_live_equity"] = {
        "times": [t["time"] for t in timeline],
        "equity": [t["equity"] for t in timeline],
        "start": lp.get("start_equity", 100000),
    }

    json_blob = json.dumps(chart_data, default=str)

    # Color palette for chart traces
    colors = [
        "#3b82f6",
        "#22c55e",
        "#f59e0b",
        "#ef4444",
        "#8b5cf6",
        "#ec4899",
        "#06b6d4",
        "#f97316",
    ]

    return f'''
<script type="application/json" id="chartData">{json_blob}</script>
<script>
document.addEventListener('DOMContentLoaded', function() {{
  // ---- Reveal on scroll ----
  var obs = new IntersectionObserver(function(entries) {{
    entries.forEach(function(e) {{ if(e.isIntersecting) e.target.classList.add('visible'); }});
  }}, {{ threshold: 0.08, rootMargin: '0px 0px -30px 0px' }});
  document.querySelectorAll('.reveal').forEach(function(el) {{ obs.observe(el); }});

  // ---- Parse chart data ----
  var raw = document.getElementById('chartData').textContent;
  var cd = JSON.parse(raw);
  var symbols = Object.keys(cd).sort();
  var colors = {json.dumps(colors)};

  // ---- Main equity chart (Percentage Growth) ----
  var traces = [];
  symbols.forEach(function(sym, i) {{
    var eq = cd[sym].equity;
    if(!eq || eq.length === 0) return;
    
    // Normalize to percentage growth
    var startVal = eq[0];
    var pctGrowth = eq.map(function(val) {{ return ((val - startVal) / startVal) * 100; }});
    
    traces.push({{
      x: cd[sym].dates,
      y: pctGrowth,
      type: 'scatter',
      mode: 'lines',
      name: sym,
      line: {{ width: 2, color: colors[i % colors.length] }},
      hovertemplate: '%{{y:.2f}}%<extra>' + sym + '</extra>'
    }});
  }});

  var layout = {{
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {{ color: '#94a3b8', family: 'Inter, system-ui, sans-serif', size: 12 }},
    xaxis: {{
      gridcolor: 'rgba(255,255,255,0.04)',
      linecolor: 'rgba(255,255,255,0.06)',
      zeroline: false
    }},
    yaxis: {{
      title: {{ text: 'Cumulative Return (%)', font: {{ size: 12, color: '#64748b' }} }},
      gridcolor: 'rgba(255,255,255,0.04)',
      linecolor: 'rgba(255,255,255,0.06)',
      zeroline: true,
      zerolinecolor: 'rgba(255,255,255,0.1)'
    }},
    legend: {{
      orientation: 'h',
      y: -0.15,
      x: 0.5,
      xanchor: 'center',
      font: {{ color: '#94a3b8', size: 11 }}
    }},
    margin: {{ l: 60, r: 20, t: 20, b: 60 }},
    hovermode: 'x unified'
  }};

  var eqEl = document.getElementById('equity-chart');
  if(eqEl) Plotly.newPlot(eqEl, traces, layout, {{ responsive: true, displayModeBar: false }});

  // ---- Live Equity Chart (Paper Trading Session) ----
  var liveEqEl = document.getElementById('live-equity-chart');
  if (liveEqEl && cd._live_equity && cd._live_equity.times.length > 0) {{
    var le = cd._live_equity;
    // Downsample if too many points
    var step = Math.max(1, Math.floor(le.times.length / 300));
    var sampledT = [], sampledV = [];
    var startEq = le.start > 0 ? le.start : 100000;
    
    // Use raw equity values (Dollars)
    for (var si = 0; si < le.times.length; si += step) {{
      sampledT.push(le.times[si]);
      sampledV.push(le.equity[si]);
    }}
    // Ensure last point is included
    if (le.times.length > 0 && sampledT[sampledT.length - 1] !== le.times[le.times.length - 1]) {{
      sampledT.push(le.times[le.times.length - 1]);
      sampledV.push(le.equity[le.equity.length - 1]);
    }}
    
    var finalVal = sampledV[sampledV.length - 1];
    var isPos = finalVal >= startEq;
    var lineCol = isPos ? '#22c55e' : '#ef4444';
    var fillCol = isPos ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)';
    
    // Main Equity Line
    var trace1 = {{
      x: sampledT,
      y: sampledV,
      type: 'scatter',
      mode: 'lines',
      name: 'Equity',
      line: {{ color: lineCol, width: 2, shape:'spline', smoothing: 1.3 }},
      fill: 'tozeroy',
      fillcolor: fillCol,
      hovertemplate: '$%{{y:,.2f}}<extra></extra>'
    }};
    
    // Start Equity Reference Line
    var trace2 = {{
      x: [sampledT[0], sampledT[sampledT.length-1]],
      y: [startEq, startEq],
      type: 'scatter',
      mode: 'lines',
      name: 'Start',
      line: {{ color: 'rgba(255,255,255,0.3)', width: 1, dash: 'dash' }},
      hoverinfo: 'none'
    }};

    var layoutLive = {{
      margin: {{ l: 40, r: 10, t: 10, b: 20 }},
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      showlegend: false,
      xaxis: {{ 
        showgrid: false, 
        zeroline: false,
        tickfont: {{ size: 9, color: '#64748b' }},
        fixedrange: true
      }},
      yaxis: {{ 
        showgrid: true, 
        gridcolor: 'rgba(255,255,255,0.05)',
        zeroline: false,
        tickfont: {{ size: 9, color: '#64748b' }},
        fixedrange: true
      }},
      hovermode: 'x unified'
    }};
    
    Plotly.newPlot(liveEqEl, [trace1, trace2], layoutLive, {{ 
      responsive: true, 
      displayModeBar: false,
      staticPlot: false 
    }});
  }}

  // ---- Sparklines for strategy cards ----
  symbols.forEach(function(sym, i) {{
    var el = document.getElementById('spark_' + sym);
    if(!el || !cd[sym]) return;
    var eq = cd[sym].equity;
    // Normalize to % change from start
    var start = eq[0] || 1;
    var pct = eq.map(function(v) {{ return (v / start - 1) * 100; }});
    var lineColor = pct[pct.length - 1] >= 0 ? '#22c55e' : '#ef4444';
    Plotly.newPlot(el, [{{
      y: pct,
      type: 'scatter',
      mode: 'lines',
      line: {{ width: 1.5, color: lineColor }},
      fill: 'tozeroy',
      fillcolor: pct[pct.length - 1] >= 0 ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)',
      hoverinfo: 'skip'
    }}], {{
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      xaxis: {{ visible: false }},
      yaxis: {{ visible: false }},
      margin: {{ l: 0, r: 0, t: 0, b: 0 }},
      height: 50
    }}, {{ responsive: true, displayModeBar: false, staticPlot: true }});
  }});

  // ---- Smooth scroll for nav links ----
  document.querySelectorAll('.topnav-links a').forEach(function(a) {{
    a.addEventListener('click', function(e) {{
      e.preventDefault();
      var target = document.querySelector(this.getAttribute('href'));
      if(target) target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }});
  }});

  // ---- Lazy Load Trade History Curves ----
  document.querySelectorAll('details.trade-history-details').forEach(function(det) {{
    det.addEventListener('toggle', function() {{
      if (this.open) {{
        var sym = this.getAttribute('data-symbol');
        var internalChartDiv = this.querySelector('[id^="trade_curve_"]');
        var chartEl = internalChartDiv;

        if (!chartEl) {{
             var chartId = 'trade_curve_' + sym;
             chartEl = document.getElementById(chartId);
        }}

        // Render only once
        if (chartEl && !chartEl.dataset.rendered && cd[sym]) {{
           // Detect if this is a LIVE panel card (id starts with trade_curve_live_)
           var isLive = chartEl.id.indexOf('trade_curve_live_') === 0;
           // Live cards: no backtest trades, only recent price + open position + signal
           var trades = isLive ? [] : (cd[sym].trades || []);
           renderTradeCurve(sym, trades, chartEl, isLive);
           chartEl.dataset.rendered = "true";
        }}
      }}
    }});
  }});

  function renderTradeCurve(sym, trades, container, isLive) {{
    // Get price history
    var prices = (cd[sym] && cd[sym].prices) ? cd[sym].prices : [];
    var currentSig = (cd[sym] && cd[sym].current_signal) ? cd[sym].current_signal : null;
    var openPos = (cd[sym] && cd[sym].open_position) ? cd[sym].open_position : null;

    // Live cards: limit to last 30 trading days for a focused view
    if (isLive && prices.length > 30) {{
        prices = prices.slice(prices.length - 30);
    }}
    
    var traces = [];
    
    // 1. Price Line Trace
    if (prices.length > 0) {{
        traces.push({{
            x: prices.map(function(p) {{ return p.date; }}),
            y: prices.map(function(p) {{ return p.close; }}),
            type: 'scatter',
            mode: 'lines',
            name: 'Price',
            line: {{ color: 'rgba(148, 163, 184, 0.5)', width: 1.5 }},
            hoverinfo: 'x+y'
        }});
    }}
    
    // 2. Prepare Buy/Sell Markers from History
    var buyX=[], buyY=[], buyText=[], buyColor=[];
    var sellX=[], sellY=[], sellText=[], sellColor=[];
    
    trades.forEach(function(t) {{
        var d = t.direction; 
        var pnl = t.pnl;
        var win = pnl >= 0;
        
        if (d === 'LONG') {{
            buyX.push(t.entry_date); 
            buyY.push(t.entry_price);
            buyText.push('Buy (Long Entry)<br>' + t.entry_date + '<br>$' + t.entry_price);
            buyColor.push('#22c55e'); 
            
            sellX.push(t.exit_date); 
            sellY.push(t.exit_price);
            sellText.push('Sell (Long Exit)<br>PnL: ' + (win?'+':'') + '$' + Math.round(pnl));
            sellColor.push(win ? '#22c55e' : '#ef4444'); 
        }} else {{
            sellX.push(t.entry_date); 
            sellY.push(t.entry_price);
            sellText.push('Sell (Short Entry)<br>' + t.entry_date + '<br>$' + t.entry_price);
            sellColor.push('#ef4444');
            
            buyX.push(t.exit_date); 
            buyY.push(t.exit_price);
            buyText.push('Buy (Short Exit)<br>PnL: ' + (win?'+':'') + '$' + Math.round(pnl));
            buyColor.push(win ? '#22c55e' : '#ef4444');
        }}
    }});
    
    // Add Buy Trace (Green Up Triangles)
    if (buyX.length > 0) {{
        traces.push({{
            x: buyX, y: buyY, text: buyText,
            type: 'scatter', mode: 'markers', name: 'Hist. Buy',
            marker: {{ symbol: 'triangle-up', size: 10, color: '#22c55e', line: {{ width: 1, color: '#fff' }} }},
            hoverinfo: 'text'
        }});
    }}
    
    // Add Sell Trace (Red Down Triangles)
    if (sellX.length > 0) {{
        traces.push({{
            x: sellX, y: sellY, text: sellText,
            type: 'scatter', mode: 'markers', name: 'Hist. Sell',
            marker: {{ symbol: 'triangle-down', size: 10, color: '#ef4444', line: {{ width: 1, color: '#fff' }} }},
            hoverinfo: 'text'
        }});
    }}
    
    // 3. Add Current OPEN Position (if any)
    if (openPos) {{
        var symbol = openPos.side === 'LONG' ? 'triangle-up' : 'triangle-down';
        var color = openPos.side === 'LONG' ? '#22c55e' : '#ef4444';
        var lastDate = prices.length > 0 ? prices[prices.length-1].date : openPos.date;

        traces.push({{
            x: [lastDate],
            y: [openPos.entry_price],
            text: ['OPEN POSITION: ' + openPos.side + '<br>Avg Price: $' + openPos.entry_price.toFixed(2) + '<br>Qty: ' + openPos.qty + '<br>Unrealized P&L: ' + openPos.pnl_pct + '%'],
            type: 'scatter',
            mode: 'markers',
            name: 'Open ' + openPos.side,
            marker: {{
                symbol: symbol,
                size: 14,
                color: color,
                line: {{ width: 2, color: '#fff' }}
            }},
            hoverinfo: 'text'
        }});

        // Add a horizontal line at entry price (only if we have price history)
        if (prices.length > 0) {{
            traces.push({{
                x: [prices[0].date, lastDate],
                y: [openPos.entry_price, openPos.entry_price],
                type: 'scatter',
                mode: 'lines',
                name: 'Entry Level',
                line: {{ color: color, width: 1, dash: 'dash' }},
                hoverinfo: 'skip'
            }});
        }}
    }}
    
    // 4. Add Current Live Signal Marker
    if (currentSig && currentSig.price > 0 && !openPos) {{
        var sigText = 'LIVE SIGNAL: ' + currentSig.action + '<br>Conf: ' + currentSig.confidence.toFixed(1) + '%';
        var sigColor = currentSig.action === 'BUY' ? '#22c55e' : '#ef4444';
        var sigSymbol = 'star'; 
        
        traces.push({{
            x: [currentSig.date],
            y: [currentSig.price],
            text: [sigText],
            type: 'scatter',
            mode: 'markers',
            name: 'Signal',
            marker: {{ 
                symbol: sigSymbol, 
                size: 16, 
                color: sigColor, 
                line: {{ width: 2, color: '#fff' }}
            }},
            hoverinfo: 'text'
        }});
    }}

    var chartTitle = isLive ? 'Live Price Action (30d)' : 'Price Action & Trades';
    var layout = {{
        title: {{ text: chartTitle, font: {{ size: 12, color: '#94a3b8' }}, x: 0.05 }},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {{ color: '#94a3b8', family: 'Inter, system-ui, sans-serif', size: 11 }},
        margin: {{ l: 40, r: 20, t: 30, b: 30 }},
        xaxis: {{ showgrid: false, zeroline: false, color: '#64748b' }},
        yaxis: {{ 
            showgrid: true, 
            gridcolor: 'rgba(255,255,255,0.04)', 
            zeroline: false, 
            color: '#64748b'
        }},
        showlegend: true,
        legend: {{ orientation: 'h', x: 0, y: 1.1, bgcolor: 'rgba(0,0,0,0)' }}
    }};

    Plotly.newPlot(container, traces, layout, {{ responsive: true, displayModeBar: false }});
  }}

}});
</script>
</body>
</html>'''


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------


def create_comprehensive_dashboard(consolidated_data):
    """Create comprehensive AI-native dashboard HTML."""
    parts = [
        _html_head(),
        _section_hero(consolidated_data),
        _section_live_panel(consolidated_data),
        _section_live_pnl(consolidated_data),
        _section_metrics_kpi(consolidated_data),
        _section_equity_curve(consolidated_data),
        _section_strategy_cards(consolidated_data),
        _section_research(),
        _section_pipeline(),
        _section_logs(consolidated_data),
        _section_faq(),
        _section_footer(consolidated_data),
        _embedded_js(consolidated_data),
    ]
    return "\n".join(parts)


if __name__ == "__main__":
    consolidated_data = consolidate_all_data()
    dashboard_html = create_comprehensive_dashboard(consolidated_data)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_path = os.path.join(output_dir, "comprehensive_dashboard.html")

    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(dashboard_html)

    print(f"Dashboard created: {dashboard_path}")
    print("Open this file in your browser to view the dashboard.")
