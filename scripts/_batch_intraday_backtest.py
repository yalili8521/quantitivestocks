#!/usr/bin/env python3
"""Quick intraday OOS backtest for all ETF symbols with trained LGB models."""
import sys, os, json, glob, logging, warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)

from signals_engine import build_adapter
from backtester import Backtester

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("intraday_batch")
log.setLevel(logging.INFO)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "intraday")
configs = sorted(glob.glob(os.path.join(MODEL_DIR, "*_lgb_intraday_etf_config.json")))
symbols = [os.path.basename(c).replace("_lgb_intraday_etf_config.json", "") for c in configs]

# Check train_end from first config
sample_cfg = json.load(open(configs[0]))
train_end = sample_cfg.get("train_end", "2026-01-15")
# Use the day after train_end as OOS start
start = "2026-01-17"
log.info("OOS start: %s (train_end=%s)", start, train_end)

adapter = build_adapter("yahoo")
fred_key = os.environ.get("FRED_API_KEY")
results = []

log.info("Backtesting %d intraday ETF symbols from %s...", len(symbols), start)
for i, sym in enumerate(symbols):
    log.info("[%d/%d] %s...", i + 1, len(symbols), sym)
    try:
        bt = Backtester(
            symbol=sym, adapter=adapter, fred_key=fred_key,
            initial_capital=100000, model_type="etf_intraday",
            model_dir=MODEL_DIR, mode="intraday", intraday_interval="5min",
        )
        r = bt.run(start_date=start)
        if r is None:
            log.warning("  %s: None result", sym)
            continue
        results.append({
            "symbol": sym,
            "sharpe": round(float(r.sharpe_ratio), 3),
            "return_pct": round(float(r.total_return_pct), 2),
            "trades": int(r.total_trades),
            "win_rate": round(float(r.win_rate), 3),
            "max_dd": round(float(r.max_drawdown_pct), 2),
            "pf": round(float(r.profit_factor), 2) if r.profit_factor != float("inf") else 999.0,
        })
        log.info("  %s: SR=%.2f, Ret=%.1f%%, Trades=%d, WR=%.0f%%",
                 sym, r.sharpe_ratio, r.total_return_pct, r.total_trades, r.win_rate * 100)
    except Exception as e:
        log.warning("  %s: FAILED - %s", sym, e)

# Sort and display
results.sort(key=lambda x: x["sharpe"], reverse=True)
print()
print("=" * 105)
print("  INTRADAY ETF OOS BACKTEST RESULTS (sorted by Sharpe)")
print("  OOS period: %s to 2026-04-08 (truly out-of-sample)" % start)
print("=" * 105)
header = "  %-10s %8s %8s %7s %8s %8s %8s  %s" % ("Symbol", "Sharpe", "Return", "Trades", "WinRate", "MaxDD", "PF", "Status")
print(header)
print("  " + "-" * 95)
promoted = 0
for r in results:
    status = "PROMOTED" if r["sharpe"] >= 0.5 and r["trades"] >= 5 else "rejected"
    if status == "PROMOTED":
        promoted += 1
    line = "  %-10s %+8.3f %+7.1f%% %7d %7.0f%% %+7.1f%% %8.2f  %s" % (
        r["symbol"], r["sharpe"], r["return_pct"], r["trades"],
        r["win_rate"] * 100, r["max_dd"], r["pf"], status,
    )
    print(line)
print("  " + "-" * 95)
print("  Total: %d | Promoted: %d | Rejected: %d" % (len(results), promoted, len(results) - promoted))
print("=" * 105)
promoted_syms = [r["symbol"] for r in results if r["sharpe"] >= 0.5 and r["trades"] >= 5]
print("  Promoted: %s" % ", ".join(promoted_syms))

# Save
out = {"symbols": {r["symbol"]: r for r in results}, "oos_start": start, "promoted": promoted_syms}
os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, "promoted_symbols.json"), "w") as f:
    json.dump(out, f, indent=2)
print("  Saved to models/intraday/promoted_symbols.json")
