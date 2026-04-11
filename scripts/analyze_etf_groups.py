"""One-shot script: aggregate OOS backtest results for swing & intraday ETF groups."""
import json, pathlib, statistics

d = pathlib.Path("outputs/backtests")

swing, intra = [], []
for f in sorted(d.glob("backtest_*_summary.json")):
    if "intraday" in f.name:
        continue
    swing.append(json.loads(f.read_text()))

for f in sorted(d.glob("backtest_*_intraday_summary.json")):
    intra.append(json.loads(f.read_text()))


def fmt(rows, title):
    print("=" * 100)
    print(title)
    print("=" * 100)
    hdr = f"{'Symbol':<8} {'Sharpe':>7} {'Ret%':>7} {'Ann%':>7} {'MaxDD%':>7} {'WR%':>6} {'Trades':>7} {'PF':>6} {'AvgW%':>6} {'AvgL%':>6} {'AvgDur':>6}"
    print(hdr)
    print("-" * 100)
    for s in sorted(rows, key=lambda x: x.get("sharpe_ratio", 0), reverse=True):
        sym = s.get("symbol", "?")
        sr = s.get("sharpe_ratio", 0) or 0
        ret = s.get("total_return_pct", 0) or 0
        ann = s.get("annualized_return_pct", 0) or 0
        dd = s.get("max_drawdown_pct", 0) or 0
        wr_raw = s.get("win_rate", 0) or 0
        wr = wr_raw * 100 if wr_raw <= 1 else wr_raw
        nt = s.get("total_trades", 0) or 0
        pf = s.get("profit_factor", 0) or 0
        aw = s.get("avg_win_pct", 0) or 0
        al = s.get("avg_loss_pct", 0) or 0
        dur = s.get("avg_trade_duration_days", 0) or 0
        print(f"{sym:<8} {sr:>7.2f} {ret:>7.1f} {ann:>7.1f} {dd:>7.1f} {wr:>6.1f} {nt:>7} {pf:>6.2f} {aw:>6.2f} {al:>6.2f} {dur:>6.1f}")

    active = [s for s in rows if (s.get("total_trades", 0) or 0) > 0]
    srs = [s.get("sharpe_ratio", 0) or 0 for s in active]
    rets = [s.get("total_return_pct", 0) or 0 for s in active]
    pos = sum(1 for x in srs if x > 0)
    neg = sum(1 for x in srs if x <= 0)
    print(f"\n  Total: {len(rows)} symbols, {len(active)} traded, {pos} profitable (SR>0), {neg} unprofitable")
    if srs:
        print(f"  Median Sharpe: {statistics.median(srs):.2f}, Mean: {statistics.mean(srs):.2f}")
    if rets:
        print(f"  Median Ret%: {statistics.median(rets):.2f}, Mean: {statistics.mean(rets):.2f}")
    # Worst drawdowns
    worst = sorted(active, key=lambda x: x.get("max_drawdown_pct", 0) or 0)[:5]
    parts = [f"{w['symbol']}({(w.get('max_drawdown_pct',0) or 0):.1f}%)" for w in worst]
    print("  Worst drawdowns:", ", ".join(parts))
    # Best PF
    best_pf = sorted(active, key=lambda x: x.get("profit_factor", 0) or 0, reverse=True)[:5]
    parts2 = [f"{b['symbol']}({(b.get('profit_factor',0) or 0):.2f})" for b in best_pf]
    print("  Best PF:", ", ".join(parts2))
    print()


fmt(swing, "SWING ETF GROUP - OOS Backtest Results")
fmt(intra, "INTRADAY ETF GROUP - OOS Backtest Results")

# Cross-group comparison
print("=" * 100)
print("CROSS-GROUP COMPARISON")
print("=" * 100)
for sym_name in sorted(set(s["symbol"] for s in swing)):
    sw = next((s for s in swing if s["symbol"] == sym_name), None)
    it = next((s for s in intra if s["symbol"] == sym_name), None)
    if sw and it:
        sw_sr = sw.get("sharpe_ratio", 0) or 0
        it_sr = it.get("sharpe_ratio", 0) or 0
        sw_ret = sw.get("total_return_pct", 0) or 0
        it_ret = it.get("total_return_pct", 0) or 0
        better = "SWING" if sw_sr > it_sr else "INTRA" if it_sr > sw_sr else "TIE"
        print(f"  {sym_name:<8}  Swing SR={sw_sr:>6.2f} Ret={sw_ret:>6.1f}%  |  Intra SR={it_sr:>6.2f} Ret={it_ret:>6.1f}%  => {better}")
