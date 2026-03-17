"""
Screen crypto coins for 5-minute intraday/HF trading suitability.

Research-informed criteria (stricter than swing screener):
  1. Available on both BinanceUS (training) and Kraken (execution)
  2. Daily avg volume >= $50M (rolling, not snapshot)
  3. Spread proxy < 0.10% for alts, < 0.03% for BTC/ETH
  4. Volume stability: >= 90% of hours have non-zero volume
  5. No "pulse" coins: volume coefficient of variation across hours < 3.0
  6. Good 5-min bar quality (gap < 2%, zero-vol < 2%)

Target: 3-8 core HF symbols (BTC + ETH + 1-3 large-cap alts)

Usage:
    .venv/Scripts/python.exe scripts/screen_intraday_coins.py
"""
import json
import time
import sys
from pathlib import Path
from datetime import datetime, timezone

import ccxt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Thresholds (from research) ────────────────────────────────────────────
MIN_DAILY_DVOL_USD = 50_000_000     # $50M daily minimum
MIN_HOURLY_ACTIVE_PCT = 0.90        # 90%+ hours must have volume
MAX_SPREAD_MAJOR = 0.0003           # 0.03% for BTC/ETH
MAX_SPREAD_ALT = 0.0010             # 0.10% for alts
MAX_GAP_PCT = 0.02                  # 2% max gaps in 5-min bars
MAX_ZERO_VOL_PCT = 0.02             # 2% max zero-volume bars
MAX_VOL_CV = 3.0                    # volume CoV across hours (reject pulse coins)
MAX_CORE_SYMBOLS = 8                # cap core HF symbols at 3-8
MAJOR_COINS = {"BTC", "ETH"}        # tighter spread threshold + priority inclusion

# Kraken fee tiers (maker/taker for volume < $50K/month — lowest tier)
# https://www.kraken.com/features/fee-schedule
KRAKEN_FEE_TAKER = 0.0026           # 0.26% taker fee (default tier)
KRAKEN_FEE_MAKER = 0.0016           # 0.16% maker fee

# ── Load existing config ──────────────────────────────────────────────────
TRADING_CFG = PROJECT_ROOT / "config" / "trading.json"
oos_sharpe = {}
if TRADING_CFG.exists():
    with open(TRADING_CFG) as f:
        cfg = json.load(f)
    oos_sharpe = cfg.get("oos_sharpe_registry", {})

# ── Load exchange markets ─────────────────────────────────────────────────
print("[1/4] Loading exchange markets...")
binanceus = ccxt.binanceus()
binanceus.load_markets()

kraken = ccxt.kraken()
kraken.load_markets()

# ── Find intersection ─────────────────────────────────────────────────────
def extract_base_coins(exchange, quote_currencies=("USD", "USDT")):
    coins = set()
    for sym, mkt in exchange.markets.items():
        if not mkt.get("active", True):
            continue
        if mkt.get("quote") in quote_currencies and mkt.get("spot", True):
            coins.add(mkt["base"])
    return coins

binance_coins = extract_base_coins(binanceus)
kraken_coins = extract_base_coins(kraken)
both = sorted(binance_coins & kraken_coins)

STABLECOINS = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "UST", "USDP", "GUSD",
               "FRAX", "LUSD", "SUSD", "PYUSD", "FDUSD", "EURC", "EUR"}
both = [c for c in both if c not in STABLECOINS]

print(f"  BinanceUS: {len(binance_coins)} coins | Kraken: {len(kraken_coins)} coins")
print(f"  Intersection (excl stablecoins): {len(both)} coins")
print()

# ── Fetch 5-min OHLCV from Kraken (720 bars = ~2.5 days) ─────────────────
print(f"[2/4] Fetching 720 x 5-min bars for {len(both)} coins from Kraken...")
print(f"  (estimated time: {len(both) * 1.5:.0f}s with rate limiting)\n")

results = []
for i, base in enumerate(both):
    pair = None
    for quote in ("USD", "USDT"):
        candidate = f"{base}/{quote}"
        if candidate in kraken.markets:
            pair = candidate
            break
    if pair is None:
        continue

    try:
        ohlcv = kraken.fetch_ohlcv(pair, timeframe="5m", limit=720)
    except Exception as e:
        print(f"  [{i+1}/{len(both)}] {base:>8s}: fetch error: {e}")
        time.sleep(1.5)
        continue

    if not ohlcv or len(ohlcv) < 50:
        time.sleep(1.5)
        continue

    timestamps = np.array([bar[0] for bar in ohlcv], dtype=np.float64)
    highs = np.array([bar[2] for bar in ohlcv], dtype=np.float64)
    lows = np.array([bar[3] for bar in ohlcv], dtype=np.float64)
    closes = np.array([bar[4] for bar in ohlcv], dtype=np.float64)
    volumes = np.array([bar[5] for bar in ohlcv], dtype=np.float64)

    n_bars = len(ohlcv)

    # --- Bar quality ---
    dt = np.diff(timestamps)
    expected_dt = 300_000  # 5 min in ms
    n_gaps = int(np.sum(dt > expected_dt * 1.5))
    gap_pct = n_gaps / max(n_bars - 1, 1)

    zero_vol_pct = float(np.sum(volumes == 0)) / n_bars

    # --- Spread proxy (average per-bar) ---
    spread_proxy = float(np.mean((highs - lows) / np.where(closes > 0, closes, 1e-9)))

    # --- Dollar volume ---
    dollar_vol_per_bar = volumes * closes  # per 5-min bar
    avg_dvol_per_bar = float(np.mean(dollar_vol_per_bar))
    daily_dvol = avg_dvol_per_bar * 288  # 288 five-min bars per day

    # --- Volume stability: per-hour analysis ---
    # Group bars by hour-of-day (UTC), check what fraction of hours have volume
    hours_utc = np.array([
        datetime.fromtimestamp(ts / 1000, tz=timezone.utc).hour
        for ts in timestamps
    ])

    # Per-hour volume sums
    hourly_vol = {}
    for h in range(24):
        mask = hours_utc == h
        if mask.any():
            hourly_vol[h] = float(np.sum(dollar_vol_per_bar[mask]))

    # Hours with meaningful volume (> 1% of average hourly volume)
    if hourly_vol:
        avg_hourly = np.mean(list(hourly_vol.values()))
        active_hours = sum(1 for v in hourly_vol.values() if v > avg_hourly * 0.01)
        hourly_active_pct = active_hours / 24.0

        # Volume coefficient of variation across hours (lower = more uniform)
        hourly_vals = np.array(list(hourly_vol.values()))
        vol_cv = float(np.std(hourly_vals) / max(np.mean(hourly_vals), 1)) if len(hourly_vals) > 1 else 0
    else:
        hourly_active_pct = 0.0
        vol_cv = 999.0

    last_price = float(closes[-1])
    oos_key = f"{base}-USD"
    sharpe = oos_sharpe.get(oos_key, None)

    is_major = base in MAJOR_COINS
    spread_threshold = MAX_SPREAD_MAJOR if is_major else MAX_SPREAD_ALT

    # --- Pass/fail each criterion ---
    pass_dvol = daily_dvol >= MIN_DAILY_DVOL_USD
    pass_spread = spread_proxy <= spread_threshold
    pass_gap = gap_pct <= MAX_GAP_PCT
    pass_zvol = zero_vol_pct <= MAX_ZERO_VOL_PCT
    pass_active = hourly_active_pct >= MIN_HOURLY_ACTIVE_PCT
    pass_cv = vol_cv <= MAX_VOL_CV
    pass_all = pass_dvol and pass_spread and pass_gap and pass_zvol and pass_active and pass_cv
    n_pass = sum([pass_dvol, pass_spread, pass_gap, pass_zvol, pass_active, pass_cv])

    # Estimated round-trip cost: spread + 2x taker fee (entry + exit)
    est_rt_cost = spread_proxy + 2 * KRAKEN_FEE_TAKER

    results.append({
        "symbol": base,
        "pair": pair,
        "n_bars": n_bars,
        "gap_pct": round(gap_pct, 4),
        "zero_vol_pct": round(zero_vol_pct, 4),
        "spread_proxy": round(spread_proxy, 6),
        "daily_dvol": round(daily_dvol, 0),
        "hourly_active_pct": round(hourly_active_pct, 3),
        "vol_cv": round(vol_cv, 3),
        "last_price": last_price,
        "oos_sharpe": sharpe,
        "is_major": is_major,
        "est_rt_cost": round(est_rt_cost, 6),
        "pass_all": pass_all,
        "n_pass": n_pass,
        "checks": {
            "dvol": pass_dvol,
            "spread": pass_spread,
            "gap": pass_gap,
            "zvol": pass_zvol,
            "active": pass_active,
            "cv": pass_cv,
        }
    })

    tag = "PASS" if pass_all else f"FAIL({6 - n_pass})"
    print(f"  [{i+1}/{len(both)}] {base:>8s}  dvol/d=${daily_dvol:>14,.0f}  "
          f"spread={spread_proxy:.5f}  active={hourly_active_pct:.0%}  "
          f"cv={vol_cv:.2f}  [{tag}]")

    time.sleep(1.5)

# ── Sort and print final table ────────────────────────────────────────────
# Sort: pass_all first, then by daily dollar volume
results.sort(key=lambda x: (x["pass_all"], x["daily_dvol"]), reverse=True)

print()
print("=" * 140)
print(f"  INTRADAY HF SUITABILITY SCREEN  |  {len(results)} coins  |  Thresholds: "
      f"dvol>=${MIN_DAILY_DVOL_USD / 1e6:.0f}M  spread<={MAX_SPREAD_ALT * 100:.2f}%  "
      f"active>={MIN_HOURLY_ACTIVE_PCT:.0%}  gap<={MAX_GAP_PCT:.0%}  cv<={MAX_VOL_CV}")
print("=" * 140)

hdr = (f"{'#':>3s}  {'Symbol':>8s}  {'Pair':<12s}  {'Daily $Vol':>16s}  "
       f"{'Spread':>8s}  {'RT Cost':>8s}  {'Active%':>8s}  {'VolCV':>6s}  "
       f"{'Gap%':>6s}  {'ZVol%':>6s}  {'OOS_SR':>7s}  {'Status':>10s}")
print(hdr)
print("-" * len(hdr))

for rank, r in enumerate(results, 1):
    oos_str = f"{r['oos_sharpe']:+.2f}" if r["oos_sharpe"] is not None else "  n/a"
    status = "  PASS" if r["pass_all"] else f"FAIL({6 - r['n_pass']})"

    # Mark which checks failed
    if not r["pass_all"]:
        fails = [k for k, v in r["checks"].items() if not v]
        status += " " + ",".join(fails)

    print(f"{rank:3d}  {r['symbol']:>8s}  {r['pair']:<12s}  "
          f"${r['daily_dvol']:>15,.0f}  "
          f"{r['spread_proxy']:8.5f}  {r['est_rt_cost']*100:7.2f}%  "
          f"{r['hourly_active_pct']:7.0%}  "
          f"{r['vol_cv']:6.2f}  "
          f"{r['gap_pct']:5.1%}  {r['zero_vol_pct']:5.1%}  "
          f"{oos_str:>7s}  {status}")

# ── Recommended HF symbols ───────────────────────────────────────────────
passed = [r for r in results if r["pass_all"]]

# Priority: ensure majors (BTC, ETH) are included even if borderline
# Check if any major narrowly missed (failed only 1 check by small margin)
majors_passed = [r for r in passed if r["is_major"]]
majors_missed = [r for r in results if r["is_major"] and not r["pass_all"]]

for m in majors_missed:
    if m["n_pass"] >= 5:  # failed only 1 criterion
        print(f"\n  [PRIORITY] {m['symbol']} is a major coin, failed only "
              f"{[k for k, v in m['checks'].items() if not v]} — including with relaxed threshold")
        m["pass_all"] = True
        m["priority_override"] = True
        passed.append(m)

alts_passed = [r for r in passed if not r["is_major"]]

print()
print("=" * 90)
print(f"  RECOMMENDED HF SYMBOLS: {len(passed)} coins passed "
      f"({len(majors_passed) + len([m for m in majors_missed if m.get('priority_override')])} majors, "
      f"{len(alts_passed)} alts)")
print("=" * 90)

if not passed:
    print("  (none passed all filters — consider relaxing thresholds)")
else:
    for rank, r in enumerate(passed, 1):
        tier = "MAJOR" if r["is_major"] else "ALT"
        oos_str = f"OOS={r['oos_sharpe']:+.2f}" if r["oos_sharpe"] is not None else "OOS=n/a"
        override = " *priority*" if r.get("priority_override") else ""
        print(f"  {rank}. {r['symbol']:>8s}  [{tier:5s}]  "
              f"dvol/d=${r['daily_dvol']:>13,.0f}  "
              f"spread={r['spread_proxy']:.5f}  "
              f"RT_cost={r['est_rt_cost']:.4f} ({r['est_rt_cost']*100:.2f}%)  "
              f"active={r['hourly_active_pct']:.0%}  "
              f"cv={r['vol_cv']:.2f}  {oos_str}{override}")

    # Select core symbols (up to MAX_CORE_SYMBOLS, majors first, then alts by dvol)
    core = []
    # Always include majors first
    for r in passed:
        if r["is_major"] and len(core) < MAX_CORE_SYMBOLS:
            core.append(r)
    # Then add alts sorted by daily dvol
    alts_sorted = sorted([r for r in passed if not r["is_major"]],
                         key=lambda x: x["daily_dvol"], reverse=True)
    for r in alts_sorted:
        if len(core) < MAX_CORE_SYMBOLS:
            core.append(r)

    print()
    core_syms = [f"{r['symbol']}-USD" for r in core]
    print(f"  Core HF symbols ({len(core)}):")
    for r in core:
        tier = "MAJOR" if r["is_major"] else "ALT"
        print(f"    {r['symbol']:>8s} [{tier}]  RT_cost={r['est_rt_cost']*100:.2f}%  "
              f"dvol/d=${r['daily_dvol']:>13,.0f}")

    print()
    sym_list = ",".join(core_syms)
    print(f"  Train command:")
    print(f"    python main.py train-crypto-intraday --symbols {sym_list}")

# ── Save results ──────────────────────────────────────────────────────────
out_path = PROJECT_ROOT / "outputs" / "intraday_coin_screen.json"
out_path.parent.mkdir(parents=True, exist_ok=True)

# Build core_symbols list for downstream consumption
core_symbols_out = [
    {
        "symbol": f"{r['symbol']}-USD",
        "pair": r["pair"],
        "tier": "major" if r["is_major"] else "alt",
        "daily_dvol": r["daily_dvol"],
        "spread_proxy": r["spread_proxy"],
        "est_rt_cost": r["est_rt_cost"],
        "hourly_active_pct": r["hourly_active_pct"],
        "vol_cv": r["vol_cv"],
    }
    for r in core
] if 'core' in dir() else []

with open(out_path, "w") as f:
    json.dump({
        "screened_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "thresholds": {
            "min_daily_dvol_usd": MIN_DAILY_DVOL_USD,
            "min_hourly_active_pct": MIN_HOURLY_ACTIVE_PCT,
            "max_spread_major": MAX_SPREAD_MAJOR,
            "max_spread_alt": MAX_SPREAD_ALT,
            "max_gap_pct": MAX_GAP_PCT,
            "max_zero_vol_pct": MAX_ZERO_VOL_PCT,
            "max_vol_cv": MAX_VOL_CV,
        },
        "fee_model": {
            "kraken_taker": KRAKEN_FEE_TAKER,
            "kraken_maker": KRAKEN_FEE_MAKER,
        },
        "total_screened": len(results),
        "total_passed": len(passed),
        "passed_symbols": [r["symbol"] for r in passed],
        "core_symbols": core_symbols_out,
        "results": results,
    }, f, indent=2)
print(f"\n[saved] {out_path}")
