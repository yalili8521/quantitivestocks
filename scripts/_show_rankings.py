"""Quick script to show current ETF rankings and promotions."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

ROOT = os.path.join(os.path.dirname(__file__), '..')

# --- Swing ranked ---
swing_ranked_path = os.path.join(ROOT, 'models', 'swing', 'etf_candidates_ranked.json')
if os.path.exists(swing_ranked_path):
    with open(swing_ranked_path) as f:
        sr = json.load(f)
    print("=" * 80)
    print("  SWING ETF RANKINGS (etf_candidates_ranked.json)")
    print("=" * 80)
    print(f"  {'#':>3}  {'Symbol':<8} {'Score':>8} {'OOS Sharpe':>12} {'OOS Return':>12}")
    print(f"  {'-'*60}")
    for i, c in enumerate(sr.get('candidates', [])[:20]):
        sym = c.get('symbol', '?')
        score = c.get('final_score', 0)
        sharpe = c.get('oos_sharpe', 0)
        ret = c.get('oos_return_pct', c.get('total_return_pct', 0))
        print(f"  {i+1:>3}  {sym:<8} {score:>8.4f} {sharpe:>+12.3f} {ret:>+11.1f}%")
    print()

# --- Intraday ranked ---
intra_ranked_path = os.path.join(ROOT, 'models', 'swing', 'etf_candidates_intraday_ranked.json')
if os.path.exists(intra_ranked_path):
    with open(intra_ranked_path) as f:
        ir = json.load(f)
    print("=" * 80)
    print("  INTRADAY ETF RANKINGS (etf_candidates_intraday_ranked.json)")
    print("=" * 80)
    print(f"  {'#':>3}  {'Symbol':<8} {'Score':>8}")
    print(f"  {'-'*30}")
    for i, c in enumerate(ir.get('candidates', [])[:20]):
        sym = c.get('symbol', '?')
        score = c.get('intraday_rank_score', c.get('final_score', 0))
        print(f"  {i+1:>3}  {sym:<8} {score:>8.4f}")
    print()

# --- Promoted symbols (swing) ---
prom_swing = os.path.join(ROOT, 'models', 'swing', 'promoted_symbols.json')
if os.path.exists(prom_swing):
    with open(prom_swing) as f:
        ps = json.load(f)
    symbols = ps.get('symbols', [])
    thresholds = ps.get('thresholds', {})
    print("=" * 80)
    print(f"  SWING PROMOTED ({len(symbols)} symbols)")
    print(f"  Thresholds: {thresholds}")
    print("=" * 80)
    print(f"  {', '.join(symbols)}")
    print()

# --- Promoted symbols (intraday) ---
prom_intra = os.path.join(ROOT, 'models', 'intraday', 'promoted_symbols.json')
if os.path.exists(prom_intra):
    with open(prom_intra) as f:
        pi = json.load(f)
    symbols = pi.get('symbols', [])
    thresholds = pi.get('thresholds', {})
    print("=" * 80)
    print(f"  INTRADAY PROMOTED ({len(symbols)} symbols)")
    print(f"  Thresholds: {thresholds}")
    print("=" * 80)
    print(f"  {', '.join(symbols)}")
    print()

# --- What paper_trader would actually resolve ---
print("=" * 80)
print("  PAPER TRADER RESOLUTION (what would actually trade)")
print("=" * 80)
try:
    from paper_trader import _resolve_swing_symbols, _resolve_intraday_symbols
    swing = _resolve_swing_symbols()
    print(f"  Swing ({len(swing)}): {', '.join(swing)}")
except Exception as e:
    print(f"  Swing: ERROR - {e}")

try:
    from paper_trader import _resolve_intraday_symbols
    intraday = _resolve_intraday_symbols()
    print(f"  Intraday ({len(intraday)}): {', '.join(intraday)}")
except Exception as e:
    print(f"  Intraday: ERROR - {e}")
print()
