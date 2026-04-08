"""
Compare 1-level vs 2-level trailing stop with full SL/BE grid search.
Uses best trail values from prior search (trail=0.50 for single, 0.25/0.50 for 2-level).
"""
import yfinance as yf
import pandas as pd
import numpy as np
from itertools import product

print("Downloading QQQ 5m data...")
qqq = yf.download('QQQ', period='60d', interval='5m', auto_adjust=True)
if isinstance(qqq.columns, pd.MultiIndex):
    qqq.columns = qqq.columns.get_level_values(0)
qqq.index = qqq.index.tz_convert('America/New_York')
print('Bars: %d' % len(qqq))


def bt(df, p):
    close = df['Close'].values.astype(float)
    opn = df['Open'].values.astype(float)
    times = df.index
    n = len(close)
    ma_f = pd.Series(close).rolling(p['ma_fast']).mean().values
    ma_m = pd.Series(close).rolling(p['ma_mid']).mean().values
    ma_s = pd.Series(close).rolling(p['ma_slow']).mean().values
    equity = 5000.0
    position = 0
    entry_price = 0.0
    entry_bar = 0
    stop_level = 0.0
    stop1_level = 0.0
    stop2_level = 0.0
    peak_price = 0.0
    is_trailing = False
    trail1_exited = False
    remaining_pct = 1.0
    bars_since_exit = 999
    trade_pnl = 0.0
    trade_entry_eq = 0.0
    trades = []
    sess_start = p['sess_start_h'] * 60 + p['sess_start_m']
    sess_end = p['sess_end_h'] * 60 + p['sess_end_m']
    prev_long = False
    prev_short = False
    start_bar = max(p['ma_slow'], p['acc_lookback'] + 1)
    two_level = p.get('two_level', True)

    for i in range(start_bar, n):
        t = times[i]
        t_min = t.hour * 60 + t.minute
        in_session = t_min >= sess_start and t_min < sess_end
        force_close = t_min >= 960 and t_min < 970
        c = close[i]
        if np.isnan(ma_f[i]) or np.isnan(ma_m[i]) or np.isnan(ma_s[i]):
            continue

        bull = ma_f[i] > ma_m[i] and ma_m[i] > ma_s[i]
        bear = ma_f[i] < ma_m[i] and ma_m[i] < ma_s[i]
        pa = c > ma_f[i]
        pb = c < ma_f[i]

        br = 0
        bf = 0
        ns = True
        for j in range(p['acc_lookback']):
            idx = i - j
            ip = i - j - 1
            if ip < 0:
                break
            if close[idx] > close[ip]:
                br += 1
            if close[idx] < close[ip]:
                bf += 1
            bm = abs(close[idx] - opn[idx]) / opn[idx] * 100 if opn[idx] > 0 else 0
            if bm > p['max_bar_size']:
                ns = False

        acc = br >= p['acc_min_bars'] and ns
        dist = bf >= p['acc_min_bars'] and ns
        mr = ma_f[i] > ma_f[i - 1] and ma_m[i] > ma_m[i - 1]
        mf = ma_f[i] < ma_f[i - 1] and ma_m[i] < ma_m[i - 1]
        co = bars_since_exit >= p['cooldown']

        lc = bull and pa and acc and mr and in_session and co
        sc = bear and pb and dist and mf and in_session and co
        lt = lc and not prev_long
        st_sig = sc and not prev_short
        prev_long = lc
        prev_short = sc

        if position != 0:
            if position == 1 and c > peak_price:
                peak_price = c
            if position == -1 and c < peak_price:
                peak_price = c

            if not is_trailing:
                if position == 1 and c >= entry_price * (1 + p['be_trigger'] / 100):
                    is_trailing = True
                    fl = entry_price * (1 + 0.05 / 100)
                    stop1_level = max(fl, peak_price * (1 - p['trail1'] / 100))
                    stop2_level = max(fl, peak_price * (1 - p['trail2'] / 100))
                elif position == -1 and c <= entry_price * (1 - p['be_trigger'] / 100):
                    is_trailing = True
                    fl = entry_price * (1 - 0.05 / 100)
                    stop1_level = min(fl, peak_price * (1 + p['trail1'] / 100))
                    stop2_level = min(fl, peak_price * (1 + p['trail2'] / 100))

            if is_trailing:
                if position == 1:
                    ns1 = peak_price * (1 - p['trail1'] / 100)
                    if ns1 > stop1_level:
                        stop1_level = ns1
                    ns2 = peak_price * (1 - p['trail2'] / 100)
                    if ns2 > stop2_level:
                        stop2_level = ns2
                elif position == -1:
                    ns1 = peak_price * (1 + p['trail1'] / 100)
                    if ns1 < stop1_level:
                        stop1_level = ns1
                    ns2 = peak_price * (1 + p['trail2'] / 100)
                    if ns2 < stop2_level:
                        stop2_level = ns2

            ex = None

            # Fixed SL (before BE trigger)
            if not is_trailing:
                if (position == 1 and c <= stop_level) or (position == -1 and c >= stop_level):
                    ex = 'SL'

            if two_level:
                # 2-level: trail1 exits partial, trail2 exits rest
                if is_trailing and not trail1_exited:
                    if (position == 1 and c <= stop1_level) or (position == -1 and c >= stop1_level):
                        trail1_exited = True
                        t1f = p['trail1_close'] / 100.0
                        if position == 1:
                            trade_pnl += (c - entry_price) / entry_price * t1f * remaining_pct * trade_entry_eq
                        else:
                            trade_pnl += (entry_price - c) / entry_price * t1f * remaining_pct * trade_entry_eq
                        remaining_pct *= (1 - t1f)

                if is_trailing and trail1_exited:
                    if (position == 1 and c <= stop2_level) or (position == -1 and c >= stop2_level):
                        ex = 'T2'

                if is_trailing and not trail1_exited:
                    if (position == 1 and c <= stop2_level) or (position == -1 and c >= stop2_level):
                        ex = 'TS'
            else:
                # 1-level: single trail exits 100%
                if is_trailing:
                    if (position == 1 and c <= stop2_level) or (position == -1 and c >= stop2_level):
                        ex = 'TS'

            if force_close:
                ex = 'MC'

            if ex:
                if position == 1:
                    trade_pnl += (c - entry_price) / entry_price * remaining_pct * trade_entry_eq
                else:
                    trade_pnl += (entry_price - c) / entry_price * remaining_pct * trade_entry_eq
                equity += trade_pnl
                trades.append({'pnl': trade_pnl, 'exit_type': ex})
                position = 0
                bars_since_exit = 0
                is_trailing = False
                trail1_exited = False
                remaining_pct = 1.0
                trade_pnl = 0.0
                continue

        if position == 0:
            bars_since_exit += 1

        if position == 0 and lt:
            position = 1
            entry_price = c
            entry_bar = i
            stop_level = c * (1 - p['sl_pct'] / 100)
            peak_price = c
            is_trailing = False
            trail1_exited = False
            remaining_pct = 1.0
            trade_pnl = 0.0
            trade_entry_eq = equity * 0.95
            bars_since_exit = 0
        elif position == 0 and st_sig:
            position = -1
            entry_price = c
            entry_bar = i
            stop_level = c * (1 + p['sl_pct'] / 100)
            peak_price = c
            is_trailing = False
            trail1_exited = False
            remaining_pct = 1.0
            trade_pnl = 0.0
            trade_entry_eq = equity * 0.95
            bars_since_exit = 0

    if position != 0:
        c = close[-1]
        if position == 1:
            trade_pnl += (c - entry_price) / entry_price * remaining_pct * trade_entry_eq
        else:
            trade_pnl += (entry_price - c) / entry_price * remaining_pct * trade_entry_eq
        equity += trade_pnl
        trades.append({'pnl': trade_pnl, 'exit_type': 'EOD'})

    nt = len(trades)
    if nt == 0:
        return {'pnl': 0, 'pf': 0, 'wr': 0, 'trades': 0, 'avg_win': 0, 'avg_loss': 0, 'sl_count': 0}

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    gp = sum(t['pnl'] for t in wins) if wins else 0
    gl = abs(sum(t['pnl'] for t in losses)) if losses else 0.001
    aw = gp / len(wins) if wins else 0
    al = gl / len(losses) if losses else 0
    sl_c = sum(1 for t in trades if t['exit_type'] == 'SL')

    return {
        'pnl': equity - 5000, 'pf': gp / gl, 'wr': len(wins) / nt * 100,
        'trades': nt, 'avg_win': aw, 'avg_loss': al, 'sl_count': sl_c
    }


shared = {
    'ma_fast': 5, 'ma_mid': 10, 'ma_slow': 20,
    'acc_lookback': 10, 'acc_min_bars': 6, 'max_bar_size': 0.35,
    'cooldown': 5,
    'sess_start_h': 9, 'sess_start_m': 30, 'sess_end_h': 11, 'sess_end_m': 0,
}

sl_vals = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.75, 1.00]
be_vals = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

# ═══════════════════════════════════════════════════════════════
# 2-LEVEL TRAIL: Trail1=0.25%, Trail2=0.50%, close 50%
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 75)
print('2-LEVEL TRAIL (Trail1=0.25%, Trail2=0.50%, 50% exit on T1)')
print('=' * 75)

res2 = []
for sl, be in product(sl_vals, be_vals):
    if be >= sl:
        continue
    p = dict(shared)
    p['sl_pct'] = sl
    p['be_trigger'] = be
    p['trail1'] = 0.25
    p['trail2'] = 0.50
    p['trail1_close'] = 50
    p['two_level'] = True
    r = bt(qqq, p)
    res2.append((sl, be, r['pnl'], r['pf'], r['wr'], r['trades'], r['sl_count']))

res2.sort(key=lambda x: x[2], reverse=True)
print()
print('%-6s %-6s | %10s | %8s | %6s | %6s | %s' % (
    'SL%', 'BE%', 'P&L', 'PF', 'WR%', 'Trades', 'SL Hits'))
print('-' * 70)
for sl, be, pnl, pf, wr, t, slc in res2[:25]:
    print('%-6.2f %-6.2f | $%9.2f | %8.3f | %5.1f%% | %6d | %d' % (
        sl, be, pnl, pf, wr, t, slc))

# ═══════════════════════════════════════════════════════════════
# 1-LEVEL TRAIL: single trail at 0.50%
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 75)
print('1-LEVEL TRAIL (single trail=0.50%, exits 100%)')
print('=' * 75)

res1 = []
for sl, be in product(sl_vals, be_vals):
    if be >= sl:
        continue
    p = dict(shared)
    p['sl_pct'] = sl
    p['be_trigger'] = be
    p['trail1'] = 0.50  # not used in 1-level mode
    p['trail2'] = 0.50
    p['trail1_close'] = 0
    p['two_level'] = False
    r = bt(qqq, p)
    res1.append((sl, be, r['pnl'], r['pf'], r['wr'], r['trades'], r['sl_count']))

res1.sort(key=lambda x: x[2], reverse=True)
print()
print('%-6s %-6s | %10s | %8s | %6s | %6s | %s' % (
    'SL%', 'BE%', 'P&L', 'PF', 'WR%', 'Trades', 'SL Hits'))
print('-' * 70)
for sl, be, pnl, pf, wr, t, slc in res1[:25]:
    print('%-6.2f %-6.2f | $%9.2f | %8.3f | %5.1f%% | %6d | %d' % (
        sl, be, pnl, pf, wr, t, slc))

# ═══════════════════════════════════════════════════════════════
# HEAD-TO-HEAD: best of each
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 75)
print('HEAD-TO-HEAD COMPARISON')
print('=' * 75)
b2 = res2[0]
b1 = res1[0]
print()
print('  2-Level best: SL=%.2f%% BE=%.2f%% -> P&L: $%.2f | PF: %.3f | WR: %.1f%% | T: %d | SL: %d' % (
    b2[0], b2[1], b2[2], b2[3], b2[4], b2[5], b2[6]))
print('  1-Level best: SL=%.2f%% BE=%.2f%% -> P&L: $%.2f | PF: %.3f | WR: %.1f%% | T: %d | SL: %d' % (
    b1[0], b1[1], b1[2], b1[3], b1[4], b1[5], b1[6]))
print()
diff = b2[2] - b1[2]
print('  Difference: $%.2f (%s wins)' % (abs(diff), '2-Level' if diff > 0 else '1-Level'))

# Same SL/BE comparison
print()
print('=' * 75)
print('SAME SETTINGS COMPARISON (SL=0.50, BE=0.40)')
print('=' * 75)
# Find matching entries
for sl, be, pnl, pf, wr, t, slc in res2:
    if sl == 0.50 and be == 0.40:
        print('  2-Level: P&L: $%.2f | PF: %.3f | WR: %.1f%% | T: %d' % (pnl, pf, wr, t))
        break
for sl, be, pnl, pf, wr, t, slc in res1:
    if sl == 0.50 and be == 0.40:
        print('  1-Level: P&L: $%.2f | PF: %.3f | WR: %.1f%% | T: %d' % (pnl, pf, wr, t))
        break

# Also test wider trail values for 1-level
print()
print('=' * 75)
print('1-LEVEL TRAIL SENSITIVITY (different trail widths, SL=0.50, BE=0.40)')
print('=' * 75)
for trail in [0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75, 1.00]:
    p = dict(shared)
    p['sl_pct'] = 0.50
    p['be_trigger'] = 0.40
    p['trail1'] = trail
    p['trail2'] = trail
    p['trail1_close'] = 0
    p['two_level'] = False
    r = bt(qqq, p)
    print('  Trail=%.2f%% -> P&L: $%7.2f | PF: %.3f | WR: %.1f%% | T: %d' % (
        trail, r['pnl'], r['pf'], r['wr'], r['trades']))
