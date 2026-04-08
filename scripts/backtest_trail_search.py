"""Fine-grained trail % search for pure runner strategy."""
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
    peak_price = 0.0
    is_trailing = False
    bars_since_exit = 999
    trade_pnl = 0.0
    trade_entry_eq = 0.0
    trades = []
    sess_start = p['sess_start_h'] * 60 + p['sess_start_m']
    sess_end = p['sess_end_h'] * 60 + p['sess_end_m']
    prev_long = False
    prev_short = False
    start_bar = max(p['ma_slow'], p['acc_lookback'] + 1)

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
        st = sc and not prev_short
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
                    stop_level = max(fl, peak_price * (1 - p['trail'] / 100))
                elif position == -1 and c <= entry_price * (1 - p['be_trigger'] / 100):
                    is_trailing = True
                    fl = entry_price * (1 - 0.05 / 100)
                    stop_level = min(fl, peak_price * (1 + p['trail'] / 100))

            if is_trailing:
                if position == 1:
                    ns1 = peak_price * (1 - p['trail'] / 100)
                    if ns1 > stop_level:
                        stop_level = ns1
                elif position == -1:
                    ns1 = peak_price * (1 + p['trail'] / 100)
                    if ns1 < stop_level:
                        stop_level = ns1

            ex = None
            if not is_trailing:
                if (position == 1 and c <= stop_level) or (position == -1 and c >= stop_level):
                    ex = 'SL'
            if is_trailing:
                if (position == 1 and c <= stop_level) or (position == -1 and c >= stop_level):
                    ex = 'TS'
            if force_close:
                ex = 'MC'

            if ex:
                if position == 1:
                    trade_pnl = (c - entry_price) / entry_price * trade_entry_eq
                else:
                    trade_pnl = (entry_price - c) / entry_price * trade_entry_eq
                equity += trade_pnl
                trades.append({'pnl': trade_pnl, 'exit_type': ex})
                position = 0
                bars_since_exit = 0
                is_trailing = False
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
            trade_pnl = 0.0
            trade_entry_eq = equity * 0.95
            bars_since_exit = 0
        elif position == 0 and st:
            position = -1
            entry_price = c
            entry_bar = i
            stop_level = c * (1 + p['sl_pct'] / 100)
            peak_price = c
            is_trailing = False
            trade_pnl = 0.0
            trade_entry_eq = equity * 0.95
            bars_since_exit = 0

    if position != 0:
        c = close[-1]
        if position == 1:
            trade_pnl = (c - entry_price) / entry_price * trade_entry_eq
        else:
            trade_pnl = (entry_price - c) / entry_price * trade_entry_eq
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
    ts_c = sum(1 for t in trades if t['exit_type'] == 'TS')

    return {
        'pnl': equity - 5000, 'pf': gp / gl, 'wr': len(wins) / nt * 100,
        'trades': nt, 'avg_win': aw, 'avg_loss': al,
        'sl_count': sl_c, 'ts_count': ts_c
    }


shared = {
    'ma_fast': 5, 'ma_mid': 10, 'ma_slow': 20,
    'acc_lookback': 10, 'acc_min_bars': 6, 'max_bar_size': 0.35,
    'cooldown': 5,
    'sess_start_h': 9, 'sess_start_m': 30, 'sess_end_h': 11, 'sess_end_m': 0,
}

# ═══════════════════════════════════════════════════════════════
# Trail % sensitivity (with current SL=0.50, BE=0.40)
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 90)
print('TRAIL % SENSITIVITY (SL=0.50, BE=0.40)')
print('=' * 90)
trail_vals = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 1.00, 1.25, 1.50]
print()
print('%-6s | %10s | %8s | %6s | %6s | %8s | %8s | %4s | %4s' % (
    'Trail', 'P&L', 'PF', 'WR%', 'Trades', 'AvgWin', 'AvgLoss', 'SL', 'TS'))
print('-' * 85)
for tr in trail_vals:
    p = dict(shared)
    p['sl_pct'] = 0.50
    p['be_trigger'] = 0.40
    p['trail'] = tr
    r = bt(qqq, p)
    print('%-6.2f | $%9.2f | %8.3f | %5.1f%% | %6d | $%7.2f | $%7.2f | %4d | %4d' % (
        tr, r['pnl'], r['pf'], r['wr'], r['trades'], r['avg_win'], r['avg_loss'],
        r['sl_count'], r['ts_count']))

# ═══════════════════════════════════════════════════════════════
# Full grid: Trail x BE (with SL=0.50)
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 90)
print('TRAIL x BE GRID (SL=0.50, Top 25 by P&L)')
print('=' * 90)
trail_g = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75, 1.00]
be_g = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

res = []
for tr, be in product(trail_g, be_g):
    p = dict(shared)
    p['sl_pct'] = 0.50
    p['be_trigger'] = be
    p['trail'] = tr
    r = bt(qqq, p)
    res.append((tr, be, r['pnl'], r['pf'], r['wr'], r['trades'], r['sl_count'], r['ts_count']))

res.sort(key=lambda x: x[2], reverse=True)
print()
print('%-6s %-6s | %10s | %8s | %6s | %6s | %4s | %4s' % (
    'Trail', 'BE', 'P&L', 'PF', 'WR%', 'Trades', 'SL', 'TS'))
print('-' * 75)
for tr, be, pnl, pf, wr, t, slc, tsc in res[:25]:
    print('%-6.2f %-6.2f | $%9.2f | %8.3f | %5.1f%% | %6d | %4d | %4d' % (
        tr, be, pnl, pf, wr, t, slc, tsc))

# ═══════════════════════════════════════════════════════════════
# Full grid: Trail x BE x SL
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 90)
print('TRAIL x BE x SL MEGA GRID (Top 25 by P&L)')
print('=' * 90)
sl_g = [0.30, 0.40, 0.50, 0.60, 0.75]

res2 = []
for tr, be, sl in product(trail_g, be_g, sl_g):
    if be >= sl:
        continue
    p = dict(shared)
    p['sl_pct'] = sl
    p['be_trigger'] = be
    p['trail'] = tr
    r = bt(qqq, p)
    res2.append((tr, be, sl, r['pnl'], r['pf'], r['wr'], r['trades'], r['sl_count'], r['ts_count']))

res2.sort(key=lambda x: x[3], reverse=True)
print()
print('%-6s %-6s %-6s | %10s | %8s | %6s | %6s | %4s | %4s' % (
    'Trail', 'BE', 'SL', 'P&L', 'PF', 'WR%', 'Trades', 'SL', 'TS'))
print('-' * 82)
for tr, be, sl, pnl, pf, wr, t, slc, tsc in res2[:25]:
    print('%-6.2f %-6.2f %-6.2f | $%9.2f | %8.3f | %5.1f%% | %6d | %4d | %4d' % (
        tr, be, sl, pnl, pf, wr, t, slc, tsc))
