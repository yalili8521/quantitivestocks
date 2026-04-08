"""Test volume filters on pure runner strategy (trail=0.50, BE=0.20, SL=0.50)."""
import yfinance as yf
import pandas as pd
import numpy as np

print("Downloading QQQ 5m data...")
qqq = yf.download('QQQ', period='60d', interval='5m', auto_adjust=True)
if isinstance(qqq.columns, pd.MultiIndex):
    qqq.columns = qqq.columns.get_level_values(0)
qqq.index = qqq.index.tz_convert('America/New_York')
print('Bars: %d' % len(qqq))

# Precompute volume metrics
vol = qqq['Volume'].values.astype(float)
close = qqq['Close'].values.astype(float)
opn = qqq['Open'].values.astype(float)

# RVOL: current bar volume / avg volume over N bars
def calc_rvol(vol_arr, period=20):
    avg = pd.Series(vol_arr).rolling(period).mean().values
    rvol = vol_arr / np.where(avg > 0, avg, 1)
    return rvol

# Rising volume: avg volume over recent N bars vs avg volume over prior N bars
def calc_vol_trend(vol_arr, window=10):
    recent = pd.Series(vol_arr).rolling(window).mean().values
    prior = pd.Series(vol_arr).shift(window).rolling(window).mean().values
    ratio = recent / np.where(prior > 0, prior, 1)
    return ratio

# VWAP (intraday reset)
def calc_vwap(df):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    vol_s = df['Volume']
    # Group by date for daily reset
    dates = df.index.date
    vwap = np.zeros(len(df))
    for d in np.unique(dates):
        mask = dates == d
        idx = np.where(mask)[0]
        cum_tp_vol = np.cumsum(tp.values[idx] * vol_s.values[idx])
        cum_vol = np.cumsum(vol_s.values[idx])
        vwap[idx] = cum_tp_vol / np.where(cum_vol > 0, cum_vol, 1)
    return vwap

rvol_20 = calc_rvol(vol, 20)
rvol_10 = calc_rvol(vol, 10)
vol_trend = calc_vol_trend(vol, 10)
vwap = calc_vwap(qqq)


def bt(df, p, rvol_arr, vol_trend_arr, vwap_arr, vol_arr):
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
    start_bar = max(p['ma_slow'], p['acc_lookback'] + 1, 21)

    rvol_min = p.get('rvol_min', 0)
    vol_trend_min = p.get('vol_trend_min', 0)
    use_vwap = p.get('use_vwap', False)
    min_vol = p.get('min_vol', 0)

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

        # Volume filters
        vol_ok = True
        if rvol_min > 0 and rvol_arr[i] < rvol_min:
            vol_ok = False
        if vol_trend_min > 0 and vol_trend_arr[i] < vol_trend_min:
            vol_ok = False
        if use_vwap:
            if bull and c < vwap_arr[i]:
                vol_ok = False
            if bear and c > vwap_arr[i]:
                vol_ok = False
        if min_vol > 0 and vol_arr[i] < min_vol:
            vol_ok = False

        lc = bull and pa and acc and mr and in_session and co and vol_ok
        sc = bear and pb and dist and mf and in_session and co and vol_ok
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

    return {
        'pnl': equity - 5000, 'pf': gp / gl, 'wr': len(wins) / nt * 100,
        'trades': nt, 'avg_win': aw, 'avg_loss': al, 'sl_count': sl_c
    }


# Best pure runner settings (the original champion)
base = {
    'ma_fast': 5, 'ma_mid': 10, 'ma_slow': 20,
    'acc_lookback': 10, 'acc_min_bars': 6, 'max_bar_size': 0.35,
    'sl_pct': 0.50, 'be_trigger': 0.20, 'trail': 0.50,
    'cooldown': 5,
    'sess_start_h': 9, 'sess_start_m': 30, 'sess_end_h': 11, 'sess_end_m': 0,
}

# Baseline (no volume filter)
print()
print('=' * 90)
print('BASELINE (no volume filter)')
print('=' * 90)
r = bt(qqq, base, rvol_20, vol_trend, vwap, vol)
print('P&L: $%.2f | PF: %.3f | WR: %.1f%% | Trades: %d | SL: %d' % (
    r['pnl'], r['pf'], r['wr'], r['trades'], r['sl_count']))

# ═══════════════════════════════════════════════════════════════
# Test 1: RVOL filter (current bar volume vs 20-bar average)
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 90)
print('TEST 1: RVOL FILTER (bar volume / 20-bar avg volume)')
print('=' * 90)
rvol_vals = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5]
print()
print('%-8s | %10s | %8s | %6s | %6s | %4s | %8s | %8s' % (
    'RVOL>', 'P&L', 'PF', 'WR%', 'Trades', 'SL', 'AvgWin', 'AvgLoss'))
print('-' * 80)
for rv in rvol_vals:
    p = dict(base)
    p['rvol_min'] = rv
    r = bt(qqq, p, rvol_20, vol_trend, vwap, vol)
    print('%-8.1f | $%9.2f | %8.3f | %5.1f%% | %6d | %4d | $%7.2f | $%7.2f' % (
        rv, r['pnl'], r['pf'], r['wr'], r['trades'], r['sl_count'], r['avg_win'], r['avg_loss']))

# ═══════════════════════════════════════════════════════════════
# Test 2: Volume trend (recent 10-bar avg vs prior 10-bar avg)
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 90)
print('TEST 2: VOLUME TREND (recent 10-bar avg / prior 10-bar avg)')
print('=' * 90)
vt_vals = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0]
print()
print('%-8s | %10s | %8s | %6s | %6s | %4s | %8s | %8s' % (
    'VTrend>', 'P&L', 'PF', 'WR%', 'Trades', 'SL', 'AvgWin', 'AvgLoss'))
print('-' * 80)
for vt in vt_vals:
    p = dict(base)
    p['vol_trend_min'] = vt
    r = bt(qqq, p, rvol_20, vol_trend, vwap, vol)
    print('%-8.1f | $%9.2f | %8.3f | %5.1f%% | %6d | %4d | $%7.2f | $%7.2f' % (
        vt, r['pnl'], r['pf'], r['wr'], r['trades'], r['sl_count'], r['avg_win'], r['avg_loss']))

# ═══════════════════════════════════════════════════════════════
# Test 3: VWAP filter (long only above VWAP, short only below)
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 90)
print('TEST 3: VWAP FILTER (long above VWAP, short below VWAP)')
print('=' * 90)
p = dict(base)
p['use_vwap'] = True
r = bt(qqq, p, rvol_20, vol_trend, vwap, vol)
print('VWAP ON  | $%9.2f | %8.3f | %5.1f%% | %6d | %4d' % (
    r['pnl'], r['pf'], r['wr'], r['trades'], r['sl_count']))
p['use_vwap'] = False
r = bt(qqq, p, rvol_20, vol_trend, vwap, vol)
print('VWAP OFF | $%9.2f | %8.3f | %5.1f%% | %6d | %4d' % (
    r['pnl'], r['pf'], r['wr'], r['trades'], r['sl_count']))

# ═══════════════════════════════════════════════════════════════
# Test 4: Combinations
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 90)
print('TEST 4: BEST COMBINATIONS')
print('=' * 90)
combos = [
    {'label': 'RVOL>1.0 + VWAP',       'rvol_min': 1.0, 'use_vwap': True},
    {'label': 'RVOL>1.2 + VWAP',       'rvol_min': 1.2, 'use_vwap': True},
    {'label': 'RVOL>0.8 + VWAP',       'rvol_min': 0.8, 'use_vwap': True},
    {'label': 'RVOL>1.0 + VTrend>1.0', 'rvol_min': 1.0, 'vol_trend_min': 1.0},
    {'label': 'RVOL>1.0 + VTrend>1.1', 'rvol_min': 1.0, 'vol_trend_min': 1.1},
    {'label': 'RVOL>1.2 + VTrend>1.0', 'rvol_min': 1.2, 'vol_trend_min': 1.0},
    {'label': 'RVOL>0.8 + VTrend>1.0', 'rvol_min': 0.8, 'vol_trend_min': 1.0},
    {'label': 'RVOL>1.0 + VTrend>1.0 + VWAP', 'rvol_min': 1.0, 'vol_trend_min': 1.0, 'use_vwap': True},
    {'label': 'RVOL>0.8 + VTrend>1.0 + VWAP', 'rvol_min': 0.8, 'vol_trend_min': 1.0, 'use_vwap': True},
    {'label': 'VTrend>1.0 + VWAP',     'vol_trend_min': 1.0, 'use_vwap': True},
    {'label': 'VTrend>1.1 + VWAP',     'vol_trend_min': 1.1, 'use_vwap': True},
]
print()
print('%-35s | %10s | %8s | %6s | %6s | %4s' % (
    'Combo', 'P&L', 'PF', 'WR%', 'Trades', 'SL'))
print('-' * 80)
for combo in combos:
    p = dict(base)
    for k, v in combo.items():
        if k != 'label':
            p[k] = v
    r = bt(qqq, p, rvol_20, vol_trend, vwap, vol)
    print('%-35s | $%9.2f | %8.3f | %5.1f%% | %6d | %4d' % (
        combo['label'], r['pnl'], r['pf'], r['wr'], r['trades'], r['sl_count']))

# ═══════════════════════════════════════════════════════════════
# Test 5: RVOL with 10-bar lookback instead of 20
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 90)
print('TEST 5: RVOL with 10-bar lookback')
print('=' * 90)
print()
print('%-8s | %10s | %8s | %6s | %6s | %4s' % (
    'RVOL>', 'P&L', 'PF', 'WR%', 'Trades', 'SL'))
print('-' * 60)
for rv in [0.8, 1.0, 1.2, 1.5, 2.0]:
    p = dict(base)
    p['rvol_min'] = rv
    r = bt(qqq, p, rvol_10, vol_trend, vwap, vol)
    print('%-8.1f | $%9.2f | %8.3f | %5.1f%% | %6d | %4d' % (
        rv, r['pnl'], r['pf'], r['wr'], r['trades'], r['sl_count']))
