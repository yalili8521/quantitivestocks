"""
Compare pure runner strategy on TQQQ (3x leveraged QQQ) with daily filters.
4 variants: no filter, ADX>20, ADX>20+VolTrend>1.0, SMA(50)+ADX>20
Uses QQQ daily bars for daily indicators (more liquid/reliable than TQQQ daily).
"""
import yfinance as yf
import pandas as pd
import numpy as np
import datetime

# Download data
print("Downloading TQQQ 5m data...")
tqqq_5m = yf.download('TQQQ', period='60d', interval='5m', auto_adjust=True)
if isinstance(tqqq_5m.columns, pd.MultiIndex):
    tqqq_5m.columns = tqqq_5m.columns.get_level_values(0)
tqqq_5m.index = tqqq_5m.index.tz_convert('America/New_York')
print('TQQQ 5m bars: %d, %s to %s' % (len(tqqq_5m), tqqq_5m.index[0], tqqq_5m.index[-1]))

print("Downloading QQQ daily data (for daily indicators)...")
qqq_daily = yf.download('QQQ', period='1y', interval='1d', auto_adjust=True)
if isinstance(qqq_daily.columns, pd.MultiIndex):
    qqq_daily.columns = qqq_daily.columns.get_level_values(0)
print('QQQ daily bars: %d' % len(qqq_daily))

# Also download QQQ 5m for side-by-side comparison
print("Downloading QQQ 5m data (for comparison)...")
qqq_5m = yf.download('QQQ', period='60d', interval='5m', auto_adjust=True)
if isinstance(qqq_5m.columns, pd.MultiIndex):
    qqq_5m.columns = qqq_5m.columns.get_level_values(0)
qqq_5m.index = qqq_5m.index.tz_convert('America/New_York')
print('QQQ 5m bars: %d' % len(qqq_5m))

# ─── Daily indicators (from QQQ, not TQQQ) ───
daily_close = qqq_daily['Close'].values.astype(float)
daily_high = qqq_daily['High'].values.astype(float)
daily_low = qqq_daily['Low'].values.astype(float)
daily_vol = qqq_daily['Volume'].values.astype(float)
daily_dates = qqq_daily.index.date if hasattr(qqq_daily.index, 'date') else pd.to_datetime(qqq_daily.index).date

def sma(arr, n):
    return pd.Series(arr).rolling(n).mean().values

daily_sma50 = sma(daily_close, 50)

daily_vol_sma5 = sma(daily_vol, 5)
daily_vol_sma20 = sma(daily_vol, 20)
daily_vol_trend = daily_vol_sma5 / np.where(daily_vol_sma20 > 0, daily_vol_sma20, 1)

def calc_adx(high, low, close, period=14):
    n = len(close)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        hl = high[i] - low[i]
        hpc = abs(high[i] - close[i-1])
        lpc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hpc, lpc)
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        plus_dm[i] = up if up > down and up > 0 else 0
        minus_dm[i] = down if down > up and down > 0 else 0
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values / np.where(atr > 0, atr, 1)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values / np.where(atr > 0, atr, 1)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1)
    adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
    return adx

daily_adx = calc_adx(daily_high, daily_low, daily_close, 14)

daily_lookup = {}
for i in range(len(qqq_daily)):
    d = daily_dates[i]
    daily_lookup[d] = {
        'close': daily_close[i],
        'sma50': daily_sma50[i],
        'vol_trend': daily_vol_trend[i],
        'adx': daily_adx[i],
    }

print('Daily lookup: %d days' % len(daily_lookup))


def get_prev_daily(date, daily_lookup):
    d = date
    for _ in range(5):
        d = d - datetime.timedelta(days=1)
        if d in daily_lookup:
            return daily_lookup[d]
    return None


def bt(df, p, daily_lookup, label=""):
    close = df['Close'].values.astype(float)
    opn = df['Open'].values.astype(float)
    times = df.index
    n = len(close)
    ma_f = pd.Series(close).rolling(p['ma_fast']).mean().values
    ma_m = pd.Series(close).rolling(p['ma_mid']).mean().values
    ma_s = pd.Series(close).rolling(p['ma_slow']).mean().values
    equity = 5000.0
    peak_equity = 5000.0
    max_dd = 0.0
    position = 0
    entry_price = 0.0
    stop_level = 0.0
    peak_price = 0.0
    is_trailing = False
    bars_since_exit = 999
    trade_pnl = 0.0
    trade_entry_eq = 0.0
    trades = []
    equity_curve = []
    sess_start = p['sess_start_h'] * 60 + p['sess_start_m']
    sess_end = p['sess_end_h'] * 60 + p['sess_end_m']
    prev_long = False
    prev_short = False
    start_bar = max(p['ma_slow'], p['acc_lookback'] + 1)

    sma_filter = p.get('sma_filter', None)
    adx_min = p.get('adx_min', 0)
    vol_trend_min = p.get('vol_trend_min', 0)

    cached_date = None
    daily_long_ok = True
    daily_short_ok = True

    for i in range(start_bar, n):
        t = times[i]
        t_min = t.hour * 60 + t.minute
        in_session = t_min >= sess_start and t_min < sess_end
        force_close = t_min >= 960 and t_min < 970
        c = close[i]
        if np.isnan(ma_f[i]) or np.isnan(ma_m[i]) or np.isnan(ma_s[i]):
            continue

        today = t.date()
        if today != cached_date:
            cached_date = today
            cd = get_prev_daily(today, daily_lookup)
            daily_long_ok = True
            daily_short_ok = True
            if cd is not None:
                if sma_filter is not None:
                    sma_val = cd.get('sma%d' % sma_filter, np.nan)
                    if not np.isnan(sma_val):
                        if cd['close'] <= sma_val:
                            daily_long_ok = False
                        if cd['close'] >= sma_val:
                            daily_short_ok = False
                if adx_min > 0:
                    if cd.get('adx', 0) < adx_min:
                        daily_long_ok = False
                        daily_short_ok = False
                if vol_trend_min > 0:
                    if cd.get('vol_trend', 0) < vol_trend_min:
                        daily_long_ok = False
                        daily_short_ok = False

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

        lc = bull and pa and acc and mr and in_session and co and daily_long_ok
        sc = bear and pb and dist and mf and in_session and co and daily_short_ok
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
                if equity > peak_equity:
                    peak_equity = equity
                dd = (peak_equity - equity) / peak_equity * 100
                if dd > max_dd:
                    max_dd = dd
                trades.append({'pnl': trade_pnl, 'exit_type': ex})
                equity_curve.append(equity)
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
            stop_level = c * (1 - p['sl_pct'] / 100)
            peak_price = c
            is_trailing = False
            trade_pnl = 0.0
            trade_entry_eq = equity * 0.95
            bars_since_exit = 0
        elif position == 0 and st:
            position = -1
            entry_price = c
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
        equity_curve.append(equity)

    nt = len(trades)
    if nt == 0:
        return {'pnl': 0, 'pf': 0, 'wr': 0, 'trades': 0, 'avg_win': 0, 'avg_loss': 0,
                'sl_count': 0, 'max_dd': 0, 'ret_pct': 0}

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    gp = sum(t['pnl'] for t in wins) if wins else 0
    gl = abs(sum(t['pnl'] for t in losses)) if losses else 0.001
    aw = gp / len(wins) if wins else 0
    al = gl / len(losses) if losses else 0
    sl_c = sum(1 for t in trades if t['exit_type'] == 'SL')

    # Consecutive losses
    max_consec_loss = 0
    curr_consec = 0
    for t in trades:
        if t['pnl'] <= 0:
            curr_consec += 1
            if curr_consec > max_consec_loss:
                max_consec_loss = curr_consec
        else:
            curr_consec = 0

    return {
        'pnl': equity - 5000, 'pf': gp / gl, 'wr': len(wins) / nt * 100,
        'trades': nt, 'avg_win': aw, 'avg_loss': al, 'sl_count': sl_c,
        'max_dd': max_dd, 'ret_pct': (equity - 5000) / 5000 * 100,
        'max_consec_loss': max_consec_loss, 'gross_profit': gp, 'gross_loss': gl
    }


# ─── Strategy params (best pure runner) ───
# Note: for TQQQ, spike filter and trail need to be ~3x wider
# because TQQQ moves ~3x QQQ per bar
base_qqq = {
    'ma_fast': 5, 'ma_mid': 10, 'ma_slow': 20,
    'acc_lookback': 10, 'acc_min_bars': 6, 'max_bar_size': 0.35,
    'sl_pct': 0.50, 'be_trigger': 0.20, 'trail': 0.50,
    'cooldown': 5,
    'sess_start_h': 9, 'sess_start_m': 30, 'sess_end_h': 11, 'sess_end_m': 0,
}

# TQQQ version: scale SL, BE, trail, spike by ~3x
base_tqqq = {
    'ma_fast': 5, 'ma_mid': 10, 'ma_slow': 20,
    'acc_lookback': 10, 'acc_min_bars': 6, 'max_bar_size': 1.05,  # 0.35 * 3
    'sl_pct': 1.50, 'be_trigger': 0.60, 'trail': 1.50,           # 3x
    'cooldown': 5,
    'sess_start_h': 9, 'sess_start_m': 30, 'sess_end_h': 11, 'sess_end_m': 0,
}

# Also test TQQQ with QQQ-same settings (not scaled)
base_tqqq_raw = dict(base_qqq)

filters = [
    {'label': 'No filter'},
    {'label': 'ADX > 20', 'adx_min': 20},
    {'label': 'ADX>20 + VolTrend>1.0', 'adx_min': 20, 'vol_trend_min': 1.0},
    {'label': 'SMA(50) + ADX>20', 'sma_filter': 50, 'adx_min': 20},
]

# ═══════════════════════════════════════════════════════════════
# QQQ (baseline reference)
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 100)
print('QQQ (unleveraged) — Pure Runner')
print('=' * 100)
print()
print('%-30s | %8s | %6s | %6s | %6s | %4s | %6s | %8s | %8s | %s' % (
    'Filter', 'P&L', 'Ret%', 'PF', 'WR%', 'T', 'SL', 'AvgWin', 'AvgLoss', 'MaxDD%'))
print('-' * 110)
for f in filters:
    p = dict(base_qqq)
    for k, v in f.items():
        if k != 'label':
            p[k] = v
    r = bt(qqq_5m, p, daily_lookup)
    print('%-30s | $%7.2f | %5.1f%% | %6.3f | %5.1f%% | %4d | %4d | $%7.2f | $%7.2f | %.1f%%' % (
        f['label'], r['pnl'], r['ret_pct'], r['pf'], r['wr'], r['trades'], r['sl_count'],
        r['avg_win'], r['avg_loss'], r['max_dd']))

# ═══════════════════════════════════════════════════════════════
# TQQQ with 3x-scaled settings
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 100)
print('TQQQ (3x leveraged) — Pure Runner with 3x-SCALED settings (SL=1.50, trail=1.50, spike=1.05)')
print('=' * 100)
print()
print('%-30s | %8s | %6s | %6s | %6s | %4s | %4s | %8s | %8s | %s' % (
    'Filter', 'P&L', 'Ret%', 'PF', 'WR%', 'T', 'SL', 'AvgWin', 'AvgLoss', 'MaxDD%'))
print('-' * 110)
for f in filters:
    p = dict(base_tqqq)
    for k, v in f.items():
        if k != 'label':
            p[k] = v
    r = bt(tqqq_5m, p, daily_lookup)
    print('%-30s | $%7.2f | %5.1f%% | %6.3f | %5.1f%% | %4d | %4d | $%7.2f | $%7.2f | %.1f%%' % (
        f['label'], r['pnl'], r['ret_pct'], r['pf'], r['wr'], r['trades'], r['sl_count'],
        r['avg_win'], r['avg_loss'], r['max_dd']))

# ═══════════════════════════════════════════════════════════════
# TQQQ with QQQ-same settings (NOT scaled)
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 100)
print('TQQQ (3x leveraged) — Pure Runner with QQQ-SAME settings (SL=0.50, trail=0.50, spike=0.35)')
print('=' * 100)
print()
print('%-30s | %8s | %6s | %6s | %6s | %4s | %4s | %8s | %8s | %s' % (
    'Filter', 'P&L', 'Ret%', 'PF', 'WR%', 'T', 'SL', 'AvgWin', 'AvgLoss', 'MaxDD%'))
print('-' * 110)
for f in filters:
    p = dict(base_tqqq_raw)
    for k, v in f.items():
        if k != 'label':
            p[k] = v
    r = bt(tqqq_5m, p, daily_lookup)
    print('%-30s | $%7.2f | %5.1f%% | %6.3f | %5.1f%% | %4d | %4d | $%7.2f | $%7.2f | %.1f%%' % (
        f['label'], r['pnl'], r['ret_pct'], r['pf'], r['wr'], r['trades'], r['sl_count'],
        r['avg_win'], r['avg_loss'], r['max_dd']))

# ═══════════════════════════════════════════════════════════════
# Sensitivity: find best TQQQ trail/SL/BE for each filter
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 100)
print('TQQQ PARAMETER SENSITIVITY — Trail % (with SL=1.50, BE=0.60)')
print('=' * 100)
print()
trail_vals = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00]
print('%-8s | %10s | %8s | %6s | %6s | %4s | %6s' % (
    'Trail%', 'P&L', 'PF', 'WR%', 'Trades', 'SL', 'MaxDD%'))
print('-' * 65)
for tr in trail_vals:
    p = dict(base_tqqq)
    p['trail'] = tr
    r = bt(tqqq_5m, p, daily_lookup)
    print('%-8.2f | $%9.2f | %8.3f | %5.1f%% | %6d | %4d | %5.1f%%' % (
        tr, r['pnl'], r['pf'], r['wr'], r['trades'], r['sl_count'], r['max_dd']))

print()
print('=' * 100)
print('TQQQ PARAMETER SENSITIVITY — SL % (with trail=1.50, BE=0.60)')
print('=' * 100)
print()
sl_vals = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00]
print('%-8s | %10s | %8s | %6s | %6s | %4s | %6s' % (
    'SL%', 'P&L', 'PF', 'WR%', 'Trades', 'SL', 'MaxDD%'))
print('-' * 65)
for sl in sl_vals:
    p = dict(base_tqqq)
    p['sl_pct'] = sl
    r = bt(tqqq_5m, p, daily_lookup)
    print('%-8.2f | $%9.2f | %8.3f | %5.1f%% | %6d | %4d | %5.1f%%' % (
        sl, r['pnl'], r['pf'], r['wr'], r['trades'], r['sl_count'], r['max_dd']))

print()
print('=' * 100)
print('TQQQ PARAMETER SENSITIVITY — BE Trigger % (with trail=1.50, SL=1.50)')
print('=' * 100)
print()
be_vals = [0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.50]
print('%-8s | %10s | %8s | %6s | %6s | %4s | %6s' % (
    'BE%', 'P&L', 'PF', 'WR%', 'Trades', 'SL', 'MaxDD%'))
print('-' * 65)
for be in be_vals:
    p = dict(base_tqqq)
    p['be_trigger'] = be
    r = bt(tqqq_5m, p, daily_lookup)
    print('%-8.2f | $%9.2f | %8.3f | %5.1f%% | %6d | %4d | %5.1f%%' % (
        be, r['pnl'], r['pf'], r['wr'], r['trades'], r['sl_count'], r['max_dd']))
