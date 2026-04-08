"""
Test daily bar trend filters on pure runner strategy.
Uses daily SMA, volume trend, and ADX as pre-market filters.
Best pure runner: trail=0.50, BE=0.20, SL=0.50, spike=0.35
"""
import yfinance as yf
import pandas as pd
import numpy as np
from itertools import product

# Download data
print("Downloading QQQ 5m data...")
qqq_5m = yf.download('QQQ', period='60d', interval='5m', auto_adjust=True)
if isinstance(qqq_5m.columns, pd.MultiIndex):
    qqq_5m.columns = qqq_5m.columns.get_level_values(0)
qqq_5m.index = qqq_5m.index.tz_convert('America/New_York')
print('5m bars: %d' % len(qqq_5m))

print("Downloading QQQ daily data...")
qqq_daily = yf.download('QQQ', period='1y', interval='1d', auto_adjust=True)
if isinstance(qqq_daily.columns, pd.MultiIndex):
    qqq_daily.columns = qqq_daily.columns.get_level_values(0)
print('Daily bars: %d' % len(qqq_daily))

# ─── Compute daily indicators ───
daily_close = qqq_daily['Close'].values.astype(float)
daily_high = qqq_daily['High'].values.astype(float)
daily_low = qqq_daily['Low'].values.astype(float)
daily_vol = qqq_daily['Volume'].values.astype(float)
daily_dates = qqq_daily.index.date if hasattr(qqq_daily.index, 'date') else pd.to_datetime(qqq_daily.index).date

# SMA
def sma(arr, n):
    return pd.Series(arr).rolling(n).mean().values

daily_sma20 = sma(daily_close, 20)
daily_sma50 = sma(daily_close, 50)
daily_sma200 = sma(daily_close, 200)

# Volume trend: SMA(5,vol) / SMA(20,vol)
daily_vol_sma5 = sma(daily_vol, 5)
daily_vol_sma20 = sma(daily_vol, 20)
daily_vol_trend = daily_vol_sma5 / np.where(daily_vol_sma20 > 0, daily_vol_sma20, 1)

# ADX(14)
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

# Daily return (for momentum)
daily_ret = np.zeros(len(daily_close))
for i in range(1, len(daily_close)):
    daily_ret[i] = (daily_close[i] - daily_close[i-1]) / daily_close[i-1]

# 5-day momentum
daily_ret5 = np.zeros(len(daily_close))
for i in range(5, len(daily_close)):
    daily_ret5[i] = (daily_close[i] - daily_close[i-5]) / daily_close[i-5]

# Build daily lookup: date -> indicators
daily_lookup = {}
for i in range(len(qqq_daily)):
    d = daily_dates[i]
    daily_lookup[d] = {
        'close': daily_close[i],
        'sma20': daily_sma20[i],
        'sma50': daily_sma50[i],
        'sma200': daily_sma200[i],
        'vol_trend': daily_vol_trend[i],
        'adx': daily_adx[i],
        'ret': daily_ret[i],
        'ret5': daily_ret5[i],
    }

print('Daily lookup: %d days' % len(daily_lookup))
print()


def get_prev_daily(date, daily_lookup):
    """Get previous trading day's daily indicators for a given date."""
    import datetime
    d = date
    for _ in range(5):
        d = d - datetime.timedelta(days=1)
        if d in daily_lookup:
            return daily_lookup[d]
    return None


def bt(df, p, daily_lookup):
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

    sma_filter = p.get('sma_filter', None)  # 20, 50, or 200
    sma_direction = p.get('sma_direction', True)  # True = directional, False = just above
    adx_min = p.get('adx_min', 0)
    vol_trend_min = p.get('vol_trend_min', 0)
    ret5_filter = p.get('ret5_filter', False)

    cached_date = None
    cached_daily = None
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

        # Daily filter: check once per day
        today = t.date()
        if today != cached_date:
            cached_date = today
            cached_daily = get_prev_daily(today, daily_lookup)
            daily_long_ok = True
            daily_short_ok = True

            if cached_daily is not None:
                # SMA filter
                if sma_filter is not None:
                    sma_key = 'sma%d' % sma_filter
                    sma_val = cached_daily.get(sma_key, np.nan)
                    if not np.isnan(sma_val):
                        if sma_direction:
                            # Directional: above SMA = long only, below = short only
                            if cached_daily['close'] <= sma_val:
                                daily_long_ok = False
                            if cached_daily['close'] >= sma_val:
                                daily_short_ok = False
                        else:
                            # Non-directional: must be above SMA for any trade
                            if cached_daily['close'] <= sma_val:
                                daily_long_ok = False
                                daily_short_ok = False

                # ADX filter
                if adx_min > 0:
                    adx_val = cached_daily.get('adx', 0)
                    if adx_val < adx_min:
                        daily_long_ok = False
                        daily_short_ok = False

                # Volume trend filter
                if vol_trend_min > 0:
                    vt = cached_daily.get('vol_trend', 0)
                    if vt < vol_trend_min:
                        daily_long_ok = False
                        daily_short_ok = False

                # 5-day momentum filter
                if ret5_filter:
                    r5 = cached_daily.get('ret5', 0)
                    if r5 <= 0:
                        daily_long_ok = False
                    if r5 >= 0:
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


base = {
    'ma_fast': 5, 'ma_mid': 10, 'ma_slow': 20,
    'acc_lookback': 10, 'acc_min_bars': 6, 'max_bar_size': 0.35,
    'sl_pct': 0.50, 'be_trigger': 0.20, 'trail': 0.50,
    'cooldown': 5,
    'sess_start_h': 9, 'sess_start_m': 30, 'sess_end_h': 11, 'sess_end_m': 0,
}

fmt = '%-40s | %10s | %8s | %6s | %6s | %4s'
hdr = fmt % ('Filter', 'P&L', 'PF', 'WR%', 'Trades', 'SL')
sep = '-' * 85

# Baseline
print()
print('=' * 85)
print('BASELINE (no daily filter)')
print('=' * 85)
r = bt(qqq_5m, base, daily_lookup)
print(fmt % ('No filter', '$%.2f' % r['pnl'], '%.3f' % r['pf'],
    '%.1f%%' % r['wr'], str(r['trades']), str(r['sl_count'])))

# ═══════════════════════════════════════════════════════════════
# Test 1: SMA Gate (directional)
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 85)
print('TEST 1: DAILY SMA GATE (directional: above=long only, below=short only)')
print('=' * 85)
print(hdr)
print(sep)
for sma_len in [20, 50, 200]:
    p = dict(base)
    p['sma_filter'] = sma_len
    p['sma_direction'] = True
    r = bt(qqq_5m, p, daily_lookup)
    print(fmt % ('SMA(%d) directional' % sma_len, '$%.2f' % r['pnl'], '%.3f' % r['pf'],
        '%.1f%%' % r['wr'], str(r['trades']), str(r['sl_count'])))

# ═══════════════════════════════════════════════════════════════
# Test 2: ADX Filter
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 85)
print('TEST 2: DAILY ADX FILTER (only trade when ADX > threshold)')
print('=' * 85)
print(hdr)
print(sep)
for adx_thresh in [15, 20, 25, 30, 35]:
    p = dict(base)
    p['adx_min'] = adx_thresh
    r = bt(qqq_5m, p, daily_lookup)
    print(fmt % ('ADX > %d' % adx_thresh, '$%.2f' % r['pnl'], '%.3f' % r['pf'],
        '%.1f%%' % r['wr'], str(r['trades']), str(r['sl_count'])))

# ═══════════════════════════════════════════════════════════════
# Test 3: Volume Trend Filter
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 85)
print('TEST 3: DAILY VOLUME TREND (SMA5/SMA20 of daily volume)')
print('=' * 85)
print(hdr)
print(sep)
for vt in [0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.3]:
    p = dict(base)
    p['vol_trend_min'] = vt
    r = bt(qqq_5m, p, daily_lookup)
    print(fmt % ('VolTrend > %.2f' % vt, '$%.2f' % r['pnl'], '%.3f' % r['pf'],
        '%.1f%%' % r['wr'], str(r['trades']), str(r['sl_count'])))

# ═══════════════════════════════════════════════════════════════
# Test 4: 5-Day Momentum Filter
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 85)
print('TEST 4: 5-DAY MOMENTUM (only long if 5d ret > 0, short if < 0)')
print('=' * 85)
print(hdr)
print(sep)
p = dict(base)
p['ret5_filter'] = True
r = bt(qqq_5m, p, daily_lookup)
print(fmt % ('5d Momentum', '$%.2f' % r['pnl'], '%.3f' % r['pf'],
    '%.1f%%' % r['wr'], str(r['trades']), str(r['sl_count'])))
p['ret5_filter'] = False
r = bt(qqq_5m, p, daily_lookup)
print(fmt % ('No momentum filter', '$%.2f' % r['pnl'], '%.3f' % r['pf'],
    '%.1f%%' % r['wr'], str(r['trades']), str(r['sl_count'])))

# ═══════════════════════════════════════════════════════════════
# Test 5: Combinations
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 85)
print('TEST 5: COMBINATIONS')
print('=' * 85)
print(hdr)
print(sep)

combos = [
    {'label': 'SMA(50) + ADX>20',                'sma_filter': 50, 'sma_direction': True, 'adx_min': 20},
    {'label': 'SMA(50) + ADX>25',                'sma_filter': 50, 'sma_direction': True, 'adx_min': 25},
    {'label': 'SMA(50) + VolTrend>1.0',          'sma_filter': 50, 'sma_direction': True, 'vol_trend_min': 1.0},
    {'label': 'SMA(50) + VolTrend>1.05',         'sma_filter': 50, 'sma_direction': True, 'vol_trend_min': 1.05},
    {'label': 'SMA(20) + ADX>20',                'sma_filter': 20, 'sma_direction': True, 'adx_min': 20},
    {'label': 'SMA(20) + ADX>25',                'sma_filter': 20, 'sma_direction': True, 'adx_min': 25},
    {'label': 'SMA(20) + VolTrend>1.0',          'sma_filter': 20, 'sma_direction': True, 'vol_trend_min': 1.0},
    {'label': 'ADX>20 + VolTrend>1.0',           'adx_min': 20, 'vol_trend_min': 1.0},
    {'label': 'ADX>25 + VolTrend>1.0',           'adx_min': 25, 'vol_trend_min': 1.0},
    {'label': 'SMA(50) + ADX>20 + VolTrend>1.0', 'sma_filter': 50, 'sma_direction': True, 'adx_min': 20, 'vol_trend_min': 1.0},
    {'label': 'SMA(50) + ADX>25 + VolTrend>1.0', 'sma_filter': 50, 'sma_direction': True, 'adx_min': 25, 'vol_trend_min': 1.0},
    {'label': 'SMA(20) + ADX>20 + VolTrend>1.0', 'sma_filter': 20, 'sma_direction': True, 'adx_min': 20, 'vol_trend_min': 1.0},
    {'label': 'SMA(50) + 5d Momentum',           'sma_filter': 50, 'sma_direction': True, 'ret5_filter': True},
    {'label': 'ADX>20 + 5d Momentum',            'adx_min': 20, 'ret5_filter': True},
    {'label': 'SMA(50)+ADX>20+VolTr>1.0+5dMom',  'sma_filter': 50, 'sma_direction': True, 'adx_min': 20, 'vol_trend_min': 1.0, 'ret5_filter': True},
]

for combo in combos:
    p = dict(base)
    for k, v in combo.items():
        if k != 'label':
            p[k] = v
    r = bt(qqq_5m, p, daily_lookup)
    print(fmt % (combo['label'], '$%.2f' % r['pnl'], '%.3f' % r['pf'],
        '%.1f%%' % r['wr'], str(r['trades']), str(r['sl_count'])))
