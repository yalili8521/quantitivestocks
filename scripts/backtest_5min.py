"""
Backtest v3 MA Accumulation strategy on 5-min QQQ bars.
Downloads 60 days of data via yfinance, runs parameter grid search.
"""
import yfinance as yf
import pandas as pd
import numpy as np

# Download 60 days of 5-min QQQ data
print("Downloading QQQ 5m data...")
qqq = yf.download('QQQ', period='60d', interval='5m', auto_adjust=True)
if isinstance(qqq.columns, pd.MultiIndex):
    qqq.columns = qqq.columns.get_level_values(0)
print('QQQ 5m: %d bars, %s to %s' % (len(qqq), qqq.index[0], qqq.index[-1]))
qqq.index = qqq.index.tz_convert('America/New_York')


def backtest(df, p):
    """Full v3 strategy backtester. Returns dict with stats."""
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
    tp_stage = 0
    peak_price = 0.0
    is_trailing = False
    remaining_pct = 1.0
    bars_since_exit = 999
    trade_pnl = 0.0
    trade_entry_eq = 0.0
    trades = []

    sess_start = p.get('sess_start_h', 9) * 60 + p.get('sess_start_m', 30)
    sess_end = p.get('sess_end_h', 11) * 60 + p.get('sess_end_m', 0)
    prev_long_entry = False
    prev_short_entry = False

    start_bar = max(p['ma_slow'], p['acc_lookback'] + 1)

    for i in range(start_bar, n):
        t = times[i]
        t_min = t.hour * 60 + t.minute
        in_session = t_min >= sess_start and t_min < sess_end
        force_close = t_min >= sess_end and t_min < sess_end + 10

        c = close[i]
        if np.isnan(ma_f[i]) or np.isnan(ma_m[i]) or np.isnan(ma_s[i]):
            continue

        bull_stack = ma_f[i] > ma_m[i] and ma_m[i] > ma_s[i]
        bear_stack = ma_f[i] < ma_m[i] and ma_m[i] < ma_s[i]
        price_above = c > ma_f[i]
        price_below = c < ma_f[i]

        # Accumulation
        bars_rising = 0
        bars_falling = 0
        no_spike = True
        for j in range(p['acc_lookback']):
            idx = i - j
            idx_prev = i - j - 1
            if idx_prev < 0:
                break
            if close[idx] > close[idx_prev]:
                bars_rising += 1
            if close[idx] < close[idx_prev]:
                bars_falling += 1
            bar_move = abs(close[idx] - opn[idx]) / opn[idx] * 100 if opn[idx] > 0 else 0
            if bar_move > p['max_bar_size']:
                no_spike = False

        accumulating = bars_rising >= p['acc_min_bars'] and no_spike
        distributing = bars_falling >= p['acc_min_bars'] and no_spike
        mas_rising = ma_f[i] > ma_f[i-1] and ma_m[i] > ma_m[i-1]
        mas_falling = ma_f[i] < ma_f[i-1] and ma_m[i] < ma_m[i-1]
        cooled_off = bars_since_exit >= p['cooldown']

        long_cond = bull_stack and price_above and accumulating and mas_rising and in_session and cooled_off
        short_cond = bear_stack and price_below and distributing and mas_falling and in_session and cooled_off

        # Trigger = new signal (wasn't true previous bar)
        long_trigger = long_cond and not prev_long_entry
        short_trigger = short_cond and not prev_short_entry
        prev_long_entry = long_cond
        prev_short_entry = short_cond

        # ─── Exit logic ───
        if position != 0:
            bars_held = i - entry_bar

            if position == 1 and c > peak_price:
                peak_price = c
            if position == -1 and c < peak_price:
                peak_price = c

            if is_trailing:
                if position == 1:
                    ns = peak_price * (1 - p['trail_pct'] / 100)
                    if ns > stop_level:
                        stop_level = ns
                elif position == -1:
                    ns = peak_price * (1 + p['trail_pct'] / 100)
                    if ns < stop_level:
                        stop_level = ns

            # TP levels
            tp1_q = p['tp1_qty'] / 100.0
            tp2_q = p['tp2_qty'] / 100.0
            tp3_q = p['tp3_qty'] / 100.0
            tp4_q = p['tp4_qty'] / 100.0

            if position == 1:
                if tp_stage < 1 and c >= entry_price * (1 + p['tp1_pct'] / 100):
                    tp_stage = 1
                    is_trailing = True
                    stop_level = peak_price * (1 - p['trail_pct'] / 100)
                    if stop_level < entry_price:
                        stop_level = entry_price
                    trade_pnl += (c - entry_price) / entry_price * tp1_q * trade_entry_eq
                    remaining_pct -= tp1_q
                if tp_stage == 1 and c >= entry_price * (1 + p['tp2_pct'] / 100):
                    tp_stage = 2
                    trade_pnl += (c - entry_price) / entry_price * tp2_q * trade_entry_eq
                    remaining_pct -= tp2_q
                if tp_stage == 2 and c >= entry_price * (1 + p['tp3_pct'] / 100):
                    tp_stage = 3
                    trade_pnl += (c - entry_price) / entry_price * tp3_q * trade_entry_eq
                    remaining_pct -= tp3_q
                if tp_stage == 3 and c >= entry_price * (1 + p['tp4_pct'] / 100):
                    tp_stage = 4
                    trade_pnl += (c - entry_price) / entry_price * tp4_q * trade_entry_eq
                    remaining_pct -= tp4_q

            if position == -1:
                if tp_stage < 1 and c <= entry_price * (1 - p['tp1_pct'] / 100):
                    tp_stage = 1
                    is_trailing = True
                    stop_level = peak_price * (1 + p['trail_pct'] / 100)
                    if stop_level > entry_price:
                        stop_level = entry_price
                    trade_pnl += (entry_price - c) / entry_price * tp1_q * trade_entry_eq
                    remaining_pct -= tp1_q
                if tp_stage == 1 and c <= entry_price * (1 - p['tp2_pct'] / 100):
                    tp_stage = 2
                    trade_pnl += (entry_price - c) / entry_price * tp2_q * trade_entry_eq
                    remaining_pct -= tp2_q
                if tp_stage == 2 and c <= entry_price * (1 - p['tp3_pct'] / 100):
                    tp_stage = 3
                    trade_pnl += (entry_price - c) / entry_price * tp3_q * trade_entry_eq
                    remaining_pct -= tp3_q
                if tp_stage == 3 and c <= entry_price * (1 - p['tp4_pct'] / 100):
                    tp_stage = 4
                    trade_pnl += (entry_price - c) / entry_price * tp4_q * trade_entry_eq
                    remaining_pct -= tp4_q

            stopped = (position == 1 and c <= stop_level) or (position == -1 and c >= stop_level)
            time_exit = bars_held >= p['max_hold'] and tp_stage < 4

            if stopped or time_exit or force_close:
                if position == 1:
                    trade_pnl += (c - entry_price) / entry_price * remaining_pct * trade_entry_eq
                else:
                    trade_pnl += (entry_price - c) / entry_price * remaining_pct * trade_entry_eq

                equity += trade_pnl
                etype = 'Stop' if stopped else 'Time' if time_exit else 'Sess'
                trades.append({
                    'entry': times[entry_bar], 'exit': t,
                    'dir': 'L' if position == 1 else 'S',
                    'entry_p': entry_price, 'exit_p': c,
                    'pnl': trade_pnl, 'tp': tp_stage, 'exit_type': etype,
                    'bars': bars_held
                })
                position = 0
                bars_since_exit = 0
                tp_stage = 0
                is_trailing = False
                remaining_pct = 1.0
                trade_pnl = 0.0
                continue

        if position == 0:
            bars_since_exit += 1

        # ─── Entry ───
        if position == 0 and long_trigger:
            position = 1
            entry_price = c
            entry_bar = i
            stop_level = c * (1 - p['sl_pct'] / 100)
            tp_stage = 0
            peak_price = c
            is_trailing = False
            remaining_pct = 1.0
            trade_pnl = 0.0
            trade_entry_eq = equity * 0.95
            bars_since_exit = 0
        elif position == 0 and short_trigger:
            position = -1
            entry_price = c
            entry_bar = i
            stop_level = c * (1 + p['sl_pct'] / 100)
            tp_stage = 0
            peak_price = c
            is_trailing = False
            remaining_pct = 1.0
            trade_pnl = 0.0
            trade_entry_eq = equity * 0.95
            bars_since_exit = 0

    # Close open position
    if position != 0:
        c = close[-1]
        if position == 1:
            trade_pnl += (c - entry_price) / entry_price * remaining_pct * trade_entry_eq
        else:
            trade_pnl += (entry_price - c) / entry_price * remaining_pct * trade_entry_eq
        equity += trade_pnl
        trades.append({
            'entry': times[entry_bar], 'exit': times[-1],
            'dir': 'L' if position == 1 else 'S',
            'entry_p': entry_price, 'exit_p': c,
            'pnl': trade_pnl, 'tp': tp_stage, 'exit_type': 'EOD',
            'bars': len(close) - 1 - entry_bar
        })

    nt = len(trades)
    if nt == 0:
        return {'pnl': 0, 'pf': 0, 'wr': 0, 'trades': 0, 'equity': equity, 'trade_list': []}

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    gp = sum(t['pnl'] for t in wins) if wins else 0
    gl = abs(sum(t['pnl'] for t in losses)) if losses else 0.001
    pf = gp / gl if gl > 0 else 999

    return {
        'pnl': equity - 5000,
        'pf': pf,
        'wr': len(wins) / nt * 100,
        'trades': nt,
        'equity': equity,
        'trade_list': trades
    }


# ─── V3 baseline (1-min settings on 5-min bars) ───
v3_base = {
    'ma_fast': 5, 'ma_mid': 10, 'ma_slow': 20,
    'acc_lookback': 10, 'acc_min_bars': 6, 'max_bar_size': 0.15,
    'tp1_pct': 0.08, 'tp2_pct': 0.20, 'tp3_pct': 0.35, 'tp4_pct': 0.50,
    'tp1_qty': 15, 'tp2_qty': 25, 'tp3_qty': 25, 'tp4_qty': 15,
    'sl_pct': 0.20, 'trail_pct': 0.25,
    'max_hold': 30, 'cooldown': 5,
    'sess_start_h': 9, 'sess_start_m': 30, 'sess_end_h': 11, 'sess_end_m': 0
}

print('\n' + '='*60)
print('V3 BASELINE (1-min settings) on 5-min')
print('='*60)
r = backtest(qqq, v3_base)
print('P&L: $%.2f | PF: %.3f | WR: %.1f%% | Trades: %d' % (r['pnl'], r['pf'], r['wr'], r['trades']))

# ─── Single-parameter sensitivity ───
param_grid = {
    'acc_lookback':  [4, 6, 8, 10, 12],
    'acc_min_bars':  [3, 4, 5, 6],
    'max_bar_size':  [0.15, 0.25, 0.40, 0.60, 1.0],
    'tp1_pct':       [0.10, 0.15, 0.20, 0.30, 0.40],
    'tp2_pct':       [0.30, 0.40, 0.50, 0.70, 1.00],
    'tp3_pct':       [0.50, 0.70, 1.00, 1.50],
    'tp4_pct':       [0.80, 1.00, 1.50, 2.00],
    'sl_pct':        [0.20, 0.30, 0.40, 0.60, 0.80],
    'trail_pct':     [0.20, 0.25, 0.35, 0.50, 0.75, 1.00],
    'max_hold':      [6, 10, 15, 20, 30, 50],
    'cooldown':      [1, 3, 5, 8],
    'sess_end_h':    [11, 12, 13, 16],
}

print('\n' + '='*60)
print('SINGLE-PARAMETER SENSITIVITY (best per param)')
print('='*60)
best_params = {}
for param_name, values in param_grid.items():
    best_val = None
    best_pnl = -9999
    best_info = ""
    all_results = []
    for val in values:
        p = v3_base.copy()
        if param_name == 'acc_min_bars' and val > p['acc_lookback']:
            continue
        p[param_name] = val
        r = backtest(qqq, p)
        all_results.append((val, r['pnl'], r['pf'], r['wr'], r['trades']))
        if r['pnl'] > best_pnl:
            best_pnl = r['pnl']
            best_val = val
            best_info = 'PF:%.3f WR:%.1f%% T:%d' % (r['pf'], r['wr'], r['trades'])
    best_params[param_name] = best_val
    # Show all values for this param
    print('\n  %s (v3 default: %s):' % (param_name, v3_base[param_name]))
    for val, pnl, pf, wr, t in all_results:
        marker = ' <-- BEST' if val == best_val else ''
        print('    %-8s -> P&L: $%7.2f | PF: %6.3f | WR: %5.1f%% | T: %d%s' % (
            str(val), pnl, pf, wr, t, marker))

# ─── Build optimized config ───
optimized = v3_base.copy()
for k, v in best_params.items():
    optimized[k] = v

print('\n' + '='*60)
print('OPTIMIZED COMBINED (all best single-params)')
print('='*60)
r_opt = backtest(qqq, optimized)
print('P&L: $%.2f | PF: %.3f | WR: %.1f%% | Trades: %d' % (r_opt['pnl'], r_opt['pf'], r_opt['wr'], r_opt['trades']))
print('\nChanged params:')
for k in optimized:
    if optimized[k] != v3_base[k]:
        print('  %s: %s -> %s' % (k, v3_base[k], optimized[k]))

# ─── Session window comparison with optimized params ───
print('\n' + '='*60)
print('SESSION WINDOW (with optimized params)')
print('='*60)
for end_h, end_m in [(11, 0), (12, 0), (13, 0), (15, 30), (16, 0)]:
    p = optimized.copy()
    p['sess_end_h'] = end_h
    p['sess_end_m'] = end_m
    r = backtest(qqq, p)
    print('  9:30-%d:%02d -> P&L: $%7.2f | PF: %6.3f | WR: %5.1f%% | T: %d' % (
        end_h, end_m, r['pnl'], r['pf'], r['wr'], r['trades']))

# ─── Trade log for optimized ───
print('\n' + '='*60)
print('TRADE LOG (Optimized)')
print('='*60)
for i, t in enumerate(r_opt['trade_list'][:40], 1):
    print('%2d. %s %s -> %s | $%.2f->$%.2f | $%+.2f | TP%d | %s | %db' % (
        i, t['dir'], str(t['entry'])[:16], str(t['exit'])[:16],
        t['entry_p'], t['exit_p'], t['pnl'], t['tp'], t['exit_type'], t['bars']))
