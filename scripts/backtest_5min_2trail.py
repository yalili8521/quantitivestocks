"""
Grid search Trail 1 / Trail 2 percentages for v3-5m 2-level trail runner.
Downloads 60 days QQQ 5-min data, tests all combos, finds the sweet spot.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from itertools import product

print("Downloading QQQ 5m data...")
qqq = yf.download('QQQ', period='60d', interval='5m', auto_adjust=True)
if isinstance(qqq.columns, pd.MultiIndex):
    qqq.columns = qqq.columns.get_level_values(0)
print('QQQ 5m: %d bars, %s to %s' % (len(qqq), qqq.index[0], qqq.index[-1]))
qqq.index = qqq.index.tz_convert('America/New_York')


def backtest_2trail(df, p):
    """2-level trail stop backtester matching Pine v3-5m logic."""
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
    force_close_start = 16 * 60  # 16:00 ET
    force_close_end = 16 * 60 + 10

    prev_long_entry = False
    prev_short_entry = False
    start_bar = max(p['ma_slow'], p['acc_lookback'] + 1)

    for i in range(start_bar, n):
        t = times[i]
        t_min = t.hour * 60 + t.minute
        in_session = t_min >= sess_start and t_min < sess_end
        force_close = t_min >= force_close_start and t_min < force_close_end

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

        long_trigger = long_cond and not prev_long_entry
        short_trigger = short_cond and not prev_short_entry
        prev_long_entry = long_cond
        prev_short_entry = short_cond

        # ─── Exit logic ───
        if position != 0:
            # Track peak
            if position == 1 and c > peak_price:
                peak_price = c
            if position == -1 and c < peak_price:
                peak_price = c

            # BE trigger: activate both trails
            if not is_trailing:
                if position == 1 and c >= entry_price * (1 + p['be_trigger'] / 100):
                    is_trailing = True
                    floor = entry_price * (1 + 0.05 / 100)
                    stop1_level = max(floor, peak_price * (1 - p['trail1'] / 100))
                    stop2_level = max(floor, peak_price * (1 - p['trail2'] / 100))
                elif position == -1 and c <= entry_price * (1 - p['be_trigger'] / 100):
                    is_trailing = True
                    floor = entry_price * (1 - 0.05 / 100)
                    stop1_level = min(floor, peak_price * (1 + p['trail1'] / 100))
                    stop2_level = min(floor, peak_price * (1 + p['trail2'] / 100))

            # Update trailing stops
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

            exit_reason = None
            exit_pct = 0.0  # fraction of position to close

            # Fixed SL (before BE trigger)
            if not is_trailing:
                if (position == 1 and c <= stop_level) or (position == -1 and c >= stop_level):
                    exit_reason = 'SL'
                    exit_pct = 1.0

            # Trail 1 (tight) — exit first half
            if is_trailing and not trail1_exited:
                if (position == 1 and c <= stop1_level) or (position == -1 and c >= stop1_level):
                    trail1_exited = True
                    t1_frac = p['trail1_close'] / 100.0
                    # Book P&L for trail 1 portion
                    if position == 1:
                        trade_pnl += (c - entry_price) / entry_price * t1_frac * remaining_pct * trade_entry_eq
                    else:
                        trade_pnl += (entry_price - c) / entry_price * t1_frac * remaining_pct * trade_entry_eq
                    remaining_pct *= (1 - t1_frac)

            # Trail 2 (wide) — exit rest
            if is_trailing and trail1_exited:
                if (position == 1 and c <= stop2_level) or (position == -1 and c >= stop2_level):
                    exit_reason = 'T2'
                    exit_pct = 1.0

            # Safety: trail2 hit before trail1 (shouldn't happen, but just in case)
            if is_trailing and not trail1_exited:
                if (position == 1 and c <= stop2_level) or (position == -1 and c >= stop2_level):
                    exit_reason = 'TS'
                    exit_pct = 1.0

            # Force close at market close
            if force_close:
                exit_reason = 'MC'
                exit_pct = 1.0

            if exit_reason and exit_pct >= 1.0:
                # Close remaining position
                if position == 1:
                    trade_pnl += (c - entry_price) / entry_price * remaining_pct * trade_entry_eq
                else:
                    trade_pnl += (entry_price - c) / entry_price * remaining_pct * trade_entry_eq

                equity += trade_pnl
                trades.append({
                    'entry': times[entry_bar], 'exit': t,
                    'dir': 'L' if position == 1 else 'S',
                    'entry_p': entry_price, 'exit_p': c,
                    'pnl': trade_pnl, 'exit_type': exit_reason,
                    'bars': i - entry_bar,
                    'trail1_hit': trail1_exited
                })
                position = 0
                bars_since_exit = 0
                is_trailing = False
                trail1_exited = False
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
            peak_price = c
            is_trailing = False
            trail1_exited = False
            remaining_pct = 1.0
            trade_pnl = 0.0
            trade_entry_eq = equity * 0.95
            bars_since_exit = 0
        elif position == 0 and short_trigger:
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

    # Close open position at end
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
            'pnl': trade_pnl, 'exit_type': 'EOD',
            'bars': len(close) - 1 - entry_bar,
            'trail1_hit': trail1_exited
        })

    nt = len(trades)
    if nt == 0:
        return {'pnl': 0, 'pf': 0, 'wr': 0, 'trades': 0, 'equity': equity, 'avg_win': 0, 'avg_loss': 0, 'trade_list': []}

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    gp = sum(t['pnl'] for t in wins) if wins else 0
    gl = abs(sum(t['pnl'] for t in losses)) if losses else 0.001
    pf = gp / gl if gl > 0 else 999
    avg_win = gp / len(wins) if wins else 0
    avg_loss = gl / len(losses) if losses else 0

    return {
        'pnl': equity - 5000,
        'pf': pf,
        'wr': len(wins) / nt * 100,
        'trades': nt,
        'equity': equity,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'trade_list': trades
    }


# ─── Base params (matching Pine v3-5m) ───
base = {
    'ma_fast': 5, 'ma_mid': 10, 'ma_slow': 20,
    'acc_lookback': 10, 'acc_min_bars': 6, 'max_bar_size': 0.35,
    'sl_pct': 0.50, 'be_trigger': 0.20,
    'trail1': 0.35, 'trail2': 0.75, 'trail1_close': 50,
    'cooldown': 5,
    'sess_start_h': 9, 'sess_start_m': 30,
    'sess_end_h': 11, 'sess_end_m': 0,
}

# ─── Current baseline ───
print('\n' + '='*70)
print('CURRENT BASELINE (Trail1=0.35%, Trail2=0.75%)')
print('='*70)
r = backtest_2trail(qqq, base)
print('P&L: $%.2f | PF: %.3f | WR: %.1f%% | Trades: %d | AvgW: $%.2f | AvgL: $%.2f' % (
    r['pnl'], r['pf'], r['wr'], r['trades'], r['avg_win'], r['avg_loss']))

# ─── Grid search: Trail1 x Trail2 ───
print('\n' + '='*70)
print('GRID SEARCH: Trail 1 % x Trail 2 %')
print('='*70)

trail1_vals = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
trail2_vals = [0.40, 0.50, 0.60, 0.75, 0.90, 1.00, 1.25, 1.50]

results = []
for t1, t2 in product(trail1_vals, trail2_vals):
    if t1 >= t2:  # trail1 must be tighter than trail2
        continue
    p = base.copy()
    p['trail1'] = t1
    p['trail2'] = t2
    r = backtest_2trail(qqq, p)
    results.append((t1, t2, r['pnl'], r['pf'], r['wr'], r['trades'], r['avg_win'], r['avg_loss']))

# Sort by P&L
results.sort(key=lambda x: x[2], reverse=True)

print('\n%-8s %-8s | %10s | %8s | %6s | %6s | %8s | %8s | %s' % (
    'Trail1', 'Trail2', 'P&L', 'PF', 'WR%', 'Trades', 'AvgWin', 'AvgLoss', 'Ratio'))
print('-' * 95)
for t1, t2, pnl, pf, wr, trades, aw, al in results:
    ratio = t2 / t1
    print('%-8.2f %-8.2f | $%9.2f | %8.3f | %5.1f%% | %6d | $%7.2f | $%7.2f | %.1fx' % (
        t1, t2, pnl, pf, wr, trades, aw, al, ratio))

# ─── Top 5 by P&L ───
print('\n' + '='*70)
print('TOP 5 BY P&L')
print('='*70)
for i, (t1, t2, pnl, pf, wr, trades, aw, al) in enumerate(results[:5], 1):
    ratio = t2 / t1
    print('#%d Trail1=%.2f%% Trail2=%.2f%% (%.1fx ratio) -> P&L: $%.2f | PF: %.3f | WR: %.1f%% | T: %d' % (
        i, t1, t2, ratio, pnl, pf, wr, trades))

# ─── Also search BE trigger + SL combos with best trail pair ───
best_t1, best_t2 = results[0][0], results[0][1]
print('\n' + '='*70)
print('SENSITIVITY: BE Trigger + SL (with best trails: %.2f/%.2f)' % (best_t1, best_t2))
print('='*70)

be_vals = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
sl_vals = [0.30, 0.40, 0.50, 0.60, 0.75, 1.00]

results2 = []
for be, sl in product(be_vals, sl_vals):
    p = base.copy()
    p['trail1'] = best_t1
    p['trail2'] = best_t2
    p['be_trigger'] = be
    p['sl_pct'] = sl
    r = backtest_2trail(qqq, p)
    results2.append((be, sl, r['pnl'], r['pf'], r['wr'], r['trades']))

results2.sort(key=lambda x: x[2], reverse=True)
print('\n%-8s %-8s | %10s | %8s | %6s | %6s' % ('BE%', 'SL%', 'P&L', 'PF', 'WR%', 'Trades'))
print('-' * 60)
for be, sl, pnl, pf, wr, trades in results2[:15]:
    print('%-8.2f %-8.2f | $%9.2f | %8.3f | %5.1f%% | %6d' % (be, sl, pnl, pf, wr, trades))

# ─── Trail1 close % sensitivity (how much to exit on trail 1) ───
print('\n' + '='*70)
print('SENSITIVITY: Trail 1 Close %% (with best trails: %.2f/%.2f)' % (best_t1, best_t2))
print('='*70)

close_vals = [20, 30, 40, 50, 60, 70, 80]
for cv in close_vals:
    p = base.copy()
    p['trail1'] = best_t1
    p['trail2'] = best_t2
    p['trail1_close'] = cv
    r = backtest_2trail(qqq, p)
    marker = ' <-- current' if cv == 50 else ''
    print('  Trail1 exits %d%% -> P&L: $%7.2f | PF: %.3f | WR: %.1f%% | T: %d%s' % (
        cv, r['pnl'], r['pf'], r['wr'], r['trades'], marker))

# ─── Compare vs pure runner (single trail) ───
print('\n' + '='*70)
print('COMPARISON: 2-Level Trail vs Pure Runner (single trail)')
print('='*70)

# Pure runner = trail1_close=0 (never exit on trail1, only trail2)
for trail_pct in [0.35, 0.50, 0.75, 1.00]:
    p = base.copy()
    p['trail1'] = trail_pct
    p['trail2'] = trail_pct
    p['trail1_close'] = 0  # effectively single trail
    r = backtest_2trail(qqq, p)
    print('  Single trail %.2f%% -> P&L: $%7.2f | PF: %.3f | WR: %.1f%% | T: %d' % (
        trail_pct, r['pnl'], r['pf'], r['wr'], r['trades']))

# Best 2-level
p = base.copy()
p['trail1'] = best_t1
p['trail2'] = best_t2
r = backtest_2trail(qqq, p)
print('  2-Level %.2f/%.2f%%  -> P&L: $%7.2f | PF: %.3f | WR: %.1f%% | T: %d' % (
    best_t1, best_t2, r['pnl'], r['pf'], r['wr'], r['trades']))
