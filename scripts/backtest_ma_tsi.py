"""
MA Accumulation + TSI Divergence Backtester
Replicates the TQQQ MA+TSI v1 Pine Script logic in Python.
Tests multiple parameter combinations for session window, cooldown, recency.
Uses yfinance 5-min data (max ~60 days).
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product


def fetch_5min_data(sym, days=59):
    """Fetch 5-min data from yfinance."""
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(sym, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                     interval="5m", progress=False)
    if df.empty:
        return None
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if df.index.tz:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    return df


def compute_tsi(close, long_len=13, short_len=7, sig_len=7):
    """Compute TSI and signal line."""
    pc = close.diff()
    smooth1 = pc.ewm(span=long_len, adjust=False).mean()
    double_smooth_pc = smooth1.ewm(span=short_len, adjust=False).mean()

    abs_pc = pc.abs()
    smooth1_abs = abs_pc.ewm(span=long_len, adjust=False).mean()
    double_smooth_apc = smooth1_abs.ewm(span=short_len, adjust=False).mean()

    tsi = 100 * double_smooth_pc / double_smooth_apc.replace(0, np.nan)
    tsi = tsi.fillna(0)
    tsi_signal = tsi.ewm(span=sig_len, adjust=False).mean()
    return tsi, tsi_signal


def find_pivots(series, length=5):
    """Find pivot highs and lows. Returns dict of bar_index -> value."""
    highs = {}
    lows = {}
    vals = series.values
    idx = series.index

    for i in range(length, len(vals) - length):
        # Pivot high
        is_ph = True
        for j in range(1, length + 1):
            if vals[i] <= vals[i - j] or vals[i] <= vals[i + j]:
                is_ph = False
                break
        if is_ph:
            highs[i] = vals[i]

        # Pivot low
        is_pl = True
        for j in range(1, length + 1):
            if vals[i] >= vals[i - j] or vals[i] >= vals[i + j]:
                is_pl = False
                break
        if is_pl:
            lows[i] = vals[i]

    return highs, lows


def backtest_ma_tsi(df, params):
    """Run one backtest with given parameters."""
    p = params
    capital_start = 5000
    capital = capital_start

    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    opn = df['Open'].values
    times = df.index

    n = len(df)

    # MAs
    ma5 = pd.Series(close).rolling(p['ma_fast']).mean().values
    ma10 = pd.Series(close).rolling(p['ma_mid']).mean().values
    ma20 = pd.Series(close).rolling(p['ma_slow']).mean().values

    # TSI
    tsi_series, tsi_sig_series = compute_tsi(
        pd.Series(close, index=df.index),
        p['tsi_long'], p['tsi_short'], p['tsi_sig']
    )
    tsi = tsi_series.values
    tsi_sig = tsi_sig_series.values

    # Find pivots on price (high for pivot highs, low for pivot lows)
    price_pivot_highs, _ = find_pivots(pd.Series(high, index=range(n)), p['pivot_len'])
    _, price_pivot_lows = find_pivots(pd.Series(low, index=range(n)), p['pivot_len'])

    # Find pivots on TSI
    tsi_pivot_highs, _ = find_pivots(pd.Series(tsi, index=range(n)), p['pivot_len'])
    _, tsi_pivot_lows = find_pivots(pd.Series(tsi, index=range(n)), p['pivot_len'])

    # Build divergence arrays
    # Track last two pivot lows and highs
    last_bull_div_bar = -9999
    last_bear_div_bar = -9999

    prev_pl_price, last_pl_price = None, None
    prev_pl_tsi, last_pl_tsi = None, None
    prev_pl_bar, last_pl_bar = 0, 0

    prev_ph_price, last_ph_price = None, None
    prev_ph_tsi, last_ph_tsi = None, None
    prev_ph_bar, last_ph_bar = 0, 0

    bull_div_bars = set()
    bear_div_bars = set()

    for i in range(n):
        # Check pivot low confirmed at i (actual pivot was at i - pivot_len, confirmed now)
        confirm_bar = i - p['pivot_len']
        if confirm_bar in price_pivot_lows:
            prev_pl_price = last_pl_price
            prev_pl_tsi = last_pl_tsi
            prev_pl_bar = last_pl_bar
            last_pl_price = price_pivot_lows[confirm_bar]
            last_pl_tsi = tsi_pivot_lows.get(confirm_bar, tsi[confirm_bar])
            last_pl_bar = confirm_bar

            # Check hidden bullish: price higher low + TSI lower low
            if (prev_pl_price is not None and prev_pl_tsi is not None and
                last_pl_price > prev_pl_price and last_pl_tsi < prev_pl_tsi and
                (last_pl_bar - prev_pl_bar) <= p['max_pivot_dist']):
                bull_div_bars.add(i)
                last_bull_div_bar = i

        if confirm_bar in price_pivot_highs:
            prev_ph_price = last_ph_price
            prev_ph_tsi = last_ph_tsi
            prev_ph_bar = last_ph_bar
            last_ph_price = price_pivot_highs[confirm_bar]
            last_ph_tsi = tsi_pivot_highs.get(confirm_bar, tsi[confirm_bar])
            last_ph_bar = confirm_bar

            # Check hidden bearish: price lower high + TSI higher high
            if (prev_ph_price is not None and prev_ph_tsi is not None and
                last_ph_price < prev_ph_price and last_ph_tsi > prev_ph_tsi and
                (last_ph_bar - prev_ph_bar) <= p['max_pivot_dist']):
                bear_div_bars.add(i)
                last_bear_div_bar = i

    # Now run the trading simulation
    trades = []
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0
    stop_level = 0
    peak_price = 0
    is_trailing = False
    bars_since_exit = 999
    last_bull_seen = -9999
    last_bear_seen = -9999

    for i in range(max(p['ma_slow'] + 1, p['pivot_len'] * 2 + 5), n):
        # Update divergence recency tracking
        if i in bull_div_bars:
            last_bull_seen = i
        if i in bear_div_bars:
            last_bear_seen = i

        t = times[i]
        h = t.hour
        m = t.minute
        et_mins = h * 60 + m

        sess_start = p['sess_start_h'] * 60 + p['sess_start_m']
        sess_end = p['sess_end_h'] * 60 + p['sess_end_m']
        in_session = et_mins >= sess_start and et_mins < sess_end
        force_close = et_mins >= 960 and et_mins < 970  # 16:00-16:10

        # MA stack
        if np.isnan(ma5[i]) or np.isnan(ma10[i]) or np.isnan(ma20[i]):
            continue

        bull_stack = ma5[i] > ma10[i] and ma10[i] > ma20[i]
        bear_stack = ma5[i] < ma10[i] and ma10[i] < ma20[i]
        price_above = close[i] > ma5[i]
        price_below = close[i] < ma5[i]

        # Accumulation
        lookback = p['acc_lookback']
        if i < lookback + 1:
            continue

        bars_rising = sum(1 for j in range(lookback) if close[i - j] > close[i - j - 1])
        bars_falling = sum(1 for j in range(lookback) if close[i - j] < close[i - j - 1])

        no_spike = True
        for j in range(lookback):
            bar_move = abs(close[i - j] - opn[i - j]) / opn[i - j] * 100
            if bar_move > p['max_bar_size']:
                no_spike = False
                break

        accumulating = bars_rising >= p['acc_min_bars'] and no_spike
        distributing = bars_falling >= p['acc_min_bars'] and no_spike

        mas_rising = ma5[i] > ma5[i - 1] and ma10[i] > ma10[i - 1]
        mas_falling = ma5[i] < ma5[i - 1] and ma10[i] < ma10[i - 1]

        # TSI filter
        recent_bull = (i - last_bull_seen) <= p['tsi_recency']
        recent_bear = (i - last_bear_seen) <= p['tsi_recency']

        # Cooldown
        if position == 0:
            bars_since_exit += 1
        cooled_off = bars_since_exit >= p['cooldown']

        # Track peak for existing position
        if position == 1:
            if close[i] > peak_price:
                peak_price = close[i]
        elif position == -1:
            if close[i] < peak_price:
                peak_price = close[i]

        # BE trigger
        if position == 1 and not is_trailing:
            if close[i] >= entry_price * (1 + p['be_trigger'] / 100):
                is_trailing = True
                floor_level = entry_price * (1 + 0.05 / 100)
                new_stop = peak_price * (1 - p['trail_pct'] / 100)
                stop_level = max(floor_level, new_stop)

        if position == -1 and not is_trailing:
            if close[i] <= entry_price * (1 - p['be_trigger'] / 100):
                is_trailing = True
                floor_level = entry_price * (1 - 0.05 / 100)
                new_stop = peak_price * (1 + p['trail_pct'] / 100)
                stop_level = min(floor_level, new_stop)

        # Update trail
        if is_trailing and position == 1:
            new_stop = peak_price * (1 - p['trail_pct'] / 100)
            if new_stop > stop_level:
                stop_level = new_stop
        if is_trailing and position == -1:
            new_stop = peak_price * (1 + p['trail_pct'] / 100)
            if new_stop < stop_level:
                stop_level = new_stop

        # Exit checks
        if position == 1 and close[i] <= stop_level:
            pnl_pct = (close[i] - entry_price) / entry_price * 100
            pnl_usd = capital * 0.95 * pnl_pct / 100
            capital += pnl_usd
            trades.append({
                'date': t, 'dir': 'LONG',
                'result': 'Trail Stop' if is_trailing else 'Stop Loss',
                'pnl': pnl_usd, 'pnl_pct': pnl_pct
            })
            position = 0
            is_trailing = False
            bars_since_exit = 0
            continue

        if position == -1 and close[i] >= stop_level:
            pnl_pct = (entry_price - close[i]) / entry_price * 100
            pnl_usd = capital * 0.95 * pnl_pct / 100
            capital += pnl_usd
            trades.append({
                'date': t, 'dir': 'SHORT',
                'result': 'Trail Stop' if is_trailing else 'Stop Loss',
                'pnl': pnl_usd, 'pnl_pct': pnl_pct
            })
            position = 0
            is_trailing = False
            bars_since_exit = 0
            continue

        # Force close
        if force_close and position != 0:
            if position == 1:
                pnl_pct = (close[i] - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - close[i]) / entry_price * 100
            pnl_usd = capital * 0.95 * pnl_pct / 100
            capital += pnl_usd
            trades.append({
                'date': t, 'dir': 'LONG' if position == 1 else 'SHORT',
                'result': 'Market Close',
                'pnl': pnl_usd, 'pnl_pct': pnl_pct
            })
            position = 0
            is_trailing = False
            bars_since_exit = 0
            continue

        # Entry signals
        if position != 0 or not in_session or not cooled_off:
            continue

        long_entry = (bull_stack and price_above and accumulating and
                      mas_rising and recent_bull)
        short_entry = (bear_stack and price_below and distributing and
                       mas_falling and recent_bear)

        if long_entry:
            position = 1
            entry_price = close[i]
            stop_level = close[i] * (1 - p['sl_pct'] / 100)
            peak_price = close[i]
            is_trailing = False
            bars_since_exit = 0

        elif short_entry:
            position = -1
            entry_price = close[i]
            stop_level = close[i] * (1 + p['sl_pct'] / 100)
            peak_price = close[i]
            is_trailing = False
            bars_since_exit = 0

    return trades, capital


def summarize(trades, capital_start=5000, label=""):
    """Print trade summary."""
    if not trades:
        print(f"  {label}: NO TRADES")
        return {'trades': 0, 'pnl': 0, 'wr': 0, 'pf': 0}

    total = len(trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] < 0]
    gp = sum(t['pnl'] for t in wins)
    gl = abs(sum(t['pnl'] for t in losses))
    pf = gp / gl if gl > 0 else float('inf')
    wr = len(wins) / total * 100
    total_pnl = sum(t['pnl'] for t in trades)
    ret = total_pnl / capital_start * 100

    # Max DD
    cum = 0
    peak = 0
    max_dd = 0
    for t in trades:
        cum += t['pnl']
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd

    print(f"  {label}: {total} trades | WR {wr:.0f}% | PF {pf:.2f} | "
          f"P&L ${total_pnl:>7.2f} ({ret:>+.1f}%) | DD ${max_dd:.0f}")

    return {'trades': total, 'pnl': total_pnl, 'wr': wr, 'pf': pf, 'dd': max_dd, 'ret': ret}


def main():
    sym = "TQQQ"
    print(f"Fetching {sym} 5-min data...")
    df = fetch_5min_data(sym, days=59)
    if df is None or len(df) == 0:
        print("No data!")
        return

    dates = df.index.date
    print(f"Data: {dates[0]} to {dates[-1]} ({len(set(dates))} days, {len(df)} bars)")

    # Base params (matching v1.3 locked + TSI defaults)
    base = {
        'ma_fast': 5, 'ma_mid': 10, 'ma_slow': 20,
        'acc_lookback': 10, 'acc_min_bars': 6, 'max_bar_size': 1.05,
        'be_trigger': 1.50, 'sl_pct': 1.50, 'trail_pct': 0.75,
        'tsi_long': 13, 'tsi_short': 7, 'tsi_sig': 7,
        'pivot_len': 5, 'max_pivot_dist': 50,
        'tsi_recency': 15, 'cooldown': 8,
        'sess_start_h': 9, 'sess_start_m': 30,
        'sess_end_h': 11, 'sess_end_m': 0,
    }

    print("\n" + "=" * 90)
    print("TEST 1: Vary SESSION WINDOW (keeping recency=15, cooldown=8)")
    print("=" * 90)
    sessions = [
        (9, 30, 11, 0, "9:30-11:00 (baseline)"),
        (9, 30, 12, 0, "9:30-12:00"),
        (9, 30, 13, 0, "9:30-13:00"),
        (9, 30, 14, 0, "9:30-14:00"),
        (9, 30, 15, 0, "9:30-15:00"),
        (9, 30, 15, 30, "9:30-15:30"),
    ]
    for sh, sm, eh, em, label in sessions:
        p = {**base, 'sess_start_h': sh, 'sess_start_m': sm,
             'sess_end_h': eh, 'sess_end_m': em}
        trades, cap = backtest_ma_tsi(df, p)
        summarize(trades, label=label)

    print("\n" + "=" * 90)
    print("TEST 2: Vary TSI RECENCY (session 9:30-15:00, cooldown=8)")
    print("=" * 90)
    for recency in [10, 15, 20, 25, 30, 40, 50]:
        p = {**base, 'tsi_recency': recency,
             'sess_end_h': 15, 'sess_end_m': 0}
        trades, cap = backtest_ma_tsi(df, p)
        summarize(trades, label=f"recency={recency}")

    print("\n" + "=" * 90)
    print("TEST 3: Vary COOLDOWN (session 9:30-15:00, recency=15)")
    print("=" * 90)
    for cd in [2, 3, 5, 8, 10, 15]:
        p = {**base, 'cooldown': cd,
             'sess_end_h': 15, 'sess_end_m': 0}
        trades, cap = backtest_ma_tsi(df, p)
        summarize(trades, label=f"cooldown={cd}")

    print("\n" + "=" * 90)
    print("TEST 4: Vary PIVOT LENGTH (session 9:30-15:00, recency=15)")
    print("=" * 90)
    for pl in [3, 4, 5, 6, 7, 8]:
        p = {**base, 'pivot_len': pl,
             'sess_end_h': 15, 'sess_end_m': 0}
        trades, cap = backtest_ma_tsi(df, p)
        summarize(trades, label=f"pivot_len={pl}")

    print("\n" + "=" * 90)
    print("TEST 5: NO TSI FILTER (pure MA accumulation baseline)")
    print("=" * 90)
    for sh, sm, eh, em, label in sessions:
        p = {**base, 'tsi_recency': 99999,  # effectively no filter
             'sess_start_h': sh, 'sess_start_m': sm,
             'sess_end_h': eh, 'sess_end_m': em}
        trades, cap = backtest_ma_tsi(df, p)
        summarize(trades, label=f"NO TSI {label}")

    print("\n" + "=" * 90)
    print("TEST 6: BEST COMBOS (grid search top candidates)")
    print("=" * 90)
    best = []
    for recency in [15, 20, 25, 30]:
        for cd in [3, 5, 8]:
            for eh in [13, 14, 15]:
                for pl in [3, 4, 5]:
                    p = {**base, 'tsi_recency': recency, 'cooldown': cd,
                         'sess_end_h': eh, 'sess_end_m': 0, 'pivot_len': pl}
                    trades, cap = backtest_ma_tsi(df, p)
                    if trades:
                        wins = [t for t in trades if t['pnl'] > 0]
                        losses = [t for t in trades if t['pnl'] < 0]
                        gp = sum(t['pnl'] for t in wins)
                        gl = abs(sum(t['pnl'] for t in losses))
                        pf = gp / gl if gl > 0 else 0
                        total_pnl = sum(t['pnl'] for t in trades)
                        best.append({
                            'recency': recency, 'cooldown': cd,
                            'sess_end': eh, 'pivot_len': pl,
                            'trades': len(trades), 'pnl': total_pnl,
                            'pf': pf, 'wr': len(wins)/len(trades)*100
                        })

    # Sort by P&L, show top 15
    best.sort(key=lambda x: x['pnl'], reverse=True)
    print(f"{'Recency':>8} {'CD':>3} {'SessEnd':>7} {'PivotL':>6} | "
          f"{'Trades':>6} {'WR':>5} {'PF':>6} {'P&L':>10}")
    print("-" * 70)
    for b in best[:15]:
        print(f"{b['recency']:>8} {b['cooldown']:>3} {b['sess_end']:>7}:00 {b['pivot_len']:>6} | "
              f"{b['trades']:>6} {b['wr']:>4.0f}% {b['pf']:>6.2f} ${b['pnl']:>9.2f}")

    # Also show by PF (min 5 trades)
    print("\nTop 15 by Profit Factor (min 5 trades):")
    best_pf = [b for b in best if b['trades'] >= 5]
    best_pf.sort(key=lambda x: x['pf'], reverse=True)
    for b in best_pf[:15]:
        print(f"{b['recency']:>8} {b['cooldown']:>3} {b['sess_end']:>7}:00 {b['pivot_len']:>6} | "
              f"{b['trades']:>6} {b['wr']:>4.0f}% {b['pf']:>6.2f} ${b['pnl']:>9.2f}")


if __name__ == "__main__":
    main()
