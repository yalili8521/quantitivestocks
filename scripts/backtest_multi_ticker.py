"""
MA Accumulation Pure Runner — Multi-Ticker Backtest
Tests the same MA Accumulation strategy across multiple leveraged & non-leveraged ETFs.
Runs each independently with $5K each, then shows combined portfolio.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def fetch_5min_data(sym, days=59):
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


def backtest_ma(df, params, capital_start=5000):
    """Pure MA Accumulation backtest (no TSI filter)."""
    p = params
    capital = capital_start
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    opn = df['Open'].values
    times = df.index
    n = len(df)

    ma5 = pd.Series(close).rolling(p['ma_fast']).mean().values
    ma10 = pd.Series(close).rolling(p['ma_mid']).mean().values
    ma20 = pd.Series(close).rolling(p['ma_slow']).mean().values

    trades = []
    position = 0
    entry_price = 0
    stop_level = 0
    peak_price = 0
    is_trailing = False
    bars_since_exit = 999

    for i in range(p['ma_slow'] + 2, n):
        t = times[i]
        h = t.hour
        m = t.minute
        et_mins = h * 60 + m

        sess_start = p['sess_start_h'] * 60 + p['sess_start_m']
        sess_end = p['sess_end_h'] * 60 + p['sess_end_m']
        in_session = et_mins >= sess_start and et_mins < sess_end
        force_close = et_mins >= 960 and et_mins < 970

        if np.isnan(ma5[i]) or np.isnan(ma10[i]) or np.isnan(ma20[i]):
            continue

        bull_stack = ma5[i] > ma10[i] and ma10[i] > ma20[i]
        bear_stack = ma5[i] < ma10[i] and ma10[i] < ma20[i]
        price_above = close[i] > ma5[i]
        price_below = close[i] < ma5[i]

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

        if position == 0:
            bars_since_exit += 1
        cooled_off = bars_since_exit >= p['cooldown']

        # Track peak
        if position == 1 and close[i] > peak_price:
            peak_price = close[i]
        elif position == -1 and close[i] < peak_price:
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

        # Exit: stop/trail
        if position == 1 and close[i] <= stop_level:
            pnl_pct = (close[i] - entry_price) / entry_price * 100
            pnl_usd = capital * 0.95 * pnl_pct / 100
            capital += pnl_usd
            trades.append({'date': t, 'dir': 'LONG', 'result': 'Trail Stop' if is_trailing else 'Stop Loss',
                           'pnl': pnl_usd, 'pnl_pct': pnl_pct, 'entry': entry_price, 'exit': close[i]})
            position = 0; is_trailing = False; bars_since_exit = 0
            continue

        if position == -1 and close[i] >= stop_level:
            pnl_pct = (entry_price - close[i]) / entry_price * 100
            pnl_usd = capital * 0.95 * pnl_pct / 100
            capital += pnl_usd
            trades.append({'date': t, 'dir': 'SHORT', 'result': 'Trail Stop' if is_trailing else 'Stop Loss',
                           'pnl': pnl_usd, 'pnl_pct': pnl_pct, 'entry': entry_price, 'exit': close[i]})
            position = 0; is_trailing = False; bars_since_exit = 0
            continue

        # Force close
        if force_close and position != 0:
            if position == 1:
                pnl_pct = (close[i] - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - close[i]) / entry_price * 100
            pnl_usd = capital * 0.95 * pnl_pct / 100
            capital += pnl_usd
            trades.append({'date': t, 'dir': 'LONG' if position == 1 else 'SHORT',
                           'result': 'Market Close', 'pnl': pnl_usd, 'pnl_pct': pnl_pct,
                           'entry': entry_price, 'exit': close[i]})
            position = 0; is_trailing = False; bars_since_exit = 0
            continue

        # Entry
        if position != 0 or not in_session or not cooled_off:
            continue

        long_entry = bull_stack and price_above and accumulating and mas_rising
        short_entry = p.get('allow_short', True) and bear_stack and price_below and distributing and mas_falling

        if long_entry:
            position = 1; entry_price = close[i]
            stop_level = close[i] * (1 - p['sl_pct'] / 100)
            peak_price = close[i]; is_trailing = False; bars_since_exit = 0
        elif short_entry:
            position = -1; entry_price = close[i]
            stop_level = close[i] * (1 + p['sl_pct'] / 100)
            peak_price = close[i]; is_trailing = False; bars_since_exit = 0

    return trades, capital


def summarize(sym, trades, capital_start=5000):
    if not trades:
        print(f"  {sym:8s}: NO TRADES")
        return None

    total = len(trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] < 0]
    gp = sum(t['pnl'] for t in wins)
    gl = abs(sum(t['pnl'] for t in losses))
    pf = gp / gl if gl > 0 else float('inf')
    wr = len(wins) / total * 100
    total_pnl = sum(t['pnl'] for t in trades)
    ret = total_pnl / capital_start * 100

    cum = 0; peak = 0; max_dd = 0
    for t in trades:
        cum += t['pnl']
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > max_dd: max_dd = dd

    # By exit type
    by_exit = {}
    for t in trades:
        r = t['result']
        if r not in by_exit:
            by_exit[r] = {'count': 0, 'pnl': 0}
        by_exit[r]['count'] += 1
        by_exit[r]['pnl'] += t['pnl']

    exit_str = " | ".join(f"{k}: {v['count']}/${v['pnl']:+.0f}" for k, v in by_exit.items())

    print(f"  {sym:8s}: {total:3d} trades | WR {wr:4.0f}% | PF {pf:5.2f} | "
          f"P&L ${total_pnl:>8.2f} ({ret:>+5.1f}%) | DD ${max_dd:>6.0f} | {exit_str}")

    return {'sym': sym, 'trades': total, 'pnl': total_pnl, 'wr': wr, 'pf': pf,
            'dd': max_dd, 'ret': ret, 'trade_list': trades}


def main():
    # Tickers to test — mix of leveraged, unleveraged, sectors
    tickers = [
        # 3x leveraged
        "TQQQ", "SOXL", "UPRO", "TNA", "LABU", "TECL", "FAS",
        # 2x leveraged
        "QLD", "SSO", "USD",
        # 1x ETFs
        "QQQ", "SPY", "IWM", "SMH", "XLK", "XLF", "XLE", "XLV",
        "ARKK", "KWEB", "EEM",
        # High-beta single stocks
        "NVDA", "TSLA", "AMD", "META", "AAPL", "AMZN", "GOOGL", "MSTR",
        "COIN", "HOOD", "PLTR", "SOFI",
    ]

    # TQQQ v1.3 locked params (3x leveraged)
    params_3x = {
        'ma_fast': 5, 'ma_mid': 10, 'ma_slow': 20,
        'acc_lookback': 10, 'acc_min_bars': 6, 'max_bar_size': 1.05,
        'be_trigger': 1.50, 'sl_pct': 1.50, 'trail_pct': 0.75,
        'cooldown': 8, 'allow_short': True,
        'sess_start_h': 9, 'sess_start_m': 30,
        'sess_end_h': 11, 'sess_end_m': 0,
    }

    # QQQ v3 locked params (1x)
    params_1x = {
        'ma_fast': 5, 'ma_mid': 10, 'ma_slow': 20,
        'acc_lookback': 10, 'acc_min_bars': 6, 'max_bar_size': 0.35,
        'be_trigger': 0.20, 'sl_pct': 0.50, 'trail_pct': 0.50,
        'cooldown': 5, 'allow_short': True,
        'sess_start_h': 9, 'sess_start_m': 30,
        'sess_end_h': 11, 'sess_end_m': 0,
    }

    # 2x params (in between)
    params_2x = {
        'ma_fast': 5, 'ma_mid': 10, 'ma_slow': 20,
        'acc_lookback': 10, 'acc_min_bars': 6, 'max_bar_size': 0.70,
        'be_trigger': 0.75, 'sl_pct': 1.00, 'trail_pct': 0.60,
        'cooldown': 6, 'allow_short': True,
        'sess_start_h': 9, 'sess_start_m': 30,
        'sess_end_h': 11, 'sess_end_m': 0,
    }

    # Stock params (similar to 1x but wider stops for individual stock vol)
    params_stock = {
        'ma_fast': 5, 'ma_mid': 10, 'ma_slow': 20,
        'acc_lookback': 10, 'acc_min_bars': 6, 'max_bar_size': 0.80,
        'be_trigger': 0.30, 'sl_pct': 0.60, 'trail_pct': 0.50,
        'cooldown': 5, 'allow_short': True,
        'sess_start_h': 9, 'sess_start_m': 30,
        'sess_end_h': 11, 'sess_end_m': 0,
    }

    leverage_3x = {"TQQQ", "SOXL", "UPRO", "TNA", "LABU", "TECL", "FAS"}
    leverage_2x = {"QLD", "SSO", "USD"}
    etfs_1x = {"QQQ", "SPY", "IWM", "SMH", "XLK", "XLF", "XLE", "XLV", "ARKK", "KWEB", "EEM"}

    print("Fetching data for all tickers...")
    data = {}
    for sym in tickers:
        df = fetch_5min_data(sym, days=59)
        if df is not None and len(df) > 100:
            data[sym] = df
            print(f"  {sym}: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")
        else:
            print(f"  {sym}: FAILED or insufficient data")

    print(f"\n{'='*100}")
    print("RESULTS — MA Accumulation Pure Runner (9:30-11:00, 5-min)")
    print(f"{'='*100}")

    results = []
    all_trades = []

    print("\n--- 3x Leveraged ETFs ---")
    for sym in sorted(leverage_3x & data.keys()):
        trades, cap = backtest_ma(data[sym], params_3x)
        r = summarize(sym, trades)
        if r:
            results.append(r)
            for t in trades:
                t['sym'] = sym
            all_trades.extend(trades)

    print("\n--- 2x Leveraged ETFs ---")
    for sym in sorted(leverage_2x & data.keys()):
        trades, cap = backtest_ma(data[sym], params_2x)
        r = summarize(sym, trades)
        if r:
            results.append(r)
            for t in trades:
                t['sym'] = sym
            all_trades.extend(trades)

    print("\n--- 1x ETFs ---")
    for sym in sorted(etfs_1x & data.keys()):
        trades, cap = backtest_ma(data[sym], params_1x)
        r = summarize(sym, trades)
        if r:
            results.append(r)
            for t in trades:
                t['sym'] = sym
            all_trades.extend(trades)

    print("\n--- Individual Stocks ---")
    stocks = set(tickers) - leverage_3x - leverage_2x - etfs_1x
    for sym in sorted(stocks & data.keys()):
        trades, cap = backtest_ma(data[sym], params_stock)
        r = summarize(sym, trades)
        if r:
            results.append(r)
            for t in trades:
                t['sym'] = sym
            all_trades.extend(trades)

    # Portfolio summary
    print(f"\n{'='*100}")
    print("PORTFOLIO SUMMARY")
    print(f"{'='*100}")

    profitable = [r for r in results if r['pnl'] > 0]
    losing = [r for r in results if r['pnl'] <= 0]
    total_trades = sum(r['trades'] for r in results)
    total_pnl = sum(r['pnl'] for r in results)
    total_alloc = len(results) * 5000

    all_wins = [t for t in all_trades if t['pnl'] > 0]
    all_losses = [t for t in all_trades if t['pnl'] < 0]
    gp = sum(t['pnl'] for t in all_wins)
    gl = abs(sum(t['pnl'] for t in all_losses))
    combined_pf = gp / gl if gl > 0 else 0
    combined_wr = len(all_wins) / len(all_trades) * 100 if all_trades else 0

    print(f"Tickers tested: {len(results)}")
    print(f"Profitable tickers: {len(profitable)} | Losing: {len(losing)}")
    print(f"Total trades: {total_trades}")
    print(f"Combined WR: {combined_wr:.1f}%")
    print(f"Combined PF: {combined_pf:.2f}")
    print(f"Combined P&L: ${total_pnl:.2f}")
    print(f"If $5K per ticker ({len(results)} tickers = ${total_alloc:,}): {total_pnl/total_alloc*100:.1f}% return")

    # Top performers
    results.sort(key=lambda x: x['pnl'], reverse=True)
    print(f"\nTop 10 by P&L:")
    for r in results[:10]:
        print(f"  {r['sym']:8s}: ${r['pnl']:>8.2f} ({r['ret']:>+5.1f}%) | {r['trades']} trades | WR {r['wr']:.0f}% | PF {r['pf']:.2f}")

    print(f"\nBottom 5 by P&L:")
    for r in results[-5:]:
        print(f"  {r['sym']:8s}: ${r['pnl']:>8.2f} ({r['ret']:>+5.1f}%) | {r['trades']} trades | WR {r['wr']:.0f}% | PF {r['pf']:.2f}")

    # Realistic portfolio: pick top-5 tickers, $5K each = $25K total
    print(f"\n{'='*100}")
    print("REALISTIC PORTFOLIO: Top-5 tickers by P&L, $5K each = $25K")
    print(f"{'='*100}")
    top5 = results[:5]
    top5_pnl = sum(r['pnl'] for r in top5)
    top5_trades = sum(r['trades'] for r in top5)
    top5_all = [t for t in all_trades if t['sym'] in {r['sym'] for r in top5}]
    top5_wins = [t for t in top5_all if t['pnl'] > 0]
    top5_losses = [t for t in top5_all if t['pnl'] < 0]
    top5_gp = sum(t['pnl'] for t in top5_wins)
    top5_gl = abs(sum(t['pnl'] for t in top5_losses))
    top5_pf = top5_gp / top5_gl if top5_gl > 0 else 0

    for r in top5:
        print(f"  {r['sym']:8s}: ${r['pnl']:>8.2f} | {r['trades']} trades | WR {r['wr']:.0f}% | PF {r['pf']:.2f}")
    print(f"  {'─'*60}")
    print(f"  TOTAL:   ${top5_pnl:.2f} on $25K = {top5_pnl/25000*100:.1f}% | {top5_trades} trades | PF {top5_pf:.2f}")
    print(f"  Annualized est: ${top5_pnl * 365 / 40:.2f} ({top5_pnl * 365 / 40 / 25000 * 100:.0f}%)")

    # Same for top-3 (more concentrated)
    print(f"\nCONCENTRATED: Top-3 tickers, $5K each = $15K")
    top3 = results[:3]
    top3_pnl = sum(r['pnl'] for r in top3)
    for r in top3:
        print(f"  {r['sym']:8s}: ${r['pnl']:>8.2f} | {r['trades']} trades")
    print(f"  TOTAL:   ${top3_pnl:.2f} on $15K = {top3_pnl/15000*100:.1f}%")
    print(f"  Annualized est: ${top3_pnl * 365 / 40:.2f} ({top3_pnl * 365 / 40 / 15000 * 100:.0f}%)")


if __name__ == "__main__":
    main()
