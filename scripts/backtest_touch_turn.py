"""
Touch and Turn Scalper Backtester
- First 15-min candle after open: measure high/low range
- Liquidity candle: range >= 25% of ATR(14)
- Fade: RED opening -> LONG at low, GREEN -> SHORT at high
- TP = 38.2% of range from entry edge
- SL = TP/2 (2:1 R:R)
- Only within 90 min of open (9:30-11:00 ET)
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def get_1m_data(sym):
    """Fetch 1-min data in 7-day chunks (yfinance limit)."""
    chunks = []
    end = datetime.now()
    for i in range(4):
        start = end - timedelta(days=7)
        chunk = yf.download(
            sym,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1m",
            progress=False,
        )
        if len(chunk) > 0:
            chunks.append(chunk)
        end = start
    if not chunks:
        return None
    intra = pd.concat(chunks).sort_index()
    intra = intra[~intra.index.duplicated()]
    intra.columns = [c[0] if isinstance(c, tuple) else c for c in intra.columns]
    if intra.index.tz:
        intra.index = intra.index.tz_convert("America/New_York").tz_localize(None)
    return intra


def backtest(sym, capital_start=5000):
    print("=" * 80)
    print(f"TOUCH AND TURN SCALPER: {sym}")
    print("=" * 80)

    # Daily ATR
    daily = yf.download(sym, period="6mo", interval="1d", progress=False)
    daily.columns = [c[0] if isinstance(c, tuple) else c for c in daily.columns]
    daily["tr"] = np.maximum(
        daily["High"] - daily["Low"],
        np.maximum(
            abs(daily["High"] - daily["Close"].shift(1)),
            abs(daily["Low"] - daily["Close"].shift(1)),
        ),
    )
    daily["atr14"] = daily["tr"].rolling(14).mean()

    intra = get_1m_data(sym)
    if intra is None or len(intra) == 0:
        print("No intraday data!")
        return

    intra["date"] = intra.index.date
    dates = sorted(intra["date"].unique())
    print(f"Testing {len(dates)} days: {dates[0]} to {dates[-1]}")

    trades = []
    skip_liq = 0
    skip_entry = 0
    capital = capital_start

    for d in dates:
        day_bars = intra[intra["date"] == d]
        rth = day_bars.between_time("09:30", "15:59")
        first_15 = rth.between_time("09:30", "09:44")
        if len(first_15) < 5:
            continue

        ch = first_15["High"].max()
        cl = first_15["Low"].min()
        co = first_15["Open"].iloc[0]
        cc = first_15["Close"].iloc[-1]
        cr = ch - cl
        if cr <= 0:
            continue

        d_pd = pd.Timestamp(d)
        mask = daily.index <= d_pd
        if mask.sum() == 0:
            continue
        atr = daily.loc[mask, "atr14"].iloc[-1]
        if pd.isna(atr) or atr <= 0:
            continue

        thresh = atr * 0.25
        if cr < thresh:
            skip_liq += 1
            continue

        is_bear = cc < co
        if is_bear:
            entry_price = cl
            tp_dist = 0.382 * cr
            tp_price = entry_price + tp_dist
            sl_dist = tp_dist / 2
            sl_price = entry_price - sl_dist
            direction = "LONG"
        else:
            entry_price = ch
            tp_dist = 0.382 * cr
            tp_price = entry_price - tp_dist
            sl_dist = tp_dist / 2
            sl_price = entry_price + sl_dist
            direction = "SHORT"

        # Trade window 9:45-11:00
        tw = rth.between_time("09:45", "10:59")
        if len(tw) == 0:
            continue

        result = None
        entered = False
        entry_time = None
        exit_time = None

        for ts, bar in tw.iterrows():
            if not entered:
                if direction == "LONG" and bar["Low"] <= entry_price:
                    entered = True
                    entry_time = ts
                elif direction == "SHORT" and bar["High"] >= entry_price:
                    entered = True
                    entry_time = ts
                if not entered:
                    continue

            if direction == "LONG":
                if bar["Low"] <= sl_price:
                    result = "SL"
                    exit_time = ts
                    break
                if bar["High"] >= tp_price:
                    result = "TP"
                    exit_time = ts
                    break
            else:
                if bar["High"] >= sl_price:
                    result = "SL"
                    exit_time = ts
                    break
                if bar["Low"] <= tp_price:
                    result = "TP"
                    exit_time = ts
                    break

        if not entered:
            skip_entry += 1
            continue

        if result is None:
            last = tw.iloc[-1]
            if direction == "LONG":
                to_pnl = (last["Close"] - entry_price) / entry_price * 100
            else:
                to_pnl = (entry_price - last["Close"]) / entry_price * 100
            result = "TIMEOUT"
        else:
            to_pnl = 0

        pos = capital * 0.95
        if result == "TP":
            pnl = pos * tp_dist / entry_price
        elif result == "SL":
            pnl = -pos * sl_dist / entry_price
        else:
            pnl = pos * to_pnl / 100

        trades.append(
            {
                "date": d,
                "dir": direction,
                "entry": entry_price,
                "range": cr,
                "atr": atr,
                "thresh": thresh,
                "result": result,
                "pnl": pnl,
                "tp_dist": tp_dist,
                "sl_dist": sl_dist,
                "entry_time": entry_time,
                "exit_time": exit_time,
            }
        )
        capital += pnl

    print(f"Skipped (no liquidity candle): {skip_liq}")
    print(f"Skipped (no entry touch): {skip_entry}")
    print(f"Trades taken: {len(trades)}")

    if not trades:
        print("No trades!")
        return

    wins = sum(1 for t in trades if t["result"] == "TP")
    losses = sum(1 for t in trades if t["result"] == "SL")
    timeouts = sum(1 for t in trades if t["result"] == "TIMEOUT")
    total_pnl = sum(t["pnl"] for t in trades)
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    pf = gp / gl if gl > 0 else float("inf")

    cum = 0
    peak = 0
    max_dd = 0
    for t in trades:
        cum += t["pnl"]
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd

    print(f"\nTP: {wins}, SL: {losses}, Timeout: {timeouts}")
    if wins + losses > 0:
        print(f"Win Rate (TP vs SL): {wins / (wins + losses) * 100:.1f}%")
    print(f"Total P&L: ${total_pnl:.2f} ({total_pnl / capital_start * 100:.1f}%)")
    print(f"PF: {pf:.3f}, MaxDD: ${max_dd:.2f}")

    avg_w = np.mean([t["pnl"] for t in trades if t["result"] == "TP"]) if wins else 0
    avg_l = np.mean([t["pnl"] for t in trades if t["result"] == "SL"]) if losses else 0
    print(f"Avg Win: ${avg_w:.2f}, Avg Loss: ${avg_l:.2f}")
    if avg_l != 0:
        print(f"Realized R:R: {abs(avg_w / avg_l):.2f}:1")

    cal_days = (dates[-1] - dates[0]).days
    if cal_days > 0 and len(dates) > 0:
        ann = total_pnl * 252 / len(dates)
        print(f"Annualized est: ${ann:.2f} ({ann / capital_start * 100:.1f}%)")

    print(f"\n--- Trades ---")
    cum = 0
    for t in trades:
        cum += t["pnl"]
        et = t["entry_time"].strftime("%H:%M") if t["entry_time"] else "?"
        xt = t["exit_time"].strftime("%H:%M") if t["exit_time"] else "?"
        print(
            f"  {t['date']} {t['dir']:5s} ${t['entry']:.2f} "
            f"rng=${t['range']:.2f} "
            f"in={et} out={xt} "
            f"{t['result']:7s} ${t['pnl']:>8.2f} cum=${cum:>9.2f}"
        )


if __name__ == "__main__":
    for s in ["QQQ", "TQQQ", "SPY"]:
        backtest(s)
        print("\n")
