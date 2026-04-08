"""TQQQ final grid: Trail x BE x Daily Filter."""
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from itertools import product

print("Downloading data...")
tqqq = yf.download('TQQQ', period='60d', interval='5m', auto_adjust=True)
if isinstance(tqqq.columns, pd.MultiIndex):
    tqqq.columns = tqqq.columns.get_level_values(0)
tqqq.index = tqqq.index.tz_convert('America/New_York')

qqq_d = yf.download('QQQ', period='1y', interval='1d', auto_adjust=True)
if isinstance(qqq_d.columns, pd.MultiIndex):
    qqq_d.columns = qqq_d.columns.get_level_values(0)
print("TQQQ 5m: %d | QQQ daily: %d" % (len(tqqq), len(qqq_d)))

dc = qqq_d['Close'].values.astype(float)
dh = qqq_d['High'].values.astype(float)
dl = qqq_d['Low'].values.astype(float)
dv = qqq_d['Volume'].values.astype(float)
dd = qqq_d.index.date if hasattr(qqq_d.index, 'date') else pd.to_datetime(qqq_d.index).date

def sma(a, n):
    return pd.Series(a).rolling(n).mean().values

sma50 = sma(dc, 50)
vs5 = sma(dv, 5)
vs20 = sma(dv, 20)
vol_trend = vs5 / np.where(vs20 > 0, vs20, 1)

def calc_adx(h, l, c, p=14):
    n = len(c)
    tr = np.zeros(n)
    pdm = np.zeros(n)
    mdm = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
        u = h[i] - h[i-1]
        d = l[i-1] - l[i]
        pdm[i] = u if u > d and u > 0 else 0
        mdm[i] = d if d > u and d > 0 else 0
    atr = pd.Series(tr).ewm(span=p, adjust=False).mean().values
    pdi = 100 * pd.Series(pdm).ewm(span=p, adjust=False).mean().values / np.where(atr > 0, atr, 1)
    mdi = 100 * pd.Series(mdm).ewm(span=p, adjust=False).mean().values / np.where(atr > 0, atr, 1)
    dx = 100 * np.abs(pdi - mdi) / np.where((pdi + mdi) > 0, pdi + mdi, 1)
    return pd.Series(dx).ewm(span=p, adjust=False).mean().values

adx_arr = calc_adx(dh, dl, dc, 14)

dlk = {}
for i in range(len(qqq_d)):
    dlk[dd[i]] = {
        'close': dc[i], 'sma50': sma50[i],
        'vol_trend': vol_trend[i], 'adx': adx_arr[i]
    }

def gpd(date):
    d = date
    for _ in range(5):
        d = d - datetime.timedelta(days=1)
        if d in dlk:
            return dlk[d]
    return None


def bt(df, p):
    close = df['Close'].values.astype(float)
    opn = df['Open'].values.astype(float)
    times = df.index
    n = len(close)
    ma_f = pd.Series(close).rolling(p['ma_fast']).mean().values
    ma_m = pd.Series(close).rolling(p['ma_mid']).mean().values
    ma_s = pd.Series(close).rolling(p['ma_slow']).mean().values
    equity = 5000.0
    peak_eq = 5000.0
    max_dd = 0.0
    pos = 0
    ep = 0.0
    sl = 0.0
    pp = 0.0
    itr = False
    bse = 999
    teq = 0.0
    trades = []
    ss = p['ssh'] * 60 + p['ssm']
    se = p['seh'] * 60 + p['sem']
    prev_l = False
    prev_s = False
    sb = max(p['ma_slow'], p['acc_lb'] + 1)
    sf = p.get('sma_filter', None)
    am = p.get('adx_min', 0)
    vtm = p.get('vt_min', 0)
    cached_d = None
    d_long = True
    d_short = True

    for i in range(sb, n):
        t = times[i]
        tm = t.hour * 60 + t.minute
        ins = tm >= ss and tm < se
        fc = tm >= 960 and tm < 970
        c = close[i]
        if np.isnan(ma_f[i]) or np.isnan(ma_m[i]) or np.isnan(ma_s[i]):
            continue

        td = t.date()
        if td != cached_d:
            cached_d = td
            di = gpd(td)
            d_long = True
            d_short = True
            if di:
                if sf:
                    sv = di.get("sma%d" % sf, np.nan)
                    if not np.isnan(sv):
                        if di['close'] <= sv:
                            d_long = False
                        if di['close'] >= sv:
                            d_short = False
                if am > 0 and di.get('adx', 0) < am:
                    d_long = False
                    d_short = False
                if vtm > 0 and di.get('vol_trend', 0) < vtm:
                    d_long = False
                    d_short = False

        bull = ma_f[i] > ma_m[i] and ma_m[i] > ma_s[i]
        bear = ma_f[i] < ma_m[i] and ma_m[i] < ma_s[i]
        pa = c > ma_f[i]
        pb = c < ma_f[i]

        br = 0
        bf = 0
        ns = True
        for j in range(p['acc_lb']):
            idx = i - j
            ip = i - j - 1
            if ip < 0:
                break
            if close[idx] > close[ip]:
                br += 1
            if close[idx] < close[ip]:
                bf += 1
            bm = abs(close[idx] - opn[idx]) / opn[idx] * 100 if opn[idx] > 0 else 0
            if bm > p['spike']:
                ns = False

        acc = br >= p['acc_min'] and ns
        dist = bf >= p['acc_min'] and ns
        mr = ma_f[i] > ma_f[i-1] and ma_m[i] > ma_m[i-1]
        mf = ma_f[i] < ma_f[i-1] and ma_m[i] < ma_m[i-1]
        co = bse >= p['cd']

        lc = bull and pa and acc and mr and ins and co and d_long
        sc = bear and pb and dist and mf and ins and co and d_short
        lt = lc and not prev_l
        st = sc and not prev_s
        prev_l = lc
        prev_s = sc

        if pos != 0:
            if pos == 1 and c > pp:
                pp = c
            if pos == -1 and c < pp:
                pp = c
            if not itr:
                if pos == 1 and c >= ep * (1 + p['be'] / 100):
                    itr = True
                    fl = ep * (1 + 0.05 / 100)
                    sl = max(fl, pp * (1 - p['tr'] / 100))
                elif pos == -1 and c <= ep * (1 - p['be'] / 100):
                    itr = True
                    fl = ep * (1 - 0.05 / 100)
                    sl = min(fl, pp * (1 + p['tr'] / 100))
            if itr:
                if pos == 1:
                    ns1 = pp * (1 - p['tr'] / 100)
                    if ns1 > sl:
                        sl = ns1
                elif pos == -1:
                    ns1 = pp * (1 + p['tr'] / 100)
                    if ns1 < sl:
                        sl = ns1

            ex = None
            if not itr:
                if (pos == 1 and c <= sl) or (pos == -1 and c >= sl):
                    ex = 'SL'
            if itr:
                if (pos == 1 and c <= sl) or (pos == -1 and c >= sl):
                    ex = 'TS'
            if fc:
                ex = 'MC'

            if ex:
                if pos == 1:
                    tpnl = (c - ep) / ep * teq
                else:
                    tpnl = (ep - c) / ep * teq
                equity += tpnl
                if equity > peak_eq:
                    peak_eq = equity
                dd_now = (peak_eq - equity) / peak_eq * 100
                if dd_now > max_dd:
                    max_dd = dd_now
                trades.append({'pnl': tpnl, 'et': ex})
                pos = 0
                bse = 0
                itr = False
                continue

        if pos == 0:
            bse += 1

        if pos == 0 and lt:
            pos = 1
            ep = c
            sl = c * (1 - p['sl'] / 100)
            pp = c
            itr = False
            teq = equity * 0.95
            bse = 0
        elif pos == 0 and st:
            pos = -1
            ep = c
            sl = c * (1 + p['sl'] / 100)
            pp = c
            itr = False
            teq = equity * 0.95
            bse = 0

    if pos != 0:
        c = close[-1]
        if pos == 1:
            tpnl = (c - ep) / ep * teq
        else:
            tpnl = (ep - c) / ep * teq
        equity += tpnl
        trades.append({'pnl': tpnl, 'et': 'EOD'})

    nt = len(trades)
    if nt == 0:
        return {'pnl': 0, 'pf': 0, 'wr': 0, 'trades': 0, 'aw': 0, 'al': 0,
                'slc': 0, 'mdd': 0, 'mcl': 0}

    w = [t for t in trades if t['pnl'] > 0]
    l = [t for t in trades if t['pnl'] <= 0]
    gp = sum(t['pnl'] for t in w) if w else 0
    gl = abs(sum(t['pnl'] for t in l)) if l else 0.001
    slc = sum(1 for t in trades if t['et'] == 'SL')
    mcl = 0
    cc = 0
    for t in trades:
        if t['pnl'] <= 0:
            cc += 1
            mcl = max(mcl, cc)
        else:
            cc = 0

    return {
        'pnl': equity - 5000, 'pf': gp / gl, 'wr': len(w) / nt * 100,
        'trades': nt, 'aw': gp / len(w) if w else 0, 'al': gl / len(l) if l else 0,
        'slc': slc, 'mdd': max_dd, 'mcl': mcl
    }


# Grid
trails = [0.75, 1.00, 1.25]
bes = [0.80, 1.00, 1.20, 1.50]
filters = [
    {'label': 'No filter'},
    {'label': 'ADX>20', 'adx_min': 20},
    {'label': 'ADX>20+VT>1.0', 'adx_min': 20, 'vt_min': 1.0},
    {'label': 'SMA50+ADX>20', 'sma_filter': 50, 'adx_min': 20},
]

hdr = "%-6s %-6s | %10s | %6s | %6s | %6s | %4s | %6s | %8s | %8s | %4s"
sep = "-" * 100

for f in filters:
    print()
    print("=" * 100)
    print("TQQQ: %s" % f['label'])
    print("=" * 100)
    print(hdr % ('Trail', 'BE', 'P&L', 'Ret%', 'PF', 'WR%', 'T', 'MaxDD', 'AvgWin', 'AvgLoss', 'MCL'))
    print(sep)
    results = []
    for tr, be in product(trails, bes):
        p = {
            'ma_fast': 5, 'ma_mid': 10, 'ma_slow': 20,
            'acc_lb': 10, 'acc_min': 6, 'spike': 1.05,
            'sl': 1.50, 'be': be, 'tr': tr, 'cd': 5,
            'ssh': 9, 'ssm': 30, 'seh': 11, 'sem': 0,
        }
        for k, v in f.items():
            if k != 'label':
                p[k] = v
        r = bt(tqqq, p)
        results.append((tr, be, r))
        print("%-6.2f %-6.2f | $%9.2f | %5.1f%% | %6.3f | %5.1f%% | %4d | %5.1f%% | $%7.2f | $%7.2f | %4d" % (
            tr, be, r['pnl'], r['pnl'] / 50, r['pf'], r['wr'], r['trades'],
            r['mdd'], r['aw'], r['al'], r['mcl']))

    valid = [x for x in results if x[2]['trades'] >= 5]
    if valid:
        bp = max(valid, key=lambda x: x[2]['pf'])
        bpnl = max(valid, key=lambda x: x[2]['pnl'])
        print()
        print("  >> Best PF:  Trail=%.2f BE=%.2f -> PF=%.3f | P&L=$%.2f | WR=%.1f%% | T=%d | MaxDD=%.1f%%" % (
            bp[0], bp[1], bp[2]['pf'], bp[2]['pnl'], bp[2]['wr'], bp[2]['trades'], bp[2]['mdd']))
        print("  >> Best P&L: Trail=%.2f BE=%.2f -> P&L=$%.2f | PF=%.3f | WR=%.1f%% | T=%d | MaxDD=%.1f%%" % (
            bpnl[0], bpnl[1], bpnl[2]['pnl'], bpnl[2]['pf'], bpnl[2]['wr'], bpnl[2]['trades'], bpnl[2]['mdd']))

# Final comparison
print()
print("=" * 100)
print("FINAL COMPARISON: Best of each filter")
print("=" * 100)
print()
print("%-35s | %8s | %6s | %6s | %6s | %4s | %6s | %4s" % (
    'Config', 'P&L', 'Ret%', 'PF', 'WR%', 'T', 'MaxDD', 'MCL'))
print("-" * 95)

for f in filters:
    best_pnl_r = None
    best_pf_r = None
    for tr, be in product(trails, bes):
        p = {
            'ma_fast': 5, 'ma_mid': 10, 'ma_slow': 20,
            'acc_lb': 10, 'acc_min': 6, 'spike': 1.05,
            'sl': 1.50, 'be': be, 'tr': tr, 'cd': 5,
            'ssh': 9, 'ssm': 30, 'seh': 11, 'sem': 0,
        }
        for k, v in f.items():
            if k != 'label':
                p[k] = v
        r = bt(tqqq, p)
        if r['trades'] >= 5:
            if best_pnl_r is None or r['pnl'] > best_pnl_r[2]['pnl']:
                best_pnl_r = (tr, be, r)
            if best_pf_r is None or r['pf'] > best_pf_r[2]['pf']:
                best_pf_r = (tr, be, r)

    if best_pnl_r:
        tr, be, r = best_pnl_r
        lbl = "%s (T%.2f/BE%.2f)" % (f['label'], tr, be)
        print("%-35s | $%7.0f | %5.1f%% | %6.3f | %5.1f%% | %4d | %5.1f%% | %4d" % (
            lbl, r['pnl'], r['pnl'] / 50, r['pf'], r['wr'], r['trades'], r['mdd'], r['mcl']))
