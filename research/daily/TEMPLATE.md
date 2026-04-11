# Market Intelligence Brief — YYYY-MM-DD

> *Research and information only — not financial advice. All prices and performance figures sourced from web searches conducted on [DATE].*

<!--
  AUTHORING NOTES:
  - Fill every section. Use N/A with a reason if a section has no data.
  - Narrative sections (1-11) can use colorful language.
  - Signal Manifest section MUST use strict vocabulary (see below) — it is
    machine-parsed by src/research_reader.py and feeds the ETF selector.
  - When a section references yesterday's call, update Section 11 accordingly.
-->

---

## Today's Key Takeaway

<!--
  2-3 sentence executive summary. If a binary scenario is active (e.g. ceasefire
  deadline, Fed decision, earnings), state which way it resolved and the
  resulting price action in a single sentence.
-->

---

## 1. US Market Overview

| Index | Prior Close | Change | Pre-Market / Futures | Notes |
|---|---|---|---|---|
| S&P 500 | | | | |
| NASDAQ | | | | |
| Dow Jones | | | | |
| VIX | | | | |
| 10-Year Treasury | | | | |

**Session recap:** <!-- What happened yesterday, what drove the move -->

**Sector expectations for today:** <!-- Which sectors lead/lag -->

**Key levels:**
- S&P 500: resistance / support
- VIX: trigger levels
- Additional asset levels as relevant

---

## 2. Crypto Markets

| Asset | Price | 24h Change | Notes |
|---|---|---|---|
| BTC | | | |
| ETH | | | |
| SOL | | | |
| XRP | | | |
| + top movers | | | |

**Total market cap / BTC dominance / Fear & Greed:** <!-- numbers + context -->

**BTC analysis:** <!-- key levels, momentum, flows -->

**ETH analysis:** <!-- ETH/BTC ratio, narrative -->

**Altcoin narratives:** <!-- at least 2-3 specific altcoin theses when crypto is front-and-center -->

**Regulatory backdrop:** <!-- SEC, ETF flows, classification changes -->

---

## 3. ETF Watchlist

| ETF | Status | Catalyst / Notes |
|---|---|---|
| | ACTIVE / MONITORING / REVERSAL / NEW | |

**Status key:** ACTIVE (thesis playing out), MONITORING (watching for trigger), REVERSAL (thesis flipped), NEW (added today).

**Positioning shifts:** <!-- Which ETFs moved between statuses today and why -->

---

## 4. Macro Trends

**Federal Reserve:** <!-- Current rate, next meeting, data dependencies -->

**Inflation outlook:** <!-- CPI/PPI trajectory, oil pass-through -->

**DXY:** <!-- level, trend, divergences -->

**Oil:** <!-- WTI, Brent, supply/demand balance -->

**Geopolitical:** <!-- Active conflicts, binary events, countdowns -->

---

## 5. Earnings & Corporate Events

**Today's reports:**

| Time | Ticker | EPS Est | Rev Est | Key metric to watch |
|---|---|---|---|---|
| | | | | |

**This week ahead:**

| Date | Ticker | Event | Significance |
|---|---|---|---|
| | | | |

**Notable guidance/pre-announcements:** <!-- capex plans, warnings, raises -->

---

## 6. Insider & Institutional Activity

**Notable insider transactions (last 5 trading days):**

| Ticker | Direction | Amount | Insider / Title | Context |
|---|---|---|---|---|
| | BUY / SELL | | | |

**13F filings / fund flows:** <!-- Recent large position changes -->

**Aggregate metrics:** <!-- e.g. "$16.1B Mag 7 insider selling (trailing 2y)" -->

---

## 7. Options Flow & Unusual Activity

| Ticker | Type | Strike | Expiry | Volume / OI Ratio | Notes |
|---|---|---|---|---|---|
| | CALLS / PUTS | | | | |

**Block trades / sweeps:** <!-- Large single prints worth flagging -->

**Implied vol standouts:** <!-- IV rank >80, IV crush candidates -->

---

## 8. Trading Signals & Opportunities

**Today's Calendar:**

| Time | Event | Significance |
|---|---|---|
| | | |

**Week Ahead (remaining):**

| Date | Event |
|---|---|
| | |

**52-week highs / lows:** <!-- Where momentum is / capitulation candidates -->

**Key levels (updated):** <!-- Actionable price triggers -->

---

## 9. AI & Tech Sector Focus

| Ticker | Latest | Outlook |
|---|---|---|
| NVDA | | |
| MSFT | | |
| GOOG | | |
| AAPL | | |
| AMZN | | |
| META | | |

**TSMC / semiconductor supply chain:** <!-- monthly sales, capex, bottlenecks -->

**Hyperscaler AI capex:** <!-- aggregate 2026 spend, trajectory -->

**Notable AI news:** <!-- model launches, orders, regulation -->

---

## 10. Sector Rotation & Fund Flows

**ETF inflows (last 5 days):** <!-- Where money is going -->

**ETF outflows (last 5 days):** <!-- Where money is leaving -->

**Rotation signals:** <!-- Growth→Value, US→Intl, Large→Small, etc. -->

**Breadth indicators:** <!-- Advance/decline, % above 50dma, new highs vs lows -->

---

## 11. Signal Accuracy Tracker

**Yesterday's calls — outcome review:**

| Date | Call | Outcome |
|---|---|---|
| YYYY-MM-DD | TICKER: bullish/bearish | CONFIRMED / INVALIDATED / PENDING — actual result |

**Running accuracy (last 7 days):** <!-- hit rate % for sector / ticker / macro calls -->

**Standing theses (check if still valid):**
- AI infrastructure / hyperscaler capex: <!-- still intact? -->
- Onshoring / reshoring / infrastructure buildout: <!-- still intact? -->
- Energy transition / grid modernization: <!-- still intact? -->
- <!-- Add/retire as appropriate; prevents thesis dropout -->

---

## Signal Manifest

<!--
  STRICT MACHINE-READABLE SECTION.
  Parser: src/research_reader.py

  Sector vocabulary (use EXACTLY one per sector):
    STRONG | BULLISH | RECOVERING | NEUTRAL | WATCH |
    WEAK | REVERSAL | SHARP REVERSAL | BEARISH

  Ticker vocabulary (use EXACTLY one per ticker):
    bullish | bullish reversal | leaning bullish | hot mover |
    monitoring | neutral | cautious |
    leaning bearish | bearish | bearish reversal

  Each sub-section below must exist, even if empty (use "- none" if so).
-->

## Macro

<!--
  One paragraph. Include explicit regime keywords:
  RISK-ON | RISK-OFF | STAGFLATION | GOLDILOCKS | NEUTRAL
  And rate direction: HAWKISH | DOVISH | NEUTRAL
-->

## Sectors

- Sector Name: STRONG/BULLISH/RECOVERING/NEUTRAL/WATCH/WEAK/REVERSAL/SHARP REVERSAL/BEARISH — reason
- <!-- one line per sector -->

## Tickers

- TICKER: bullish/bullish reversal/leaning bullish/hot mover/monitoring/neutral/cautious/leaning bearish/bearish/bearish reversal, reason
- <!-- one line per ticker -->

## Insider Signals

- TICKER: insider_buy / insider_sell, $amount, insider_name/title
- <!-- use "- none" if no notable transactions -->

## Options Signals

- TICKER: unusual_calls / unusual_puts, strike, expiry, volume_oi_ratio
- <!-- use "- none" if no notable activity -->

## Sector Flows

- Sector: INFLOW / OUTFLOW / NEUTRAL, magnitude (high/medium/low)
- <!-- use "- none" if flow data unavailable -->

## Key Watchlist Items

- **Date / time: Event** — why it matters
- <!-- bulleted list of catalysts -->

## Themes

<!-- Comma-separated active themes. Keep thesis names consistent across days. -->

## Accuracy

- YYYY-MM-DD TICKER call: CONFIRMED / INVALIDATED / PENDING, actual_result
- <!-- one line per prior-day call being scored -->
