<!--
  This is the prompt body to paste into the existing Cowork scheduled task at:
    https://claude.ai/code/scheduled
  It replaces the March-era prompt. The task writes to:
    C:\Users\yalil\OneDrive\Desktop\AI-projects\quantitivestocks\research\daily\Market_Intel_YYYY-MM-DD.md
  The output is parsed by quantitivestocks/src/research_reader.py — the
  Signal Manifest section is machine-read with STRICT vocabulary (see bottom).
-->

# Daily Market Intelligence Report — Generator Prompt

You are generating a daily Market Intelligence brief for a quantitative ETF selector pipeline. The output is BOTH read by a human trader AND machine-parsed into signals that drive composite ETF scores, so follow the structure exactly.

## Output

Create a file named `Market_Intel_YYYY-MM-DD.md` (today's date, ET) in the mounted research folder. Overwrite if it already exists. The file must follow the structure in `research/daily/TEMPLATE.md` — all 11 numbered sections plus the Signal Manifest.

## Required sections (in order)

0. **Today's Key Takeaway** — 2-3 sentences. If a binary event resolved (Fed, CPI, ceasefire deadline, earnings), state the resolution and price action.
1. **US Market Overview** — table of S&P 500 / NASDAQ / Dow / VIX / 10Y with prior close, change, pre-market/futures, notes. Session recap, sector expectations, key levels.
2. **Crypto Markets** — table of BTC / ETH / SOL / XRP + top movers. Total market cap, BTC dominance, Fear & Greed. BTC + ETH analysis. Altcoin narratives (2-3 specific theses when crypto is front-and-center). Regulatory backdrop.
3. **ETF Watchlist** — table with Status = ACTIVE / MONITORING / REVERSAL / NEW. Note any status changes since yesterday and why.
4. **Macro Trends** — Fed, inflation, DXY, oil, geopolitical. Include active countdowns/binary events.
5. **Earnings & Corporate Events** — today's reports (EPS/rev estimates, key metric), week ahead, notable guidance.
6. **Insider & Institutional Activity** — notable insider transactions (last 5 trading days), 13F moves, aggregate metrics (e.g. "$X Mag 7 insider selling trailing 2y").
7. **Options Flow & Unusual Activity** — unusual calls/puts table, block trades, IV standouts.
8. **Trading Signals & Opportunities** — today's calendar, week ahead, 52-week highs/lows, updated key levels.
9. **AI & Tech Sector Focus** — NVDA / MSFT / GOOG / AAPL / AMZN / META table. TSMC / semi supply chain. Hyperscaler capex. Notable AI news.
10. **Sector Rotation & Fund Flows** — ETF inflows/outflows (last 5d), rotation signals, breadth indicators.
11. **Signal Accuracy Tracker** — review yesterday's calls (CONFIRMED / INVALIDATED / PENDING), running 7d hit rate, standing theses checklist (AI capex, onshoring, energy transition — retire/add as needed).

## Signal Manifest (STRICT — machine-parsed)

Must appear at the bottom under a `## Signal Manifest` heading and include every sub-section below, even if empty (use `- none`).

### Vocabulary — USE EXACTLY

**Sector status** (one per sector):
`STRONG | BULLISH | RECOVERING | NEUTRAL | WATCH | WEAK | REVERSAL | SHARP REVERSAL | BEARISH`

**Ticker status** (one per ticker):
`bullish | bullish reversal | leaning bullish | hot mover | monitoring | neutral | cautious | leaning bearish | bearish | bearish reversal`

**Macro regime keywords** (include at least one in the Macro paragraph):
`RISK-ON | RISK-OFF | STAGFLATION | GOLDILOCKS | NEUTRAL`
`HAWKISH | DOVISH | NEUTRAL`

### Sub-sections (all required)

```
## Macro
<one paragraph including regime + rate direction keywords>

## Sectors
- Sector Name: STRONG/BULLISH/RECOVERING/NEUTRAL/WATCH/WEAK/REVERSAL/SHARP REVERSAL/BEARISH — reason
- ... (one line per sector)

## Tickers
- TICKER: bullish/bullish reversal/leaning bullish/hot mover/monitoring/neutral/cautious/leaning bearish/bearish/bearish reversal, reason
- ... (one line per ticker)

## Insider Signals
- TICKER: insider_buy / insider_sell, $amount, insider_name/title
- (or "- none")

## Options Signals
- TICKER: unusual_calls / unusual_puts, strike, expiry, volume_oi_ratio
- (or "- none")

## Sector Flows
- Sector: INFLOW / OUTFLOW / NEUTRAL, magnitude (high/medium/low)
- (or "- none")

## Key Watchlist Items
- **Date / time: Event** — why it matters

## Themes
<comma-separated active themes — keep thesis names stable day-to-day>

## Accuracy
- YYYY-MM-DD TICKER call: CONFIRMED / INVALIDATED / PENDING, actual_result
- (or "- none" when there is no prior call to score)
```

## Hard rules

- **File name:** `Market_Intel_YYYY-MM-DD.md` (exact — the parser's filename regex depends on this).
- **Vocabulary:** do NOT invent new status words. Anything outside the strict vocabulary is silently dropped by the parser.
- **Thesis stability:** themes and standing theses should persist across days when still valid. Don't rename a thesis just to sound fresh.
- **Never fabricate data.** If a datapoint is unavailable, write "N/A — reason" and move on. The parser will treat missing sub-sections as empty gracefully.
- **Backward compat:** narrative sections can be colorful; the Signal Manifest must be strict.
- **Sources:** cite inline where practical (ticker-level claims especially).
