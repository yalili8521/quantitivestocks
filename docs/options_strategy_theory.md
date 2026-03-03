# Options Trading Strategy — Basic Theories

Summary of the theories and rules used in this project’s options backtester and paper trader. References: **McMillan**, **Natenberg**, **Sinclair**, **Davey**.

---

## 1. Two Strategies

| Strategy | Idea | When to enter |
|----------|------|----------------|
| **A — Directional** | Bet on direction with ITM call or put | LSTM says UP/DOWN with enough confidence; IV not extreme; theta acceptable. |
| **B — Vol expansion straddle** | Buy cheap volatility; profit when vol expands | LSTM says *low* direction confidence; vol model says expansion likely; IV rank low; IV not too rich vs realized. |

---

## 2. Option Pricing & Greeks

- **Pricing**: **Black–Scholes** for synthetic option values (no live options chain needed).
- **IV proxy**: VIX, scaled per symbol (`IV_SCALE`), e.g. SPY ≈ 1:1, QQQ/commodities adjusted.
- **Greeks used**:
  - **Delta**: N(d1) for calls; used to pick **ITM strike** (e.g. delta ≈ 0.68) so the directional bet has enough “stock-like” move.
  - **Theta**: Daily time decay; used to **reject** trades where theta is too large vs premium (e.g. theta &lt; 1.5% of entry cost per day — Natenberg Ch.7).

---

## 3. Volatility Edge (IV vs RV)

- **IV/RV ratio** = Implied Volatility / Realized Volatility (e.g. 20-day realized).
- **Sinclair (Volatility Trading, Ch.2)**:
  - Ratio **&lt; 1.0** → options cheap vs recent realized vol → better for **buying** premium.
  - Ratio **&gt; ~1.25–1.5** → options expensive → avoid buying (or size down).
- **In the code**:
  - Directional: enter only if **IV/RV ≤ 1.75** (don’t overpay for directional premium).
  - Straddle: enter only if **IV/RV ≤ 1.70** (buy vol when it’s not in “fear spike” territory).

---

## 4. IV Rank

- **IV Rank** = where current IV sits in the last N days (e.g. 0–100%).
- **Use**:
  - **Directional**: cap at **IV rank ≤ 55%** — avoid entering when IV is in the top half of its range (expensive).
  - **Straddle**: **IV rank ≤ 25%** — buy straddles when vol is in the **lower** part of its range so expansion has room and premium is cheaper.

---

## 5. Directional Strategy (A) — Theory

- **Instrument**: Single **ITM call** (if UP) or **ITM put** (if DOWN).
- **Strike**: Chosen so **|delta| ≈ 0.68** (more movement per dollar than ATM).
- **Expiry**: e.g. **28 DTE**.
- **Entry**:
  - LSTM **direction** = UP or DOWN and **confidence ≥ threshold** (e.g. 0.10).
  - **IV rank ≤ 55%**, **IV/RV ≤ 1.75**.
  - **Theta budget**: daily theta &lt; 1.5% of entry cost (Natenberg Ch.7).
- **Exit**:
  - **+50%** profit target.
  - **−25%** stop loss.
  - **7 DTE** forced exit (avoid gamma/theta blow-up).
  - **Direction flip**: close if LSTM flips to the opposite direction with confidence ≥ 0.50.

---

## 6. Vol Expansion Straddle (B) — Theory

- **Instrument**: **ATM straddle** (buy call + put, same strike, same expiry).
- **Strike**: **ATM** (round to nearest $5 for underlyings ≥ $100).
- **Expiry**: e.g. **30 DTE**.
- **Entry**:
  - **Direction confidence &lt; 0.45** (no strong directional view — we’re betting on vol, not direction).
  - **Vol model** says expansion likely: **vol_expansion_prob ≥ 0.30** (LSTM vol predictor).
  - **IV rank ≤ 25%** (buy when vol is low).
  - **IV/RV ≤ 1.70** (premium not in extreme fear spike).
- **Exit**:
  - **+80%** on **either** leg → close entire straddle (profit target).
  - **−40%** on **total** straddle value → stop loss.
  - **7 DTE** forced exit.
  - **Leg management (McMillan)**: if one leg is **≥ 3×** the other, **close the losing leg** and let the winning leg run (reduces theta drag, locks in part of the gain).

---

## 7. Position Sizing (Kelly)

- **Kelly criterion** (Sinclair Ch.8, Davey):  
  **f = p − (1−p)/b**  
  - *p* = win probability (here: **confidence** for directional, **vol_expansion_prob** for straddle).  
  - *b* = win/loss ratio = profit_target / stop_loss (e.g. 50%/25% → 2, or 80%/40% → 2).
- **Fractional Kelly**: use **¼ Kelly** to reduce variance and model error (Davey).
- **In the code**:  
  Risk per trade = `max_risk_pct × (0.5 + 0.5 × kelly)` so we always use at least half of the risk budget and scale up with confidence/vol_prob.

---

## 8. Risk & Execution

- **Max risk per trade**: e.g. **2%** of equity (`max_risk_pct`). Position size (cost) is capped by this and the Kelly-derived fraction.
- **Slippage**: **+2%** on buy, **−2%** on sell (worse fill than mid) to make backtest conservative.

---

## 9. Summary Table

| Concept | Source | Role |
|--------|--------|------|
| Black–Scholes | Standard | Price options; derive delta/theta. |
| VIX as IV proxy | Practical | No options chain; scale by symbol. |
| IV/RV ratio | Sinclair Ch.2 | Gate: don’t buy when options are expensive vs realized. |
| IV Rank | Common | Directional: avoid high IV; Straddle: buy low IV. |
| Theta vs cost | Natenberg Ch.7 | Reject directional trades with too much daily decay. |
| Delta ~0.68 ITM | Natenberg | Directional: enough leverage, still option-like. |
| Straddle leg 3× rule | McMillan | Close weak leg when one leg dominates. |
| Kelly + ¼ Kelly | Sinclair Ch.8, Davey | Size by edge (confidence / vol_prob) and payoff ratio. |

This is the set of **basic theories** behind the option trading strategy in this repo.
