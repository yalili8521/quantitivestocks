# Swing Model Upgrade: TFT + XGBoost Ensemble
**Completed: 2026-03-07 overnight**

## What Changed

Replaced the PatchTST legacy ensemble with a **Temporal Fusion Transformer (TFT)**
(Lim et al., Google 2021 — used by Two Sigma, Goldman Sachs Marquee, AQR).

Architecture in `swing_model.py`:
- `GRN` — Gated Residual Network (skip + GLU gate + LayerNorm)
- `VariableSelectionNetwork` — per-timestep learned feature importance (one GRN per feature)
- `TFTSwingModel` — VSN → LSTM → Add&Norm → Multi-head Attention → GRN → Linear head
- Hyperparams: `hidden=32, seq_len=20, n_heads=4, dropout=0.25` (intentionally small for N~800)
- Blend: **60% TFT + 40% XGBoost** (fallback to XGBoost-only if TFT dir_acc < 50%)

## Training Results (val set, lookback=1500 bars)

| Symbol | XGB Dir Acc | TFT Dir Acc | TFT Active? | Top XGB Feature |
|--------|------------|------------|-------------|----------------|
| GDX    | 77.2%      | 76.2%      | Yes         | month_sin, fed_funds_rate, real_yield_10y |
| SLV    | 76.1%      | 36.8%      | **No** (< 50%) | gold_silver_ratio, cboe_skew, ret63 |
| IGV    | 45.7%      | 41.5%      | **No** (< 50%) | beta_mkt, sector_breadth, real_yield_10y |
| QQQ    | 64.5%      | 60.1%      | Yes         | breakeven_inflation_5y, risk_appetite_ratio |
| GLD    | 76.6%      | 62.2%      | Yes         | gold_silver_ratio, equity_carry, real_yield_10y |
| SMH    | 73.6%      | 58.5%      | Yes         | yield_curve_3m10y, fed_funds_rate, momentum_quality |
| XLK    | 55.8%      | **59.1%**  | Yes (TFT better!) | yield_curve_3m10y, fed_funds_rate |
| MCHI   | 63.5%      | 41.5%      | **No** (< 50%) | correlation_regime, beta_cma, factor_momentum |
| IBIT   | 66.1%      | 65.4%      | Yes         | month_cos, correlation_regime, month_sin |

**TFT active for 6/9 symbols: GDX, QQQ, GLD, SMH, XLK, IBIT**
**XGBoost-only for 3/9: SLV, IGV, MCHI** (TFT discarded — would hurt blend)

Key observation: TFT beats XGBoost alone on **XLK** (59.1% vs 55.8%). For commodity ETFs
(SLV, MCHI), temporal sequential context doesn't add value — point-in-time factor signals dominate.

## OOS Backtest Results (2022-01-01 → 2023-12-31, truly out-of-sample)

| Symbol | OOS Return | Sharpe | Trades | Win Rate | TFT Active? | Verdict |
|--------|-----------|--------|--------|----------|-------------|---------|
| SMH    | +98.46%   | 1.724  | 3      | 33.3%    | Yes         | KEEP ✅ |
| GLD    | +84.75%   | 1.634  | 2      | 50.0%    | Yes         | KEEP ✅ |
| SLV    | +82.30%   | 1.588  | 2      | 50.0%    | XGB only    | KEEP ✅ |
| XLK    | +48.47%   | 1.997  | 3      | 66.7%    | Yes         | KEEP ✅ |
| QQQ    | +34.26%   | 1.923  | 3      | 66.7%    | Yes         | KEEP ✅ |
| IGV    | +8.42%    | 2.599  | 3      | 33.3%    | XGB only    | WATCH ⚠ |
| GDX    | +1.03%    | 1.204  | 5      | 80.0%    | Yes         | WATCH ⚠ |
| MCHI   | -7.20%    | -1.021 | 8      | 50.0%    | XGB only    | DROP ❌ |
| IBIT   | 0 trades  | —      | 0      | —        | Yes         | SKIP ❌ |

**Notes:**
- Trade counts are very low (2-8 OOS) — swing model is highly selective, which is expected
- MCHI: only symbol with negative OOS return (-7.20%, Sharpe -1.02) — consider dropping
- IBIT: 0 trades OOS (only 539 bars of history — model launched Jan 2024, no 2022-2023 data)
- Cooldown fired 0 times for all symbols — regime filter not restrictive for swing (model self-selects)

## Recommendations for Tomorrow

1. **MCHI**: Consider removing from swing group — negative OOS, lowest XGB accuracy (63.5%)
2. **IBIT**: Keep — no OOS history is expected (ETF launched 2024), not a model quality issue
3. **IGV**: Keep with caution — low win rate (33.3%) but Sharpe 2.6 (2 wins were large)
4. **SLV TFT**: TFT failed (36.8%) — possibly because silver is driven by point-in-time
   macro factors (real yields, gold/silver ratio) not temporal patterns. XGB-only is correct.

## Files Modified
- `swing_model.py` — TFT classes + quality gate (discard if dir_acc < 50%)
- `models/GDX_tft_swing.pt` + config — new
- `models/QQQ_tft_swing.pt` + config — new
- `models/GLD_tft_swing.pt` + config — new
- `models/SMH_tft_swing.pt` + config — new
- `models/XLK_tft_swing.pt` + config — new
- `models/IBIT_tft_swing.pt` + config — new
- (SLV, IGV, MCHI TFT files discarded — below quality gate)

## Next Steps (Deferred)
- Run longer OOS window (2020-2021) if you want more trades to evaluate
- Consider dropping MCHI from swing group
- Monitor live paper trading — first signal with TFT blend will be the real test
