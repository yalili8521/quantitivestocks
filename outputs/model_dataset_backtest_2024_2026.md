# Model Dataset — Backtest Results 2024-01-01 → 2026-03-06
Generated: 2026-03-06 | Capital: $100,000 | Provider: Alpaca (intraday) / Yahoo (swing)

---

## Intraday Strategy (LightGBM v2, 24 features)
**Entry**: first-30m close (~10:00 ET) | **Exit**: EOD (~15:30 ET) | **Data**: Alpaca 5-min, 2yr

| Symbol | Total Return | Ann. Return | Sharpe | Max DD  | Win Rate | Trades | Profit Factor | Notes |
|--------|-------------|-------------|--------|---------|----------|--------|---------------|-------|
| SMH    | +368.77%    | +104.02%    | 5.090  | -2.46%  | 86.8%    | 190    | 22.675        | best overall |
| IWM    | +197.92%    | +65.97%     | 4.920  | -4.01%  | 80.7%    | 212    | 7.662         | |
| IGV    | +165.10%    | +48.65%     | 3.618  | -7.21%  | 69.6%    | 293    | 2.897         | |
| GDX    | +143.79%    | +50.99%     | 2.644  | -11.72% | 61.2%    | 371    | 1.651         | higher DD |
| SLV    | +140.15%    | +50.85%     | 2.677  | -17.93% | 67.0%    | 352    | 1.880         | high DD |
| EWT    | +136.54%    | +49.90%     | 3.652  | -3.71%  | 72.2%    | 266    | 4.302         | |
| XLK    | +135.32%    | +57.48%     | 3.561  | -9.04%  | 71.2%    | 260    | 3.790         | |
| XLE    | +136.21%    | +48.91%     | 3.612  | -5.53%  | 70.9%    | 261    | 3.091         | |
| SPY    | +120.19%    | +44.05%     | 3.957  | -3.54%  | 67.7%    | 393    | 2.764         | |
| SOXX   | +101.48%    | +38.17%     | 2.331  | -10.26% | 79.4%    | 102    | 5.968         | |
| USO    | +104.14%    | +39.69%     | 3.195  | -5.35%  | 79.0%    | 124    | 6.096         | |
| QQQ    | +83.09%     | +32.33%     | 3.378  | -2.74%  | 88.5%    | 122    | 9.926         | |
| EEM    | +63.19%     | +26.06%     | 3.372  | -3.89%  | 71.9%    | 263    | 2.641         | |
| GLD    | +59.44%     | +24.07%     | 3.022  | -7.14%  | 63.1%    | 347    | 1.978         | |
| EWJ    | +44.01%     | +18.70%     | 2.954  | -1.77%  | 84.0%    | 100    | 8.480         | |
| EWS    | +40.69%     | +17.84%     | 1.851  | -3.92%  | 54.2%    | 421    | 1.567         | low win rate |
| MCHI   | +30.16%     | —           | 2.649  | —       | 72.4%    | 134    | 2.678         | <60% return |
| INDA   | +24.10%     | +10.68%     | 2.303  | -3.31%  | 72.1%    | 172    | 3.063         | low return |
| IBIT   | +0.00%      | —           | 0.000  | —       | 0.0%     | 0      | —             | no Alpaca data |

---

## Swing Strategy (XGBoost daily regression)
**Entry/Exit**: signal-decay exits | **Data**: Yahoo daily, 2yr

| Symbol | Total Return | Ann. Return | Sharpe | Max DD  | Win Rate | Trades | Profit Factor | Notes |
|--------|-------------|-------------|--------|---------|----------|--------|---------------|-------|
| GDX    | +187.73%    | —           | 1.689  | -19.10% | 50.0%    | 2      | 32.910        | too few trades |
| SLV    | +117.79%    | —           | 1.338  | -30.68% | 50.0%    | 2      | 38.063        | too few trades + high DD |
| IGV    | +100.48%    | —           | 2.690  | -7.26%  | 70.6%    | 34     | 6.833         | |
| QQQ    | +91.48%     | —           | 2.783  | -3.26%  | 72.7%    | 22     | 19.993        | |
| GLD    | +88.30%     | —           | 1.999  | -11.12% | 50.0%    | 2      | 53.908        | too few trades |
| SMH    | +87.59%     | —           | 1.191  | -28.15% | 100.0%   | 1      | 999.000       | 1 trade only |
| XLK    | +72.78%     | —           | 2.261  | -10.15% | 81.8%    | 11     | 513.403       | |
| MCHI   | +67.58%     | —           | 1.976  | -5.08%  | 74.2%    | 31     | 16.798        | |
| IBIT   | +64.44%     | —           | 2.337  | -7.91%  | 65.0%    | 20     | 7.424         | |
| ITA    | +49.13%     | —           | 1.660  | -8.88%  | 50.0%    | 2      | 20.481        | too few trades |
| EWS    | +44.94%     | —           | 1.839  | -7.29%  | 58.3%    | 24     | 3.845         | |
| SPY    | +43.57%     | —           | 2.207  | -4.56%  | 62.5%    | 8      | 22.673        | few trades |
| SOXX   | +39.57%     | —           | 0.924  | -18.97% | 50.0%    | 2      | 13.955        | high DD + few trades |
| XLF    | +30.12%     | —           | 1.766  | -5.38%  | 68.2%    | 22     | 4.630         | |
| FXI    | +26.81%     | —           | 1.139  | -8.81%  | 33.3%    | 6      | 7.703         | low win rate |
| EWT    | +28.74%     | —           | 1.053  | -13.01% | 50.0%    | 2      | 17.407        | too few trades |
| IWM    | +20.42%     | —           | 1.144  | -5.44%  | 70.0%    | 10     | 35.596        | |
| XLI    | +20.85%     | —           | 1.280  | -8.15%  | 100.0%   | 1      | 999.000       | 1 trade only |
| EEM    | +14.88%     | —           | 1.119  | -6.02%  | 50.0%    | 2      | 17.391        | too few trades |
| EWJ    | +15.62%     | —           | 0.909  | -5.87%  | 100.0%   | 1      | 999.000       | 1 trade only |
| USO    | +15.01%     | —           | 2.247  | -1.27%  | 76.7%    | 30     | 10.995        | |
| XLE    | +7.50%      | —           | 0.801  | -4.28%  | 50.0%    | 2      | 12.619        | too few trades |
| EWZ    | +2.26%      | —           | 0.383  | -2.44%  | 44.4%    | 9      | 1.907         | weak signal |
| XLV    | +0.57%      | —           | 0.193  | -2.18%  | 33.3%    | 3      | 1.651         | weak signal |
| INDA   | +0.78%      | —           | 0.121  | -4.90%  | 100.0%   | 1      | 999.000       | 1 trade only |
| TLT    | +0.88%      | —           | 0.698  | -0.46%  | 81.2%    | 16     | 1.983         | weak signal |

---

## Decision Framework

### Selection rule: total return > 60% over 2024-2026 backtest period

### Intraday group (LightGBM v2) — FINAL
✅ **13 symbols**: SMH, IWM, IGV, GDX, SLV, EWT, XLK, XLE, SPY, SOXX, USO, QQQ, EEM

Cross-test additions (swing symbols tested intraday):
- SMH: +369% Sharpe 5.09 ← **best in universe**
- IGV: +165% Sharpe 3.62 ← added
- GDX: +144% Sharpe 2.64 ← added (DD -11.7%, acceptable)
- XLK: +135% Sharpe 3.56 ← added
- MCHI: +30% ← dropped (<60%)
- IBIT: 0 trades ← dropped (no Alpaca intraday data)

### Swing group (XGBoost daily) — FINAL
✅ **9 symbols**: GDX, SLV, IGV, QQQ, GLD, SMH, XLK, MCHI, IBIT

Note: GDX/SLV/IGV/QQQ/SMH/XLK appear in both groups intentionally (separate Alpaca accounts, separate models, different time horizons)

### Expansion group — RETIRED
The expansion XGBoost factor model (EWJ, EWS, XLE, INDA) produced <60% returns on its factor feature set.
Those symbols are either covered by intraday (EWT +137%, XLE +136%, EEM +63%) or insufficient (EWJ +44%, EWS +41%, INDA +24%).
The "expansion" group key is preserved in code for CLI compatibility; add symbols when a new strategy qualifies.

### Flagged — too few trades (<5) to trust signal (swing)
⚠️  GDX (2), SLV (2), GLD (2), SMH (1), EWT (2), EWJ (1), EEM (2), XLE (2), ITA (2), XLI (1), INDA (1)

### Drop — weak signal
❌ EWZ (Sharpe 0.38), XLV (0.19), INDA swing (0.12), EWJ swing (0.91), XLE swing (0.80), SOXX swing (0.92), FXI (33% WR)
