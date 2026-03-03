#!/usr/bin/env python3
"""
Generate tables of training and backtest metrics (return, accuracy, etc.).

Collects:
  - Equity LSTM: val_loss, val_acc (from models/*_lstm*_metrics.json)
  - Equity Meta RF: val_accuracy, threshold (from models/*_meta_rf*_config.json)
  - Vol LSTM: val_loss, val_acc (from models/*_vol_lstm_metrics.json)
  - Vol Meta RF: val_accuracy (from models/*_vol_meta_rf_config.json)
  - Backtest: return %, Sharpe, max DD, trades, win rate (from outputs/backtest_*_summary.json
    or computed from outputs/backtest_*.csv if no summary)

Outputs:
  - outputs/training_results.csv  (combined table)
  - outputs/training_results.html (readable HTML tables)

Usage:
  python scripts/generate_training_tables.py
  python main.py training-tables   # if wired in main
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")


def _load_json(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def collect_lstm_metrics() -> pd.DataFrame:
    rows = []
    if not os.path.isdir(MODELS_DIR):
        return pd.DataFrame()
    for fname in os.listdir(MODELS_DIR):
        if "_metrics.json" in fname and "vol_lstm" in fname:
            d = _load_json(os.path.join(MODELS_DIR, fname))
            if d:
                rows.append({
                    "symbol": d.get("symbol", ""),
                    "model_type": "vol_lstm",
                    "best_val_loss": d.get("best_val_loss"),
                    "best_val_acc": d.get("best_val_acc"),
                    "epochs_run": d.get("epochs_run"),
                })
        elif "_metrics.json" in fname and "vol" not in fname:
            d = _load_json(os.path.join(MODELS_DIR, fname))
            if d:
                rows.append({
                    "symbol": d.get("symbol", ""),
                    "mode": d.get("mode", ""),
                    "model_type": "lstm",
                    "best_val_loss": d.get("best_val_loss"),
                    "best_val_acc": d.get("best_val_acc"),
                    "epochs_run": d.get("epochs_run"),
                })
    return pd.DataFrame(rows)


def collect_meta_configs() -> pd.DataFrame:
    rows = []
    if not os.path.isdir(MODELS_DIR):
        return pd.DataFrame()
    for fname in os.listdir(MODELS_DIR):
        if "_config.json" not in fname or "meta" not in fname:
            continue
        d = _load_json(os.path.join(MODELS_DIR, fname))
        if not d:
            continue
        is_vol = "vol_meta" in fname
        row = {
            "symbol": d.get("symbol", ""),
            "val_accuracy": d.get("val_accuracy"),
            "val_precision": d.get("val_precision"),
            "val_coverage": d.get("val_coverage"),
            "threshold": d.get("threshold"),
        }
        if is_vol:
            row["model_type"] = "vol_meta_rf"
        else:
            row["mode"] = d.get("mode", "")
            row["model_type"] = "meta_rf"
        rows.append(row)
    return pd.DataFrame(rows)


def collect_backtest_summaries() -> pd.DataFrame:
    rows = []
    if not os.path.isdir(OUTPUTS_DIR):
        return pd.DataFrame()
    for fname in os.listdir(OUTPUTS_DIR):
        if not fname.startswith("backtest_") or not fname.endswith("_summary.json"):
            continue
        symbol = fname.replace("backtest_", "").replace("_summary.json", "")
        d = _load_json(os.path.join(OUTPUTS_DIR, fname))
        if d:
            rows.append({
                "symbol": symbol,
                "mode": d.get("mode", "daily"),
                "start_date": d.get("start_date"),
                "end_date": d.get("end_date"),
                "initial_capital": d.get("initial_capital"),
                "final_equity": d.get("final_equity"),
                "total_return_pct": d.get("total_return_pct"),
                "annualized_return_pct": d.get("annualized_return_pct"),
                "sharpe_ratio": d.get("sharpe_ratio"),
                "max_drawdown_pct": d.get("max_drawdown_pct"),
                "total_trades": d.get("total_trades"),
                "win_rate": d.get("win_rate"),
                "avg_win_pct": d.get("avg_win_pct"),
                "avg_loss_pct": d.get("avg_loss_pct"),
                "profit_factor": d.get("profit_factor"),
                "avg_trade_duration_days": d.get("avg_trade_duration_days"),
            })
    # Fallback: compute return from equity curve CSV if no summary
    for fname in os.listdir(OUTPUTS_DIR):
        if fname.startswith("backtest_") and fname.endswith(".csv") and "_summary" not in fname:
            symbol = fname.replace("backtest_", "").replace(".csv", "")
            if any(r.get("symbol") == symbol for r in rows):
                continue
            csv_path = os.path.join(OUTPUTS_DIR, fname)
            try:
                df = pd.read_csv(csv_path)
                if "equity" in df.columns and len(df) >= 2:
                    initial = df["equity"].iloc[0]
                    final = df["equity"].iloc[-1]
                    ret = (final / initial - 1) * 100 if initial and initial > 0 else None
                    start = df["date"].iloc[0] if "date" in df.columns else ""
                    end = df["date"].iloc[-1] if "date" in df.columns else ""
                    rows.append({
                        "symbol": symbol,
                        "mode": "daily",
                        "start_date": start,
                        "end_date": end,
                        "total_return_pct": round(ret, 2) if ret is not None else None,
                        "final_equity": final,
                        "sharpe_ratio": None,
                        "max_drawdown_pct": None,
                        "total_trades": None,
                        "win_rate": None,
                    })
            except Exception:
                pass
    return pd.DataFrame(rows)


def build_equity_training_table(lstm_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """One row per symbol+mode: LSTM metrics + meta metrics."""
    lstm_eq = lstm_df[(lstm_df["model_type"] == "lstm")] if "model_type" in lstm_df.columns and not lstm_df.empty else lstm_df
    meta_eq = meta_df[(meta_df["model_type"] == "meta_rf")] if "model_type" in meta_df.columns and not meta_df.empty else meta_df
    if meta_eq.empty and lstm_eq.empty:
        return pd.DataFrame()
    if lstm_eq.empty:
        cols = [c for c in ["symbol", "mode", "val_accuracy", "val_precision", "val_coverage", "threshold"] if c in meta_eq.columns]
        return meta_eq[cols].copy() if cols else pd.DataFrame()
    if meta_eq.empty:
        return lstm_eq[["symbol", "mode", "best_val_loss", "best_val_acc", "epochs_run"]].copy() if "symbol" in lstm_eq.columns else pd.DataFrame()
    if "symbol" not in lstm_eq.columns or "mode" not in lstm_eq.columns or "symbol" not in meta_eq.columns or "mode" not in meta_eq.columns:
        return pd.DataFrame()
    merged = lstm_eq.merge(meta_eq, on=["symbol", "mode"], how="outer", suffixes=("_lstm", "_meta"))
    cols = ["symbol", "mode", "best_val_loss", "best_val_acc", "epochs_run", "val_accuracy", "threshold"]
    return merged[[c for c in cols if c in merged.columns]]


def build_vol_training_table(lstm_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """One row per symbol: vol LSTM + vol meta."""
    vol_lstm = lstm_df[(lstm_df["model_type"] == "vol_lstm")] if "model_type" in lstm_df.columns and not lstm_df.empty else pd.DataFrame()
    vol_meta = meta_df[(meta_df["model_type"] == "vol_meta_rf")] if "model_type" in meta_df.columns and not meta_df.empty else pd.DataFrame()
    if vol_lstm.empty and vol_meta.empty:
        return pd.DataFrame()
    if vol_lstm.empty:
        return vol_meta
    if vol_meta.empty:
        return vol_lstm
    if "symbol" not in vol_lstm.columns or "symbol" not in vol_meta.columns:
        return pd.DataFrame()
    merged = vol_lstm.merge(vol_meta, on="symbol", how="outer", suffixes=("", "_meta"))
    cols = ["symbol", "best_val_loss", "best_val_acc", "epochs_run", "val_accuracy", "val_precision", "val_coverage", "threshold"]
    return merged[[c for c in cols if c in merged.columns]]


def _table_html(title: str, df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    html = f"  <h2>{title}</h2>\n  <table>\n    <thead><tr>\n"
    for c in df.columns:
        html += f"      <th>{c}</th>\n"
    html += "    </tr></thead>\n    <tbody>\n"
    for _, row in df.iterrows():
        html += "    <tr>\n"
        for c in df.columns:
            v = row[c]
            if pd.isna(v):
                v = "—"
            elif isinstance(v, float):
                v = f"{v:.4f}" if 0 < abs(v) < 1 else f"{v:.2f}"
            html += f"      <td class=\"num\">{v}</td>\n"
        html += "    </tr>\n"
    html += "    </tbody>\n  </table>\n"
    return html


def write_html(by_mode: dict) -> str:
    """by_mode = { 'daily': { 'equity_training': df, 'backtest': df }, 'intraday': {...}, 'vol': { 'vol_training': df } }"""
    html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Training &amp; Backtest Results</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; background: #0f172a; color: #e2e8f0; }
    h1 { font-size: 1.5rem; margin-bottom: 8px; }
    h2 { font-size: 1.1rem; margin: 24px 0 12px; color: #94a3b8; }
    h3 { font-size: 1rem; margin: 20px 0 8px; color: #64748b; }
    table { border-collapse: collapse; width: 100%; max-width: 1200px; font-size: 13px; }
    th, td { border: 1px solid #334155; padding: 8px 10px; text-align: left; }
    th { background: #1e293b; color: #94a3b8; font-weight: 600; }
    tr:nth-child(even) { background: rgba(30,41,59,0.5); }
    .num { text-align: right; }
  </style>
</head>
<body>
  <h1>Training &amp; Backtest Results</h1>
  <p>Organized by mode: Daily, Intraday, Vol (options).</p>
"""
    for mode_key in ("daily", "intraday", "vol"):
        blocks = by_mode.get(mode_key) or {}
        if not blocks:
            continue
        label = mode_key.capitalize()
        html += f"  <h3>—— {label} ——</h3>\n"
        if "equity_training" in blocks and not blocks["equity_training"].empty:
            html += _table_html(f"Equity LSTM + Meta ({label})", blocks["equity_training"])
        if "backtest" in blocks and not blocks["backtest"].empty:
            html += _table_html(f"Backtest ({label})", blocks["backtest"])
        if "vol_training" in blocks and not blocks["vol_training"].empty:
            html += _table_html("Vol LSTM + Meta (options timing)", blocks["vol_training"])
    html += "</body>\n</html>"
    return html


def _df_to_records(df: pd.DataFrame) -> list:
    if df.empty:
        return []
    out = []
    for row in df.to_dict(orient="records"):
        rec = {}
        for k, v in row.items():
            if pd.isna(v):
                rec[k] = None
            elif isinstance(v, (int,)):
                rec[k] = int(v)
            elif isinstance(v, float):
                rec[k] = round(v, 6) if abs(v) < 1e10 else v
            else:
                rec[k] = v
        out.append(rec)
    return out


def main() -> None:
    sys.path.insert(0, PROJECT_ROOT)

    lstm_df = collect_lstm_metrics()
    meta_df = collect_meta_configs()
    backtest_df = collect_backtest_summaries()

    equity_table = build_equity_training_table(lstm_df, meta_df)
    vol_table = build_vol_training_table(lstm_df, meta_df)

    # Split by mode for CSVs and HTML/JSON (missing mode → treat as daily)
    if not equity_table.empty and "mode" in equity_table.columns:
        equity_daily = equity_table[equity_table["mode"] == "daily"].copy()
        equity_intraday = equity_table[equity_table["mode"] == "intraday"].copy()
    else:
        equity_daily = equity_table.copy() if not equity_table.empty else pd.DataFrame()
        equity_intraday = pd.DataFrame()
    if not backtest_df.empty and "mode" in backtest_df.columns:
        backtest_daily = backtest_df[backtest_df["mode"] == "daily"].copy()
        backtest_intraday = backtest_df[backtest_df["mode"] == "intraday"].copy()
    else:
        backtest_daily = backtest_df.copy() if not backtest_df.empty else pd.DataFrame()
        backtest_intraday = pd.DataFrame()

    by_mode = {
        "daily": {"equity_training": equity_daily, "backtest": backtest_daily},
        "intraday": {"equity_training": equity_intraday, "backtest": backtest_intraday},
        "vol": {"vol_training": vol_table},
    }

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # CSVs per mode type
    if not equity_daily.empty:
        equity_daily.to_csv(os.path.join(OUTPUTS_DIR, "training_results_equity_daily.csv"), index=False)
        print("  Wrote outputs/training_results_equity_daily.csv")
    if not equity_intraday.empty:
        equity_intraday.to_csv(os.path.join(OUTPUTS_DIR, "training_results_equity_intraday.csv"), index=False)
        print("  Wrote outputs/training_results_equity_intraday.csv")
    if not vol_table.empty:
        vol_table.to_csv(os.path.join(OUTPUTS_DIR, "training_results_vol.csv"), index=False)
        print("  Wrote outputs/training_results_vol.csv")
    if not backtest_daily.empty:
        backtest_daily.to_csv(os.path.join(OUTPUTS_DIR, "training_results_backtest_daily.csv"), index=False)
        print("  Wrote outputs/training_results_backtest_daily.csv")
    if not backtest_intraday.empty:
        backtest_intraday.to_csv(os.path.join(OUTPUTS_DIR, "training_results_backtest_intraday.csv"), index=False)
        print("  Wrote outputs/training_results_backtest_intraday.csv")

    html_path = os.path.join(OUTPUTS_DIR, "training_results.html")
    with open(html_path, "w") as f:
        f.write(write_html(by_mode))
    print(f"  Wrote {html_path}")

    # Single JSON file organized by mode
    tables_json = {
        "daily": {
            "equity_training": _df_to_records(equity_daily),
            "backtest": _df_to_records(backtest_daily),
        },
        "intraday": {
            "equity_training": _df_to_records(equity_intraday),
            "backtest": _df_to_records(backtest_intraday),
        },
        "vol": {
            "vol_training": _df_to_records(vol_table),
        },
    }
    json_path = os.path.join(OUTPUTS_DIR, "training_tables.json")
    with open(json_path, "w") as f:
        json.dump(tables_json, f, indent=2)
    print(f"  Wrote {json_path}")

    if equity_table.empty and vol_table.empty and backtest_df.empty:
        print("  No training or backtest data found. Run train/backtest first to generate metrics.")


if __name__ == "__main__":
    main()
