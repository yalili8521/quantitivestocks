#!/usr/bin/env python3
"""
A/B Exit Variant Comparison Dashboard
======================================
Loads trade CSVs from variant-specific dirs (outputs/trades_A/, outputs/trades_B/),
computes per-variant metrics, and generates an interactive HTML report.

Usage:
    python scripts/analyze_exits_ab.py
    python scripts/analyze_exits_ab.py --variants A,B --output outputs/ab_report.html
    python scripts/analyze_exits_ab.py --include-base   # also compare against base params
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_variant_trades(variant: str) -> pd.DataFrame:
    """Load all trade CSVs for a given variant (A, B, or 'base')."""
    if variant == "base":
        trades_dir = os.path.join(OUTPUTS_DIR, "trades")
    else:
        trades_dir = os.path.join(OUTPUTS_DIR, f"trades_{variant}")

    if not os.path.isdir(trades_dir):
        print(f"  WARNING: {trades_dir} not found, skipping variant '{variant}'")
        return pd.DataFrame()

    frames = []
    for fname in sorted(os.listdir(trades_dir)):
        if not fname.endswith(".csv") or not fname.startswith("trades_"):
            continue
        fpath = os.path.join(trades_dir, fname)
        try:
            df = pd.read_csv(fpath)
            # Ensure variant column exists
            if "variant" not in df.columns:
                df["variant"] = variant
            frames.append(df)
        except Exception as e:
            print(f"  WARNING: failed to read {fpath}: {e}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_variant_summaries(variant: str) -> list[dict]:
    """Load all backtest summary JSONs for a variant."""
    if variant == "base":
        bt_dir = os.path.join(OUTPUTS_DIR, "backtests")
    else:
        bt_dir = os.path.join(OUTPUTS_DIR, f"backtests_{variant}")

    if not os.path.isdir(bt_dir):
        return []

    summaries = []
    for fname in sorted(os.listdir(bt_dir)):
        if not fname.endswith("_summary.json"):
            continue
        fpath = os.path.join(bt_dir, fname)
        try:
            with open(fpath) as f:
                s = json.load(f)
            s["variant"] = variant
            summaries.append(s)
        except Exception:
            pass
    return summaries


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute aggregate metrics from a trades DataFrame."""
    if df.empty:
        return {}

    n = len(df)
    wins = df[df["return_pct"] > 0]
    losses = df[df["return_pct"] <= 0]

    win_rate = len(wins) / n if n > 0 else 0
    avg_return = df["return_pct"].mean()
    avg_win = wins["return_pct"].mean() if len(wins) > 0 else 0
    avg_loss = losses["return_pct"].mean() if len(losses) > 0 else 0

    # Profit factor
    gross_win = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe-like ratio on trade returns
    std_ret = df["return_pct"].std()
    trade_sharpe = avg_return / std_ret if std_ret > 0 else 0

    # MFE/MAE stats
    has_mfe = "mfe_pct" in df.columns and df["mfe_pct"].notna().any()
    has_mae = "mae_pct" in df.columns and df["mae_pct"].notna().any()

    avg_mfe = df["mfe_pct"].mean() if has_mfe else None
    avg_mae = df["mae_pct"].mean() if has_mae else None
    median_mfe = df["mfe_pct"].median() if has_mfe else None
    median_mae = df["mae_pct"].median() if has_mae else None

    # MFE retention: what fraction of MFE was captured as final return
    if has_mfe:
        mfe_positive = df[df["mfe_pct"] > 0]
        if len(mfe_positive) > 0:
            retention = (mfe_positive["return_pct"] / mfe_positive["mfe_pct"]).clip(-5, 5)
            mfe_retention = retention.mean()
        else:
            mfe_retention = 0
    else:
        mfe_retention = None

    # Exit layer breakdown
    layer_counts = {}
    if "exit_layer" in df.columns:
        layer_counts = df["exit_layer"].value_counts().to_dict()

    # Per-exit-layer win rate
    layer_winrate = {}
    if "exit_layer" in df.columns:
        for layer in df["exit_layer"].unique():
            ldf = df[df["exit_layer"] == layer]
            if len(ldf) > 0:
                layer_winrate[layer] = len(ldf[ldf["return_pct"] > 0]) / len(ldf)

    return {
        "trades": n,
        "win_rate": round(win_rate * 100, 1),
        "avg_return_pct": round(avg_return, 3),
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "trade_sharpe": round(trade_sharpe, 3),
        "total_pnl": round(df["pnl"].sum(), 2),
        "avg_mfe_pct": round(avg_mfe, 3) if avg_mfe is not None else "-",
        "avg_mae_pct": round(avg_mae, 3) if avg_mae is not None else "-",
        "median_mfe_pct": round(median_mfe, 3) if median_mfe is not None else "-",
        "median_mae_pct": round(median_mae, 3) if median_mae is not None else "-",
        "mfe_retention_pct": round(mfe_retention * 100, 1) if mfe_retention is not None else "-",
        "exit_layer_counts": layer_counts,
        "exit_layer_winrate": {k: round(v * 100, 1) for k, v in layer_winrate.items()},
    }


def compute_per_symbol_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics broken down by symbol."""
    if df.empty:
        return pd.DataFrame()

    rows = []
    for sym in sorted(df["symbol"].unique()):
        sdf = df[df["symbol"] == sym]
        m = compute_metrics(sdf)
        m["symbol"] = sym
        m["tier"] = sdf["tier"].iloc[0] if "tier" in sdf.columns else ""
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def generate_html_report(
    variant_data: dict[str, pd.DataFrame],
    variant_metrics: dict[str, dict],
    variant_per_symbol: dict[str, pd.DataFrame],
    output_path: str,
) -> str:
    """Generate interactive HTML comparison dashboard."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        has_plotly = True
    except ImportError:
        has_plotly = False
        print("  WARNING: plotly not installed, generating table-only report")

    variants = sorted(variant_data.keys())
    colors = {"A": "#2196F3", "B": "#FF9800", "base": "#4CAF50"}

    # --- Build HTML ---
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<title>Exit Variant A/B Comparison</title>",
        "<style>",
        "  body { font-family: 'Segoe UI', sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; }",
        "  h1 { color: #00d4ff; }",
        "  h2 { color: #bb86fc; margin-top: 30px; }",
        "  table { border-collapse: collapse; margin: 10px 0; width: 100%; }",
        "  th, td { border: 1px solid #333; padding: 8px 12px; text-align: right; }",
        "  th { background: #16213e; color: #00d4ff; }",
        "  td { background: #0f3460; }",
        "  tr:nth-child(even) td { background: #1a1a2e; }",
        "  .metric-label { text-align: left; font-weight: bold; }",
        "  .winner { color: #4CAF50; font-weight: bold; }",
        "  .loser { color: #f44336; }",
        "  .chart-container { margin: 20px 0; }",
        "</style>",
        "</head><body>",
        "<h1>Exit Variant A/B Comparison</h1>",
    ]

    # --- Summary table ---
    html_parts.append("<h2>Aggregate Metrics</h2>")
    html_parts.append("<table>")
    header = "<tr><th class='metric-label'>Metric</th>"
    for v in variants:
        header += f"<th>Variant {v}</th>"
    header += "</tr>"
    html_parts.append(header)

    metric_labels = [
        ("trades", "Total Trades"),
        ("win_rate", "Win Rate (%)"),
        ("avg_return_pct", "Avg Return (%)"),
        ("avg_win_pct", "Avg Win (%)"),
        ("avg_loss_pct", "Avg Loss (%)"),
        ("profit_factor", "Profit Factor"),
        ("trade_sharpe", "Trade Sharpe"),
        ("total_pnl", "Total P&L ($)"),
        ("avg_mfe_pct", "Avg MFE (%)"),
        ("avg_mae_pct", "Avg MAE (%)"),
        ("median_mfe_pct", "Median MFE (%)"),
        ("median_mae_pct", "Median MAE (%)"),
        ("mfe_retention_pct", "MFE Retention (%)"),
    ]

    # Determine winner for numeric metrics (higher = better, except avg_loss and avg_mae)
    lower_is_better = {"avg_loss_pct", "avg_mae_pct", "median_mae_pct"}

    for key, label in metric_labels:
        row = f"<tr><td class='metric-label'>{label}</td>"
        vals = {}
        for v in variants:
            val = variant_metrics.get(v, {}).get(key, "-")
            vals[v] = val
            row += f"<td>{val}</td>"
        row += "</tr>"
        html_parts.append(row)

    html_parts.append("</table>")

    # --- Exit layer breakdown ---
    html_parts.append("<h2>Exit Layer Breakdown</h2>")
    all_layers = set()
    for v in variants:
        all_layers.update(variant_metrics.get(v, {}).get("exit_layer_counts", {}).keys())
    all_layers = sorted(all_layers)

    if all_layers:
        html_parts.append("<table>")
        header = "<tr><th class='metric-label'>Exit Layer</th>"
        for v in variants:
            header += f"<th>Variant {v} (count)</th><th>Variant {v} (WR%)</th>"
        header += "</tr>"
        html_parts.append(header)
        for layer in all_layers:
            row = f"<tr><td class='metric-label'>{layer}</td>"
            for v in variants:
                count = variant_metrics.get(v, {}).get("exit_layer_counts", {}).get(layer, 0)
                wr = variant_metrics.get(v, {}).get("exit_layer_winrate", {}).get(layer, "-")
                row += f"<td>{count}</td><td>{wr}</td>"
            row += "</tr>"
            html_parts.append(row)
        html_parts.append("</table>")

    # --- Per-symbol comparison ---
    html_parts.append("<h2>Per-Symbol Comparison</h2>")
    all_symbols = set()
    for v in variants:
        if v in variant_per_symbol and not variant_per_symbol[v].empty:
            all_symbols.update(variant_per_symbol[v]["symbol"].tolist())
    all_symbols = sorted(all_symbols)

    if all_symbols:
        html_parts.append("<table>")
        header = "<tr><th class='metric-label'>Symbol</th><th>Tier</th>"
        for v in variants:
            header += (f"<th>{v} Trades</th><th>{v} WR%</th>"
                       f"<th>{v} AvgRet%</th><th>{v} PF</th><th>{v} P&L</th>")
        header += "</tr>"
        html_parts.append(header)
        for sym in all_symbols:
            row = f"<tr><td class='metric-label'>{sym}</td>"
            tier_val = ""
            for v in variants:
                psdf = variant_per_symbol.get(v, pd.DataFrame())
                if not psdf.empty and sym in psdf["symbol"].values:
                    srow = psdf[psdf["symbol"] == sym].iloc[0]
                    if not tier_val:
                        tier_val = str(srow.get("tier", ""))
                    row_data = (f"<td>{srow.get('trades', 0)}</td>"
                                f"<td>{srow.get('win_rate', '-')}</td>"
                                f"<td>{srow.get('avg_return_pct', '-')}</td>"
                                f"<td>{srow.get('profit_factor', '-')}</td>"
                                f"<td>{srow.get('total_pnl', '-')}</td>")
                else:
                    row_data = "<td>-</td>" * 5
                row += row_data
            row = row[:row.index("</td>") + 5]  # insert tier after symbol
            # Rebuild with tier
            row = f"<tr><td class='metric-label'>{sym}</td><td>{tier_val}</td>"
            for v in variants:
                psdf = variant_per_symbol.get(v, pd.DataFrame())
                if not psdf.empty and sym in psdf["symbol"].values:
                    srow = psdf[psdf["symbol"] == sym].iloc[0]
                    row += (f"<td>{srow.get('trades', 0)}</td>"
                            f"<td>{srow.get('win_rate', '-')}</td>"
                            f"<td>{srow.get('avg_return_pct', '-')}</td>"
                            f"<td>{srow.get('profit_factor', '-')}</td>"
                            f"<td>{srow.get('total_pnl', '-')}</td>")
                else:
                    row += "<td>-</td>" * 5
            row += "</tr>"
            html_parts.append(row)
        html_parts.append("</table>")

    # --- Plotly charts ---
    if has_plotly and len(variants) >= 2:
        # Chart 1: MFE retention boxplot
        fig1 = go.Figure()
        for v in variants:
            df = variant_data.get(v, pd.DataFrame())
            if df.empty or "mfe_pct" not in df.columns:
                continue
            valid = df[(df["mfe_pct"] > 0) & df["return_pct"].notna()]
            if valid.empty:
                continue
            retention = (valid["return_pct"] / valid["mfe_pct"]).clip(-3, 3) * 100
            fig1.add_trace(go.Box(
                y=retention, name=f"Variant {v}",
                marker_color=colors.get(v, "#999"),
            ))
        fig1.update_layout(
            title="MFE Retention (% of peak captured)",
            yaxis_title="Retention %",
            template="plotly_dark",
            height=400,
        )
        html_parts.append("<div class='chart-container'>")
        html_parts.append(fig1.to_html(full_html=False, include_plotlyjs="cdn"))
        html_parts.append("</div>")

        # Chart 2: Exit layer bar chart
        if all_layers:
            fig2 = go.Figure()
            for v in variants:
                counts = variant_metrics.get(v, {}).get("exit_layer_counts", {})
                fig2.add_trace(go.Bar(
                    x=all_layers,
                    y=[counts.get(l, 0) for l in all_layers],
                    name=f"Variant {v}",
                    marker_color=colors.get(v, "#999"),
                ))
            fig2.update_layout(
                title="Exit Layer Distribution",
                xaxis_title="Exit Layer",
                yaxis_title="Count",
                barmode="group",
                template="plotly_dark",
                height=400,
            )
            html_parts.append("<div class='chart-container'>")
            html_parts.append(fig2.to_html(full_html=False, include_plotlyjs=False))
            html_parts.append("</div>")

        # Chart 3: MAE distribution histogram
        fig3 = go.Figure()
        for v in variants:
            df = variant_data.get(v, pd.DataFrame())
            if df.empty or "mae_pct" not in df.columns:
                continue
            fig3.add_trace(go.Histogram(
                x=df["mae_pct"].dropna(),
                name=f"Variant {v}",
                marker_color=colors.get(v, "#999"),
                opacity=0.6,
                nbinsx=30,
            ))
        fig3.update_layout(
            title="MAE Distribution (Max Adverse Excursion %)",
            xaxis_title="MAE %",
            yaxis_title="Count",
            barmode="overlay",
            template="plotly_dark",
            height=400,
        )
        html_parts.append("<div class='chart-container'>")
        html_parts.append(fig3.to_html(full_html=False, include_plotlyjs=False))
        html_parts.append("</div>")

        # Chart 4: Cumulative P&L per variant
        fig4 = go.Figure()
        for v in variants:
            df = variant_data.get(v, pd.DataFrame())
            if df.empty or "pnl" not in df.columns:
                continue
            # Sort by exit_date for cumulative
            if "exit_date" in df.columns:
                df_sorted = df.sort_values("exit_date")
            else:
                df_sorted = df
            cum_pnl = df_sorted["pnl"].cumsum()
            fig4.add_trace(go.Scatter(
                x=list(range(len(cum_pnl))),
                y=cum_pnl,
                name=f"Variant {v}",
                line=dict(color=colors.get(v, "#999"), width=2),
            ))
        fig4.update_layout(
            title="Cumulative P&L by Trade Number",
            xaxis_title="Trade #",
            yaxis_title="Cumulative P&L ($)",
            template="plotly_dark",
            height=400,
        )
        html_parts.append("<div class='chart-container'>")
        html_parts.append(fig4.to_html(full_html=False, include_plotlyjs=False))
        html_parts.append("</div>")

    html_parts.extend([
        "<p style='color:#666; margin-top:30px;'>Generated by analyze_exits_ab.py</p>",
        "</body></html>",
    ])

    html = "\n".join(html_parts)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="A/B Exit Variant Comparison Dashboard")
    parser.add_argument("--variants", default="A,B",
                        help="Comma-separated variant names to compare (default: A,B)")
    parser.add_argument("--include-base", action="store_true",
                        help="Also include base (non-variant) trades for comparison")
    parser.add_argument("--output", default=None,
                        help="Output HTML path (default: outputs/ab_exit_report.html)")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",")]
    if args.include_base and "base" not in variants:
        variants.insert(0, "base")

    output_path = args.output or os.path.join(OUTPUTS_DIR, "ab_exit_report.html")

    print(f"\n  Loading trades for variants: {', '.join(variants)}")
    variant_data = {}
    for v in variants:
        df = load_variant_trades(v)
        if not df.empty:
            variant_data[v] = df
            print(f"    Variant {v}: {len(df)} trades loaded")
        else:
            print(f"    Variant {v}: no trades found")

    if len(variant_data) < 1:
        print("\n  ERROR: No trade data found. Run backtests with --exit-variant first:")
        print("    python main.py backtest --symbol SLV --start 2024-01-01 --model swing --exit-variant A")
        print("    python main.py backtest --symbol SLV --start 2024-01-01 --model swing --exit-variant B")
        sys.exit(1)

    # Compute metrics
    print("\n  Computing metrics...")
    variant_metrics = {}
    variant_per_symbol = {}
    for v, df in variant_data.items():
        variant_metrics[v] = compute_metrics(df)
        variant_per_symbol[v] = compute_per_symbol_metrics(df)

    # Print console summary
    print("\n" + "=" * 70)
    print("  A/B EXIT VARIANT COMPARISON")
    print("=" * 70)
    for v in variant_data:
        m = variant_metrics[v]
        print(f"\n  Variant {v}:")
        print(f"    Trades:        {m.get('trades', 0)}")
        print(f"    Win Rate:      {m.get('win_rate', '-')}%")
        print(f"    Avg Return:    {m.get('avg_return_pct', '-')}%")
        print(f"    Profit Factor: {m.get('profit_factor', '-')}")
        print(f"    Total P&L:     ${m.get('total_pnl', 0):,.2f}")
        print(f"    Avg MFE:       {m.get('avg_mfe_pct', '-')}%")
        print(f"    Avg MAE:       {m.get('avg_mae_pct', '-')}%")
        print(f"    MFE Retention: {m.get('mfe_retention_pct', '-')}%")
        layers = m.get("exit_layer_counts", {})
        if layers:
            print(f"    Exit Layers:   {layers}")

    # Generate HTML
    print(f"\n  Generating HTML report...")
    path = generate_html_report(variant_data, variant_metrics,
                                variant_per_symbol, output_path)
    print(f"  Report saved to: {path}")

    # Try to open in browser
    try:
        import webbrowser
        webbrowser.open(path)
    except Exception:
        pass


if __name__ == "__main__":
    main()
