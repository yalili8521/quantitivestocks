#!/usr/bin/env python3
"""
Quantitative Stocks - Backtest Dashboard
=========================================
Interactive dashboard to display backtest results from multiple ETFs/stocks.

Features:
- Portfolio performance summary
- Equity curves for all symbols  
- Trade analysis
- Risk metrics
- Comparison charts

Usage:
    python dashboard.py
    # Opens dashboard in browser at http://localhost:8501
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import glob

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")

# Page configuration
st.set_page_config(
    page_title="Quantitative Stocks Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_backtest_data():
    """Load all backtest and trading data."""
    data = {}
    
    # Find all backtest CSV files
    backtest_files = glob.glob(os.path.join(OUTPUT_DIR, "backtest_*.csv"))
    trade_files = glob.glob(os.path.join(OUTPUT_DIR, "trades_*.csv"))
    
    for file in backtest_files:
        symbol = os.path.basename(file).replace("backtest_", "").replace(".csv", "")
        try:
            equity_df = pd.read_csv(file)
            equity_df['date'] = pd.to_datetime(equity_df['date'])
            data[symbol] = {"equity": equity_df}
        except Exception as e:
            st.warning(f"Error loading {file}: {e}")
    
    # Load trade data
    for file in trade_files:
        symbol = os.path.basename(file).replace("trades_", "").replace(".csv", "")
        try:
            trades_df = pd.read_csv(file)
            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
            if symbol in data:
                data[symbol]["trades"] = trades_df
        except Exception as e:
            st.warning(f"Error loading {file}: {e}")
    
    return data

def calculate_metrics(equity_df, trades_df=None):
    """Calculate performance metrics."""
    if equity_df.empty:
        return {}
    
    equity = equity_df['equity'].values
    dates = equity_df['date'].values
    
    # Basic metrics
    initial_capital = equity[0]
    final_equity = equity[-1]
    total_return = (final_equity - initial_capital) / initial_capital
    
    # Calculate daily returns
    daily_returns = np.diff(equity) / equity[:-1]
    annual_returns = daily_returns * 252  # Annualized
    
    # Sharpe ratio (assuming 2% risk-free rate)
    risk_free_rate = 0.02
    excess_returns = annual_returns - risk_free_rate/252
    sharpe = np.mean(excess_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
    
    # Maximum drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # Trading metrics
    trade_metrics = {}
    if trades_df is not None and not trades_df.empty:
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (total_trades - winning_trades) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Average trade duration
        duration = (trades_df['exit_date'] - trades_df['entry_date']).dt.days.mean()
        
        trade_metrics = {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "avg_duration": duration
        }
    
    return {
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return": total_return,
        "annualized_return": (1 + total_return) ** (252 / len(equity)) - 1,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "volatility": np.std(daily_returns) * np.sqrt(252),
        **trade_metrics
    }

def create_equity_curve_chart(data):
    """Create combined equity curve chart."""
    fig = go.Figure()
    
    for symbol, symbol_data in data.items():
        equity_df = symbol_data.get("equity", pd.DataFrame())
        if not equity_df.empty:
            fig.add_trace(go.Scatter(
                x=equity_df['date'],
                y=equity_df['equity'],
                mode='lines',
                name=f'{symbol} Equity',
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title="Portfolio Equity Curves",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_drawdown_chart(data):
    """Create drawdown chart for all symbols."""
    fig = go.Figure()
    
    for symbol, symbol_data in data.items():
        equity_df = symbol_data.get("equity", pd.DataFrame())
        if not equity_df.empty:
            equity = equity_df['equity'].values
            running_max = np.maximum.accumulate(equity)
            drawdown = (equity - running_max) / running_max * 100
            
            fig.add_trace(go.Scatter(
                x=equity_df['date'],
                y=drawdown,
                mode='lines',
                name=f'{symbol} Drawdown',
                fill='tonexty' if symbol == list(data.keys())[0] else None,
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title="Portfolio Drawdowns",
        xaxis_title="Date", 
        yaxis_title="Drawdown (%)",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_trade_analysis_chart(data):
    """Create trade analysis charts."""
    all_trades = []
    
    for symbol, symbol_data in data.items():
        trades_df = symbol_data.get("trades", pd.DataFrame())
        if not trades_df.empty:
            trades_copy = trades_df.copy()
            trades_copy['symbol'] = symbol
            all_trades.append(trades_copy)
    
    if not all_trades:
        return None, None
        
    combined_trades = pd.concat(all_trades, ignore_index=True)
    
    # PnL distribution
    pnl_fig = px.histogram(
        combined_trades, 
        x='pnl', 
        color='symbol',
        title="Trade P&L Distribution",
        nbins=20
    )
    pnl_fig.update_layout(height=400)
    
    # Returns by symbol
    returns_fig = px.box(
        combined_trades,
        x='symbol',
        y='return_pct',
        title="Return Distribution by Symbol"
    )
    returns_fig.update_layout(height=400)
    
    return pnl_fig, returns_fig

def main():
    st.title("📈 Quantitative Stocks - Backtest Dashboard")
    st.markdown("*ML-Driven Trading Strategy Performance Analysis*")
    
    # Load data
    with st.spinner("Loading backtest data..."):
        data = load_backtest_data()
    
    if not data:
        st.error("No backtest data found! Run backtests first using:")
        st.code("python main.py backtest --symbol SPY --start 2024-01-01")
        return
    
    st.success(f"Loaded data for {len(data)} symbols: {', '.join(data.keys())}")
    
    # Sidebar - Symbol selection
    st.sidebar.header("📊 Dashboard Controls")
    selected_symbols = st.sidebar.multiselect(
        "Select Symbols",
        options=list(data.keys()),
        default=list(data.keys())[:3] if len(data) > 3 else list(data.keys())
    )
    
    filtered_data = {k: v for k, v in data.items() if k in selected_symbols}
    
    # Main dashboard content
    if not filtered_data:
        st.warning("Please select at least one symbol from the sidebar.")
        return
    
    # Key metrics summary
    st.header("🎯 Performance Summary")
    
    # Calculate combined metrics
    metrics_data = []
    for symbol, symbol_data in filtered_data.items():
        equity_df = symbol_data.get("equity", pd.DataFrame())
        trades_df = symbol_data.get("trades", pd.DataFrame())
        metrics = calculate_metrics(equity_df, trades_df)
        metrics['symbol'] = symbol
        metrics_data.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Display key metrics in columns
    if not metrics_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_return = metrics_df['total_return'].mean() * 100
            st.metric("Avg Total Return", f"{avg_return:.1f}%")
            
        with col2:
            avg_sharpe = metrics_df['sharpe_ratio'].mean()
            st.metric("Avg Sharpe Ratio", f"{avg_sharpe:.2f}")
            
        with col3:
            avg_dd = metrics_df['max_drawdown'].mean() * 100
            st.metric("Avg Max Drawdown", f"{avg_dd:.1f}%")
            
        with col4:
            total_trades = metrics_df['total_trades'].sum()
            st.metric("Total Trades", f"{int(total_trades) if not pd.isna(total_trades) else 0}")
    
    # Detailed metrics table
    if not metrics_df.empty:
        st.subheader("📋 Detailed Metrics by Symbol")
        
        display_df = metrics_df.copy()
        for col in ['total_return', 'annualized_return', 'max_drawdown', 'volatility']:
            if col in display_df.columns:
                display_df[col] = (display_df[col] * 100).round(2).astype(str) + '%'
        
        for col in ['sharpe_ratio']:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(3)
                
        if 'win_rate' in display_df.columns:
            display_df['win_rate'] = (display_df['win_rate'] * 100).round(1).astype(str) + '%'
        
        # Rename columns for display
        column_mapping = {
            'symbol': 'Symbol',
            'final_equity': 'Final Equity ($)',
            'total_return': 'Total Return',
            'annualized_return': 'Annual Return', 
            'sharpe_ratio': 'Sharpe Ratio',
            'max_drawdown': 'Max Drawdown',
            'volatility': 'Volatility',
            'total_trades': 'Total Trades',
            'win_rate': 'Win Rate',
            'profit_factor': 'Profit Factor',
            'avg_duration': 'Avg Duration (days)'
        }
        
        display_cols = [col for col in column_mapping.keys() if col in display_df.columns]
        display_df = display_df[display_cols].rename(columns=column_mapping)
        
        st.dataframe(display_df, use_container_width=True)
    
    # Charts section
    st.header("📈 Portfolio Performance Charts")
    
    # Equity curves
    eq_chart = create_equity_curve_chart(filtered_data)
    st.plotly_chart(eq_chart, use_container_width=True)
    
    # Drawdown chart
    dd_chart = create_drawdown_chart(filtered_data)
    st.plotly_chart(dd_chart, use_container_width=True)
    
    # Trade analysis
    st.header("🔄 Trade Analysis")
    pnl_chart, returns_chart = create_trade_analysis_chart(filtered_data)
    
    if pnl_chart and returns_chart:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(pnl_chart, use_container_width=True)
        with col2:
            st.plotly_chart(returns_chart, use_container_width=True)
    else:
        st.info("No trade data available for selected symbols.")
    
    # Recent trades
    st.header("📊 Recent Trades")
    
    recent_trades_data = []
    for symbol, symbol_data in filtered_data.items():
        trades_df = symbol_data.get("trades", pd.DataFrame())
        if not trades_df.empty:
            trades_copy = trades_df.copy()
            trades_copy['symbol'] = symbol
            recent_trades_data.append(trades_copy)
    
    if recent_trades_data:
        all_trades = pd.concat(recent_trades_data, ignore_index=True)
        all_trades = all_trades.sort_values('exit_date', ascending=False)
        
        # Format for display
        display_trades = all_trades[['symbol', 'direction', 'entry_date', 'exit_date', 
                                   'entry_price', 'exit_price', 'shares', 'pnl', 
                                   'return_pct', 'exit_reason']].head(20)
        
        display_trades['entry_date'] = display_trades['entry_date'].dt.strftime('%Y-%m-%d')
        display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('%Y-%m-%d')
        display_trades['pnl'] = display_trades['pnl'].round(2)
        display_trades['return_pct'] = (display_trades['return_pct']).round(2)
        
        st.dataframe(display_trades, use_container_width=True)
    else:
        st.info("No trade data available.")
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard generated from backtest results in `outputs/`*")
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        # Start Streamlit
        os.system("streamlit run dashboard.py")
    else:
        main()