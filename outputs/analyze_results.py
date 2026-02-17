#!/usr/bin/env python3
"""
Quick Backtest Results Analysis
===============================
Show summary of backtest performance without interactive dashboard.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import glob

# Project root and output directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

def analyze_backtest_results():
    """Analyze and display backtest results."""
    print("=" * 80)
    print("  📈 QUANTITATIVE STOCKS - BACKTEST RESULTS ANALYSIS")
    print("=" * 80)
    print()
    
    # Find all backtest files
    backtest_files = glob.glob(os.path.join(OUTPUT_DIR, "backtest_*.csv"))
    trade_files = glob.glob(os.path.join(OUTPUT_DIR, "trades_*.csv"))
    
    if not backtest_files:
        print("❌ No backtest files found!")
        print("Run backtests first: python main.py backtest --symbol SPY --start 2024-01-01")
        return
    
    print(f"📊 Found {len(backtest_files)} backtest results")
    print(f"📊 Found {len(trade_files)} trade log files")
    print()
    
    results = []
    
    for file in backtest_files:
        symbol = os.path.basename(file).replace("backtest_", "").replace(".csv", "")
        
        try:
            # Load equity curve
            equity_df = pd.read_csv(file)
            equity_df['date'] = pd.to_datetime(equity_df['date'])
            
            # Load trades if available
            trade_file = os.path.join(OUTPUT_DIR, f"trades_{symbol}.csv")
            trades_df = None
            if os.path.exists(trade_file):
                trades_df = pd.read_csv(trade_file)
                trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
                trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
            
            # Calculate metrics
            metrics = calculate_performance_metrics(equity_df, trades_df)
            metrics['symbol'] = symbol
            results.append(metrics)
            
        except Exception as e:
            print(f"⚠️  Error processing {symbol}: {e}")
    
    if not results:
        print("❌ No valid backtest data found!")
        return
    
    # Display results
    display_results_table(results)
    display_trade_summary(results)

def calculate_performance_metrics(equity_df, trades_df=None):
    """Calculate comprehensive performance metrics."""
    equity = equity_df['equity'].values
    dates = equity_df['date'].values
    
    # Basic metrics
    initial_capital = equity[0]
    final_equity = equity[-1]
    total_return = (final_equity - initial_capital) / initial_capital
    
    # Time period
    start_date = dates[0]
    end_date = dates[-1]
    total_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    years = total_days / 365.25
    
    # Annualized return
    annualized_return = (final_equity / initial_capital) ** (1 / years) - 1 if years > 0 else 0
    
    # Daily returns for risk calculation
    daily_returns = np.diff(equity) / equity[:-1]
    
    # Volatility (annualized)
    volatility = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 else 0
    
    # Sharpe ratio (assuming 2% risk-free rate)
    risk_free_rate = 0.02
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Maximum drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # Trading metrics
    trade_metrics = {}
    if trades_df is not None and not trades_df.empty:
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Average win/loss
        wins = trades_df[trades_df['pnl'] > 0]['pnl']
        losses = trades_df[trades_df['pnl'] < 0]['pnl']
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        # Profit factor
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average holding period
        duration = (trades_df['exit_date'] - trades_df['entry_date']).dt.days
        avg_duration = duration.mean() if not duration.empty else 0
        
        # Largest win/loss
        largest_win = wins.max() if len(wins) > 0 else 0
        largest_loss = losses.min() if len(losses) > 0 else 0
        
        trade_metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_duration': avg_duration,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    return {
        'start_date': start_date,
        'end_date': end_date,
        'total_days': total_days,
        'initial_capital': initial_capital,
        'final_equity': final_equity,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        **trade_metrics
    }

def display_results_table(results):
    """Display performance results in a formatted table."""
    print("📈 PERFORMANCE SUMMARY")
    print("-" * 80)
    
    # Header
    print(f"{'Symbol':<8} {'Return':<10} {'Annual':<10} {'Sharpe':<8} {'MaxDD':<8} {'Trades':<8} {'Win%':<8}")
    print("-" * 80)
    
    # Data rows
    for result in results:
        symbol = result['symbol']
        total_ret = result['total_return'] * 100
        annual_ret = result['annualized_return'] * 100
        sharpe = result['sharpe_ratio']
        max_dd = result['max_drawdown'] * 100
        trades = result.get('total_trades', 0)
        win_rate = result.get('win_rate', 0) * 100
        
        print(f"{symbol:<8} {total_ret:>8.1f}% {annual_ret:>8.1f}% {sharpe:>6.2f} {max_dd:>6.1f}% {trades:>6.0f} {win_rate:>6.1f}%")
    
    print("-" * 80)
    
    # Summary statistics
    total_returns = [r['total_return'] for r in results]
    annual_returns = [r['annualized_return'] for r in results]
    sharpe_ratios = [r['sharpe_ratio'] for r in results]
    
    avg_total = np.mean(total_returns) * 100
    avg_annual = np.mean(annual_returns) * 100
    avg_sharpe = np.mean(sharpe_ratios)
    
    print(f"{'AVERAGE':<8} {avg_total:>8.1f}% {avg_annual:>8.1f}% {avg_sharpe:>6.2f}")
    print()

def display_trade_summary(results):
    """Display trading activity summary."""
    print("🔄 TRADING ACTIVITY SUMMARY")
    print("-" * 80)
    
    all_trades = 0
    all_wins = 0
    total_profit = 0
    total_loss = 0
    
    for result in results:
        if 'total_trades' in result:
            symbol = result['symbol']
            trades = result['total_trades']
            wins = result.get('winning_trades', 0)
            losses = result.get('losing_trades', 0)
            profit = result.get('gross_profit', 0)
            loss = result.get('gross_loss', 0)
            
            all_trades += trades
            all_wins += wins
            total_profit += profit
            total_loss += loss
            
            print(f"{symbol:<8} {trades:>6.0f} trades, {wins:>3.0f} wins, {losses:>3.0f} losses")
            print(f"         Profit: ${profit:>8.0f}, Loss: ${loss:>8.0f}")
    
    if all_trades > 0:
        print("-" * 40)
        overall_win_rate = all_wins / all_trades * 100
        net_profit = total_profit - total_loss
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        print(f"TOTAL:   {all_trades:>6.0f} trades, {overall_win_rate:>5.1f}% win rate")
        print(f"         Net P&L: ${net_profit:>8.0f}")
        print(f"         Profit Factor: {profit_factor:>5.2f}")
    
    print()

def show_best_period(results):
    """Show the best performing period for each symbol."""
    print("🏆 TOP PERFORMERS")
    print("-" * 40)
    
    if results:
        # Sort by Sharpe ratio
        sorted_results = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)
        
        print("By Sharpe Ratio:")
        for i, result in enumerate(sorted_results[:3], 1):
            symbol = result['symbol']
            sharpe = result['sharpe_ratio']
            annual_ret = result['annualized_return'] * 100
            print(f"{i}. {symbol}: {sharpe:.2f} Sharpe, {annual_ret:.1f}% annual return")
        
        print()
        
        # Sort by total return
        sorted_by_return = sorted(results, key=lambda x: x['total_return'], reverse=True)
        
        print("By Total Return:")
        for i, result in enumerate(sorted_by_return[:3], 1):
            symbol = result['symbol']
            total_ret = result['total_return'] * 100
            days = result['total_days']
            print(f"{i}. {symbol}: {total_ret:.1f}% over {days:.0f} days")

if __name__ == "__main__":
    try:
        analyze_backtest_results()
        show_best_period([])  # Will implement if needed
        
        # Auto-generate comprehensive dashboard
        try:
            import subprocess
            dashboard_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "create_comprehensive_dashboard.py")
            if os.path.exists(dashboard_script):
                print("🔄 Updating comprehensive dashboard with latest data...")
                subprocess.run(["python", dashboard_script], capture_output=True)
                print("✅ Comprehensive dashboard updated!")
        except Exception:
            pass
        
        print("=" * 80)
        print("🌐 To view comprehensive dashboard, open:")
        print("   comprehensive_dashboard.html")
        print("   (All individual results consolidated in one interactive view)")
        print()
        print("📁 All results available in:")
        print(f"   {OUTPUT_DIR}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()