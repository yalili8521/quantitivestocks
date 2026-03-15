"""
Portfolio Allocation Module - Grinold & Kahn Principles
=======================================================

Based on "Active Portfolio Management" by Richard Grinold and Ronald Kahn.

Key concepts:
1. IC (Information Coefficient) - prediction accuracy
2. Correlation - diversification benefit
3. Mean-Variance Optimization - efficient frontier
4. Risk Parity - equal risk contribution

Usage:
    from portfolio_allocator import PortfolioAllocator
    allocator = PortfolioAllocator(symbols=["BTC/USD", "ETH/USD", ...])
    allocation = allocator.get_allocation(equity=100000)
"""

import os
import json
from typing import Dict, List, Optional


DEFAULT_CRYPTO_SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD",
    "DOGE/USD", "DOT/USD", "SUSHI/USD", "ADA/USD", "CRV/USD",
    "AAVE/USD", "RENDER/USD"
]


class PortfolioAllocator:
    """Grinold-Kahn inspired portfolio allocator."""
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        backtest_dir: str = "outputs",
        use_ic_weighting: bool = True,
        use_correlation: bool = True,
        risk_parity: bool = True,
        max_positions: int = 4,
        min_signal_threshold: float = 0.3,
    ):
        self.symbols = symbols or DEFAULT_CRYPTO_SYMBOLS
        self.backtest_dir = backtest_dir
        self.use_ic_weighting = use_ic_weighting
        self.use_correlation = use_correlation
        self.risk_parity = risk_parity
        self.max_positions = max_positions
        self.min_signal_threshold = min_signal_threshold
        
        self.ic_weights = self._load_ic_weights()
        self.corr_matrix = None
        
    def _load_ic_weights(self) -> Dict[str, float]:
        """Load backtest results - use Sharpe ratio as IC proxy."""
        weights = {}
        
        for symbol in self.symbols:
            file_symbol = symbol.replace("/", "-")
            filepath = os.path.join(
                self.backtest_dir, 
                f"backtest_{file_symbol}_summary.json"
            )
            
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    sharpe = data.get("sharpe_ratio", 0)
                    win_rate = data.get("win_rate", 0.5)
                    total_trades = data.get("total_trades", 0)
                    
                    trade_factor = min(1.0, total_trades / 20)
                    ic_proxy = max(0, sharpe) * (0.7 + 0.3 * trade_factor)
                    weights[symbol] = ic_proxy
                    
                except (json.JSONDecodeError, KeyError):
                    weights[symbol] = 0.0
            else:
                weights[symbol] = 0.0
                
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            weights = {k: 1.0 / len(self.symbols) for k in self.symbols}
            
        return weights
    
    def get_ic_weight(self, symbol: str) -> float:
        return self.ic_weights.get(symbol, 0.0)
    
    def calculate_correlation_matrix(self) -> Dict:
        corr = {}
        for s1 in self.symbols:
            corr[s1] = {}
            for s2 in self.symbols:
                if s1 == s2:
                    corr[s1][s2] = 1.0
                else:
                    base = 0.5
                    if s1 in ["BTC/USD", "ETH/USD"] and s2 in ["BTC/USD", "ETH/USD"]:
                        base = 0.7
                    elif s1 in ["SOL/USD", "AVAX/USD", "LINK/USD", "ADA/USD", 
                               "DOT/USD", "CRV/USD", "AAVE/USD", "RENDER/USD"] and \
                         s2 in ["SOL/USD", "AVAX/USD", "LINK/USD", "ADA/USD", 
                               "DOT/USD", "CRV/USD", "AAVE/USD", "RENDER/USD"]:
                        base = 0.6
                    elif s1 in ["DOGE/USD"] or s2 in ["DOGE/USD"]:
                        base = 0.4
                    corr[s1][s2] = base
        self.corr_matrix = corr
        return corr
    
    def get_diversification_bonus(self, symbol: str) -> float:
        if self.corr_matrix is None:
            self.calculate_correlation_matrix()
            
        if symbol not in self.corr_matrix:
            return 1.0
            
        avg_corr = sum(self.corr_matrix[symbol].values()) / len(self.corr_matrix[symbol])
        bonus = 1.0 + (0.5 - avg_corr)
        return max(0.5, min(1.5, bonus))
    
    def get_volatility_weight(self, symbol: str) -> float:
        file_symbol = symbol.replace("/", "-")
        filepath = os.path.join(
            self.backtest_dir,
            f"backtest_{file_symbol}_summary.json"
        )
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                max_dd = abs(data.get("max_drawdown_pct", 0)) / 100
                if max_dd > 0:
                    return 1.0 / (1 + max_dd * 10)
                return 1.0
            except:
                return 0.5
        return 0.5
    
    def get_allocation(self, equity: float) -> Dict[str, float]:
        weights = {}
        for symbol in self.symbols:
            ic_w = self.get_ic_weight(symbol)
            div_bonus = self.get_diversification_bonus(symbol)
            vol_w = self.get_volatility_weight(symbol)
            
            if self.use_ic_weighting:
                base = ic_w
            else:
                base = 1.0 / len(self.symbols)
            
            if self.use_correlation:
                base *= div_bonus
                
            if self.risk_parity:
                base *= vol_w
                
            weights[symbol] = base
            
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
            
        for symbol in list(weights.keys()):
            if self.ic_weights.get(symbol, 0) < self.min_signal_threshold:
                weights[symbol] = 0
                
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
            
        if self.max_positions > 0:
            sorted_syms = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            for i, (symbol, _) in enumerate(sorted_syms):
                if i >= self.max_positions:
                    weights[symbol] = 0
                    
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
            
        allocation = {symbol: weight * equity for symbol, weight in weights.items()}
        
        return allocation
    
    def get_position_size(self, symbol: str, current_price: float, total_equity: float) -> Dict:
        allocation = self.get_allocation(total_equity)
        
        if symbol not in allocation or allocation[symbol] <= 0:
            return {
                "qty": 0,
                "allocation": 0,
                "weight": 0,
                "signal_quality": "low"
            }
            
        dollar_amount = allocation[symbol]
        qty = dollar_amount / current_price
        
        if "BTC" in symbol:
            qty = round(qty, 6)
        else:
            qty = int(qty)
            
        return {
            "qty": qty,
            "allocation": dollar_amount,
            "weight": dollar_amount / total_equity,
            "signal_quality": "high" if self.ic_weights.get(symbol, 0) >= self.min_signal_threshold else "low"
        }
    
    def print_allocation_summary(self, equity: float) -> None:
        allocation = self.get_allocation(equity)
        
        print(f"\n{'='*60}")
        print(f"Portfolio Allocation (Grinold-Kahn)")
        print(f"{'='*60}")
        print(f"Total equity: ${equity:,.2f}")
        print(f"Max positions: {self.max_positions}")
        print(f"Min signal threshold: {self.min_signal_threshold}")
        print(f"\n{'Symbol':<15} {'IC Weight':>12} {'Vol Adj':>10} {'Allocation':>15} {'Weight':>10}")
        print(f"{'-'*60}")
        
        for symbol in self.symbols:
            ic_w = self.get_ic_weight(symbol)
            vol_w = self.get_volatility_weight(symbol)
            alloc = allocation.get(symbol, 0)
            weight = alloc / equity if equity > 0 else 0
            
            if alloc > 0:
                print(f"{symbol:<15} {ic_w:>12.3f} {vol_w:>10.3f} ${alloc:>14,.2f} {weight:>9.1%}")
                
        print(f"{'-'*60}")
        print(f"{'Total':<15} {'':<12} {'':<10} ${sum(allocation.values()):>14,.2f} {sum(allocation.values())/equity:>9.1%}")


def load_crypto_allocator(
    backtest_dir: str = "outputs",
    max_positions: int = 4,
) -> PortfolioAllocator:
    return PortfolioAllocator(
        symbols=DEFAULT_CRYPTO_SYMBOLS,
        backtest_dir=backtest_dir,
        use_ic_weighting=True,
        use_correlation=True,
        risk_parity=True,
        max_positions=max_positions,
        min_signal_threshold=0.3,
    )


if __name__ == "__main__":
    # Use threshold 0.0 to show all symbols
    allocator = PortfolioAllocator(
        symbols=DEFAULT_CRYPTO_SYMBOLS,
        backtest_dir="outputs",
        use_ic_weighting=True,
        use_correlation=True,
        risk_parity=True,
        max_positions=4,
        min_signal_threshold=0.0,  # Show all
    )
    allocator.print_allocation_summary(equity=100000)
