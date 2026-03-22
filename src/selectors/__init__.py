"""
Layer 1 selectors — cross-sectional symbol ranking per trading group.

Each selector ranks a universe of symbols for Layer 2 (per-symbol prediction
models). Position limits are applied downstream by paper_trader/risk_config.
Selectors are optional; groups without a trained selector fall back to their
static symbol list.

Usage:
    from selectors import get_selector
    selector = get_selector("crypto", model_dir="models/crypto")
    output = selector.rank(universe_data)
    top_symbols = output.selected
"""
from selectors.base import BaseSelector, SelectorOutput

__all__ = ["BaseSelector", "SelectorOutput", "get_selector"]


def get_selector(group: str, **kwargs):
    """Factory: return the right selector for a trading group, or None."""
    if group == "crypto":
        from selectors.crypto import CryptoSelector
        try:
            return CryptoSelector(**kwargs)
        except FileNotFoundError:
            return None
    if group == "swing":
        from selectors.swing import SwingSelector
        try:
            return SwingSelector(**kwargs)
        except FileNotFoundError:
            return None
    if group == "intraday":
        from selectors.intraday import IntradaySelector
        try:
            return IntradaySelector(**kwargs)
        except FileNotFoundError:
            return None
    return None
