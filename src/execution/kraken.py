"""
Kraken executor — re-exports from kraken_executor.py with BaseExecutor interface.
"""
from execution.base import BaseExecutor
from kraken_executor import KrakenExecutor as _KrakenExecutor


class KrakenExecutor(_KrakenExecutor, BaseExecutor):
    """Kraken executor implementing BaseExecutor interface."""
    pass
