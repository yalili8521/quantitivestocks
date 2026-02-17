"""
Configuration settings for the quantitative trading system.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SECRETS_DIR = PROJECT_ROOT / "secrets"
LOGS_DIR = PROJECT_ROOT / "logs"

# Model paths
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
SCALERS_DIR = MODELS_DIR / "scalers"

# Results paths
BACKTESTS_DIR = RESULTS_DIR / "backtests"
TRADES_DIR = RESULTS_DIR / "trades"
CHARTS_DIR = RESULTS_DIR / "charts"

# Trading configuration
DEFAULT_SYMBOLS = ['SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'XLE', 'IGV']
CONFIDENCE_THRESHOLDS = {
    'LONG': 0.20,
    'SHORT': 0.15,
    'EXIT': 0.10
}

# Alpaca configuration
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'
ALPACA_ENV_FILE = SECRETS_DIR / 'alpaca.env'

# ML Model configuration
MODEL_FEATURES = ['close', 'volume', 'sma_20', 'rsi', 'macd']
SEQUENCE_LENGTH = 60
PREDICTION_HORIZON = 1

# Risk management
MAX_POSITION_SIZE = 0.05  # 5% max position size
STOP_LOSS_PCT = 0.02      # 2% stop loss
TRAILING_STOP_PCT = 0.015  # 1.5% trailing stop

# Options trading
OPTIONS_EXPIRATION_DAYS = 30
MAX_OPTIONS_RISK = 1000    # Max risk per options trade
SPREAD_WIDTH = 5           # Strike width for spreads

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'