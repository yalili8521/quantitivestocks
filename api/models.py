"""Vercel serverless function: GET /api/models

Returns metadata about the current trained model inventory across all four
trading groups by scanning the saved model config files on disk.
"""

from http.server import BaseHTTPRequestHandler
import json
import os
from datetime import datetime, timezone


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRADING_CONFIG = os.path.join(PROJECT_ROOT, "config", "trading.json")

MODEL_SOURCES = [
    {
        "group": "swing",
        "mode": "daily",
        "dir": os.path.join(PROJECT_ROOT, "models", "swing"),
        "suffixes": {
            "_xgb_swing_config.json": "XGBoost",
            "_tft_swing_config.json": "TFT",
        },
    },
    {
        "group": "intraday",
        "mode": "intraday",
        "dir": os.path.join(PROJECT_ROOT, "models", "intraday"),
        "suffixes": {
            "_lgb_intraday_config.json": "LightGBM",
        },
    },
    {
        "group": "crypto",
        "mode": "daily",
        "dir": os.path.join(PROJECT_ROOT, "models", "crypto"),
        "suffixes": {
            "_xgb_swing_config.json": "XGBoost",
            "_tft_swing_config.json": "TFT",
        },
    },
    {
        "group": "crypto_intraday",
        "mode": "intraday",
        "dir": os.path.join(PROJECT_ROOT, "models", "crypto_intraday"),
        "suffixes": {
            "_lgb_intraday_crypto_config.json": "LightGBM+GRU",
        },
    },
]


def _load_trading_config() -> dict:
    try:
        with open(TRADING_CONFIG, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _normalize_symbol(symbol: str, group: str) -> str:
    if group.startswith("crypto"):
        return symbol.replace("/", "-")
    return symbol


def _status_for_symbol(symbol: str, group: str, cfg: dict) -> str:
    caps = cfg.get("symbol_caps", {}).get(group, {})
    sym = _normalize_symbol(symbol, group)
    cap = caps.get(sym)
    if cap is not None and cap <= 0:
        return "excluded"
    return "active"


def _feature_count(config: dict) -> int:
    feature_names = config.get("feature_names")
    if isinstance(feature_names, list):
        return len(feature_names)
    n_features = config.get("n_features")
    return int(n_features) if isinstance(n_features, int) else 0


def _direction_accuracy(config: dict):
    for key in ("val_direction_accuracy", "val_accuracy", "val_dir_acc"):
        val = config.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _discover_models() -> list[dict]:
    cfg = _load_trading_config()
    models = []

    for source in MODEL_SOURCES:
        model_dir = source["dir"]
        if not os.path.isdir(model_dir):
            continue

        try:
            files = sorted(os.listdir(model_dir))
        except Exception:
            continue

        for fname in files:
            matched_suffix = None
            matched_type = None
            for suffix, model_type in source["suffixes"].items():
                if fname.endswith(suffix):
                    matched_suffix = suffix
                    matched_type = model_type
                    break
            if matched_suffix is None:
                continue

            config_path = os.path.join(model_dir, fname)
            try:
                with open(config_path, encoding="utf-8") as f:
                    model_cfg = json.load(f)
            except Exception:
                model_cfg = {}

            symbol = model_cfg.get("symbol") or fname[: -len(matched_suffix)]
            trained_at = None
            try:
                stat = os.stat(config_path)
                trained_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            except Exception:
                trained_at = ""

            models.append({
                "symbol": symbol,
                "mode": source["mode"],
                "group": source["group"],
                "model_type": matched_type,
                "features": _feature_count(model_cfg),
                "dir_acc": _direction_accuracy(model_cfg),
                "trained_at": trained_at,
                "status": _status_for_symbol(symbol, source["group"], cfg),
                "note": model_cfg.get("train_end") or model_cfg.get("horizon", ""),
            })

    models.sort(key=lambda item: (item["symbol"], item["group"], item["model_type"]))
    return models


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = {
            "models": _discover_models(),
            "asof": datetime.now(timezone.utc).isoformat(),
        }

        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "s-maxage=300")
        self.end_headers()
        self.wfile.write(body)
