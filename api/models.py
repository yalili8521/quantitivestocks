"""Vercel serverless function: GET /api/models

Returns metadata about all trained LSTM models.
Discovers models via *_scaler*.json (*.pt excluded from deploy via .vercelignore).
"""

from http.server import BaseHTTPRequestHandler
import json
import os
from datetime import datetime, timezone

_candidates = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
    os.path.join(os.getcwd(), "models"),
    "/var/task/models",
]
MODELS_DIR = next((p for p in _candidates if os.path.isdir(p)), _candidates[0])

# All models share the same arch; actual .pt excluded via .vercelignore
_PT_SIZE_KB = 512.5


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        models = []

        try:
            files = sorted(os.listdir(MODELS_DIR))
        except Exception:
            files = []

        for fname in files:
            if not fname.endswith("_scaler.json") and not fname.endswith("_scaler_5min.json"):
                continue

            scaler_path = os.path.join(MODELS_DIR, fname)
            stat = os.stat(scaler_path)
            trained_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

            if fname.endswith("_scaler_5min.json"):
                symbol = fname.replace("_scaler_5min.json", "")
                mode = "intraday"
            else:
                symbol = fname.replace("_scaler.json", "")
                mode = "daily"

            try:
                with open(scaler_path) as f:
                    scaler = json.load(f)
                features = list(scaler.get("mean", {}).keys())
                n_features = len(features)
            except Exception:
                features = []
                n_features = 0

            models.append({
                "symbol": symbol,
                "mode": mode,
                "features": n_features,
                "feature_names": features,
                "size_kb": _PT_SIZE_KB,
                "trained_at": trained_at,
            })

        payload = {
            "models": models,
            "total": len(models),
            "asof": datetime.now(timezone.utc).isoformat(),
        }

        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "s-maxage=3600")
        self.end_headers()
        self.wfile.write(body)
