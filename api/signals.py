"""Vercel serverless function: GET /api/signals

Returns live VIX data from Yahoo Finance (real-time).
ML prediction is skipped (torch too large for serverless).
"""

from http.server import BaseHTTPRequestHandler
import json
from datetime import datetime, timezone

import requests as req

YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        vix_val = None
        vix_prev = None
        vix_change = None

        try:
            resp = req.get(
                YAHOO_QUOTE_URL,
                params={"range": "2d", "interval": "1d"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            data = resp.json()
            result = data["chart"]["result"][0]
            meta = result["meta"]
            vix_val = round(meta["regularMarketPrice"], 2)
            vix_prev = round(meta["chartPreviousClose"], 2)
            if vix_prev:
                vix_change = round((vix_val - vix_prev) / vix_prev * 100, 2)
        except Exception:
            pass

        payload = {
            "vix": vix_val,
            "vix_change_pct": vix_change,
            "asof": datetime.now(timezone.utc).isoformat(),
        }

        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "s-maxage=120")
        self.end_headers()
        self.wfile.write(body)
