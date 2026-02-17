"""Vercel serverless function: GET /api/signals

Returns live VIX data from FRED API.
ML prediction is skipped (torch too large for serverless).
"""

from http.server import BaseHTTPRequestHandler
import json
import os
from datetime import datetime, timezone, timedelta

import requests as req

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        fred_key = os.environ.get("FRED_API_KEY", "")

        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = (datetime.now(timezone.utc) - timedelta(days=40)).strftime("%Y-%m-%d")

        params = {
            "series_id": "VIXCLS",
            "observation_start": start,
            "observation_end": end,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 5,
            "api_key": fred_key or "DEMO_KEY",
        }

        vix_val = None
        vix_prev = None
        vix_change = None
        vix_date = None

        try:
            resp = req.get(FRED_BASE, params=params, timeout=10)
            obs = resp.json().get("observations", [])
            valid = [o for o in obs if o.get("value", ".") != "."]
            if len(valid) >= 1:
                vix_val = float(valid[0]["value"])
                vix_date = valid[0]["date"]
            if len(valid) >= 2:
                vix_prev = float(valid[1]["value"])
                if vix_prev:
                    vix_change = round((vix_val - vix_prev) / vix_prev * 100, 2)
        except Exception:
            pass

        data = {
            "vix": vix_val,
            "vix_change_pct": vix_change,
            "vix_date": vix_date,
            "asof": datetime.now(timezone.utc).isoformat(),
        }

        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "s-maxage=300")
        self.end_headers()
        self.wfile.write(body)
