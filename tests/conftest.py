"""Local Alpaca-format HTTP fixtures; no replacement downloader or external calls."""

import json
import threading
from datetime import UTC, date, datetime, time, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import pytest
import yaml

from dead_reckoning.market_data import alpaca


@pytest.fixture
def market_server(monkeypatch):
    state = SimpleNamespace(requests=[], failures=[])
    ny = ZoneInfo("America/New_York")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            parsed = urlparse(self.path)
            params = {key: value[0] for key, value in parse_qs(parsed.query).items()}
            state.requests.append((parsed.path, params))
            assert self.headers["APCA-API-KEY-ID"] == "fixture-key"
            assert self.headers["APCA-API-SECRET-KEY"] == "fixture-secret"
            if state.failures:
                status = state.failures.pop(0)
                if status:
                    self.send_response(status)
                    self.end_headers()
                    self.wfile.write(b'{"message":"fixture failure"}')
                    return
            if parsed.path == "/v2/assets":
                assert "status" not in params
                payload = [
                    {
                        "id": str(index),
                        "symbol": symbol,
                        "name": f"{symbol} fixture",
                        "class": "us_equity",
                        "exchange": "NASDAQ",
                        "status": "inactive" if symbol == "CCC" else "active",
                        "tradable": True,
                    }
                    for index, symbol in enumerate(["AAA", "BBB", "CCC"])
                ]
            elif parsed.path == "/v2/stocks/bars":
                assert params["feed"] == "sip"
                assert params["adjustment"] == "raw"
                assert params["asof"] == "-"
                lower = datetime.fromisoformat(params["start"].replace("Z", "+00:00"))
                upper = datetime.fromisoformat(params["end"].replace("Z", "+00:00"))
                entries = []
                for symbol in sorted(params["symbols"].split(",")):
                    index = 0
                    day = date(2025, 1, 2)
                    while day < date(2025, 2, 15):
                        if day.weekday() < 5:
                            values = (
                                [(0, 0)] if params["timeframe"] == "1Day" else [(9, 30), (9, 31)]
                            )
                            for hour, minute in values:
                                stamp = datetime.combine(day, time(hour, minute), ny).astimezone(
                                    UTC
                                )
                                if lower <= stamp <= upper:
                                    volume = (
                                        {
                                            "AAA": 100 if index < 3 else 1,
                                            "BBB": 10 if index < 3 else 1000,
                                            "CCC": 0.1,
                                        }
                                    )[symbol]
                                    entries.append(
                                        (
                                            symbol,
                                            {
                                                "t": stamp.isoformat(),
                                                "o": 10.0,
                                                "h": 11.0,
                                                "l": 9.0,
                                                "c": 10.0,
                                                "v": volume,
                                                "n": 5,
                                                "vw": 10.2,
                                            },
                                        )
                                    )
                            index += 1
                        day += timedelta(days=1)
                offset = int(params.get("page_token", "0"))
                limit = int(params["limit"])
                page = entries[offset : offset + limit]
                bars = {}
                for symbol, bar in page:
                    bars.setdefault(symbol, []).append(bar)
                next_offset = offset + limit
                payload = {
                    "bars": bars,
                    "next_page_token": (str(next_offset) if next_offset < len(entries) else None),
                }
            else:
                self.send_error(404)
                return
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr(alpaca, "DATA_URL", url)
    monkeypatch.setattr(alpaca, "ASSET_URL", url)
    monkeypatch.setenv("ALPACA_API_KEY", "fixture-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fixture-secret")
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {"root": str(tmp_path / "external-data"), "start_date": "2025-01-02"},
                "universe": {"lookback_days": 3, "min_history_days": 3, "ingestion_rank": 1},
                "ingestion": {
                    "page_size": 3,
                    "requests_per_minute": 100000000,
                    "daily_batch_size": 3,
                    "minute_batch_size": 2,
                    "max_retries": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    return path
