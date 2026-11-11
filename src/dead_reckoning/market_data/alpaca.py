"""SDK request models with explicit, bounded HTTP pagination and retries."""

import logging
import random
import time
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from datetime import time as day_time
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

import polars as pl
import requests
from alpaca.common.enums import Sort
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import AssetClass
from alpaca.trading.requests import GetAssetsRequest

from dead_reckoning.config import Credentials, IngestionConfig

log = logging.getLogger(__name__)
DATA_URL = "https://data.alpaca.markets"
ASSET_URL = "https://paper-api.alpaca.markets"
NY = ZoneInfo("America/New_York")

BAR_SCHEMA = {
    "symbol": pl.String,
    "timestamp": pl.Datetime("us", "UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "trade_count": pl.Int64,
    "vwap": pl.Float64,
}
BAR_FIELDS = {
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "n": "trade_count",
    "vw": "vwap",
}


def session_bounds(start: date, end: date) -> tuple[datetime, datetime]:
    return (
        datetime.combine(start, day_time(), NY).astimezone(UTC),
        datetime.combine(end, day_time(), NY).astimezone(UTC),
    )


def bars_frame(bars: dict) -> pl.DataFrame:
    records = [
        {
            "symbol": symbol,
            "timestamp": bar["t"],
            **{name: bar.get(key) for key, name in BAR_FIELDS.items()},
        }
        for symbol, values in bars.items()
        for bar in values
    ]
    if not records:
        return pl.DataFrame(schema=BAR_SCHEMA)
    frame = (
        pl.from_dicts(records, infer_schema_length=None)
        .with_columns(pl.col("timestamp").str.to_datetime(time_zone="UTC", time_unit="us"))
        .cast(BAR_SCHEMA)
    )
    required = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    if frame.select(pl.any_horizontal(pl.col(required).is_null()).any()).item():
        raise ValueError("Alpaca returned null required bar fields")
    numeric = ["open", "high", "low", "close", "volume"]
    if frame.filter(
        ~pl.all_horizontal(pl.col(numeric).is_finite())
        | (pl.col("volume") < 0)
        | (pl.col("trade_count") < 0)
    ).height:
        raise ValueError("Alpaca returned invalid bar values")
    return frame


class AlpacaClient:
    def __init__(self, credentials: Credentials, options: IngestionConfig):
        self.options = options
        self.session = requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": credentials.api_key.get_secret_value(),
                "APCA-API-SECRET-KEY": credentials.secret_key.get_secret_value(),
                "User-Agent": "dead-reckoning-ml/0.1",
            }
        )
        self._next_request = 0.0

    def close(self) -> None:
        self.session.close()

    def _get(self, url: str, params: dict):
        for attempt in range(self.options.max_retries + 1):
            time.sleep(max(0, self._next_request - time.monotonic()))
            self._next_request = time.monotonic() + 60 / self.options.requests_per_minute
            response = None
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.options.timeout_seconds,
                    allow_redirects=False,
                )
                if response.status_code not in {408, 429, 500, 502, 503, 504}:
                    response.raise_for_status()
                    if response.status_code != 200:
                        raise ValueError(f"Unexpected Alpaca HTTP status {response.status_code}")
                    return response.json()
                response.raise_for_status()
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                if response is not None and response.status_code not in {
                    408,
                    429,
                    500,
                    502,
                    503,
                    504,
                }:
                    raise RuntimeError(
                        f"Alpaca HTTP {response.status_code}; check credentials, SIP entitlement, "
                        "and requested symbols. No coverage was marked complete."
                    ) from exc
                if attempt == self.options.max_retries:
                    raise RuntimeError(
                        "Alpaca request failed after retries; rerun to resume"
                    ) from exc
                delay = min(60, 2**attempt) + random.uniform(0, 0.5)
                if response is not None:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            delay = max(
                                delay, parsedate_to_datetime(retry_after).timestamp() - time.time()
                            )
                    reset = response.headers.get("X-RateLimit-Reset")
                    if response.status_code == 429 and reset:
                        delay = max(delay, float(reset) - time.time())
                self._next_request = max(self._next_request, time.monotonic() + delay)
                log.warning("Transient Alpaca error; retry %s in %.1fs", attempt + 1, delay)

    def assets(self) -> list[dict]:
        # Omit status to include active AND inactive assets; no tradable filter.
        params = GetAssetsRequest(asset_class=AssetClass.US_EQUITY).to_request_fields()
        result = self._get(f"{ASSET_URL}/v2/assets", params)
        if not isinstance(result, list) or not result:
            raise ValueError("Alpaca returned an invalid or empty asset master")
        return result

    def bar_pages(
        self,
        symbols: list[str],
        timeframe: str,
        start: date,
        end: date,
    ) -> Iterator[pl.DataFrame]:
        if timeframe not in {"1Day", "1Min"} or not symbols or start >= end:
            raise ValueError("Invalid bar request")
        lower, upper = session_bounds(start, end)
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day if timeframe == "1Day" else TimeFrame.Minute,
            start=lower,
            end=upper - timedelta(microseconds=1),
            feed=DataFeed.SIP,
            adjustment=Adjustment.RAW,
            asof="-",
            sort=Sort.ASC,
            limit=self.options.page_size,
        )
        params = request.to_request_fields()
        params["symbols"] = ",".join(symbols)
        params.pop("symbol_or_symbols", None)
        seen = set()
        while True:
            result = self._get(f"{DATA_URL}/v2/stocks/bars", params)
            if not isinstance(result, dict) or not isinstance(result.get("bars"), dict):
                raise ValueError("Invalid Alpaca bars response; coverage remains incomplete")
            frame = bars_frame(result["bars"])
            if frame.filter(~pl.col("symbol").is_in(symbols)).height:
                raise ValueError("Unexpected symbol in Alpaca response")
            yield frame.filter((pl.col("timestamp") >= lower) & (pl.col("timestamp") < upper))
            token = result.get("next_page_token")
            if not token:
                break
            if token in seen:
                raise ValueError("Alpaca repeated a pagination token")
            seen.add(token)
            params["page_token"] = token
