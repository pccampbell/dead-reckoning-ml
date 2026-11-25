from datetime import UTC, date, datetime

import polars as pl
import pytest

from dead_reckoning.config import Credentials, load_config
from dead_reckoning.ingestion.bars import ingest_bars
from dead_reckoning.ingestion.manifest import Manifest, missing_intervals
from dead_reckoning.market_data.alpaca import BAR_SCHEMA, AlpacaClient, bars_frame
from dead_reckoning.storage.parquet import read_metadata, write_pages


def test_missing_coverage_gaps_and_extensions():
    def day(number):
        return date(2025, 1, number)

    complete = [(day(2), day(5)), (day(4), day(8)), (day(10), day(12))]
    assert missing_intervals(day(1), day(14), complete) == [
        (day(1), day(2)),
        (day(8), day(10)),
        (day(12), day(14)),
    ]
    assert not missing_intervals(day(2), day(8), complete)
    assert missing_intervals(day(2), day(10), complete) == [(day(8), day(10))]


def test_pagination_ignores_short_and_empty_pages(config_path):
    config = load_config(config_path)
    client = AlpacaClient(Credentials(api_key="fixture", secret_key="fixture"), config.ingestion)
    stamp = "2025-01-02T15:00:00Z"
    bar = {"t": stamp, "o": 1, "h": 1, "l": 1, "c": 1, "v": 5, "n": 1, "vw": 1}
    responses = iter(
        [
            {"bars": {"AAA": [bar]}, "next_page_token": "a"},
            {"bars": {}, "next_page_token": "b"},
            {"bars": {"BBB": [bar]}, "next_page_token": None},
        ]
    )
    tokens = []

    def get(_, params):
        tokens.append(params.get("page_token"))
        return next(responses)

    client._get = get
    pages = list(client.bar_pages(["AAA", "BBB"], "1Min", date(2025, 1, 2), date(2025, 1, 3)))
    assert tokens == [None, "a", "b"]
    assert pl.concat(pages)["symbol"].to_list() == ["AAA", "BBB"]
    client.close()


def test_permanent_api_error_shows_provider_reason_without_credentials(config_path, market_server):
    market_server.failures = [400]
    market_server.failure_message = "invalid symbol: 0029900E0 (fixture-key fixture-secret)"
    client = AlpacaClient(Credentials(), load_config(config_path).ingestion)
    try:
        with pytest.raises(RuntimeError, match="Alpaca HTTP 400") as error:
            list(client.bar_pages(["AAA"], "1Day", date(2025, 1, 2), date(2025, 1, 3)))
        assert "fixture-key" not in str(error.value)
        assert "fixture-secret" not in str(error.value)
        assert "invalid symbol: 0029900E0" in str(error.value)
        assert '"code": 40010001' in str(error.value)
        assert len(market_server.requests) == 1  # Permanent errors are not retried.
        assert client.telemetry.snapshot()["http_requests_total"] == 1
    finally:
        client.close()


def test_http_telemetry_counts_retries_pages_and_received_rows(config_path, market_server):
    market_server.failures = [503]
    client = AlpacaClient(Credentials(), load_config(config_path).ingestion)
    client.telemetry.start_stage("daily_bars")
    try:
        pages = list(client.bar_pages(["AAA"], "1Day", date(2025, 1, 2), date(2025, 1, 10)))
        snapshot = client.telemetry.snapshot()
        assert sum(page.height for page in pages) == 5
        assert snapshot["rows_received"] == 5
        assert snapshot["http_requests"] == len(market_server.requests) == 3
    finally:
        client.close()


def test_atomic_roundtrip_and_page_boundary_dedup(tmp_path):
    frame = pl.DataFrame(
        {
            "symbol": ["AAA"],
            "timestamp": [datetime(2025, 1, 2, 15, tzinfo=UTC)],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [10.5],
            "trade_count": [None],
            "vwap": [None],
        },
        schema=BAR_SCHEMA,
    )
    path = tmp_path / "bars.parquet"
    counts = write_pages(path, [frame, frame], {"symbols": ["AAA"], "feed": "sip"})
    assert counts == {"AAA": 1}
    assert pl.read_parquet(path).equals(frame)
    assert read_metadata(path)["feed"] == "sip"

    def broken_pages():
        yield frame
        raise ConnectionError("interrupted")

    with pytest.raises(ConnectionError):
        write_pages(path, broken_pages(), {"symbols": ["AAA"]})
    assert pl.read_parquet(path).equals(frame)  # Existing final file survives a failed replacement.
    assert not list(tmp_path.glob("*.partial"))


def test_invalid_bar_values_cannot_enter_raw_storage():
    bar = {"t": "2025-01-02T15:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 5}
    for field, value in (("o", float("nan")), ("v", float("inf")), ("v", -1), ("c", None)):
        with pytest.raises(ValueError, match="bar"):
            bars_frame({"AAA": [{**bar, field: value}]})


def test_add_symbols_empty_period_and_missing_file(config_path, market_server):
    config = load_config(config_path)
    client = AlpacaClient(Credentials(), config.ingestion)
    with Manifest(config.data.root) as manifest:
        # Weekend: typed empty file still seals queried coverage.
        start, end = date(2025, 1, 4), date(2025, 1, 6)
        result = ingest_bars(config, manifest, client, "daily_bars", ["AAA"], start, end)
        assert result["rows"] == 0 and result["batches"] == 1
        assert client.telemetry.snapshot()["http_requests"] == 1
        assert client.telemetry.snapshot()["rows_received"] == 0
        assert (
            ingest_bars(config, manifest, client, "daily_bars", ["AAA"], start, end)["batches"] == 0
        )
        assert client.telemetry.snapshot()["http_requests"] == 0
        assert client.telemetry.snapshot()["http_requests_total"] == 1
        request_count = len(market_server.requests)
        ingest_bars(config, manifest, client, "daily_bars", ["AAA", "BBB"], start, end)
        assert market_server.requests[request_count][1]["symbols"] == "BBB"
        path = manifest.files("daily_bars")[0]
    path.unlink()
    with Manifest(config.data.root) as manifest:
        assert len(manifest.files("daily_bars")) == 1
        result = ingest_bars(config, manifest, client, "daily_bars", ["AAA", "BBB"], start, end)
        assert result["batches"] == 1
    client.close()
