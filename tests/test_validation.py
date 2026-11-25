import json
from datetime import UTC, date, datetime, timedelta

import duckdb
import polars as pl
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from dead_reckoning.cli import app
from dead_reckoning.config import load_config
from dead_reckoning.ingestion.manifest import Manifest
from dead_reckoning.ingestion.validation import validate_data
from dead_reckoning.market_data.alpaca import BAR_SCHEMA
from dead_reckoning.storage.parquet import METADATA_KEY, write_pages


def store(root, name="a", frame=None, symbols=None):
    if frame is None:
        frame = pl.DataFrame(
            {
                "symbol": ["AAA"],
                "timestamp": [datetime(2025, 1, 2, 15, tzinfo=UTC)],
                "open": [10.0],
                "high": [11.0],
                "low": [9.0],
                "close": [10.0],
                "volume": [10.0],
                "trade_count": [None],
                "vwap": [None],
            },
            schema=BAR_SCHEMA,
        )
    meta = {
        "schema_version": 1,
        "provider": "alpaca",
        "dataset": "daily_bars",
        "timeframe": "1Day",
        "feed": "sip",
        "adjustment": "raw",
        "asof": "-",
        "symbols": symbols or ["AAA"],
        "start": "2025-01-02",
        "end": "2025-01-03",
        "ingested_at": "2025-01-04T00:00:00+00:00",
    }
    path = root / f"raw/daily_bars/{name}.parquet"
    counts = write_pages(path, [frame], meta)
    with Manifest(root) as manifest:
        manifest.record(path, meta, "complete", counts)
    return path


def rewrite(path, transform):
    table = pq.ParquetFile(path).read()
    metadata = table.schema.metadata
    frame = transform(pl.from_arrow(table))
    pq.write_table(frame.to_arrow().replace_schema_metadata(metadata), path)


def daily_issues(root, **kwargs):
    # These focused fixtures contain only daily data; missing minute data is expected.
    return [
        issue for issue in validate_data(root, **kwargs)["issues"] if "minute_bars" not in issue
    ]


def test_empty_symbols_and_read_only_validation(config_path):
    root = load_config(config_path).data.root
    path = store(root, symbols=["AAA", "BBB"])
    # Also exercise a completely empty completed request.
    empty = store(root, "empty", pl.DataFrame(schema=BAR_SCHEMA), symbols=["CCC"])
    before = {
        p: (p.stat().st_mtime_ns, p.read_bytes())
        for p in (path, empty, root / "metadata/ingestion.duckdb")
    }
    assert not daily_issues(root, full=True)
    assert all((p.stat().st_mtime_ns, p.read_bytes()) == value for p, value in before.items())


@pytest.mark.parametrize("damage", ["ohlc", "duplicate", "range", "symbol", "null", "infinite"])
def test_bad_rows_that_status_would_trust(config_path, damage):
    root = load_config(config_path).data.root
    path = store(root, symbols=["AAA", "BBB"])
    if damage == "duplicate":
        rewrite(path, lambda f: pl.concat([f, f]))
        # Keep manifest and footer totals consistent: only scanning keys finds the issue.
        with duckdb.connect(str(root / "metadata/ingestion.duckdb")) as db:
            db.execute("UPDATE coverage SET row_count = 2 WHERE symbol = 'AAA'")
    else:
        column, value = {
            "ohlc": ("high", 5.0),
            "range": ("timestamp", datetime(2025, 1, 4, 15, tzinfo=UTC)),
            "symbol": ("symbol", "BBB"),  # Same total, wrong per-symbol counts.
            "null": ("close", None),
            "infinite": ("vwap", float("inf")),
        }[damage]
        rewrite(
            path, lambda f: f.with_columns(pl.lit(value).cast(BAR_SCHEMA[column]).alias(column))
        )
    with Manifest(root) as manifest:
        assert manifest.summary("daily_bars")[1] == (2 if damage == "duplicate" else 1)
        assert manifest.coverage("daily_bars")["AAA"] == [(date(2025, 1, 2), date(2025, 1, 3))]
    assert daily_issues(root, full=True)


def test_missing_corrupt_metadata_and_count_mismatch(config_path):
    root = load_config(config_path).data.root
    path = store(root)
    with duckdb.connect(str(root / "metadata/ingestion.duckdb")) as db:
        db.execute("UPDATE coverage SET row_count = 10")
    assert any("footer row count" in issue for issue in daily_issues(root))
    path.write_bytes(b"broken parquet")
    assert any("a.parquet" in issue for issue in daily_issues(root))
    path.unlink()
    assert any("completed file is missing" in issue for issue in daily_issues(root))
    # Validation didn't erase the missing-file entry, unlike status reconciliation.
    with duckdb.connect(str(root / "metadata/ingestion.duckdb"), read_only=True) as db:
        assert db.execute("SELECT count(*) FROM coverage").fetchone()[0] == 1


def test_overlapping_files_and_metadata_disagreement(config_path):
    root = load_config(config_path).data.root
    store(root)
    other = store(root, "b")
    assert any("overlapping" in issue for issue in daily_issues(root))
    table = pq.read_table(other)
    meta = json.loads(table.schema.metadata[METADATA_KEY])
    meta["symbols"] = ["BBB"]
    pq.write_table(table.replace_schema_metadata({METADATA_KEY: json.dumps(meta).encode()}), other)
    assert any("request metadata disagrees" in issue for issue in daily_issues(root))


def test_duplicate_across_streaming_batch_boundary(config_path):
    root = load_config(config_path).data.root
    path = store(root)
    first = datetime(2025, 1, 2, 15, tzinfo=UTC)
    timestamps = [first + timedelta(microseconds=i) for i in range(100_000)]
    timestamps.append(timestamps[-1])
    # Bypass the production writer's deduplication to simulate corrupted stored data.
    rewrite(
        path,
        lambda f: (
            f.join(pl.DataFrame({"timestamp": timestamps}), how="cross", suffix="_new")
            .drop("timestamp")
            .rename({"timestamp_new": "timestamp"})
            .select(list(BAR_SCHEMA))
        ),
    )
    with duckdb.connect(str(root / "metadata/ingestion.duckdb")) as db:
        db.execute("UPDATE coverage SET row_count = 100001")
    assert any("across batches" in issue for issue in daily_issues(root, full=True))


def test_full_scan_catches_damage_outside_sample(config_path):
    root = load_config(config_path).data.root
    first = store(root, "a")
    frame = pl.read_parquet(first)
    damaged = store(root, "b", frame.with_columns(pl.lit("BBB").alias("symbol")), ["BBB"])
    store(root, "c", frame.with_columns(pl.lit("CCC").alias("symbol")), ["CCC"])
    rewrite(damaged, lambda f: f.with_columns(pl.lit(-1.0).alias("volume")))
    # Sampling is explicitly limited; intact footers don't guarantee intact unscanned rows.
    assert not daily_issues(root, sample_files=1)
    assert any("invalid bars" in issue for issue in daily_issues(root, full=True))


def test_cli_sample_full_and_failure_exit_code(config_path, market_server):
    runner = CliRunner()
    base = ["--config", str(config_path)]
    bootstrap = runner.invoke(
        app, [*base, "data", "bootstrap", "--start", "2025-01-02", "--end", "2025-01-10"]
    )
    assert bootstrap.exit_code == 0, bootstrap.output
    requests = len(market_server.requests)
    for options, label in (([], "sampled row scan"), (["--full"], "full row scan")):
        result = runner.invoke(app, [*base, "data", "validate", *options])
        assert result.exit_code == 0, result.output + repr(result.exception)
        assert label in result.output and "rows scanned" in result.output
    assert len(market_server.requests) == requests
    root = load_config(config_path).data.root
    path = next((root / "raw/daily_bars").rglob("*.parquet"))
    rewrite(path, lambda f: f.with_columns(pl.lit(-1.0).alias("volume")))
    result = runner.invoke(app, [*base, "data", "validate", "--full"])
    assert result.exit_code == 1 and "invalid bars" in result.output
