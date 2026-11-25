"""Read-only checks of durable bars against the existing coverage index."""

import json
import logging
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from time import monotonic

import duckdb
import polars as pl
import pyarrow.parquet as pq

from dead_reckoning.market_data.alpaca import BAR_SCHEMA, session_bounds
from dead_reckoning.storage.parquet import METADATA_KEY

log = logging.getLogger(__name__)


def _scan_bars(parquet: pq.ParquetFile, meta: dict, expected: dict[str, int]) -> int:
    """Bounded batches; last timestamps catch duplicates even across batch boundaries."""
    lower, upper = session_bounds(
        date.fromisoformat(meta["start"]), date.fromisoformat(meta["end"])
    )
    counts = Counter(dict.fromkeys(expected, 0))
    last = {}
    for batch in parquet.iter_batches(batch_size=100_000):
        frame = pl.from_arrow(batch)
        required = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        invalid = frame.filter(
            pl.any_horizontal(pl.col(required).is_null())
            | ~pl.all_horizontal(pl.col("open", "high", "low", "close", "volume").is_finite())
            | (pl.min_horizontal("open", "high", "low", "close") <= 0)
            | (pl.col("low") > pl.min_horizontal("open", "close", "high"))
            | (pl.col("high") < pl.max_horizontal("open", "close", "low"))
            | (pl.col("volume") < 0)
            | (pl.col("trade_count") < 0).fill_null(False)
            | (~pl.col("vwap").is_finite()).fill_null(False)
            | ~pl.col("symbol").is_in(list(expected))
            | (pl.col("timestamp") < lower)
            | (pl.col("timestamp") >= upper)
        )
        if invalid.height:
            example = invalid.select("symbol", "timestamp").row(0)
            raise ValueError(f"{invalid.height:,} invalid bars in batch; first key {example}")
        if frame.select((pl.col("timestamp").diff().over("symbol") <= pl.duration()).any()).item():
            raise ValueError("duplicate or out-of-order symbol/timestamp keys within batch")
        stats = frame.group_by("symbol").agg(
            pl.len().alias("rows"),
            pl.col("timestamp").min().alias("first"),
            pl.col("timestamp").max().alias("last"),
        )
        for symbol, rows, first, final in stats.iter_rows():
            if symbol in last and first <= last[symbol]:
                raise ValueError(
                    f"duplicate or out-of-order timestamps across batches for {symbol}"
                )
            counts[symbol] += rows
            last[symbol] = final
    if dict(counts) != expected:
        raise ValueError(
            "per-symbol stored row counts disagree with manifest (including empty symbols)"
        )
    return counts.total()


def validate_data(root: Path, *, sample_files: int = 3, full: bool = False) -> dict:
    """Check all raw bar footers and sample whole files, without reconciling or repairing."""
    if sample_files < 1:
        raise ValueError("sample_files must be positive")
    database = root / "metadata/ingestion.duckdb"
    if not database.is_file():
        raise ValueError("No ingestion manifest; ingest data first")
    expected = defaultdict(list)
    issues = []
    # Do not open Manifest: its reconciliation would modify evidence before checking it.
    with duckdb.connect(str(database), read_only=True) as db:
        for row in db.execute(
            "SELECT path, dataset, symbol, timeframe, feed, adjustment, start_date, end_date, "
            "row_count FROM coverage WHERE status = 'complete' ORDER BY path, symbol"
        ).fetchall():
            expected[row[0]].append(row[1:])
        overlaps = db.execute("""
            WITH intervals AS (
                SELECT dataset, symbol, start_date,
                    max(end_date) OVER (PARTITION BY dataset, symbol ORDER BY start_date, end_date
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_end
                FROM coverage WHERE status = 'complete'
            )
            SELECT dataset, count(*) FROM intervals WHERE start_date < prior_end GROUP BY dataset
        """).fetchall()
        for dataset, count in overlaps:
            issues.append(
                f"{dataset}: {count:,} overlapping completed symbol intervals; possible duplicates"
            )

    summaries = []
    last_log = monotonic()
    for dataset in ("daily_bars", "minute_bars"):
        files = sorted((root / "raw" / dataset).rglob("*.parquet"))
        relatives = {str(path.relative_to(root)) for path in files}
        for relative, entries in expected.items():
            if entries[0][0] == dataset and relative not in relatives:
                issues.append(f"{relative}: completed file is missing")
        if not files:
            issues.append(f"{dataset}: no raw bar files")
        # Sample nonempty requests so an all-empty sample cannot give false confidence.
        candidates = [
            path
            for path in files
            if sum(entry[-1] for entry in expected.get(str(path.relative_to(root)), [])) > 0
        ]
        # Spread over sorted request paths; no promise of statistical coverage.
        size = min(sample_files, len(candidates))
        selected = (
            set(files)
            if full
            else {candidates[i * (len(candidates) - 1) // max(1, size - 1)] for i in range(size)}
        )
        scanned_files = scanned_rows = 0
        log.info(
            "Validating %s: %s footers; %s whole files selected for row scans",
            dataset,
            len(files),
            len(selected),
        )
        for index, path in enumerate(files, 1):
            relative = str(path.relative_to(root))
            try:
                entries = expected.get(relative)
                if not entries:
                    raise ValueError(
                        "raw file has no completed manifest entry; run data status to reconcile"
                    )
                with pq.ParquetFile(path) as parquet:
                    meta = json.loads(parquet.metadata.metadata[METADATA_KEY])
                    if (
                        meta["schema_version"] != 1
                        or meta["provider"] != "alpaca"
                        or meta["dataset"] != dataset
                        or meta["feed"] != "sip"
                        or meta["adjustment"] != "raw"
                        or meta["asof"] != "-"
                        or meta["timeframe"] != ("1Day" if dataset == "daily_bars" else "1Min")
                        or meta["start"] >= meta["end"]
                    ):
                        raise ValueError("unexpected request metadata")
                    if pl.Schema(parquet.schema_arrow) != pl.Schema(BAR_SCHEMA):
                        raise ValueError("bar schema/types disagree with schema version 1")
                    symbols = {entry[1]: entry[-1] for entry in entries}
                    request = (
                        dataset,
                        meta["timeframe"],
                        meta["feed"],
                        meta["adjustment"],
                        date.fromisoformat(meta["start"]),
                        date.fromisoformat(meta["end"]),
                    )
                    if set(meta["symbols"]) != set(symbols) or any(
                        (stored_dataset, timeframe, feed, adjustment, start, end) != request
                        for stored_dataset, _, timeframe, feed, adjustment, start, end, _ in entries
                    ):
                        raise ValueError("request metadata disagrees with manifest")
                    if parquet.metadata.num_rows != sum(symbols.values()):
                        raise ValueError("footer row count disagrees with manifest")
                    if path in selected:
                        scanned_rows += _scan_bars(parquet, meta, symbols)
                        scanned_files += 1
            except (OSError, ValueError, KeyError, TypeError, pl.exceptions.PolarsError) as exc:
                issues.append(f"{relative}: {exc}")
            if monotonic() - last_log >= 30:
                log.info("Validation %s: %s/%s files checked", dataset, index, len(files))
                last_log = monotonic()
        summaries.append(
            f"{dataset}: {len(files):,} files inspected; "
            f"{scanned_files:,} whole files / {scanned_rows:,} rows scanned"
        )
    return {"summaries": summaries, "issues": issues}
