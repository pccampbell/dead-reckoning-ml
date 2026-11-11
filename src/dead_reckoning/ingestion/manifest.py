"""DuckDB coverage index, reconstructed from durable raw Parquet when needed."""

import hashlib
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import duckdb
import polars as pl

from dead_reckoning.storage.parquet import read_metadata


def missing_intervals(
    start: date,
    end: date,
    completed: list[tuple[date, date]],
) -> list[tuple[date, date]]:
    cursor = start
    missing = []
    for lower, upper in sorted(completed):
        if upper <= cursor or lower >= end:
            continue
        if lower > cursor:
            missing.append((cursor, min(lower, end)))
        cursor = max(cursor, upper)
        if cursor >= end:
            break
    if cursor < end:
        missing.append((cursor, end))
    return missing


class Manifest:
    def __init__(self, root: Path):
        self.root = root
        (root / "metadata").mkdir(parents=True, exist_ok=True)
        # One CLI writer per data root. DuckDB's OS lock also survives stale process exits.
        self.db = duckdb.connect(str(root / "metadata/ingestion.duckdb"))
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS coverage (
                path VARCHAR, dataset VARCHAR, symbol VARCHAR, timeframe VARCHAR,
                feed VARCHAR, adjustment VARCHAR, start_date DATE, end_date DATE,
                status VARCHAR, row_count BIGINT, updated_at TIMESTAMPTZ,
                PRIMARY KEY (path, symbol)
            )
        """)
        try:
            self.reconcile()
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        self.db.close()

    def record(self, path: Path, meta: dict, status: str, counts: dict[str, int]) -> None:
        rows = [
            (
                str(path.relative_to(self.root)),
                meta["dataset"],
                symbol,
                meta["timeframe"],
                meta["feed"],
                meta["adjustment"],
                meta["start"],
                meta["end"],
                status,
                counts.get(symbol, 0),
                meta["ingested_at"],
            )
            for symbol in meta["symbols"]
        ]
        self.db.execute("BEGIN")
        try:
            self.db.executemany(
                "INSERT OR REPLACE INTO coverage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def reconcile(self) -> None:
        known = {
            row[0]
            for row in self.db.execute(
                "SELECT DISTINCT path FROM coverage WHERE status = 'complete'"
            ).fetchall()
        }
        for relative in known.copy():
            if not (self.root / relative).is_file():
                self.db.execute("DELETE FROM coverage WHERE path = ?", [relative])
                known.remove(relative)
        for dataset in ("daily_bars", "minute_bars"):
            for path in sorted((self.root / "raw" / dataset).rglob("*.parquet")):
                if str(path.relative_to(self.root)) in known:
                    continue
                meta = read_metadata(path)
                if meta["schema_version"] != 1 or meta["dataset"] != dataset:
                    raise ValueError(f"Unsupported raw metadata in {path}")
                counts = dict(pl.scan_parquet(path).group_by("symbol").len().collect().iter_rows())
                self.record(path, meta, "complete", counts)

    def coverage(self, dataset: str) -> dict[str, list[tuple[date, date]]]:
        result = defaultdict(list)
        for symbol, start, end in self.db.execute(
            "SELECT symbol, start_date, end_date FROM coverage "
            "WHERE dataset = ? AND status = 'complete' AND feed = 'sip' AND adjustment = 'raw'",
            [dataset],
        ).fetchall():
            result[symbol].append((start, end))
        return result

    def files(self, dataset: str) -> list[Path]:
        return [
            self.root / row[0]
            for row in self.db.execute(
                "SELECT DISTINCT path FROM coverage WHERE dataset = ? AND status = 'complete' "
                "ORDER BY path",
                [dataset],
            ).fetchall()
        ]

    def fingerprint(self, dataset: str) -> str:
        rows = self.db.execute(
            "SELECT path, symbol, start_date, end_date, row_count, epoch_us(updated_at) "
            "FROM coverage WHERE dataset = ? AND status = 'complete' ORDER BY path, symbol",
            [dataset],
        ).fetchall()
        return hashlib.sha256(json.dumps(rows, default=str).encode()).hexdigest()

    def bounds(self, dataset: str) -> tuple[date | None, date | None]:
        return self.db.execute(
            "SELECT min(start_date), max(end_date) FROM coverage "
            "WHERE dataset = ? AND status = 'complete'",
            [dataset],
        ).fetchone()

    def summary(self, dataset: str) -> tuple:
        return self.db.execute(
            "SELECT count(DISTINCT symbol), coalesce(sum(row_count), 0), "
            "min(start_date), max(end_date), count(*) FILTER (WHERE row_count = 0) "
            "FROM coverage WHERE dataset = ? AND status = 'complete'",
            [dataset],
        ).fetchone()
