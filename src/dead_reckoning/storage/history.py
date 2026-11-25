"""Bounded local historical extraction shared by research and datasets."""

from datetime import timedelta
from pathlib import Path

import duckdb
import polars as pl


def minute_inventory(root: Path, start, end) -> pl.DataFrame:
    """Read only relevant completed minute-request coverage, without touching a provider."""
    path = root / "metadata/ingestion.duckdb"
    if not path.is_file():
        raise ValueError("No ingestion manifest; dataset generation never downloads data")
    with duckdb.connect(str(path), read_only=True) as db:
        return db.sql(
            """
            SELECT * FROM coverage WHERE status='complete' AND dataset='minute_bars'
                AND timeframe='1Min' AND feed='sip' AND adjustment='raw'
                AND start_date < $end AND end_date > $start ORDER BY path, symbol
            """,
            params={"start": start, "end": end},
        ).pl()


def inventory(root: Path, sessions: pl.DataFrame) -> tuple[pl.DataFrame, list[str], dict]:
    path = root / "metadata/ingestion.duckdb"
    if not path.is_file():
        raise ValueError("No ingestion manifest; ingest historical data first")
    with duckdb.connect(str(path), read_only=True) as db:
        coverage = db.sql(
            """
            SELECT * FROM coverage WHERE status='complete' AND feed='sip' AND adjustment='raw'
                AND start_date <= $last
                AND (dataset='daily_bars' OR end_date > $first)
        """,
            params={"first": min(sessions["date"]), "last": max(sessions["date"])},
        ).pl()
    end = max(sessions["date"]) + timedelta(days=1)
    daily = (
        coverage.filter((pl.col("dataset") == "daily_bars") & (pl.col("start_date") < end))["path"]
        .unique()
        .sort()
        .to_list()
    )
    if not daily:
        raise ValueError("No historical daily Parquet files in manifest")
    import hashlib

    signature = hashlib.sha256(
        coverage.sort(["path", "symbol"]).hash_rows(seed=42).to_numpy().tobytes()
    ).hexdigest()
    return coverage, [str(root / p) for p in daily], {"manifest_fingerprint": signature}


def minute_paths(coverage: pl.DataFrame, root: Path, day, symbols: list[str]):
    rows = coverage.filter(
        (pl.col("dataset") == "minute_bars")
        & pl.col("symbol").is_in(symbols)
        & (pl.col("start_date") <= day)
        & (pl.col("end_date") > day)
    )
    missing = sorted(set(symbols) - set(rows["symbol"]))
    if missing:
        raise ValueError(f"Minute manifest missing {len(missing)} symbols on {day}: {missing[:5]}")
    paths = [str(root / p) for p in rows["path"].unique().sort()]
    if any(not Path(p).is_file() for p in paths):
        raise ValueError(f"Manifest references missing minute files on {day}")
    return paths


def read_minutes(db, paths, symbols, opening, closing) -> pl.DataFrame:
    return db.sql(
        """
        SELECT symbol, timestamp, open, high, low, close
        FROM read_parquet($paths, hive_partitioning=false)
        WHERE symbol IN (SELECT unnest($symbols)) AND timestamp >= $opening AND timestamp < $closing
        ORDER BY symbol, timestamp
    """,
        params={"paths": paths, "symbols": symbols, "opening": opening, "closing": closing},
    ).pl()
