import json
import logging
import os
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

import duckdb
import polars as pl

from dead_reckoning.config import Config, validate_range
from dead_reckoning.ingestion.manifest import Manifest, missing_intervals
from dead_reckoning.market_data.alpaca import session_bounds
from dead_reckoning.storage.parquet import atomic_json

log = logging.getLogger(__name__)


def current_universe(root: Path) -> tuple[Path, dict] | None:
    pointer = root / "curated/universe/current.json"
    if not pointer.exists():
        return None
    meta = json.loads(pointer.read_text(encoding="utf-8"))
    folder = pointer.parent / meta["generation"]
    if (
        not (folder / "symbols.parquet").is_file()
        or not (folder / "daily_liquidity.parquet").is_file()
    ):
        return None
    return folder, meta


def build_universe(
    config: Config,
    manifest: Manifest,
    start: date | None = None,
    end: date | None = None,
) -> tuple[Path, dict]:
    files = manifest.files("daily_bars")
    if not files:
        raise ValueError("No daily coverage; run 'ingest daily-bars' first")
    lower, upper = manifest.bounds("daily_bars")
    start, end = start or lower, end or upper
    validate_range(start, end)
    incomplete = [
        symbol
        for symbol, intervals in manifest.coverage("daily_bars").items()
        if missing_intervals(start, end, intervals)
    ]
    if incomplete:
        raise ValueError(
            f"Daily coverage is incomplete for {len(incomplete)} known symbols over "
            f"[{start}, {end}); run daily ingestion or bootstrap for this range first"
        )
    signature = {
        "schema_version": 1,
        "provider": "alpaca",
        "feed": "sip",
        "adjustment": "raw",
        "asof": "-",
        "daily_fingerprint": manifest.fingerprint("daily_bars"),
        "start": start.isoformat(),
        "end": end.isoformat(),
        **config.universe.model_dump(),
    }
    current = current_universe(config.data.root)
    if current and all(current[1].get(key) == value for key, value in signature.items()):
        log.info("Universe: unchanged inputs, existing generation skipped")
        return current

    # DuckDB can spill sorts/window calculations to disk instead of loading years into Python.
    base = config.data.root / "curated/universe"
    generation = uuid4().hex
    staging = base / f".{generation}.partial"
    staging.mkdir(parents=True)
    final = base / generation
    options = config.universe
    log.info(
        "Universe: building liquidity with DuckDB memory_limit=%s, threads=%s",
        config.query.memory_limit,
        config.query.threads,
    )
    with duckdb.connect(
        config={
            "memory_limit": config.query.memory_limit,
            "threads": config.query.threads,
            "temp_directory": str(config.data.root / "metadata/query_tmp"),
            "preserve_insertion_order": False,
        }
    ) as db:
        # Deduplicate defensively by natural key. Rolling precedes requested-date filtering,
        # so previously downloaded warm-up history is included. No current-day volume enters
        # either the historical sample count or median used in that day's ranking.
        query = f"""
            WITH daily AS (
                SELECT symbol, CAST(timezone('America/New_York', timestamp) AS DATE) AS date,
                       close * volume AS dollar_volume
                FROM read_parquet($paths, hive_partitioning=false)
                WHERE timestamp < $end_boundary
                QUALIFY row_number() OVER (PARTITION BY symbol, timestamp) = 1
            ), rolling AS (
                SELECT *,
                    median(dollar_volume) OVER prior_bars AS trailing_median_dollar_volume,
                    count(dollar_volume) OVER prior_bars AS history_days
                FROM daily
                WINDOW prior_bars AS (PARTITION BY symbol ORDER BY date
                    ROWS BETWEEN {options.lookback_days} PRECEDING AND 1 PRECEDING)
            ), eligible AS (
                SELECT *, history_days >= {options.min_history_days} AS eligible
                FROM rolling WHERE date >= $start AND date < $end
            ), ranked AS (
                SELECT *, CASE WHEN eligible THEN row_number() OVER (
                    PARTITION BY date, eligible
                    ORDER BY trailing_median_dollar_volume DESC, symbol ASC
                ) END AS liquidity_rank
                FROM eligible
            )
            SELECT * FROM ranked ORDER BY date, symbol
        """
        db.sql(
            query,
            params={
                "paths": [str(path) for path in files],
                "start": start,
                "end": end,
                "end_boundary": session_bounds(start, end)[1],
            },
        ).write_parquet(str(staging / "daily_liquidity.parquet"), compression="zstd")
        db.sql(
            """
            SELECT symbol, min(date) AS first_qualifying_date,
                max(date) AS last_qualifying_date, min(liquidity_rank) AS best_liquidity_rank,
                max(trailing_median_dollar_volume) AS peak_qualifying_median_dollar_volume
            FROM read_parquet($path) WHERE liquidity_rank <= $rank
            GROUP BY symbol ORDER BY symbol
        """,
            params={
                "path": str(staging / "daily_liquidity.parquet"),
                "rank": options.ingestion_rank,
            },
        ).write_parquet(str(staging / "symbols.parquet"), compression="zstd")
    count = pl.scan_parquet(staging / "symbols.parquet").select(pl.len()).collect().item()
    meta = {
        **signature,
        "generation": generation,
        "symbol_count": count,
        "built_at": datetime.now(UTC).isoformat(),
        "criterion": f"ever top {options.ingestion_rank} by prior "
        f"{options.lookback_days}-observation median dollar volume",
    }
    atomic_json(staging / "metadata.json", meta)
    os.replace(staging, final)
    # One pointer swaps both artifacts together; interruptions preserve the previous generation.
    atomic_json(base / "current.json", meta)
    log.info("Universe: %s symbols; %s", count, meta["criterion"])
    return final, meta


def universe_symbols(folder: Path) -> list[str]:
    return pl.read_parquet(folder / "symbols.parquet", columns=["symbol"])["symbol"].to_list()
