import json
import logging
import os
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import duckdb
import exchange_calendars as xcals
import polars as pl

from dead_reckoning.config import Config, validate_range
from dead_reckoning.ingestion.assets import asset_symbols
from dead_reckoning.ingestion.manifest import Manifest, missing_intervals
from dead_reckoning.market_data.alpaca import session_bounds
from dead_reckoning.storage.parquet import atomic_json

log = logging.getLogger(__name__)
UNIVERSE_ALGORITHM = "prior_session_asof_v2"


def session_dates(start: date, end: date) -> pl.DataFrame:
    """A calendar independent of any stock's current-day bar availability."""
    calendar = xcals.get_calendar("XNYS", start=str(start), end=str(end))
    return pl.DataFrame(
        {
            "date": pl.Series(
                [stamp.date() for stamp in calendar.sessions if start <= stamp.date() < end],
                dtype=pl.Date,
            )
        }
    )


def prior_liquidity_sql(lookback: int, minimum: int, staleness: int) -> str:
    """Effective prior-state intervals implement a strict ASOF without a huge cross join.

    The next observation only bounds when a prior snapshot stops being current.
    The interval includes that next observation's date: its values cannot enter
    eligibility or ranks until a later session. All output measures are historical.
    """
    return f"""
        WITH daily AS (
            SELECT symbol, CAST(timezone('America/New_York', timestamp) AS DATE) AS feature_date,
                   close * volume AS dollar_volume
            FROM read_parquet($paths, hive_partitioning=false)
            WHERE timestamp < $end_boundary
                AND isfinite(close) AND close > 0 AND isfinite(volume) AND volume >= 0
            QUALIFY row_number() OVER (
                PARTITION BY symbol, timestamp ORDER BY close, volume
            ) = 1
        ), rolling AS (
            SELECT *,
                median(dollar_volume) OVER prior_state AS trailing_median_dollar_volume,
                count(dollar_volume) OVER prior_state AS history_days,
                lead(feature_date) OVER (PARTITION BY symbol ORDER BY feature_date) AS next_date
            FROM daily WINDOW prior_state AS (PARTITION BY symbol ORDER BY feature_date
                ROWS BETWEEN {lookback - 1} PRECEDING AND CURRENT ROW)
        ), prior AS (
            SELECT r.symbol, s.date, r.feature_date,
                r.dollar_volume AS last_observed_dollar_volume,
                r.trailing_median_dollar_volume, r.history_days,
                r.history_days >= {minimum} AND r.trailing_median_dollar_volume > 0 AS eligible
            FROM rolling r JOIN trading_sessions s
                ON s.date > r.feature_date AND s.date <= r.feature_date + {staleness}
                AND (r.next_date IS NULL OR s.date <= r.next_date)
        ), ranked AS (
            SELECT *, CASE WHEN eligible THEN row_number() OVER (
                PARTITION BY date, eligible
                ORDER BY trailing_median_dollar_volume DESC, symbol ASC
            ) END AS liquidity_rank FROM prior
        )
        SELECT * FROM ranked ORDER BY date, symbol
    """


def compare_universes(previous, folder: Path, start: date, end: date) -> dict:
    old = set(universe_symbols(previous[0])) if previous else set()
    new = set(universe_symbols(folder))
    records = [
        {
            "symbol": symbol,
            "status": "unchanged"
            if symbol in old & new
            else "newly_included"
            if symbol in new
            else "no_longer_required",
        }
        for symbol in sorted(old | new)
    ]
    pl.DataFrame(records, schema={"symbol": pl.String, "status": pl.String}).write_csv(
        folder / "symbol_changes.csv"
    )
    result = {
        "previous_generation": previous[1]["generation"] if previous else None,
        "previous_count": len(old),
        "corrected_count": len(new),
        "unchanged": len(old & new),
        "newly_included": len(new - old),
        "no_longer_required": len(old - new),
        "new_symbols": sorted(new - old),
        "removed_symbols": sorted(old - new),
        "same_date_range": bool(
            previous
            and previous[1].get("start") == str(start)
            and previous[1].get("end") == str(end)
        ),
        "previous_start": previous[1].get("start") if previous else None,
        "previous_end": previous[1].get("end") if previous else None,
        "start": str(start),
        "end": str(end),
        "note": "Old generation and all raw bars are retained; removed means outside this union.",
    }
    atomic_json(folder / "comparison.json", result)
    return result


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
    populations = manifest.daily_populations()
    if not populations and any((config.data.root / "raw/assets").glob("*.parquet")):
        # Older downloads have no saved scope. Default to the broad asset population;
        # rerunning an explicit --symbols ingestion records an intentional smaller scope.
        populations = [(lower, upper, asset_symbols(config.data.root))]
    if populations:
        lower = min(lower, *(requested_start for requested_start, _, _ in populations))
        upper = max(upper, *(requested_end for _, requested_end, _ in populations))
    start, end = start or lower, end or upper
    validate_range(start, end)
    coverage = manifest.coverage("daily_bars")
    incomplete = {
        symbol for symbol, intervals in coverage.items() if missing_intervals(start, end, intervals)
    }
    for requested_start, requested_end, symbols in populations:
        required_start, required_end = max(start, requested_start), min(end, requested_end)
        if required_start < required_end:
            incomplete.update(
                symbol
                for symbol in symbols
                if missing_intervals(required_start, required_end, coverage.get(symbol, []))
            )
    if incomplete:
        raise ValueError(
            f"Daily coverage is incomplete for {len(incomplete)} symbols "
            f"({', '.join(sorted(incomplete)[:5])}) over "
            f"[{start}, {end}); run daily ingestion or bootstrap for this range first"
        )
    signature = {
        "schema_version": 2,
        "algorithm": UNIVERSE_ALGORITHM,
        "calendar": "XNYS",
        "calendar_version": version("exchange-calendars"),
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
        sessions = session_dates(start, end)
        db.register("trading_sessions", sessions.to_arrow())
        query = prior_liquidity_sql(
            options.lookback_days, options.min_history_days, options.max_staleness_days
        )
        db.sql(
            query,
            params={
                "paths": [str(path) for path in files],
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
    comparison = compare_universes(current, staging, start, end)
    meta = {
        **signature,
        "generation": generation,
        "symbol_count": count,
        "built_at": datetime.now(UTC).isoformat(),
        "criterion": f"ever top {options.ingestion_rank} by prior "
        f"{options.lookback_days}-observation median dollar volume; "
        f"last observation <= {options.max_staleness_days} calendar days old; "
        "no current-day bar requirement",
        "session_count": sessions.height,
        "previous_generation": comparison["previous_generation"],
    }
    atomic_json(staging / "metadata.json", meta)
    os.replace(staging, final)
    # One pointer swaps both artifacts together; interruptions preserve the previous generation.
    atomic_json(base / "current.json", meta)
    log.info("Universe: %s symbols; %s", count, meta["criterion"])
    return final, meta


def universe_symbols(folder: Path) -> list[str]:
    return pl.read_parquet(folder / "symbols.parquet", columns=["symbol"])["symbol"].to_list()
