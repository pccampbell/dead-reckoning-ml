"""Calendar sampling, strictly prior daily eligibility, and bounded minute retrieval."""

import logging
import random
from pathlib import Path

import duckdb
import exchange_calendars as xcals
import numpy as np
import polars as pl

from dead_reckoning.config import Config, LabelStudyConfig
from dead_reckoning.storage.history import inventory, minute_paths, read_minutes  # noqa: F401

log = logging.getLogger(__name__)


def sampled_sessions(options: LabelStudyConfig) -> pl.DataFrame:
    calendar = xcals.get_calendar("XNYS", start=str(options.start), end=str(options.end))
    schedule = calendar.schedule
    groups = {}
    for stamp, row in schedule.iterrows():
        day = stamp.date()
        if options.start <= day < options.end:
            groups.setdefault((day.year, day.month), []).append(
                {
                    "date": day,
                    "open": row["open"].to_pydatetime(),
                    "close": row["close"].to_pydatetime(),
                }
            )
    rng = random.Random(options.seed)
    selected = []
    for days in groups.values():
        selected.extend(rng.sample(days, min(len(days), options.sessions_per_month)))
    selected.sort(key=lambda row: row["date"])
    if options.max_sessions and len(selected) > options.max_sessions:
        # Deterministic spread through the calendar, retaining multi-year coverage.
        selected = [
            selected[i] for i in np.linspace(0, len(selected) - 1, options.max_sessions, dtype=int)
        ]
    if not selected:
        raise ValueError("No valid XNYS sessions in the research range")
    return pl.from_dicts(selected)


def decision_grid(minutes: int, options: LabelStudyConfig) -> tuple[np.ndarray, np.ndarray]:
    decisions = np.arange(0, minutes, options.decision_minutes)
    entries = decisions + options.execution_delay
    keep = entries + max(options.horizons) <= minutes
    return decisions[keep], entries[keep]


def query_connection(config: Config, temp: Path):
    return duckdb.connect(
        config={
            "threads": config.research.threads,
            "memory_limit": config.research.memory_limit,
            "temp_directory": str(temp),
            "preserve_insertion_order": False,
        }
    )


def prior_universe(db, paths: list[str], sessions: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Rank the full daily population without requiring a bar on the decision date.

    Last 20 observed dollar volumes and last 20 log close returns strictly before
    each session. At most seven calendar days since the last prior daily bar.
    Current-day prices, existence, returns, and volatility never gate eligibility.
    """
    db.register("sample_dates", sessions.select("date").to_arrow())
    if config.universe.lookback_days != 20 or config.universe.min_history_days != 20:
        raise ValueError("Label study requires a prior 20-observation universe configuration")
    log.info("Research: deriving prior daily features from %s files", len(paths))
    db.execute(
        """
        CREATE TEMP TABLE daily_features AS
        WITH daily AS (
            SELECT symbol, CAST(timezone('America/New_York', timestamp) AS DATE) AS date,
                   close, close * volume AS dv
            FROM read_parquet($paths, hive_partitioning=false)
            WHERE timestamp < $end_boundary
            QUALIFY row_number() OVER (PARTITION BY symbol, timestamp ORDER BY close, volume) = 1
        ), returns AS (
            SELECT *, CASE WHEN close > 0 AND isfinite(close) THEN
                ln(close / nullif(lag(close) OVER (PARTITION BY symbol ORDER BY date), 0))
                END AS log_return
            FROM daily WHERE close > 0 AND isfinite(close) AND dv >= 0 AND isfinite(dv)
        )
        SELECT symbol, date, median(dv) OVER w AS prior_dollar_volume,
               count(*) OVER w AS observations,
               stddev_samp(log_return) OVER w AS prior_volatility,
               count(log_return) OVER w AS return_observations
        FROM returns WINDOW w AS (
            PARTITION BY symbol ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        )
    """,
        {
            "paths": paths,
            "end_boundary": sessions["close"].max(),
        },
    )
    # ASOF matches the last strictly earlier observed date, including absent-on-day symbols.
    frame = db.sql(
        """
        WITH symbols AS (SELECT DISTINCT symbol FROM daily_features), prior AS (
            SELECT d.date, s.symbol, f.date AS feature_date, f.prior_dollar_volume,
                   f.prior_volatility, f.observations, f.return_observations
            FROM (sample_dates d CROSS JOIN symbols s)
            ASOF LEFT JOIN daily_features f ON s.symbol=f.symbol AND d.date > f.date
        ), ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY date ORDER BY prior_dollar_volume DESC, symbol
            ) AS liquidity_rank
            FROM prior WHERE observations=20 AND prior_dollar_volume > 0
                AND date-feature_date <= $stale
        )
        SELECT * FROM ranked WHERE liquidity_rank <= $size ORDER BY date, liquidity_rank
    """,
        params={
            "stale": config.research.max_staleness_days,
            "size": config.research.universe_size,
        },
    ).pl()
    db.execute("""
        CREATE OR REPLACE TEMP TABLE daily_support_quality AS
        SELECT count(*) AS supporting_daily_observations, count(DISTINCT symbol) AS daily_symbols,
               min(date)::VARCHAR AS first_daily_date, max(date)::VARCHAR AS last_daily_date
        FROM daily_features
    """)
    db.execute("DROP TABLE daily_features")
    if frame.is_empty():
        raise ValueError("No eligible prior-history universe; daily data is unavailable")
    # Daily cross-sectional terciles, using only prior features; unknown gets its own bucket.
    frame = (
        frame.with_columns(
            pl.col("prior_volatility").rank("ordinal").over("date").alias("vol_rank"),
            pl.len().over("date").alias("universe_count"),
        )
        .with_columns(
            pl.when(pl.col("return_observations") < 20)
            .then(pl.lit("unknown"))
            .when(pl.col("vol_rank") <= pl.col("universe_count") / 3)
            .then(pl.lit("low"))
            .when(pl.col("vol_rank") <= 2 * pl.col("universe_count") / 3)
            .then(pl.lit("mid"))
            .otherwise(pl.lit("high"))
            .alias("volatility_bucket"),
            pl.when(pl.col("liquidity_rank") <= pl.col("universe_count") / 3)
            .then(pl.lit("top"))
            .when(pl.col("liquidity_rank") <= 2 * pl.col("universe_count") / 3)
            .then(pl.lit("mid"))
            .otherwise(pl.lit("lower"))
            .alias("liquidity_bucket"),
        )
        .drop("vol_rank")
    )
    assert frame.filter(pl.col("feature_date") >= pl.col("date")).is_empty()
    return frame
