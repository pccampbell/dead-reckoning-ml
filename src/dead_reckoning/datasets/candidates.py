"""Policy-independent scheduled decisions from the corrected prior-only universe."""

import logging
from datetime import date
from importlib.metadata import version
from time import monotonic

import polars as pl

from dead_reckoning.config import validate_range
from dead_reckoning.datasets.artifacts import (
    artifact_path,
    completed,
    connection,
    digest,
    finish_run,
    implementation,
    open_artifact,
    plan_partitions,
    sessions,
    write_partition,
    writer_lock,
)
from dead_reckoning.ingestion.manifest import Manifest
from dead_reckoning.storage.history import minute_inventory, minute_paths
from dead_reckoning.universe.liquidity import UNIVERSE_ALGORITHM, current_universe

log = logging.getLogger(__name__)
UTC_US = pl.Datetime("us", "UTC")


def candidate_definition(config):
    return {
        "kind": "candidates",
        "schema_version": 1,
        "configuration": config.dataset.candidates.model_dump(mode="json"),
        "universe": config.universe.model_dump(exclude={"ingestion_rank"}),
        "universe_algorithm": UNIVERSE_ALGORITHM,
        "calendar": "XNYS",
        "calendar_version": version("exchange-calendars"),
        "versions": {name: version(name) for name in ("duckdb", "polars")},
        "data_conventions": {
            "feed": config.data.feed,
            "adjustment": config.data.adjustment,
            "provider": config.data.provider,
            "asof": "-",
        },
        "implementation": implementation(
            "datasets/candidates.py",
            "datasets/artifacts.py",
            "storage/history.py",
            "universe/liquidity.py",
        ),
    }


def make_candidates(members, session, options):
    """Every regular-session decision remains present, even if no price bar exists."""
    minutes = int((session["close"] - session["open"]).total_seconds() / 60)
    grid = pl.DataFrame(
        {
            "decision_minute": pl.Series(
                range(0, minutes, options.decision_minutes),
                dtype=pl.UInt16,
            )
        }
    )
    result = (
        members.select(
            "symbol",
            pl.col("liquidity_rank").cast(pl.UInt32).alias("universe_rank"),
            pl.col("trailing_median_dollar_volume").cast(pl.Float64).alias("prior_liquidity"),
            pl.col("feature_date").alias("universe_feature_date"),
        )
        .join(grid, how="cross")
        .with_columns(
            pl.lit(session["session_date"], dtype=pl.Date).alias("session_date"),
            pl.lit(session["open"], dtype=UTC_US).alias("session_open_utc"),
            pl.lit(session["close"], dtype=UTC_US).alias("session_close_utc"),
            (
                pl.lit(session["open"], dtype=UTC_US)
                + pl.duration(minutes=pl.col("decision_minute"))
            ).alias("decision_timestamp_utc"),
        )
        .sort(["symbol", "decision_timestamp_utc"])
    )
    if result.filter(pl.col("universe_feature_date") >= pl.col("session_date")).height:
        raise ValueError("Universe uses current/future daily information")
    return result


def checked_universe(config, start, end):
    current = current_universe(config.data.root)
    if current is None:
        raise ValueError("Corrected universe is missing; run 'universe build' first")
    folder, meta = current
    expected = {
        **config.universe.model_dump(exclude={"ingestion_rank"}),
        "algorithm": UNIVERSE_ALGORITHM,
        "calendar_version": version("exchange-calendars"),
    }
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("Corrected universe definition is stale; run 'universe build' first")
    if date.fromisoformat(meta["start"]) > start or date.fromisoformat(meta["end"]) < end:
        raise ValueError("Corrected universe does not cover this range; run 'universe build'")
    with Manifest(config.data.root) as manifest:
        if meta["daily_fingerprint"] != manifest.fingerprint("daily_bars"):
            raise ValueError("Daily sources changed; run 'universe build' before candidates")
    return folder, meta


def build_candidates(config, start=None, end=None, output_id=None, rebuild=False):
    started = monotonic()
    start, end = start or config.dataset.start, end or config.dataset.end
    validate_range(start, end)
    schedule = sessions(start, end)
    if not schedule:
        raise ValueError("No regular trading sessions in requested range")
    source, source_meta = checked_universe(config, start, end)
    coverage = minute_inventory(config.data.root, start, end)
    definition = candidate_definition(config)
    output_id = output_id or f"candidates_{digest(definition)[:16]}"
    built = skipped = 0
    with writer_lock(artifact_path(config.data.root, "candidates", output_id)):
        folder, meta = open_artifact(
            config.data.root, "candidates", "candidates", definition, output_id
        )
        plan_partitions(folder, meta, [s["session_date"] for s in schedule])
        with connection(config, config.data.root) as db:
            members = db.sql(
                """SELECT * FROM read_parquet($path) WHERE date >= $start AND date < $end
                    AND liquidity_rank <= $size ORDER BY date, liquidity_rank""",
                params={
                    "path": str(source / "daily_liquidity.parquet"),
                    "start": start,
                    "end": end,
                    "size": config.dataset.candidates.universe_size,
                },
            ).pl()
            for session in schedule:
                day = session["session_date"]
                selected = members.filter(pl.col("date") == day)
                if selected.is_empty():
                    raise ValueError(
                        f"No eligible universe on {day}; no candidates silently omitted"
                    )
                # Request completion gates the build, never which individual rows qualify.
                minute_paths(coverage, config.data.root, day, selected["symbol"].to_list())
                dependency = {
                    "prior_members": digest(
                        selected.select(
                            "symbol",
                            "liquidity_rank",
                            "feature_date",
                            "trailing_median_dollar_volume",
                        )
                        .sort("symbol")
                        .to_dicts()
                    ),
                    "session_open": str(session["open"]),
                    "session_close": str(session["close"]),
                }
                if completed(folder, meta, day, dependency, rebuild):
                    skipped += 1
                    continue
                frame = make_candidates(selected, session, config.dataset.candidates)
                write_partition(
                    folder,
                    meta,
                    day,
                    frame,
                    dependency,
                    {
                        "symbols": selected.height,
                        "universe_generation": source_meta["generation"],
                        "decisions_per_symbol": frame.height // selected.height,
                        "minute_request_coverage": "complete; actual bars never gate candidates",
                    },
                )
                built += 1
                log.info("Candidates %s: %s rows", day, frame.height)
        return finish_run(folder, meta, start, end, built, skipped, started, config)
