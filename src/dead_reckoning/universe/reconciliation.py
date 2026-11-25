"""Compare immutable universe generations and fill only missing request coverage."""

import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from dead_reckoning.config import Config
from dead_reckoning.ingestion.bars import ingest_bars
from dead_reckoning.ingestion.manifest import Manifest, missing_intervals
from dead_reckoning.storage.parquet import atomic_json
from dead_reckoning.universe.liquidity import universe_symbols

log = logging.getLogger(__name__)


def coverage_plan(config: Config, manifest: Manifest, folder: Path, meta: dict):
    start, end = date.fromisoformat(meta["start"]), date.fromisoformat(meta["end"])
    coverage = manifest.coverage("minute_bars")
    comparison = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
    added = set(comparison["new_symbols"])
    rows = []
    for symbol in universe_symbols(folder):
        missing = missing_intervals(start, end, coverage.get(symbol, []))
        rows.append(
            {
                "symbol": symbol,
                "newly_included": symbol in added,
                "coverage_complete": not missing,
                "missing_interval_count": len(missing),
                "missing_intervals": json.dumps([[str(a), str(b)] for a, b in missing]),
            }
        )
    frame = pl.DataFrame(
        rows,
        schema={
            "symbol": pl.String,
            "newly_included": pl.Boolean,
            "coverage_complete": pl.Boolean,
            "missing_interval_count": pl.Int64,
            "missing_intervals": pl.String,
        },
    )
    return frame, {
        "required_symbols": frame.height,
        "complete_symbols": frame.filter(pl.col("coverage_complete")).height,
        "incomplete_symbols": frame.filter(~pl.col("coverage_complete")).height,
        "newly_included_symbols": frame.filter(pl.col("newly_included")).height,
        "newly_included_complete": frame.filter(
            pl.col("newly_included") & pl.col("coverage_complete")
        ).height,
        "newly_included_incomplete": frame.filter(
            pl.col("newly_included") & ~pl.col("coverage_complete")
        ).height,
        "missing_intervals": int(frame["missing_interval_count"].sum() or 0),
    }


def reconcile_minutes(
    config: Config, manifest: Manifest, folder: Path, meta: dict, *, client=None
) -> dict:
    """No client means report only. An authorized caller supplies the existing client.

    The pre-download plan is saved before the first request. Interrupted runs use
    the same generation and replan from durable manifest coverage on the next run.
    This never deletes symbols or raw files, nor force-refreshes complete intervals.
    """
    frame, before = coverage_plan(config, manifest, folder, meta)
    frame.write_csv(folder / "minute_coverage.csv")
    comparison = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
    result = {
        "generation": meta["generation"],
        "comparison": comparison,
        "start": meta["start"],
        "end": meta["end"],
        "coverage_before": before,
        "coverage_after": before,
        "download": None,
        "note": "Coverage means completed SIP/raw request intervals, including empty responses. "
        "It does not guarantee one bar per scheduled minute.",
        "updated_at": datetime.now(UTC).isoformat(),
    }
    atomic_json(folder / "reconciliation.json", result)
    log.info("Universe reconciliation: %s", comparison)
    log.info("Minute coverage before: %s", before)
    pending = frame.filter(~pl.col("coverage_complete"))["symbol"].to_list()
    if pending and client is not None:
        result["download"] = ingest_bars(
            config,
            manifest,
            client,
            "minute_bars",
            pending,
            date.fromisoformat(meta["start"]),
            date.fromisoformat(meta["end"]),
        )
        frame, after = coverage_plan(config, manifest, folder, meta)
        frame.write_csv(folder / "minute_coverage.csv")
        result["coverage_after"] = after
        result["updated_at"] = datetime.now(UTC).isoformat()
        atomic_json(folder / "reconciliation.json", result)
        log.info("Minute coverage after: %s", after)
        if after["incomplete_symbols"]:
            raise RuntimeError("Minute ingestion returned with required intervals still incomplete")
    return result
