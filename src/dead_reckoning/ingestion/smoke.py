"""Bounded live verification using the production ingestion and storage paths."""

from datetime import date

import polars as pl

from dead_reckoning.config import Config
from dead_reckoning.ingestion.assets import asset_symbols, ingest_assets
from dead_reckoning.ingestion.bars import ingest_bars
from dead_reckoning.ingestion.manifest import Manifest, missing_intervals
from dead_reckoning.market_data.alpaca import AlpacaClient, session_bounds
from dead_reckoning.universe.liquidity import build_universe, universe_symbols

SMOKE_SYMBOLS = ["AAPL", "MSFT"]
SMOKE_START = date(2025, 1, 2)
SMOKE_END = date(2025, 2, 8)


def _check_bars(manifest: Manifest, dataset: str, symbols: list[str]) -> int:
    files = manifest.files(dataset)
    if not files:
        raise RuntimeError(f"Smoke check failed: no {dataset} files were stored")
    stats = (
        pl.scan_parquet(files, hive_partitioning=False)
        .group_by("symbol")
        .agg(
            pl.len().alias("rows"),
            pl.col("timestamp").n_unique().alias("unique_rows"),
            pl.col("timestamp").min().alias("first"),
            pl.col("timestamp").max().alias("last"),
        )
        .collect()
    )
    if set(stats["symbol"].to_list()) != set(symbols):
        raise RuntimeError(f"Smoke check failed: {dataset} has missing or unexpected symbols")
    coverage = manifest.coverage(dataset)
    lower, upper = session_bounds(SMOKE_START, SMOKE_END)
    for row in stats.iter_rows(named=True):
        if row["rows"] != row["unique_rows"]:
            raise RuntimeError(f"Smoke check failed: duplicate {dataset} timestamps")
        if not lower <= row["first"] <= row["last"] < upper:
            raise RuntimeError(f"Smoke check failed: {dataset} timestamps outside requested range")
        if missing_intervals(SMOKE_START, SMOKE_END, coverage.get(row["symbol"], [])):
            raise RuntimeError(f"Smoke check failed: incomplete {dataset} coverage")
    rows = stats["rows"].sum()
    if rows != manifest.summary(dataset)[1]:
        raise RuntimeError(f"Smoke check failed: {dataset} rows disagree with manifest")
    return rows


def run_smoke(config: Config, manifest: Manifest, client: AlpacaClient) -> dict[str, int]:
    """Download at most two symbols' bars, validate storage, then verify a no-op rerun."""
    ingest_assets(config.data.root, client)
    assets = asset_symbols(config.data.root)
    if not set(SMOKE_SYMBOLS).issubset(assets):
        raise RuntimeError("Smoke check failed: requested symbols are missing from asset master")

    ingest_bars(config, manifest, client, "daily_bars", SMOKE_SYMBOLS, SMOKE_START, SMOKE_END)
    daily_rows = _check_bars(manifest, "daily_bars", SMOKE_SYMBOLS)
    folder, _ = build_universe(config, manifest, SMOKE_START, SMOKE_END)
    selected = universe_symbols(folder)
    if not selected or not set(selected).issubset(SMOKE_SYMBOLS):
        raise RuntimeError("Smoke check failed: empty or unexpected historical universe")
    ingest_bars(config, manifest, client, "minute_bars", selected, SMOKE_START, SMOKE_END)
    minute_rows = _check_bars(manifest, "minute_bars", selected)
    fingerprints = {
        dataset: manifest.fingerprint(dataset) for dataset in ("daily_bars", "minute_bars")
    }

    # Same production calls, same dates: the second pass must do no bar ingestion.
    ingest_assets(config.data.root, client)
    daily_rerun = ingest_bars(
        config, manifest, client, "daily_bars", SMOKE_SYMBOLS, SMOKE_START, SMOKE_END
    )
    rerun_folder, _ = build_universe(config, manifest, SMOKE_START, SMOKE_END)
    minute_rerun = ingest_bars(
        config, manifest, client, "minute_bars", selected, SMOKE_START, SMOKE_END
    )
    if any(result["batches"] or result["rows"] for result in (daily_rerun, minute_rerun)):
        raise RuntimeError("Smoke check failed: rerun downloaded already-complete bars")
    if rerun_folder != folder or any(
        manifest.fingerprint(dataset) != fingerprint
        for dataset, fingerprint in fingerprints.items()
    ):
        raise RuntimeError("Smoke check failed: rerun changed completed data or universe")
    return {
        "assets": len(assets),
        "daily_rows": daily_rows,
        "minute_rows": minute_rows,
        "universe_symbols": len(selected),
    }
