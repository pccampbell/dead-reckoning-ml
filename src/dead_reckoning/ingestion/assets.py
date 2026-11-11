import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import polars as pl

from dead_reckoning.market_data.alpaca import AlpacaClient
from dead_reckoning.storage.parquet import read_metadata, write_pages

log = logging.getLogger(__name__)
ASSET_SCHEMA = {
    "id": pl.String,
    "symbol": pl.String,
    "name": pl.String,
    "class": pl.String,
    "exchange": pl.String,
    "status": pl.String,
    "tradable": pl.Boolean,
    "marginable": pl.Boolean,
    "shortable": pl.Boolean,
    "easy_to_borrow": pl.Boolean,
    "fractionable": pl.Boolean,
}


def ingest_assets(root: Path, client: AlpacaClient, refresh: bool = False) -> None:
    folder = root / "raw/assets"
    snapshots = sorted(folder.glob("*.parquet"))
    if snapshots and not refresh:
        latest = datetime.fromisoformat(read_metadata(snapshots[-1])["ingested_at"])
        ny = ZoneInfo("America/New_York")
        if latest.astimezone(ny).date() == datetime.now(ny).date():
            log.info("Assets: today's snapshot skipped (use --refresh to update)")
            return
    records = client.assets()
    frame = (
        pl.from_dicts(
            [{field: asset.get(field) for field in ASSET_SCHEMA} for asset in records],
            schema=ASSET_SCHEMA,
        )
        .unique("id")
        .sort("symbol")
    )
    if frame["symbol"].null_count() or frame["id"].null_count():
        raise ValueError("Asset master contains missing symbols or identifiers")
    now = datetime.now(UTC)
    path = folder / f"{now:%Y%m%dT%H%M%S}-{uuid4().hex[:8]}.parquet"
    write_pages(
        path,
        [frame],
        {
            "schema_version": 1,
            "provider": "alpaca",
            "dataset": "assets",
            "status_filter": "all",
            "asset_class": "us_equity",
            "ingested_at": now.isoformat(),
        },
    )
    log.info("Assets: stored %s active/inactive reference records", frame.height)


def asset_symbols(root: Path) -> list[str]:
    files = sorted((root / "raw/assets").glob("*.parquet"))
    if not files:
        raise ValueError("No asset snapshot; run 'ingest assets' first")
    # Union snapshots so refreshing the asset master never silently loses old symbols.
    return (
        pl.scan_parquet(files)
        .select("symbol")
        .unique()
        .sort("symbol")
        .collect()["symbol"]
        .to_list()
    )
