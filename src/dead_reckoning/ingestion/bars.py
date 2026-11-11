import hashlib
import json
import logging
from collections import defaultdict
from datetime import UTC, date, datetime

from dead_reckoning.config import Config, validate_range
from dead_reckoning.ingestion.manifest import Manifest, missing_intervals
from dead_reckoning.market_data.alpaca import AlpacaClient
from dead_reckoning.storage.parquet import write_pages

log = logging.getLogger(__name__)


def windows(start: date, end: date, dataset: str):
    """Yearly daily requests, monthly minute requests; clip to exact missing range."""
    while start < end:
        if dataset == "daily_bars":
            boundary = date(start.year + 1, 1, 1)
        else:
            boundary = date(start.year + start.month // 12, start.month % 12 + 1, 1)
        upper = min(end, boundary)
        yield start, upper
        start = upper


def ingest_bars(
    config: Config,
    manifest: Manifest,
    client: AlpacaClient,
    dataset: str,
    symbols: list[str],
    start: date,
    end: date,
) -> dict[str, int]:
    validate_range(start, end)
    if dataset not in {"daily_bars", "minute_bars"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    timeframe = "1Day" if dataset == "daily_bars" else "1Min"
    batch_size = (
        config.ingestion.daily_batch_size
        if timeframe == "1Day"
        else config.ingestion.minute_batch_size
    )
    coverage = manifest.coverage(dataset)
    groups = defaultdict(list)
    skipped = 0
    for symbol in sorted(set(symbols)):
        missing = missing_intervals(start, end, coverage.get(symbol, []))
        if not missing:
            skipped += 1
        for lower, upper in missing:
            for window in windows(lower, upper, dataset):
                groups[window].append(symbol)
    requests = sum((len(group) + batch_size - 1) // batch_size for group in groups.values())
    log.info(
        "%s: %s symbols already covered; %s request batches pending", dataset, skipped, requests
    )
    rows = 0
    for (lower, upper), group in sorted(groups.items()):
        for offset in range(0, len(group), batch_size):
            batch = group[offset : offset + batch_size]
            meta = {
                "schema_version": 1,
                "provider": "alpaca",
                "dataset": dataset,
                "timeframe": timeframe,
                "feed": config.data.feed,
                "adjustment": config.data.adjustment,
                "asof": "-",
                "symbols": batch,
                "start": lower.isoformat(),
                "end": upper.isoformat(),
                "date_convention": "New York session dates; end exclusive",
            }
            key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:24]
            path = config.data.root / "raw" / dataset / f"year={lower.year}" / f"{key}.parquet"
            meta["ingested_at"] = datetime.now(UTC).isoformat()
            manifest.record(path, meta, "running", {})
            try:
                counts = write_pages(path, client.bar_pages(batch, timeframe, lower, upper), meta)
                manifest.record(path, meta, "complete", counts)
            except BaseException:
                # If rename succeeded, reconciliation will recover the final file on restart.
                manifest.record(path, meta, "failed", {})
                raise
            rows += sum(counts.values())
            log.info(
                "%s: %s to %s, %s symbols, %s rows stored",
                dataset,
                lower,
                upper,
                len(batch),
                sum(counts.values()),
            )
    return {"batches": requests, "rows": rows, "skipped_symbols": skipped}
