import json
import os
from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

import polars as pl
import pyarrow.parquet as pq

METADATA_KEY = b"dead_reckoning"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    try:
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_pages(path: Path, pages: Iterable[pl.DataFrame], metadata: dict) -> dict[str, int]:
    """Stream pages into one atomic file; require sorted keys across pages.

    A failure leaves no final file. Per-symbol last keys prevent duplicates even
    across page boundaries without holding an entire request in memory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    counts = dict.fromkeys(metadata.get("symbols", []), 0)
    last_keys = {}
    writer = None
    try:
        for frame in pages:
            if "timestamp" in frame.columns:
                frame = frame.unique(["symbol", "timestamp"], keep="last").sort(
                    ["symbol", "timestamp"]
                )
                # One operation per symbol/page, never per historical row.
                kept = []
                for group in frame.partition_by("symbol", maintain_order=True):
                    symbol = group["symbol"][0]
                    if symbol in last_keys:
                        if group["timestamp"].min() < last_keys[symbol]:
                            raise ValueError("Bar pages are out of order")
                        group = group.filter(pl.col("timestamp") > last_keys[symbol])
                    if group.height:
                        last_keys[symbol] = group["timestamp"][-1]
                        counts[symbol] = counts.get(symbol, 0) + group.height
                        kept.append(group)
                frame = pl.concat(kept) if kept else frame.head(0)
            table = frame.to_arrow()
            if writer is None:
                schema = table.schema.with_metadata({METADATA_KEY: json.dumps(metadata).encode()})
                writer = pq.ParquetWriter(temp, schema, compression="zstd")
            writer.write_table(table, row_group_size=100_000)
        if writer is None:
            raise ValueError("No pages supplied, refusing to record coverage")
        writer.close()
        writer = None
        # Windows requires a writable descriptor for FlushFileBuffers/os.fsync.
        with temp.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temp, path)
        return counts
    finally:
        if writer:
            writer.close()
        temp.unlink(missing_ok=True)


def read_metadata(path: Path) -> dict:
    return json.loads(pq.read_metadata(path).metadata[METADATA_KEY])
