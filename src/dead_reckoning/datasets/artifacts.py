"""Small JSON manifests and atomic daily Parquet partitions; one writer per artifact."""

import hashlib
import json
import re
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from time import monotonic

import duckdb
import exchange_calendars as xcals
import polars as pl

from dead_reckoning.storage.parquet import atomic_json, write_pages

KEYS = ["symbol", "decision_timestamp_utc"]
SCHEMA_VERSION = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def implementation(*relative_paths):
    base = Path(__file__).parents[1]
    return {name: file_hash(base / name) for name in relative_paths}


def utc_now():
    return datetime.now(UTC).isoformat()


def peak_memory_mb():
    try:
        import psutil
    except ImportError:
        return None
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss) / 1024**2


def sessions(start, end):
    calendar = xcals.get_calendar("XNYS", start=str(start), end=str(end))
    return [
        {
            "session_date": stamp.date(),
            "open": row["open"].to_pydatetime(),
            "close": row["close"].to_pydatetime(),
        }
        for stamp, row in calendar.schedule.iterrows()
        if start <= stamp.date() < end
    ]


def connection(config, root):
    return duckdb.connect(
        config={
            "memory_limit": config.query.memory_limit,
            "threads": config.query.threads,
            "temp_directory": str(root / "metadata/dataset_query_tmp"),
            "preserve_insertion_order": False,
        }
    )


def artifact_path(root, stage, identifier):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", identifier):
        raise ValueError("Artifact identifiers must contain only letters, digits, '_' or '-'")
    return root / "datasets" / stage / identifier


@contextmanager
def writer_lock(folder):
    """DuckDB supplies a Windows-compatible OS lock; process death releases it."""
    folder.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(folder / "writer.duckdb")):
        yield


def open_artifact(root, stage, prefix, definition, identifier=None):
    fingerprint = digest(definition)
    identifier = identifier or f"{prefix}_{fingerprint[:16]}"
    folder = artifact_path(root, stage, identifier)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "metadata.json"
    if path.exists():
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("definition_fingerprint") != fingerprint:
            raise ValueError(f"Incompatible definition for {identifier}; use a new artifact ID")
    else:
        meta = {
            "artifact_id": identifier,
            "artifact_version": 1,
            "stage": stage,
            "schema_version": SCHEMA_VERSION,
            "definition": definition,
            "definition_fingerprint": fingerprint,
            "created_at": utc_now(),
            "keys": KEYS,
            "partitions": {},
            "runs": [],
        }
        atomic_json(path, meta)
    return folder, meta


def read_artifact(root, stage, identifier):
    folder = artifact_path(root, stage, identifier)
    path = folder / "metadata.json"
    if not path.exists():
        raise ValueError(f"Artifact {identifier} does not exist; generate its stage first")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("schema_version") != SCHEMA_VERSION or meta.get("stage") != stage:
        raise ValueError(f"Unsupported artifact schema: {identifier}")
    if digest(meta["definition"]) != meta.get("definition_fingerprint"):
        raise ValueError(f"Artifact definition fingerprint changed: {identifier}")
    return folder, meta


def partition_path(folder, day):
    return folder / f"session_date={day}" / "part.parquet"


def completed(folder, meta, day, dependency=None, rebuild=False):
    part = meta["partitions"].get(str(day), {})
    path = partition_path(folder, day)
    if rebuild or part.get("status") != "complete" or not path.is_file():
        return False
    if file_hash(path) != part["sha256"]:
        raise ValueError(f"Partition content changed for {day}; explicitly rebuild it")
    if dependency is not None and part["dependency"] != dependency:
        raise ValueError(f"Stale dependency for {day}; explicitly rebuild this partition")
    return True


def write_partition(folder, meta, day, frame, dependency, summary):
    if frame.select(pl.struct(KEYS).n_unique()).item() != frame.height:
        raise ValueError("Observation keys are not unique")
    if frame.select(pl.any_horizontal(pl.col(KEYS).is_null()).any()).item():
        raise ValueError("Observation keys cannot be null")
    schema = {name: str(dtype) for name, dtype in frame.schema.items()}
    if "schema" in meta and meta["schema"] != schema:
        raise ValueError("Partition schema differs from artifact schema")
    meta["schema"] = schema
    meta["partitions"][str(day)] = {"status": "building", "dependency": dependency}
    atomic_json(folder / "metadata.json", meta)
    path = partition_path(folder, day)
    write_pages(
        path,
        [frame.sort(KEYS)],
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_id": meta["artifact_id"],
            "definition_fingerprint": meta["definition_fingerprint"],
            "session_date": str(day),
            "dependency": dependency,
        },
    )
    meta["partitions"][str(day)] = {
        "status": "complete",
        "path": str(path.relative_to(folder)),
        "dependency": dependency,
        "rows": frame.height,
        "bytes": path.stat().st_size,
        "sha256": file_hash(path),
        "completed_at": utc_now(),
        "summary": summary,
    }
    meta["updated_at"] = utc_now()
    atomic_json(folder / "metadata.json", meta)


def plan_partitions(folder, meta, days):
    for day in days:
        meta["partitions"].setdefault(str(day), {"status": "pending"})
    atomic_json(folder / "metadata.json", meta)


def finish_run(folder, meta, start, end, built, skipped, started, config):
    parts = [p for p in meta["partitions"].values() if p["status"] == "complete"]
    days = sorted(k for k, p in meta["partitions"].items() if p["status"] == "complete")
    meta["coverage"] = {
        "start": days[0] if days else None,
        "end": str(date.fromisoformat(days[-1]) + timedelta(days=1)) if days else None,
        "complete_partitions": len(parts),
        "rows": sum(p["rows"] for p in parts),
        "parquet_bytes": sum(p["bytes"] for p in parts),
    }
    result = {
        "artifact_id": meta["artifact_id"],
        "start": str(start),
        "end": str(end),
        "built_partitions": built,
        "skipped_partitions": skipped,
        "elapsed_seconds": monotonic() - started,
        "peak_process_mb": peak_memory_mb(),
        "generation_timestamp": utc_now(),
        "query": config.query.model_dump(),
        "versions": {name: version(name) for name in ("numpy", "duckdb", "polars")},
        **meta["coverage"],
    }
    # Coverage can be non-contiguous; per-day states remain authoritative.
    result["requested_start"], result["requested_end"] = str(start), str(end)
    meta["runs"].append(result)
    atomic_json(folder / "metadata.json", meta)
    return result


def dataset_status(root):
    results = []
    for path in sorted((root / "datasets").glob("*/*/metadata.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        states = [p["status"] for p in meta["partitions"].values()]
        results.append(
            {
                "artifact_id": meta["artifact_id"],
                "stage": meta["stage"],
                "kind": meta["definition"]["kind"],
                "complete": states.count("complete"),
                "pending": len(states) - states.count("complete"),
                "missing_files": sum(
                    not (path.parent / p["path"]).is_file()
                    for p in meta["partitions"].values()
                    if p["status"] == "complete"
                ),
                "coverage": meta.get("coverage"),
            }
        )
    return results
