"""Generate independent targets from candidate partitions with shared bounded extraction."""

import logging
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path
from time import monotonic

import numpy as np
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
    partition_path,
    plan_partitions,
    read_artifact,
    sessions,
    write_partition,
    writer_lock,
)
from dead_reckoning.datasets.targets import barrier_targets, fixed_targets, target_summary
from dead_reckoning.labels.paths import price_grid, scan_paths
from dead_reckoning.storage.history import minute_inventory, minute_paths, read_minutes
from dead_reckoning.storage.parquet import read_metadata

log = logging.getLogger(__name__)


def label_definition(candidate_meta, policy=None, horizons=None):
    return {
        "kind": "triple_barrier" if policy else "fixed_horizon",
        "schema_version": 1,
        "configuration": policy.model_dump(mode="json") if policy else {"horizons": horizons},
        "source_candidate_id": candidate_meta["artifact_id"],
        "source_candidate_definition": candidate_meta["definition_fingerprint"],
        "execution_delay": candidate_meta["definition"]["configuration"]["execution_delay"],
        "implementation": implementation(
            "labels/barriers.py",
            "labels/paths.py",
            "datasets/targets.py",
            "datasets/labels.py",
            "datasets/artifacts.py",
            "storage/history.py",
        ),
        "versions": {n: version(n) for n in ("numpy", "duckdb", "polars")},
    }


def raw_dependency(coverage, root, day, symbols, paths, cache):
    rows = coverage.filter(
        (pl.col("start_date") <= day) & (pl.col("end_date") > day) & pl.col("symbol").is_in(symbols)
    ).sort(["path", "symbol"])
    files = []
    for path in paths:
        if path not in cache:
            info = read_metadata(path)
            if any(
                info.get(k) != v
                for k, v in {
                    "feed": "sip",
                    "adjustment": "raw",
                    "asof": "-",
                    "timeframe": "1Min",
                }.items()
            ):
                raise ValueError(f"Unexpected minute data conventions in {path}")
            cache[path] = info
        stat = Path(path).stat()
        files.append((str(Path(path).relative_to(root)), stat.st_size, stat.st_mtime_ns))
    return digest({"coverage": rows.to_dicts(), "files": files})


def build_labels(
    config,
    candidate_id,
    start=None,
    end=None,
    policy_names=None,
    fixed=True,
    output_id=None,
    fixed_id=None,
    rebuild=False,
):
    started = monotonic()
    start, end = start or config.dataset.start, end or config.dataset.end
    validate_range(start, end)
    schedule = sessions(start, end)
    if not schedule:
        raise ValueError("No regular trading sessions in requested range")
    source, candidate_meta = read_artifact(config.data.root, "candidates", candidate_id)
    if candidate_meta["definition"]["kind"] != "candidates":
        raise ValueError("Expected a candidate artifact")
    names = ["original_60m"] if policy_names is None else policy_names
    options = {p.name: p for p in config.dataset.policies}
    if set(names) - set(options):
        raise ValueError(f"Unknown label policies: {sorted(set(names) - set(options))}")
    policies = [options[name] for name in dict.fromkeys(names)]
    if output_id and len(policies) != 1:
        raise ValueError("--output-id requires exactly one barrier policy")
    delay = candidate_meta["definition"]["configuration"]["execution_delay"]
    if any(p.execution_delay != delay for p in policies):
        raise ValueError("Label execution_delay must match the source candidate convention")
    horizons = config.dataset.fixed_horizons if fixed else []
    if not policies and not horizons:
        raise ValueError("Select a barrier policy or fixed-horizon targets")
    coverage = minute_inventory(config.data.root, start, end)
    jobs, cache = [], {}
    with ExitStack() as stack:
        for policy in [*policies, *([None] if horizons else [])]:
            definition = label_definition(candidate_meta, policy, horizons)
            prefix = policy.name if policy else "fixed"
            identifier = (
                output_id if policy else fixed_id
            ) or f"{prefix}_{digest(definition)[:16]}"
            stack.enter_context(writer_lock(artifact_path(config.data.root, "labels", identifier)))
            folder, meta = open_artifact(config.data.root, "labels", prefix, definition, identifier)
            plan_partitions(folder, meta, [s["session_date"] for s in schedule])
            jobs.append(
                {"folder": folder, "meta": meta, "policy": policy, "built": 0, "skipped": 0}
            )
        db = stack.enter_context(connection(config, config.data.root))
        for session in schedule:
            day = session["session_date"]
            if not completed(source, candidate_meta, day):
                raise ValueError(
                    f"Candidate partition {day} is not complete; build candidates first"
                )
            candidate_path = partition_path(source, day)
            candidates = pl.read_parquet(candidate_path)
            symbols = candidates["symbol"].unique().sort().to_list()
            paths = minute_paths(coverage, config.data.root, day, symbols)
            dependency = {
                "candidate_partition": candidate_meta["partitions"][str(day)]["sha256"],
                "minute_sources": raw_dependency(
                    coverage, config.data.root, day, symbols, paths, cache
                ),
            }
            pending = []
            for job in jobs:
                if completed(job["folder"], job["meta"], day, dependency, rebuild):
                    job["skipped"] += 1
                else:
                    pending.append(job)
            if not pending:
                continue
            minutes = int((session["close"] - session["open"]).total_seconds() / 60)
            bars = read_minutes(db, paths, symbols, session["open"], session["close"])
            groups = {g["symbol"][0]: g for g in bars.partition_by("symbol")}
            active_policies = [j["policy"] for j in pending if j["policy"] is not None]
            active_fixed = horizons if any(j["policy"] is None for j in pending) else []
            scan_horizons = sorted({p.horizon for p in active_policies} | set(active_fixed))
            upper = sorted({p.tp for p in active_policies})
            lower = sorted({p.sl for p in active_policies})
            frames = [[] for _ in pending]
            for cohort in candidates.partition_by("symbol", maintain_order=True):
                cohort = cohort.sort("decision_timestamp_utc")
                symbol = cohort["symbol"][0]
                decisions = cohort["decision_minute"].to_numpy().astype(int)
                starts = decisions + delay
                expected = int(session["open"].timestamp()) * 1_000_000 + decisions * 60_000_000
                if not np.array_equal(
                    cohort["decision_timestamp_utc"].cast(pl.Int64).to_numpy(), expected
                ):
                    raise ValueError("Candidate timestamps disagree with their session grid")
                prices, present, good = price_grid(groups.get(symbol), session["open"], minutes)
                evidence = scan_paths(prices, present, good, starts, scan_horizons, upper, lower)
                for index, job in enumerate(pending):
                    frame = (
                        barrier_targets(cohort, evidence, job["policy"], session["open"])
                        if job["policy"]
                        else fixed_targets(cohort, evidence, horizons, session["open"])
                    )
                    frames[index].append(frame)
            for job, pieces in zip(pending, frames, strict=True):
                frame = pl.concat(pieces)
                if frame.height != candidates.height:
                    raise ValueError("Label artifact lost candidate observations")
                kind = job["meta"]["definition"]["kind"]
                write_partition(
                    job["folder"],
                    job["meta"],
                    day,
                    frame,
                    dependency,
                    target_summary(frame, kind, horizons),
                )
                job["built"] += 1
                log.info("Labels %s %s: %s rows", job["meta"]["artifact_id"], day, frame.height)
        return [
            finish_run(
                j["folder"], j["meta"], start, end, j["built"], j["skipped"], started, config
            )
            for j in jobs
        ]
