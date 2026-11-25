"""Offline correctness, independent stages, compatibility and interrupted partitions."""

import json
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from dead_reckoning.cli import app
from dead_reckoning.config import CandidateConfig, Config, LabelPolicy
from dead_reckoning.datasets.artifacts import (
    KEYS,
    completed,
    dataset_status,
    file_hash,
    partition_path,
    read_artifact,
    sessions,
)
from dead_reckoning.datasets.candidates import build_candidates, make_candidates
from dead_reckoning.datasets.labels import build_labels
from dead_reckoning.datasets.targets import barrier_targets, fixed_targets
from dead_reckoning.ingestion.manifest import Manifest
from dead_reckoning.labels.paths import price_grid, scan_paths
from dead_reckoning.storage.parquet import atomic_json, read_metadata, write_pages
from dead_reckoning.universe.liquidity import build_universe


@pytest.fixture
def dataset_config(tmp_path):
    config = Config.model_validate(
        {
            "data": {"root": tmp_path},
            "query": {"memory_limit": "256MB", "threads": 1},
            "dataset": {
                "start": "2019-11-27",
                "end": "2019-11-30",
                "candidates": {"universe_size": 2},
            },
        }
    )
    daily = pl.from_dicts(
        [
            {
                "symbol": symbol,
                "timestamp": datetime(2019, 10, 20, 4, tzinfo=UTC) + timedelta(days=k),
                "close": 100.0,
                "volume": volume,
            }
            for symbol, volume in [("AAA", 10000), ("BBB", 1000)]
            for k in range(38)
        ]
    )
    minute_rows = []
    for session in sessions(config.dataset.start, config.dataset.end):
        for symbol in ["AAA", "BBB"]:
            minutes = int((session["close"] - session["open"]).total_seconds() / 60)
            for k in range(minutes):
                if (symbol == "AAA" and k in [10, 11]) or (
                    symbol == "BBB" and k in [1, 50, 51, 52]
                ):
                    continue
                row = {
                    "symbol": symbol,
                    "timestamp": session["open"] + timedelta(minutes=k),
                    "open": 100.0,
                    "high": 100.1,
                    "low": 99.9,
                    "close": 100.0,
                }
                if symbol == "AAA" and k == 14:
                    row.update(open=99.0, high=100.1, low=98.9)
                if symbol == "AAA" and k == 100:
                    row.update(high=101.0, low=99.0)
                minute_rows.append(row)
    for name, frame, timeframe in [
        ("daily_bars", daily, "1Day"),
        ("minute_bars", pl.from_dicts(minute_rows), "1Min"),
    ]:
        write_pages(
            tmp_path / "raw" / name / "test.parquet",
            [frame],
            {
                "schema_version": 1,
                "dataset": name,
                "provider": "alpaca",
                "feed": "sip",
                "adjustment": "raw",
                "asof": "-",
                "timeframe": timeframe,
                "start": "2019-10-01" if name == "daily_bars" else "2019-11-27",
                "end": "2019-11-30",
                "symbols": ["AAA", "BBB"],
                "ingested_at": "2026-01-01T00:00:00+00:00",
            },
        )
    with Manifest(tmp_path) as manifest:
        build_universe(config, manifest, config.dataset.start, config.dataset.end)
    return config


def all_rows(root, stage, identifier):
    folder, meta = read_artifact(root, stage, identifier)
    return pl.concat(
        [
            pl.read_parquet(partition_path(folder, day))
            for day in sorted(meta["partitions"])
            if meta["partitions"][day]["status"] == "complete"
        ]
    )


def test_independent_candidates_labels_incremental_and_join_keys(dataset_config):
    c = dataset_config
    first = build_candidates(c, end=date(2019, 11, 28), output_id="cohort")
    assert first["rows"] == 156  # Full 78-decision grid, never cut to 120m common horizon.
    result = build_candidates(c, output_id="cohort")
    assert result["rows"] == 240 and result["built_partitions"] == result["skipped_partitions"] == 1
    before = all_rows(c.data.root, "candidates", "cohort")
    assert before.select(pl.struct(KEYS).n_unique()).item() == before.height
    assert before.filter(pl.col("universe_feature_date") >= pl.col("session_date")).is_empty()
    assert before.filter(pl.col("decision_timestamp_utc") >= pl.col("session_close_utc")).is_empty()
    results = build_labels(c, "cohort")
    barrier = all_rows(c.data.root, "labels", results[0]["artifact_id"])
    fixed = all_rows(c.data.root, "labels", results[1]["artifact_id"])
    assert barrier.height == fixed.height == before.height
    assert before.join(barrier, on=KEYS).height == before.height
    a = barrier.filter(
        (pl.col("symbol") == "AAA") & (pl.col("decision_timestamp_utc").dt.minute() == 30)
    )
    assert a["raw_outcome"][0] == "SL_FIRST" and a["stop_gap"][0]
    assert a["realized_trade_return"][0] == pytest.approx(-0.01)
    assert not a["strict_valid"][0] and a["max_consecutive_missing_minutes"][0] == 2
    b = barrier.filter(
        (pl.col("symbol") == "BBB") & (pl.col("decision_timestamp_utc").dt.minute() == 30)
    )
    assert b["invalid_reason"][0] == "missing_entry" and b["realized_trade_return"][0] is None
    assert barrier.filter(pl.col("invalid_reason") == "insufficient_session_time").height == 48
    assert (
        barrier.filter(pl.col("invalid_reason") == "insufficient_session_time")[
            "missing_minutes"
        ].null_count()
        == 48
    )
    assert barrier.filter(pl.col("invalid_reason") == "excessive_interior_gap").height > 0
    assert fixed.filter(pl.col("valid_60m")).height > barrier.filter(pl.col("label_valid")).height
    again = build_labels(c, "cohort")
    assert all(r["built_partitions"] == 0 and r["skipped_partitions"] == 2 for r in again)
    other = build_labels(c, "cohort", policy_names=["original_30m", "wide_60m"], fixed=False)
    assert len(other) == 2
    assert all_rows(c.data.root, "candidates", "cohort").equals(before)
    assert all_rows(c.data.root, "labels", other[0]["artifact_id"]).height == before.height


def test_rebuild_resume_staleness_and_definition_compatibility(dataset_config, monkeypatch):
    c = dataset_config
    build_candidates(c, output_id="cohort")
    from dead_reckoning.datasets import artifacts

    real_write = artifacts.write_pages
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated interruption")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(artifacts, "write_pages", fail_second)
    with pytest.raises(OSError, match="interruption"):
        build_labels(c, "cohort", fixed=False, output_id="labels")
    folder, meta = read_artifact(c.data.root, "labels", "labels")
    assert meta["partitions"]["2019-11-29"]["status"] == "building"
    assert not completed(folder, meta, "2019-11-29")
    monkeypatch.setattr(artifacts, "write_pages", real_write)
    resumed = build_labels(c, "cohort", fixed=False, output_id="labels")[0]
    assert resumed["built_partitions"] == resumed["skipped_partitions"] == 1
    old_hash = file_hash(partition_path(folder, "2019-11-27"))
    selected = build_labels(
        c, "cohort", start=date(2019, 11, 29), fixed=False, output_id="labels", rebuild=True
    )[0]
    assert selected["built_partitions"] == 1
    assert file_hash(partition_path(folder, "2019-11-27")) == old_hash
    changed = c.model_copy(deep=True)
    changed.dataset.policies[0].tp = 0.006
    with pytest.raises(ValueError, match="Incompatible"):
        build_labels(changed, "cohort", fixed=False, output_id="labels")
    changed.dataset.policies[0].execution_delay = 2
    with pytest.raises(ValueError, match="execution_delay"):
        build_labels(changed, "cohort", fixed=False)
    meta = json.loads((folder / "metadata.json").read_text())
    meta["partitions"]["2019-11-27"]["dependency"]["minute_sources"] = "changed"
    atomic_json(folder / "metadata.json", meta)
    with pytest.raises(ValueError, match="Stale dependency"):
        build_labels(c, "cohort", fixed=False, output_id="labels")


def test_coverage_required_without_selecting_on_future_availability(dataset_config):
    c = dataset_config
    result = build_candidates(c, output_id="cohort")
    path = c.data.root / "raw/minute_bars/test.parquet"
    original = pl.read_parquet(path)
    before = all_rows(c.data.root, "candidates", "cohort")
    # Complete requests can contain no bars; this must not shrink or alter candidates.
    metadata = read_metadata(path)
    counts = write_pages(path, [original.head(0)], metadata)
    with Manifest(c.data.root) as manifest:
        manifest.record(path, metadata, "complete", counts)
    build_candidates(c, output_id="empty_future")
    assert before.equals(all_rows(c.data.root, "candidates", "empty_future"))
    # An absent raw file is a genuine uncovered request and cannot be silently dropped.
    path.unlink()
    with pytest.raises(ValueError, match="Minute manifest missing"):
        build_candidates(c, output_id="another")
    assert result["rows"] == 240 and original.height > 0


@pytest.mark.parametrize("ambiguity", ["conservative", "optimistic", "exclude"])
def test_execution_ambiguity_endpoints_quality_and_scheduled_targets(ambiguity):
    opening = datetime(2019, 1, 2, 14, 30, tzinfo=UTC)
    members = pl.DataFrame(
        {
            "symbol": ["AAA"],
            "liquidity_rank": [1],
            "trailing_median_dollar_volume": [100.0],
            "feature_date": [date(2018, 12, 31)],
        }
    )
    cohort = make_candidates(
        members,
        {"session_date": opening.date(), "open": opening, "close": opening + timedelta(minutes=90)},
        CandidateConfig(),
    )
    prices = np.full((90, 4), 100.0)
    prices[3, [1, 2]] = [101, 99]
    present = np.ones(90, dtype=bool)
    present[2] = False
    prices[2] = np.nan
    evidence = scan_paths(
        prices,
        present,
        present,
        cohort["decision_minute"].to_numpy() + 1,
        [15, 60],
        [0.005],
        [0.004],
    )
    policy = LabelPolicy(ambiguity=ambiguity)
    barrier = barrier_targets(cohort, evidence, policy, opening)
    assert barrier["raw_outcome"][0] == "AMBIGUOUS"
    assert barrier["conservative_outcome"][0] == "SL_FIRST"
    assert barrier["conservative_trade_return"][0] == pytest.approx(-0.004)
    assert barrier["optimistic_trade_return"][0] == pytest.approx(0.005)
    assert barrier["label_valid"][0] == (ambiguity != "exclude")
    assert (
        barrier["outcome"][0]
        == {"conservative": "SL_FIRST", "optimistic": "TP_FIRST", "exclude": None}[ambiguity]
    )
    fixed = fixed_targets(cohort, evidence, [15, 60], opening)
    assert fixed["valid_60m"][0] and fixed["return_60m"][0] == 0
    strict = barrier_targets(cohort, evidence, LabelPolicy(missing_tolerance=0), opening)
    assert not strict["label_valid"][0] and fixed["valid_60m"][0]
    assert fixed["valid_15m"][-1] is False
    assert fixed["valid_15m"][12] is True and fixed["valid_60m"][12] is False
    good = present.copy()
    good[3] = False
    bad = barrier_targets(
        cohort,
        scan_paths(prices, present, good, evidence.starts, [60], [0.005], [0.004]),
        policy,
        opening,
    )
    assert bad["invalid_reason"][0] == "invalid_data" and bad["realized_trade_return"][0] is None
    present[60] = False
    prices[60] = np.nan
    missing_exit = scan_paths(prices, present, present, evidence.starts, [60], [0.005], [0.004])
    assert not fixed_targets(cohort, missing_exit, [60], opening)["valid_60m"][0]
    # The early established barrier exit is valid even without a later fixed endpoint.
    assert barrier_targets(cohort, missing_exit, LabelPolicy(), opening)["label_valid"][0]


def test_duplicate_unaligned_grid_and_cli_status(dataset_config, tmp_path):
    opening = datetime(2019, 1, 2, 14, 30, tzinfo=UTC)
    bars = pl.DataFrame(
        {
            "timestamp": [opening, opening, opening + timedelta(minutes=1, seconds=1)],
            "open": [100.0] * 3,
            "high": [100.0] * 3,
            "low": [100.0] * 3,
            "close": [100.0] * 3,
        }
    )
    _, present, good = price_grid(bars, opening, 60)
    assert present[:2].all() and not good[:2].any()
    c = dataset_config
    build_candidates(c, output_id="cohort")
    assert dataset_status(c.data.root)[0]["complete"] == 2
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f'data:\n  root: "{c.data.root.as_posix()}"\n')
    result = CliRunner().invoke(app, ["--config", str(cfg), "dataset", "status"])
    assert result.exit_code == 0 and "cohort" in result.output
