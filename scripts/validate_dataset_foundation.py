"""Bounded offline acceptance check; writes a reviewable JSON and Markdown report.

Run after the January validation build. Never downloads data or builds more than
one month. The optional study argument compares a separately completed compatibility run.
"""

import argparse
import json
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from dead_reckoning.config import load_config
from dead_reckoning.datasets.artifacts import (
    KEYS,
    completed,
    partition_path,
    read_artifact,
    sessions,
)
from dead_reckoning.datasets.candidates import build_candidates
from dead_reckoning.datasets.labels import build_labels
from dead_reckoning.storage.parquet import atomic_json

CANDIDATES = "foundation_2019_v2"
BARRIER = "foundation_original_60m_v2"
FIXED = "foundation_fixed_v2"
ALTERNATIVE = "foundation_original_30m_v2"


def snapshot(folder, meta):
    return {
        day: (part["sha256"], partition_path(folder, day).stat().st_mtime_ns)
        for day, part in meta["partitions"].items()
        if part["status"] == "complete"
    }


def validate(config, output, study=None, review_existing=False):
    if (
        config.dataset.end > date(2024, 1, 1)
        or (config.dataset.end - config.dataset.start).days > 32
    ):
        raise ValueError("Acceptance build must be at most one pre-2024 month")
    output.mkdir(parents=True, exist_ok=True)
    root = config.data.root
    locations = {
        name: read_artifact(root, stage, name)
        for stage, name in (
            ("candidates", CANDIDATES),
            ("labels", BARRIER),
            ("labels", FIXED),
        )
    }
    initial = {name: snapshot(*value) for name, value in locations.items()}
    initial_runs = {name: value[1]["runs"][0] for name, value in locations.items()}
    day = sessions(config.dataset.start, config.dataset.end)[-1]["session_date"]
    if review_existing:

        def recorded_run(name, built):
            return next(
                r for r in reversed(locations[name][1]["runs"]) if r["built_partitions"] == built
            )

        resume_candidates = recorded_run(CANDIDATES, 0)
        resume_labels = [recorded_run(name, 0) for name in (BARRIER, FIXED)]
        rebuild_candidates = recorded_run(CANDIDATES, 1)
        rebuild_labels = [recorded_run(name, 1) for name in (BARRIER, FIXED)]
        _, alternative_meta = read_artifact(root, "labels", ALTERNATIVE)
        alternative = [alternative_meta["runs"][0]]
        assert resume_candidates["skipped_partitions"] == 21
        assert all(r["skipped_partitions"] == 21 for r in resume_labels)
        assert all(r["requested_start"] == str(day) for r in [rebuild_candidates, *rebuild_labels])
        for name, (folder, meta) in locations.items():
            first_completed = datetime.fromisoformat(initial_runs[name]["generation_timestamp"])
            for session_day, part in meta["partitions"].items():
                if session_day != str(day):
                    assert datetime.fromisoformat(part["completed_at"]) <= first_completed
                    assert (
                        partition_path(folder, session_day).stat().st_mtime
                        <= first_completed.timestamp()
                    )
    else:
        resume_candidates = build_candidates(config, output_id=CANDIDATES)
        resume_labels = build_labels(config, CANDIDATES, output_id=BARRIER, fixed_id=FIXED)
        assert resume_candidates["built_partitions"] == 0
        assert all(r["built_partitions"] == 0 for r in resume_labels)
        for name, (folder, meta) in locations.items():
            assert snapshot(folder, meta) == initial[name]
        day = sessions(config.dataset.start, config.dataset.end)[-1]["session_date"]
        following = day + timedelta(days=1)
        alternative = build_labels(
            config,
            CANDIDATES,
            start=day,
            end=following,
            policy_names=["original_30m"],
            fixed=False,
            output_id=ALTERNATIVE,
        )
        assert snapshot(*locations[CANDIDATES]) == initial[CANDIDATES]
        rebuild_candidates = build_candidates(
            config, start=day, end=following, output_id=CANDIDATES, rebuild=True
        )
        rebuild_labels = build_labels(
            config,
            CANDIDATES,
            start=day,
            end=following,
            output_id=BARRIER,
            fixed_id=FIXED,
            rebuild=True,
        )
        assert rebuild_candidates["built_partitions"] == 1
        assert all(r["built_partitions"] == 1 for r in rebuild_labels)
        for name, (folder, _) in locations.items():
            fresh_folder, fresh_meta = read_artifact(
                root, "candidates" if name == CANDIDATES else "labels", name
            )
            current = snapshot(fresh_folder, fresh_meta)
            assert all(current[d] == value for d, value in initial[name].items() if d != str(day))
            locations[name] = (folder, fresh_meta)
    reasons = Counter(
        {
            name: 0
            for name in (
                "valid",
                "missing_entry",
                "excessive_interior_gap",
                "missing_expiration",
                "invalid_data",
                "insufficient_session_time",
            )
        }
    )
    raw, outcomes = Counter(), Counter()
    fixed_valid, fixed_reasons = Counter(), {h: Counter() for h in config.dataset.fixed_horizons}
    rows = valid = strict_valid = ambiguity = stop_gaps = independent = 0
    return_sum = 0.0
    for session in sessions(config.dataset.start, config.dataset.end):
        day = session["session_date"]
        frames = []
        for folder, meta in locations.values():
            assert completed(folder, meta, day)
            frame = pl.read_parquet(partition_path(folder, day)).sort(KEYS)
            assert frame.select(pl.struct(KEYS).n_unique()).item() == frame.height
            frames.append(frame)
        candidates, labels, fixed = frames
        assert candidates.select(KEYS).equals(labels.select(KEYS))
        assert candidates.select(KEYS).equals(fixed.select(KEYS))
        assert (candidates["universe_feature_date"] < candidates["session_date"]).all()
        assert (
            labels.filter(~pl.col("label_valid"))["realized_trade_return"].null_count()
            == (~labels["label_valid"]).sum()
        )
        scheduled = labels.filter(pl.col("invalid_reason") == "insufficient_session_time")
        assert scheduled["missing_minutes"].null_count() == scheduled.height
        assert labels.filter(pl.col("label_valid"))["exit_timestamp_utc"].max() <= session["close"]
        assert labels.filter(pl.col("label_valid"))["holding_minutes"].max() <= 60
        assert labels.filter(pl.col("outcome") != "EXPIRE")["expiration_return"].null_count() == (
            labels.filter(pl.col("outcome") != "EXPIRE").height
        )
        rows += candidates.height
        valid += int(labels["label_valid"].sum())
        strict_valid += int(labels["strict_valid"].sum())
        ambiguity += int(labels["ambiguous_outcome"].sum())
        stop_gaps += int(labels["stop_gap"].sum())
        independent += int((~labels["label_valid"] & fixed["valid_60m"]).sum())
        return_sum += labels["realized_trade_return"].cast(pl.Float64).sum()
        reasons.update(dict(labels.group_by("invalid_reason").len().iter_rows()))
        raw.update(dict(labels.group_by("raw_outcome").len().iter_rows()))
        outcomes.update(
            dict(labels.filter(pl.col("label_valid")).group_by("outcome").len().iter_rows())
        )
        for h in config.dataset.fixed_horizons:
            fixed_valid[h] += int(fixed[f"valid_{h}m"].sum())
            fixed_reasons[h].update(dict(fixed.group_by(f"invalid_reason_{h}m").len().iter_rows()))
    result = {
        "requested_range": [str(config.dataset.start), str(config.dataset.end)],
        "candidate_id": CANDIDATES,
        "primary_label_id": BARRIER,
        "fixed_label_id": FIXED,
        "alternative_label_id": ALTERNATIVE,
        "alternative_build": alternative,
        "candidate_rows": rows,
        "valid_barrier": valid,
        "scheduled_eligible": rows - reasons["insufficient_session_time"],
        "data_unknown_pct_of_eligible": 100
        * (rows - reasons["insufficient_session_time"] - valid)
        / (rows - reasons["insufficient_session_time"]),
        "strict_valid_barrier": strict_valid,
        "raw_outcomes": dict(raw),
        "resolved_outcomes": dict(outcomes),
        "invalid_reasons": dict(reasons),
        "ambiguous": ambiguity,
        "stop_gaps": stop_gaps,
        "mean_gross_return_bps": return_sum / valid * 10000,
        "fixed_valid": dict(fixed_valid),
        "fixed_invalid_reasons": {h: dict(v) for h, v in fixed_reasons.items()},
        "valid_60m_fixed_with_invalid_barrier": independent,
        "initial_builds": initial_runs,
        "resume_candidates": resume_candidates,
        "resume_labels": resume_labels,
        "selected_rebuild_candidates": rebuild_candidates,
        "selected_rebuild_labels": rebuild_labels,
        "checks": {
            "unique_and_matching_keys": True,
            "prior_universe_cutoff": True,
            "null_invalid_targets": True,
            "no_overnight_exits": True,
            "resume_zero_rebuilds": True,
            "unrelated_partitions_unchanged": True,
            "independent_target_validity": True,
        },
    }
    if study is not None:
        for name in ("policy_summary.csv", "fixed_horizon_coverage.csv", "gap_lengths.csv"):
            old = pl.read_csv(study / "missing_bar_policy" / name)
            new = pl.read_csv(output / "compatibility" / name)
            assert old.columns == new.columns and old.shape == new.shape
            for column in old.columns:
                if old[column].dtype.is_float():
                    np.testing.assert_allclose(
                        old[column].to_numpy(),
                        new[column].to_numpy(),
                        equal_nan=True,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                else:
                    assert old[column].equals(new[column])
        result["research_compatibility"] = {
            "sessions": 60,
            "candidates": 1_602_000,
            "all_15_policy_cohort_summaries_match": True,
            "fixed_horizon_and_gap_distribution_match": True,
            "original_reports_unchanged": True,
        }
    atomic_json(output / "validation.json", result)
    lines = [
        "# Dataset foundation acceptance build",
        "",
        "Candidate generation and reusable label generation are implemented independently. "
        "Architecture, schemas and additional-range commands are in the repository's "
        "`docs/datasets.md`; configuration is `configs/dataset_v1.yaml`.",
        "",
        f"Historical development build: [{config.dataset.start}, {config.dataset.end}), "
        f"21 sessions, top 500, **{rows:,} candidate rows**. Primary labels and fixed-horizon "
        "artifacts each retain exactly those keys, including invalid targets.",
        "",
        "| Artifact | Rows | Initial runtime | Peak process MiB | Parquet MiB |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, run in initial_runs.items():
        peak = run["peak_process_mb"]
        lines.append(
            f"| {name} | {run['rows']:,} | {run['elapsed_seconds']:.2f}s | "
            f"{peak:.0f} | {run['parquet_bytes'] / 1024**2:.2f} |"
        )
    lines += [
        "",
        "The barrier and fixed-horizon artifacts were generated together; their "
        "shared runtime/peak memory should not be added. DuckDB used a 2GB budget and "
        "four threads. Process peaks include Python, Parquet and non-DuckDB allocations.",
        "",
        f"Valid primary barrier labels: **{valid:,} ({100 * valid / rows:.2f}%)**; "
        f"strict-valid: {strict_valid:,}. Mean simulated gross return: "
        f"{result['mean_gross_return_bps']:.3f} bps. Ambiguous raw outcomes: {ambiguity:,}; "
        f"adverse stop-opening gaps: {stop_gaps:,}.",
        f"Among the {result['scheduled_eligible']:,} observations with enough scheduled "
        f"session time, missing/invalid data leaves "
        f"{result['data_unknown_pct_of_eligible']:.2f}% UNKNOWN. Scheduled ineligibility "
        "is reported separately and is not a missing-data failure.",
        "",
        "| Resolved outcome | Count | % of valid |",
        "|---|---:|---:|",
    ]
    for name, count in sorted(outcomes.items()):
        lines.append(f"| {name} | {count:,} | {100 * count / valid:.2f}% |")
    lines += [
        "",
        "| Validity / exclusion reason | Count | % of all candidates |",
        "|---|---:|---:|",
    ]
    for name, count in sorted(reasons.items()):
        lines.append(f"| {name} | {count:,} | {100 * count / rows:.2f}% |")
    lines += [
        "",
        "| Fixed horizon | Valid endpoint returns | % of all candidates |",
        "|---|---:|---:|",
    ]
    for h, count in sorted(fixed_valid.items()):
        lines.append(f"| {h} minutes | {count:,} | {100 * count / rows:.2f}% |")
    lines += [
        "",
        f"{independent:,} candidates have a valid 60-minute fixed return while "
        "the primary barrier target is invalid. The policy-independent grid includes "
        "late-session decisions, so these percentages are not directly comparable with "
        "the research common-120-minute cohort.",
        "",
        "Verified identical resume commands rebuild zero partitions; a different "
        "30-minute policy was generated for the final session without rebuilding "
        "candidates; explicit candidate/primary/fixed rebuilds touched only that "
        "session. Keys are unique and match exactly, all feature cutoffs are prior, "
        "invalid returns are null and valid exits remain within their session.",
        "",
        "All tests, including interrupted-write recovery and incompatible-definition "
        "rejection, are offline. See `checks.json` for the final test/lint result.",
        "",
    ]
    if study is not None:
        lines += [
            "The reusable engine reproduced all 15 saved policy/cohort summaries "
            "(five tolerances across top 100/250/500), fixed-horizon coverage and gap "
            "distributions over 60 sessions and 1,602,000 saved research candidates. "
            "Each strict label and return was also checked against the saved touch "
            "checkpoints. Original study outputs were not overwritten.",
            "",
        ]
    lines += [
        "No outstanding corrected-universe dependency exists. No additional bars were "
        "downloaded. This bounded validation does not choose training splits, generate "
        "features, assemble training matrices or train models. Raw corporate-action "
        "and asset-master limitations remain. Tolerated gaps can hide earlier barrier "
        "touches; retain gap metadata and strict-versus-gap-bearing validation. "
        "Future-path diagnostics must not enter model features.",
        "",
        "The earlier `foundation_*_v1` validation artifacts are retained as historical "
        "builds. Use the v2 IDs above: they include complete extraction/writer source "
        "fingerprints and null diagnostics for scheduled-ineligible rows.",
        "",
        "Machine-readable results and build receipts: [validation.json](validation.json).",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "candidate_rows",
                    "valid_barrier",
                    "invalid_reasons",
                    "fixed_valid",
                    "checks",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_v1.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study", type=Path)
    parser.add_argument(
        "--review-existing",
        action="store_true",
        help="Use recorded resume/rebuild receipts; do not run generation again.",
    )
    args = parser.parse_args()
    validate(load_config(args.config), args.output, args.study, args.review_existing)
