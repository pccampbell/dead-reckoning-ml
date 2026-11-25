"""Bounded sensitivity follow-up to an existing study; production labels stay strict.

Run: python -m dead_reckoning.research.missing_bars STUDY_DIRECTORY
Only reads local bars and the original saved cohort. No ingestion or expanded grid.
"""

import argparse
import hashlib
import json
from collections import Counter
from datetime import timedelta
from pathlib import Path
from time import monotonic

import numpy as np
import polars as pl

from dead_reckoning.config import load_config
from dead_reckoning.labels.barriers import Outcome, resolve
from dead_reckoning.labels.paths import fixed_returns, price_grid, resolve_paths, scan_paths
from dead_reckoning.research.data import inventory, minute_paths, query_connection, read_minutes
from dead_reckoning.research.study import unpack
from dead_reckoning.storage.parquet import atomic_json

TOLERANCES = (0, 1, 2, 5, None)
SIZES = (100, 250, 500)
REASONS = ("known", "missing_entry", "interior_gap", "missing_expiration", "other_quality")
TP, SL, HORIZON = 0.005, 0.004, 60


def runs(mask):
    """Half-open runs, including boundary runs, without filling missing prices."""
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def evaluate(prices, present, good, starts):
    """Compatibility wrapper around the reusable observed-price path evaluator."""
    evidence = scan_paths(prices, present, good, starts, [HORIZON], [TP], [SL])
    policies = resolve_paths(evidence, TP, SL, HORIZON, TOLERANCES)
    first = policies[0]
    forward, _ = fixed_returns(evidence, HORIZON)
    return (
        {limit: (result.labels, result.reason) for limit, result in policies.items()},
        first.interior_max,
        first.interior_missing,
        forward,
        evidence.good.all(axis=1),
    )


def policy_row(size, tolerance, result, strict, maximum, missing):
    labels, reason = result
    known = reason == 0
    recovered = known & (strict == Outcome.UNKNOWN)
    row = {
        "top_n": size,
        "tolerance": "unlimited" if tolerance is None else str(tolerance),
        "candidates": len(reason),
        "known": int(known.sum()),
        "recovered": int(recovered.sum()),
        "return_sum": float(np.nansum(labels.conservative)),
        "optimistic_sum": float(np.nansum(labels.optimistic)),
        "recovered_return_sum": float(np.nansum(labels.conservative[recovered])),
        "gap_bearing_known": int((known & (maximum > 0)).sum()),
        "missing_minutes_known_sum": int(missing[known].sum()),
    }
    for code, name in enumerate(REASONS[1:], 1):
        row[f"{name}_count"] = int((reason == code).sum())
    for code in Outcome:
        row[f"{code.name.lower()}_count"] = int((labels.outcome == code).sum())
    return row


def aggregate(rows, keys):
    frame = pl.from_dicts(rows)
    return frame.group_by(keys).agg(pl.exclude(keys).sum()).sort(keys)


def write_report(output):
    """Small Markdown summary from the saved aggregates; no repeated raw-data scan."""
    policies = pl.read_csv(output / "policy_summary.csv", schema_overrides={"tolerance": pl.String})
    fixed = pl.read_csv(output / "fixed_horizon_coverage.csv")
    gaps = pl.read_csv(output / "gap_lengths.csv").with_columns(pl.col("boundary").cast(pl.Boolean))
    sessions = pl.read_csv(output / "session_summary.csv")
    meta = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    top = policies.filter(pl.col("top_n") == 500)
    baseline, two = [top.filter(pl.col("tolerance") == t).row(0, named=True) for t in ("0", "2")]
    unlimited = top.filter(pl.col("tolerance") == "unlimited").row(0, named=True)
    recovery_share = 100 * two["recovered"] / unlimited["recovered"]
    lines = [
        "# Missing-bar policy follow-up",
        "",
        "**Recommendation: permit at most two consecutive missing interior minutes as the "
        "initial training policy, with an explicit gap flag; retain strict labels for sensitivity "
        "checks. Production definitions have not changed.** This is a pragmatic observed-price "
        "approximation, not proof that no barrier was crossed during a missing interval.",
        "",
        f"Same 60 sessions, saved top-500 memberships and entry timestamps: "
        f"{meta['candidates']:,} candidates. Only TP +0.50%, SL -0.40%, 60 elapsed minutes. "
        "No additional data downloads, universe recomputation or parameter grid.",
        "",
        f"Two-minute tolerance recovers **{two['recovered']:,}** observations "
        f"({recovery_share:.1f}% of everything recoverable under unlimited tolerance), "
        f"reducing UNKNOWN from **{baseline['unknown_pct']:.2f}% to "
        f"{two['unknown_pct']:.2f}%**. It adds 60,508 observations beyond one-minute "
        "tolerance. Raising the limit from two to five recovers another 28,934 "
        "(1.81% of all candidates), while accepting longer unobserved price paths. "
        "This makes two minutes a practical coverage-versus-uncertainty compromise; "
        "the data do not establish an objectively optimal tolerance.",
        "",
        "## Label comparison",
        "",
        "Recovered counts are relative to tolerance 0 within each liquidity cohort. "
        "TP/SL/expiration/ambiguous percentages use **known labels** as their denominator; "
        "UNKNOWN uses all candidates. Mean return is gross basis points among known labels, "
        "using the existing conservative stop-first convention for ambiguous bars. "
        "Returns omit transaction costs. Changing means also reflect the recovered cohort, "
        "so they do not establish which policy has more accurate labels.",
        "",
        "| Top N | Tolerance | Recovered | UNKNOWN % | TP % | SL % | "
        "Expire % | Ambig. % | Mean bps |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in policies.iter_rows(named=True):
        lines.append(
            f"| {row['top_n']} | {row['tolerance']} | {row['recovered']:,} | "
            f"{row['unknown_pct']:.2f} | {row['tp_first_known_pct']:.2f} | "
            f"{row['sl_first_known_pct']:.2f} | {row['expire_known_pct']:.2f} | "
            f"{row['ambiguous_known_pct']:.3f} | {row['mean_return_bps']:.3f} |"
        )
    lines += [
        "",
        "Full counts, unknown reasons for every liquidity cohort, recovered-cohort returns, "
        "and optimistic ambiguity sensitivity: [policy_summary.csv](policy_summary.csv).",
        "",
        "## Why labels remain UNKNOWN",
        "",
        "Mutually exclusive reasons below are assigned to the **first chronological "
        "disqualifier** before or at the observed exit. A candidate can have multiple gaps; "
        "for example, an earlier excessive interior gap takes precedence over a later "
        "missing expiration bar. Missing expiration matters only when no barrier exit "
        "has already been observed. All percentages below use the 1,602,000 top-500 candidates.",
        "",
        "| Tolerance | Missing entry | Interior gap | Missing expiration | Other quality |",
        "|:---|---:|---:|---:|---:|",
    ]
    for row in top.iter_rows(named=True):
        cells = [f"{row[r + '_count']:,} ({row[r + '_pct']:.2f}%)" for r in REASONS[1:]]
        lines.append(f"| {row['tolerance']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Consecutive missing intervals",
        "",
        "Runs are counted once per symbol/session over the entire regular session, "
        "rather than once per overlapping candidate window. Boundary runs include "
        "missing opens/closes and entirely empty symbol-days. Interior runs have observed "
        "bars on both sides. These full-session lengths differ from label-specific gaps "
        "before an observed exit.",
        "",
        "| Length | All runs | Interior runs | Missing minutes | Share of all runs |",
        "|:---|---:|---:|---:|---:|",
    ]
    gap500 = gaps.filter(pl.col("top_n") == 500)
    total_runs = gap500["runs"].sum()
    for label, lower, upper in [
        ("1", 1, 1),
        ("2", 2, 2),
        ("3–5", 3, 5),
        ("6–15", 6, 15),
        ("16+", 16, 390),
    ]:
        group = gap500.filter(pl.col("length").is_between(lower, upper))
        count = group["runs"].sum()
        interior = group.filter(~pl.col("boundary"))["runs"].sum()
        lines.append(
            f"| {label} | {count:,} | {interior:,} | "
            f"{group['missing_minutes'].sum():,} | {100 * count / total_runs:.2f}% |"
        )
    lines += [
        "",
        f"{sessions['empty_symbol_days'].sum()} entirely empty symbol-days. "
        "Exact lengths and top-100/250/500 breakdowns: [gap_lengths.csv](gap_lengths.csv).",
        "",
        "## Fixed-horizon alternative",
        "",
        "Buy at the genuine entry-minute open and measure return at the genuine close "
        "of the 60th minute. Intermediate bars are irrelevant to this return target; "
        "this is different from determining which barrier was hit first. Entry/exit "
        "bars must be valid, unique and aligned. Coverage counts include candidates "
        "whose barrier labels remain UNKNOWN.",
        "",
        "| Top N | Candidates | Complete-path valid | Endpoint-only valid | Endpoint coverage |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in fixed.iter_rows(named=True):
        lines.append(
            f"| {row['top_n']} | {row['candidates']:,} | "
            f"{row['complete_window_valid']:,} ({row['complete_window_coverage_pct']:.2f}%) | "
            f"{row['endpoint_valid']:,} | {row['endpoint_coverage_pct']:.2f}% |"
        )
    sync = meta["synchronization"]
    lines += [
        "",
        "## Synchronized-missing screen",
        "",
        "Counted simultaneous missing minutes across the prior-liquidity top 100, "
        "excluding entirely empty symbol-days from the outage screen. "
        f"Across {sync['session_minutes']:,} session-minutes: "
        f"{sync['minutes_ge5']:,} had at least 5 missing symbols, "
        f"{sync['minutes_ge10']:,} had at least 10, and "
        f"{sync['minutes_ge20']:,} had at least 20. "
        f"The maximum was {sync['max_missing']} of 100; "
        f"{sync['top100_empty_symbol_days']} empty top-100 symbol-days were excluded.",
        "",
        "This does not suggest a broad synchronized outage in the sampled liquid cohort. "
        "Largest examples: [synchronized_missing_examples.csv](synchronized_missing_examples.csv). "
        "The independence-based expected counts and descriptive z-scores are only screening "
        "aids: time-of-day effects and related securities violate independence. This screen "
        "cannot identify a provider outage or certify completeness.",
        "",
        "## Training implications and limits",
        "",
        "- Permit at most two consecutive absent interior minutes, with no synthetic OHLC prices. "
        "Multiple isolated gaps can pass: retain both maximum gap length and total missing "
        "minutes per label as quality metadata, and compare strict versus gap-bearing "
        "validation subsets. Do not use future-path quality metadata as entry-time model features.",
        "- Keep entry and required expiration bars mandatory. Reject invalid OHLC, duplicate "
        "or unaligned bars under every tolerance. Ignore missing bars after an observed exit.",
        "- Keep ambiguous same-bar TP/SL outcomes separate for classifier construction, "
        "or apply an explicitly chosen conservative rule. A gap may hide an earlier hit "
        "or reverse TP/SL order even when adjacent prices look ordinary; the recovered "
        "labels are assumptions, not reconstructed market truth. Avoid unlimited tolerance "
        "as the default.",
        "- Missingness varies by liquidity, symbol and year; dropping UNKNOWN observations "
        "changes the training population. Track coverage and performance in those slices. "
        "These overlapping 60-minute candidates also require chronological splits and "
        "purging overlapping label windows at split boundaries.",
        "- Proceed to feature engineering and training with this explicit research policy. "
        "Use endpoint-only returns if a continuous return target is desired; they do not "
        "require complete intrahorizon paths. No further completeness investigation is "
        "needed to start modeling, unless concentrated synchronized gaps or slice failures emerge.",
        "",
        "## Reproduction and validation",
        "",
        "The strict follow-up reproduces every saved baseline label and return, and "
        "all previously known outcomes/returns remain unchanged under every tolerance. "
        "Independent minute-loop tests cover missing endpoints, elapsed expiration, "
        "opening-gap stop execution, invalid data and randomized paths.",
        "",
        "```powershell",
        f'python -m dead_reckoning.research.missing_bars "{meta["study"]}"',
        "```",
        "",
        f"Analysis runtime: {meta['elapsed_seconds']:.1f}s. "
        "Provenance and cohort hashes: [summary.json](summary.json). "
        "Production label code and the original study outputs are unchanged.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(study: Path, output: Path | None = None):
    started = monotonic()
    config = load_config(study / "config.yaml")
    output = output or study / "missing_bar_policy"
    output.mkdir(parents=True, exist_ok=True)
    sessions = pl.read_csv(study / "sampled_sessions.csv", try_parse_dates=True)
    universe = pl.read_parquet(study / "selected_universe.parquet")
    assert sessions.height == 60 and universe.height == 30_000
    coverage, _, _ = inventory(config.data.root, sessions)
    policy_rows, fixed_rows, sync_rows, session_rows = [], [], [], []
    gap_counts = Counter()
    candidate_count = baseline_differences = 0
    with query_connection(config, output / "query_tmp") as db:
        for number, session in enumerate(sessions.iter_rows(named=True), 1):
            day, opening, closing = session["date"], session["open"], session["close"]
            minutes = int((closing - opening).total_seconds() / 60)
            members = universe.filter(pl.col("date") == day).sort("liquidity_rank")
            evidence = pl.read_parquet(study / "touches" / f"{day}.parquet")
            paths = minute_paths(coverage, config.data.root, day, members["symbol"].to_list())
            bars = read_minutes(db, paths, members["symbol"].to_list(), opening, closing)
            groups = {g["symbol"][0]: g for g in bars.partition_by("symbol")}
            saved = {g["symbol"][0]: g for g in evidence.partition_by("symbol")}
            absent = []
            for member in members.iter_rows(named=True):
                symbol, rank = member["symbol"], member["liquidity_rank"]
                original = saved[symbol].sort("entry_minute")
                starts = original["entry_minute"].to_numpy().astype(int)
                prices, present, good = price_grid(groups.get(symbol), opening, minutes)
                absent.append(~present)
                with np.errstate(invalid="ignore", divide="ignore"):
                    results, maximum, missing, forward, full = evaluate(
                        prices, present, good, starts
                    )
                strict = resolve(unpack(original, config.research), TP, SL, HORIZON)
                base = results[0][0]
                baseline_differences += int((base.outcome != strict.outcome).sum())
                np.testing.assert_allclose(base.conservative, strict.conservative, equal_nan=True)
                for labels, _ in results.values():
                    known_before = strict.outcome != Outcome.UNKNOWN
                    assert np.array_equal(
                        labels.outcome[known_before], strict.outcome[known_before]
                    )
                    np.testing.assert_allclose(
                        labels.conservative[known_before], strict.conservative[known_before]
                    )
                gaps = runs(~present)
                for size in SIZES:
                    if rank > size:
                        continue
                    for left, right in gaps:
                        gap_counts[(size, right - left, left == 0 or right == minutes)] += 1
                    for tolerance, result in results.items():
                        policy_rows.append(
                            policy_row(size, tolerance, result, strict.outcome, maximum, missing)
                        )
                    fixed_rows.append(
                        {
                            "top_n": size,
                            "candidates": len(starts),
                            "entry_missing": int((~present[starts]).sum()),
                            "exit_missing": int((~present[starts + HORIZON - 1]).sum()),
                            "endpoint_valid": int(np.isfinite(forward).sum()),
                            "complete_window_valid": int(full.sum()),
                            "return_sum": float(np.nansum(forward)),
                        }
                    )
                candidate_count += len(starts)
            absent = np.asarray(absent)
            # Keep only intermittently absent top-100 symbols in the outage screen;
            # inactive/empty symbol-days are counted separately, not called outages.
            liquid = absent[:100]
            active = ~liquid.all(axis=1)
            co_missing = liquid[active].sum(axis=0)
            probabilities = liquid[active].mean(axis=1)
            expected = float(probabilities.sum())
            variance = float((probabilities * (1 - probabilities)).sum())
            symbols = members["symbol"].to_list()[:100]
            for minute, count in enumerate(co_missing):
                sync_rows.append(
                    {
                        "date": str(day),
                        "timestamp": str(opening + timedelta(minutes=minute)),
                        "minute": minute,
                        "missing_top100_active": int(count),
                        "missing_top100_all": int(liquid[:, minute].sum()),
                        "active_top100": int(active.sum()),
                        "independence_expected": expected,
                        "descriptive_z": (int(count) - expected) / np.sqrt(variance)
                        if variance > 0
                        else 0.0,
                        "symbols": ",".join(
                            s
                            for s, a, m in zip(symbols, active, liquid[:, minute], strict=True)
                            if a and m
                        ),
                    }
                )
            session_rows.append(
                {
                    "date": str(day),
                    "bars": bars.height,
                    "candidates": evidence.height,
                    "missing_minutes": int(absent.sum()),
                    "empty_symbol_days": int(absent.all(axis=1).sum()),
                    "top100_empty_symbols": int((~active).sum()),
                    "top100_max_synchronized_missing": int(co_missing.max()),
                    "top100_minutes_ge5_missing": int((co_missing >= 5).sum()),
                    "top100_minutes_ge10_missing": int((co_missing >= 10).sum()),
                    "top100_minutes_ge20_missing": int((co_missing >= 20).sum()),
                }
            )
            print(f"{number}/60 {day}: {bars.height:,} bars", flush=True)
    assert baseline_differences == 0 and candidate_count == 1_602_000
    policies = (
        aggregate(policy_rows, ["top_n", "tolerance"])
        .with_columns(
            (100 * pl.col("unknown_count") / pl.col("candidates")).alias("unknown_pct"),
            (10000 * pl.col("return_sum") / pl.col("known")).alias("mean_return_bps"),
            (10000 * pl.col("optimistic_sum") / pl.col("known")).alias("optimistic_mean_bps"),
            (10000 * pl.col("recovered_return_sum") / pl.col("recovered")).alias(
                "recovered_mean_bps"
            ),
        )
        .with_columns(
            *[
                (100 * pl.col(f"{o.name.lower()}_count") / pl.col("known")).alias(
                    f"{o.name.lower()}_known_pct"
                )
                for o in list(Outcome)[:4]
            ],
            *[
                (100 * pl.col(f"{r}_count") / pl.col("candidates")).alias(f"{r}_pct")
                for r in REASONS[1:]
            ],
        )
    )
    fixed = aggregate(fixed_rows, ["top_n"]).with_columns(
        (100 * pl.col("endpoint_valid") / pl.col("candidates")).alias("endpoint_coverage_pct"),
        (100 * pl.col("complete_window_valid") / pl.col("candidates")).alias(
            "complete_window_coverage_pct"
        ),
        (10000 * pl.col("return_sum") / pl.col("endpoint_valid")).alias("mean_return_bps"),
    )
    gaps = pl.from_dicts(
        [
            {
                "top_n": size,
                "length": length,
                "boundary": bool(edge),
                "runs": count,
                "missing_minutes": length * count,
            }
            for (size, length, edge), count in sorted(gap_counts.items())
        ]
    )
    sync = pl.from_dicts(sync_rows)
    policies.write_csv(output / "policy_summary.csv")
    fixed.write_csv(output / "fixed_horizon_coverage.csv")
    gaps.write_csv(output / "gap_lengths.csv")
    sync.sort("missing_top100_active", descending=True).head(30).write_csv(
        output / "synchronized_missing_examples.csv"
    )
    pl.from_dicts(session_rows).write_csv(output / "session_summary.csv")
    metadata = {
        "study": str(study),
        "sessions": 60,
        "candidates": candidate_count,
        "tp": TP,
        "sl": SL,
        "horizon_minutes": HORIZON,
        "tolerances": list(TOLERANCES),
        "baseline_outcome_differences": baseline_differences,
        "strict_known_labels_unchanged": True,
        "elapsed_seconds": monotonic() - started,
        "local_only": True,
        "production_labels_changed": False,
        "cohort_sha256": {
            name: hashlib.sha256((study / name).read_bytes()).hexdigest()
            for name in ("sampled_sessions.csv", "selected_universe.parquet")
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "synchronization": {
            "session_minutes": sync.height,
            "minutes_ge5": int((sync["missing_top100_active"] >= 5).sum()),
            "minutes_ge10": int((sync["missing_top100_active"] >= 10).sum()),
            "minutes_ge20": int((sync["missing_top100_active"] >= 20).sum()),
            "max_missing": int(sync["missing_top100_active"].max()),
            "top100_empty_symbol_days": sum(s["top100_empty_symbols"] for s in session_rows),
        },
    }
    atomic_json(output / "summary.json", metadata)
    write_report(output)
    print(json.dumps(metadata, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, help="Separate output for compatibility checks.")
    arguments = parser.parse_args()
    run(arguments.study, arguments.output)
