"""Session checkpoints and bounded exploratory analysis of compact first-touch evidence."""

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import psutil
import yaml

from dead_reckoning.config import Config
from dead_reckoning.labels.barriers import Touches, first_touches, valid_ohlc
from dead_reckoning.research.data import (
    decision_grid,
    inventory,
    minute_paths,
    prior_universe,
    query_connection,
    read_minutes,
    sampled_sessions,
)
from dead_reckoning.storage.parquet import atomic_json, read_metadata

log = logging.getLogger(__name__)


def memory_mb() -> float:
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss) / 1024**2


def thresholds(options):
    return sorted({p.tp for p in options.pairs}), sorted({p.sl for p in options.pairs})


def pack(touches: Touches, options) -> dict:
    upper, lower = thresholds(options)
    out = {"first_bad": touches.first_bad}
    for i, t in enumerate(upper):
        out[f"u{i}"] = touches.upper[t]
        out[f"uo{i}"] = touches.upper_open[t]
    for i, t in enumerate(lower):
        out[f"d{i}"] = touches.lower[t]
        out[f"do{i}"] = touches.lower_open[t]
        out[f"sr{i}"] = touches.stop_return[t]
    for h in options.horizons:
        out[f"f{h}"] = touches.forward[h]
    return out


def unpack(frame: pl.DataFrame, options) -> Touches:
    upper, lower = thresholds(options)
    a = lambda name: frame[name].to_numpy()  # noqa: E731
    return Touches(
        a("first_bad"),
        {t: a(f"u{i}") for i, t in enumerate(upper)},
        {t: a(f"d{i}") for i, t in enumerate(lower)},
        {t: a(f"uo{i}") for i, t in enumerate(upper)},
        {t: a(f"do{i}") for i, t in enumerate(lower)},
        {t: a(f"sr{i}") for i, t in enumerate(lower)},
        {h: a(f"f{h}") for h in options.horizons},
    )


def session_evidence(bars, members, session, options):
    opening, closing = session["open"], session["close"]
    minutes = int((closing - opening).total_seconds() / 60)
    decisions, starts = decision_grid(minutes, options)
    if len(starts) == 0:
        raise ValueError(f"No common-cohort entries fit session {session['date']}")
    opening_us = int(opening.timestamp()) * 1_000_000
    upper, lower = thresholds(options)
    groups = {g["symbol"][0]: g for g in bars.partition_by("symbol")}
    quality = {
        "session": str(session["date"]),
        "minute_rows": bars.height,
        "expected_symbol_minutes": minutes * members.height,
        "missing_minutes": 0,
        "invalid_ohlc_rows": 0,
        "duplicate_rows": 0,
        "unaligned_rows": 0,
        "extreme_discontinuities": 0,
        "empty_symbols": 0,
        "candidate_entries": len(starts) * members.height,
        "decisions_per_symbol": len(starts),
        "session_minutes": minutes,
        "first_decision_et": (opening + timedelta(minutes=int(decisions[0])))
        .astimezone(ZoneInfo("America/New_York"))
        .isoformat(),
        "last_decision_et": (opening + timedelta(minutes=int(decisions[-1])))
        .astimezone(ZoneInfo("America/New_York"))
        .isoformat(),
        "examples": [],
    }
    frames = []
    for member in members.iter_rows(named=True):
        symbol = member["symbol"]
        prices = np.full((minutes, 4), np.nan)
        good = np.zeros(minutes, dtype=bool)
        group = groups.get(symbol)
        if group is None:
            quality["empty_symbols"] += 1
            quality["missing_minutes"] += minutes
        else:
            stamp = group["timestamp"].cast(pl.Int64).to_numpy()
            offset = stamp - opening_us
            aligned = offset % 60_000_000 == 0
            index = (offset // 60_000_000).astype(int)
            values = group.select("open", "high", "low", "close").to_numpy()
            valid = valid_ohlc(values)
            quality["invalid_ohlc_rows"] += int((~valid).sum())
            quality["unaligned_rows"] += int((~aligned).sum())
            counts = np.bincount(index, minlength=minutes)
            quality["duplicate_rows"] += int(np.maximum(counts - 1, 0).sum())
            quality["missing_minutes"] += int((counts == 0).sum())
            prices[index] = values
            good[index] = valid & aligned
            good[counts > 1] = False
            # Report consecutive-minute >20% open/previous-close moves; no outcome-based filter.
            with np.errstate(invalid="ignore", divide="ignore"):
                jumps = (
                    np.flatnonzero(
                        (np.abs(prices[1:, 0] / prices[:-1, 3] - 1) > 0.20) & good[1:] & good[:-1]
                    )
                    + 1
                )
            quality["extreme_discontinuities"] += len(jumps)
            for k in jumps[:3]:
                if len(quality["examples"]) < 20:
                    quality["examples"].append(
                        {
                            "symbol": symbol,
                            "minute": int(k),
                            "open": float(prices[k, 0]),
                            "previous_close": float(prices[k - 1, 3]),
                        }
                    )
        t = first_touches(prices, starts, options.horizons, upper, lower, good)
        n = len(starts)
        frame = pl.DataFrame(
            {
                "symbol": [symbol] * n,
                "date": [session["date"]] * n,
                "year": [str(session["date"].year)] * n,
                "decision_minute": decisions.astype(np.int16),
                "entry_minute": starts.astype(np.int16),
                "entry_timestamp": [opening + timedelta(minutes=int(k)) for k in starts],
                "decision_timestamp": [opening + timedelta(minutes=int(k)) for k in decisions],
                "entry_price": prices[starts, 0],
                "time_of_day": np.where(
                    decisions < 60,
                    "09:30-10:29",
                    np.where(decisions < 180, "10:30-12:29", "12:30+"),
                ),
                "liquidity": [member["liquidity_bucket"]] * n,
                "volatility": [member["volatility_bucket"]] * n,
                "liquidity_rank": [member["liquidity_rank"]] * n,
                **pack(t, options),
            }
        )
        frames.append(frame)
    return pl.concat(frames), quality


def run_study(config: Config, resume: Path | None = None) -> Path:
    # Lazy import keeps existing ingestion CLI independent of optional research dependencies.
    from dead_reckoning.research.metrics import aggregate
    from dead_reckoning.research.report import write_report

    started = monotonic()
    options = config.research
    base = options.output
    if not base.is_absolute():
        base = config.data.root / base
    output = resume or base / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6])
    snapshot = config.model_dump(mode="json")
    if resume:
        saved = yaml.safe_load((output / "config.yaml").read_text(encoding="utf-8"))
        if saved != snapshot:
            raise ValueError("Resume configuration differs from saved config.yaml")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.yaml").write_text(
            yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8"
        )
    (output / "touches").mkdir(exist_ok=True)
    (output / "session_quality").mkdir(exist_ok=True)
    sessions = sampled_sessions(options)
    sessions.write_csv(output / "sampled_sessions.csv")
    coverage, daily_paths, provenance = inventory(config.data.root, sessions)
    provenance.update(
        {
            "python": platform.python_version(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in [
                    "duckdb",
                    "polars",
                    "numpy",
                    "pyarrow",
                    "exchange-calendars",
                    "matplotlib",
                ]
            },
            "source_sha256": {
                str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in [
                    *Path(__file__).parent.glob("*.py"),
                    Path(__file__).parents[1] / "labels/barriers.py",
                    Path(__file__).parents[1] / "config.py",
                ]
            },
        }
    )
    if resume:
        previous = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
        if previous != provenance:
            raise ValueError("Source, dependency versions, or manifest changed; start a fresh run")
    else:
        atomic_json(output / "provenance.json", provenance)
    log.info("Research: %s sessions; output %s", sessions.height, output)
    with query_connection(config, output / "query_tmp") as db:
        universe_path = output / "selected_universe.parquet"
        if universe_path.exists():
            universe = pl.read_parquet(universe_path)
        else:
            universe = prior_universe(db, daily_paths, sessions, config)
            universe.write_parquet(universe_path, compression="zstd")
            atomic_json(
                output / "daily_support.json",
                db.sql("SELECT * FROM daily_support_quality").pl().row(0, named=True),
            )
        if set(universe["date"]) != set(sessions["date"]):
            raise ValueError("Some sampled sessions have no eligible historical securities")
        for number, session in enumerate(sessions.iter_rows(named=True), 1):
            batch_start = monotonic()
            day = session["date"]
            target = output / "touches" / f"{day}.parquet"
            quality_path = output / "session_quality" / f"{day}.json"
            if target.exists() and quality_path.exists():
                log.info("Research: checkpoint %s reused", day)
                continue
            members = universe.filter(pl.col("date") == day)
            symbols = members["symbol"].to_list()
            paths = minute_paths(coverage, config.data.root, day, symbols)
            metadata = read_metadata(Path(paths[0]))
            if any(
                metadata.get(k) != v
                for k, v in {
                    "feed": "sip",
                    "adjustment": "raw",
                    "asof": "-",
                    "timeframe": "1Min",
                }.items()
            ):
                raise ValueError(f"Unexpected minute conventions in {paths[0]}")
            bars = read_minutes(db, paths, symbols, session["open"], session["close"])
            frame, quality = session_evidence(bars, members, session, options)
            quality.update(
                {
                    "files_selected": len(paths),
                    "universe_symbols": len(symbols),
                    "manifest_covered_symbols": len(symbols),
                    "elapsed_seconds": monotonic() - batch_start,
                    "peak_process_mb": memory_mb(),
                    "checkpoint_rows": frame.height,
                }
            )
            temp = target.with_suffix(".partial")
            frame.write_parquet(temp, compression="zstd")
            os.replace(temp, target)
            atomic_json(quality_path, quality)
            log.info(
                "Research %s/%s %s: %s bars, %s entries, %.2fs, peak process %.0f MB",
                number,
                sessions.height,
                day,
                bars.height,
                frame.height,
                quality["elapsed_seconds"],
                quality["peak_process_mb"],
            )
        quality = collect_quality(output, universe, started)
        quality.update(json.loads((output / "daily_support.json").read_text(encoding="utf-8")))
        aggregate(db, output, options)
        quality["analysis_elapsed_seconds"] = monotonic() - started
        quality["peak_process_mb"] = memory_mb()
        atomic_json(output / "data_quality.json", quality)
        write_report(output, config, quality)
    quality["total_elapsed_seconds"] = monotonic() - started
    quality["peak_process_mb"] = memory_mb()
    quality["output_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    atomic_json(output / "data_quality.json", quality)
    log.info(
        "Research complete: %.1fs; peak process %.0f MB; %s",
        monotonic() - started,
        memory_mb(),
        output / "report.md",
    )
    return output


def collect_quality(output, universe, started):
    batches = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((output / "session_quality").glob("*.json"))
    ]
    fields = [
        "minute_rows",
        "expected_symbol_minutes",
        "missing_minutes",
        "invalid_ohlc_rows",
        "duplicate_rows",
        "unaligned_rows",
        "extreme_discontinuities",
        "empty_symbols",
        "candidate_entries",
        "universe_symbols",
        "manifest_covered_symbols",
    ]
    return {
        **{key: sum(b[key] for b in batches) for key in fields},
        "sessions": len(batches),
        "batches": batches,
        "extraction_batch_seconds": sum(b["elapsed_seconds"] for b in batches),
        "elapsed_before_aggregation_seconds": monotonic() - started,
        "prior_feature_date_violations": universe.filter(
            pl.col("feature_date") >= pl.col("date")
        ).height,
        "prior_volatility_over_20pct": universe.filter(pl.col("prior_volatility") > 0.2).height,
        "universe_definition": "prior 20 observations, no current-day existence filter; "
        "last observation within configured calendar-day staleness limit",
        "manifest_policy": "all sampled symbol-session requests must be complete; "
        "missing stored minutes are separate strict UNKNOWN evidence",
    }
