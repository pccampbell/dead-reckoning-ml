from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from dead_reckoning.cli import app
from dead_reckoning.config import Config, LabelStudyConfig
from dead_reckoning.ingestion.manifest import Manifest
from dead_reckoning.research.data import prior_universe, query_connection, sampled_sessions
from dead_reckoning.research.study import run_study, session_evidence
from dead_reckoning.storage.parquet import write_pages


def store(root, dataset, frame, start, end, symbols):
    path = root / "raw" / dataset / "synthetic.parquet"
    write_pages(
        path,
        [frame],
        {
            "schema_version": 1,
            "provider": "alpaca",
            "dataset": dataset,
            "feed": "sip",
            "adjustment": "raw",
            "asof": "-",
            "timeframe": "1Day" if dataset == "daily_bars" else "1Min",
            "start": str(start),
            "end": str(end),
            "symbols": symbols,
            "ingested_at": "2026-01-01T00:00:00+00:00",
        },
    )
    return path


@pytest.fixture
def study(tmp_path):
    config = Config.model_validate(
        {
            "data": {"root": tmp_path},
            "research": {
                "start": "2019-11-29",
                "end": "2019-11-30",
                "universe_size": 2,
                "horizons": [15, 60, 120],
                "pairs": [{"name": "original", "tp": 0.005, "sl": 0.004}],
                "memory_limit": "128MB",
                "threads": 1,
            },
        }
    )
    session = sampled_sessions(config.research).row(0, named=True)
    day = session["date"]
    daily = []
    for symbol, volume in [("AAA", 10000.0), ("BBB", 1000.0)]:
        for k in range(35):
            prior = day - timedelta(days=35 - k)
            daily.append(
                {
                    "symbol": symbol,
                    "timestamp": datetime.combine(prior, datetime.min.time(), UTC)
                    + timedelta(hours=5),
                    "close": 100 + k,
                    "volume": volume,
                }
            )
    daily_path = store(
        tmp_path,
        "daily_bars",
        pl.from_dicts(daily),
        day - timedelta(days=50),
        day + timedelta(days=1),
        ["AAA", "BBB"],
    )
    bars = []
    for symbol in ["AAA", "BBB"]:
        for minute in range(210):
            if symbol == "BBB" and minute == 5:
                continue
            bars.append(
                {
                    "symbol": symbol,
                    "timestamp": session["open"] + timedelta(minutes=minute),
                    "open": 100.0,
                    "high": 100.1,
                    "low": 99.9,
                    "close": 100.0,
                }
            )
    store(
        tmp_path, "minute_bars", pl.from_dicts(bars), day, day + timedelta(days=1), ["AAA", "BBB"]
    )
    with Manifest(tmp_path):
        pass
    return config, daily_path, session


def test_prior_population_absence_and_current_future_invariance(study, tmp_path):
    config, path, session = study
    sessions = sampled_sessions(config.research)
    with query_connection(config, tmp_path / "temp") as db:
        before = prior_universe(db, [str(path)], sessions, config)
        assert before["symbol"].to_list() == ["AAA", "BBB"]
        assert (before["feature_date"] < session["date"]).all()
        expected = np.std(np.log(np.arange(115, 135) / np.arange(114, 134)), ddof=1)
        assert before["prior_volatility"][0] == pytest.approx(expected)
        frame = pl.read_parquet(path)
        additions = pl.DataFrame(
            {
                "symbol": ["BBB", "AAA"],
                "timestamp": [session["open"], session["open"] + timedelta(days=1)],
                "close": [1e9, 1e9],
                "volume": [1e15, 1e15],
            }
        ).cast(frame.schema)
        changed = tmp_path / "changed.parquet"
        pl.concat([frame, additions]).write_parquet(changed)
        after = prior_universe(db, [str(changed)], sessions, config)
        assert before.equals(after)


def test_offline_end_to_end_checkpoint_report_and_denominators(study):
    config, _, _ = study
    output = run_study(config)
    summary = pl.read_csv(output / "summary.csv")
    assert summary.height == 3
    assert summary["candidates"].to_list() == [36] * 3
    assert summary["unknown_count"].to_list() == [1] * 3
    assert summary["expire_count"].to_list() == [35] * 3
    assert summary["valid"].to_list() == [35] * 3
    assert summary["gross_bps_mean"].to_list() == [0.0] * 3
    assert summary["expire_pct"][0] == pytest.approx(3500 / 36)
    assert pl.read_csv(output / "fixed_horizon_summary.csv")["valid"].to_list() == [35] * 3
    assert (output / "report.md").is_file()
    assert len(list((output / "charts").glob("*.png"))) == 5
    before = [(p, p.stat().st_mtime_ns) for p in (output / "touches").glob("*.parquet")]
    run_study(config, resume=output)
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in before)
    changed = config.model_copy(deep=True)
    changed.research.seed += 1
    with pytest.raises(ValueError, match="configuration differs"):
        run_study(changed, resume=output)


def test_empty_symbol_duplicate_and_off_grid_quality(study):
    config, _, session = study
    members = pl.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "liquidity_rank": [1, 2],
            "liquidity_bucket": ["top", "lower"],
            "volatility_bucket": ["low", "high"],
        }
    )
    bars = pl.DataFrame(
        {
            "symbol": ["AAA"] * 3,
            "timestamp": [session["open"] + timedelta(minutes=1)] * 2
            + [session["open"] + timedelta(minutes=2, seconds=1)],
            "open": [100.0] * 3,
            "high": [100.0] * 3,
            "low": [100.0] * 3,
            "close": [100.0] * 3,
        }
    )
    evidence, quality = session_evidence(bars, members, session, config.research)
    assert quality["empty_symbols"] == 1
    assert quality["duplicate_rows"] == 1
    assert quality["unaligned_rows"] == 1
    assert evidence["first_bad"][0] == 0


def test_reserved_dates_validation_and_cli(tmp_path):
    with pytest.raises(ValidationError, match="reserved holdout"):
        LabelStudyConfig(end=date(2025, 1, 1))
    path = tmp_path / "config.yaml"
    path.write_text(f'data:\n  root: "{tmp_path.as_posix()}"\n')
    result = CliRunner().invoke(app, ["--config", str(path), "research", "labels", "--smoke"])
    assert result.exit_code == 1
    assert "No ingestion manifest" in result.output
