import json
from datetime import UTC, date, datetime

import duckdb
import polars as pl

from dead_reckoning.config import Credentials, load_config
from dead_reckoning.ingestion.bars import ingest_bars
from dead_reckoning.ingestion.manifest import Manifest
from dead_reckoning.market_data.alpaca import AlpacaClient, session_bounds
from dead_reckoning.storage.parquet import atomic_json
from dead_reckoning.universe.liquidity import (
    compare_universes,
    prior_liquidity_sql,
    session_dates,
)
from dead_reckoning.universe.reconciliation import reconcile_minutes


def test_prior_population_current_day_invariance_and_staleness(tmp_path):
    days = [date(2025, 1, d) for d in [2, 3, 6]]
    records = [
        {
            "symbol": symbol,
            "timestamp": datetime(day.year, day.month, day.day, 5, tzinfo=UTC),
            "close": 1.0,
            "volume": volume,
        }
        for symbol, volume in [("AAA", 100.0), ("BBB", 10.0)]
        for day in days
    ]
    records.append(
        {
            "symbol": "BBB",
            "timestamp": datetime(2025, 1, 7, 5, tzinfo=UTC),
            "close": 1.0,
            "volume": 1e12,
        }
    )
    path = tmp_path / "daily.parquet"
    pl.DataFrame(records).write_parquet(path)
    start, end = date(2025, 1, 2), date(2025, 1, 18)
    with duckdb.connect() as db:
        db.register("trading_sessions", session_dates(start, end).to_arrow())

        def calculate():
            return db.sql(
                prior_liquidity_sql(3, 3, 7),
                params={
                    "paths": [str(path)],
                    "end_boundary": session_bounds(start, end)[1],
                },
            ).pl()

        before = calculate()
        on_day = before.filter(pl.col("date") == date(2025, 1, 7))
        assert on_day.sort("liquidity_rank")["symbol"].to_list() == ["AAA", "BBB"]
        assert on_day["eligible"].to_list() == [True, True]
        assert on_day["feature_date"].to_list() == [date(2025, 1, 6)] * 2
        assert on_day["trailing_median_dollar_volume"].to_list() == [100.0, 10.0]
        # AAA qualifies without any current-day record. Adding huge same-day values
        # must not change any feature or rank for this session.
        records.append(
            {
                "symbol": "AAA",
                "timestamp": datetime(2025, 1, 7, 5, tzinfo=UTC),
                "close": 1e10,
                "volume": 1e20,
            }
        )
        pl.DataFrame(records).write_parquet(path)
        after = calculate()
        assert on_day.equals(after.filter(pl.col("date") == date(2025, 1, 7)))
    assert (
        before.filter((pl.col("symbol") == "AAA") & (pl.col("date") == date(2025, 1, 13))).height
        == 1
    )
    assert before.filter(
        (pl.col("symbol") == "AAA") & (pl.col("date") > date(2025, 1, 13))
    ).is_empty()
    assert date(2025, 1, 9) not in before["date"]  # Special XNYS closure.
    assert before.filter(pl.col("feature_date") >= pl.col("date")).is_empty()


def test_comparison_and_download_only_missing_intervals(config_path, market_server, tmp_path):
    config = load_config(config_path)
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    pl.DataFrame({"symbol": ["BBB", "CCC"]}).write_parquet(old / "symbols.parquet")
    pl.DataFrame({"symbol": ["AAA", "BBB"]}).write_parquet(new / "symbols.parquet")
    start, end = date(2025, 1, 2), date(2025, 1, 14)
    comparison = compare_universes(
        (old, {"generation": "old", "start": str(start), "end": str(end)}), new, start, end
    )
    assert (
        comparison["unchanged"],
        comparison["newly_included"],
        comparison["no_longer_required"],
    ) == (1, 1, 1)
    assert comparison["new_symbols"] == ["AAA"]
    assert comparison["removed_symbols"] == ["CCC"]
    assert comparison["same_date_range"]
    meta = {"generation": "new", "start": str(start), "end": str(end)}
    with Manifest(config.data.root) as manifest:
        client = AlpacaClient(Credentials(), config.ingestion)
        try:
            ingest_bars(config, manifest, client, "minute_bars", ["BBB"], start, end)
            ingest_bars(config, manifest, client, "minute_bars", ["AAA"], start, date(2025, 1, 10))
            count = len(market_server.requests)
            preview = reconcile_minutes(config, manifest, new, meta)
            assert len(market_server.requests) == count
            assert preview["coverage_before"]["newly_included_incomplete"] == 1
            result = reconcile_minutes(config, manifest, new, meta, client=client)
            requests = market_server.requests[count:]
            assert requests
            assert all(params["symbols"] == "AAA" for _, params in requests)
            assert all(params["start"].startswith("2025-01-10") for _, params in requests)
            assert result["coverage_after"]["incomplete_symbols"] == 0
            assert result["download"]["batches"] == 1
            assert (old / "symbols.parquet").is_file()
            count = len(market_server.requests)
            repeated = reconcile_minutes(config, manifest, new, meta, client=client)
            assert repeated["download"] is None
            assert len(market_server.requests) == count
            saved = json.loads((new / "reconciliation.json").read_text())
            assert saved["coverage_after"]["complete_symbols"] == 2
        finally:
            client.close()


def test_empty_union_needs_no_requests(config_path, tmp_path):
    config = load_config(config_path)
    pl.DataFrame(schema={"symbol": pl.String}).write_parquet(tmp_path / "symbols.parquet")
    atomic_json(tmp_path / "comparison.json", {"new_symbols": []})
    with Manifest(config.data.root) as manifest:
        result = reconcile_minutes(
            config,
            manifest,
            tmp_path,
            {"generation": "empty", "start": "2025-01-02", "end": "2025-01-03"},
        )
    assert result["coverage_after"]["required_symbols"] == 0
    assert result["coverage_after"]["incomplete_symbols"] == 0
