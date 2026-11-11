from datetime import UTC, date, datetime, timedelta

import polars as pl

from dead_reckoning.config import load_config
from dead_reckoning.ingestion.manifest import Manifest
from dead_reckoning.market_data.alpaca import BAR_SCHEMA
from dead_reckoning.storage.parquet import write_pages
from dead_reckoning.universe.liquidity import build_universe


def test_prior_window_union_and_future_invariance(config_path):
    config = load_config(config_path)
    days = [date(2025, 1, 2) + timedelta(days=index) for index in range(7)]

    def store(path, first, last, values):
        records = []
        for symbol, volumes in values.items():
            for index, volume in enumerate(volumes):
                records.append(
                    {
                        "symbol": symbol,
                        "timestamp": datetime.combine(days[index + first], datetime.min.time(), UTC)
                        + timedelta(hours=5),
                        "open": 2.0,
                        "high": 2.0,
                        "low": 2.0,
                        "close": 2.0,
                        "volume": volume,
                        "trade_count": 1,
                        "vwap": 2.0,
                    }
                )
        meta = {
            "schema_version": 1,
            "provider": "alpaca",
            "dataset": "daily_bars",
            "timeframe": "1Day",
            "feed": "sip",
            "adjustment": "raw",
            "asof": "-",
            "start": days[first].isoformat(),
            "end": (days[last] + timedelta(days=1)).isoformat(),
            "symbols": list(values),
            "ingested_at": "2025-02-01T00:00:00+00:00",
        }
        write_pages(path, [pl.from_dicts(records, schema=BAR_SCHEMA)], meta)

    folder = config.data.root / "raw/daily_bars"
    store(
        folder / "initial.parquet",
        0,
        5,
        {
            "AAA": [100, 80, 60, 0, 0, 0],
            "BBB": [10, 10, 10, 1000, 1000, 1000],
            "CCC": [1, 1, 1, 1, 1, 1],
        },
    )
    with Manifest(config.data.root) as manifest:
        output, _ = build_universe(config, manifest)
        before = pl.read_parquet(output / "daily_liquidity.parquet")
        union = pl.read_parquet(output / "symbols.parquet")
        assert union["symbol"].to_list() == ["AAA", "BBB"]
        row = before.filter((pl.col("symbol") == "AAA") & (pl.col("date") == days[3]))
        assert row["trailing_median_dollar_volume"].item() == 160.0
        assert row["history_days"].item() == 3
        assert row["liquidity_rank"].item() == 1
        # Today's BBB volume is huge, yet today's ranking uses only the prior 3 days.
        row = before.filter((pl.col("symbol") == "BBB") & (pl.col("date") == days[3]))
        assert row["trailing_median_dollar_volume"].item() == 20.0
        assert row["liquidity_rank"].item() == 2
        assert before.filter(pl.col("date") < days[3])["liquidity_rank"].null_count() == 9

    store(folder / "future.parquet", 6, 6, {"AAA": [1e12], "BBB": [1e15], "CCC": [1e18]})
    with Manifest(config.data.root) as manifest:
        output, _ = build_universe(config, manifest)
        after = pl.read_parquet(output / "daily_liquidity.parquet")
        assert before.equals(after.filter(pl.col("date") <= days[5]))
        # Filtering the result start must not discard earlier rolling warm-up observations.
        output, _ = build_universe(config, manifest, days[3], days[6])
        subset = pl.read_parquet(output / "daily_liquidity.parquet")
        assert subset.equals(before.filter(pl.col("date") >= days[3]))
