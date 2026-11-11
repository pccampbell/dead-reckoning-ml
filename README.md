# dead-reckoning-ml

Local-first research project for ML-based equity trading. Currently provides
Alpaca historical data ingestion and liquidity-based symbol selection.

## Setup

Use Python 3.12.x. Check that `python --version` reports 3.12 before installing:

```powershell
python --version
pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in `.env` or environment variables.
Use paper-account keys for the asset endpoint.
Configure `data.root` in `configs/data.yaml` for an external data directory;
the default is `~/dead-reckoning-data`. Credentials are never stored in YAML.

`query.memory_limit` and `query.threads` control DuckDB's liquidity calculations;
the defaults are 8GB and 8 threads. These do not control download concurrency or
cap total process memory. Large query intermediates can spill to
`<data.root>/metadata/query_tmp`; use a fast local drive for the data root.

## Historical data pipeline

Run the complete pipeline:

```powershell
dead-reckoning data bootstrap --start 2016-01-01 --end 2026-11-01
```

Or run individual stages:

```powershell
dead-reckoning ingest assets
dead-reckoning ingest daily-bars --start 2016-01-01 --end 2026-11-01
dead-reckoning universe build
dead-reckoning ingest minute-bars --start 2016-01-01 --end 2026-11-01
dead-reckoning data status
```

Dates use New York sessions: start inclusive, end exclusive. Select another YAML
with `--config PATH` before the subcommand.

Bars use SIP, raw prices, and unmapped tickers (`asof=-`). The historical universe
contains every symbol that ever ranked in the top 2,000 by trailing 20-observation
median dollar volume. Each date's rank uses only prior observations, with at least
20 required; dates without enough prior history are ineligible. These thresholds
are configurable. Minute ingestion automatically
uses this union and downloads each symbol's full requested history.

## Storage

All datasets live under `data.root`:

```text
raw/assets/              Asset snapshots, including available inactive assets
raw/daily_bars/          Daily SIP bars
raw/minute_bars/         One-minute SIP bars
curated/universe/        Daily liquidity metrics and historical symbol union
metadata/ingestion.duckdb
```

Raw ZSTD Parquet is the durable source of truth. Atomic writes and per-symbol
coverage tracking make downloads resumable; reruns fetch only missing intervals.
Curated artifacts and the coverage index can be rebuilt from raw data.

Alpaca's asset snapshots are not a complete historical security master;
survivorship and ticker reuse remain limitations. SIP availability depends on
account access. Completed intervals are not automatically refreshed for vendor corrections.

## Development

```powershell
pytest
ruff check .
ruff format --check .
pytest tests/test_pipeline.py -v -s
```

Pytest runs offline using synthetic HTTP responses. To check real Alpaca credentials,
SIP access, storage, and rerun skips, run the live smoke test:

```powershell
dead-reckoning data smoke
```

This uses AAPL/MSFT for January 2–February 7, 2025, with the configured universe
rules. It prints a separate temporary data directory, validates stored bars and
coverage, and reruns ingestion to confirm zero new bar batches. Data remains there
for inspection; the configured research data root is unchanged.
