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
dead-reckoning data validate
```

`status` reports indexed download coverage. `validate` opens every raw bar file's
footer and checks schema, request metadata, row totals, missing/unindexed files,
and overlapping coverage. It also reads three whole nonempty files per dataset, selected
reproducibly across sorted paths, to check per-symbol counts, duplicate/out-of-order
timestamps, request boundaries, and invalid prices/volumes/OHLC relationships.
Use `--sample-files 10` for a larger sample or `--full` to read every bar row
(potentially hours of disk I/O). Failures return exit code 1. Validation uses no
credentials, makes no repairs, and cannot prove Alpaca supplied every expected bar.

Dates use New York sessions: start inclusive, end exclusive. Select another YAML
with `--config PATH` before the subcommand.

Timestamped throughput logs appear about every 30 seconds during ingestion and at
each stage's end. HTTP counts include retries and pagination; `total` spans the
client session, including asset requests. Current RPM estimates the last up to
60 seconds; average RPM, received bar rows, rows/sec, and elapsed time cover the
current daily/minute stage, including pacing, retries, and local writes. Received
rows precede storage deduplication. Compare sustained RPM with `target_rpm` before
considering download concurrency; short runs have a warm-up estimate.
The synchronous request pace defaults to 198/minute, just below the Basic plan's
200/minute allowance. Network and local processing time can keep actual RPM lower.

Bars use SIP, raw prices, and unmapped tickers (`asof=-`). The historical universe
contains every symbol that ever ranked in the top 2,000 by trailing 20-observation
median dollar volume. Each date's rank uses only prior observations, with at least
20 required; dates without enough prior history are ineligible. These thresholds
are configurable. Rankings are generated on the XNYS session calendar, without
requiring a current-day bar. A symbol's most recent prior daily observation must
be within `universe.max_staleness_days` (default seven calendar days), preventing
old observations from keeping discontinued symbols eligible indefinitely.
Each ranked row records its strictly earlier `feature_date`. Minute ingestion automatically
uses this union and downloads each symbol's full requested history.

To compare the corrected universe with the previous generation and fill only
missing minute request intervals:

```powershell
dead-reckoning universe reconcile --download-missing
```

Omit `--download-missing` for comparison and coverage reporting without API calls.
Optional `--start` and `--end` follow the same exclusive-end convention as `build`;
by default the existing daily request envelope is used. Each immutable generation
contains `comparison.json`, `symbol_changes.csv`, `minute_coverage.csv`, and
`reconciliation.json`. The comparison identifies unchanged, added, and removed
symbols and explicitly records whether the date ranges match. Previous generations
and raw data remain available. Repeating the command reuses the current generation
and downloads no already-complete intervals. Completed intervals may contain
missing minutes or be empty; this command does not force-refresh those intervals.

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

Daily ingestion saves its intended symbols and dates before downloading. Universe
builds reject missing coverage, including symbols never attempted; completed empty
responses count as covered. Explicit `--symbols` runs retain their selected scope.
For older data without a saved scope, the asset snapshots supply the broad population;
rerun the scoped daily command to declare an older smoke/debug dataset's scope.

Alpaca's asset snapshots are not a complete historical security master;
survivorship and ticker reuse remain limitations. SIP availability depends on
account access. Completed intervals are not automatically refreshed for vendor corrections.
SIP candidates exclude asset records with `exchange: OTC`; active/tradable status
is not used as a filter. Bar discovery retains inactive tickers, uppercases symbols,
and maps `_DELISTED`
aliases to their underlying tickers. Non-ticker reference identifiers are excluded
from bar requests; their original asset records remain in the raw snapshots.

## Exploratory label research

Install the optional local research dependencies, run a small offline study, then the
default 2019–2023 sample:

```powershell
pip install -e ".[dev,research]"
dead-reckoning research labels --config configs/label_study_v1.yaml --smoke
dead-reckoning --config configs/label_study_v1.yaml research labels
```

Both placements of `--config` are supported for this command. Set `data.root` in
the selected YAML to reuse another data location. No API credentials are needed.
The default samples one reproducibly random XNYS session per month, 500 stocks
ranked by prior 20-observation median dollar volume, and a common five-minute
entry grid with one-minute execution delay and room for the full 120-minute
window. Research rejects end dates after 2024-01-01 to preserve the holdout.
`--smoke` caps the universe at 25 and selects two calendar-spread sampled sessions;
the other label definitions and all 25 policies remain the same.

Edit `research` YAML settings for dates, seed, sessions per month, session cap,
universe size, decision grid, delay, barrier pairs, horizons, cost assumptions,
output base, and DuckDB resources. Paths under `research.output` are relative to
`data.root`, unless absolute. Default results are in
`<data.root>/research/label_study/<run_id>/`, including `report.md`, five PNG charts,
configuration, session dates, exact summary/segmentation/cost/fixed-horizon CSVs,
data-quality measurements, provenance, selected universe, and compact first-touch
checkpoints. Original minute bars are neither copied nor changed.

To reuse completed session batches after interruption, use identical settings:

```powershell
dead-reckoning --config configs/label_study_v1.yaml research labels --resume <run-directory>
```

Include `--smoke` when resuming a smoke run. Resume rejects changed source,
dependency versions, manifest, or configuration, and recalculates aggregates.

The reusable engine is `dead_reckoning.labels.barriers`. It retains TP, SL,
expiration, ambiguity, and UNKNOWN evidence. Opens precede intrabar high/low checks;
adverse stop gaps fill at the open, favorable TP gaps at the threshold. Missing or
invalid minutes before an exit yield UNKNOWN; gaps after a resolved exit do not.
No bars are forward-filled. Conservative ambiguity assigns SL, with optimistic
and excluded sensitivity metrics. Fixed-horizon targets require the entire path.
Intrabar holding times use minute end as an explicit approximation.

Research independently reconstructs eligibility from strictly prior daily observations,
using the same prior-only population convention as the corrected ingestion universe.
It requires a last daily observation within seven calendar days, configurable in
YAML, and retains absent-on-day securities as candidates. This avoids using future
bar availability for eligibility; raw prices and the incomplete historical asset
master still limit corporate-action and survivorship accuracy. No post-2023
outcomes, model training, portfolio backtest, or policy optimization are involved.

## Training dataset foundation

Candidate and label generation are independent offline stages. The default dataset
grid keeps every regular-session five-minute decision; each target separately checks
whether its holding window fits. Labels use a configurable two-minute interior-gap
tolerance, genuine entry/expiration bars, observed opening-gap execution and preserved
ambiguity. Independent endpoint-only 15/30/60/90/120-minute returns remain usable
when intermediate gaps invalidate barrier labels.

```powershell
dead-reckoning --config configs/dataset_v1.yaml dataset candidates --output-id foundation_2019_v2
dead-reckoning --config configs/dataset_v1.yaml dataset labels --candidate-id foundation_2019_v2
dead-reckoning --config configs/dataset_v1.yaml dataset status
```

These defaults build only January 2019, with no downloads. Configuration fingerprints,
daily atomic Parquet partitions and manifests support resume, additional date ranges,
selected partition rebuilds and independent label policies. Future feature groups
join using the same symbol/UTC decision keys. Features, assembly and training are not
implemented here. See [architecture, schemas and command examples](docs/datasets.md).

## Development

```powershell
pytest
ruff check .
ruff format --check .
pytest tests/test_pipeline.py -v -s
```

Install `.[dev,research]` to run all research and ingestion tests. Pytest runs offline
using synthetic price paths and HTTP responses. Research tests compare the optimized
first-touch engine with an independent minute-loop reference, exercise point-in-time
eligibility and calendar boundaries, and generate a complete synthetic report.
To check real Alpaca credentials,
SIP access, storage, and rerun skips, run the live smoke test:

```powershell
dead-reckoning data smoke
```

This uses AAPL/MSFT for January 2–February 7, 2025, with the configured universe
rules. It prints a separate temporary data directory, validates stored bars and
coverage, and reruns ingestion to confirm zero new bar batches. Data remains there
for inspection; the configured research data root is unchanged.
