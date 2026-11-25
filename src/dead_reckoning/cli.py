"""Small synchronous CLI; all actual ingestion lives in reusable modules."""

import logging
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic
from typing import Annotated

import duckdb
import polars as pl
import typer
from pydantic import ValidationError

from dead_reckoning.config import Config, Credentials, load_config, validate_range
from dead_reckoning.ingestion.assets import asset_symbols, ingest_assets
from dead_reckoning.ingestion.bars import ingest_bars
from dead_reckoning.ingestion.manifest import Manifest, missing_intervals
from dead_reckoning.ingestion.smoke import SMOKE_END, SMOKE_START, SMOKE_SYMBOLS, run_smoke
from dead_reckoning.ingestion.validation import validate_data
from dead_reckoning.market_data.alpaca import AlpacaClient, stock_bar_symbol
from dead_reckoning.universe.liquidity import (
    build_universe,
    current_universe,
    universe_symbols,
)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
ingest = typer.Typer(no_args_is_help=True)
universe = typer.Typer(no_args_is_help=True)
data = typer.Typer(no_args_is_help=True)
research = typer.Typer(no_args_is_help=True)
dataset = typer.Typer(no_args_is_help=True)
app.add_typer(ingest, name="ingest")
app.add_typer(universe, name="universe")
app.add_typer(data, name="data")
app.add_typer(research, name="research")
app.add_typer(dataset, name="dataset")
ConfigOption = Annotated[Path, typer.Option("--config", help="Non-secret YAML configuration.")]
StartOption = Annotated[str | None, typer.Option(help="YYYY-MM-DD; defaults to data.start_date.")]
EndOption = Annotated[str, typer.Option(help="Exclusive New York session date, YYYY-MM-DD.")]
SymbolsOption = Annotated[str | None, typer.Option(help="Optional daily subset, e.g. AAPL,MSFT.")]


@contextmanager
def friendly_errors():
    try:
        yield
    except (
        ValueError,
        OSError,
        RuntimeError,
        ValidationError,
        duckdb.Error,
        pl.exceptions.PolarsError,
    ) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.callback()
def main(ctx: typer.Context, config: ConfigOption = Path("configs/data.yaml")):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
        force=True,
    )
    with friendly_errors():
        ctx.obj = load_config(config)


@contextmanager
def client_for(config: Config):
    client = AlpacaClient(Credentials(), config.ingestion)
    try:
        yield client
    finally:
        client.close()


def dates(config: Config, start: str | None, end: str) -> tuple[date, date]:
    lower = date.fromisoformat(start) if start else config.data.start_date
    upper = date.fromisoformat(end)
    validate_range(lower, upper)
    return lower, upper


def daily_symbols(config: Config, symbols: str | None) -> list[str]:
    if symbols is None:
        return asset_symbols(config.data.root)
    selected = sorted({symbol.strip().upper() for symbol in symbols.split(",") if symbol.strip()})
    if not selected:
        raise ValueError("--symbols must contain at least one symbol")
    return selected


def print_result(stage: str, result: dict, start: date, end: date) -> None:
    typer.echo(
        f"{stage} complete [{start}, {end}): "
        f"{result['batches']:,} new request batches; {result['rows']:,} new rows; "
        f"{result['skipped_symbols']:,} symbols already covered."
    )


@ingest.command("assets")
def assets(ctx: typer.Context, refresh: bool = False):
    """Store all available U.S. equity assets, including inactive assets."""
    config: Config = ctx.obj
    with friendly_errors(), Manifest(config.data.root), client_for(config) as client:
        ingest_assets(config.data.root, client, refresh)


@ingest.command("daily-bars")
def daily_bars(
    ctx: typer.Context, end: EndOption, start: StartOption = None, symbols: SymbolsOption = None
):
    """Download broad daily SIP history, or an explicitly limited smoke subset."""
    config: Config = ctx.obj
    with friendly_errors():
        lower, upper = dates(config, start, end)
        selected = daily_symbols(config, symbols)
        with Manifest(config.data.root) as manifest, client_for(config) as client:
            print_result(
                "Daily bars",
                ingest_bars(config, manifest, client, "daily_bars", selected, lower, upper),
                lower,
                upper,
            )


@universe.command("build")
def build(ctx: typer.Context, start: str | None = None, end: str | None = None):
    """Persist point-in-time daily liquidity and its historical qualifying union."""
    config: Config = ctx.obj
    with friendly_errors(), Manifest(config.data.root) as manifest:
        _, meta = build_universe(
            config,
            manifest,
            date.fromisoformat(start) if start else None,
            date.fromisoformat(end) if end else None,
        )
        typer.echo(f"Historical universe: {meta['symbol_count']} symbols; {meta['criterion']}")


@universe.command("reconcile")
def reconcile_universe(
    ctx: typer.Context,
    start: str | None = None,
    end: str | None = None,
    download_missing: Annotated[
        bool, typer.Option(help="Fetch only missing minute request intervals; needs credentials.")
    ] = False,
):
    """Rebuild prior-only ranks, compare unions, and optionally fill missing minute coverage."""
    from dead_reckoning.universe.reconciliation import reconcile_minutes

    config: Config = ctx.obj
    with friendly_errors(), Manifest(config.data.root) as manifest:
        folder, meta = build_universe(
            config,
            manifest,
            date.fromisoformat(start) if start else None,
            date.fromisoformat(end) if end else None,
        )
        result = reconcile_minutes(config, manifest, folder, meta)
        if download_missing and result["coverage_after"]["incomplete_symbols"]:
            with client_for(config) as client:
                result = reconcile_minutes(config, manifest, folder, meta, client=client)
        comparison = result["comparison"]
        typer.echo(
            f"Universe: {comparison['previous_count']:,} -> {comparison['corrected_count']:,}; "
            f"{comparison['unchanged']:,} unchanged, {comparison['newly_included']:,} added, "
            f"{comparison['no_longer_required']:,} no longer required."
        )
        coverage = result["coverage_after"]
        typer.echo(
            f"Minute request coverage: {coverage['complete_symbols']:,}/"
            f"{coverage['required_symbols']:,} complete; "
            f"{coverage['incomplete_symbols']:,} pending."
        )
        typer.echo(f"Comparison and coverage: {folder / 'reconciliation.json'}")


def download_minutes(
    config: Config, manifest: Manifest, client: AlpacaClient, lower: date, upper: date
) -> None:
    folder, _ = build_universe(config, manifest, lower, upper)
    selected = universe_symbols(folder)
    if not selected:
        raise ValueError(
            "Historical universe is empty; ingest more daily history or check criteria"
        )
    print_result(
        "Minute bars",
        ingest_bars(config, manifest, client, "minute_bars", selected, lower, upper),
        lower,
        upper,
    )


@ingest.command("minute-bars")
def minute_bars(ctx: typer.Context, end: EndOption, start: StartOption = None):
    """Download full requested history for the automatically derived historical union."""
    config: Config = ctx.obj
    with friendly_errors():
        lower, upper = dates(config, start, end)
        with Manifest(config.data.root) as manifest, client_for(config) as client:
            download_minutes(config, manifest, client, lower, upper)


@data.command("bootstrap")
def bootstrap(
    ctx: typer.Context,
    end: EndOption,
    start: StartOption = None,
    symbols: SymbolsOption = None,
    refresh_assets: bool = False,
):
    """Assets -> daily bars -> historical universe -> minute bars, without manual steps."""
    config: Config = ctx.obj
    with friendly_errors():
        started = monotonic()
        lower, upper = dates(config, start, end)
        logging.info("Bootstrap starting [%s, %s); data root: %s", lower, upper, config.data.root)
        with Manifest(config.data.root) as manifest, client_for(config) as client:
            ingest_assets(config.data.root, client, refresh_assets)
            print_result(
                "Daily bars",
                ingest_bars(
                    config,
                    manifest,
                    client,
                    "daily_bars",
                    daily_symbols(config, symbols),
                    lower,
                    upper,
                ),
                lower,
                upper,
            )
            download_minutes(config, manifest, client, lower, upper)
        typer.echo(
            f"Bootstrap completed successfully for {lower} to {upper} (end exclusive). "
            "Assets, daily bars, historical universe, and minute bars are ready. "
            f"Elapsed: {timedelta(seconds=int(monotonic() - started))}."
        )
        typer.echo(f"Data root: {config.data.root}")


@data.command("smoke")
def smoke(ctx: typer.Context):
    """Run a small LIVE Alpaca check in a separate temporary data root; requires credentials."""
    with friendly_errors():
        config: Config = ctx.obj.model_copy(deep=True)
        config.data.root = Path(mkdtemp(prefix="dead-reckoning-smoke-"))
        typer.echo(f"Live smoke: {','.join(SMOKE_SYMBOLS)}, [{SMOKE_START}, {SMOKE_END})")
        typer.echo(f"Smoke data root: {config.data.root}")
        with Manifest(config.data.root) as manifest, client_for(config) as client:
            result = run_smoke(config, manifest, client)
        ctx.obj = config
        status(ctx)
        typer.echo(
            f"Live smoke passed: {result['daily_rows']:,} daily rows, "
            f"{result['minute_rows']:,} minute rows, "
            f"{result['universe_symbols']} universe symbols; "
            "rerun fetched zero bar batches."
        )


@data.command("validate")
def validate(
    ctx: typer.Context,
    full: Annotated[
        bool, typer.Option(help="Read every bar row; potentially hours of disk I/O.")
    ] = False,
    sample_files: Annotated[
        int, typer.Option(min=1, help="Nonempty files per dataset to scan.")
    ] = 3,
):
    """Check stored bars against the manifest; no API calls or repairs."""
    with friendly_errors():
        result = validate_data(ctx.obj.data.root, sample_files=sample_files, full=full)
    for summary in result["summaries"]:
        typer.echo(summary)
    for issue in result["issues"][:20]:
        typer.echo(f"FAIL: {issue}", err=True)
    if result["issues"]:
        typer.echo(
            f"Validation failed: {len(result['issues']):,} issues (first 20 shown).", err=True
        )
        raise typer.Exit(1)
    typer.echo(f"Validation passed ({'full row scan' if full else 'sampled row scan'}).")
    typer.echo("Checks storage integrity and bar consistency; cannot prove vendor completeness.")


@data.command("status")
def status(ctx: typer.Context):
    """Show stored request coverage, row counts, and pending universe symbols."""
    config: Config = ctx.obj
    with friendly_errors(), Manifest(config.data.root) as manifest:
        root = config.data.root
        typer.echo(f"Data root: {root}")
        snapshots = list((root / "raw/assets").glob("*.parquet"))
        typer.echo(
            f"Assets: {len(asset_symbols(root))} symbols in {len(snapshots)} snapshots"
            if snapshots
            else "Assets: pending"
        )
        for dataset, title in (("daily_bars", "Daily bars"), ("minute_bars", "Minute bars")):
            symbols, rows, start, end, empty = manifest.summary(dataset)
            typer.echo(
                f"{title}: {symbols} queried symbols; {rows:,} rows; "
                f"request envelope [{start}, {end}); {empty} empty symbol intervals"
            )
            if start and end:
                coverage = manifest.coverage(dataset)
                complete = sum(
                    not missing_intervals(start, end, intervals) for intervals in coverage.values()
                )
                typer.echo(f"  Fully covered over envelope: {complete}/{symbols} queried symbols")
        current = current_universe(root)
        if current:
            folder, meta = current
            from dead_reckoning.universe.liquidity import UNIVERSE_ALGORITHM

            stale = (
                meta.get("algorithm") != UNIVERSE_ALGORITHM
                or meta["daily_fingerprint"] != manifest.fingerprint("daily_bars")
                or any(
                    meta.get(key) != value for key, value in config.universe.model_dump().items()
                )
            )
            typer.echo(
                f"Historical universe: {meta['symbol_count']} symbols; {meta['criterion']}"
                + (" (stale; minute ingestion will rebuild)" if stale else "")
            )
            start, end = date.fromisoformat(meta["start"]), date.fromisoformat(meta["end"])
            coverage = manifest.coverage("minute_bars")
            selected = universe_symbols(folder)
            complete = sum(
                not missing_intervals(start, end, coverage.get(symbol, [])) for symbol in selected
            )
            typer.echo(
                f"  Minute coverage for union [{start}, {end}): "
                f"{complete} completed; {len(selected) - complete} pending symbols"
            )
        else:
            typer.echo("Historical universe: pending")
        unfinished_attempts = manifest.db.execute(
            "SELECT symbol FROM coverage WHERE status != 'complete'"
        ).fetchall()
        unfinished = sum(stock_bar_symbol(symbol) is not None for (symbol,) in unfinished_attempts)
        excluded = len(unfinished_attempts) - unfinished
        if excluded:
            typer.echo(f"Excluded reference-identifier attempts retained in manifest: {excluded}")
        typer.echo(f"Unfinished symbol intervals: {unfinished}; reruns fetch missing coverage only")


@research.command("labels")
def research_labels(
    ctx: typer.Context,
    study_config: Annotated[
        Path | None, typer.Option("--config", help="Study YAML override.")
    ] = None,
    smoke: Annotated[
        bool, typer.Option(help="Two calendar-spread sessions, at most 25 symbols.")
    ] = False,
    resume: Annotated[Path | None, typer.Option(help="Resume an identical run directory.")] = None,
):
    """Run offline pre-2024 exploratory minute-bar labeling and generate a research report."""
    with friendly_errors():
        config = load_config(study_config) if study_config else ctx.obj.model_copy(deep=True)
        if smoke:
            config.research.max_sessions = 2
            config.research.universe_size = min(25, config.research.universe_size)
        try:
            from dead_reckoning.research.study import run_study
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                'Install research dependencies: pip install -e ".[research]"'
            ) from exc
        output = run_study(config, resume)
        typer.echo(f"Label study report: {output / 'report.md'}")


@dataset.command("candidates")
def dataset_candidates(
    ctx: typer.Context,
    start: str | None = None,
    end: str | None = None,
    output_id: str | None = None,
    rebuild: Annotated[
        bool, typer.Option(help="Replace partitions in this selected date range.")
    ] = False,
):
    """Generate policy-independent candidates offline; resume completed session partitions."""
    from dead_reckoning.datasets.candidates import build_candidates

    with friendly_errors():
        result = build_candidates(
            ctx.obj,
            date.fromisoformat(start) if start else None,
            date.fromisoformat(end) if end else None,
            output_id,
            rebuild,
        )
        typer.echo(
            f"Candidates {result['artifact_id']}: {result['rows']:,} total rows; "
            f"{result['built_partitions']} built, {result['skipped_partitions']} skipped; "
            f"{result['elapsed_seconds']:.1f}s."
        )


@dataset.command("labels")
def dataset_labels(
    ctx: typer.Context,
    candidate_id: Annotated[str, typer.Option(help="Existing candidate artifact identifier.")],
    start: str | None = None,
    end: str | None = None,
    policy: Annotated[
        list[str] | None, typer.Option(help="Repeat to share extraction across policies.")
    ] = None,
    fixed: Annotated[bool, typer.Option(help="Generate independent fixed-horizon targets.")] = True,
    fixed_only: bool = False,
    output_id: str | None = None,
    fixed_id: str | None = None,
    rebuild: Annotated[
        bool, typer.Option(help="Replace partitions in this selected date range.")
    ] = False,
):
    """Generate independent target artifacts without changing or rebuilding candidates."""
    from dead_reckoning.datasets.labels import build_labels

    with friendly_errors():
        if fixed_only and policy:
            raise ValueError("--fixed-only cannot be combined with --policy")
        results = build_labels(
            ctx.obj,
            candidate_id,
            date.fromisoformat(start) if start else None,
            date.fromisoformat(end) if end else None,
            [] if fixed_only else policy,
            fixed,
            output_id,
            fixed_id,
            rebuild,
        )
        for result in results:
            typer.echo(
                f"Labels {result['artifact_id']}: {result['rows']:,} total rows; "
                f"{result['built_partitions']} built, {result['skipped_partitions']} skipped; "
                f"{result['elapsed_seconds']:.1f}s."
            )


@dataset.command("status")
def dataset_state(ctx: typer.Context):
    """List candidate/target artifacts, covered dates and unfinished partitions."""
    from dead_reckoning.datasets.artifacts import dataset_status

    with friendly_errors():
        results = dataset_status(ctx.obj.data.root)
        typer.echo(f"Data root: {ctx.obj.data.root}")
        for item in results:
            coverage = item["coverage"] or {}
            typer.echo(
                f"{item['stage']}/{item['artifact_id']} ({item['kind']}): "
                f"{item['complete']} complete, {item['pending']} pending, "
                f"{item['missing_files']} missing files; "
                f"{coverage.get('rows', 0):,} rows "
                f"[{coverage.get('start')}, {coverage.get('end')})."
            )
        if not results:
            typer.echo("No dataset artifacts yet.")


if __name__ == "__main__":
    app()
