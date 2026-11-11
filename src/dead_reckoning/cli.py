"""Small synchronous CLI; all actual ingestion lives in reusable modules."""

import logging
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from tempfile import mkdtemp
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
from dead_reckoning.market_data.alpaca import AlpacaClient
from dead_reckoning.universe.liquidity import (
    build_universe,
    current_universe,
    universe_symbols,
)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
ingest = typer.Typer(no_args_is_help=True)
universe = typer.Typer(no_args_is_help=True)
data = typer.Typer(no_args_is_help=True)
app.add_typer(ingest, name="ingest")
app.add_typer(universe, name="universe")
app.add_typer(data, name="data")
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
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


def print_result(result: dict) -> None:
    typer.echo(
        f"New request batches: {result['batches']}; new rows: {result['rows']}; "
        f"already covered symbols: {result['skipped_symbols']}"
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
                ingest_bars(config, manifest, client, "daily_bars", selected, lower, upper)
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


def download_minutes(
    config: Config, manifest: Manifest, client: AlpacaClient, lower: date, upper: date
) -> None:
    folder, _ = build_universe(config, manifest, lower, upper)
    selected = universe_symbols(folder)
    if not selected:
        raise ValueError(
            "Historical universe is empty; ingest more daily history or check criteria"
        )
    print_result(ingest_bars(config, manifest, client, "minute_bars", selected, lower, upper))


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
        lower, upper = dates(config, start, end)
        with Manifest(config.data.root) as manifest, client_for(config) as client:
            ingest_assets(config.data.root, client, refresh_assets)
            print_result(
                ingest_bars(
                    config,
                    manifest,
                    client,
                    "daily_bars",
                    daily_symbols(config, symbols),
                    lower,
                    upper,
                )
            )
            download_minutes(config, manifest, client, lower, upper)


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
            stale = meta["daily_fingerprint"] != manifest.fingerprint("daily_bars") or any(
                meta.get(key) != value for key, value in config.universe.model_dump().items()
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
        unfinished = manifest.db.execute(
            "SELECT count(*) FROM coverage WHERE status != 'complete'"
        ).fetchone()[0]
        typer.echo(f"Unfinished symbol intervals: {unfinished}; reruns fetch missing coverage only")


if __name__ == "__main__":
    app()
