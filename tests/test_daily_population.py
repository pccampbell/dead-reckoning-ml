from datetime import date

import polars as pl
import yaml
from typer.testing import CliRunner

from dead_reckoning.cli import app
from dead_reckoning.config import load_config
from dead_reckoning.ingestion.manifest import Manifest, missing_intervals
from dead_reckoning.universe.liquidity import current_universe


def test_unattempted_broad_symbol_blocks_build_and_resume_skips_completed(
    config_path,
    market_server,
):
    runner = CliRunner()
    base = ["--config", str(config_path)]
    assert runner.invoke(app, [*base, "ingest", "assets"]).exit_code == 0
    config = load_config(config_path)
    # One symbol per batch; AAA's two pages finish, BBB fails, CCC is never attempted.
    config.ingestion.daily_batch_size = 1
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    market_server.failures = [0, 0, 403]
    command = [*base, "ingest", "daily-bars", "--start", "2025-01-02", "--end", "2025-01-10"]
    interrupted = runner.invoke(app, command)
    assert interrupted.exit_code == 1, interrupted.output
    with Manifest(config.data.root) as manifest:
        assert set(manifest.coverage("daily_bars")) == {"AAA"}
        assert (
            manifest.db.execute("SELECT count(*) FROM coverage WHERE symbol = 'CCC'").fetchone()[0]
            == 0
        )
    # Retrying a completed subset cannot shrink the earlier broad population.
    scoped = runner.invoke(app, [*command, "--symbols", "AAA"])
    assert scoped.exit_code == 0, scoped.output
    # Explicit dates and implicit defaults must both check the persisted intended set.
    for bounds in ([], ["--start", "2025-01-02", "--end", "2025-01-10"]):
        rejected = runner.invoke(app, [*base, "universe", "build", *bounds])
        assert rejected.exit_code == 1
        assert "Daily coverage is incomplete" in rejected.output
        assert "CCC" in rejected.output
    assert current_universe(config.data.root) is None

    count = len(market_server.requests)
    resumed = runner.invoke(app, command)
    assert resumed.exit_code == 0, resumed.output
    requested = {
        params["symbols"]
        for path, params in market_server.requests[count:]
        if path.endswith("/bars")
    }
    assert requested == {"BBB", "CCC"}
    built = runner.invoke(app, [*base, "universe", "build"])
    assert built.exit_code == 0, built.output


def test_completed_zero_row_broad_population_is_complete(config_path, market_server):
    runner = CliRunner()
    base = ["--config", str(config_path)]
    assert runner.invoke(app, [*base, "ingest", "assets"]).exit_code == 0
    # A weekend returns no rows for any symbol, but all requests completed successfully.
    result = runner.invoke(
        app, [*base, "ingest", "daily-bars", "--start", "2025-01-04", "--end", "2025-01-06"]
    )
    assert result.exit_code == 0, result.output
    config = load_config(config_path)
    with Manifest(config.data.root) as manifest:
        assert manifest.summary("daily_bars")[:2] == (3, 0)
        for symbol in ("AAA", "BBB", "CCC"):
            assert not missing_intervals(
                date(2025, 1, 4), date(2025, 1, 6), manifest.coverage("daily_bars")[symbol]
            )
    built = runner.invoke(app, [*base, "universe", "build"])
    assert built.exit_code == 0, built.output


def test_explicit_scope_ignores_unrequested_assets_and_bootstraps(config_path, market_server):
    runner = CliRunner()
    base = ["--config", str(config_path)]
    assert runner.invoke(app, [*base, "ingest", "assets"]).exit_code == 0
    scope = ["--symbols", "AAA,BBB", "--start", "2025-01-02", "--end", "2025-01-10"]
    result = runner.invoke(app, [*base, "ingest", "daily-bars", *scope])
    assert result.exit_code == 0, result.output
    built = runner.invoke(app, [*base, "universe", "build"])
    assert built.exit_code == 0, built.output
    config = load_config(config_path)
    with Manifest(config.data.root) as manifest:
        assert "CCC" not in manifest.coverage("daily_bars")
    folder, _ = current_universe(config.data.root)
    assert pl.read_parquet(folder / "symbols.parquet")["symbol"].to_list() == ["AAA", "BBB"]
    bootstrap = runner.invoke(app, [*base, "data", "bootstrap", *scope])
    assert bootstrap.exit_code == 0, bootstrap.output
    assert "Bootstrap completed successfully" in bootstrap.output
