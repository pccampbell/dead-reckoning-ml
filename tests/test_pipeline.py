"""Executable offline smoke: real CLI -> real HTTP transport -> real external storage."""

import polars as pl
from typer.testing import CliRunner

from dead_reckoning import cli
from dead_reckoning.cli import app
from dead_reckoning.config import load_config
from dead_reckoning.ingestion import smoke
from dead_reckoning.ingestion.manifest import Manifest
from dead_reckoning.universe.liquidity import current_universe


def test_end_to_end_smoke(config_path, market_server):
    runner = CliRunner()
    config = load_config(config_path)
    root = config.data.root

    def run(*args):
        result = runner.invoke(app, ["--config", str(config_path), *args])
        assert result.exit_code == 0, result.output + repr(result.exception)
        print(f"dead-reckoning {' '.join(args)}\n{result.output.strip()}")
        return result

    market_server.failures = [503]  # Exercise real retry/backoff, then successful assets.
    run("ingest", "assets")
    run("ingest", "daily-bars", "--start", "2025-01-02", "--end", "2025-01-10")
    run("universe", "build")
    run("ingest", "minute-bars", "--start", "2025-01-02", "--end", "2025-01-10")
    folder, _ = current_universe(root)
    assert pl.read_parquet(folder / "symbols.parquet")["symbol"].to_list() == ["AAA", "BBB"]
    assert (
        pl.scan_parquet(root / "raw/daily_bars/**/*.parquet").select(pl.len()).collect().item()
        == 18
    )
    minute = pl.read_parquet(root / "raw/minute_bars/**/*.parquet")
    assert minute.height == 24
    assert minute.unique(["symbol", "timestamp"]).height == 24
    assert minute["timestamp"].min().date().isoformat() == "2025-01-02"
    report = run("data", "status").output
    assert "2 completed; 0 pending" in report
    assert "3 symbols in 1 snapshots" in report

    # A whole bootstrap rerun must make ZERO HTTP calls, including asset ingestion.
    request_count = len(market_server.requests)
    result = run("data", "bootstrap", "--start", "2025-01-02", "--end", "2025-01-10")
    assert "New request batches: 0; new rows: 0; already covered symbols: 3" in result.output
    assert len(market_server.requests) == request_count

    # Lost index / rename-before-commit recovery must preserve all completed raw coverage.
    db = root / "metadata/ingestion.duckdb"
    db.unlink()
    run("data", "bootstrap", "--start", "2025-01-02", "--end", "2025-01-10")
    assert len(market_server.requests) == request_count

    # A forward extension requests only the missing tail, then automatically rebuilds the union.
    run("data", "bootstrap", "--start", "2025-01-02", "--end", "2025-01-14")
    requests = market_server.requests[request_count:]
    assert requests
    assert all(params["start"].startswith("2025-01-10") for _, params in requests)
    minute = pl.read_parquet(root / "raw/minute_bars/**/*.parquet")
    assert minute.height == 32
    assert minute.unique(["symbol", "timestamp"]).height == minute.height
    run("data", "status")

    # A longer minute range must not silently derive its union from incomplete daily data.
    count = len(market_server.requests)
    rejected = runner.invoke(
        app,
        [
            "--config",
            str(config_path),
            "ingest",
            "minute-bars",
            "--start",
            "2025-01-02",
            "--end",
            "2025-01-15",
        ],
    )
    assert rejected.exit_code == 1
    assert "Daily coverage is incomplete" in rejected.output
    assert len(market_server.requests) == count


def test_interrupted_pagination_resumes(config_path, market_server):
    runner = CliRunner()
    base = [
        "--config",
        str(config_path),
        "ingest",
        "daily-bars",
        "--symbols",
        "AAA,BBB",
        "--start",
        "2025-01-02",
        "--end",
        "2025-01-10",
    ]
    market_server.failures = [0, 403]  # Fail after the first actual bar page.
    first = runner.invoke(app, base)
    assert first.exit_code == 1, first.output
    root = load_config(config_path).data.root
    assert not list((root / "raw/daily_bars").rglob("*.parquet"))
    assert not list((root / "raw/daily_bars").rglob("*.partial"))
    with Manifest(root) as manifest:
        assert not manifest.coverage("daily_bars")
    resumed = runner.invoke(app, base)
    assert resumed.exit_code == 0, resumed.output
    frame = pl.read_parquet(root / "raw/daily_bars/**/*.parquet")
    assert frame.height == 12
    count = len(market_server.requests)
    assert runner.invoke(app, base).exit_code == 0
    assert len(market_server.requests) == count


def test_live_smoke_command_offline(config_path, market_server, monkeypatch, tmp_path):
    # Test the opt-in live command against local HTTP fixtures; normal pytest stays offline.
    root = tmp_path / "smoke-data"
    root.mkdir()
    monkeypatch.setattr(cli, "mkdtemp", lambda **_: str(root))
    monkeypatch.setattr(smoke, "SMOKE_SYMBOLS", ["AAA", "BBB"])
    monkeypatch.setattr(cli, "SMOKE_SYMBOLS", ["AAA", "BBB"])
    result = CliRunner().invoke(app, ["--config", str(config_path), "data", "smoke"])
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert "Live smoke passed" in result.output
    assert "rerun fetched zero bar batches" in result.output
    assert "2 completed; 0 pending" in result.output
    assert (root / "metadata/ingestion.duckdb").is_file()
    assert not load_config(config_path).data.root.exists()
    # One initial daily request and two initial monthly minute requests; none on the rerun.
    initial_requests = [
        params
        for path, params in market_server.requests
        if path.endswith("/bars") and "page_token" not in params
    ]
    assert len(initial_requests) == 3
