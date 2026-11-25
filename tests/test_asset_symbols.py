import polars as pl
from typer.testing import CliRunner

from dead_reckoning.cli import app
from dead_reckoning.config import load_config
from dead_reckoning.ingestion.assets import asset_symbols
from dead_reckoning.ingestion.manifest import Manifest
from dead_reckoning.market_data.alpaca import stock_bar_symbols
from dead_reckoning.storage.parquet import atomic_json


def test_stock_bar_identifiers_keep_inactive_tickers_and_supported_suffixes():
    assert stock_bar_symbols(
        [
            "OLD",
            "COL_DELISTED",
            "brk.b",
            "BAC.PRA",
            "ahpaw",
            "COL",
            "0029900E0",
            "G4770P115",
            "N014776_DELISTED",
            "OPP-C",
        ]
    ) == ["AHPAW", "BAC.PRA", "BRK.B", "COL", "OLD"]


def test_sip_candidates_exclude_otc_and_retain_inactive_nontradable_assets(config_path):
    root = load_config(config_path).data.root
    folder = root / "raw/assets"
    folder.mkdir(parents=True)
    snapshot = pl.DataFrame(
        {
            "symbol": ["ACTIVE", "INACTIVE", "OTCACTIVE", "OTCINACTIVE"],
            "exchange": ["NYSE", "NASDAQ", "OTC", "OTC"],
            "status": ["active", "inactive", "active", "inactive"],
            "tradable": [True, False, True, False],
        }
    )
    path = folder / "snapshot.parquet"
    snapshot.write_parquet(path)
    assert asset_symbols(root) == ["ACTIVE", "INACTIVE"]
    assert pl.read_parquet(path).equals(snapshot)


def test_broad_selection_preserves_raw_assets_and_repairs_old_population(
    config_path, market_server
):
    config = load_config(config_path)
    folder = config.data.root / "raw/assets"
    folder.mkdir(parents=True)
    references = ["AAA", "BBB", "CCC_DELISTED", "aaa", "0029900E0", "G4770P115", "ZZOTC"]
    path = folder / "snapshot.parquet"
    pl.DataFrame(
        {
            "symbol": references,
            "status": ["inactive"] * len(references),
            "exchange": ["NASDAQ"] * (len(references) - 1) + ["OTC"],
            "tradable": [False] * len(references),
        }
    ).write_parquet(path)
    # Simulate the population saved by the old broad-ingestion code before it failed.
    population = config.data.root / "metadata/daily_requests/2025-01-02_2025-01-10.json"
    atomic_json(population, {"start": "2025-01-02", "end": "2025-01-10", "symbols": references})
    assert asset_symbols(config.data.root) == ["AAA", "BBB", "CCC"]
    with Manifest(config.data.root) as manifest:
        assert manifest.daily_populations()[0][2] == ["AAA", "BBB", "CCC"]

    runner = CliRunner()
    base = ["--config", str(config_path)]
    command = [*base, "ingest", "daily-bars", "--start", "2025-01-02", "--end", "2025-01-10"]
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert all(params["symbols"] == "AAA,BBB,CCC" for _, params in market_server.requests)
    assert pl.read_parquet(path)["symbol"].to_list() == references
    with Manifest(config.data.root) as manifest:
        assert set(manifest.coverage("daily_bars")) == {"AAA", "BBB", "CCC"}
    built = runner.invoke(app, [*base, "universe", "build"])
    assert built.exit_code == 0, built.output
    count = len(market_server.requests)
    assert runner.invoke(app, command).exit_code == 0
    assert len(market_server.requests) == count
