from datetime import date

import pytest
from pydantic import ValidationError

from dead_reckoning.config import Config, Credentials, load_config, validate_range


def test_config_is_validated_and_credentials_are_separate(config_path, monkeypatch, tmp_path):
    config = load_config(config_path)
    assert config.data.feed == "sip"
    assert config.data.root == config_path.parent / "external-data"
    assert "key" not in config.model_dump_json()
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(ValidationError):
        Credentials(_env_file=None)
    for invalid in (
        {"data": {}},
        {"data": {"root": str(tmp_path), "api_key": "must-not-go-here"}},
        {"data": {"root": str(tmp_path), "feed": "iex"}},
        {"data": {"root": str(tmp_path)}, "universe": {"lookback_days": 2, "min_history_days": 3}},
        {"data": {"root": str(tmp_path)}, "query": {"memory_limit": "0GB"}},
        {"data": {"root": str(tmp_path)}, "query": {"threads": 0}},
    ):
        with pytest.raises(ValidationError):
            Config.model_validate(invalid)


def test_repo_storage_and_invalid_ranges_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    from pathlib import Path

    root = Path.cwd().as_posix()
    path.write_text(f'data:\n  root: "{root}/data"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        load_config(path)
    with pytest.raises(ValueError, match="precede"):
        validate_range(date(2025, 1, 2), date(2025, 1, 2))
    with pytest.raises(ValueError, match="completed"):
        validate_range(date(2025, 1, 2), date(2100, 1, 2))
