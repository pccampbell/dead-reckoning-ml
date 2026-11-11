"""Validated non-secret YAML configuration and separate environment credentials."""

import os
from datetime import date
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Credentials(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="ALPACA_", extra="ignore")
    api_key: SecretStr = Field(min_length=1)
    secret_key: SecretStr = Field(min_length=1)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictModel):
    root: Path
    provider: Literal["alpaca"] = "alpaca"
    feed: Literal["sip"] = "sip"
    adjustment: Literal["raw"] = "raw"
    start_date: date = date(2016, 1, 1)


class UniverseConfig(StrictModel):
    lookback_days: int = Field(default=20, ge=1)
    ingestion_rank: int = Field(default=2000, ge=1)
    min_history_days: int = Field(default=20, ge=1)

    @model_validator(mode="after")
    def valid_window(self) -> Self:
        if self.min_history_days > self.lookback_days:
            raise ValueError("min_history_days must be <= lookback_days")
        return self


class IngestionConfig(StrictModel):
    daily_batch_size: int = Field(default=100, ge=1, le=1000)
    minute_batch_size: int = Field(default=10, ge=1, le=1000)
    requests_per_minute: int = Field(default=180, ge=1)
    page_size: int = Field(default=10000, ge=1, le=10000)
    max_retries: int = Field(default=5, ge=0, le=20)
    timeout_seconds: float = Field(default=60, gt=0)


class QueryConfig(StrictModel):
    """DuckDB liquidity-query budget; allocations outside its buffer manager are separate."""

    memory_limit: str = Field(default="8GB", pattern=r"^[1-9]\d*(MB|GB)$")
    threads: int = Field(default=8, ge=1)


class Config(StrictModel):
    data: DataConfig
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)


def load_config(path: Path) -> Config:
    config = Config.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    root = Path(os.path.expandvars(str(config.data.root))).expanduser()
    config.data.root = (path.resolve().parent / root).resolve()
    # Prevent accidentally committing datasets, even if the YAML is stored elsewhere.
    for candidate in (Path.cwd(), Path(__file__).resolve().parents[2]):
        if (candidate / "pyproject.toml").exists() and config.data.root.is_relative_to(candidate):
            raise ValueError("data.root must be outside the code repository")
    return config


def validate_range(start: date, end: date) -> None:
    """All public dates use [start, end), in America/New_York session dates."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    if start >= end:
        raise ValueError("start must precede end (end is exclusive)")
    if end > datetime.now(ZoneInfo("America/New_York")).date():
        raise ValueError("end must be today or earlier; only completed sessions are ingested")
