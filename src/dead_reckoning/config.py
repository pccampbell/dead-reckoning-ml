"""Validated non-secret YAML configuration and separate environment credentials."""

import math
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
    max_staleness_days: int = Field(default=7, ge=1)

    @model_validator(mode="after")
    def valid_window(self) -> Self:
        if self.min_history_days > self.lookback_days:
            raise ValueError("min_history_days must be <= lookback_days")
        return self


class IngestionConfig(StrictModel):
    daily_batch_size: int = Field(default=100, ge=1, le=1000)
    minute_batch_size: int = Field(default=10, ge=1, le=1000)
    requests_per_minute: int = Field(default=198, ge=1)
    page_size: int = Field(default=10000, ge=1, le=10000)
    max_retries: int = Field(default=5, ge=0, le=20)
    timeout_seconds: float = Field(default=60, gt=0)


class QueryConfig(StrictModel):
    """DuckDB liquidity-query budget; allocations outside its buffer manager are separate."""

    memory_limit: str = Field(default="8GB", pattern=r"^[1-9]\d*(MB|GB)$")
    threads: int = Field(default=8, ge=1)


class BarrierPair(StrictModel):
    name: str = Field(min_length=1)
    tp: float = Field(gt=0, lt=1)
    sl: float = Field(gt=0, lt=1)


class LabelStudyConfig(StrictModel):
    start: date = date(2019, 1, 1)
    end: date = date(2024, 1, 1)
    seed: int = 42
    sessions_per_month: int = Field(default=1, ge=1, le=23)
    max_sessions: int | None = Field(default=None, ge=1)
    universe_size: int = Field(default=500, ge=1)
    decision_minutes: int = Field(default=5, ge=1)
    execution_delay: int = Field(default=1, ge=0)
    horizons: list[int] = Field(default_factory=lambda: [15, 30, 60, 90, 120])
    pairs: list[BarrierPair] = Field(
        default_factory=lambda: [
            BarrierPair(name="tight", tp=0.0025, sl=0.002),
            BarrierPair(name="original", tp=0.005, sl=0.004),
            BarrierPair(name="symmetric", tp=0.005, sl=0.005),
            BarrierPair(name="higher_rr", tp=0.0075, sl=0.004),
            BarrierPair(name="wide", tp=0.01, sl=0.008),
        ]
    )
    costs_bps: list[float] = Field(default_factory=lambda: [0, 2, 5, 10, 20])
    threads: int = Field(default=2, ge=1, le=16)
    memory_limit: str = Field(default="2GB", pattern=r"^[1-9]\d*(MB|GB)$")
    max_staleness_days: int = Field(default=7, ge=1)
    output: Path = Path("research/label_study")

    @model_validator(mode="after")
    def valid_study(self) -> Self:
        if self.start >= self.end or self.end > date(2024, 1, 1):
            raise ValueError("Research requires start < end <= 2024-01-01 (reserved holdout)")
        if not self.horizons or any(h < 1 or h > 390 for h in self.horizons):
            raise ValueError("horizons must contain minute durations in [1, 390]")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be unique")
        self.horizons.sort()
        if not self.pairs or len({p.name for p in self.pairs}) != len(self.pairs):
            raise ValueError("pairs must be nonempty with unique names")
        if not self.costs_bps or any(c < 0 or not math.isfinite(c) for c in self.costs_bps):
            raise ValueError("costs_bps must be nonempty, finite, and nonnegative")
        return self


class CandidateConfig(StrictModel):
    universe_size: int = Field(default=500, ge=1)
    decision_minutes: int = Field(default=5, ge=1, le=390)
    execution_delay: int = Field(default=1, ge=0, le=390)
    side: Literal["long"] = "long"


class LabelPolicy(StrictModel):
    name: str = Field(default="original_60m", pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    tp: float = Field(default=0.005, gt=0, lt=1)
    sl: float = Field(default=0.004, gt=0, lt=1)
    horizon: int = Field(default=60, ge=1, le=390)
    execution_delay: int = Field(default=1, ge=0, le=390)
    missing_tolerance: int | None = Field(default=2, ge=0, le=390)
    ambiguity: Literal["conservative", "optimistic", "exclude"] = "conservative"


class DatasetConfig(StrictModel):
    start: date = date(2019, 1, 1)
    end: date = date(2019, 2, 1)
    candidates: CandidateConfig = Field(default_factory=CandidateConfig)
    policies: list[LabelPolicy] = Field(
        default_factory=lambda: [
            LabelPolicy(),
            LabelPolicy(name="original_30m", horizon=30),
            LabelPolicy(name="wide_60m", tp=0.01, sl=0.008),
        ]
    )
    fixed_horizons: list[int] = Field(default_factory=lambda: [15, 30, 60, 90, 120])

    @model_validator(mode="after")
    def valid_dataset(self) -> Self:
        if self.start >= self.end:
            raise ValueError("dataset.start must precede dataset.end")
        if len({p.name for p in self.policies}) != len(self.policies):
            raise ValueError("Dataset label policy names must be unique")
        if len(set(self.fixed_horizons)) != len(self.fixed_horizons) or any(
            h < 1 or h > 390 for h in self.fixed_horizons
        ):
            raise ValueError("fixed_horizons must be unique minute durations in [1, 390]")
        self.fixed_horizons.sort()
        return self


class Config(StrictModel):
    data: DataConfig
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    research: LabelStudyConfig = Field(default_factory=LabelStudyConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)


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
