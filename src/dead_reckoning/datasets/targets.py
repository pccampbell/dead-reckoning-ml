"""Keyed barrier and fixed-horizon target schemas, independent of artifact I/O."""

import numpy as np
import polars as pl

from dead_reckoning.datasets.artifacts import KEYS
from dead_reckoning.datasets.candidates import UTC_US
from dead_reckoning.labels.barriers import Outcome
from dead_reckoning.labels.paths import REASONS, fixed_returns, resolve_paths


def timestamp_column(name, opening, offsets, valid):
    values = int(opening.timestamp()) * 1_000_000 + np.asarray(offsets, dtype=np.int64) * 60_000_000
    return pl.Series(name, values).cast(UTC_US).set(pl.Series(~np.asarray(valid)), None)


def nullable_float(name, values, valid, dtype=pl.Float64):
    return pl.Series(name, np.asarray(values), dtype=dtype).set(pl.Series(~np.asarray(valid)), None)


def entry_columns(evidence, opening):
    good = evidence.good[:, 0] & (evidence.starts < evidence.session_minutes)
    return [
        timestamp_column(
            "entry_timestamp_utc",
            opening,
            evidence.starts,
            evidence.starts < evidence.session_minutes,
        ),
        nullable_float("entry_price", evidence.prices[:, 0, 0], good),
    ]


def barrier_targets(candidates, evidence, policy, opening):
    r = resolve_paths(evidence, policy.tp, policy.sl, policy.horizon, (policy.missing_tolerance,))[
        policy.missing_tolerance
    ]
    labels, t = r.labels, evidence.touches
    path_valid = r.reason == 0
    ambiguous = path_valid & (labels.outcome == Outcome.AMBIGUOUS)
    conservative = labels.outcome.copy()
    conservative[ambiguous] = Outcome.SL_FIRST
    resolved = conservative.copy()
    realized = labels.conservative.copy()
    valid = path_valid.copy()
    reasons = np.array(REASONS, dtype=object)[r.reason]
    if policy.ambiguity == "optimistic":
        resolved[ambiguous] = Outcome.TP_FIRST
        realized = labels.optimistic.copy()
    elif policy.ambiguity == "exclude":
        valid[ambiguous] = False
        reasons[ambiguous] = "ambiguous_excluded"
    names = np.array([o.name for o in Outcome], dtype=object)
    # Holding times already preserve open-before-intrabar execution from the core resolver.
    exit_offset = evidence.starts + np.nan_to_num(labels.holding).astype(int)
    entry = evidence.prices[:, 0, 0]
    lower_hit = t.lower_open[policy.sl] & (labels.outcome == Outcome.SL_FIRST)
    stop_index = np.minimum(t.lower[policy.sl], evidence.prices.shape[1] - 1)
    stop_open = evidence.prices[np.arange(len(entry)), stop_index, 0]
    stop_gap = path_valid & lower_hit & (stop_open < entry * (1 - policy.sl))
    frame = candidates.select(*KEYS, "session_date").with_columns(
        *entry_columns(evidence, opening),
        timestamp_column("exit_timestamp_utc", opening, exit_offset, valid),
        nullable_float("exit_price", entry * (1 + realized), valid),
        pl.Series("raw_outcome", names[labels.outcome], dtype=pl.String),
        pl.Series("outcome", names[resolved], dtype=pl.String).set(pl.Series(~valid), None),
        pl.Series("conservative_outcome", names[conservative], dtype=pl.String).set(
            pl.Series(~path_valid), None
        ),
        nullable_float("conservative_trade_return", labels.conservative, path_valid, pl.Float32),
        nullable_float("optimistic_trade_return", labels.optimistic, path_valid, pl.Float32),
        nullable_float("realized_trade_return", realized, valid, pl.Float32),
        nullable_float(
            "expiration_return", realized, valid & (labels.outcome == Outcome.EXPIRE), pl.Float32
        ),
        nullable_float("holding_minutes", labels.holding, valid, pl.Float32),
        pl.Series("label_valid", valid),
        pl.Series("path_valid", path_valid),
        pl.Series("invalid_reason", reasons, dtype=pl.String),
        pl.Series("ambiguous_outcome", ambiguous),
        pl.Series("stop_gap", stop_gap),
        pl.Series("missing_minutes", r.missing_minutes, dtype=pl.UInt16).set(
            pl.Series(r.reason == 5), None
        ),
        pl.Series("max_consecutive_missing_minutes", r.maximum_missing, dtype=pl.UInt16).set(
            pl.Series(r.reason == 5), None
        ),
        pl.Series("permitted_interior_gap", path_valid & (r.interior_max > 0)),
        pl.Series("strict_valid", r.strict_valid & ~(ambiguous & (policy.ambiguity == "exclude"))),
    )
    return frame


def fixed_targets(candidates, evidence, horizons, opening):
    frame = candidates.select(*KEYS, "session_date").with_columns(*entry_columns(evidence, opening))
    for horizon in horizons:
        returns, reason = fixed_returns(evidence, horizon)
        frame = frame.with_columns(
            nullable_float(f"return_{horizon}m", returns, reason == 0, pl.Float32),
            pl.Series(f"valid_{horizon}m", reason == 0),
            pl.Series(
                f"invalid_reason_{horizon}m",
                np.array(REASONS, dtype=object)[reason],
                dtype=pl.String,
            ),
            timestamp_column(
                f"exit_timestamp_{horizon}m_utc",
                opening,
                evidence.starts + horizon,
                evidence.starts + horizon <= evidence.session_minutes,
            ),
        )
    return frame


def target_summary(frame, kind, horizons=()):
    if kind == "fixed_horizon":
        return {
            f"{h}m": {
                "valid": int(frame[f"valid_{h}m"].sum()),
                "reasons": dict(frame.group_by(f"invalid_reason_{h}m").len().iter_rows()),
            }
            for h in horizons
        }
    return {
        "valid": int(frame["label_valid"].sum()),
        "strict_valid": int(frame["strict_valid"].sum()),
        "raw_outcomes": dict(frame.group_by("raw_outcome").len().iter_rows()),
        "outcomes": dict(frame.filter(pl.col("label_valid")).group_by("outcome").len().iter_rows()),
        "invalid_reasons": dict(frame.group_by("invalid_reason").len().iter_rows()),
        "stop_gaps": int(frame["stop_gap"].sum()),
        "mean_return_bps": frame["realized_trade_return"].mean() * 10000
        if frame["realized_trade_return"].count()
        else None,
    }
