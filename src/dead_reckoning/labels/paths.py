"""Observed-price path quality shared by research and reusable dataset labels.

The strict barrier resolver remains the single authority for execution and
ambiguity. Only the first disqualifying interval and endpoint return change.
"""

from dataclasses import dataclass, replace

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from dead_reckoning.labels.barriers import Labels, Touches, first_touches, resolve, valid_ohlc

REASONS = (
    "valid",
    "missing_entry",
    "excessive_interior_gap",
    "missing_expiration",
    "invalid_data",
    "insufficient_session_time",
)


@dataclass
class PathEvidence:
    touches: Touches
    prices: np.ndarray
    present: np.ndarray
    good: np.ndarray
    starts: np.ndarray
    session_minutes: int


@dataclass
class PathResult:
    labels: Labels
    reason: np.ndarray
    interior_max: np.ndarray
    interior_missing: np.ndarray
    missing_minutes: np.ndarray
    maximum_missing: np.ndarray
    strict_valid: np.ndarray
    end: np.ndarray


def price_grid(group, opening, minutes):
    """Put only real bars on the scheduled grid; flag duplicates and off-minute rows."""
    import polars as pl

    prices = np.full((minutes, 4), np.nan)
    present = np.zeros(minutes, dtype=bool)
    good = present.copy()
    if group is not None and group.height:
        stamp = group["timestamp"].cast(pl.Int64).to_numpy()
        offset = stamp - int(opening.timestamp()) * 1_000_000
        index = (offset // 60_000_000).astype(int)
        if np.any(index < 0) or np.any(index >= minutes):
            raise ValueError("Price grid received bars outside its session")
        aligned = offset % 60_000_000 == 0
        values = group.select("open", "high", "low", "close").to_numpy()
        counts = np.bincount(index, minlength=minutes)
        prices[index] = values
        present[index] = True
        good[index] = valid_ohlc(values) & aligned
        good[counts > 1] = False
    return prices, present, good


def scan_paths(prices, present, good, starts, horizons, upper, lower):
    """Scan distinct barriers once. Padding only represents unavailable time, not prices.

    Late candidates are retained; each target independently checks scheduled fit.
    No padded interval is allowed to produce a valid target.
    """
    starts = np.asarray(starts, dtype=int)
    maximum = max(horizons)
    minutes = len(prices)
    padding = max(0, int(starts.max(initial=0)) + maximum - minutes)
    safe = np.pad(prices, ((0, padding), (0, 0)), constant_values=np.nan)
    seen = np.pad(present, (0, padding), constant_values=False)
    valid = np.pad(good, (0, padding), constant_values=False)
    safe[~valid] = np.nan
    touches = first_touches(safe, starts, horizons, upper, lower, valid)
    # Preserve original prices for endpoint diagnostics; invalid endpoints still fail good.
    original = np.pad(prices, ((0, padding), (0, 0)), constant_values=np.nan)
    path = sliding_window_view(original, maximum, axis=0)[starts].transpose(0, 2, 1)
    return PathEvidence(
        touches,
        path,
        sliding_window_view(seen, maximum)[starts],
        sliding_window_view(valid, maximum)[starts],
        starts,
        minutes,
    )


def fixed_returns(evidence, horizon):
    """Endpoint-only prices; intermediate bars never affect this independent target."""
    e = evidence
    reason = np.zeros(len(e.starts), dtype=np.int8)
    reason[~e.present[:, horizon - 1]] = 3
    reason[e.present[:, horizon - 1] & ~e.good[:, horizon - 1]] = 4
    reason[e.present[:, 0] & ~e.good[:, 0]] = 4
    reason[~e.present[:, 0]] = 1
    reason[e.starts + horizon > e.session_minutes] = 5
    with np.errstate(invalid="ignore", divide="ignore"):
        returns = e.prices[:, horizon - 1, 3] / e.prices[:, 0, 0] - 1
    return np.where(reason == 0, returns, np.nan), reason


def resolve_paths(evidence, tp, sl, horizon, tolerances=(2,)):
    """First-disqualifier attribution and consecutive *absent* interior-minute limits."""
    e, t = evidence, evidence.touches
    n = len(e.starts)
    first = np.minimum(t.upper[tp], t.lower[sl])
    end = np.minimum(first, horizon - 1)
    offsets = np.arange(horizon)[None, :]
    seen, valid = e.present[:, :horizon], e.good[:, :horizon]
    missing = ~seen
    before_exit = offsets <= end[:, None]
    interior = missing & (offsets > 0) & (offsets < horizon - 1) & before_exit
    quality = seen & ~valid & before_exit
    quality_first = np.where(quality, offsets, horizon).min(axis=1)
    limits = set(tolerances) | {0}
    bad_gap = {limit: np.full(n, horizon, dtype=np.int16) for limit in limits if limit is not None}
    lengths = np.zeros(n, dtype=np.int16)
    maximum = lengths.copy()
    all_lengths, all_max = lengths.copy(), lengths.copy()
    for k in range(horizon):
        lengths = np.where(interior[:, k], lengths + 1, 0)
        maximum = np.maximum(maximum, lengths)
        all_lengths = np.where(missing[:, k] & before_exit[:, k], all_lengths + 1, 0)
        all_max = np.maximum(all_max, all_lengths)
        for limit in bad_gap:
            bad_gap[limit] = np.minimum(
                bad_gap[limit], np.where(lengths > limit, k - lengths + 1, horizon)
            )
    forward, _ = fixed_returns(e, horizon)
    output = {}
    for limit in limits:
        bad = quality_first.copy()
        reason = np.where(bad < horizon, 4, 0).astype(np.int8)
        bad[missing[:, 0]] = 0
        if limit is not None:
            reason[bad_gap[limit] < bad] = 2
            bad = np.minimum(bad, bad_gap[limit])
        expiry = (first >= horizon) & missing[:, -1] & (horizon - 1 < bad)
        reason[expiry], bad[expiry] = 3, horizon - 1
        reason[missing[:, 0]] = 1
        scheduled = e.starts + horizon > e.session_minutes
        reason[scheduled], bad[scheduled] = 5, 0
        touched = replace(t, first_bad=bad, forward={**t.forward, horizon: forward})
        labels = resolve(touched, tp, sl, horizon)
        output[limit] = (labels, reason)
    strict = output[0][1] == 0
    return {
        limit: PathResult(
            output[limit][0],
            output[limit][1],
            maximum,
            interior.sum(axis=1),
            (missing & before_exit).sum(axis=1),
            all_max,
            strict,
            end,
        )
        for limit in tolerances
    }
