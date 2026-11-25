"""Reusable long-only minute OHLC labels. No I/O, calendars, or research features.

Indices are zero-based offsets from the entry minute; sentinel = maximum horizon.
An invalid interval at or before an exit makes that policy UNKNOWN. Prices remain
float64 throughout comparisons. Intrabar exit times use minute end as a proxy;
opening exits use minute start. No candle-direction ordering is inferred.
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Outcome(IntEnum):
    TP_FIRST = 0
    SL_FIRST = 1
    EXPIRE = 2
    AMBIGUOUS = 3
    UNKNOWN = 4


@dataclass
class Touches:
    first_bad: np.ndarray
    upper: dict[float, np.ndarray]
    lower: dict[float, np.ndarray]
    upper_open: dict[float, np.ndarray]
    lower_open: dict[float, np.ndarray]
    stop_return: dict[float, np.ndarray]
    forward: dict[int, np.ndarray]


@dataclass
class Labels:
    outcome: np.ndarray
    conservative: np.ndarray
    optimistic: np.ndarray
    holding: np.ndarray


def valid_ohlc(prices: np.ndarray) -> np.ndarray:
    o, h, low, c = prices.T
    return (
        np.isfinite(prices).all(axis=1)
        & (prices > 0).all(axis=1)
        & (low <= np.minimum(o, c))
        & (h >= np.maximum(o, c))
        & (low <= h)
    )


def _first(mask: np.ndarray) -> np.ndarray:
    return np.where(mask.any(axis=1), mask.argmax(axis=1), mask.shape[1]).astype(np.int16)


def first_touches(
    prices: np.ndarray,
    starts: np.ndarray,
    horizons: list[int],
    upper: list[float],
    lower: list[float],
    valid: np.ndarray | None = None,
) -> Touches:
    """Scan each distinct threshold once for all entries in one symbol/session.

    Missing minutes must be represented by NaN rows on a scheduled minute grid.
    Duplicate/unaligned timestamps can be marked invalid by the caller. Windows
    must fit the session; eligibility must be decided before calling this method.
    """
    prices = np.asarray(prices, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64)
    maximum = max(horizons)
    if np.any(starts < 0) or np.any(starts + maximum > len(prices)):
        raise ValueError("Forward windows must fit within a single scheduled session")
    good = valid_ohlc(prices)
    if valid is not None:
        good &= valid
    path = sliding_window_view(prices, maximum, axis=0)[starts].transpose(0, 2, 1)
    path_good = sliding_window_view(good, maximum)[starts]
    first_bad = _first(~path_good)
    entry = path[:, 0, 0]
    rows = np.arange(len(starts))
    result = Touches(first_bad, {}, {}, {}, {}, {}, {})
    with np.errstate(invalid="ignore", divide="ignore"):
        for threshold in sorted(set(upper)):
            price = entry[:, None] * (1 + threshold)
            opened = path[:, :, 0] >= price
            touched = opened | (path[:, :, 1] >= price)
            index = _first(touched)
            result.upper[threshold] = index
            result.upper_open[threshold] = opened[rows, np.minimum(index, maximum - 1)]
        for threshold in sorted(set(lower)):
            price = entry[:, None] * (1 - threshold)
            opened = path[:, :, 0] <= price
            touched = opened | (path[:, :, 2] <= price)
            index = _first(touched)
            safe_index = np.minimum(index, maximum - 1)
            result.lower[threshold] = index
            result.lower_open[threshold] = opened[rows, safe_index]
            result.stop_return[threshold] = np.where(
                result.lower_open[threshold], path[rows, safe_index, 0] / entry - 1, -threshold
            )
        for horizon in horizons:
            result.forward[horizon] = np.where(
                first_bad >= horizon, path[:, horizon - 1, 3] / entry - 1, np.nan
            )
    return result


def resolve(t: Touches, tp: float, sl: float, horizon: int) -> Labels:
    """Resolve first-touch evidence without rescanning OHLC or dropping ambiguity."""
    up, down = t.upper[tp], t.lower[sl]
    first = np.minimum(up, down)
    hit = first < horizon
    same = up == down
    take = hit & ((up < down) | (same & t.upper_open[tp]))
    stop = hit & ((down < up) | (same & t.lower_open[sl])) & ~take
    ambiguous = hit & ~take & ~stop
    unknown = t.first_bad <= np.minimum(first, horizon - 1)
    code = np.full(len(up), Outcome.EXPIRE, dtype=np.int8)
    code[take], code[stop], code[ambiguous] = (
        Outcome.TP_FIRST,
        Outcome.SL_FIRST,
        Outcome.AMBIGUOUS,
    )
    code[unknown] = Outcome.UNKNOWN
    conservative = np.where(
        take, tp, np.where(stop | ambiguous, t.stop_return[sl], t.forward[horizon])
    )
    optimistic = np.where(ambiguous, tp, conservative)
    opening = (take & t.upper_open[tp]) | (stop & t.lower_open[sl])
    holding = np.where(hit, first + (~opening).astype(int), horizon).astype(float)
    conservative[unknown], optimistic[unknown], holding[unknown] = np.nan, np.nan, np.nan
    return Labels(code, conservative, optimistic, holding)
