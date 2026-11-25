"""Independent execution oracle for the research-only missing-bar sensitivity."""

import numpy as np
import pytest

from dead_reckoning.labels.barriers import Outcome, valid_ohlc
from dead_reckoning.research.missing_bars import HORIZON, SL, TP, evaluate, runs


def oracle(prices, present, good, tolerance):
    if not present[0] or not good[0]:
        return Outcome.UNKNOWN, np.nan
    entry = prices[0, 0]
    upper, lower = entry * (1 + TP), entry * (1 - SL)
    consecutive = 0
    for k in range(HORIZON):
        if not present[k]:
            if k == HORIZON - 1:
                return Outcome.UNKNOWN, np.nan
            consecutive += 1
            if tolerance is not None and consecutive > tolerance:
                return Outcome.UNKNOWN, np.nan
            continue
        if not good[k]:
            return Outcome.UNKNOWN, np.nan
        consecutive = 0
        o, h, low, close = prices[k]
        if o >= upper:
            return Outcome.TP_FIRST, TP
        if o <= lower:
            return Outcome.SL_FIRST, o / entry - 1
        if h >= upper and low <= lower:
            return Outcome.AMBIGUOUS, -SL
        if h >= upper:
            return Outcome.TP_FIRST, TP
        if low <= lower:
            return Outcome.SL_FIRST, -SL
    return Outcome.EXPIRE, close / entry - 1


def test_gap_boundaries_real_endpoints_and_opening_execution():
    prices = np.full((70, 4), 100.0)
    present = np.ones(70, dtype=bool)
    present[10:12] = False
    prices[10:12] = np.nan
    prices[12] = [99, 101, 98, 100]  # Opening stop wins despite both barriers touched.
    good = valid_ohlc(prices)
    results, maximum, _, _, _ = evaluate(prices, present, good, np.array([0]))
    assert maximum[0] == 2
    for tolerance, (labels, reason) in results.items():
        if tolerance in (0, 1):
            assert labels.outcome[0] == Outcome.UNKNOWN and reason[0] == 2
        else:
            assert labels.outcome[0] == Outcome.SL_FIRST
            assert labels.conservative[0] == pytest.approx(-0.01)
    prices[12] = 100
    prices[60, 1] = 110  # A post-expiration touch must never extend the elapsed window.
    results, _, _, forward, _ = evaluate(prices, present, valid_ohlc(prices), np.array([0]))
    assert results[2][0].outcome[0] == Outcome.EXPIRE
    assert forward[0] == 0
    present[59] = False
    prices[59] = np.nan
    results, *_ = evaluate(prices, present, valid_ohlc(prices), np.array([0]))
    assert results[None][1][0] == 3  # No genuine expiration price.
    present[0] = False
    prices[0] = np.nan
    results, *_ = evaluate(prices, present, valid_ohlc(prices), np.array([0]))
    assert all(reason[0] == 1 for _, reason in results.values())


def test_quality_is_never_tolerated_and_gaps_after_exit_do_not_matter():
    prices = np.full((60, 4), 100.0)
    present = np.ones(60, dtype=bool)
    good = present.copy()
    good[2] = False  # E.g. duplicate timestamp, despite valid-looking prices.
    results, *_ = evaluate(prices, present, good, np.array([0]))
    assert all(reason[0] == 4 for _, reason in results.values())
    prices[1, 1] = 101
    present[10:] = False
    prices[10:] = np.nan
    results, maximum, *_ = evaluate(prices, present, good & present, np.array([0]))
    assert maximum[0] == 0
    assert all(labels.outcome[0] == Outcome.TP_FIRST for labels, _ in results.values())
    assert runs(np.array([True, True, False, True])) == [(0, 2), (3, 4)]


def test_randomized_missing_only_policy_matches_minute_loop():
    rng = np.random.default_rng(17)
    for _ in range(30):
        o = 100 * np.exp(rng.normal(0, 0.001, 90).cumsum())
        c = o * np.exp(rng.normal(0, 0.0005, 90))
        prices = np.column_stack(
            [
                o,
                np.maximum(o, c) * 1.001,
                np.minimum(o, c) * 0.999,
                c,
            ]
        )
        present = rng.random(90) > 0.08
        prices[~present] = np.nan
        good = present & (rng.random(90) > 0.005)
        starts = np.arange(30)
        results, *_ = evaluate(prices, present, good, starts)
        for tolerance, (labels, _) in results.items():
            for i in starts:
                expected, ret = oracle(prices[i:], present[i:], good[i:], tolerance)
                assert labels.outcome[i] == expected
                np.testing.assert_allclose(labels.conservative[i], ret, equal_nan=True)


@pytest.mark.parametrize("length", [1, 2, 5, 6])
def test_exact_tolerance_cutoffs_and_multiple_separate_gaps(length):
    prices = np.full((60, 4), 100.0)
    present = np.ones(60, dtype=bool)
    present[10 : 10 + length] = False
    present[30 : 30 + length] = False
    prices[~present] = np.nan
    results, maximum, missing, *_ = evaluate(prices, present, present, np.array([0]))
    assert maximum[0] == length and missing[0] == 2 * length
    for tolerance, (labels, _) in results.items():
        expected = Outcome.EXPIRE if tolerance is None or tolerance >= length else Outcome.UNKNOWN
        assert labels.outcome[0] == expected
