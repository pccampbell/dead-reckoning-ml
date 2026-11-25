"""Independent minute-loop oracle and boundary cases for the reusable label engine."""

import numpy as np
import pytest

from dead_reckoning.config import LabelStudyConfig
from dead_reckoning.labels.barriers import Outcome, first_touches, resolve, valid_ohlc
from dead_reckoning.research.data import decision_grid, sampled_sessions


def reference(prices, tp, sl, horizon):
    entry = prices[0, 0]
    upper, lower = entry * (1 + tp), entry * (1 - sl)
    for k, bar in enumerate(prices[:horizon]):
        if not valid_ohlc(bar[None, :])[0]:
            return Outcome.UNKNOWN, np.nan, np.nan, np.nan
        o, h, low, _ = bar
        if o >= upper:
            return Outcome.TP_FIRST, tp, tp, k
        if o <= lower:
            return Outcome.SL_FIRST, o / entry - 1, o / entry - 1, k
        if h >= upper and low <= lower:
            return Outcome.AMBIGUOUS, -sl, tp, k + 1
        if h >= upper:
            return Outcome.TP_FIRST, tp, tp, k + 1
        if low <= lower:
            return Outcome.SL_FIRST, -sl, -sl, k + 1
    ret = prices[horizon - 1, 3] / entry - 1
    return Outcome.EXPIRE, ret, ret, horizon


@pytest.mark.parametrize(
    "name,offset,bar,expected,ret",
    [
        ("tp", 1, [100, 102, 100, 101], Outcome.TP_FIRST, 0.01),
        ("sl", 1, [100, 100, 98, 99], Outcome.SL_FIRST, -0.01),
        ("entry_touch", 0, [100, 102, 100, 101], Outcome.TP_FIRST, 0.01),
        ("final_touch", 2, [100, 102, 100, 101], Outcome.TP_FIRST, 0.01),
        ("after_expiration", 3, [100, 102, 100, 101], Outcome.EXPIRE, 0),
        ("ambiguous", 1, [100, 102, 98, 100], Outcome.AMBIGUOUS, -0.01),
        ("gap_tp_resolves_both", 1, [102, 103, 98, 100], Outcome.TP_FIRST, 0.01),
        ("gap_sl_resolves_both", 1, [98, 102, 97, 100], Outcome.SL_FIRST, -0.02),
        ("expire", 2, [100, 100.5, 100, 100.4], Outcome.EXPIRE, 0.004),
    ],
)
def test_named_paths(name, offset, bar, expected, ret):
    prices = np.full((5, 4), 100.0)
    prices[offset] = bar
    t = first_touches(prices, np.array([0]), [3, 5], [0.01], [0.01])
    result = resolve(t, 0.01, 0.01, 3)
    assert result.outcome[0] == expected, name
    assert result.conservative[0] == pytest.approx(ret)


@pytest.mark.parametrize(
    "missing,touch,expected",
    [
        (0, 2, Outcome.UNKNOWN),
        (1, 2, Outcome.UNKNOWN),
        (2, 0, Outcome.TP_FIRST),
        (3, 2, Outcome.TP_FIRST),
        (1, None, Outcome.UNKNOWN),
        (3, None, Outcome.EXPIRE),
    ],
)
def test_missing_before_after_exit(missing, touch, expected):
    prices = np.full((5, 4), 100.0)
    prices[missing] = np.nan
    if touch is not None:
        prices[touch, 1] = 102
    t = first_touches(prices, np.array([0]), [3, 5], [0.01], [0.01])
    assert resolve(t, 0.01, 0.01, 3).outcome[0] == expected
    if expected == Outcome.TP_FIRST and missing < 3:
        assert np.isnan(t.forward[3][0])  # fixed horizon still needs its entire window


def test_randomized_reference_equivalence_and_monotonicity():
    rng = np.random.default_rng(42)
    for _ in range(60):
        opens = 100 * np.exp(rng.normal(0, 0.006, 50).cumsum())
        closes = opens * np.exp(rng.normal(0, 0.003, 50))
        highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.012, 50))
        lows = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.012, 50))
        prices = np.stack([opens, highs, lows, closes], axis=1)
        if rng.random() < 0.5:
            prices[rng.integers(0, 50)] = np.nan
        starts = np.arange(20)
        horizons = [5, 15, 30]
        t = first_touches(prices, starts, horizons, [0.005, 0.01], [0.004, 0.01])
        for tp, sl in [(0.005, 0.004), (0.01, 0.01)]:
            unresolved = []
            for horizon in horizons:
                result = resolve(t, tp, sl, horizon)
                for i, start in enumerate(starts):
                    expected = reference(prices[start:], tp, sl, horizon)
                    assert result.outcome[i] == expected[0]
                    np.testing.assert_allclose(
                        [result.conservative[i], result.optimistic[i], result.holding[i]],
                        expected[1:],
                        equal_nan=True,
                        atol=1e-14,
                    )
                fully_observed = t.first_bad >= 30
                unresolved.append(np.sum(result.outcome[fully_observed] == Outcome.EXPIRE))
            assert unresolved == sorted(unresolved, reverse=True)


def test_execution_grid_dst_early_closes_and_no_overnight():
    options = LabelStudyConfig(start="2019-01-01", end="2020-01-01", sessions_per_month=23)
    sessions = sampled_sessions(options)
    assert sampled_sessions(options).equals(sessions)
    by_day = {str(row["date"]): row for row in sessions.iter_rows(named=True)}
    assert "2019-07-04" not in by_day
    assert by_day["2019-11-29"]["close"].hour == 18
    assert by_day["2019-01-02"]["open"].hour == 14
    assert by_day["2019-07-03"]["open"].hour == 13
    for minutes in [390, 210]:
        decisions, entries = decision_grid(minutes, options)
        assert (entries == decisions + 1).all()
        assert (entries + 120 <= minutes).all()
        assert entries[6] == 31  # 10:00 decision -> 10:01 entry; 60m -> 11:00 close
    with pytest.raises(ValueError, match="single scheduled session"):
        first_touches(np.full((20, 4), 100.0), np.array([10]), [15], [0.01], [0.01])


def test_invalid_prices_and_duplicate_validity_mask():
    for bad in [[100, 99, 100, 100], [0, 1, 0, 1], [100, np.inf, 100, 100]]:
        prices = np.full((5, 4), 100.0)
        prices[1] = bad
        t = first_touches(prices, np.array([0]), [5], [0.01], [0.01])
        assert resolve(t, 0.01, 0.01, 5).outcome[0] == Outcome.UNKNOWN
    t = first_touches(
        np.full((5, 4), 100.0),
        np.array([0]),
        [5],
        [0.01],
        [0.01],
        np.array([True, False, True, True, True]),
    )
    assert resolve(t, 0.01, 0.01, 5).outcome[0] == Outcome.UNKNOWN
