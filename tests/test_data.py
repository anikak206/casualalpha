"""Tests for data alignment and leakage properties."""

import numpy as np
import pandas as pd

from src.data import align_and_clean, compute_returns, remove_bad_ticks


def test_ffill_limit_respected():
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    df = pd.DataFrame({"A": [1.0, np.nan, np.nan, np.nan, np.nan, 2.0]}, index=idx)
    out = align_and_clean(df)
    # with limit=3, a 4-nan gap cannot be fully filled -> row dropped
    assert out.shape[0] < df.shape[0]


def test_compute_returns_no_lookahead():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    prices = pd.DataFrame({"A": [100, 101, 102, 103, 104]}, index=idx, dtype=float)
    rets = compute_returns(prices)
    # return at t uses prices t and t-1 only
    expected = np.log(101 / 100)
    assert np.isclose(rets["A"].iloc[0], expected)
    # changing future price does not alter past return
    prices2 = prices.copy()
    prices2.iloc[-1, 0] = 999
    rets2 = compute_returns(prices2)
    assert np.allclose(rets["A"].iloc[:-1], rets2["A"].iloc[:-1])


def test_remove_bad_ticks_flags_extreme():
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    df = pd.DataFrame({"A": [100.0, 300.0, 301.0]}, index=idx)
    out = remove_bad_ticks(df)
    assert pd.isna(out["A"].iloc[1])
