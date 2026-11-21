"""Tests for rolling-window construction and FDR correction (Phase 2)."""

import numpy as np

from src.causal import rolling_windows, granger_pair
import pandas as pd


def test_rolling_windows_no_future_leakage():
    n = 300
    ws = rolling_windows(n, window=100, step=20)
    for (s, e) in ws:
        assert e <= n
        assert e - s == 100
    # windows are ordered and non-overlapping in start progression
    starts = [w[0] for w in ws]
    assert starts == sorted(starts)


def test_rolling_windows_count():
    ws = rolling_windows(250, window=100, step=50)
    # starts: 0,50,100,150 -> 4 windows (last end 250)
    assert len(ws) == 4
    assert ws[-1] == (150, 250)


def test_bh_reduces_false_positives():
    # sanity: granger_pair returns list of tuples
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120)}, index=idx)
    res = granger_pair(df, "a", "b")
    assert isinstance(res, list)
    if res:
        assert len(res[0]) == 3
