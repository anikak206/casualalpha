"""Tests for walk-forward splits and feature alignment (Phase 4)."""

import numpy as np
import pandas as pd

from src.backtest import backtest_strategy, portfolio_returns_from_positions
from src.strategy import stable_edges


def _make_returns(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.normal(0, 0.01, size=(n, 3))
    return pd.DataFrame(data, index=idx, columns=["A", "B", "C"])


def test_walk_forward_no_train_test_overlap():
    r = _make_returns(300)
    out = backtest_strategy(r, train_size=100, test_size=30, cost_bps=1.0)
    pos = out["positions"]
    # positions should be zero in first train window
    assert (pos.iloc[:100].abs().sum().sum() == 0)
    # function runs and returns aligned frame
    assert pos.shape == r.shape


def test_portfolio_returns_alignment():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    pos = pd.DataFrame([[1.0, 0], [1.0, 0], [0, 1.0], [0, 1.0], [0, 0]], index=idx, columns=["A", "B"])
    rets = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04, 0.05],
                         "B": [0.05, 0.04, 0.03, 0.02, 0.01]}, index=idx)
    net = portfolio_returns_from_positions(pos, rets, cost_bps=0.0)
    # return on day 1 (index1) should use position from day0 * return day1
    assert np.isclose(net.iloc[1], (1.0 * 0.02 + 0.0 * 0.04) / 2)


def test_stable_edges_returns_list():
    r = _make_returns(120)
    edges = stable_edges(r)
    assert isinstance(edges, list)
