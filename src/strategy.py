"""Phase 4: signal construction from (past) stable edges."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests

from src.causal import granger_pair


def stable_edges(train_returns: pd.DataFrame, alpha: float = 0.05) -> List[Tuple[str, str, int]]:
    """Return list of (leader, follower, lag) edges BH-significant in training window."""
    markets = list(train_returns.columns)
    tests = []
    for leader in markets:
        for follower in markets:
            if leader == follower:
                continue
            gr = granger_pair(train_returns, leader, follower)
            for (lag, f, p) in gr:
                tests.append((leader, follower, lag, f, p))
    if not tests:
        return []
    pvals = [t[4] for t in tests]
    reject, pvals_bh, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    out = []
    for idx, (leader, follower, lag, f, p) in enumerate(tests):
        if reject[idx]:
            out.append((leader, follower, lag))
    # deduplicate: keep best lag (min p) per (leader, follower)
    best = {}
    for (leader, follower, lag) in out:
        key = (leader, follower)
        # if multiple lags, keep first (tests ordered by lag)
        if key not in best:
            best[key] = lag
    return [(l, f, lag) for (l, f), lag in best.items()]


def build_signals(returns: pd.DataFrame, edges: List[Tuple[str, str, int]]) -> pd.DataFrame:
    """Build per-market signal at time t using data up to t (no look-ahead).

    signal_{follower,t} = sum over edges of leader's lagged return.
    Returns DataFrame aligned with returns index.
    """
    sig = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    for (leader, follower, lag) in edges:
        # leader return at t-lag predicts follower at t+1; signal at t uses lagged leader
        sig[follower] = sig[follower] + returns[leader].shift(lag)
    return sig


def signals_to_positions(signals: pd.DataFrame) -> pd.DataFrame:
    pos = np.sign(signals).fillna(0.0)
    # clip to [-1, 1] (already)
    return pos


def momentum_positions(returns: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    mom = returns.rolling(lookback).sum()
    return np.sign(mom).fillna(0.0)


def mean_reversion_positions(returns: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    mom = returns.rolling(lookback).sum()
    return (-np.sign(mom)).fillna(0.0)


def random_positions(returns: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = pd.DataFrame(
        rng.choice([-1.0, 1.0], size=returns.shape),
        index=returns.index,
        columns=returns.columns,
    )
    return out
