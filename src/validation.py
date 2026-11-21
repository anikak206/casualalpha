"""Phase 4: validation utilities (DSR, PBO, purged k-fold)."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def purged_kfold_indices(n: int, k: int = 5, purge: int = 5, embargo: int = 5):
    """Yield (train_idx, test_idx) with purging and embargo.

    Splits indices [0..n) into k contiguous folds. For each test fold,
    remove training observations within `purge` of test boundaries and
    embargo `embargo` observations after the test fold.
    """
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1
    bounds = np.concatenate([[0], np.cumsum(fold_sizes)])
    for i in range(k):
        test_start = bounds[i]
        test_end = bounds[i + 1]
        test_idx = np.arange(test_start, test_end)
        # training = all other folds
        train_idx = np.concatenate(
            [np.arange(bounds[j], bounds[j + 1]) for j in range(k) if j != i]
        )
        # purge: remove train obs too close to test boundaries
        purge_mask = (train_idx < test_start - purge) | (train_idx > test_end - 1 + purge)
        # embargo: remove train obs within embargo after test end
        embargo_mask = train_idx > test_end - 1 + embargo
        # keep train obs that are NOT (purged or embargoed)
        # purge condition: distance to test <= purge
        dist_to_test = np.minimum(
            np.abs(train_idx - (test_start - 1)),
            np.abs(train_idx - test_end),
        )
        # Actually purge removes within `purge` lag of test set
        too_close = (train_idx >= test_start - purge) & (train_idx <= test_end - 1 + purge)
        embargoed = (train_idx > test_end - 1) & (train_idx <= test_end - 1 + embargo)
        keep = ~(too_close | embargoed)
        train_idx = train_idx[keep]
        yield train_idx, test_idx


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int) -> dict:
    r = returns.dropna()
    T = len(r)
    if T < 2 or r.std(ddof=1) == 0:
        return {"dsr": 0.0, "p_value": 0.5, "sr": 0.0}
    sr = float(r.mean() / r.std(ddof=1) * np.sqrt(252))
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    # variance of Sharpe estimator
    sr_var = (1.0 + (1.0 - skew) * sr / 6.0 + (kurt - 3.0) * sr ** 2 / 24.0) / (T - 1)
    sr_std = float(np.sqrt(max(sr_var, 1e-12)))
    # expected maximum Sharpe under null across n_trials
    from scipy.stats import norm
    emc = (1 - np.euler_gamma) * norm.ppf(1 - 1.0 / n_trials) + np.euler_gamma * norm.ppf(
        1 - 1.0 / (n_trials * np.e)
    )
    sr0 = emc * sr_std
    dsr = (sr - sr0) / sr_std
    p_value = float(norm.cdf(dsr))
    return {"dsr": float(dsr), "p_value": p_value, "sr": sr, "sr0": float(sr0), "n_trials": int(n_trials)}


def probability_of_backtest_overfitting(variant_returns: dict, n_blocks: int = 8) -> float:
    """CSCV-lite PBO: probability that in-sample best strategy is in bottom half OOS."""
    # align
    df = pd.DataFrame(variant_returns).dropna()
    if df.shape[1] < 2:
        return float("nan")
    T = len(df)
    block_size = T // n_blocks
    if block_size < 2:
        return float("nan")
    # trim to multiple of block_size
    Ttrim = block_size * n_blocks
    df = df.iloc[:Ttrim]
    block_ids = np.repeat(np.arange(n_blocks), block_size)
    n_half = n_blocks // 2
    bad = 0
    total = 0
    for in_blocks in combinations(range(n_blocks), n_half):
        in_set = set(in_blocks)
        in_mask = np.array([b in in_set for b in block_ids])
        out_mask = ~in_mask
        # in-sample sharpes
        in_sr = df[in_mask].mean() / df[in_mask].std(ddof=1)
        best = in_sr.idxmax()
        # out-of-sample ranks
        out_sr = df[out_mask].mean() / df[out_mask].std(ddof=1)
        # rank of best OOS (higher better). Bottom half if rank <= median
        rank = out_sr.rank(ascending=True)[best]
        median_rank = (len(out_sr) + 1) / 2.0
        if rank <= median_rank:
            bad += 1
        total += 1
    return float(bad / total) if total > 0 else float("nan")
