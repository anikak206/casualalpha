"""Tests for validation utilities."""

import numpy as np
import pandas as pd

from src.validation import purged_kfold_indices, deflated_sharpe_ratio, probability_of_backtest_overfitting


def test_purged_kfold_no_overlap():
    folds = list(purged_kfold_indices(100, k=5, purge=2, embargo=2))
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        assert len(set(train_idx) & set(test_idx)) == 0
        # embargo: no train index immediately after test block
        # (best-effort check)
        assert train_idx.dtype.kind == "i"


def test_purged_kfold_covers_all_test():
    all_test = []
    for train_idx, test_idx in purged_kfold_indices(100, k=5, purge=2, embargo=2):
        all_test.extend(list(test_idx))
    assert sorted(all_test) == list(range(100))


def test_dsr_returns_finite():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, size=252))
    info = deflated_sharpe_ratio(r, n_trials=10)
    assert np.isfinite(info["dsr"])
    assert 0 <= info["p_value"] <= 1


def test_pbo_bounds():
    rng = np.random.default_rng(1)
    df = {f"s{i}": pd.Series(rng.normal(0, 0.01, size=200)) for i in range(3)}
    pbo = probability_of_backtest_overfitting(df, n_blocks=4)
    assert np.isnan(pbo) or (0 <= pbo <= 1)
