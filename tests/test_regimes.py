"""Tests for causal (filtered) HMM regime assignment."""

import numpy as np

from src.regimes import fit_hmm, filtered_states


def test_filtered_states_causal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    model = fit_hmm(X[:40], 2)
    base = filtered_states(model, X)
    # alter future observations; past filtered states must be unchanged
    X2 = X.copy()
    X2[60:] += 5.0
    new = filtered_states(model, X2)
    assert np.array_equal(base[:60], new[:60])


def test_filtered_states_length():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 2))
    model = fit_hmm(X[:25], 2)
    st = filtered_states(model, X)
    assert st.shape == (50,)
    assert set(np.unique(st)).issubset({0, 1})
