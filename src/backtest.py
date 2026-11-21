"""Phase 4: backtesting engine and metrics."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.strategy import (
    build_signals,
    signals_to_positions,
    momentum_positions,
    mean_reversion_positions,
    random_positions,
    stable_edges,
)


COST_BPS_DEFAULT: float = 2.0


def portfolio_returns_from_positions(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = COST_BPS_DEFAULT,
) -> pd.Series:
    """Compute daily net portfolio return (equal-weight across markets).

    positions at t earn returns at t+1. Cost applied on position changes.
    """
    # align
    positions = positions.reindex(returns.index).fillna(0.0)
    # gross: position_t * return_{t+1}? We'll shift returns back: use return at same day as position next day.
    # Convention: positions indexed by decision date t; return realized t+1. We store portfolio return at t+1.
    gross = (positions * returns).mean(axis=1)  # same-day (will shift)
    # Actually positions at t should multiply returns at t+1. Shift positions down by one day.
    pos_shifted = positions.shift(1).fillna(0.0)
    gross = (pos_shifted * returns).mean(axis=1)
    # costs on changes in positions (decision dates), applied next day
    changes = positions.diff().abs().sum(axis=1)
    n_markets = positions.shape[1]
    turnover_cost = (cost_bps / 10000.0) * (changes / n_markets)
    # align cost to return day (t+1)
    cost_series = turnover_cost.shift(1).fillna(0.0)
    net = gross - cost_series
    net.name = "portfolio_return"
    return net


def max_drawdown(cumulative: pd.Series) -> float:
    peak = cumulative.cummax()
    dd = (cumulative - peak) / peak
    return float(dd.min())


def compute_metrics(returns: pd.Series, positions: pd.DataFrame = None) -> Dict[str, float]:
    r = returns.dropna()
    if len(r) == 0:
        return {}
    ann_ret = float(r.mean() * 252)
    ann_vol = float(r.std(ddof=1) * np.sqrt(252))
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if r.std(ddof=1) > 0 else 0.0
    cum = (1 + r).cumprod()
    mdd = max_drawdown(cum)
    # turnover
    if positions is not None:
        changes = positions.diff().abs().sum(axis=1)
        turnover = float((changes / positions.shape[1]).mean())
    else:
        turnover = 0.0
    hit_rate = float((r > 0).mean())
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "turnover": turnover,
        "hit_rate": hit_rate,
        "n_days": int(len(r)),
    }


def bootstrap_ci(returns: pd.Series, n_boot: int = 1000, seed: int = 42, metric: str = "sharpe") -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    arr = returns.dropna().values
    n = len(arr)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        if metric == "sharpe":
            s = sample.mean() / sample.std(ddof=1) * np.sqrt(252) if sample.std(ddof=1) > 0 else 0.0
        elif metric == "ann_return":
            s = sample.mean() * 252
        else:
            s = sample.mean()
        vals.append(s)
    lo = float(np.percentile(vals, 2.5))
    hi = float(np.percentile(vals, 97.5))
    return lo, hi


from typing import Tuple


def backtest_strategy(
    returns: pd.DataFrame,
    train_size: int = 756,
    test_size: int = 63,
    cost_bps: float = COST_BPS_DEFAULT,
    seed: int = 42,
) -> Dict[str, object]:
    """Walk-forward backtest. Returns dict with portfolio returns series and positions."""
    n = len(returns)
    all_net = pd.Series(index=returns.index, dtype=float)
    all_pos = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    # initial positions zero until first test
    train_end = train_size
    while train_end + test_size <= n:
        train = returns.iloc[:train_end]
        test_idx = returns.index[train_end : train_end + test_size]
        edges = stable_edges(train)
        # build signals for a window that includes train tail + test for lag availability
        # Use returns up to test end
        window_returns = returns.iloc[: train_end + test_size]
        sigs = build_signals(window_returns, edges)
        pos = signals_to_positions(sigs)
        # extract test positions
        test_pos = pos.loc[test_idx]
        all_pos.loc[test_idx] = test_pos
        train_end += test_size

    # fill any remaining dates with 0 (no position)
    all_pos = all_pos.fillna(0.0)
    # compute net returns over whole series (positions only where set)
    net = portfolio_returns_from_positions(all_pos, returns, cost_bps=cost_bps)
    return {"returns": net, "positions": all_pos, "cost_bps": cost_bps}


def run_baselines(returns: pd.DataFrame, cost_bps: float = COST_BPS_DEFAULT, seed: int = 42) -> Dict[str, pd.Series]:
    out = {}
    # buy and hold
    bh_pos = pd.DataFrame(1.0, index=returns.index, columns=returns.columns)
    out["buy_hold"] = portfolio_returns_from_positions(bh_pos, returns, cost_bps=0.0)
    # momentum
    mom_pos = momentum_positions(returns)
    out["momentum"] = portfolio_returns_from_positions(mom_pos, returns, cost_bps=cost_bps)
    # mean reversion
    mr_pos = mean_reversion_positions(returns)
    out["mean_reversion"] = portfolio_returns_from_positions(mr_pos, returns, cost_bps=cost_bps)
    # random
    rnd_pos = random_positions(returns, seed=seed)
    out["random"] = portfolio_returns_from_positions(rnd_pos, returns, cost_bps=cost_bps)
    return out
