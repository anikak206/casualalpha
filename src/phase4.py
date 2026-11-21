"""Phase 4: strategy backtest, baselines, validation, verdict."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.backtest import (
    backtest_strategy,
    run_baselines,
    compute_metrics,
    bootstrap_ci,
)
from src.causal import load_returns
from src.validation import deflated_sharpe_ratio, probability_of_backtest_overfitting
from src.strategy import stable_edges, build_signals, signals_to_positions

ROOT = None


def load_returns_df() -> pd.DataFrame:
    from src.causal import load_returns
    return load_returns()


def run_phase4() -> dict:
    from src.data import project_root
    root = project_root()
    (root / "results").mkdir(parents=True, exist_ok=True)

    returns = load_returns_df()

    # Strategy (walk-forward)
    strat_out = backtest_strategy(returns, train_size=756, test_size=63, cost_bps=2.0)
    strat_ret = strat_out["returns"]
    strat_pos = strat_out["positions"]

    baselines = run_baselines(returns, cost_bps=2.0, seed=42)

    # Evaluate on the common out-of-sample period only (after first train window)
    oos_start = returns.index[756]
    strat_ret = strat_ret.loc[oos_start:]
    strat_pos = strat_pos.loc[oos_start:]
    baselines = {k: v.loc[oos_start:] for k, v in baselines.items()}

    # Metrics
    rows = []
    series_map = {"causal_strategy": strat_ret}
    series_map.update(baselines)

    positions_map = {"causal_strategy": strat_pos}

    for name, r in series_map.items():
        pos = positions_map.get(name)
        m = compute_metrics(r, positions=pos if pos is not None else None)
        # bootstrap CI for sharpe and ann_return
        ci_sharpe = bootstrap_ci(r, n_boot=500, seed=42, metric="sharpe")
        ci_ret = bootstrap_ci(r, n_boot=500, seed=42, metric="ann_return")
        m["sharpe_ci_lo"] = ci_sharpe[0]
        m["sharpe_ci_hi"] = ci_sharpe[1]
        m["ann_return_ci_lo"] = ci_ret[0]
        m["ann_return_ci_hi"] = ci_ret[1]
        m["strategy"] = name
        rows.append(m)

    comp = pd.DataFrame(rows).set_index("strategy")
    comp.to_csv(root / "results" / "strategy_comparison.csv")

    # Equity curves
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, r in series_map.items():
        cum = (1 + r.fillna(0)).cumprod()
        ax.plot(cum.index, cum.values, label=name)
    ax.set_title("Equity curves (walk-forward, net of costs)")
    ax.set_ylabel("Cumulative return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "results" / "equity_curves.png", dpi=120)
    plt.close(fig)

    # DSR
    n_trials = 72  # directed edges tested
    dsr_info = deflated_sharpe_ratio(strat_ret, n_trials=n_trials)

    # PBO
    variant_returns = {name: r for name, r in series_map.items()}
    pbo = probability_of_backtest_overfitting(variant_returns, n_blocks=8)

    # Verdict generated from numbers
    best_baseline = comp.drop(index="causal_strategy")["sharpe"].idxmax()
    best_baseline_sharpe = float(comp.drop(index="causal_strategy")["sharpe"].max())
    strat_sharpe = float(comp.loc["causal_strategy", "sharpe"])
    strat_mdd = float(comp.loc["causal_strategy", "max_drawdown"])
    if strat_sharpe > best_baseline_sharpe:
        verdict = (
            f"Strategy Sharpe ({strat_sharpe:.3f}) exceeds best baseline ({best_baseline}, "
            f"{best_baseline_sharpe:.3f}) out-of-sample. DSR p-value={dsr_info['p_value']:.3f}; "
            f"PBO={pbo:.3f}. Interpret with caution."
        )
    else:
        verdict = (
            f"Strategy did NOT beat baselines out-of-sample: causal Sharpe {strat_sharpe:.3f} vs "
            f"best baseline {best_baseline} {best_baseline_sharpe:.3f}. "
            f"DSR p-value={dsr_info['p_value']:.3f}; PBO={pbo:.3f}. "
            f"Negative result: the lead-lag signal does not survive costs/OOS under this protocol."
        )
    (root / "results" / "verdict.txt").write_text(verdict)

    # Save DSR/PBO summary
    pd.DataFrame([{"dsr": dsr_info["dsr"], "dsr_p_value": dsr_info["p_value"],
                   "n_trials": dsr_info["n_trials"], "pbo": pbo,
                   "best_baseline": best_baseline, "best_baseline_sharpe": best_baseline_sharpe,
                   "strategy_sharpe": strat_sharpe}]).to_csv(
        root / "results" / "validation_summary.csv", index=False
    )

    print(comp.to_string())
    print("\n" + verdict)
    print("\nDSR:", dsr_info)
    print("PBO:", pbo)

    return {
        "comparison": comp,
        "strat_returns": strat_ret,
        "baselines": baselines,
        "dsr": dsr_info,
        "pbo": pbo,
        "verdict": verdict,
    }


if __name__ == "__main__":
    run_phase4()
