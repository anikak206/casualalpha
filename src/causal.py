"""Phase 2: rolling Granger-causality lead-lag graph with FDR correction."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests

from src.data import project_root

WINDOW: int = 252
STEP: int = 21
MAX_LAG: int = 5
ALPHA: float = 0.05


def load_returns() -> pd.DataFrame:
    p = project_root() / "data" / "processed" / "returns.parquet"
    return pd.read_parquet(p)


def rolling_windows(n: int, window: int = WINDOW, step: int = STEP) -> List[slice]:
    """Return list of (start, end) index positions for rolling windows."""
    out = []
    start = 0
    while start + window <= n:
        out.append((start, start + window))
        start += step
    return out


def granger_pair(df_window: pd.DataFrame, leader: str, follower: str, max_lag: int = MAX_LAG):
    """Run Granger causality: does leader predict follower? Returns list of (lag, f_stat, p_value)."""
    data = df_window[[follower, leader]].dropna()
    if len(data) < max_lag + 10:
        return []
    try:
        res = grangercausalitytests(data, maxlag=max_lag)
    except Exception:
        return []
    out = []
    for lag in range(1, max_lag + 1):
        test = res[lag]
        f_stat, p_value = test[0]["ssr_ftest"][0], test[0]["ssr_ftest"][1]
        out.append((lag, float(f_stat), float(p_value)))
    return out


def run_phase2() -> dict:
    root = project_root()
    (root / "results").mkdir(parents=True, exist_ok=True)
    returns = load_returns()
    markets = list(returns.columns)
    windows = rolling_windows(len(returns))

    records = []  # per (window, leader, follower, lag)
    window_counts = []

    for (s, e) in windows:
        wdf = returns.iloc[s:e]
        w_start = returns.index[s]
        w_end = returns.index[e - 1]

        # collect all (leader, follower, lag) tests
        tests = []
        for leader in markets:
            for follower in markets:
                if leader == follower:
                    continue
                gr = granger_pair(wdf, leader, follower)
                for (lag, f, p) in gr:
                    tests.append((leader, follower, lag, f, p))

        if not tests:
            continue

        pvals = [t[4] for t in tests]
        reject, pvals_bh, _, _ = multipletests(pvals, alpha=ALPHA, method="fdr_bh")

        for idx, (leader, follower, lag, f, p) in enumerate(tests):
            records.append(
                {
                    "window_start": w_start.date().isoformat(),
                    "window_end": w_end.date().isoformat(),
                    "leader": leader,
                    "follower": follower,
                    "lag": lag,
                    "f_stat": f,
                    "p_value": p,
                    "p_value_bh": float(pvals_bh[idx]),
                    "significant_raw": bool(p < ALPHA),
                    "significant_bh": bool(reject[idx]),
                }
            )

        # edge-level: any lag significant
        edge_raw = set()
        edge_bh = set()
        for idx, (leader, follower, lag, f, p) in enumerate(tests):
            if p < ALPHA:
                edge_raw.add((leader, follower))
            if reject[idx]:
                edge_bh.add((leader, follower))
        window_counts.append(
            {
                "window_start": w_start.date().isoformat(),
                "window_end": w_end.date().isoformat(),
                "n_tests": len(tests),
                "n_edges_raw": len(edge_raw),
                "n_edges_bh": len(edge_bh),
            }
        )

    tests_df = pd.DataFrame(records)
    tests_df.to_csv(root / "results" / "granger_tests.csv", index=False)
    counts_df = pd.DataFrame(window_counts)
    counts_df.to_csv(root / "results" / "significance_counts.csv", index=False)

    # Edge stability: fraction of windows where edge significant (BH), aggregated over lags
    # An edge is significant in a window if any lag significant (BH).
    windows_list = counts_df["window_start"].tolist()
    # build per-window edge sets already; recompute from tests_df
    edge_window = (
        tests_df[tests_df["significant_bh"]]
        .groupby(["window_start", "leader", "follower"])
        .size()
        .reset_index(name="n_lags")
    )
    n_windows = len(windows_list)
    stab_rows = []
    for leader in markets:
        for follower in markets:
            if leader == follower:
                continue
            wins = edge_window[
                (edge_window["leader"] == leader) & (edge_window["follower"] == follower)
            ]["window_start"].nunique()
            raw_wins = (
                tests_df[(tests_df["leader"] == leader) & (tests_df["follower"] == follower) & (tests_df["significant_raw"])]
                .groupby("window_start")
                .size()
            )
            # count windows with any raw significant lag
            raw_n = tests_df[
                (tests_df["leader"] == leader)
                & (tests_df["follower"] == follower)
                & (tests_df["significant_raw"])
            ]["window_start"].nunique()
            # effect size: mean f_stat over significant bh lags (or all)
            eff = tests_df[
                (tests_df["leader"] == leader)
                & (tests_df["follower"] == follower)
                & (tests_df["significant_bh"])
            ]["f_stat"].mean()
            stab_rows.append(
                {
                    "leader": leader,
                    "follower": follower,
                    "edge": f"{leader} -> {follower}",
                    "n_windows_total": n_windows,
                    "n_windows_sig_raw": int(raw_n),
                    "n_windows_sig_bh": int(wins),
                    "stability_raw": raw_n / n_windows if n_windows else 0.0,
                    "stability_bh": wins / n_windows if n_windows else 0.0,
                    "mean_f_stat_sig": float(eff) if not np.isnan(eff) else 0.0,
                }
            )
    stab_df = pd.DataFrame(stab_rows).sort_values("stability_bh", ascending=False)
    stab_df.to_csv(root / "results" / "edge_stability.csv", index=False)

    # Heatmap of stability (BH)
    mat = pd.DataFrame(0.0, index=markets, columns=markets)
    for _, r in stab_df.iterrows():
        mat.loc[r["leader"], r["follower"]] = r["stability_bh"]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1, ax=ax)
    ax.set_title("Edge stability (fraction of rolling windows significant, BH)")
    fig.tight_layout()
    fig.savefig(root / "results" / "edge_stability_heatmap.png", dpi=120)
    plt.close(fig)

    # Example graphs: adjacency for first, middle, last window
    selected = [windows_list[0], windows_list[len(windows_list) // 2], windows_list[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, ws in zip(axes, selected):
        sub = tests_df[(tests_df["window_start"] == ws) & (tests_df["significant_bh"])]
        edges = set(zip(sub["leader"], sub["follower"]))
        adj = pd.DataFrame(0, index=markets, columns=markets)
        for (l, f) in edges:
            adj.loc[l, f] = 1
        sns.heatmap(adj, cmap="Blues", cbar=False, ax=ax, linewidths=0.5)
        ax.set_title(f"Sig edges (BH)\n{ws}")
    fig.tight_layout()
    fig.savefig(root / "results" / "example_graphs.png", dpi=120)
    plt.close(fig)

    # Summary counts
    total_edges = len(markets) * (len(markets) - 1)
    raw_sig_edges = int((stab_df["n_windows_sig_raw"] > 0).sum())
    bh_sig_edges = int((stab_df["n_windows_sig_bh"] > 0).sum())
    print(f"Windows: {n_windows}")
    print(f"Total directed edges: {total_edges}")
    print(f"Edges significant in >=1 window (raw): {raw_sig_edges}")
    print(f"Edges significant in >=1 window (BH): {bh_sig_edges}")
    print(f"Mean edges/window raw: {counts_df['n_edges_raw'].mean():.2f}")
    print(f"Mean edges/window BH: {counts_df['n_edges_bh'].mean():.2f}")
    print("Top 10 edges by stability (BH):")
    print(stab_df.head(10).to_string(index=False))

    return {
        "tests_df": tests_df,
        "stab_df": stab_df,
        "counts_df": counts_df,
        "n_windows": n_windows,
    }


if __name__ == "__main__":
    run_phase2()
