"""Phase 3: causal (filtered) Gaussian HMM regimes + regime-conditional edge stability."""

from __future__ import annotations

from pathlib import Path
from collections import Counter
from scipy.stats import multivariate_normal

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

from src.data import project_root
from src.causal import load_returns, rolling_windows, WINDOW, STEP

TRAIN_SIZE: int = 504   # ~2 years for initial HMM fit
N_STATES_CANDIDATES = [2, 3]


def build_features(returns: pd.DataFrame) -> pd.DataFrame:
    """Market-level features: VIX level, rolling vol of SP500, SP500 return."""
    vix = returns["VIX"] if "VIX" in returns else None
    # VIX level from prices? returns has VIX returns, not level. Load prices.
    prices = pd.read_parquet(project_root() / "data" / "processed" / "prices.parquet")
    vix_level = prices["VIX"]
    sp_ret = returns["SP500"]
    sp_vol = sp_ret.rolling(21).std() * np.sqrt(252)
    feat = pd.DataFrame(
        {
            "vix_level": vix_level,
            "sp500_return": sp_ret,
            "sp500_vol21": sp_vol,
        },
        index=returns.index,
    )
    # align lengths
    feat = feat.loc[returns.index]
    return feat


def fit_hmm(features_scaled: np.ndarray, n_states: int, seed: int = 42) -> GaussianHMM:
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=200,
        random_state=seed,
    )
    model.fit(features_scaled)
    return model


def bic(model: GaussianHMM, X: np.ndarray) -> float:
    # BIC = -2 logL + k log n
    logL = model.score(X)
    n = X.shape[0]
    # number of params: startprob (k-1) + transmat (k*(k-1)) + means (k*d) + covars (k*d)
    k = model.n_components
    d = X.shape[1]
    n_params = (k - 1) + k * (k - 1) + k * d + k * d
    return -2 * logL + n_params * np.log(n)


def filtered_states(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Causal filtered state sequence using forward algorithm (no future)."""
    startprob = model.startprob_
    transmat = model.transmat_
    means = model.means_
    covars = model.covars_  # diag: (k, d)
    k = model.n_components
    n = X.shape[0]
    d = X.shape[1]
    # precompute emission log-probs
    # log N(x | mu, diag cov)
    # var = covars
    states = np.zeros(n, dtype=int)
    # alpha in log-space for stability
    log_alpha = np.zeros(k)
    for t in range(n):
        x = X[t]
        # emission log prob for each state (causal: uses x_t only)
        log_emiss = np.zeros(k)
        for j in range(k):
            try:
                log_emiss[j] = multivariate_normal.logpdf(x, mean=means[j], cov=covars[j], allow_singular=True)
            except Exception:
                log_emiss[j] = -1e10
        if t == 0:
            log_alpha = np.log(startprob + 1e-300) + log_emiss
        else:
            # log-sum-exp over i: log sum_i exp(log_alpha_prev[i] + log transmat[i,j])
            # transmat may be zero; handle
            new = np.zeros(k)
            for j in range(k):
                vals = log_alpha + np.log(transmat[:, j] + 1e-300)
                m = np.max(vals)
                new[j] = m + np.log(np.sum(np.exp(vals - m))) + log_emiss[j]
            log_alpha = new
        states[t] = int(np.argmax(log_alpha))
    return states


def run_phase3() -> dict:
    root = project_root()
    (root / "results").mkdir(parents=True, exist_ok=True)

    returns = load_returns()
    features = build_features(returns)
    # drop rows with NaN (rolling vol warmup)
    features = features.dropna()
    returns = returns.loc[features.index]

    # scaler fit on training window only
    train_feat = features.iloc[:TRAIN_SIZE]
    scaler = StandardScaler()
    scaler.fit(train_feat.values)
    X_all = scaler.transform(features.values)
    X_train = X_all[:TRAIN_SIZE]

    # choose n_states by BIC on training window
    best = None
    for ns in N_STATES_CANDIDATES:
        m = fit_hmm(X_train, ns)
        b = bic(m, X_train)
        if best is None or b < best[1]:
            best = (m, b, ns)
    model, best_bic, best_ns = best

    # filtered states for entire series (causal)
    states = filtered_states(model, X_all)
    regime_series = pd.Series(states, index=features.index, name="regime")

    # regime timeline plot
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(regime_series.index, regime_series.values, drawstyle="steps-post", lw=1)
    ax.set_title(f"Regime timeline (HMM, {best_ns} states, filtered)")
    ax.set_ylabel("Regime")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(root / "results" / "regime_timeline.png", dpi=120)
    plt.close(fig)

    # regime distribution
    regime_counts = regime_series.value_counts().sort_index()
    regime_counts.to_csv(root / "results" / "regime_counts.csv")

    # regime-conditional edge stability
    # load granger tests
    tests = pd.read_csv(root / "results" / "granger_tests.csv")
    tests["window_start"] = pd.to_datetime(tests["window_start"])
    tests["window_end"] = pd.to_datetime(tests["window_end"])

    # dominant regime per rolling window
    def dominant_regime(s, e):
        seg = regime_series.loc[s:e]
        if seg.empty:
            return np.nan
        return int(seg.mode().iloc[0])

    # build window list from tests
    windows = tests[["window_start", "window_end"]].drop_duplicates()
    window_regime = {}
    for _, r in windows.iterrows():
        window_regime[(r["window_start"], r["window_end"])] = dominant_regime(r["window_start"], r["window_end"])

    tests["regime"] = tests.apply(lambda r: window_regime.get((r["window_start"], r["window_end"])), axis=1)

    # edge significance by regime: for each regime, for each (leader,follower), fraction of windows in that regime where edge sig (BH)
    rows = []
    for regime in sorted(regime_series.unique()):
        sub_all = tests[tests["regime"] == regime]
        n_windows_regime = sub_all["window_start"].nunique()
        # edges
        for leader in returns.columns:
            for follower in returns.columns:
                if leader == follower:
                    continue
                sub = sub_all[(sub_all["leader"] == leader) & (sub_all["follower"] == follower)]
                # windows in this regime where any lag sig BH
                sig_windows = sub[sub["significant_bh"]]["window_start"].nunique()
                stab = sig_windows / n_windows_regime if n_windows_regime > 0 else 0.0
                rows.append(
                    {
                        "regime": int(regime),
                        "leader": leader,
                        "follower": follower,
                        "edge": f"{leader} -> {follower}",
                        "n_windows_regime": int(n_windows_regime),
                        "n_windows_sig_bh": int(sig_windows),
                        "stability_bh": float(stab),
                    }
                )
    reg_edge = pd.DataFrame(rows)
    reg_edge.to_csv(root / "results" / "regime_edge_stability.csv", index=False)

    # heatmap for top 15 edges by overall stability
    overall = pd.read_csv(root / "results" / "edge_stability.csv").sort_values("stability_bh", ascending=False).head(15)
    top_edges = list(overall["edge"])
    pivot = reg_edge[reg_edge["edge"].isin(top_edges)].pivot_table(
        index="edge", columns="regime", values="stability_bh", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1, ax=ax)
    ax.set_title("Regime x edge stability (top 15 edges)")
    fig.tight_layout()
    fig.savefig(root / "results" / "regime_edge_heatmap.png", dpi=120)
    plt.close(fig)

    # save regime series
    regime_series.to_frame().to_parquet(root / "data" / "processed" / "regimes.parquet")

    print(f"Chosen states: {best_ns} (BIC {best_bic:.1f})")
    print("Regime counts:\n", regime_counts.to_string())
    print("Regime-edge stability saved. Top 5 edges overall:")
    print(overall.head(5).to_string(index=False))

    return {
        "model": model,
        "n_states": best_ns,
        "regime_series": regime_series,
        "reg_edge": reg_edge,
    }


if __name__ == "__main__":
    run_phase3()
