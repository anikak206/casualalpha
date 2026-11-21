"""Phase 1: data download, cleaning, alignment, returns, ADF tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

# --- constants ---------------------------------------------------------------
TICKERS: Dict[str, str] = {
    "Brent": "BZ=F",
    "WTI": "CL=F",
    "Gold": "GC=F",
    "Copper": "HG=F",
    "NatGas": "NG=F",
    "SP500": "ES=F",
    "US10Y": "ZN=F",
    "Dollar": "DX-Y.NYB",
    "VIX": "^VIX",
}

FFILL_LIMIT: int = 3          # max consecutive forward-fills after a gap
BAD_TICK_RET_THRESHOLD: float = 0.5  # |daily log return| above this -> treat as bad tick
START_DATE: str = "2010-01-01"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def raw_cache_path() -> Path:
    return project_root() / "data" / "raw" / "prices_raw.parquet"


def processed_path() -> Path:
    return project_root() / "data" / "processed" / "prices.parquet"


def returns_path() -> Path:
    return project_root() / "data" / "processed" / "returns.parquet"


def download_raw_prices(
    start: str = START_DATE,
    end: Optional[str] = None,
    force: bool = False,
) -> pd.DataFrame:
    """Download daily close prices for all tickers, cache to parquet."""
    cache = raw_cache_path()
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    frames = {}
    for name, sym in TICKERS.items():
        df = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
        if df is None or df.empty:
            raise RuntimeError(f"No data returned for {name} ({sym})")
        # yfinance may return MultiIndex columns; flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        frames[name] = close

    raw = pd.concat(frames, axis=1, sort=False)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw = raw.sort_index()
    cache.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(cache)
    return raw


def remove_bad_ticks(prices: pd.DataFrame) -> pd.DataFrame:
    """Set prices on days with extreme daily moves to NaN (suspected bad ticks)."""
    out = prices.copy()
    # non-positive prices cannot be log-transformed; treat as missing
    out = out.mask(out <= 0)
    logret = np.log(out).diff()
    bad = logret.abs() > BAD_TICK_RET_THRESHOLD
    out = out.mask(bad)
    return out


def align_and_clean(prices: pd.DataFrame) -> pd.DataFrame:
    """Remove bad ticks, forward-fill small gaps (limit), drop remaining NaNs."""
    cleaned = remove_bad_ticks(prices)
    # union of all dates already present in index; ffill bounded gaps
    cleaned = cleaned.ffill(limit=FFILL_LIMIT)
    cleaned = cleaned.dropna(how="any")
    return cleaned


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff().dropna()


def adf_summary(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in returns.columns:
        s = returns[col].dropna()
        stat, pval, usedlag, nobs, crit, icbest = adfuller(s, autolag="AIC", result_object=False)
        rows.append(
            {
                "market": col,
                "adf_statistic": stat,
                "p_value": pval,
                "n_obs": nobs,
                "used_lag": usedlag,
                "crit_1pct": crit.get("1%"),
                "crit_5pct": crit.get("5%"),
                "crit_10pct": crit.get("10%"),
                "stationary_5pct": bool(pval < 0.05),
            }
        )
    return pd.DataFrame(rows)


def data_summary(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in prices.columns:
        p = prices[col]
        r = returns[col]
        rows.append(
            {
                "market": col,
                "ticker": TICKERS[col],
                "n_prices": int(p.notna().sum()),
                "start": p.index.min().date().isoformat(),
                "end": p.index.max().date().isoformat(),
                "price_mean": float(p.mean()),
                "price_min": float(p.min()),
                "price_max": float(p.max()),
                "return_mean": float(r.mean()),
                "return_std": float(r.std()),
                "return_min": float(r.min()),
                "return_max": float(r.max()),
                "n_returns": int(r.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def make_correlation_heatmap(returns: pd.DataFrame, out_path: Path) -> None:
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Daily log-return correlation (2010-present)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_phase1(force_download: bool = False) -> dict:
    root = project_root()
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)

    raw = download_raw_prices(force=force_download)
    prices = align_and_clean(raw)
    prices.to_parquet(processed_path())
    rets = compute_returns(prices)
    rets.to_parquet(returns_path())

    dsum = data_summary(prices, rets)
    dsum.to_csv(root / "results" / "data_summary.csv", index=False)

    adf = adf_summary(rets)
    adf.to_csv(root / "results" / "adf_summary.csv", index=False)

    make_correlation_heatmap(rets, root / "results" / "correlation_heatmap.png")

    return {
        "prices": prices,
        "returns": rets,
        "data_summary": dsum,
        "adf_summary": adf,
        "raw_shape": raw.shape,
        "clean_shape": prices.shape,
    }


if __name__ == "__main__":
    out = run_phase1()
    print("Raw shape:", out["raw_shape"])
    print("Clean shape:", out["clean_shape"])
    print("\nData summary:\n", out["data_summary"].to_string(index=False))
    print("\nADF summary:\n", out["adf_summary"].to_string(index=False))
    print("\nSaved: data/processed/prices.parquet, returns.parquet")
    print("Saved: results/data_summary.csv, results/adf_summary.csv, results/correlation_heatmap.png")
