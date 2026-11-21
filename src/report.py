"""Phase 6: regenerate README auto-generated tables from results/."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import project_root


def build_tables() -> str:
    root = project_root()
    res = root / "results"
    parts = []
    parts.append("<!-- AUTO-GENERATED:START -->")
    parts.append("## Auto-generated results (do not edit by hand)\n")

    # Data summary
    p = res / "data_summary.csv"
    if p.exists():
        df = pd.read_csv(p)
        parts.append("### Data summary\n")
        parts.append(df.to_markdown(index=False))
        parts.append("")

    # ADF summary
    p = res / "adf_summary.csv"
    if p.exists():
        df = pd.read_csv(p)
        parts.append("### ADF stationarity tests (5% level)\n")
        parts.append(df[["market", "adf_statistic", "p_value", "stationary_5pct"]].to_markdown(index=False))
        parts.append("")

    # Edge stability top 10
    p = res / "edge_stability.csv"
    if p.exists():
        df = pd.read_csv(p).head(10)
        parts.append("### Top 10 edges by stability (BH)\n")
        parts.append(df[["edge", "n_windows_sig_bh", "stability_bh", "stability_raw"]].to_markdown(index=False))
        parts.append("")

    # Significance counts summary
    p = res / "significance_counts.csv"
    if p.exists():
        df = pd.read_csv(p)
        parts.append("### Multiple-testing inflation\n")
        parts.append(f"- Windows: {len(df)}")
        parts.append(f"- Mean edges/window (raw): {df['n_edges_raw'].mean():.2f}")
        parts.append(f"- Mean edges/window (BH): {df['n_edges_bh'].mean():.2f}")
        parts.append("")

    # Regime counts
    p = res / "regime_counts.csv"
    if p.exists():
        df = pd.read_csv(p)
        parts.append("### Regime counts\n")
        parts.append(df.to_markdown(index=False))
        parts.append("")

    # Strategy comparison
    p = res / "strategy_comparison.csv"
    if p.exists():
        df = pd.read_csv(p, index_col=0)
        parts.append("### Strategy vs baselines (out-of-sample)\n")
        parts.append(df[["ann_return", "ann_vol", "sharpe", "max_drawdown", "turnover", "hit_rate"]].to_markdown())
        parts.append("")

    # Validation summary
    p = res / "validation_summary.csv"
    if p.exists():
        df = pd.read_csv(p)
        parts.append("### Validation summary\n")
        parts.append(df.to_markdown(index=False))
        parts.append("")

    # Verdict
    p = res / "verdict.txt"
    if p.exists():
        parts.append("### Verdict\n")
        parts.append("```\n" + p.read_text() + "\n```")
        parts.append("")

    parts.append("<!-- AUTO-GENERATED:END -->")
    return "\n".join(parts)


def run_phase6_report() -> None:
    root = project_root()
    readme = root / "README.md"
    text = readme.read_text() if readme.exists() else ""
    block = build_tables()
    start = text.find("<!-- AUTO-GENERATED:START -->")
    end = text.find("<!-- AUTO-GENERATED:END -->")
    if start != -1 and end != -1:
        end = text.find("-->", end) + 3
        text = text[:start] + block + text[end:]
    else:
        text = text + "\n\n" + block
    readme.write_text(text)
    print("README auto-generated tables updated.")


if __name__ == "__main__":
    run_phase6_report()
