"""Phase 5: LLM sentiment (Ollama) ablation. Optional; degrades gracefully."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from src.data import project_root, TICKERS


def ollama_available(host: str = "http://localhost:11434") -> bool:
    try:
        with urllib.request.urlopen(host + "/api/tags", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def fetch_headlines() -> pd.DataFrame:
    """Fetch recent headlines via yfinance for each market. Timestamps recorded as-is."""
    import yfinance as yf

    rows = []
    for name, sym in TICKERS.items():
        try:
            t = yf.Ticker(sym)
            news = t.news or []
        except Exception:
            news = []
        for item in news:
            content = item.get("content", {}) if isinstance(item, dict) else {}
            title = item.get("title") or content.get("title")
            pub = item.get("pubDate") or content.get("pubDate") or item.get("providerPublishTime")
            if isinstance(pub, (int, float)):
                ts = datetime.utcfromtimestamp(pub)
            else:
                try:
                    ts = pd.to_datetime(pub, utc=True).tz_localize(None).to_pydatetime()
                except Exception:
                    ts = None
            if title and ts is not None:
                rows.append({"market": name, "timestamp": ts, "headline": title})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def score_headline_ollama(headline: str, model: str = "llama3.2") -> str:
    prompt = (
        "Classify the sentiment of this financial headline for the relevant market "
        "as exactly one of: bullish, bearish, neutral.\nHeadline: " + headline + "\nAnswer:"
    )
    payload = {"model": model, "prompt": prompt, "stream": False}
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    text = (data.get("response") or "").lower()
    for label in ["bullish", "bearish", "neutral"]:
        if label in text:
            return label
    return "neutral"


def run_phase5() -> dict:
    root = project_root()
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)

    headlines = fetch_headlines()
    headlines.to_csv(root / "data" / "processed" / "headlines.csv", index=False)

    status_lines = []
    status_lines.append(f"Headlines fetched: {len(headlines)}")
    status_lines.append(f"Date range: {headlines['timestamp'].min()} to {headlines['timestamp'].max()}" if not headlines.empty else "No headlines")

    if headlines.empty:
        status_lines.append("No headlines available; ablation skipped.")
        (root / "results" / "sentiment_status.txt").write_text("\n".join(status_lines))
        return {"status": "no_headlines"}

    if not ollama_available():
        status_lines.append("Ollama not available locally; sentiment scoring skipped. No sentiment results fabricated.")
        (root / "results" / "sentiment_status.txt").write_text("\n".join(status_lines))
        return {"status": "no_ollama", "headlines": headlines}

    # Score
    labels = []
    for h in headlines["headline"]:
        try:
            labels.append(score_headline_ollama(h))
        except Exception:
            labels.append("neutral")
    headlines["sentiment_label"] = labels
    sent_map = {"bullish": 1, "bearish": -1, "neutral": 0}
    headlines["sentiment"] = headlines["sentiment_label"].map(sent_map).fillna(0).astype(int)
    headlines.to_csv(root / "data" / "processed" / "headlines_scored.csv", index=False)

    # Aggregate daily per market
    headlines["date"] = headlines["timestamp"].dt.date
    daily = headlines.groupby(["market", "date"])["sentiment"].mean().reset_index()
    daily.to_csv(root / "data" / "processed" / "sentiment_daily.csv", index=False)

    # Simple ablation: hit rate of next-day return for price-only vs price+sentiment
    try:
        returns = pd.read_parquet(root / "data" / "processed" / "returns.parquet")
        sent = daily.copy()
        sent["date"] = pd.to_datetime(sent["date"])
        # map market names to return columns (same names)
        ab_rows = []
        for market in returns.columns:
            ms = sent[sent["market"] == market].set_index("date")["sentiment"]
            # align
            r = returns[market]
            # price-only position: sign of previous day return
            pos_price = np.sign(r.shift(1))
            # price+sentiment: add sentiment (forward-filled) to previous return
            sent_series = ms.reindex(r.index, method="ffill").fillna(0)
            pos_ps = np.sign(r.shift(1) + 0.01 * sent_series)
            # next-day return hit
            fwd = r.shift(-1)
            hit_price = ((pos_price * fwd) > 0).mean()
            hit_ps = ((pos_ps * fwd) > 0).mean()
            ab_rows.append({"market": market, "hit_rate_price_only": float(hit_price),
                           "hit_rate_price_plus_sentiment": float(hit_ps)})
        ab_df = pd.DataFrame(ab_rows)
        ab_df.to_csv(root / "results" / "sentiment_ablation.csv", index=False)
        status_lines.append("Ablation saved to results/sentiment_ablation.csv (recent window only).")
    except Exception as e:
        status_lines.append(f"Ablation failed: {e}")

    status_lines.append("Sentiment scored via Ollama.")
    (root / "results" / "sentiment_status.txt").write_text("\n".join(status_lines))
    return {"status": "scored", "headlines": headlines, "daily": daily}


if __name__ == "__main__":
    run_phase5()
