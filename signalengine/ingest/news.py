"""News ingest (Tiingo News API, included with the Power tier) and LLM event
extraction.

Two stages, deliberately separate:

news_raw.parquet     one row per (article, ticker): published (UTC), ticker,
                     source, title, description, url, article_id. Collected
                     nightly; backfillable from Tiingo's archive.
news_events.parquet  LLM-extracted structure per article: event_type,
                     sentiment (-1..1), materiality (0..1). Each article is
                     scored ONCE (keyed by article_id) with a cheap model
                     forced into a fixed schema — the LLM turns text into
                     facts; it is never asked to predict prices. Requires
                     ANTHROPIC_API_KEY; without it the raw feed still
                     accumulates and extraction catches up later.

Timestamp hygiene is the whole game: features built from these tables must
only use articles with published < that trading day's close, or the backtest
becomes a lookahead machine. The join lives in features/ when benched.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .lake import upsert
from .providers import _get

TIINGO_NEWS_URL = "https://api.tiingo.com/tiingo/news"
BATCH_TICKERS = 50
EVENT_TYPES = ["earnings_beat", "earnings_miss", "guidance_raise", "guidance_cut",
               "upgrade", "downgrade", "merger_acquisition", "lawsuit_regulatory",
               "hack_breach", "listing_delisting", "partnership", "product",
               "macro", "other"]


def ingest_news(lake_file: Path, tickers: list[str], backfill_days: int = 0) -> None:
    """Pull per-ticker tagged articles. Incremental = since newest stored row
    (minus a day of overlap); backfill_days > 0 reaches into the archive."""
    key = os.environ.get("TIINGO_API_KEY")
    if not key:
        print("  TIINGO_API_KEY not set — skipping news")
        return

    if backfill_days:
        start = date.today() - timedelta(days=backfill_days)
    elif lake_file.is_file():
        newest = pd.read_parquet(lake_file, columns=["published"])["published"].max()
        start = (newest - pd.Timedelta(days=1)).date()
    else:
        start = date.today() - timedelta(days=7)

    frames = []
    for i in range(0, len(tickers), BATCH_TICKERS):
        batch = tickers[i : i + BATCH_TICKERS]
        offset = 0
        while True:  # Tiingo pages at 1000 articles
            resp = _get(TIINGO_NEWS_URL, params={
                "tickers": ",".join(batch), "startDate": start.isoformat(),
                "limit": 1000, "offset": offset, "token": key,
            })
            articles = resp.json()
            if not articles:
                break
            rows = []
            for a in articles:
                for ticker in a.get("tickers", []):
                    ticker = ticker.upper()
                    if ticker in batch:
                        rows.append({
                            "article_id": str(a["id"]),
                            "ticker": ticker,
                            "published": pd.Timestamp(a["publishedDate"]).tz_convert("UTC").tz_localize(None),
                            "source": a.get("source", ""),
                            "title": a.get("title", ""),
                            "description": (a.get("description") or "")[:2000],
                            "url": a.get("url", ""),
                        })
            frames.append(pd.DataFrame(rows))
            if len(articles) < 1000:
                break
            offset += 1000
            time.sleep(0.2)
        if (i // BATCH_TICKERS) % 5 == 4:
            print(f"  ...{i + len(batch)}/{len(tickers)} tickers", flush=True)

    if frames:
        new = pd.concat(frames, ignore_index=True)
        added, total = upsert(lake_file, new, ["article_id", "ticker"])
        print(f"  {lake_file.name}: +{added:,} rows (total {total:,})")


EXTRACTION_PROMPT = """Extract trading-relevant structure from this financial news article.
Respond with ONLY a JSON object, no other text:
{"event_type": one of %s,
 "sentiment": float -1.0 (very negative for the company/asset) to 1.0 (very positive),
 "materiality": float 0.0 (noise/PR fluff) to 1.0 (major price-relevant event)}

Title: %%s
Description: %%s""" % json.dumps(EVENT_TYPES)


def extract_events(raw_file: Path, events_file: Path, batch_limit: int = 2000,
                   tickers: list[str] | None = None) -> None:
    """Score unscored articles with a small Claude model. Idempotent by
    article_id; runs until caught up or batch_limit. `tickers` restricts
    scoring to a pilot set — collection is free, extraction is metered, so
    the news-feature experiment spends only on symbols in the pilot."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  ANTHROPIC_API_KEY not set — raw news accumulating, extraction deferred")
        return
    if not raw_file.is_file():
        print("  no raw news yet")
        return

    import anthropic

    raw = pd.read_parquet(raw_file)
    if tickers is not None:
        raw = raw[raw["ticker"].isin(set(tickers))]
    raw = raw.drop_duplicates("article_id")
    done: set[str] = set()
    if events_file.is_file():
        done = set(pd.read_parquet(events_file, columns=["article_id"])["article_id"])
    todo = raw[~raw["article_id"].isin(done)].head(batch_limit)
    if todo.empty:
        print("  news extraction: caught up")
        return

    client = anthropic.Anthropic(api_key=api_key)
    rows, failures = [], 0
    for _, a in todo.iterrows():
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=200,
                messages=[{"role": "user",
                           "content": EXTRACTION_PROMPT % (a["title"], a["description"])}],
            )
            parsed = json.loads(msg.content[0].text)
            rows.append({
                "article_id": a["article_id"],
                "published": a["published"],
                "event_type": parsed.get("event_type", "other"),
                "sentiment": float(parsed.get("sentiment", 0.0)),
                "materiality": float(parsed.get("materiality", 0.0)),
            })
        except Exception:
            failures += 1
            if failures > 25:
                print("  too many extraction failures — stopping this run", flush=True)
                break
    if rows:
        added, total = upsert(events_file, pd.DataFrame(rows), ["article_id"])
        print(f"  {events_file.name}: +{added:,} scored (total {total:,}, {failures} failures)")
