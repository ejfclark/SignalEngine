"""Daily stock fundamentals snapshot via Financial Modeling Prep batch quotes.

Optional: needs FMP_API_KEY (the legacy DataReader used FMP, so a key exists).
Appends one row per ticker per run day to stock_fundamentals.parquet:

    ticker, date, eps, pe, mcap, shares_outstanding, avg_volume, earnings_date

These are point-in-time snapshots — the same shape as the fundamentals the
legacy EdStock DailyPrice table carried, so history and new data line up.

WHERE MORE DATA WOULD HELP: FMP also serves earnings *surprise* history
(actual vs estimate) — a natural extension of this job once the engine's
days_to_earnings feature earns its keep.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd

from .lake import upsert
from .providers import _get

BATCH_SIZE = 100


def ingest_fundamentals(lake_file: Path, tickers: list[str]) -> None:
    key = os.environ.get("FMP_API_KEY")
    if not key:
        print("  FMP_API_KEY not set — skipping fundamentals (set it in .env to enable)")
        return

    rows = []
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        url = f"https://financialmodelingprep.com/api/v3/quote/{','.join(batch)}"
        for q in _get(url, params={"apikey": key}).json():
            rows.append({
                "ticker": q.get("symbol"),
                "date": pd.Timestamp(date.today()),
                "eps": q.get("eps"),
                "pe": q.get("pe"),
                "mcap": q.get("marketCap"),
                "shares_outstanding": q.get("sharesOutstanding"),
                "avg_volume": q.get("avgVolume"),
                "earnings_date": pd.to_datetime(q.get("earningsAnnouncement"), errors="coerce"),
            })
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"]).dt.tz_localize(None)
    added, total = upsert(lake_file, df, ["ticker", "date"])
    print(f"  {lake_file.name}: +{added:,} rows (total {total:,})")
