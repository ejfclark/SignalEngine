"""Stock / ETF / macro-index daily price ingest into the lake.

Backfill pulls BACKFILL_YEARS of adjusted history; incremental resumes from
each ticker's last stored bar. One request per ticker either way, politely
rate-limited, individual failures logged and skipped (the job reports them
but keeps going — one delisted ticker must not kill the nightly run).
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .lake import last_date, upsert
from .providers import BACKFILL_YEARS, RateLimited, fetch_daily

# Seconds between price requests. Override with SIGNALENGINE_PAUSE for a
# slow-drip run when the IP is rate-limited (e.g. 30 rides out a Yahoo block).
REQUEST_PAUSE_S = float(os.environ.get("SIGNALENGINE_PAUSE", "0.4"))


def _resume_dates(path: Path, tickers: list[str], backfill: bool) -> dict[str, date]:
    start_default = date.today() - timedelta(days=365 * BACKFILL_YEARS)
    if backfill or not path.is_file():
        return {t: start_default for t in tickers}
    per_ticker = last_date(path, "ticker")
    return {
        t: (per_ticker[t].date() + timedelta(days=1)) if t in per_ticker.index else start_default
        for t in tickers
    }


CHECKPOINT_EVERY = 25  # tickers per lake write: partial runs keep their progress
BREAKER_LIMIT = 6      # consecutive failures = provider is capped; stop wasting the run


def ingest_prices(lake_file: Path, tickers: list[str], backfill: bool = False) -> None:
    """Resumable: progress is checkpointed to the lake every CHECKPOINT_EVERY
    tickers, and _resume_dates skips whatever is already stored — so if a run
    dies (rate limit, reboot), simply run the job again WITHOUT --backfill and
    it continues where it stopped, including finishing a partial backfill
    (tickers with no rows yet restart from the full history window)."""
    starts = _resume_dates(lake_file, tickers, backfill)
    # Most-behind first: a quota-capped pass then spends its requests on
    # tickers missing years of history, not on refreshing yesterday's leaders —
    # otherwise a backfill can starve behind daily top-ups forever.
    tickers = sorted(tickers, key=lambda t: starts[t])
    frames, failed, fetched = [], [], 0
    consecutive_failures = 0

    def flush():
        nonlocal frames
        if frames:
            added, total = upsert(lake_file, pd.concat(frames, ignore_index=True), ["ticker", "date"])
            print(f"  {lake_file.name}: +{added:,} rows (total {total:,})", flush=True)
            frames = []

    for i, ticker in enumerate(tickers, 1):
        if starts[ticker] > date.today():
            continue
        try:
            df = fetch_daily(ticker, starts[ticker])
            if len(df):
                frames.append(df)
                fetched += 1
            consecutive_failures = 0
        except RateLimited as e:
            failed.append(ticker)
            consecutive_failures += 1
            print(f"  ! {ticker}: {e}", flush=True)
            if consecutive_failures >= BREAKER_LIMIT:
                print(f"  breaker tripped after {BREAKER_LIMIT} consecutive rate-limit failures — "
                      "provider capped; stopping early (rerun resumes here)", flush=True)
                break
        except Exception as e:  # ticker not carried / bad symbol — skip, no breaker
            failed.append(ticker)
            print(f"  ! {ticker}: {e}", flush=True)
        if i % CHECKPOINT_EVERY == 0:
            print(f"  ...{i}/{len(tickers)} tickers", flush=True)
            flush()
        time.sleep(REQUEST_PAUSE_S)

    flush()
    print(f"  done: {fetched} tickers fetched, {len(failed)} failed")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
