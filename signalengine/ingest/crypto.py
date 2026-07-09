"""Crypto ingest via ccxt public endpoints (no API keys, no cost).

Two lake files:

crypto_prices.parquet       ticker, date, open, high, low, close, volume
                            Daily spot OHLCV (TICKER/USDT), full history.
crypto_derivatives.parquet  ticker, date, funding_rate, open_interest
                            Daily mean of the 8-hour perp funding rates —
                            the positioning/sentiment signal the price-only
                            model lacked. Open interest is best-effort
                            (exchanges only serve ~30 days of history, so it
                            accumulates forward from when the job starts).

Exchange preference binance -> bybit -> okx: a symbol missing on one is tried
on the next; stablecoins and unlisted tickers are skipped with a note.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .lake import last_date, upsert
from .providers import BACKFILL_YEARS
from .universe import STABLECOINS

EXCHANGES = ["binance", "bybit", "okx"]


def _clients():
    import ccxt

    out = []
    for name in EXCHANGES:
        try:
            out.append(getattr(ccxt, name)({"enableRateLimit": True}))
        except Exception as e:  # exchange unreachable from this network
            print(f"  ! {name} unavailable: {e}")
    return out


def _fetch_ohlcv_paged(client, symbol: str, since_ms: int) -> list[list]:
    rows, cursor = [], since_ms
    while True:
        batch = client.fetch_ohlcv(symbol, timeframe="1d", since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        cursor = batch[-1][0] + 1
    return rows


def ingest_crypto_prices(lake_file: Path, tickers: list[str], backfill: bool = False) -> None:
    clients = _clients()
    if not clients:
        print("  no exchange reachable — aborting crypto ingest")
        return

    default_start = datetime.now(timezone.utc) - timedelta(days=365 * BACKFILL_YEARS)
    per_ticker = None if (backfill or not lake_file.is_file()) else last_date(lake_file, "ticker")

    frames, skipped = [], []
    for i, ticker in enumerate(tickers, 1):
        if ticker in STABLECOINS:
            continue
        since = default_start
        if per_ticker is not None and ticker in per_ticker.index:
            since = (per_ticker[ticker] + pd.Timedelta(days=1)).tz_localize(timezone.utc)
        since_ms = int(since.timestamp() * 1000)

        rows = None
        for client in clients:
            try:
                rows = _fetch_ohlcv_paged(client, f"{ticker}/USDT", since_ms)
                if rows:
                    break
            except Exception:
                continue
        if not rows:
            skipped.append(ticker)
            continue

        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
        df.insert(0, "ticker", ticker)
        frames.append(df[["ticker", "date", "open", "high", "low", "close", "volume"]])
        if i % 20 == 0:
            print(f"  ...{i}/{len(tickers)} coins")

    if frames:
        added, total = upsert(lake_file, pd.concat(frames, ignore_index=True), ["ticker", "date"])
        print(f"  {lake_file.name}: +{added:,} rows (total {total:,})")
    if skipped:
        print(f"  no USDT market ({len(skipped)}): {', '.join(skipped)}")


def _fetch_funding_paged(client, symbol: str, since_ms: int) -> list[dict]:
    rows, cursor = [], since_ms
    while True:
        batch = client.fetch_funding_rate_history(symbol, since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        cursor = batch[-1]["timestamp"] + 1
    return rows


def ingest_crypto_derivatives(lake_file: Path, tickers: list[str], backfill: bool = False) -> None:
    clients = _clients()
    if not clients:
        return

    default_start = datetime.now(timezone.utc) - timedelta(days=365 * BACKFILL_YEARS)
    per_ticker = None if (backfill or not lake_file.is_file()) else last_date(lake_file, "ticker")

    frames, skipped = [], []
    for i, ticker in enumerate(tickers, 1):
        if ticker in STABLECOINS:
            continue
        since = default_start
        if per_ticker is not None and ticker in per_ticker.index:
            since = (per_ticker[ticker] + pd.Timedelta(days=1)).tz_localize(timezone.utc)
        since_ms = int(since.timestamp() * 1000)
        symbol = f"{ticker}/USDT:USDT"  # linear perp

        funding = None
        oi_now = None
        for client in clients:
            try:
                funding = _fetch_funding_paged(client, symbol, since_ms)
                if funding:
                    try:
                        oi_now = client.fetch_open_interest(symbol)
                    except Exception:
                        oi_now = None
                    break
            except Exception:
                continue
        if not funding:
            skipped.append(ticker)
            continue

        df = pd.DataFrame({
            "date": pd.to_datetime([f["timestamp"] for f in funding], unit="ms").normalize(),
            "funding_rate": [f.get("fundingRate") for f in funding],
        })
        daily = df.groupby("date", as_index=False)["funding_rate"].mean()
        daily.insert(0, "ticker", ticker)
        daily["open_interest"] = float("nan")
        if oi_now and oi_now.get("openInterestAmount"):
            daily.loc[daily.index[-1], "open_interest"] = float(oi_now["openInterestAmount"])
        frames.append(daily)
        if i % 20 == 0:
            print(f"  ...{i}/{len(tickers)} perps")

    if frames:
        added, total = upsert(lake_file, pd.concat(frames, ignore_index=True), ["ticker", "date"])
        print(f"  {lake_file.name}: +{added:,} rows (total {total:,})")
    if skipped:
        print(f"  no perp market ({len(skipped)}): {', '.join(skipped)}")
