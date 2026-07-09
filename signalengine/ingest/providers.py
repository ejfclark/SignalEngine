"""Daily-price providers for stocks/ETFs. Both return the same frame:

    ticker, date, open, high, low, close, volume
    (prices split/dividend-adjusted)

Yahoo needs no key and works today; Tiingo is the preferred long-term source
(explicit API terms, cleaner data) and is used automatically when
TIINGO_API_KEY is set — the free tier covers 500 unique symbols/month and
1000 requests/day, enough for this universe's daily refresh AND its backfill.
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta

import pandas as pd
import requests

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
BACKFILL_YEARS = 5


class ProviderError(Exception):
    pass


def _get(url: str, *, params: dict | None = None, headers: dict | None = None,
         retries: int = 4, backoff: float = 2.0) -> requests.Response:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=60)
        except requests.RequestException as e:  # resets/timeouts: retry too
            if attempt < retries - 1:
                time.sleep(backoff * (2**attempt))
                continue
            raise ProviderError(f"{url} -> {e}") from e
        if resp.status_code == 200:
            return resp
        if resp.status_code in (429, 500, 502, 503) and attempt < retries - 1:
            time.sleep(backoff * (2**attempt))
            continue
        raise ProviderError(f"{url} -> HTTP {resp.status_code}")
    raise ProviderError(f"{url} -> retries exhausted")


_yahoo_session: requests.Session | None = None


def _yahoo() -> requests.Session:
    """Session with Yahoo's cookies (a bare API hit gets 429; a warmed one doesn't)."""
    global _yahoo_session
    if _yahoo_session is None:
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT})
        s.get("https://finance.yahoo.com", timeout=30)
        _yahoo_session = s
    return _yahoo_session


def fetch_yahoo_daily(ticker: str, start: date, end: date | None = None) -> pd.DataFrame:
    """Yahoo chart API (same endpoint the old C# YahooChartProvider used)."""
    end = end or date.today() + timedelta(days=1)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": int(pd.Timestamp(start).timestamp()),
        "period2": int(pd.Timestamp(end).timestamp()),
        "interval": "1d",
        "events": "splits,dividends",
    }
    # Fail fast on a persistent 429: ingest_prices is resumable, so a skipped
    # ticker just gets picked up by the next run instead of stalling this one.
    global _yahoo_session
    for attempt in range(3):
        resp = _yahoo().get(url, params=params, timeout=30)
        if resp.status_code == 200:
            break
        if resp.status_code == 429:
            time.sleep(10.0 * (attempt + 1))
            if attempt == 1:
                _yahoo_session = None  # refresh cookies once
            continue
        raise ProviderError(f"{ticker}: Yahoo HTTP {resp.status_code}")
    else:
        raise ProviderError(f"{ticker}: Yahoo rate limit persisted")

    result = resp.json().get("chart", {}).get("result")
    if not result or not result[0].get("timestamp"):
        raise ProviderError(f"Yahoo returned no data for {ticker}")
    r = result[0]
    quote = r["indicators"]["quote"][0]
    adj = r["indicators"].get("adjclose", [{}])[0].get("adjclose")

    df = pd.DataFrame({
        "date": pd.to_datetime(r["timestamp"], unit="s").normalize(),
        "open": quote.get("open"), "high": quote.get("high"),
        "low": quote.get("low"), "close": quote.get("close"),
        "volume": quote.get("volume"),
    })
    # Rescale OHLC by adjclose/close so history is split/dividend adjusted.
    if adj is not None:
        factor = pd.Series(adj, dtype="float64") / df["close"]
        for col in ("open", "high", "low", "close"):
            df[col] = df[col] * factor
    df.insert(0, "ticker", ticker)
    df["volume"] = df["volume"].astype("float64")
    return df.dropna(subset=["close"]).reset_index(drop=True)


def fetch_tiingo_daily(ticker: str, start: date, end: date | None = None) -> pd.DataFrame:
    key = os.environ.get("TIINGO_API_KEY")
    if not key:
        raise ProviderError("TIINGO_API_KEY not set")
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {"startDate": start.isoformat(), "token": key, "format": "json"}
    if end:
        params["endDate"] = end.isoformat()
    rows = _get(url, params=params).json()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "ticker": ticker,
        "date": pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize(),
        "open": df["adjOpen"], "high": df["adjHigh"],
        "low": df["adjLow"], "close": df["adjClose"],
        "volume": df["adjVolume"],
    })
    return out.dropna(subset=["close"]).reset_index(drop=True)


class RateLimited(ProviderError):
    """Provider quota exhausted — worth aborting the whole run, unlike a
    per-ticker failure (404 = ticker not carried), which is not."""


def fetch_daily(ticker: str, start: date) -> pd.DataFrame:
    """Tiingo when a key is configured (falls back to Yahoo), otherwise Yahoo.
    (Stooq was evaluated as a third leg but now sits behind a JS browser check.)"""
    tiingo_error: ProviderError | None = None
    if os.environ.get("TIINGO_API_KEY"):
        try:
            return fetch_tiingo_daily(ticker, start)
        except ProviderError as e:
            tiingo_error = e
    try:
        return fetch_yahoo_daily(ticker, start)
    except ProviderError as e:
        detail = f"tiingo: {tiingo_error}; yahoo: {e}" if tiingo_error else str(e)
        if "429" in str(tiingo_error or "") or "retries exhausted" in str(tiingo_error or ""):
            raise RateLimited(detail) from e
        if tiingo_error is None and ("429" in str(e) or "rate limit" in str(e)):
            raise RateLimited(detail) from e
        raise ProviderError(detail) from e
