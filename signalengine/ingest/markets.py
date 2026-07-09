"""Market-context ingest: retires the C# StockIngest app and the SQL bridge.

macro   — VIX (VIXCLS), broad dollar index (DTWEXBGS), 10y (DGS10) and 2y
          (DGS2) treasury yields from FRED's keyless fredgraph CSV endpoint.
          Official data, no API key, no scraping.
markets — country P/E and S&P sector P/E scraped from worldperatio.com;
          faithful port of the C# MarketPeParser / SectorPeParser (both pages
          embed a JS array the parsers cut out and clean up). Scrapers throw
          FormatException-style errors when the page layout changes, so a
          break is loud in the nightly log, never silent zeros.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd

from .lake import upsert
from .providers import BACKFILL_YEARS, USER_AGENT, _get

# ---------------------------------------------------------------- FRED macro

FRED_SERIES = {"VIXCLS": "vix", "DTWEXBGS": "dxy", "DGS10": "us10y", "DGS2": "us2y"}
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_fred_series(series_id: str, start: date) -> pd.Series:
    # FRED tarpits python-requests' TLS fingerprint but serves curl instantly,
    # so shell out to curl (present on Windows 10+ and every Linux distro).
    import subprocess

    url = f"{FRED_CSV_URL}?id={series_id}&cosd={start.isoformat()}"
    proc = subprocess.run(["curl", "-s", "--fail", "-m", "60", url],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise ValueError(f"FRED fetch failed for {series_id} (curl exit {proc.returncode})")
    df = pd.read_csv(StringIO(proc.stdout))
    date_col, value_col = df.columns[0], df.columns[1]
    out = pd.Series(
        pd.to_numeric(df[value_col], errors="coerce").values,  # "." on holidays -> NaN
        index=pd.to_datetime(df[date_col]),
        name=series_id,
    )
    return out.dropna()


def ingest_macro(lake_file: Path, backfill: bool = False) -> None:
    """Writes macro.parquet: date, vix, dxy, us10y, us2y. The full FRED CSV is
    small, so incremental just refetches a recent window and upserts."""
    start = date.today() - timedelta(days=365 * (BACKFILL_YEARS + 1))
    if not backfill and lake_file.is_file():
        start = date.today() - timedelta(days=45)

    series = {}
    for series_id, name in FRED_SERIES.items():
        try:
            series[name] = fetch_fred_series(series_id, start)
        except Exception as e:
            print(f"  ! FRED {series_id}: {e}")
    if not series:
        return
    wide = pd.DataFrame(series)
    wide.index.name = "date"
    wide = wide.reset_index()
    added, total = upsert(lake_file, wide, ["date"])
    print(f"  {lake_file.name}: +{added:,} rows (total {total:,})")


# ------------------------------------------------------- worldperatio scrape

MARKET_URL = "https://worldperatio.com/"
SECTORS_URL = "https://worldperatio.com/sp-500-sectors/"
_DESC_PREFIX = re.compile(r'"desc"\s*:\s*\'.*?P/E:\s*<b>', re.S)
_TRAILING_COMMA = re.compile(r",\s*([\]}])")


def _loads_js(raw: str):
    """json.loads that tolerates the trailing commas JS allows."""
    return json.loads(_TRAILING_COMMA.sub(r"\1", raw))


def _cut_array(text: str, marker: str) -> str:
    """Return the balanced [...] JS array that starts at `marker`."""
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"worldperatio: '{marker}' not found — page layout has changed")
    start += len(marker) - 1  # position of '['
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("worldperatio: data array is not terminated")


def parse_market_pe(html: str, asof: date) -> pd.DataFrame:
    """Country P/E from the home page's data[5] array. The desc field is an
    HTML snippet ending in "P/E: <b>NN.NN</b>"; strip markup so it parses as
    the bare number (same trick as the C# parser)."""
    raw = _cut_array(html, "data[5] = [")
    cleaned = _DESC_PREFIX.sub('"desc":', raw).replace("</b>'", "")
    entries = _loads_js(cleaned)
    rows = [  # same columns the SQL bridge produced — schema stays stable
        {"date": pd.Timestamp(asof), "code": e.get("flag", ""),
         "pe": float(e["desc"]), "dev_5yr_pct": float(e.get("value", 0.0))}
        for e in entries if e.get("name")
    ]
    if not rows:
        raise ValueError("worldperatio: market P/E array parsed to zero rows")
    return pd.DataFrame(rows)


def parse_sector_pe(html: str, asof: date) -> pd.DataFrame:
    """S&P sector P/E bubble points from sectors_xy_60_points (5-year window).
    Z (quoted string in the page JSON) is the P/E level; X the 5y deviation."""
    raw = _cut_array(html, "sectors_xy_60_points = [")
    entries = _loads_js(raw)
    rows = [
        {"date": pd.Timestamp(asof), "code": e["code"],
         "pe": float(e["z"]), "dev_5yr_pct": float(e.get("x", 0.0))}
        for e in entries if e.get("code")
    ]
    if not rows:
        raise ValueError("worldperatio: sector P/E array parsed to zero rows")
    return pd.DataFrame(rows)


def ingest_market_pe(lake_dir: Path) -> None:
    """Scrape both worldperatio pages into market_pe / sector_pe (same schema
    the SQL bridge produced, so features are untouched). Daily snapshot only —
    history up to today came over in the EdStock migration."""
    asof = date.today()
    headers = {"User-Agent": USER_AGENT}

    market = parse_market_pe(_get(MARKET_URL, headers=headers).text, asof)
    added, total = upsert(lake_dir / "market_pe.parquet", market, ["code", "date"])
    print(f"  market_pe.parquet: +{added:,} rows (total {total:,})")

    sector = parse_sector_pe(_get(SECTORS_URL, headers=headers).text, asof)
    added, total = upsert(lake_dir / "sector_pe.parquet", sector, ["code", "date"])
    print(f"  sector_pe.parquet: +{added:,} rows (total {total:,})")
