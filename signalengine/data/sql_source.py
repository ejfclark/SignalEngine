"""Reads the EdStock Azure SQL tables StockIngest writes to.

Plain SELECTs on base tables — no stored procedures — so the identical frames can
come from Parquet later. Azure SQL serverless may be asleep; connect retries.
"""

from __future__ import annotations

import time

import pandas as pd
import pyodbc

from .source import normalize_prices

_STOCK_QUERY = """
SELECT Ticker AS ticker, Timestamp AS date,
       [Open] AS [open], High AS high, Low AS low, [Close] AS [close],
       Volume AS volume, AvgVolume AS avg_volume,
       EPS AS eps, PE AS pe, MCap AS mcap, SharesOutstanding AS shares_outstanding,
       EarningsAnnouncement AS earnings_date,
       PriceAvg50 AS price_avg50, PriceAvg200 AS price_avg200
FROM dbo.DailyPrice
"""

_CRYPTO_QUERY = """
SELECT Ticker AS ticker, CAST(Timestamp AS date) AS date,
       [Open] AS [open], High AS high, Low AS low, [Close] AS [close],
       Volume AS volume, MarketCap AS mcap,
       CirculatingSupply AS circulating_supply, TotalSupply AS total_supply
FROM dbo.CurrencyPrice
WHERE QuoteCurrency = 'USD'
"""

_INSTRUMENT_QUERY = """
SELECT i.Ticker AS ticker, i.InstrumentName AS name,
       c.CategoryName AS category, c.Sector AS sector
FROM dbo.Instrument i
LEFT JOIN dbo.Category c ON c.CategoryId = i.CategoryId
"""

_BOND_QUERY = """
SELECT Market AS country, Timestamp AS date, Yield AS yield_pct,
       [1Day] AS chg_1d_bps, [1Month] AS chg_1m_bps, [1Year] AS chg_1y_bps
FROM dbo.BondYield
"""

_MARKET_PE_QUERY = """
SELECT Timestamp AS date, Code AS code, PE AS pe, [5yrChange] AS dev_5yr_pct
FROM dbo.MarketPE
"""

# SectorPE stores the worldperatio bubble chart: Z is the P/E level, X the
# deviation from the 5-year average in percent (Y is the daily change).
_SECTOR_PE_QUERY = """
SELECT Timestamp AS date, Code AS code, Z AS pe, X AS dev_5yr_pct
FROM dbo.SectorPE
"""


class SqlSource:
    def __init__(self, connection_string: str, attempts: int = 5, delay_s: float = 15.0):
        self._conn_str = connection_string
        self._attempts = attempts
        self._delay_s = delay_s
        self._conn: pyodbc.Connection | None = None

    def _connect(self) -> pyodbc.Connection:
        if self._conn is not None:
            return self._conn
        last_error: Exception | None = None
        for attempt in range(1, self._attempts + 1):
            try:
                self._conn = pyodbc.connect(self._conn_str)
                return self._conn
            except pyodbc.Error as e:  # serverless DB waking up
                last_error = e
                if attempt < self._attempts:
                    print(f"DB connect attempt {attempt} failed (database waking?); retrying in {self._delay_s:.0f}s")
                    time.sleep(self._delay_s)
        raise ConnectionError(f"Could not connect to EdStock after {self._attempts} attempts") from last_error

    def _read(self, query: str) -> pd.DataFrame:
        cursor = self._connect().cursor()
        cursor.execute(query)
        columns = [c[0] for c in cursor.description]
        df = pd.DataFrame.from_records(cursor.fetchall(), columns=columns)
        cursor.close()
        return df

    def load_stock_prices(self) -> pd.DataFrame:
        df = self._read(_STOCK_QUERY)
        # DateTime.MaxValue means "provider had no announcement date".
        df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")
        df.loc[df["earnings_date"] > pd.Timestamp("2100-01-01"), "earnings_date"] = pd.NaT
        return normalize_prices(df)

    def load_crypto_prices(self) -> pd.DataFrame:
        return normalize_prices(self._read(_CRYPTO_QUERY))

    def load_instruments(self) -> pd.DataFrame:
        return self._read(_INSTRUMENT_QUERY)

    def load_bond_yields(self) -> pd.DataFrame:
        df = self._read(_BOND_QUERY)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def load_market_pe(self) -> pd.DataFrame:
        df = self._read(_MARKET_PE_QUERY)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def load_sector_pe(self) -> pd.DataFrame:
        df = self._read(_SECTOR_PE_QUERY)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # The ingest-job datasets never lived in SQL; the lake is their only home.
    def load_stock_fundamentals(self) -> pd.DataFrame:
        return pd.DataFrame()

    def load_etf_prices(self) -> pd.DataFrame:
        return pd.DataFrame()

    def load_macro(self) -> pd.DataFrame:
        return pd.DataFrame()

    def load_crypto_derivatives(self) -> pd.DataFrame:
        return pd.DataFrame()
