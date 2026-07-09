"""Data access behind one interface so SQL can be swapped for Parquet without
touching the engine.

Canonical frames (all snake_case, `date` is datetime64[ns], numerics are float64):

prices (stocks)   ticker, date, open, high, low, close, volume, avg_volume,
                  eps, pe, mcap, shares_outstanding, earnings_date,
                  price_avg50, price_avg200
prices (crypto)   ticker, date, open, high, low, close, volume, mcap,
                  circulating_supply, total_supply
instruments       ticker, name, category, sector   (sector = SPDR ETF code, joins SectorPE)
bond_yields       date, country, yield_pct, chg_1d_bps, chg_1m_bps, chg_1y_bps
market_pe         date, code, pe, dev_5yr_pct
sector_pe         date, code, pe, dev_5yr_pct
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from ..config import Config


class DataSource(Protocol):
    def load_stock_prices(self) -> pd.DataFrame: ...
    def load_crypto_prices(self) -> pd.DataFrame: ...
    def load_instruments(self) -> pd.DataFrame: ...
    def load_bond_yields(self) -> pd.DataFrame: ...
    def load_market_pe(self) -> pd.DataFrame: ...
    def load_sector_pe(self) -> pd.DataFrame: ...
    # ingest-job datasets (may be empty if the job hasn't run):
    def load_stock_fundamentals(self) -> pd.DataFrame: ...
    def load_etf_prices(self) -> pd.DataFrame: ...
    def load_macro(self) -> pd.DataFrame: ...
    def load_crypto_derivatives(self) -> pd.DataFrame: ...


def get_source(cfg: Config) -> DataSource:
    if cfg.source == "parquet":
        from .parquet_source import ParquetSource

        return ParquetSource(cfg.parquet_dir)
    from .sql_source import SqlSource

    return SqlSource(cfg.connection_string)


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Shared cleanup for either source: types, ordering, obvious bad rows."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in df.columns:
        if col not in ("ticker", "date", "earnings_date"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    # A zero/negative close is an ingest artefact, not a price.
    df = df[df["close"] > 0]
    # The legacy ingest never populated Open (zero through 2024, patchy after):
    # treat non-positive opens as missing so nothing downstream reads 0 as a price.
    if "open" in df.columns:
        df.loc[df["open"] <= 0, "open"] = float("nan")
    df = df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    return df.reset_index(drop=True)
