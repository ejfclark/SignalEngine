"""Reads the canonical frames from a local Parquet lake.

The lake is written by `signalengine export-parquet` (see export.py) and is a
drop-in replacement for the SQL source: same frames, same columns. Once the
ingest apps write Parquet directly, point config [data].source at this and the
Azure SQL database is no longer on the training path.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .source import normalize_prices

FILES = {
    "stock_prices": "stock_prices.parquet",
    "crypto_prices": "crypto_prices.parquet",
    "instruments": "instruments.parquet",
    "bond_yields": "bond_yields.parquet",
    "market_pe": "market_pe.parquet",
    "sector_pe": "sector_pe.parquet",
    # written by the ingest jobs; may not exist yet — loaders return empty
    "stock_fundamentals": "stock_fundamentals.parquet",
    "etf_prices": "etf_prices.parquet",
    "macro": "macro.parquet",
    "crypto_derivatives": "crypto_derivatives.parquet",
}


class ParquetSource:
    def __init__(self, directory: Path):
        self._dir = Path(directory)
        if not self._dir.is_dir():
            raise FileNotFoundError(
                f"Parquet lake not found at {self._dir}. Run `signalengine export-parquet` first."
            )

    def _read(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(self._dir / FILES[key])

    def load_stock_prices(self) -> pd.DataFrame:
        return normalize_prices(self._read("stock_prices"))

    def load_crypto_prices(self) -> pd.DataFrame:
        return normalize_prices(self._read("crypto_prices"))

    def load_instruments(self) -> pd.DataFrame:
        return self._read("instruments")

    def load_bond_yields(self) -> pd.DataFrame:
        return self._read("bond_yields")

    def load_market_pe(self) -> pd.DataFrame:
        return self._read("market_pe")

    def load_sector_pe(self) -> pd.DataFrame:
        return self._read("sector_pe")

    def _read_optional(self, key: str) -> pd.DataFrame:
        path = self._dir / FILES[key]
        return pd.read_parquet(path) if path.is_file() else pd.DataFrame()

    def load_stock_fundamentals(self) -> pd.DataFrame:
        return self._read_optional("stock_fundamentals")

    def load_etf_prices(self) -> pd.DataFrame:
        return self._read_optional("etf_prices")

    def load_macro(self) -> pd.DataFrame:
        return self._read_optional("macro")

    def load_crypto_derivatives(self) -> pd.DataFrame:
        return self._read_optional("crypto_derivatives")
