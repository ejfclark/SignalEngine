"""One-shot export of the EdStock tables to the local Parquet lake.

This is both the migration path away from Azure SQL and a cheap full backup of
the historical data (the real asset here). Re-running overwrites the lake with
a fresh snapshot.
"""

from __future__ import annotations

from pathlib import Path

from .parquet_source import FILES
from .sql_source import SqlSource


def export_to_parquet(sql: SqlSource, directory: Path) -> dict[str, int]:
    directory.mkdir(parents=True, exist_ok=True)
    loaders = {
        "stock_prices": sql.load_stock_prices,
        "crypto_prices": sql.load_crypto_prices,
        "instruments": sql.load_instruments,
        "bond_yields": sql.load_bond_yields,
        "market_pe": sql.load_market_pe,
        "sector_pe": sql.load_sector_pe,
    }
    counts: dict[str, int] = {}
    for key, load in loaders.items():
        df = load()
        df.to_parquet(directory / FILES[key], index=False)
        counts[key] = len(df)
    return counts
