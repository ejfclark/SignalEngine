"""Ticker universes: plain text files under universe/, one ticker per line.

Editable by hand — add a line, run the ingest, the engine picks it up. Lines
starting with # are comments. Initially generated from the EdStock Instrument
table; from here on these files are the master list, not the database.
"""

from __future__ import annotations

from pathlib import Path

SECTOR_ETFS = ["SPY", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]

# No perpetual futures / no meaningful swing setup on pegged assets.
STABLECOINS = {"USDT", "USDC", "DAI", "USD1", "USDE", "TUSD", "USDD", "FDUSD", "PYUSD", "BUSD"}


def load_universe(root: Path, name: str) -> list[str]:
    path = root / "universe" / f"{name}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"universe file missing: {path}")
    tickers = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip().upper()
        if line:
            tickers.append(line)
    return sorted(dict.fromkeys(tickers))
