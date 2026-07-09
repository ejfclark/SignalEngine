"""Legacy bridge: the one-time EdStock snapshot.

(The former `ingest context` SQL bridge is retired — market/sector P/E now
come from `ingest markets` and rates/VIX/DXY from `ingest macro` via FRED,
so nothing depends on the Azure SQL database or the C# StockIngest app.)

`ingest legacy-snapshot` — one-time: archive the full EdStock export and carve
    the useful parts into the lake:
      - fundamentals history (eps/pe/mcap/earnings dates) from DailyPrice —
        the new price feed doesn't carry these, and pe_ratio/days_to_earnings
        are top-10 features;
      - instruments (ticker -> sector mapping), deduplicated.
    Old EdStock OHLC is NOT merged into the price files: it is unadjusted and
    open-less; the provider backfill supersedes it. It stays in the archive.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..config import Config
from .lake import upsert


def legacy_snapshot(cfg: Config) -> None:
    lake = cfg.parquet_dir
    archive = cfg.root / "data" / "edstock_snapshot"

    # 1. Archive the raw EdStock export (created by `signalengine export-parquet`).
    archive.mkdir(parents=True, exist_ok=True)
    for f in lake.glob("*.parquet"):
        shutil.copy2(f, archive / f.name)
    print(f"  archived EdStock export -> {archive}")

    # 2. Fundamentals history out of the legacy DailyPrice rows.
    import pandas as pd

    legacy = pd.read_parquet(archive / "stock_prices.parquet")
    cols = ["ticker", "date", "eps", "pe", "mcap", "shares_outstanding",
            "avg_volume", "earnings_date"]
    fundamentals = legacy[[c for c in cols if c in legacy.columns]].copy()
    fundamentals = fundamentals[(fundamentals["eps"].notna()) | (fundamentals["pe"].notna())]
    added, total = upsert(lake / "stock_fundamentals.parquet", fundamentals, ["ticker", "date"])
    print(f"  stock_fundamentals.parquet: +{added:,} rows (total {total:,})")

    # 3. Clean instruments (sector mapping) — dedupe the known duplicate tickers.
    inst = pd.read_parquet(archive / "instruments.parquet").drop_duplicates("ticker")
    inst.to_parquet(lake / "instruments.parquet", index=False)
    print(f"  instruments.parquet: {len(inst)} tickers")

    # 4. Remove the legacy price files from the lake: the provider backfill
    #    (adjusted, with opens) replaces them. The archive keeps the originals.
    for name in ("stock_prices.parquet", "crypto_prices.parquet"):
        target = lake / name
        if target.is_file():
            target.unlink()
            print(f"  removed legacy {name} from lake (archived copy kept)")
