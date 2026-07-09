"""Parquet lake primitives: idempotent upsert with atomic replace."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def upsert(path: Path, new: pd.DataFrame, keys: list[str]) -> tuple[int, int]:
    """Merge `new` into the parquet file at `path`, new rows winning on `keys`.

    Returns (rows_added_or_updated, total_rows). Write is atomic (tmp + replace)
    so a crash mid-write never corrupts the lake.
    """
    if new.empty:
        total = len(pd.read_parquet(path)) if path.is_file() else 0
        return 0, total

    new = new.drop_duplicates(keys, keep="last")
    if path.is_file():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new], ignore_index=True)
        combined = combined.drop_duplicates(keys, keep="last")
        added = len(combined) - len(existing.drop_duplicates(keys))
    else:
        combined = new
        added = len(new)

    combined = combined.sort_values(keys).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return max(added, len(new) if not path.is_file() else added), len(combined)


def last_date(path: Path, group_col: str | None = None) -> pd.Timestamp | pd.Series | None:
    """Latest date in a lake file — overall, or per group (e.g. per ticker)."""
    if not path.is_file():
        return None
    df = pd.read_parquet(path, columns=["date", group_col] if group_col else ["date"])
    if df.empty:
        return None
    if group_col:
        return df.groupby(group_col)["date"].max()
    return df["date"].max()
