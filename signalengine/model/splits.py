"""Purged walk-forward cross-validation.

Time series + overlapping labels make random splits a leakage machine (the old
MLSignalAnalysis.py used train_test_split — that alone invalidated its results).
Here:

  - The date axis is cut into n_folds+1 contiguous blocks.
  - Fold i trains on blocks[0..i] and tests on block[i+1] (train is always
    strictly earlier than test — expanding window).
  - The last `purge_days` trading days are removed from the end of each train
    window. A label with a 10-day horizon computed 3 days before the test block
    "knows" what happened inside it; purging cuts that overlap.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def purged_walk_forward(
    dates: pd.Series, n_folds: int = 5, purge_days: int = 10
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yields (train_mask, test_mask) boolean arrays aligned with `dates`."""
    unique_dates = np.array(sorted(pd.unique(dates)))
    blocks = np.array_split(unique_dates, n_folds + 1)

    date_values = dates.to_numpy()
    for i in range(n_folds):
        train_dates = np.concatenate(blocks[: i + 1])
        if purge_days > 0 and len(train_dates) > purge_days:
            train_dates = train_dates[:-purge_days]
        test_dates = blocks[i + 1]

        train_mask = np.isin(date_values, train_dates)
        test_mask = np.isin(date_values, test_dates)
        yield train_mask, test_mask
