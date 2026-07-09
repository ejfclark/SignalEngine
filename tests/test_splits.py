import numpy as np
import pandas as pd

from signalengine.model.splits import purged_walk_forward


def make_dates(n_days=120, rows_per_day=3):
    days = pd.date_range("2024-01-01", periods=n_days, freq="B")
    return pd.Series(np.repeat(days, rows_per_day))


def test_train_always_before_test():
    dates = make_dates()
    for train_mask, test_mask in purged_walk_forward(dates, n_folds=4, purge_days=5):
        assert dates[train_mask].max() < dates[test_mask].min()


def test_no_row_in_both():
    dates = make_dates()
    for train_mask, test_mask in purged_walk_forward(dates, n_folds=4, purge_days=5):
        assert not (train_mask & test_mask).any()


def test_purge_gap_at_least_purge_days():
    dates = make_dates()
    unique = np.array(sorted(dates.unique()))
    for train_mask, test_mask in purged_walk_forward(dates, n_folds=4, purge_days=5):
        train_end = dates[train_mask].max()
        test_start = dates[test_mask].min()
        gap = np.searchsorted(unique, test_start) - np.searchsorted(unique, train_end)
        assert gap > 5  # strictly more trading days than the purge window


def test_expanding_window():
    dates = make_dates()
    sizes = [train.sum() for train, _ in purged_walk_forward(dates, n_folds=4, purge_days=5)]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


def test_every_fold_has_test_rows():
    dates = make_dates()
    folds = list(purged_walk_forward(dates, n_folds=5, purge_days=10))
    assert len(folds) == 5
    assert all(test.sum() > 0 for _, test in folds)
