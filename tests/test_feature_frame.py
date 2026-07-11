import numpy as np
import pandas as pd

from signalengine.features.pipeline import FEATURE_COLUMNS, SECTOR_CODES, feature_frame


def make_panel(sectors):
    n = len(sectors)
    panel = pd.DataFrame({col: np.random.default_rng(0).normal(size=n) for col in FEATURE_COLUMNS})
    panel["ticker"] = [f"T{i}" for i in range(n)]
    panel["sector"] = sectors
    return panel


def test_without_sector_matches_feature_columns():
    X = feature_frame(make_panel(["XLK", "XLF"]))
    assert list(X.columns) == FEATURE_COLUMNS


def test_sector_codes_stable_across_subsets():
    # The invariant that keeps train and predict consistent: a sector's
    # category code must not depend on which sectors happen to be present.
    full = feature_frame(make_panel(["XLK", "XLF", "XLE", "XLV"]), include_sector=True)
    subset = feature_frame(make_panel(["XLV"]), include_sector=True)
    code_full = full["sector"].cat.codes.iloc[3]
    code_subset = subset["sector"].cat.codes.iloc[0]
    assert code_full == code_subset == SECTOR_CODES.index("XLV")


def test_unmapped_and_missing_sector_are_nan():
    X = feature_frame(make_panel(["XLK", "BOGUS", None]), include_sector=True)
    assert X["sector"].cat.codes.tolist() == [SECTOR_CODES.index("XLK"), -1, -1]

    no_col = make_panel(["XLK"]).drop(columns=["sector"])
    X2 = feature_frame(no_col, include_sector=True)
    assert X2["sector"].isna().all()
