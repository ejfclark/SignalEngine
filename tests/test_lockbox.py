import pandas as pd

from signalengine.config import Config
from signalengine.lockbox import split_lockbox


def make_labeled(rows):
    df = pd.DataFrame(rows, columns=["date", "exit_date", "label"])
    df["date"] = pd.to_datetime(df["date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    return df


def test_split_removes_boundary_crossers(tmp_path):
    cfg = Config(root=tmp_path)
    cfg.cv.lockbox_start = "2026-03-15"
    df = make_labeled([
        ("2026-02-01", "2026-02-12", 1.0),   # safely historical
        ("2026-03-10", "2026-03-20", 0.0),   # trade CROSSES the boundary -> neither side
        ("2026-03-20", "2026-04-01", 1.0),   # inside the lockbox
    ])
    history, lockbox = split_lockbox(cfg, df)
    assert list(history["date"]) == [pd.Timestamp("2026-02-01")]
    assert list(lockbox["date"]) == [pd.Timestamp("2026-03-20")]


def test_disabled_lockbox_passes_everything(tmp_path):
    cfg = Config(root=tmp_path)
    cfg.cv.lockbox_start = ""
    df = make_labeled([("2026-03-20", "2026-04-01", 1.0)])
    history, lockbox = split_lockbox(cfg, df)
    assert len(history) == 1 and len(lockbox) == 0
