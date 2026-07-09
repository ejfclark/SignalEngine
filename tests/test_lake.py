import pandas as pd

from signalengine.ingest.lake import last_date, upsert


def make(rows):
    return pd.DataFrame(rows, columns=["ticker", "date", "close"]).assign(
        date=lambda d: pd.to_datetime(d["date"])
    )


def test_upsert_creates_and_appends(tmp_path):
    path = tmp_path / "prices.parquet"
    upsert(path, make([("A", "2024-01-01", 1.0), ("A", "2024-01-02", 2.0)]), ["ticker", "date"])
    _, total = upsert(path, make([("A", "2024-01-03", 3.0)]), ["ticker", "date"])
    assert total == 3


def test_upsert_is_idempotent(tmp_path):
    path = tmp_path / "prices.parquet"
    df = make([("A", "2024-01-01", 1.0), ("B", "2024-01-01", 5.0)])
    upsert(path, df, ["ticker", "date"])
    _, total = upsert(path, df, ["ticker", "date"])
    assert total == 2


def test_upsert_new_rows_win(tmp_path):
    path = tmp_path / "prices.parquet"
    upsert(path, make([("A", "2024-01-01", 1.0)]), ["ticker", "date"])
    upsert(path, make([("A", "2024-01-01", 9.0)]), ["ticker", "date"])
    out = pd.read_parquet(path)
    assert len(out) == 1 and out["close"].iloc[0] == 9.0


def test_upsert_empty_frame_is_noop(tmp_path):
    path = tmp_path / "prices.parquet"
    upsert(path, make([("A", "2024-01-01", 1.0)]), ["ticker", "date"])
    added, total = upsert(path, pd.DataFrame(), ["ticker", "date"])
    assert added == 0 and total == 1


def test_last_date_per_group(tmp_path):
    path = tmp_path / "prices.parquet"
    upsert(path, make([("A", "2024-01-01", 1.0), ("A", "2024-01-05", 2.0),
                       ("B", "2024-01-03", 3.0)]), ["ticker", "date"])
    per = last_date(path, "ticker")
    assert per["A"] == pd.Timestamp("2024-01-05")
    assert per["B"] == pd.Timestamp("2024-01-03")
    assert last_date(tmp_path / "missing.parquet") is None
