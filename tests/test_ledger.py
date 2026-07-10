import numpy as np
import pandas as pd
import pytest

from signalengine import ledger as L
from signalengine.config import Config


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    cfg = Config(root=tmp_path)
    cfg.artifacts_dir = tmp_path / "artifacts"
    cfg.artifacts_dir.mkdir()
    return cfg


def write_signals(cfg, tag, rows):
    df = pd.DataFrame(rows)
    df.to_csv(cfg.artifacts_dir / f"{tag}_signals.csv", index=False)


def make_prices(rows):
    df = pd.DataFrame(rows, columns=["ticker", "date", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df


class FakeSource:
    def __init__(self, crypto):
        self._crypto = crypto

    def load_stock_prices(self):
        return make_prices([])

    def load_crypto_prices(self):
        return self._crypto


def patch_source(monkeypatch, crypto_prices):
    import signalengine.data as data_mod
    monkeypatch.setattr(data_mod, "get_source", lambda cfg: FakeSource(crypto_prices))


def test_record_is_idempotent(cfg):
    write_signals(cfg, "crypto", [
        {"ticker": "BTC", "date": "2026-07-01", "probability": 0.72,
         "stop": 95.0, "target": 112.0},
        {"ticker": "ETH", "date": "2026-07-01", "probability": 0.50,  # below threshold
         "stop": 9.0, "target": 12.0},
    ])
    assert L.record_signals(cfg) == 1
    assert L.record_signals(cfg) == 0  # same day, no duplicates
    book = L.load_ledger(cfg)
    assert list(book["ticker"]) == ["BTC"] and book.iloc[0]["status"] == "pending"


def test_entry_fill_and_target_exit(cfg, monkeypatch):
    write_signals(cfg, "crypto", [
        {"ticker": "BTC", "date": "2026-07-01", "probability": 0.70,
         "stop": 95.0, "target": 112.0},
    ])
    L.record_signals(cfg)
    prices = make_prices([
        ("BTC", "2026-07-02", 100.0, 105.0, 99.0, 104.0),   # entry at open 100
        ("BTC", "2026-07-03", 104.0, 113.0, 103.0, 112.5),  # intraday touch of 112
    ])
    patch_source(monkeypatch, prices)
    result = L.update_positions(cfg)
    assert result == {"filled": 1, "closed": 1}
    row = L.load_ledger(cfg).iloc[0]
    assert row["status"] == "target"
    assert row["entry_price"] == 100.0 and row["exit_price"] == 112.0
    assert row["net_return"] == pytest.approx(0.12 - 0.004)


def test_short_stop_gap_fill(cfg, monkeypatch):
    write_signals(cfg, "crypto-short", [
        {"ticker": "SOL", "date": "2026-07-01", "probability": 0.75,
         "stop": 103.0, "target": 94.0},
    ])
    L.record_signals(cfg)
    prices = make_prices([
        ("SOL", "2026-07-02", 100.0, 101.0, 99.0, 100.5),   # entry 100 (short)
        ("SOL", "2026-07-03", 106.0, 107.0, 105.0, 106.0),  # gaps OPEN above stop
    ])
    patch_source(monkeypatch, prices)
    L.update_positions(cfg)
    row = L.load_ledger(cfg).iloc[0]
    assert row["status"] == "stop"
    assert row["exit_price"] == 106.0  # gap fill at open, not 103
    assert row["net_return"] == pytest.approx(-(0.06) - 0.004)


def test_timeout_closes_at_horizon(cfg, monkeypatch):
    write_signals(cfg, "crypto", [
        {"ticker": "BTC", "date": "2026-07-01", "probability": 0.70,
         "stop": 90.0, "target": 130.0},
    ])
    L.record_signals(cfg)
    bars = [("BTC", f"2026-07-{d:02d}", 100.0, 101.0, 99.0, 100.0) for d in range(2, 22)]
    patch_source(monkeypatch, make_prices(bars))
    L.update_positions(cfg)
    row = L.load_ledger(cfg).iloc[0]
    assert row["status"] == "timeout"
    assert pd.Timestamp(row["exit_date"]) >= row["horizon_end"] - pd.Timedelta(days=1)


def test_open_position_stays_open(cfg, monkeypatch):
    write_signals(cfg, "crypto", [
        {"ticker": "BTC", "date": "2026-07-01", "probability": 0.70,
         "stop": 90.0, "target": 130.0},
    ])
    L.record_signals(cfg)
    patch_source(monkeypatch, make_prices([
        ("BTC", "2026-07-02", 100.0, 102.0, 99.0, 101.0),
    ]))
    result = L.update_positions(cfg)
    assert result == {"filled": 1, "closed": 0}
    assert L.load_ledger(cfg).iloc[0]["status"] == "open"
