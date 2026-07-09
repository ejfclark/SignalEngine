import numpy as np
import pandas as pd

from signalengine.labels.triple_barrier import apply_triple_barrier


def make_panel(closes, highs=None, lows=None, atr_value=2.0):
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "ticker": "T",
        "date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "open": closes,  # open == close for easy reasoning
        "high": np.asarray(highs, dtype=float) if highs is not None else closes,
        "low": np.asarray(lows, dtype=float) if lows is not None else closes,
        "close": closes,
        "atr_14": atr_value,
    })


def test_target_hit_first():
    # entry at open[1]=100; target=100+3*2=106; stop=100-1.5*2=97
    closes = [100, 100, 102, 107, 90, 90, 90, 90, 90, 90, 90, 90]
    highs = [c + 0.5 for c in closes]
    panel = make_panel(closes, highs=highs, lows=closes)
    out = apply_triple_barrier(panel, horizon=10, target_mult=3.0, stop_mult=1.5)
    row = out.iloc[0]
    assert row["label"] == 1.0 and row["outcome"] == "target"
    assert row["exit_price"] == row["target_price"] == 106.0
    assert row["trade_return"] == (106.0 / 100.0) - 1.0


def test_stop_hit_first():
    closes = [100, 100, 96, 110, 110, 110, 110, 110, 110, 110, 110, 110]
    panel = make_panel(closes)
    out = apply_triple_barrier(panel, horizon=10, target_mult=3.0, stop_mult=1.5)
    row = out.iloc[0]
    assert row["label"] == 0.0 and row["outcome"] == "stop"
    assert row["exit_price"] == row["stop_price"] == 97.0


def test_both_hit_same_day_is_conservative_stop():
    closes = [100, 100, 100] + [100] * 10
    highs = [100, 100, 120] + [100] * 10   # day 2 breaches both
    lows = [100, 100, 90] + [100] * 10
    panel = make_panel(closes, highs=highs, lows=lows)
    row = apply_triple_barrier(panel, horizon=10, target_mult=3.0, stop_mult=1.5).iloc[0]
    assert row["outcome"] == "stop" and row["label"] == 0.0


def test_timeout_label():
    closes = [100.0] * 15  # never moves
    panel = make_panel(closes)
    row = apply_triple_barrier(panel, horizon=10, target_mult=3.0, stop_mult=1.5).iloc[0]
    assert row["label"] == 0.0 and row["outcome"] == "timeout"
    assert row["exit_price"] == 100.0


def test_incomplete_window_stays_unlabeled():
    closes = [100.0] * 8  # fewer bars than horizon after the first row
    panel = make_panel(closes)
    out = apply_triple_barrier(panel, horizon=10, target_mult=3.0, stop_mult=1.5)
    assert out["label"].isna().all()  # no timeout labels invented from short windows


def test_entry_is_next_day_open_no_lookahead():
    closes = [100, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105]
    panel = make_panel(closes)
    row = apply_triple_barrier(panel, horizon=10, target_mult=3.0, stop_mult=1.5).iloc[0]
    assert row["entry_price"] == 105.0  # open of day 1, not day 0's close


def test_multi_ticker_isolation():
    a = make_panel([100] * 15)
    b = make_panel([100, 100, 200] + [200] * 12)
    b["ticker"] = "B"
    out = apply_triple_barrier(pd.concat([a, b], ignore_index=True), 10, 3.0, 1.5)
    assert out[out["ticker"] == "T"].iloc[0]["outcome"] == "timeout"
    assert out[out["ticker"] == "B"].iloc[0]["outcome"] == "target"
