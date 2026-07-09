import pandas as pd

from signalengine.labels.triple_barrier import apply_triple_barrier


def make_panel(opens, highs, lows, closes, atr_value=2.0):
    n = len(closes)
    return pd.DataFrame({
        "ticker": "T",
        "date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "open": [float(x) for x in opens],
        "high": [float(x) for x in highs],
        "low": [float(x) for x in lows],
        "close": [float(x) for x in closes],
        "atr_14": atr_value,
    })


def test_gap_down_through_stop_fills_at_open():
    # entry open[1]=100, stop=97; day 2 gaps open at 92
    panel = make_panel(
        opens=[100, 100, 92] + [92] * 10,
        highs=[100, 100, 93] + [92] * 10,
        lows=[100, 99, 91] + [92] * 10,
        closes=[100, 100, 92] + [92] * 10,
    )
    row = apply_triple_barrier(panel, horizon=10, target_mult=3.0, stop_mult=1.5).iloc[0]
    assert row["outcome"] == "stop"
    assert row["exit_price"] == 92.0          # open fill, not the 97 stop price
    assert row["trade_return"] == (92.0 / 100.0) - 1.0


def test_gap_up_through_target_fills_at_open():
    # entry open[1]=100, target=106; day 2 gaps open at 110
    panel = make_panel(
        opens=[100, 100, 110] + [110] * 10,
        highs=[100, 100, 111] + [110] * 10,
        lows=[100, 99.5, 109] + [110] * 10,
        closes=[100, 100, 110] + [110] * 10,
    )
    row = apply_triple_barrier(panel, horizon=10, target_mult=3.0, stop_mult=1.5).iloc[0]
    assert row["outcome"] == "target"
    assert row["exit_price"] == 110.0         # favorable gap keeps the extra

def test_intraday_touch_still_fills_at_barrier():
    # no gap: opens inside the range, low touches the stop intraday
    panel = make_panel(
        opens=[100, 100, 100] + [100] * 10,
        highs=[100, 100, 101] + [100] * 10,
        lows=[100, 99.5, 96] + [100] * 10,
        closes=[100, 100, 100] + [100] * 10,
    )
    row = apply_triple_barrier(panel, horizon=10, target_mult=3.0, stop_mult=1.5).iloc[0]
    assert row["outcome"] == "stop"
    assert row["exit_price"] == 97.0
