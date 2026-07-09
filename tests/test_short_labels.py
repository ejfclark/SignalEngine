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


def flat(n, price=100.0):
    return [price] * n


def test_short_target_below_entry():
    # entry open[1]=100; short target=100-3*2=94, stop=100+1.5*2=103; price slides to 93
    panel = make_panel(
        opens=flat(2) + [96, 93] + flat(9, 93),
        highs=flat(2) + [96.5, 93.5] + flat(9, 93),
        lows=flat(2) + [95, 92.5] + flat(9, 93),
        closes=flat(2) + [96, 93] + flat(9, 93),
    )
    row = apply_triple_barrier(panel, 10, 3.0, 1.5, direction="short").iloc[0]
    assert row["outcome"] == "target" and row["label"] == 1.0
    assert row["target_price"] == 94.0 and row["stop_price"] == 103.0
    # day 3 gaps open at 93 through the 94 target -> favorable gap fill at 93
    assert row["exit_price"] == 93.0
    assert row["trade_return"] == (1.0 - 93.0 / 100.0)  # short gain is positive


def test_short_stopped_by_rally():
    # price rallies through the 103 stop
    panel = make_panel(
        opens=flat(2) + [102, 104] + flat(9, 104),
        highs=flat(2) + [102.5, 104.5] + flat(9, 104),
        lows=flat(2) + [101, 103.5] + flat(9, 104),
        closes=flat(2) + [102, 104] + flat(9, 104),
    )
    row = apply_triple_barrier(panel, 10, 3.0, 1.5, direction="short").iloc[0]
    assert row["outcome"] == "stop" and row["label"] == 0.0
    assert row["exit_price"] == 104.0  # gap open above stop fills at open
    assert row["trade_return"] < 0


def test_short_timeout_keeps_signed_return():
    panel = make_panel(flat(15, 100), flat(15, 100.4), flat(15, 99.6),
                       [100] * 2 + [99] * 13)
    row = apply_triple_barrier(panel, 10, 3.0, 1.5, direction="short").iloc[0]
    assert row["outcome"] == "timeout"
    assert row["trade_return"] > 0  # drifted down 1% -> small short gain


def test_long_unchanged_by_direction_param():
    panel = make_panel(flat(15), flat(15, 101), flat(15, 99), flat(15))
    long_default = apply_triple_barrier(panel, 10, 3.0, 1.5)
    long_explicit = apply_triple_barrier(panel, 10, 3.0, 1.5, direction="long")
    pd.testing.assert_frame_equal(long_default, long_explicit)
