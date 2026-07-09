import numpy as np
import pandas as pd

from signalengine.features.indicators import compute_indicators


def make_ohlcv(closes, highs=None, lows=None, volumes=None):
    closes = pd.Series(closes, dtype=float)
    return pd.DataFrame({
        "ticker": "T",
        "date": pd.date_range("2023-01-02", periods=len(closes), freq="B"),
        "open": closes.shift(1).fillna(closes.iloc[0]),
        "high": pd.Series(highs, dtype=float) if highs is not None else closes * 1.01,
        "low": pd.Series(lows, dtype=float) if lows is not None else closes * 0.99,
        "close": closes,
        "volume": pd.Series(volumes, dtype=float) if volumes is not None else pd.Series(1e6, index=closes.index),
    })


def test_atr_ratio_detects_compression():
    rng = np.random.default_rng(3)
    wild = 100 + np.cumsum(rng.normal(0, 3, 80))
    calm = np.full(60, wild[-1]) + rng.normal(0, 0.2, 60)
    df = compute_indicators(make_ohlcv(np.concatenate([wild, calm])))
    assert df["atr_ratio_10_60"].iloc[-1] < 0.85          # compressed vs long-run
    assert df["atr_ratio_10_60"].iloc[-1] < df["atr_ratio_10_60"].iloc[79]  # tighter than wild era


def test_days_since_high_counts_up():
    closes = list(np.linspace(100, 120, 30)) + [110] * 10  # high at day 29, then drift
    df = compute_indicators(make_ohlcv(closes))
    assert df["days_since_20d_high"].iloc[-1] >= 9


def test_dist_to_breakout_level():
    closes = [100] * 40 + [104]  # jumps near the 20d high set at 105
    highs = [105 if i == 20 else c * 1.001 for i, c in enumerate(closes)]
    df = compute_indicators(make_ohlcv(closes, highs=highs))
    assert -0.05 < df["dist_20d_high"].iloc[-1] <= 0.0


def test_vol_dryup_low_when_volume_fades():
    volumes = [2e6] * 60 + [4e5] * 10
    df = compute_indicators(make_ohlcv([100] * 70, volumes=volumes))
    assert df["vol_dryup"].iloc[-1] < 0.5


def test_no_lookahead_in_new_features():
    rng = np.random.default_rng(4)
    closes = 100 + np.cumsum(rng.normal(0, 1, 150))
    full = compute_indicators(make_ohlcv(closes))
    trunc = compute_indicators(make_ohlcv(closes[:100]))
    cols = ["atr_ratio_10_60", "bb_width_pctile", "range_contraction", "inside_days_5",
            "dist_20d_high", "dist_60d_high", "days_since_20d_high", "higher_lows_20", "vol_dryup"]
    pd.testing.assert_frame_equal(
        full.iloc[:100][cols].reset_index(drop=True),
        trunc[cols].reset_index(drop=True),
    )
