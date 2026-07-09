import numpy as np
import pandas as pd
import pytest

from signalengine.features.indicators import adx, atr, compute_indicators, rsi, stochastic


def make_ohlcv(closes, seed=0):
    rng = np.random.default_rng(seed)
    close = pd.Series(closes, dtype=float)
    high = close * (1 + rng.uniform(0.001, 0.01, len(close)))
    low = close * (1 - rng.uniform(0.001, 0.01, len(close)))
    return pd.DataFrame({
        "ticker": "TEST",
        "date": pd.date_range("2024-01-01", periods=len(close), freq="B"),
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": high, "low": low, "close": close,
        "volume": rng.integers(1e5, 1e6, len(close)).astype(float),
    })


def test_rsi_bounds_and_direction():
    up = rsi(pd.Series(np.linspace(100, 200, 60))).dropna()
    down = rsi(pd.Series(np.linspace(200, 100, 60))).dropna()
    assert up.between(0, 100).all() and down.between(0, 100).all()
    assert up.iloc[-1] > 95      # relentless rally -> RSI ~100
    assert down.iloc[-1] < 5     # relentless slide -> RSI ~0


def test_rsi_reference_value():
    # Wilder's classic worked example (14-period)
    closes = pd.Series([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
                        45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28])
    value = rsi(closes, 14).iloc[-1]
    assert value == pytest.approx(70.46, abs=0.6)


def test_atr_positive_and_scales_with_volatility():
    calm = make_ohlcv(100 + np.sin(np.arange(80)) * 0.5)
    wild = make_ohlcv(100 + np.sin(np.arange(80)) * 10)
    a_calm = atr(calm["high"], calm["low"], calm["close"]).iloc[-1]
    a_wild = atr(wild["high"], wild["low"], wild["close"]).iloc[-1]
    assert a_calm > 0 and a_wild > 5 * a_calm


def test_adx_strong_trend_vs_chop():
    trend = make_ohlcv(np.linspace(100, 300, 100))
    chop = make_ohlcv(100 + np.sin(np.arange(100) * 2.1) * 2)
    adx_trend = adx(trend["high"], trend["low"], trend["close"])["adx"].iloc[-1]
    adx_chop = adx(chop["high"], chop["low"], chop["close"])["adx"].iloc[-1]
    assert adx_trend > 40
    assert adx_trend > adx_chop


def test_stochastic_bounds():
    df = make_ohlcv(100 + np.cumsum(np.random.default_rng(1).normal(0, 1, 120)))
    st = stochastic(df["high"], df["low"], df["close"]).dropna()
    assert st["stoch_k"].between(-1e-9, 100 + 1e-9).all()


def test_compute_indicators_no_lookahead():
    """Truncating the input must not change earlier feature values."""
    df = make_ohlcv(100 + np.cumsum(np.random.default_rng(2).normal(0, 1, 150)))
    full = compute_indicators(df)
    truncated = compute_indicators(df.iloc[:100].copy())
    feature_cols = [c for c in full.columns if c not in df.columns]
    pd.testing.assert_frame_equal(
        full.iloc[:100][feature_cols].reset_index(drop=True),
        truncated[feature_cols].reset_index(drop=True),
    )
