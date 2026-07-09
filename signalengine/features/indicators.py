"""Vectorized technical indicators, computed per ticker on a date-sorted frame.

Pure pandas/numpy (no TA-Lib binary): deterministic, testable, installs anywhere.
All momentum-style features are expressed relative to price (percent / normalized)
rather than absolute levels, so a $5 stock and a $500 stock are comparable and
the model generalizes across the universe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def wilder(s: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (used by RSI/ATR/ADX): EMA with alpha = 1/period."""
    return s.ewm(alpha=1.0 / period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI: SMA-seeded, then recursive smoothing (matches TA-Lib)."""
    delta = close.diff().to_numpy(float)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    n = len(close)
    out = np.full(n, np.nan)
    if n > period:
        avg_gain = gain[1 : period + 1].mean()
        avg_loss = loss[1 : period + 1].mean()
        for i in range(period, n):
            if i > period:
                avg_gain = (avg_gain * (period - 1) + gain[i]) / period
                avg_loss = (avg_loss * (period - 1) + loss[i]) / period
            total = avg_gain + avg_loss
            out[i] = 50.0 if total == 0 else 100.0 * avg_gain / total
    return pd.Series(out, index=close.index)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return wilder(tr, period)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Returns adx, plus_di, minus_di."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)

    atr_ = atr(high, low, close, period).replace(0.0, np.nan)
    plus_di = 100.0 * wilder(plus_dm, period) / atr_
    minus_di = 100.0 * wilder(minus_dm, period) / atr_
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return pd.DataFrame({"adx": wilder(dx, period), "plus_di": plus_di, "minus_di": minus_di})


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, smooth: int = 3
) -> pd.DataFrame:
    """Slow stochastic %K/%D."""
    ll = low.rolling(k_period).min()
    hh = high.rolling(k_period).max()
    fast_k = 100.0 * (close - ll) / (hh - ll).replace(0.0, np.nan)
    slow_k = fast_k.rolling(smooth).mean()
    slow_d = slow_k.rolling(smooth).mean()
    return pd.DataFrame({"stoch_k": slow_k, "stoch_d": slow_d})


def compute_indicators(g: pd.DataFrame) -> pd.DataFrame:
    """Add indicator columns to one ticker's date-sorted OHLCV frame."""
    g = g.copy()
    close, high, low = g["close"], g["high"], g["low"]

    # Multi-horizon momentum and volatility
    for n in (1, 5, 10, 20):
        g[f"ret_{n}d"] = close.pct_change(n)
    g["vol_20d"] = g["ret_1d"].rolling(20).std() * np.sqrt(252)

    g["atr_14"] = atr(high, low, close)
    g["atr_pct"] = g["atr_14"] / close

    g["rsi_14"] = rsi(close)

    macd_line = ema(close, 12) - ema(close, 26)
    macd_signal = ema(macd_line, 9)
    # Normalized by price so it is comparable across tickers.
    g["macd_pct"] = macd_line / close
    g["macd_hist_pct"] = (macd_line - macd_signal) / close

    a = adx(high, low, close)
    g["adx_14"] = a["adx"]
    g["di_spread"] = a["plus_di"] - a["minus_di"]  # >0 bullish, <0 bearish

    st = stochastic(high, low, close)
    g["stoch_k"] = st["stoch_k"]
    g["stoch_kd_spread"] = st["stoch_k"] - st["stoch_d"]

    # EMA ribbon (5/8/13/21): how much of the stack is in bullish order, 0..4
    e5, e8, e13, e21 = (ema(close, n) for n in (5, 8, 13, 21))
    g["ema_ribbon"] = (
        (close > e5).astype(float) + (e5 > e8).astype(float)
        + (e8 > e13).astype(float) + (e13 > e21).astype(float)
    )

    # Distance from moving averages, in percent
    g["dist_sma20"] = close / close.rolling(20).mean() - 1.0
    sma50 = g["price_avg50"] if "price_avg50" in g and g["price_avg50"].notna().any() else close.rolling(50).mean()
    g["dist_sma50"] = close / sma50.replace(0.0, np.nan) - 1.0
    sma200 = g["price_avg200"] if "price_avg200" in g and g["price_avg200"].notna().any() else close.rolling(200).mean()
    g["dist_sma200"] = close / sma200.replace(0.0, np.nan) - 1.0

    # Volume: z-score vs its own 20-day history; spikes often precede swings
    vol = g["volume"]
    vol_std = vol.rolling(20).std().replace(0.0, np.nan)
    g["volume_z"] = (vol - vol.rolling(20).mean()) / vol_std

    # Daily structure
    g["gap_pct"] = g["open"] / close.shift(1) - 1.0
    g["range_pct"] = (high - low) / close

    # 52-week-ish position within available history (min 60 days)
    roll_max = close.rolling(252, min_periods=60).max()
    roll_min = close.rolling(252, min_periods=60).min()
    g["pct_off_high"] = close / roll_max - 1.0
    g["pos_in_range"] = (close - roll_min) / (roll_max - roll_min).replace(0.0, np.nan)

    # Stock fundamentals when present
    if "earnings_date" in g:
        days = (g["earnings_date"] - g["date"]).dt.days
        g["days_to_earnings"] = days.where((days >= 0) & (days <= 90))
    if "pe" in g:
        g["pe_ratio"] = g["pe"].where(g["pe"] > 0)

    return g
