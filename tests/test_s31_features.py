import numpy as np
import pandas as pd
import pytest

from signalengine.features.indicators import compute_indicators
from signalengine.features.pipeline import build_features


def make_ohlcv(closes, start="2024-01-02"):
    closes = pd.Series(closes, dtype=float)
    return pd.DataFrame({
        "ticker": "T",
        "date": pd.date_range(start, periods=len(closes), freq="B"),
        "open": closes.shift(1).fillna(closes.iloc[0]),
        "high": closes * 1.01,
        "low": closes * 0.99,
        "close": closes,
        "volume": pd.Series(1e6, index=closes.index),
    })


def with_earnings(df, event_iloc, jump_to=None):
    """As-of next-earnings column: the event date, rolling to a later one
    the bar after it passes (how the fundamentals feed behaves)."""
    event = df["date"].iloc[event_iloc]
    later = df["date"].iloc[-1] + pd.Timedelta(days=40)
    df = df.copy()
    df["earnings_date"] = np.where(df["date"] <= event, event, later)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    if jump_to is not None:
        df["close"] = np.where(df["date"] > event, jump_to, df["close"])
        df["high"] = df["close"] * 1.01
        df["low"] = df["close"] * 0.99
    return df


def test_earnings_reaction_value_and_timing():
    df = with_earnings(make_ohlcv([100.0] * 40), event_iloc=20, jump_to=110.0)
    g = compute_indicators(df)
    # Reaction = close before event (100) -> first close after (110) = +10%.
    post = g[g["date"] > df["date"].iloc[20]]
    assert post["earnings_reaction"].iloc[0] == pytest.approx(0.10)
    # Never populated on or before the event day itself (that close is the
    # first observable evidence; the day-of row cannot see the post close).
    on_or_before = g[g["date"] <= df["date"].iloc[20]]
    assert on_or_before["earnings_reaction"].isna().all()
    # days_since_earnings counts up from the event.
    assert post["days_since_earnings"].iloc[0] >= 1


def test_earnings_features_no_lookahead():
    closes = 100 + np.cumsum(np.random.default_rng(7).normal(0, 1, 120))
    df = with_earnings(make_ohlcv(closes), event_iloc=50)
    full = compute_indicators(df)
    trunc = compute_indicators(df.iloc[:80].copy())
    cols = ["days_to_earnings", "days_since_earnings", "earnings_reaction"]
    pd.testing.assert_frame_equal(
        full.iloc[:80][cols].reset_index(drop=True),
        trunc[cols].reset_index(drop=True),
    )


def test_beta_of_levered_ticker_is_two():
    rng = np.random.default_rng(11)
    spy_ret = rng.normal(0.0005, 0.01, 200)
    spy_close = 100 * np.cumprod(1 + spy_ret)
    t_close = 100 * np.cumprod(1 + 2 * spy_ret)  # exactly 2x SPY daily moves

    prices = make_ohlcv(t_close)
    etf = make_ohlcv(spy_close)
    etf["ticker"] = "SPY"

    panel = build_features(prices, etf_prices=etf)
    last = panel.iloc[-1]
    assert abs(last["beta_spy_120d"] - 2.0) < 0.05
    assert last["corr_spy_120d"] > 0.99
    assert last["idio_vol_share"] < 0.05


def test_dollar_vol_rank_orders_by_liquidity():
    a = make_ohlcv([100.0] * 60)
    b = make_ohlcv([100.0] * 60)
    b["ticker"] = "B"
    b["volume"] = 1e8  # far more liquid
    panel = build_features(pd.concat([a, b], ignore_index=True), with_market_context=False)
    last_day = panel[panel["date"] == panel["date"].max()].set_index("ticker")
    assert last_day.loc["B", "dollar_vol_rank"] > last_day.loc["T", "dollar_vol_rank"]


def test_eps_chg_yoy_uses_full_year_lookback():
    n = 400
    eps = pd.Series([1.0] * 252 + [1.5] * (n - 252))
    df = make_ohlcv([100.0] * n)
    df["eps"] = eps
    g = compute_indicators(df)
    # At day 300 (48 trading days after the step), YoY compares to day 48
    # (eps=1.0), a full year back -- not the 63d quarterly window.
    assert g["eps_chg_yoy"].iloc[300] == pytest.approx(0.5)
    assert g["eps_chg_yoy"].iloc[:252].isna().all()


def test_eps_chg_yoy_no_lookahead():
    rng = np.random.default_rng(9)
    df = make_ohlcv(100 + np.cumsum(rng.normal(0, 1, 400)))
    df["eps"] = np.repeat(rng.normal(2, 0.3, 8), 50)
    full = compute_indicators(df)
    trunc = compute_indicators(df.iloc[:300].copy())
    pd.testing.assert_frame_equal(
        full.iloc[:300][["eps_chg_yoy"]].reset_index(drop=True),
        trunc[["eps_chg_yoy"]].reset_index(drop=True),
    )
