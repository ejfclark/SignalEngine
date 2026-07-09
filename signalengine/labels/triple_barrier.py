"""Triple-barrier labeling (López de Prado).

For every (ticker, date) row, simulate the trade the engine would flag:

    entry  = next day's OPEN            (a signal seen after today's close can
                                         only be acted on tomorrow — no lookahead;
                                         falls back to today's CLOSE where the feed
                                         has no open — true for all 2024 rows —
                                         which ignores the overnight gap but keeps
                                         that history usable)
    target = entry + target_mult * ATR14(today)
    stop   = entry - stop_mult   * ATR14(today)
    give up after `horizon` trading days

The label is which barrier is hit first, scanned over daily highs/lows:

    label 1  -> target first (the swing worked)
    label 0  -> stop first, or timeout

Timeout exits at the close of the last day; its realized return keeps the sign
of whatever happened. If high and low breach both barriers on the same day we
cannot know the order from daily bars, so the STOP is assumed first — the
conservative choice; it can only understate performance, never inflate it.

Volatility-scaled barriers mean every trade is labeled against its own risk:
a 2%-a-day crypto coin and a 0.5%-a-day mega-cap get proportionate targets,
and the stop/target the model is trained on is exactly the stop/target the
signal report emits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOME_TARGET = "target"
OUTCOME_STOP = "stop"
OUTCOME_TIMEOUT = "timeout"


def _label_one_ticker(
    g: pd.DataFrame, horizon: int, target_mult: float, stop_mult: float
) -> pd.DataFrame:
    n = len(g)
    open_ = g["open"].to_numpy(float)
    high = g["high"].to_numpy(float)
    low = g["low"].to_numpy(float)
    close = g["close"].to_numpy(float)
    atr = g["atr_14"].to_numpy(float)
    dates = g["date"].to_numpy()

    label = np.full(n, np.nan)
    outcome = np.full(n, None, dtype=object)
    entry_price = np.full(n, np.nan)
    stop_price = np.full(n, np.nan)
    target_price = np.full(n, np.nan)
    exit_price = np.full(n, np.nan)
    exit_idx = np.full(n, -1)

    for t in range(n):
        first = t + 1  # entry day
        last = min(t + horizon, n - 1)
        if first > last or not np.isfinite(atr[t]) or atr[t] <= 0:
            continue
        entry = open_[first]
        if not np.isfinite(entry) or entry <= 0:
            entry = close[t]  # feed has no open for this bar: enter at signal close
        if not np.isfinite(entry) or entry <= 0:
            continue
        tgt = entry + target_mult * atr[t]
        stp = entry - stop_mult * atr[t]

        entry_price[t], target_price[t], stop_price[t] = entry, tgt, stp
        hit = False
        for d in range(first, last + 1):
            # Realistic fills: a gap through a barrier fills at the open, not
            # at the barrier price — unfavorable for stops (gap down), and
            # favorable for targets (gap up). The open is checked before the
            # intraday range so overnight gaps resolve the both-hit ambiguity.
            day_open = open_[d] if np.isfinite(open_[d]) and open_[d] > 0 else None
            if day_open is not None and day_open <= stp:
                label[t], outcome[t] = 0.0, OUTCOME_STOP
                exit_price[t], exit_idx[t] = day_open, d
                hit = True
                break
            if day_open is not None and day_open >= tgt:
                label[t], outcome[t] = 1.0, OUTCOME_TARGET
                exit_price[t], exit_idx[t] = day_open, d
                hit = True
                break
            if low[d] <= stp:  # both-hit day counts as stop (conservative)
                label[t], outcome[t] = 0.0, OUTCOME_STOP
                exit_price[t], exit_idx[t] = stp, d
                hit = True
                break
            if high[d] >= tgt:
                label[t], outcome[t] = 1.0, OUTCOME_TARGET
                exit_price[t], exit_idx[t] = tgt, d
                hit = True
                break
        if not hit:
            # Only a completed window is a real timeout; a window truncated by
            # the end of data stays unlabeled (we don't know the outcome yet).
            if t + horizon <= n - 1:
                label[t], outcome[t] = 0.0, OUTCOME_TIMEOUT
                exit_price[t], exit_idx[t] = close[last], last

    out = g.copy()
    out["label"] = label
    out["outcome"] = outcome
    out["entry_price"] = entry_price
    out["stop_price"] = stop_price
    out["target_price"] = target_price
    out["exit_price"] = exit_price
    out["trade_return"] = exit_price / entry_price - 1.0
    out["exit_date"] = pd.Series(
        [dates[i] if i >= 0 else pd.NaT for i in exit_idx], index=g.index
    )
    return out


def apply_triple_barrier(
    panel: pd.DataFrame, horizon: int = 10, target_mult: float = 3.0, stop_mult: float = 1.5
) -> pd.DataFrame:
    """Label a multi-ticker panel (needs open/high/low/close/atr_14, sorted by date)."""
    parts = [
        _label_one_ticker(g, horizon, target_mult, stop_mult)
        for _, g in panel.groupby("ticker", sort=False)
    ]
    return pd.concat(parts, ignore_index=True)
