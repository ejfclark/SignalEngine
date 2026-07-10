"""Paper-trade ledger: the live-evidence gate between backtest and real money.

Nightly, after signal generation:

    signalengine ledger update    advance open positions against new bars
    signalengine ledger record    open virtual positions from today's signals
    signalengine ledger report    live stats vs backtest expectation

State lives in data/ledger.parquet — one row per virtual trade:
    opened (signal date), ticker, asset, direction, probability,
    entry_price (NaN until the entry bar arrives), stop, target, horizon_end,
    status: pending -> open -> target/stop/timeout (or 'expired' if no
    entry bar ever arrives), exit_date, exit_price, net_return.

Fills follow the exact label semantics: entry at the first bar's open after
the signal; a bar OPENING beyond a barrier fills at that open (gap), an
intraday touch fills at the barrier; both-hit days count as stops; timeout
exits at the close of the last bar inside the horizon. Costs are charged at
the configured round trip. If live results diverge badly from the OOS
backtest, that is the ledger doing its job — believe it, not the backtest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

LEDGER_FILE = "ledger.parquet"

# What qualifies as an actionable signal per book. Kept in one place so the
# ledger, the portal, and future execution agree on what "we would trade" means.
BOOKS = {
    "crypto": {"threshold": 0.65, "direction": "long"},
    "crypto-short": {"threshold": 0.70, "direction": "short"},
    "stock": {"threshold": 0.70, "direction": "long"},  # experimental book
}


def _ledger_path(cfg: Config) -> Path:
    return cfg.root / "data" / LEDGER_FILE


def load_ledger(cfg: Config) -> pd.DataFrame:
    path = _ledger_path(cfg)
    if path.is_file():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=[
        "opened", "ticker", "asset", "direction", "probability",
        "entry_price", "stop", "target", "horizon_end",
        "status", "exit_date", "exit_price", "net_return",
    ])


def _save(cfg: Config, ledger: pd.DataFrame) -> None:
    path = _ledger_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(path, index=False)


def record_signals(cfg: Config) -> int:
    """Open pending positions from today's signal CSVs (idempotent per day)."""
    ledger = load_ledger(cfg)
    added = 0
    for tag, rule in BOOKS.items():
        path = cfg.artifacts_dir / f"{tag}_signals.csv"
        if not path.is_file():
            continue
        signals = pd.read_csv(path, parse_dates=["date"])
        picks = signals[signals["probability"] >= rule["threshold"]]
        for _, s in picks.iterrows():
            duplicate = (
                (ledger["ticker"] == s["ticker"])
                & (ledger["asset"] == tag)
                & (ledger["opened"] == s["date"])
            )
            if duplicate.any():
                continue
            ledger = pd.concat([ledger, pd.DataFrame([{
                "opened": s["date"],
                "ticker": s["ticker"],
                "asset": tag,
                "direction": rule["direction"],
                "probability": float(s["probability"]),
                "entry_price": np.nan,
                "stop": float(s["stop"]),
                "target": float(s["target"]),
                "horizon_end": s["date"] + pd.Timedelta(days=int(
                    cfg.labels.horizon_days * 1.6  # trading -> calendar days
                )),
                "status": "pending",
                "exit_date": pd.NaT,
                "exit_price": np.nan,
                "net_return": np.nan,
            }])], ignore_index=True)
            added += 1
    _save(cfg, ledger)
    return added


def _bars_after(prices: pd.DataFrame, ticker: str, after: pd.Timestamp) -> pd.DataFrame:
    bars = prices[(prices["ticker"] == ticker) & (prices["date"] > after)]
    return bars.sort_values("date")


def update_positions(cfg: Config) -> dict:
    """Advance pending/open positions using the latest bars in the lake."""
    from .data import get_source

    ledger = load_ledger(cfg)
    if ledger.empty:
        return {"filled": 0, "closed": 0}

    source = get_source(cfg)
    prices = {"stock": source.load_stock_prices(), "crypto": source.load_crypto_prices()}
    cost = 2.0 * (cfg.backtest.fee_bps + cfg.backtest.slippage_bps) / 1e4
    filled = closed = 0

    for i, t in ledger.iterrows():
        if t["status"] not in ("pending", "open"):
            continue
        panel = prices["stock" if t["asset"] == "stock" else "crypto"]
        bars = _bars_after(panel, t["ticker"], t["opened"])
        if bars.empty:
            if pd.Timestamp.now() > t["horizon_end"] + pd.Timedelta(days=5):
                ledger.loc[i, "status"] = "expired"  # no bar ever arrived
            continue

        entry = t["entry_price"]
        if t["status"] == "pending":
            first = bars.iloc[0]
            entry = first["open"] if np.isfinite(first["open"]) and first["open"] > 0 else first["close"]
            ledger.loc[i, "entry_price"] = entry
            ledger.loc[i, "status"] = "open"
            filled += 1

        sign = 1.0 if t["direction"] == "long" else -1.0
        stp, tgt = t["stop"], t["target"]
        for _, bar in bars.iterrows():
            day_open = bar["open"] if np.isfinite(bar["open"]) and bar["open"] > 0 else None
            exit_price = outcome = None
            if day_open is not None and sign * (day_open - stp) <= 0:
                exit_price, outcome = day_open, "stop"
            elif day_open is not None and sign * (day_open - tgt) >= 0:
                exit_price, outcome = day_open, "target"
            else:
                bar_stop = bar["low"] if sign > 0 else bar["high"]
                bar_target = bar["high"] if sign > 0 else bar["low"]
                if sign * (bar_stop - stp) <= 0:
                    exit_price, outcome = stp, "stop"
                elif sign * (bar_target - tgt) >= 0:
                    exit_price, outcome = tgt, "target"
                elif bar["date"] >= t["horizon_end"]:
                    exit_price, outcome = bar["close"], "timeout"
            if exit_price is not None:
                ledger.loc[i, ["status", "exit_date", "exit_price"]] = (
                    outcome, bar["date"], exit_price)
                ledger.loc[i, "net_return"] = sign * (exit_price / entry - 1.0) - cost
                closed += 1
                break

    _save(cfg, ledger)
    return {"filled": filled, "closed": closed}


def report(cfg: Config) -> pd.DataFrame:
    """Per-book live stats. The comparison row is the backtest expectation."""
    ledger = load_ledger(cfg)
    if ledger.empty:
        print("ledger is empty — no signals recorded yet")
        return ledger
    done = ledger[ledger["status"].isin(("target", "stop", "timeout"))]
    print(f"{'book':<14} {'open':>5} {'closed':>7} {'hit':>7} {'expect':>8} {'total':>8}")
    for tag in BOOKS:
        mine = ledger[ledger["asset"] == tag]
        fin = done[done["asset"] == tag]
        hit = (fin["net_return"] > 0).mean() if len(fin) else float("nan")
        exp = fin["net_return"].mean() if len(fin) else float("nan")
        tot = fin["net_return"].sum() if len(fin) else 0.0
        n_open = int((mine["status"].isin(("pending", "open"))).sum())
        print(f"{tag:<14} {n_open:>5} {len(fin):>7} {hit:>7.1%} {exp:>8.2%} {tot:>8.2%}")
    print("\nbacktest expectation: crypto +0.85%/trade, crypto-short +0.96%, stock +0.3% (thin)")
    return ledger
