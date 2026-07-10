"""Lockbox holdout: the one-shot honesty check for a frozen system.

`split_lockbox` — used by every bench/experiment path: rows whose trade could
touch the lockbox period are removed, so iteration can never learn from it.
A row leaks if its exit lands on/after lockbox_start, so the cut is on
exit_date (with unresolved rows near the boundary dropped too).

`lockbox_eval` — the single spend: train on everything before the lockbox,
trade the lockbox period exactly as the nightly system would (this book's
threshold/sizing/gate), and report. After running this, the number is the
number — no fixing things and re-running against the same months. Move
lockbox_start forward and let the paper ledger be the ongoing clean gate.
"""

from __future__ import annotations

import json

import pandas as pd

from .backtest import run_backtest
from .config import Config


def lockbox_ts(cfg: Config) -> pd.Timestamp | None:
    return pd.Timestamp(cfg.cv.lockbox_start) if cfg.cv.lockbox_start else None


def split_lockbox(cfg: Config, labeled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(iterable_history, lockbox_rows). Boundary-crossing rows are in neither."""
    start = lockbox_ts(cfg)
    if start is None:
        return labeled, labeled.iloc[0:0]
    exit_date = pd.to_datetime(labeled["exit_date"])
    history = labeled[exit_date < start]
    lockbox = labeled[labeled["date"] >= start]
    return history.reset_index(drop=True), lockbox.reset_index(drop=True)


def lockbox_eval(cfg: Config, asset: str, direction: str = "long") -> dict:
    from .cli import _tag, build_dataset
    from .model.train import _make_model
    from .features.pipeline import FEATURE_COLUMNS

    tag = _tag(asset, direction)
    labeled = build_dataset(cfg, asset, direction)
    history, lockbox = split_lockbox(cfg, labeled)
    history = history.dropna(subset=["label"])
    lockbox = lockbox.dropna(subset=["label"])
    if lockbox.empty:
        raise ValueError("lockbox period contains no labeled rows")

    model = _make_model(cfg, history["label"].to_numpy())
    model.fit(history[FEATURE_COLUMNS], history["label"])
    scored = lockbox.copy()
    scored["probability"] = model.predict_proba(lockbox[FEATURE_COLUMNS])[:, 1]

    bt = cfg.backtest_for(tag)
    stats = run_backtest(
        scored, bt.probability_threshold, bt.fee_bps, bt.slippage_bps, bt.max_positions,
        sizing=bt.sizing, risk_pct=bt.risk_pct, top_n=bt.top_n or None,
        gate_column=bt.gate_column or None, gate_min=bt.gate_min,
    ).stats

    payload = {
        "tag": tag,
        "lockbox_start": cfg.cv.lockbox_start,
        "train_rows": len(history),
        "lockbox_rows": len(lockbox),
        "threshold": bt.probability_threshold,
        "stats": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in stats.items()},
    }
    out = cfg.artifacts_dir / "bench" / f"lockbox_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return payload
