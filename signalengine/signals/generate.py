"""Score the latest bar of every ticker and emit actionable signals.

Each signal carries the full trade plan the model was actually trained on:
probability that the +target_mult*ATR take-profit is reached before the
-stop_mult*ATR stop within the horizon, plus the concrete stop/target levels.
"""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier

from ..config import Config
from ..features.pipeline import FEATURE_COLUMNS


def generate_signals(
    features: pd.DataFrame, model: LGBMClassifier, cfg: Config, asof: pd.Timestamp | None = None
) -> pd.DataFrame:
    """features: full feature panel. Scores each ticker's latest row (<= asof)."""
    panel = features if asof is None else features[features["date"] <= asof]
    latest = panel.sort_values("date").groupby("ticker").tail(1).copy()
    latest = latest[latest["atr_14"] > 0]
    # Only score tickers with a current bar: a name whose feed stopped months
    # ago would otherwise be "signalled" off stale data.
    cutoff = panel["date"].max() - pd.Timedelta(days=7)
    stale = latest["date"] < cutoff
    if stale.any():
        print(f"  (skipping {stale.sum()} tickers with no bar since {cutoff.date()})")
        latest = latest[~stale]

    latest["probability"] = model.predict_proba(latest[FEATURE_COLUMNS])[:, 1]

    # Reference levels off the latest close; live entry will be next open.
    tgt_mult, stp_mult = cfg.labels.target_atr_mult, cfg.labels.stop_atr_mult
    latest["stop"] = latest["close"] - stp_mult * latest["atr_14"]
    latest["target"] = latest["close"] + tgt_mult * latest["atr_14"]
    latest["stop_pct"] = latest["stop"] / latest["close"] - 1.0
    latest["target_pct"] = latest["target"] / latest["close"] - 1.0
    latest["reward_risk"] = tgt_mult / stp_mult
    latest["horizon_days"] = cfg.labels.horizon_days

    cols = ["ticker", "date", "close", "probability", "stop", "target",
            "stop_pct", "target_pct", "reward_risk", "horizon_days"]
    if "sector" in latest.columns:
        cols.append("sector")
    return (
        latest[cols]
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )
