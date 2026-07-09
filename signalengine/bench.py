"""Experiment bench: measure a code state, compare against a named baseline.

    signalengine bench --name baseline --asset stock
    signalengine bench --name exp2-vol-sizing --asset stock
    signalengine bench-compare baseline exp2-vol-sizing --asset stock

Every run rebuilds the dataset from the lake (so feature changes are picked
up), trains walk-forward, backtests the OOS predictions across a threshold
grid, and writes artifacts/bench/<name>_<asset>.json. The acceptance rule for
an experiment: it must beat the baseline on the metrics it claims to improve
WITHOUT degrading fold AUC or expectancy elsewhere. Rejected experiments get
reverted; the bench file stays as the record either way.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .backtest import run_backtest
from .config import Config

THRESHOLDS = [0.55, 0.60, 0.65, 0.70]


def run_bench(cfg: Config, asset: str, name: str) -> dict:
    from .cli import build_dataset
    from .model.train import train_walk_forward

    labeled = build_dataset(cfg, asset)
    result = train_walk_forward(labeled, cfg)

    folds = result.fold_metrics.to_dict("records")
    backtests = {}
    for threshold in THRESHOLDS:
        bt = run_backtest(
            result.oos, threshold,
            cfg.backtest.fee_bps, cfg.backtest.slippage_bps, cfg.backtest.max_positions,
        )
        backtests[str(threshold)] = {
            k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
            for k, v in bt.stats.items()
        }

    payload = {
        "name": name,
        "asset": asset,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "labeled_rows": int(labeled["label"].notna().sum()),
        "mean_auc": float(np.nanmean([f["auc"] for f in folds])),
        "min_auc": float(np.nanmin([f["auc"] for f in folds])),
        "folds": [
            {k: (str(v) if not isinstance(v, (int, float, np.floating, np.integer)) else float(v))
             for k, v in f.items()} for f in folds
        ],
        "backtests": backtests,
        "top_features": result.importance.head(10).to_dict("records"),
    }
    out_dir = cfg.artifacts_dir / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_{asset}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _fmt(v, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:+.1%}" if pct else (f"{v:.3f}" if isinstance(v, float) else str(v))


def compare(cfg: Config, asset: str, base_name: str, exp_name: str) -> None:
    out_dir = cfg.artifacts_dir / "bench"
    base = json.loads((out_dir / f"{base_name}_{asset}.json").read_text(encoding="utf-8"))
    exp = json.loads((out_dir / f"{exp_name}_{asset}.json").read_text(encoding="utf-8"))

    print(f"\n{asset}: {base_name}  vs  {exp_name}")
    print(f"  mean AUC   {base['mean_auc']:.4f} -> {exp['mean_auc']:.4f}"
          f"   ({exp['mean_auc'] - base['mean_auc']:+.4f})")
    print(f"  min AUC    {base['min_auc']:.4f} -> {exp['min_auc']:.4f}")
    header = f"  {'thr':>5} {'trades':>13} {'hit':>11} {'expect':>15} {'PF':>11} {'sharpe':>11} {'maxDD':>13} {'return':>15}"
    print(header)
    for thr in THRESHOLDS:
        b = base["backtests"].get(str(thr), {})
        e = exp["backtests"].get(str(thr), {})
        def pair(key, pct=False):
            return f"{_fmt(b.get(key), pct)}->{_fmt(e.get(key), pct)}"
        print(f"  {thr:>5} {pair('n_trades'):>13} {pair('hit_rate'):>11} {pair('expectancy', True):>15}"
              f" {pair('profit_factor'):>11} {pair('sharpe'):>11} {pair('max_drawdown', True):>13}"
              f" {pair('total_return', True):>15}")
