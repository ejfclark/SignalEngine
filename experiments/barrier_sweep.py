"""Exp D: barrier/horizon sweep for the crypto long book.

Selection discipline: combos are RANKED on the early folds (0-2, through
~2024) and the ranking is then CHECKED on the late folds (3-4). Picking the
combo that's best on the same folds you report would just be overfitting the
backtest; this split keeps the choice and the evidence separate.

Cross-combo comparisons use backtest stats (same capital rules for all), not
AUC — different barriers define different prediction tasks, so AUC is not
comparable across combos.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signalengine.backtest import run_backtest
from signalengine.cli import build_dataset
from signalengine.config import load_config
from signalengine.model.train import train_walk_forward

COMBOS = [
    # (target_mult, stop_mult, horizon)
    (3.0, 1.5, 10),   # current production
    (2.0, 1.0, 10),
    (2.5, 1.25, 10),
    (4.0, 2.0, 10),
    (3.0, 1.0, 10),   # 3:1 reward:risk
    (2.0, 1.5, 10),   # tighter target
    (3.0, 1.5, 15),   # longer horizon
    (3.0, 1.5, 7),    # shorter horizon
]
THRESHOLD = 0.65


def main() -> None:
    cfg = load_config()
    results = []
    for tgt, stp, horizon in COMBOS:
        cfg.labels.target_atr_mult = tgt
        cfg.labels.stop_atr_mult = stp
        cfg.labels.horizon_days = horizon
        cfg.cv.purge_days = max(horizon, 10)

        labeled = build_dataset(cfg, "crypto")
        res = train_walk_forward(labeled, cfg)

        early = res.oos[res.oos["fold"] <= 2]
        late = res.oos[res.oos["fold"] >= 3]
        bt = cfg.backtest_for("crypto")
        row = {"target": tgt, "stop": stp, "horizon": horizon}
        for name, chunk in (("early", early), ("late", late)):
            s = run_backtest(chunk, THRESHOLD, bt.fee_bps, bt.slippage_bps,
                             bt.max_positions, sizing="vol",
                             gate_column="breadth_20d", gate_min=0.30).stats
            row[f"{name}_expect"] = s.get("expectancy", float("nan"))
            row[f"{name}_sharpe"] = s.get("sharpe", float("nan"))
            row[f"{name}_trades"] = s.get("n_trades", 0)
        results.append(row)
        print(f"tgt {tgt} stop {stp} h{horizon}: "
              f"early {row['early_expect']:+.2%}/{row['early_sharpe']:.2f}  "
              f"late {row['late_expect']:+.2%}/{row['late_sharpe']:.2f}", flush=True)

    df = pd.DataFrame(results).sort_values("early_sharpe", ascending=False)
    out = cfg.artifacts_dir / "bench" / "barrier_sweep_crypto.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nranked by EARLY sharpe (selection), check LATE column (validation):")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
