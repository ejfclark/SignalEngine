"""Live-vs-backtest divergence check: the automated version of "does the
paper ledger's realized performance look like the model's own backtest, or
has something drifted." Runs per book against `{tag}_oos_predictions.parquet`
(written by the last `train`) — no bench/lockbox involved, so this is safe to
run nightly without touching the experiment-gating machinery.

Method: backtest each walk-forward fold separately (same rules the book
actually trades under: threshold + gate from cfg.backtest_for(tag)) to get a
fold-to-fold expectancy distribution — the model's own honest range, the way
the multi-seed sweeps in docs/05-experiments.md treat run-to-run variance.
Compare the live ledger's realized expectancy against that range. Flag only
when live falls outside it — small samples inside the range are not noteworthy,
and this must not become a second place experiment verdicts get decided.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .config import Config
from .ledger import BOOKS
from .model.train import PREDICTIONS_FILE


@dataclass
class Divergence:
    tag: str
    live_trades: int
    live_expectancy: float
    fold_expectancies: list[float]
    fold_low: float
    fold_high: float
    flagged: bool
    reason: str


def _asset_direction(tag: str) -> tuple[str, str]:
    rule = BOOKS[tag]
    asset = tag.split("-")[0]
    return asset, rule["direction"]


def fold_expectancy_range(cfg: Config, tag: str) -> list[float]:
    """Backtest each fold of the CURRENT model's OOS predictions separately,
    under the book's actual production rules. Returns one expectancy per fold
    with >=20 trades (smaller folds are too noisy to call a range point)."""
    path = cfg.artifacts_dir / f"{tag}_{PREDICTIONS_FILE}"
    if not path.is_file():
        raise FileNotFoundError(f"no OOS predictions at {path} — run `train` for {tag} first")
    oos = pd.read_parquet(path)
    bt = cfg.backtest_for(tag)

    expectancies = []
    for _, fold_rows in oos.groupby("fold"):
        result = run_backtest(
            fold_rows, bt.probability_threshold, bt.fee_bps, bt.slippage_bps,
            bt.max_positions, sizing=bt.sizing, risk_pct=bt.risk_pct,
            top_n=bt.top_n or None, gate_column=bt.gate_column or None, gate_min=bt.gate_min,
        )
        if result.stats.get("n_trades", 0) >= 20:
            expectancies.append(float(result.stats["expectancy"]))
    return expectancies


def check_book(cfg: Config, tag: str, ledger: pd.DataFrame, min_live_trades: int = 30) -> Divergence | None:
    """None if there isn't enough live evidence yet to say anything meaningful."""
    done = ledger[(ledger["asset"] == tag) & (ledger["status"].isin(("target", "stop", "timeout")))]
    if len(done) < min_live_trades:
        return None

    live_exp = float(done["net_return"].mean())
    folds = fold_expectancy_range(cfg, tag)
    if not folds:
        return None

    lo, hi = min(folds), max(folds)
    # Live is judged against the fold range widened by its own live-sample
    # standard error — a small live sample can swing further than the fold
    # range by chance alone, and that alone isn't evidence of drift.
    se = float(done["net_return"].std(ddof=1)) / np.sqrt(len(done)) if len(done) > 1 else 0.0
    margin = 2.0 * se
    flagged = live_exp < (lo - margin) or live_exp > (hi + margin)
    reason = ""
    if flagged:
        side = "below" if live_exp < lo else "above"
        reason = (f"live expectancy {side} the model's own fold-to-fold range "
                  f"even after a 2-SE margin for live-sample noise")
    return Divergence(tag, len(done), live_exp, folds, lo, hi, flagged, reason)


def run_monitor(cfg: Config, min_live_trades: int = 30) -> list[Divergence]:
    from .ledger import load_ledger

    ledger = load_ledger(cfg)
    results = []
    for tag in BOOKS:
        try:
            d = check_book(cfg, tag, ledger, min_live_trades)
        except FileNotFoundError as e:
            print(f"  {tag}: {e}")
            continue
        if d is None:
            done = len(ledger[(ledger["asset"] == tag)
                              & (ledger["status"].isin(("target", "stop", "timeout")))])
            print(f"  {tag}: only {done}/{min_live_trades} closed trades — too early to check")
            continue
        results.append(d)
        flag = "!! DIVERGENCE" if d.flagged else "ok"
        print(f"  {tag:<14} [{flag}] live {d.live_expectancy:+.2%} over {d.live_trades} trades  "
              f"vs fold range [{d.fold_low:+.2%}, {d.fold_high:+.2%}] "
              f"({len(d.fold_expectancies)} folds)")
        if d.flagged:
            print(f"                 {d.reason}")
    return results
