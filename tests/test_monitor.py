import numpy as np
import pandas as pd
import pytest

from signalengine import monitor as M
from signalengine.config import BacktestConfig, Config


@pytest.fixture
def cfg(tmp_path):
    cfg = Config(root=tmp_path)
    cfg.artifacts_dir = tmp_path / "artifacts"
    cfg.artifacts_dir.mkdir()
    return cfg


def make_oos(n_per_fold, n_folds, returns_by_fold, threshold=0.60):
    """Synthetic OOS predictions: n_per_fold trades/fold, all above threshold,
    non-overlapping exit windows so every trade gets its own portfolio slot."""
    rows = []
    day = pd.Timestamp("2024-01-01")
    for fold in range(n_folds):
        rets = returns_by_fold[fold]
        for i in range(n_per_fold):
            rows.append({
                "ticker": f"T{i}", "date": day, "fold": fold,
                "probability": threshold + 0.05,
                "entry_price": 100.0, "stop_price": 95.0, "target_price": 110.0,
                "exit_price": 100.0 * (1 + rets[i % len(rets)]),
                "exit_date": day + pd.Timedelta(days=1),
                "trade_return": rets[i % len(rets)],
                "outcome": "target" if rets[i % len(rets)] > 0 else "stop",
                "label": 1 if rets[i % len(rets)] > 0 else 0,
                "close": 100.0,
            })
            day += pd.Timedelta(days=2)  # stagger so slots don't collide
    return pd.DataFrame(rows)


def write_oos(cfg, tag, df):
    df.to_parquet(cfg.artifacts_dir / f"{tag}_oos_predictions.parquet", index=False)


def make_ledger(tag, n, mean_return, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(mean_return, 0.05, n)
    return pd.DataFrame({
        "asset": tag, "status": "stop",
        "net_return": rets,
        "opened": pd.date_range("2024-06-01", periods=n, freq="D"),
    })


def test_returns_none_below_min_trades(cfg):
    write_oos(cfg, "crypto", make_oos(30, 5, [[0.02, -0.01]] * 5))
    ledger = make_ledger("crypto", 10, 0.02)
    assert M.check_book(cfg, "crypto", ledger, min_live_trades=30) is None


def test_flags_when_live_expectancy_outside_fold_range(cfg):
    # Every fold has a clearly positive expectancy (mean return ~+2%).
    write_oos(cfg, "crypto", make_oos(30, 5, [[0.05, 0.03, -0.01]] * 5))
    # Live book is consistently and clearly losing -- well outside any
    # plausible fold range, with a tight live distribution (small std) so
    # the 2-SE noise margin can't swallow it.
    ledger = make_ledger("crypto", 200, -0.05, seed=1)
    d = M.check_book(cfg, "crypto", ledger, min_live_trades=30)
    assert d is not None
    assert d.flagged
    assert d.live_expectancy < d.fold_low


def test_does_not_flag_when_live_within_fold_noise(cfg):
    # Folds span a wide range (some negative, some strongly positive) --
    # realistic run-to-run/regime variance.
    write_oos(cfg, "crypto", make_oos(30, 5, [
        [0.08, 0.05, -0.02], [-0.01, -0.02, 0.01], [0.03, -0.01, 0.02],
        [0.06, 0.04, -0.01], [0.02, 0.01, -0.01],
    ]))
    ledger = make_ledger("crypto", 40, 0.01, seed=2)
    d = M.check_book(cfg, "crypto", ledger, min_live_trades=30)
    assert d is not None
    assert not d.flagged


def test_run_monitor_reports_missing_predictions_gracefully(cfg, capsys):
    from signalengine.ledger import _save

    _save(cfg, make_ledger("crypto", 40, 0.0))
    results = M.run_monitor(cfg, min_live_trades=30)
    assert results == []
    assert "no OOS predictions" in capsys.readouterr().out
