"""Cost-aware backtest over the walk-forward OUT-OF-SAMPLE predictions only.

Every candidate row already carries its realized exit (from the triple-barrier
scan), so the simulation is: take signals above the probability threshold,
respect a max concurrent position count, charge fees+slippage both ways, and
compound an equal-split portfolio. Deliberately simple — its job is an honest
expectancy estimate, not broker emulation.

If the stats here don't clear costs, the model does not go live. That's the
gate the 2024 version of this project never had.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    stats: dict


def run_backtest(
    oos: pd.DataFrame,
    threshold: float = 0.60,
    fee_bps: float = 10.0,
    slippage_bps: float = 10.0,
    max_positions: int = 10,
) -> BacktestResult:
    cost = 2.0 * (fee_bps + slippage_bps) / 1e4  # round trip, both ways

    candidates = (
        oos[(oos["probability"] >= threshold) & oos["exit_date"].notna()]
        .sort_values(["date", "probability"], ascending=[True, False])
        .copy()
    )

    # Greedy portfolio: on each signal date take the highest-probability names
    # while slots are free; a position occupies its slot until its exit date.
    open_until: list[pd.Timestamp] = []
    taken = []
    for _, row in candidates.iterrows():
        open_until = [d for d in open_until if d >= row["date"]]
        if len(open_until) < max_positions:
            taken.append(row)
            open_until.append(row["exit_date"])
    trades = pd.DataFrame(taken)

    if trades.empty:
        return BacktestResult(trades, pd.Series(dtype=float), {"n_trades": 0})

    trades["net_return"] = trades["trade_return"] - cost

    # Daily equity: each trade contributes 1/max_positions of capital.
    weight = 1.0 / max_positions
    daily = (
        trades.groupby("exit_date")["net_return"].sum() * weight
    ).sort_index()
    all_days = pd.date_range(oos["date"].min(), oos["exit_date"].max(), freq="D")
    daily = daily.reindex(all_days, fill_value=0.0)
    equity = (1.0 + daily).cumprod()

    wins = trades[trades["net_return"] > 0]
    losses = trades[trades["net_return"] <= 0]
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    active = daily[daily != 0.0]
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)

    stats = {
        "n_trades": len(trades),
        "hit_rate": len(wins) / len(trades),
        "avg_win": wins["net_return"].mean() if len(wins) else 0.0,
        "avg_loss": losses["net_return"].mean() if len(losses) else 0.0,
        "expectancy": trades["net_return"].mean(),
        "profit_factor": (
            wins["net_return"].sum() / abs(losses["net_return"].sum())
            if len(losses) and losses["net_return"].sum() != 0 else np.inf
        ),
        "total_return": equity.iloc[-1] - 1.0,
        "cagr": equity.iloc[-1] ** (1.0 / years) - 1.0,
        "max_drawdown": drawdown.min(),
        "sharpe": (
            active.mean() / active.std() * np.sqrt(252)
            if len(active) > 2 and active.std() > 0 else np.nan
        ),
        "avg_hold_days": (trades["exit_date"] - trades["date"]).dt.days.mean(),
        "round_trip_cost": cost,
    }
    return BacktestResult(trades.reset_index(drop=True), equity, stats)
