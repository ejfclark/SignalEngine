"""SignalEngine CLI.

Data collection (writes the Parquet lake — the engine's source of truth):
    signalengine ingest legacy-snapshot  One-time: archive EdStock export, keep fundamentals
    signalengine ingest stocks [--backfill]   Adjusted daily OHLCV (Tiingo if key set, else Yahoo)
    signalengine ingest etfs   [--backfill]   Sector ETFs + SPY for relative strength
    signalengine ingest macro  [--backfill]   VIX, DXY
    signalengine ingest crypto [--backfill]   Spot OHLCV via exchange APIs (ccxt)
    signalengine ingest funding [--backfill]  Perp funding rates + open interest
    signalengine ingest fundamentals          FMP quote snapshot (needs FMP_API_KEY)
    signalengine ingest context               Pull MarketPE/SectorPE/BondYield from SQL
    signalengine ingest daily                 All incremental jobs, in order

Engine:
    signalengine export-parquet          Snapshot the SQL tables to the local Parquet lake
    signalengine train   --asset stock   Build features+labels, walk-forward train, save artifacts
    signalengine backtest --asset stock  Cost-aware backtest over the out-of-sample predictions
    signalengine signals --asset stock   Score the latest bar -> ranked signals with stop/target

The dataset is cached in artifacts/<asset>_dataset.parquet; `train --rebuild`
refreshes it from the data source.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import Config, load_config
from .data import get_source
from .features import build_features
from .labels import apply_triple_barrier

ASSETS = ("stock", "crypto")


def _tag(asset: str, direction: str) -> str:
    """Artifact naming: 'crypto' for longs (back-compat), 'crypto-short' for shorts."""
    return asset if direction == "long" else f"{asset}-{direction}"


def _dataset_path(cfg: Config, tag: str) -> Path:
    return cfg.artifacts_dir / f"{tag}_dataset.parquet"


def _merge_fundamentals(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Attach point-in-time fundamentals to price rows, forward-filled a short
    way so a weekly snapshot covers the trading days between snapshots."""
    if fundamentals.empty:
        return prices
    cols = ["eps", "pe", "mcap", "shares_outstanding", "avg_volume", "earnings_date"]
    cols = [c for c in cols if c in fundamentals.columns]
    f = fundamentals[["ticker", "date", *cols]].copy()
    f["date"] = pd.to_datetime(f["date"])
    merged = prices.merge(f, on=["ticker", "date"], how="left")
    merged = merged.sort_values(["ticker", "date"])
    merged[cols] = merged.groupby("ticker")[cols].ffill(limit=10)
    return merged.reset_index(drop=True)


def build_dataset(cfg: Config, asset: str, direction: str = "long",
                  candidate_query: str | None = None) -> pd.DataFrame:
    """candidate_query is the meta-labeling hook: a pandas query evaluated on
    the labeled feature panel. Rows failing it are dropped BEFORE training, so
    the model learns 'given this setup, does the trade work?' instead of
    scoring every bar of every ticker every day."""
    source = get_source(cfg)
    if asset == "stock":
        prices = _merge_fundamentals(source.load_stock_prices(), source.load_stock_fundamentals())
    else:
        prices = source.load_crypto_prices()
    print(f"{asset} ({direction}): {len(prices):,} price rows, {prices['ticker'].nunique()} tickers, "
          f"{prices['date'].min().date()} -> {prices['date'].max().date()}")

    features = build_features(
        prices,
        instruments=source.load_instruments() if asset == "stock" else None,
        bond_yields=source.load_bond_yields(),
        market_pe=source.load_market_pe(),
        sector_pe=source.load_sector_pe(),
        macro=source.load_macro() if asset == "stock" else None,
        etf_prices=source.load_etf_prices() if asset == "stock" else None,
        derivatives=source.load_crypto_derivatives() if asset == "crypto" else None,
        with_market_context=(asset == "stock"),
    )
    labeled = apply_triple_barrier(
        features,
        horizon=cfg.labels.horizon_days,
        target_mult=cfg.labels.target_atr_mult,
        stop_mult=cfg.labels.stop_atr_mult,
        direction=direction,
    )
    if candidate_query:
        before = len(labeled)
        labeled = labeled.query(candidate_query).reset_index(drop=True)
        print(f"{asset}: candidate filter '{candidate_query}' kept "
              f"{len(labeled):,}/{before:,} rows")
    n = labeled["label"].notna().sum()
    print(f"{asset}: {n:,} labeled rows, base rate {labeled['label'].mean():.3f}")
    return labeled


def _load_dataset(cfg: Config, asset: str, rebuild: bool, direction: str = "long") -> pd.DataFrame:
    path = _dataset_path(cfg, _tag(asset, direction))
    if path.is_file() and not rebuild:
        print(f"Using cached dataset {path} (pass --rebuild to refresh)")
        return pd.read_parquet(path)
    labeled = build_dataset(cfg, asset, direction)
    labeled.to_parquet(path, index=False)
    return labeled


def cmd_ingest(cfg: Config, args) -> None:
    from .ingest.universe import load_universe

    lake = cfg.parquet_dir
    job = args.job
    backfill = getattr(args, "backfill", False)

    def stocks():
        from .ingest.stocks import ingest_prices
        print("stocks:")
        ingest_prices(lake / "stock_prices.parquet", load_universe(cfg.root, "stocks"), backfill)

    def etfs():
        from .ingest.stocks import ingest_prices
        print("etfs:")
        ingest_prices(lake / "etf_prices.parquet", load_universe(cfg.root, "etfs"), backfill)

    def macro():
        from .ingest.markets import ingest_macro
        print("macro (FRED):")
        ingest_macro(lake / "macro.parquet", backfill)

    def markets():
        from .ingest.markets import ingest_market_pe
        print("markets (worldperatio):")
        ingest_market_pe(lake)

    def crypto():
        from .ingest.crypto import ingest_crypto_prices
        print("crypto:")
        ingest_crypto_prices(lake / "crypto_prices.parquet", load_universe(cfg.root, "crypto"), backfill)

    def funding():
        from .ingest.crypto import ingest_crypto_derivatives
        print("funding:")
        ingest_crypto_derivatives(lake / "crypto_derivatives.parquet", load_universe(cfg.root, "crypto"), backfill)

    def fundamentals():
        from .ingest.fundamentals import ingest_fundamentals
        print("fundamentals:")
        ingest_fundamentals(lake / "stock_fundamentals.parquet", load_universe(cfg.root, "stocks"))

    def legacy_snapshot():
        from .ingest.context import legacy_snapshot as snap
        print("legacy-snapshot:")
        snap(cfg)

    jobs = {"stocks": stocks, "etfs": etfs, "macro": macro, "markets": markets,
            "crypto": crypto, "funding": funding, "fundamentals": fundamentals,
            "legacy-snapshot": legacy_snapshot}
    if job == "daily":
        failures = []
        for name in ("stocks", "etfs", "macro", "markets", "crypto", "funding", "fundamentals"):
            try:
                jobs[name]()
            except Exception as e:  # one source down must not kill the nightly run
                failures.append(name)
                print(f"  ! {name} failed: {e}")
        if failures:
            sys.exit(f"daily ingest finished with failures: {', '.join(failures)}")
    else:
        jobs[job]()


def cmd_export_parquet(cfg: Config, _args) -> None:
    from .data.export import export_to_parquet
    from .data.sql_source import SqlSource

    counts = export_to_parquet(SqlSource(cfg.connection_string), cfg.parquet_dir)
    for name, n in counts.items():
        print(f"  {name:15s} {n:>10,} rows")
    print(f"Parquet lake written to {cfg.parquet_dir}")


def cmd_train(cfg: Config, args) -> None:
    from .model.train import save_artifacts, train_walk_forward

    labeled = _load_dataset(cfg, args.asset, args.rebuild, args.direction)
    result = train_walk_forward(labeled, cfg)

    print("\nWalk-forward folds (all metrics out-of-sample):")
    print(result.fold_metrics.to_string(index=False))
    print("\nTop features:")
    print(result.importance.head(15).to_string(index=False))

    save_artifacts(result, cfg.artifacts_dir, _tag(args.asset, args.direction))
    print(f"\nSaved model + OOS predictions to {cfg.artifacts_dir}")


def cmd_backtest(cfg: Config, args) -> None:
    from .backtest import run_backtest
    from .model.train import PREDICTIONS_FILE

    path = cfg.artifacts_dir / f"{_tag(args.asset, args.direction)}_{PREDICTIONS_FILE}"
    if not path.is_file():
        sys.exit(f"No OOS predictions at {path} — run `signalengine train` for that asset/direction first")
    oos = pd.read_parquet(path)

    bt = cfg.backtest_for(args.asset)
    threshold = args.threshold if args.threshold is not None else bt.probability_threshold
    result = run_backtest(
        oos, threshold, bt.fee_bps, bt.slippage_bps, bt.max_positions,
        sizing=bt.sizing, risk_pct=bt.risk_pct, top_n=bt.top_n or None,
        gate_column=bt.gate_column or None, gate_min=bt.gate_min,
    )

    print(f"\nBacktest (OOS only, threshold={threshold}, "
          f"costs={bt.fee_bps + bt.slippage_bps:.0f}bps/side, max {bt.max_positions} positions):")
    for key, value in result.stats.items():
        print(f"  {key:16s} {value:.4f}" if isinstance(value, float) else f"  {key:16s} {value}")

    trades_path = cfg.artifacts_dir / f"{args.asset}_trades.csv"
    result.trades.to_csv(trades_path, index=False)
    print(f"\nTrades written to {trades_path}")


def cmd_ledger(cfg: Config, args) -> None:
    from .ledger import record_signals, report, update_positions

    if args.action == "update":
        result = update_positions(cfg)
        print(f"ledger update: {result['filled']} entries filled, {result['closed']} closed")
    elif args.action == "record":
        added = record_signals(cfg)
        print(f"ledger record: {added} new virtual positions")
    else:
        report(cfg)


def cmd_bench(cfg: Config, args) -> None:
    from .bench import run_bench

    if args.calibrate:
        cfg.model.calibrate = True
    payload = run_bench(cfg, args.asset, args.name, args.direction, args.query)
    print(f"\nbench '{args.name}' ({args.asset}): mean AUC {payload['mean_auc']:.4f}, "
          f"min AUC {payload['min_auc']:.4f}")
    for thr, stats in payload["backtests"].items():
        print(f"  thr {thr}: {stats.get('n_trades', 0)} trades, "
              f"expectancy {stats.get('expectancy', float('nan')):+.2%}, "
              f"sharpe {stats.get('sharpe', float('nan')):.2f}, "
              f"maxDD {stats.get('max_drawdown', float('nan')):+.1%}")


def cmd_bench_compare(cfg: Config, args) -> None:
    from .bench import compare

    compare(cfg, args.asset, args.base, args.experiment)


def cmd_bench_variants(cfg: Config, args) -> None:
    from .bench import run_variants

    run_variants(cfg, args.asset)


def cmd_signals(cfg: Config, args) -> None:
    from .model.train import load_model
    from .signals import generate_signals

    tag = _tag(args.asset, args.direction)
    labeled = _load_dataset(cfg, args.asset, args.rebuild, args.direction)
    model = load_model(cfg.artifacts_dir, tag)
    asof = pd.Timestamp(args.asof) if args.asof else None

    signals = generate_signals(labeled, model, cfg, asof, direction=args.direction)
    threshold = args.threshold if args.threshold is not None else cfg.signal_threshold
    flagged = signals[signals["probability"] >= threshold]

    date = signals["date"].max().date() if len(signals) else "n/a"
    print(f"\nSignals as of {date} (probability >= {threshold}):")
    if flagged.empty:
        print("  none — market conditions don't favour the setup right now")
    else:
        print(flagged.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    out = cfg.artifacts_dir / f"{tag}_signals.csv"
    signals.to_csv(out, index=False)
    print(f"\nFull ranked list written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="signalengine", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="path to config.toml (default: search upward from cwd)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("export-parquet", help="snapshot SQL tables to the Parquet lake")

    p_ingest = sub.add_parser("ingest", help="collect data into the Parquet lake")
    p_ingest.add_argument("job", choices=["stocks", "etfs", "macro", "markets", "crypto",
                                          "funding", "fundamentals", "legacy-snapshot", "daily"])
    p_ingest.add_argument("--backfill", action="store_true", help="full history, not incremental")

    for name in ("train", "backtest", "signals"):
        p = sub.add_parser(name)
        p.add_argument("--asset", choices=ASSETS, default="stock")
        p.add_argument("--direction", choices=["long", "short"], default="long")
        p.add_argument("--rebuild", action="store_true", help="rebuild dataset from the data source")
        p.add_argument("--threshold", type=float, default=None)
        if name == "signals":
            p.add_argument("--asof", help="score as of this date (yyyy-mm-dd), default latest")

    p_bench = sub.add_parser("bench", help="train+backtest, write metrics for comparison")
    p_bench.add_argument("--name", required=True)
    p_bench.add_argument("--asset", choices=ASSETS, default="stock")
    p_bench.add_argument("--direction", choices=["long", "short"], default="long")
    p_bench.add_argument("--query", default=None,
                         help="meta-labeling candidate filter (pandas query on the panel)")
    p_bench.add_argument("--calibrate", action="store_true",
                         help="per-fold isotonic calibration (experiment flag)")
    p_cmp = sub.add_parser("bench-compare", help="compare two bench runs")
    p_cmp.add_argument("base")
    p_cmp.add_argument("experiment")
    p_cmp.add_argument("--asset", choices=ASSETS, default="stock")
    p_var = sub.add_parser("bench-variants", help="portfolio variants over saved OOS predictions")
    p_var.add_argument("--asset", choices=ASSETS, default="stock")
    p_led = sub.add_parser("ledger", help="paper-trade ledger: live signals vs backtest")
    p_led.add_argument("action", choices=["update", "record", "report"])

    args = parser.parse_args()
    cfg = load_config(args.config)

    {"export-parquet": cmd_export_parquet,
     "ingest": cmd_ingest,
     "train": cmd_train,
     "backtest": cmd_backtest,
     "signals": cmd_signals,
     "bench": cmd_bench,
     "bench-compare": cmd_bench_compare,
     "bench-variants": cmd_bench_variants,
     "ledger": cmd_ledger}[args.command](cfg, args)


if __name__ == "__main__":
    main()
