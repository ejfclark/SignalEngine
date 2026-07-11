# SignalEngine

Swing-trading ML prediction engine over the EdStock dataset.

> **Docs**: plain-English overview, architecture & tech choices, component
> reference, and the financial method — in [docs/](docs/README.md). Replaces the 2024
BulkAnalysis project (indicator snapshots + rules + a leaky RandomForest) with a
modern pipeline:

```
prices ──> features ──> triple-barrier labels ──> LightGBM ──> OOS backtest ──> signals
 (SQL         (vectorized indicators,   (ATR-scaled     (purged walk-     (fees +      (prob, stop,
  or           cross-sectional ranks,    stop/target/     forward CV)      slippage)     target)
  Parquet)     market regime)            timeout)
```

Design principles the old system lacked:

- **Labels look forward** — each row is labeled by whether a +3×ATR target is hit
  before a −1.5×ATR stop within 10 trading days, entered at the *next* bar. The
  stop/target the model trains on is the stop/target the signal report emits.
- **Validation is walk-forward with purging** — train always strictly precedes
  test, with a horizon-sized gap so overlapping labels can't leak.
- **The backtest only ever sees out-of-sample predictions**, and charges
  fees + slippage. If expectancy doesn't clear costs, the model doesn't go live.
- **The database is optional** — `export-parquet` snapshots all tables to a local
  Parquet lake; everything downstream runs identically from either source
  (`[data].source` in config.toml).

## Quickstart

```bash
pip install -e .                      # or: uv pip install -e .
cp .env.example .env                  # EDSTOCK_CONN only needed for the SQL bridge jobs

# Data collection (the lake in data/parquet is the source of truth):
signalengine ingest stocks --backfill   # 5y adjusted OHLCV (Tiingo if key set, else Yahoo)
signalengine ingest etfs --backfill     # sector ETFs + SPY
signalengine ingest macro --backfill    # VIX, DXY
signalengine ingest crypto --backfill   # 5y spot OHLCV via exchange APIs (no key)
signalengine ingest funding --backfill  # perp funding history (no key)
signalengine ingest daily               # nightly incremental of all of the above

# Engine:
signalengine train --asset stock      # features + labels + walk-forward training
signalengine backtest --asset stock   # cost-aware backtest of OOS predictions
signalengine signals --asset stock    # today's ranked signals with stop/target
```

`--asset crypto` runs the same pipeline over the crypto lake. All knobs
(barriers, horizon, folds, costs, thresholds) live in `config.toml`; ticker
lists live in `universe/*.txt` (edit, re-ingest, retrain). See `deploy/` for
running the whole thing on a Linux VPS with systemd timers.

## Reading the output

`train` prints per-fold **AUC** (0.5 = no edge) and **precision@threshold** vs the
**base rate** — the model is useful only when flagged rows hit targets clearly more
often than the average row. `backtest` prints expectancy per trade after costs,
profit factor, drawdown and Sharpe, and writes every simulated trade to
`artifacts/<asset>_trades.csv` for inspection. First results on the current data
(~22 months, 414 tech stocks): AUC 0.55–0.61 per fold, stock expectancy ≈ +1% per
trade after 40bps round-trip costs; crypto shows **no edge** with current features
(see wishlist — funding/OI data is the missing ingredient there).

Treat these as a baseline, not a verdict: one universe, one market regime,
no hyperparameter tuning, and 2024 entries are at signal-day close because the
feed has no opens (see Data quality).

## Data quality notes (found while building)

- `DailyPrice.Open` is 0 for all of 2024 and ~45% of 2025 rows. The engine
  treats those as missing and enters at signal-day close instead of next-day
  open for affected rows (slightly optimistic: ignores the overnight gap).
- `Instrument` contains duplicate tickers (ADI, ARM, AVGO, INTC, MU, NVDA, …).
  The pipeline dedupes, but the table wants a primary key.
- Stock ingestion stopped 2025-11-10 and crypto 2026-05-05 (checked 2026-07-07);
  market P/E + bond yields are current. Restart StockIngest's price jobs before
  relying on live signals.
- `CurrencyPrice` holds ~4 intraday snapshots per day; the engine collapses to
  one daily bar (last per date).

## Where more data would add value

Ranked by expected impact on this specific engine:

1. **Backfilled daily history (5+ years, survivorship-aware)** — 22 months and a
   single regime is the binding constraint on everything. One backfill from
   Tiingo/Polygon/EODHD would do more than any modelling change.
2. **Crypto funding rates + open interest** (exchange APIs / CCXT) — the crypto
   model currently has no sentiment/positioning signal, and it shows: expectancy
   is negative. Extreme funding is the classic crypto mean-reversion input.
3. **VIX level + term structure** (free: FRED/Yahoo) — the best single risk-on/off
   regime feature for the stock book; slots in next to `us10y` in `features/market.py`.
4. **Sector ETF daily prices** (XLK, XLE, … from Yahoo) — enables relative
   strength vs own sector, a classic swing screen; the sector mapping already exists.
5. **Earnings surprise history** (FMP has it) — `days_to_earnings` is already a
   top-10 feature; surprise direction/magnitude would sharpen it.
6. **BTC dominance + DXY** — cheap regime features for the crypto model.
7. **Short interest / borrow rates** (stocks) — squeeze setups are a distinct,
   labelable regime.
8. **News/filings sentiment via LLM extraction** — structured features
   (guidance cut, listing, hack, …) from headlines; the modern replacement for
   the old ChatGPT-automation idea. Valuable but build after 1–4.

Each of these lands as: a new ingest job → a new Parquet file → a join + a few
columns in `features/market.py` or `features/pipeline.py` → retrain and compare
fold metrics. The engine is deliberately shaped so that's the whole loop.

## Layout

```
signalengine/
  config.py           config.toml + .env loading
  data/               SQL + Parquet sources (one interface), export utility
  ingest/             lake.py (idempotent upsert), providers.py (Yahoo/Tiingo),
                      stocks.py, crypto.py (ccxt), fundamentals.py (FMP),
                      context.py (SQL bridge + legacy snapshot), universe.py
  features/           indicators.py (vectorized TA), market.py (regime), pipeline.py
  labels/             triple_barrier.py
  model/              splits.py (purged walk-forward), train.py (LightGBM)
  backtest/           engine.py (cost-aware, OOS-only)
  signals/            generate.py (latest-bar scoring, stale tickers excluded)
  cli.py              ingest | export-parquet | train | backtest | signals
universe/             stocks.txt, crypto.txt, etfs.txt — the master ticker lists
data/parquet/         the lake (source of truth); data/edstock_snapshot/ = archive
deploy/               systemd units + VPS setup guide
tests/                indicator correctness, label semantics, split hygiene
artifacts/            datasets, models, OOS predictions, trades, signals (gitignored)
```

## Data architecture

- **The lake is the source of truth.** Every ingest job upserts Parquet files on
  natural keys — idempotent, atomic, safe to re-run. The old EdStock export
  lives in `data/edstock_snapshot/` as a frozen archive.
- **Prices** come from Yahoo (no key) or Tiingo (free key, preferred — set
  TIINGO_API_KEY), split/dividend-adjusted, 5 years back.
- **Crypto** spot OHLCV + perp funding rates come from exchange public APIs
  (binance→bybit→okx fallback) — free, no keys.
- **Fundamentals** (EPS/PE/earnings dates): legacy history extracted from
  EdStock; ongoing snapshots via the FMP job (needs FMP_API_KEY).
- **StockIngest (C#)** keeps running only its market job (worldperatio P/E +
  bond yields → SQL); `ingest context` bridges that into the lake. Its price
  jobs are superseded and should be disabled.

## Known limitations / next steps

- Stops are assumed filled at the stop price; a gap through the stop fills worse
  in reality. Expect live results below backtest.
- Probabilities are uncalibrated across folds (fold 2 flags 6k rows, fold 0 flags
  144). Calibration (isotonic/Platt per fold) or a rank-based "top N per day"
  entry rule would stabilise trade counts.
- No hyperparameter search — deliberate until there's more history; tuning on
  22 months would mostly fit noise.
- Meta-labeling (rule-based candidate filter + ML sizing) is the natural next
  architecture step once more data lands.
