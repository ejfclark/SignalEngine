# Core components — technical reference

*Audience: developers. Assumes the architecture doc. Paths are repo-relative;
the package is `signalengine/`.*

## Data layer (`signalengine/data/`, `signalengine/ingest/`)

**Canonical frames** are defined in `data/source.py`: snake_case columns,
`date` as datetime64, numerics float64. Two interchangeable sources implement
the same protocol — `ParquetSource` (production; the lake) and `SqlSource`
(legacy EdStock reader, kept for archaeology). `normalize_prices` enforces
positive closes, null opens (the legacy feed's `Open=0` era), per-(ticker,
date) dedupe.

**The lake** (`data/parquet/`): `stock_prices`, `etf_prices`, `crypto_prices`
(OHLCV, adjusted), `crypto_derivatives` (daily-mean perp funding + spot OI),
`macro` (VIX/DXY/2y/10y from FRED), `market_pe`/`sector_pe` (worldperatio),
`stock_fundamentals` (EPS/PE/mcap/earnings dates: legacy history + FMP
snapshots), `instruments` (ticker→sector, GICS→SPDR mapped), `news_raw` /
`news_events`.

**Ingest jobs** (`ingest/`): each is idempotent via `lake.upsert(path, df,
keys)` — read-concat-dedupe-atomic-replace. Price jobs checkpoint every 25
tickers and resume from each ticker's last stored bar, ordered
**most-behind-first** so a rate-capped pass spends quota on backfill, not
refreshes. Failures are classified: `RateLimited` trips a circuit breaker
(6 consecutive = provider capped, stop the pass); per-ticker 404s just log.
Providers: Tiingo (preferred, key) with Yahoo fallback (dead from datacenter
IPs — local dev only); ccxt with binance→bybit→okx failover; FRED via `curl`
subprocess (their WAF tarpits python-requests' TLS fingerprint); worldperatio
parsers ported from the C# originals (balanced-bracket JS array extraction,
loud `FormatException`-style failures).

**News** (`ingest/news.py`): raw collection is unmetered (Tiingo);
LLM extraction (Haiku, fixed JSON schema: event_type/sentiment/materiality,
cached forever by article_id) is metered and restricted to
`universe/news-pilot.txt` while that file exists.

## Features (`signalengine/features/`)

`indicators.py` — vectorized, pure pandas/numpy (no TA-Lib binary), computed
per ticker: multi-horizon returns/volatility, ATR, Wilder RSI (SMA-seeded,
TA-Lib-matching), MACD (price-normalized), ADX/DI, stochastics, EMA-ribbon
state, MA distances, 52-week position, volume z-score, and the compression/
breakout block (ATR ratio 10/60, Bollinger bandwidth percentile, range
contraction, inside days, distance/days-since 20/60d highs, higher-lows
count, volume dry-up). All are look-back only; `test_*_no_lookahead` truncation
tests enforce it.

`market.py` + `pipeline.py` — cross-sectional daily percentile ranks
(momentum/vol/volume vs the whole universe), breadth and equal-weight
universe return, macro context (VIX + change, 10y, 2s10s curve, DXY),
market/sector P/E joins, relative strength vs sector ETF and SPY, crypto
funding features and BTC-as-market-factor. `FEATURE_COLUMNS` is the single
list the models consume; NaN is legal everywhere (LightGBM handles it), so
asset-class-inapplicable features simply stay NaN.

## Labels (`signalengine/labels/triple_barrier.py`)

Per row: entry = next bar's open (fallback signal close where the feed has no
open); target = entry ± `target_atr_mult`·ATR14; stop = entry ∓
`stop_atr_mult`·ATR14; scan `horizon_days` bars. **Realistic fills**: a bar
*opening* beyond a barrier fills at that open (gaps hurt stops, help
targets); intraday touches fill at the barrier; both-hit days resolve
conservatively to the stop; timeouts exit at the last close; windows
truncated by end-of-data stay unlabeled. `direction="short"` mirrors
everything via a sign flip; `trade_return` is signed in the trade's favor.
Label = 1 iff target first. Per-book barrier config via
`cfg.labels_for(tag)` (`[labels.crypto]` h15, `[labels.stock]`
candidate_query, …).

**Meta-labeling**: `[labels.stock] candidate_query` is a pandas query applied
to the labeled panel before training — the model learns "given this setup,
does it work?" instead of scoring every bar.

## Models (`signalengine/model/`)

`splits.py` — purged walk-forward: date axis into `n_folds+1` contiguous
blocks; fold *i* trains on blocks ≤ *i* (minus `purge_days` ≥ horizon, so
overlapping labels can't leak) and tests on block *i+1*.

`train.py` — per fold: LightGBM binary classifier (400 trees, lr .05,
63 leaves, subsample/colsample .8, auto scale_pos_weight, seed 42), OOS
probabilities collected with trade metadata + regime columns; then a final
model on all rows for live scoring. Isotonic calibration exists behind
`model.calibrate` (benched: rejected — collapses the probability scale).
Beware run-to-run nondeterminism (~±0.3pp expectancy): same-prediction
comparisons are strong, cross-run ones need margin.

## Backtest (`signalengine/backtest/engine.py`)

Consumes **only** OOS predictions. Threshold filter → optional regime gate
(`gate_column`/`gate_min`) → optional top-N/day → greedy slot allocation
(`max_positions`, position occupies its slot to exit). Sizing: `equal` or
`vol` (weight = `risk_pct` / stop-distance, capped at 2 equal-slots).
Round-trip costs charged per trade. Stats: expectancy, hit rate, PF, Sharpe
(daily), maxDD, CAGR. Per-book settings from `cfg.backtest_for(tag)`.

## Experiment discipline (`bench.py`, `lockbox.py`)

`bench --name X [--asset --direction --query --universe --calibrate]` —
rebuild dataset, walk-forward train, threshold-grid backtest, write
`artifacts/bench/X_tag.json` + OOS parquet. `bench-compare` / `bench-variants`
(portfolio-rule grids over saved predictions, no retrain). **Lockbox**:
`[cv] lockbox_start` — bench paths drop every row whose exit could touch the
sealed period; `lockbox-eval` runs the frozen system once on it (train
before, trade through, per-book production rules) and records the verdict.
Rule of the house: one variable per commit; keep only what beats the champion
on the metrics it claims to improve; rejected code stays in history with its
bench record.

## Ledger (`signalengine/ledger.py`)

State machine per virtual trade: `pending` (awaiting entry bar) → `open` →
`target|stop|timeout` (label-faithful fills incl. gap handling) or `expired`.
`BOOKS` defines per-book action thresholds — the same numbers the portal
badges use. Nightly `ledger update` then `ledger record` after signals; state
in `data/ledger.parquet`; `ledger report` prints live-vs-backtest per book.

## Portal (`signalengine/portal.py`)

Single-file Flask: scrypt-hashed single user (env-file creds, timing-safe
compare), session cookie, tables for the four books (BUY/WATCH/SHORT badges,
stop/target/R:R per row, stale-data warning tiles) + the paper-book
live-vs-expected table. Served by gunicorn in the portal container on
127.0.0.1:8050; Caddy terminates TLS for signals.clevrur.com.

## Testing & ops

`tests/` (~44): indicator correctness vs reference values, no-lookahead
truncation, triple-barrier semantics both directions incl. gap fills, ledger
state machine, purged-split hygiene, lockbox boundary, lake upsert
idempotency. Run `python -m pytest` before every commit (the habit that
caught three shipped bugs).

Ops quick-reference: `/etc/cron.d/signalengine` (schedule) ·
`/var/log/signalengine.log` (nightly output) · `/srv/signalengine/`
(app/, data/, artifacts/, universe/, signalengine.env chmod 600) ·
`docker compose build portal && docker compose up -d portal` (deploy) ·
`docker compose run --rm engine python -m signalengine.cli <cmd>` (anything
else). License fallback: `[data] stocks_universe = "stocks-core"`.
