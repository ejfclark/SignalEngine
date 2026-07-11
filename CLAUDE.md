# CLAUDE.md — SignalEngine

Nightly ML swing-trading signal engine: four books (stock/stock-short/
crypto/crypto-short), triple-barrier labels, LightGBM, purged walk-forward
validation, cost-aware backtests, paper-trade ledger, Flask portal. Full
context in `docs/` (01 overview → 04 financial method). Performance claims
live in `artifacts/bench/*.json`; adopted numbers are annotated where they
sit in `config.toml`.

## The prime directive

**Every improvement is a gated experiment.** Build → `bench` → compare vs
champion → keep only if it wins on the metrics it claims to improve →
otherwise revert (the bench JSON stays as the record). One variable per
commit. ~Half of everything tried so far was correctly rejected — that is the
system working. Never present backtest numbers without costs, and never let
any experiment touch the lockbox (below).

**Before proposing or running ANY experiment, read `docs/05-experiments.md`**
— the ledger of every experiment's hypothesis, verdict, and evidence. Don't
re-run rejected ideas without new evidence; don't trust single-run bench
comparisons (LightGBM cross-run variance at a single threshold exceeds most
claimed wins — use multi-seed sweeps or same-prediction `bench-variants`).
Append your result to the ledger when an experiment concludes.

## Commands

```
python -m pytest                                  # ~44 tests; run before EVERY commit
python -m signalengine.cli ingest <job>|daily [--backfill]   # jobs: stocks etfs macro markets crypto funding fundamentals news
python -m signalengine.cli train|backtest|signals --asset stock|crypto [--direction short] [--rebuild]
python -m signalengine.cli bench --name X --asset A [--direction D --query Q --universe U --calibrate]
python -m signalengine.cli bench-compare A B / bench-variants --asset A
python -m signalengine.cli ledger update|record|report
python -m signalengine.cli lockbox-eval --asset A --direction D   # ONE SHOT per frozen system
```

## Invariants — do not violate

- **Lockbox** (`[cv] lockbox_start` in config.toml): bench paths auto-exclude
  it; never train/tune/select against it. `lockbox-eval` spends it once, then
  the date moves forward.
- **No lookahead**: features are look-back only (truncation tests enforce);
  labels enter at the *next* bar; news joins must use published < close.
- **OOS only**: backtests consume walk-forward test-fold predictions, never
  in-sample scores.
- **Purge ≥ horizon**: build_dataset auto-raises `purge_days`; keep it that way.
- **Per-book config**: barriers/filters via `cfg.labels_for(tag)`, portfolio
  rules via `cfg.backtest_for(tag)` — tags: `stock`, `stock-short`, `crypto`,
  `crypto-short`. Evidence is per-book; never assume a rule transfers.
- Secrets live in `.env` (local) / `/srv/signalengine/signalengine.env`
  (server, chmod 600, mounted as `/app/.env`). **Never** pass them through
  compose env vars (compose interpolates `$` and corrupts scrypt hashes) and
  never commit them.

## Hard-won environment facts

- **Yahoo blocks datacenter IPs outright** — server-side stock prices are
  Tiingo-only (Power key). Free-tier fallback: `[data] stocks_universe =
  "stocks-core"` (hourly cron retries + RateLimited breaker absorb the drip).
- **FRED tarpits python-requests' TLS** — fetched via `curl` subprocess.
- **LightGBM run-to-run nondeterminism** swings small expectancies ±0.3pp:
  same-prediction comparisons (bench-variants) are strong; cross-run
  comparisons need margin.
- The stock **model** universe is `stocks-core` (421) while **collection**
  runs the full 876: the S&P additions lack fundamentals history and
  `pe_ratio` is the top stock feature — re-test expansion only after filling
  that gap (needs FMP key or Tiingo fundamentals add-on).
- News LLM extraction is metered: restricted to `universe/news-pilot.txt`
  while that file exists (delete it to extract everything). Raw collection is
  free and universe-wide. Needs `ANTHROPIC_API_KEY`.
- `ingest news --backfill` holds results in memory and writes once at the end
  — long archive pulls show nothing in the lake until they finish.

## Server (details in memory + deploy/README.md)

Shared PROD VPS (88.208.213.23) — live tempbase/DocuSeal apps; recon before
touching anything outside `/srv/signalengine`. Layout there: `app/` (code),
`data/` (lake + ledger), `artifacts/`, `universe/`, `signalengine.env`,
`docker-compose.yml`. Portal container on 127.0.0.1:8050 behind the shared
host-network Caddy (`/srv/tempbase/Caddyfile` → signals.clevrur.com).
Schedule in `/etc/cron.d/signalengine`; nightly log `/var/log/signalengine.log`.

Deploy = tar code (exclude .git/.env/.venv/artifacts/data), extract to
`app/`, `docker compose build portal && docker compose up -d portal`; batch
work via `docker compose run --rm engine ...` (`-d` for long jobs — and note
`nohup ... &` hangs paramiko exec channels; use `compose run -d` or
`systemd-run`).

## Current state (2026-07-11)

Crypto books are the money books (combined OOS ~+1.3%/trade, Sharpe ~1.75);
stock-long is promising post-meta-labeling (~+1.1%/trade, Sharpe ~1.6);
stock-short is paper-only (small-sample-suspicious, borrow costs unmodeled).
Phase: **paper-ledger validation** — judge each book at ~30+ closed trades on
the portal's live-vs-backtest table before any real money. Pending: news
pilot extraction + news features bench; expanded-universe re-test after
fundamentals coverage; Azure decommission (archives already dual-sited).
