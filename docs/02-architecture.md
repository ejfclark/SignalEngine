# System architecture

*Audience: technically curious, no finance background needed. What the pieces
are, why they exist, and the reasoning behind the key technology choices.*

## The shape of the system

```
DATA SOURCES (all free / already-paid)     NIGHTLY INGEST (Python jobs, cron)
──────────────────────────────────────    ───────────────────────────────────
Tiingo (stocks/ETFs/news, Power key) ───►  ingest stocks / etfs / news
FRED (VIX, rates, dollar — keyless)  ───►  ingest macro
worldperatio.com (scraped P/E)       ───►  ingest markets
Binance→Bybit→OKX (ccxt, no keys)    ───►  ingest crypto / funding
FMP (fundamentals, optional key)     ───►  ingest fundamentals
Anthropic API (news → events)        ───►  extraction inside ingest news
                                                      │
                                                      ▼
                                       THE LAKE  data/parquet/*.parquet
                                       (plain files; the single source of truth)
                                                      │
                                                      ▼
                                       ENGINE (nightly, 07:00 UTC)
                                       features → labels → 4 models → signals
                                                      │
                             ┌────────────────────────┼──────────────────────┐
                             ▼                        ▼                      ▼
                      artifacts/*.csv          paper ledger            bench records
                      (today's signals)     (live vs backtest)     (why every choice)
                             │
                             ▼
                      PORTAL (Flask, container) ◄── Caddy (TLS) ◄── signals.clevrur.com
```

Deployment: one Docker image on the shared IONOS VPS, run two ways — a
long-lived **portal** container, and throwaway **engine** containers launched
by cron for the nightly batch work. The lake and artifacts live on the host,
bind-mounted in.

## The elements, simply

| Element | What it is | Why it exists |
|---|---|---|
| **Ingest jobs** | Small Python collectors, one per data source | Each source fails in its own way; isolation means one broken feed never spoils the night. Every job is *idempotent* (safe to re-run) and *resumable* (picks up where it stopped). |
| **The lake** | A folder of Parquet files | The entire data estate. Replaces the old Azure SQL database. |
| **Feature pipeline** | Turns raw prices into ~60 numeric "descriptors" per asset per day | Models can't read charts; features are how a chart becomes numbers (trend, momentum, volatility, crowd positioning, market weather). |
| **Labeler** | Marks every historical day: "would this trade have worked?" | Supervised learning needs an answer key. Ours encodes the actual trade rules (stop, target, time limit) — see the financial doc. |
| **Models** | Four LightGBM classifiers (one per book) | Each nightly retrain sees all history; scoring today's data yields each signal's probability. |
| **Backtest / bench** | Simulates only *out-of-sample* predictions, with costs | The measuring stick. Every proposed change is measured here and kept only if it wins. |
| **Ledger** | Records live signals as paper trades, tracks real outcomes | The bridge between simulation and money. |
| **Portal** | Password-protected dashboard | Where the human reads the shortlist and the live-vs-promised scoreboard. |

## Key technology choices, and why

**Parquet files instead of a database.** The workload is "write once nightly,
scan everything for training" — columnar files are dramatically faster for
that, cost nothing, back up with a file copy, and can't be broken by a bad
migration. The old system's Azure SQL was its single point of failure and its
largest bill. (Postgres exists on the server for future portal needs; nothing
requires it.)

**LightGBM instead of deep learning.** For tabular financial features,
gradient-boosted trees remain state of the art — faster to train (the whole
nightly retrain is minutes), robust to missing data (crypto has no P/E and
that's fine), and interpretable (we can rank which features carry the
signal). Deep models were considered and deliberately skipped at this data
scale.

**Retrain nightly from scratch.** Training costs ~a minute per book, so
there is no model-versioning problem, no staleness, no drift management —
every morning's model has seen everything up to the edge of what's fair.

**Free/committed data sources, with fallbacks.** Yahoo blocks datacenter
IPs, so the server uses Tiingo (paid Power tier) with the rate-limit
machinery kept armed: if the subscription ever lapses, one config line drops
to a smaller universe and the hourly retry crons absorb the slower feed.
Crypto data comes from exchange public APIs (no keys); macro from FRED
(keyless CSV); market P/E is scraped with loud failure modes.

**Docker on the shared VPS, behind the existing Caddy.** The server already
runs production apps in containers behind one reverse proxy that owns
80/443. SignalEngine joined that pattern rather than fighting it: its portal
listens only on localhost, Caddy routes the domain to it and auto-manages the
TLS certificate. Batch jobs run as `docker compose run --rm` from plain cron —
visible in one file (`/etc/cron.d/signalengine`), logged to one place.

**Secrets as a mounted file, not compose env vars** — docker compose
interpolates `$` inside env values, which silently corrupts password hashes;
the app parses its own env file instead. (Learned the hard way.)

**Git + gated experiments.** Every change is a commit; every performance
claim traces to a JSON record in `artifacts/bench/` written by the run that
measured it. The repo lives at github.com/ejfclark/SignalEngine (private).

**Config over code for evidence-bearing numbers.** Thresholds, barriers,
universes, and portfolio rules live in `config.toml`, each annotated with the
experiment that chose it — per-asset and per-book overrides included. Ticker
lists are plain text files under `universe/`; edit and the next ingest picks
them up.

## Operational rhythm (UTC)

| When | What |
|---|---|
| 22:15 (+hourly retries to 06:15) | Collect stock + ETF prices (retries are no-ops when the first pass completes) |
| 22:30 | Macro, market P/E, crypto, funding, fundamentals, news + extraction |
| 07:00 | Retrain all four books → publish signals → ledger update/record |
| Anytime | Portal reads the latest artifacts on every page load |

A missed night self-heals: every collector resumes from its last stored bar.
