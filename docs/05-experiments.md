# 05 — Experiment ledger

*The canonical register of every gated experiment: hypothesis, verdict, and
where the evidence lives. **Read this before proposing or running any
experiment** — do not re-run rejected ideas without new evidence, and do not
compare against stale champion numbers. Append a row when an experiment
concludes; never edit past verdicts.*

Raw evidence: `artifacts/bench/<name>_<tag>.json` (+ `_oos.parquet`).
Adopted numbers are annotated where they live in `config.toml`.

## How to read a verdict

- **ADOPTED** — beat the champion on the metrics it claimed to improve without
  degrading elsewhere; now part of the production config/code.
- **REJECTED** — failed its claim; code reverted or left flag-gated OFF. The
  bench JSON stays as the record. Re-running requires a new hypothesis for why
  the result would differ (new data, new regime, fixed confound).
- **PAPER-ONLY** — promising but unproven for real money (unmodeled costs,
  small sample); runs in the paper ledger only.

## Standing methodology facts (learned the hard way)

- **LightGBM cross-run variance is large.** Same code + same data can swing
  expectancy well beyond ±0.3pp at a single threshold because the greedy
  slot-allocation backtest amplifies small prediction changes (observed
  2026-07-11: champion stock config produced +1.11%/trade in one run and
  +0.25%/trade in another). Single-run bench comparisons at one threshold are
  NOT evidence. Use `bench-variants` (same predictions) where possible, or a
  multi-seed sweep; a win must show up in the seed distribution, not one roll.
- **The 22-month era numbers were regime-flattered.** Anything benched before
  the 5y backfill (2026-07-09) is not comparable to current numbers.
- **Champion stock-long honest distribution (2026-07-11, 5 seeds, thr 0.65):**
  expectancy +0.32%±0.27/trade, Sharpe 0.49±0.41, AUC 0.5315±0.0016
  (`artifacts/bench/s1_seed_sweep.json`, `s1-base-seed*` arms). The adopted B1
  record (+1.11%/trade, Sharpe 1.59) was a favorable single draw. Treat the
  distribution as the champion baseline; the paper ledger is the live gate.
- **Never bench against the lockbox** (`[cv] lockbox_start`); `lockbox-eval`
  is one-shot per frozen system.
- **News archive depth:** Tiingo news API only returns ~90 days of history at
  our tier (verified 2026-07-11 — date filters below that are ignored).
  Deeper news history requires another provider (FMP has it; key not purchased
  as of 2026-07-11).

## Ledger

| # | Date | Name (bench file) | Hypothesis | Verdict | Outcome / adopted where |
|---|---|---|---|---|---|
| 1 | 2026-07-09 | `exp1-fills` | Realistic gap fills in labels (open beyond barrier fills at open) give honest, not optimistic, labels | **ADOPTED** | Label semantics in `labels/triple_barrier.py`; both assets |
| 2 | 2026-07-09 | `exp5-compression` | Compression/breakout features add stock edge | **ADOPTED** | Stock expectancy flat→+0.3–0.5%/trade; features in `indicators.py` |
| 3 | 2026-07-09 | `barrier_sweep` + variants | Crypto vol-sizing + breadth≥0.3 entry gate | **ADOPTED** | `[backtest.crypto]`; crypto +0.85%/trade, Sharpe 1.32 at the time |
| 4 | 2026-07-09 | variants | Top-N daily entry cap | **REJECTED** | Hurt expectancy; `top_n = 0` stays |
| 5 | 2026-07-09/10 | `shorts-v1` | Crypto short book via mirrored triple barrier | **ADOPTED** | Short AUC 0.623 > longs; ungated thr 0.70 (`[backtest.crypto-short]`). Gating shorts on breadth HURT — breakdowns are shortable in any tape |
| 6 | 2026-07-10 | `expB1-uptrend` | Stock meta-labeling: only score uptrend names near highs | **ADOPTED** | `[labels.stock] candidate_query`; stock +0.54→+1.11%/trade, Sharpe 0.6→1.59 (single-run numbers — see methodology note) |
| 7 | 2026-07-10 | `expB2-compress` | Compression-based candidate filter | **REJECTED** | Negative vs B1 |
| 8 | 2026-07-10 | `expC-calibrated` | Per-fold isotonic calibration | **REJECTED** | Collapsed the probability scale; code stays flag-gated OFF (`model.calibrate`) |
| 9 | 2026-07-10 | (labels sweep) | Crypto-long horizon 15 | **ADOPTED** | `[labels.crypto] horizon_days = 15` |
| 10 | 2026-07-10 | `shorts-v1_stock-short` | Stock short book | **PAPER-ONLY** | +0.51%/trade @0.70 but borrow/squeeze costs unmodeled |
| 11 | 2026-07-10 | `ab-core` vs `ab-expanded` | Expanded 876-ticker universe beats core 421 | **REJECTED (for now)** | Sharpe 1.17→0.66 — the S&P additions lack fundamentals history and `pe_ratio` is the top feature. NOT a falsification of diversification; re-test after fundamentals coverage (needs FMP). `stocks_model_universe = "stocks-core"` |
| 12 | 2026-07-10 | `lockbox_*` (SPENT) | One-shot honest eval on 2026-03-15+ | — | stock-long +1.71%/trade ✓, crypto-long +3.29% ✓, stock-short +11.8% on 38 trades (distrust — small sample), crypto-short −0.5% (rally regime, coherent) |
| 13 | 2026-07-11 | `s1-*-seed*` + `s1_seed_sweep.json` | Exp S1: sector as native LightGBM categorical lets the pooled model learn per-sector behavior | **REJECTED** | 5-seed sweep: exp65 base +0.32%±0.27 vs sector +0.50%±0.34 (paired diffs −0.03/+0.73/+0.56/−0.03/−0.31pp — noise), and AUC LOWER in 4/5 seeds (0.5315→0.5291). The single seed-42 run (Sharpe 0.39→1.64) was pure run-to-run noise. Sector-relative features already in the panel capture what the label offers. Code stays flag-gated OFF (`model.sector_feature`, `bench --sector`); `feature_frame()` refactor + tests kept |
| 14 | 2026-07-11 | S3.0 diagnostic (scratch script; pooled `s1-base-seed*` OOS trades) | Does the champion's edge vary by behavioral segment? | **HETEROGENEITY CONFIRMED** | (a) Edge is regime-conditional: at signal level, ALL beta segments profitable when entry-day VIX elevated (+0.71/+0.15/+0.75%, 5/5 & 4/5 seeds), ALL negative in low-VIX tape (mid/high beta −0.85 to −0.99%, 0/5 seeds). (b) The stock book is de facto all-tech: 16,914/17,157 sector-mapped signals are XLK — explains S1's null; makes universe diversification (blocked on fundamentals) the real segmentation lever. (c) Low-beta is the only unconditionally positive segment |
| 15 | 2026-07-11 | S3.3 VIX gate (same-prediction test over `s1-base-seed*_oos`) | Gate stock entries on elevated VIX | **ADOPTED** | `[backtest.stock] gate_column="vix", gate_min=18`. Same predictions, 5 seeds, thr 0.65: exp +0.32%±0.27→+0.89%±0.48, Sharpe 0.49→1.56, maxDD ~−39%→~−14%, better in 4/5 seeds; ~50 trades/yr remain. Caveat: hypothesis derived from the same OOS window it was tested on (though threshold 18 is a natural prior, not tuned — 20 and 25 also positive); the paper ledger is the true validator. Uncovered + fixed alongside: the paper ledger never applied ANY book's gate (crypto breadth gate wasn't reaching the paper book) — signals CSVs now carry regime columns and `record_signals` enforces `backtest_for(tag)` gates |

| 16 | 2026-07-12 | `s31-*-seed*` + `s31_sweep.json` | Exp S3.1a: behavioral-identity features (beta_spy_120d, corr_spy_120d, idio_vol_share, vix_sens_120d, dollar_vol_rank) | **ADOPTED** | Strongest result to date. Ungated: better at every threshold (thr .65: +0.32→+0.72%). Under production VIX≥18 gate: +0.89%±0.48→+1.59%±0.62/trade, Sharpe 1.56→2.31, paired better-or-flat 5/5 seeds (+0.84/+0.40/+1.08/+1.20/+0.00pp), unchanged trade count. Caveat: mean fold AUC −0.005 in 5/5 seeds (mid-ranking blurs, top tail sharpens — expectancy improved everywhere, so accepted with the deviation noted) |
| 17 | 2026-07-12 | same sweep, `earn`/`both` arms | Exp S3.1b: earnings-event features (earnings_reaction, days_since_earnings, eps_chg_63d — PEAD via market reaction) | **REJECTED (for now)** | Ungated looked alive (+0.55% at .65) but under the production gate it HURT (+0.89→+0.26%), and `both` diluted behavioral everywhere. Only 36% coverage (fundamentals start 2024). Code stays in indicators.py, columns out of FEATURE_COLUMNS. Retest when: fundamentals history deepens, FMP consensus-surprise data exists, or as a standalone event study. This also weakens the near-term prior for LLM earnings extraction |

## Queued / designed but not run

- **Exp S2** — per-sector models vs pooled (tech vs non-tech first). Only if S1
  signal is real; per-stock models are ruled out on sample size (~1,250
  bars/stock, few hundred candidates → 400-tree GBM memorizes).
- **Exp S3** — behavioral clusters (vol/liquidity/mcap) instead of GICS labels.
- **News value (Track A)** — BLOCKED on a deep-history news source (no FMP key
  yet). Design: ~15 stocks + 5 cryptos, 2y history, Gate 1 = event study
  (rank-IC of sentiment×materiality vs 1–5d forward returns, ~$5 of Haiku
  extraction on a sample), Gate 2 = model bench on that sub-universe (~$15).
  Full 130-ticker pilot extraction (~$61 sync / ~$31 batch) only after both
  gates pass. Tiingo raw collection continues free meanwhile; extraction of
  the live feed costs ~$0.70/day at current volume.
- **Earnings-announcement LLM extraction** — targeted slice of the news
  pipeline: extract only articles within ±3d of reconstructed earnings dates
  (schema already has earnings_beat/miss, guidance_raise/cut). Adds the WHY
  behind `earnings_reaction` (guidance direction drives drift more than the
  headline beat). Prior depends on the S3.1 earnings-arm verdict; size the
  article subset (free) before proposing spend. ~1 earnings cycle of news
  history so event-study only, not model features yet.
- **Influencer attribution in extraction schema** — add `notable_person` /
  `influencer_statement` to the extraction schema BEFORE any big extraction
  run so scored articles carry speaker attribution (Musk/Trump etc.).
  Hypothesis: attention spikes predict multi-day continuation/reversal
  (strongest prior in crypto: DOGE, TRUMP). NOTE: catching the initial pop is
  structurally impossible at daily bars with next-open entry — the target is
  the drift after, never the pop.
- **Earnings surprises features** — blocked on FMP key.
- **Expanded-universe re-test** — blocked on fundamentals coverage (FMP or
  Tiingo add-on); see #11.
