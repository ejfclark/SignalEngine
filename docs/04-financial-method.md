# The financial method — why this should make money

*Audience: someone who trades or wants to reason about the returns. Explains
the financial logic of every component and what the results actually mean.
All performance figures are out-of-sample simulation as of 2026-07-10; the
authoritative numbers live in `artifacts/bench/*.json` and supersede this
page.*

## The premise

Swing trading tries to capture multi-day price moves. Two empirical
regularities make a modest edge plausible at this horizon: **momentum/trend
persistence** (assets in strong trends continue more often than chance) and
**post-event drift** (prices adjust to new information over days, not
instantly). Nobody reliably predicts *prices*; the system instead estimates
the *probability that a specific, fully-defined trade plan works* — a much
narrower question, and one you can verify historically millions of times.

## Trade construction: the triple barrier

Every trade is defined before entry:

- **Entry** — next day's open after the signal (no lookahead: a signal
  computed after tonight's close is only actionable tomorrow).
- **Stop-loss** — 1.5 × ATR(14) against the position. ATR is the asset's own
  average daily range, so risk is proportionate to each asset's volatility.
- **Target** — 3.0 × ATR in favor (crypto longs get 15 trading days for it,
  other books 10, per barrier-sweep evidence).
- Whichever is hit first ends the trade; timeout exits at market.

This gives every trade the same **2:1 reward:risk geometry**, which makes
expectancy arithmetic trivial: breakeven hit rate = 1/(1+2) = **33.3%** plus
costs. The books hit 43–55% out-of-sample — the gap between that and 33% *is*
the edge. Wins average ~1.4× the size of losses; you lose more often than a
coin flip would suggest and profit anyway.

**Honest fills**: simulations charge 20bps per side (fees+slippage) and model
gaps — a stock that gaps below your stop fills at the open, not at your stop
price. Adding gap realism erased what looked like a profitable unfiltered
stock strategy; that deletion is a feature of the method, not a bug.

## What the models actually learn from

Ranked by measured importance, the signal comes from (stocks / crypto):

1. **Regime** — market breadth, BTC's own trend (crypto), VIX and yield curve
   (stocks). The single biggest lesson of training: *when* matters more than
   *what*. A mediocre setup in a strong tape beats a great setup in a bad one.
2. **Trend position** — distance from the 200-day average, % off 52-week
   high. Slow context beats fast oscillators.
3. **Valuation & events (stocks)** — P/E and days-to-next-earnings; outcomes
   near earnings are a different animal.
4. **Positioning (crypto)** — perpetual-futures funding rates: what the
   leveraged crowd is paying to hold its bias. Crypto had *no* edge until
   this arrived.
5. **Setup quality** — compression/breakout structure (the quantified content
   of classical chart patterns), relative strength vs sector.
6. Classic indicators (RSI, MACD…) rank near the bottom — they're in the mix
   but carry the least unique information. The 2024 system was built almost
   entirely from this bottom tier, which is why it didn't work.

**Meta-labeling (stocks)**: the stock model only evaluates candidates already
in an uptrend near their highs (`dist_sma200 > 0`, within 10% of the 20-day
high). Financially this is "trade pullback/breakout continuation in trending
names, and let the model rank them" — it tripled the stock book's
risk-adjusted return versus scoring everything.

## Portfolio layer — where risk is actually controlled

- **Volatility-scaled sizing**: each position risks a fixed ~1% of equity
  (position size = risk budget ÷ stop distance). A tight-stop trade gets more
  size, a wide-stop trade less; every trade contributes equal pain when
  wrong. Adopted on evidence: crypto Sharpe 0.72 → 1.04, drawdown −56% → −47%
  with identical trade selection.
- **Regime gate (crypto longs)**: stand aside when market breadth < 0.30 —
  don't buy dips in a tape where 70%+ of coins are below trend.
  Counter-intuitively, shorts run **ungated**: individual breakdowns short
  fine even in rising markets (benched, not assumed).
- **Concurrency cap**: max 10 open positions; best-probability first.
- **Two-sided books**: shorts roughly doubled the opportunity set and lifted
  the combined crypto book from Sharpe 1.32 to **1.75** — the short book
  earns most exactly when the long book is gated out.

## The results, and how much to believe them

Out-of-sample (5-year purged walk-forward, costs charged), as of 2026-07-10:

| Book | Per-trade edge | Sharpe | Notes |
|---|---|---|---|
| Crypto long | ~+0.85% | ~1.3 | Gated, vol-sized, h15 |
| Crypto short | ~+0.96% | ~1.1 | Ungated, 0.70 threshold |
| Crypto combined | ~+1.32% | ~1.75 | ~44%/yr in simulation, maxDD −38% |
| Stock long (filtered) | ~+1.1% | ~1.6 | Post meta-labeling; maxDD −22% |
| Stock short | +0.5% (sim) | — | Lockbox result implausibly good; paper-only, borrow costs unmodeled |

**Lockbox check** (system frozen, then run once on four sealed months it had
never influenced): stock long +1.71%/trade, crypto long +3.29%/trade — both
validate; crypto short lost ~0.5%/trade in what was a strong rally
(regime-coherent, judged across a full cycle by the ledger).

**Discounting rules** we apply to ourselves:
- Halve backtest Sharpe for live expectations (the standard live-decay
  observation). Combined-crypto 1.75 → live ~0.9 would still be excellent.
- Distrust beautiful small samples (see stock-short).
- The **paper ledger** is the final arbiter: every night's actual signals are
  tracked to their actual outcomes and compared to the promises above, on the
  portal. Judgment point is ~30+ closed trades per book. Real capital follows
  only books that survive that comparison — sized so the historical worst
  drawdown (−38%) would be tolerable if it repeated, because something like
  it eventually will.

## Costs of running it

Data: Tiingo Power $30/mo (prices + news archive); exchange/FRED/scrape
sources free; LLM news extraction ~$15–40 one-off pilot then ~$1/day;
server: marginal (shared VPS). Total: roughly the cost of one bad trade per
month — which is the right way to think about every component here: each
must earn more than it costs, and each has a bench record showing whether it
does.
