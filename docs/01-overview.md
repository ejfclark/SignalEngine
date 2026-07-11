# SignalEngine — what it is, in plain English

*Audience: anyone. No trading or technology background assumed.*

## What it does

SignalEngine is an automated analyst for **swing trading** — the style of
trading where you buy (or bet against) a stock or cryptocurrency and hold it
for a few days to a few weeks, trying to catch one "swing" in its price.

Every night, while you sleep, it:

1. **Collects the day's data** — prices for ~880 stocks and ~130
   cryptocurrencies, market indicators (fear gauges, interest rates),
   valuations, crypto positioning data, and news.
2. **Re-teaches itself** — four machine-learning models retrain from scratch
   on five years of history, one for each "book": stocks-long, stocks-short,
   crypto-long, crypto-short. ("Long" = betting a price rises; "short" =
   betting it falls.)
3. **Publishes a shortlist** to a private website. Each idea comes with the
   full trade plan: the probability the model assigns to it working, the exact
   **stop-loss** (where to cut the loss if it goes wrong) and **target**
   (where to take the profit), and how many days to give it.

You read the shortlist over coffee. It never places trades — every decision
and every order remains human.

## What a signal actually means

The models are asked a very specific question about every candidate, every
day: *"If we entered this trade at tomorrow's open, would the price hit the
profit target before it hit the stop-loss, within the time limit?"* The
targets and stops are sized to each asset's own volatility — a wild crypto
coin gets wider ones than a calm blue-chip — and the target is always twice
as far away as the stop. That means the system only needs to be right about
**1 time in 3** to break even; anything above that is profit. In testing it
is right roughly 43–55% of the time, depending on the book.

## How we know it isn't fooling us

This is the heart of the project. Almost every amateur trading system looks
brilliant in hindsight and fails with real money, so SignalEngine is built
around three layers of enforced honesty:

- **Walk-forward testing** — every claimed result comes from simulating the
  past as it was actually lived: the model is repeatedly trained only on data
  *before* a period, then judged on that unseen period, across five years
  including the 2022 crash. Costs and realistic order fills are charged.
- **The lockbox** — the most recent months of history are sealed off. No
  improvement is ever allowed to learn from them. When the system is frozen,
  it gets exactly one graded exam on that sealed period.
- **The paper ledger** — every night's real signals are recorded as pretend
  trades and tracked to their real outcomes. The private website shows, side
  by side, what the testing *promised* and what live signals actually
  *delivered*. Until that comparison holds up over dozens of trades, no real
  money follows it.

Improvements follow the same rule: every idea (new data, new logic) is built,
measured against the current system on unseen data, and **kept only if it
provably helps**. About half of everything we tried was rejected — which is
the system working, not failing.

## Where it stands (July 2026)

- The **crypto books** are the strongest: in five-year simulated testing the
  combined long+short crypto book earned about **1.3% per trade after
  costs** with a strong risk-adjusted score. The **stock-long book** became
  genuinely promising after learning to only consider stocks in rising trends
  near their highs. The **stock-short book** looks impressive but on too small
  a sample to trust yet.
- Everything runs unattended on a rented server; the shortlist lives at a
  private, password-protected website.
- The current phase is **paper trading**: letting the ledger accumulate live
  evidence before any real money is committed. Expect testing numbers to be
  roughly **halved** in real life — that is normal for trading systems, and
  planning around it is part of the design.

## What it is not

- Not a get-rich-quick machine — the honest expectation is a modest edge,
  compounded patiently, with drawdowns that will test nerves.
- Not autonomous with money — it recommends; humans decide.
- Not finished — it is an evidence-generating machine that gets better (or
  honestly reports that an idea doesn't work) every week it runs.
