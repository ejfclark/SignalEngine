"""Assembles the model-ready feature panel from the canonical data frames."""

from __future__ import annotations

import pandas as pd

from .indicators import compute_indicators
from .market import build_breadth, build_etf_returns, build_market_context, build_sector_features

# Everything the model sees. LightGBM handles NaN natively, so features that
# don't exist for an asset class (e.g. pe_ratio for crypto) just stay NaN.
FEATURE_COLUMNS = [
    # per-ticker technicals
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "vol_20d", "atr_pct",
    "rsi_14", "macd_pct", "macd_hist_pct", "adx_14", "di_spread",
    "stoch_k", "stoch_kd_spread", "ema_ribbon",
    "dist_sma20", "dist_sma50", "dist_sma200",
    "volume_z", "gap_pct", "range_pct", "pct_off_high", "pos_in_range",
    # fundamentals (stocks only)
    "pe_ratio", "days_to_earnings",
    # cross-sectional: where this ticker sits vs the rest of the universe today
    "rank_ret_20d", "rank_vol_20d", "rank_volume_z",
    # universe regime
    "breadth_20d", "universe_ret_5d",
    # market context (stocks; NaN for crypto until crypto context sources exist)
    "us10y", "us10y_chg_1m", "mkt_pe", "mkt_pe_dev", "spy_pe",
    # macro regime (from ingest macro job, FRED)
    "vix", "vix_chg_5d", "dxy_ret_20d", "curve_2s10s",
    # sector context (stocks with a mapped sector)
    "sector_pe", "sector_pe_dev", "sector_rel_pe",
    # relative strength vs sector ETF and SPY (from ingest etfs job)
    "spy_ret_20d", "sector_ret_20d", "rel_sector_20d", "rel_spy_20d",
    # crypto positioning (from ingest funding job) + BTC market factor
    "funding_rate", "funding_rate_7d", "btc_ret_20d", "rel_btc_20d",
]

# Columns carried through for labeling/backtesting/reporting, not fed to the model.
META_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "atr_14"]


def build_features(
    prices: pd.DataFrame,
    instruments: pd.DataFrame | None = None,
    bond_yields: pd.DataFrame | None = None,
    market_pe: pd.DataFrame | None = None,
    sector_pe: pd.DataFrame | None = None,
    macro: pd.DataFrame | None = None,
    etf_prices: pd.DataFrame | None = None,
    derivatives: pd.DataFrame | None = None,
    with_market_context: bool = True,
) -> pd.DataFrame:
    """prices -> one row per (ticker, date) with FEATURE_COLUMNS populated."""
    parts = [compute_indicators(g) for _, g in prices.groupby("ticker", sort=False)]
    panel = pd.concat(parts, ignore_index=True)

    # Cross-sectional daily percentile ranks (0..1). Momentum rank in particular
    # is often more predictive than the absolute value.
    by_date = panel.groupby("date")
    panel["rank_ret_20d"] = by_date["ret_20d"].rank(pct=True)
    panel["rank_vol_20d"] = by_date["vol_20d"].rank(pct=True)
    panel["rank_volume_z"] = by_date["volume_z"].rank(pct=True)

    panel = panel.merge(build_breadth(panel), on="date", how="left")

    if with_market_context:
        ctx = build_market_context(bond_yields, market_pe, sector_pe, macro)
        if len(ctx):
            panel = panel.merge(ctx, on="date", how="left")

        if instruments is not None and "sector" in instruments.columns:
            # The Instrument table holds duplicate tickers (e.g. NVDA twice);
            # dedupe or the merge silently multiplies price rows.
            inst = instruments[["ticker", "sector"]].dropna().drop_duplicates("ticker")
            panel = panel.merge(inst, on="ticker", how="left")
            if sector_pe is not None and len(sector_pe):
                sectors = build_sector_features(sector_pe)
                panel = panel.merge(
                    sectors, left_on=["date", "sector"], right_on=["date", "code"], how="left"
                ).drop(columns=["code"], errors="ignore")

        # Relative strength vs SPY and own sector ETF (sector codes are ETF tickers).
        etf_ret = build_etf_returns(etf_prices)
        if len(etf_ret):
            spy = etf_ret[etf_ret["code"] == "SPY"][["date", "etf_ret_20d"]]
            panel = panel.merge(spy.rename(columns={"etf_ret_20d": "spy_ret_20d"}),
                                on="date", how="left")
            panel["rel_spy_20d"] = panel["ret_20d"] - panel["spy_ret_20d"]
            if "sector" in panel.columns:
                panel = panel.merge(
                    etf_ret.rename(columns={"etf_ret_20d": "sector_ret_20d"}),
                    left_on=["date", "sector"], right_on=["date", "code"], how="left",
                ).drop(columns=["code"], errors="ignore")
                panel["rel_sector_20d"] = panel["ret_20d"] - panel["sector_ret_20d"]

    # Crypto positioning: perp funding (daily mean of 8h rates) + 7-day mean.
    if derivatives is not None and len(derivatives):
        deriv = derivatives.sort_values(["ticker", "date"]).copy()
        deriv["funding_rate_7d"] = (
            deriv.groupby("ticker")["funding_rate"].transform(lambda s: s.rolling(7).mean())
        )
        panel = panel.merge(
            deriv[["ticker", "date", "funding_rate", "funding_rate_7d"]],
            on=["ticker", "date"], how="left",
        )

    # BTC as the crypto market factor: every coin sees BTC momentum + its own
    # return relative to it (alt rotation signal).
    if "BTC" in set(panel["ticker"].unique()):
        btc = panel[panel["ticker"] == "BTC"][["date", "ret_20d"]]
        panel = panel.merge(btc.rename(columns={"ret_20d": "btc_ret_20d"}),
                            on="date", how="left")
        panel["rel_btc_20d"] = panel["ret_20d"] - panel["btc_ret_20d"]

    # Guarantee every feature column exists so dataset assembly never KeyErrors.
    for col in FEATURE_COLUMNS:
        if col not in panel.columns:
            panel[col] = float("nan")

    keep = META_COLUMNS + [c for c in ("sector",) if c in panel.columns] + FEATURE_COLUMNS
    return panel[keep].sort_values(["ticker", "date"]).reset_index(drop=True)
