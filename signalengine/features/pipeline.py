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
    # compression / breakout structure (the quantified content of chart patterns)
    "atr_ratio_10_60", "bb_width_pctile", "range_contraction", "inside_days_5",
    "dist_20d_high", "dist_60d_high", "days_since_20d_high", "higher_lows_20", "vol_dryup",
    # fundamentals (stocks only)
    "pe_ratio", "days_to_earnings",
    # NOTE: earnings-event features (days_since_earnings, earnings_reaction,
    # eps_chg_63d) are computed in indicators.py but REJECTED from the model
    # (Exp S3.1 2026-07-12: hurt the gated book; only 36% coverage). Retest
    # once fundamentals history deepens or FMP surprise data exists.
    # eps_chg_yoy (Exp S3.1c, free YoY-vs-consensus proxy) REJECTED 2026-07-13:
    # hurt the gated book in 5/5 seeds. Computed in indicators.py, not fed to
    # the model. Real consensus-surprise data (FMP or similar) is still the
    # only path to a genuine earnings-surprise feature — see docs/05-experiments.md.
    # behavioral identity (Exp S3.1, ADOPTED 2026-07-12): who this ticker is,
    # so regime features can interact with character — gated book
    # +0.89% -> +1.59%/trade, Sharpe 1.56 -> 2.31, 5/5 seeds
    "beta_spy_120d", "corr_spy_120d", "idio_vol_share", "vix_sens_120d", "dollar_vol_rank",
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

# The 11 SPDR sector ETFs that instruments.parquet maps GICS sectors onto.
# Fixed list (not derived from data) so pandas category codes are identical
# at train time and predict time regardless of which sectors appear.
SECTOR_CODES = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]


def feature_frame(panel: pd.DataFrame, include_sector: bool = False) -> pd.DataFrame:
    """The model input matrix. With include_sector, `sector` rides along as a
    pandas categorical that LightGBM splits natively (subset splits, not
    ordinal); crypto/unmapped rows are NaN, which LightGBM routes on its own."""
    if not include_sector:
        return panel[FEATURE_COLUMNS]
    X = panel[FEATURE_COLUMNS].copy()
    sector = panel["sector"] if "sector" in panel.columns else pd.Series(pd.NA, index=panel.index)
    sector = sector.where(sector.isin(SECTOR_CODES))  # unknown codes -> NaN, not an error
    X["sector"] = pd.Categorical(sector, categories=SECTOR_CODES)
    return X


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

    # Liquidity identity: where this ticker's 20d dollar volume sits in the
    # universe today (illiquid names gap through stops; the model should know).
    dv = (panel["close"] * panel["volume"]).groupby(panel["ticker"]).transform(
        lambda s: s.rolling(20, min_periods=10).mean())
    panel["dollar_vol_rank"] = dv.groupby(panel["date"]).rank(pct=True)

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

        # Behavioral identity vs the market (Exp S3.1): rolling beta/correlation
        # to SPY, idiosyncratic share, and sensitivity to VIX changes. All
        # look-back rolling windows over daily returns.
        if etf_prices is not None and len(etf_prices):
            spy_px = etf_prices[etf_prices["ticker"] == "SPY"].sort_values("date")
            if len(spy_px):
                panel = panel.merge(pd.DataFrame({
                    "date": spy_px["date"],
                    "spy_ret_1d": spy_px["close"].pct_change(),
                }), on="date", how="left")
            if "vix" in panel.columns:
                vday = panel[["date", "vix"]].dropna().drop_duplicates("date").sort_values("date")
                panel = panel.merge(pd.DataFrame({
                    "date": vday["date"], "vix_chg_1d_tmp": vday["vix"].diff(),
                }), on="date", how="left")

            def _behavioral(d: pd.DataFrame) -> pd.DataFrame:
                out = pd.DataFrame(index=d.index)
                if "spy_ret_1d" in d:
                    var = d["spy_ret_1d"].rolling(120, min_periods=60).var()
                    out["beta_spy_120d"] = (
                        d["ret_1d"].rolling(120, min_periods=60).cov(d["spy_ret_1d"]) / var
                    )
                    out["corr_spy_120d"] = (
                        d["ret_1d"].rolling(120, min_periods=60).corr(d["spy_ret_1d"])
                    )
                if "vix_chg_1d_tmp" in d:
                    out["vix_sens_120d"] = (
                        d["ret_1d"].rolling(120, min_periods=60).corr(d["vix_chg_1d_tmp"])
                    )
                return out

            if "spy_ret_1d" in panel.columns or "vix_chg_1d_tmp" in panel.columns:
                behav = panel.groupby("ticker", sort=False).apply(
                    _behavioral, include_groups=False).reset_index(level=0, drop=True)
                for col in behav.columns:
                    panel[col] = behav[col]
                if "corr_spy_120d" in panel.columns:
                    panel["idio_vol_share"] = 1.0 - panel["corr_spy_120d"] ** 2

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
