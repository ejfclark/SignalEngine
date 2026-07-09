"""Market/regime context joined onto every (ticker, date) row.

Swing setups behave very differently in risk-on vs risk-off tape; these features
let the model condition on that. Values are forward-filled up to a week so
weekend/holiday gaps in the scraped sources don't punch holes in the panel.

WHERE MORE DATA WOULD HELP (in rough order of value for this engine):
  - VIX level + term structure   -> the single best risk-on/off feature for stocks;
                                    free from FRED/Yahoo. Slot it in exactly like us10y.
  - Crypto funding rates + open interest (per coin, via exchange APIs / CCXT)
                                 -> the closest crypto analogue to sentiment; strong
                                    mean-reversion signal when funding is extreme.
  - BTC dominance                -> regime split between BTC-led and alt-led markets.
  - DXY (dollar index)           -> risk appetite for both asset classes; FRED.
  - 2y-10y yield spread          -> already have 10y; adding 2y gives the curve.
  - Sector ETF prices (XLK etc.) -> relative-strength of a stock vs its own sector,
                                    a classic swing screen; Yahoo has them for free.
"""

from __future__ import annotations

import pandas as pd

FFILL_LIMIT = 7


def _daily(df: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    """Collapse to one row per date (last write wins), rename, sort."""
    out = df.sort_values("date").drop_duplicates("date", keep="last")
    return out[["date", *cols]].rename(columns=cols).reset_index(drop=True)


def build_market_context(
    bond_yields: pd.DataFrame,
    market_pe: pd.DataFrame,
    sector_pe: pd.DataFrame,
    macro: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per calendar date with market-wide features."""
    frames = []
    macro_has_rates = macro is not None and len(macro) and "us10y" in macro.columns
    if macro is not None and len(macro):
        m = macro.sort_values("date").copy()
        # Changes computed on trading days, before the calendar reindex.
        if "vix" in m.columns:
            m["vix_chg_5d"] = m["vix"].pct_change(5)
        if "dxy" in m.columns:
            m["dxy_ret_20d"] = m["dxy"].pct_change(20)
        if macro_has_rates:
            m["us10y_chg_1m"] = m["us10y"].diff(21) * 100.0  # bps over ~1 trading month
            if "us2y" in m.columns:
                m["curve_2s10s"] = m["us10y"] - m["us2y"]
        frames.append(m)
    # Legacy Bloomberg-scraped yields — only when FRED macro doesn't cover rates.
    if not macro_has_rates and bond_yields is not None and len(bond_yields):
        frames.append(_daily(
            bond_yields[bond_yields["country"] == "United States"],
            {"yield_pct": "us10y", "chg_1m_bps": "us10y_chg_1m"},
        ))
    if market_pe is not None and len(market_pe):
        frames.append(_daily(
            market_pe[market_pe["code"].str.lower() == "us"],
            {"pe": "mkt_pe", "dev_5yr_pct": "mkt_pe_dev"},
        ))
    if sector_pe is not None and len(sector_pe):
        frames.append(_daily(sector_pe[sector_pe["code"] == "SPY"], {"pe": "spy_pe"}))

    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame({"date": []})
    start = min(f["date"].min() for f in frames)
    end = max(f["date"].max() for f in frames)
    ctx = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    for frame in frames:
        ctx = ctx.merge(frame, on="date", how="left")
    ctx = ctx.set_index("date").ffill(limit=FFILL_LIMIT).reset_index()
    return ctx


def build_sector_features(sector_pe: pd.DataFrame) -> pd.DataFrame:
    """Per (date, sector_code): sector P/E and its premium over SPY."""
    pe = sector_pe.sort_values("date").drop_duplicates(["date", "code"], keep="last")
    spy = pe[pe["code"] == "SPY"][["date", "pe"]].rename(columns={"pe": "spy_pe"})
    out = pe[pe["code"] != "SPY"].merge(spy, on="date", how="left")
    out["sector_pe"] = out["pe"]
    out["sector_pe_dev"] = out["dev_5yr_pct"]
    out["sector_rel_pe"] = out["pe"] / out["spy_pe"]
    return out[["date", "code", "sector_pe", "sector_pe_dev", "sector_rel_pe"]]


def build_etf_returns(etf_prices: pd.DataFrame) -> pd.DataFrame:
    """Per (date, code): 20-day ETF return, for relative-strength features."""
    if etf_prices is None or etf_prices.empty:
        return pd.DataFrame()
    out = []
    for code, g in etf_prices.sort_values("date").groupby("ticker"):
        out.append(pd.DataFrame({
            "date": g["date"],
            "code": code,
            "etf_ret_20d": g["close"].pct_change(20),
        }))
    return pd.concat(out, ignore_index=True)


def build_breadth(panel: pd.DataFrame) -> pd.DataFrame:
    """Universe-internal regime features, computed from the price panel itself:
    breadth (share of tickers above their 20-day mean) and the equal-weight
    5-day universe return. Needs no external data and works for crypto too."""
    by_date = panel.groupby("date")
    breadth = by_date.apply(
        lambda d: (d["dist_sma20"] > 0).mean(), include_groups=False
    ).rename("breadth_20d")
    uni_ret = by_date["ret_5d"].mean().rename("universe_ret_5d")
    return pd.concat([breadth, uni_ret], axis=1).reset_index()
