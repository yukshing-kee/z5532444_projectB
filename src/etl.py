"""Station 1 - ETL: load and clean the three datasets.

Load raw data through src.data_access. Integrity checks match Part A findings:
missing-date audit, duplicate check, outlier screen on returns. News dates are
UTC-aware; normalise before any merge. Returns are computed here (within each
panel) so they are never differenced across the calendar merge.
"""
from __future__ import annotations

import pandas as pd
from src import data_access


# ---------------------------------------------------------------------------
# Equities
# ---------------------------------------------------------------------------

def load_clean_equities() -> tuple[pd.DataFrame, dict]:
    """Load equity prices and run Station 1 integrity checks.

    Returns (clean_df, report). The df includes a 'ret' column (daily returns
    computed within the equity panel only).
    """
    df = data_access.load_equity_prices()
    report: dict = {}

    # 1. Duplicate ticker-date rows
    dups = df.duplicated(subset=["ticker", "date"]).sum()
    report["duplicate_ticker_date"] = int(dups)
    if dups:
        df = df.drop_duplicates(subset=["ticker", "date"])

    # 2. Missing-date audit against the union equity trading calendar
    full_calendar = pd.Index(sorted(df["date"].unique()))
    missing_by_ticker: dict = {}
    for ticker, grp in df.groupby("ticker"):
        ticker_dates = pd.Index(sorted(grp["date"].unique()))
        gaps = full_calendar.difference(ticker_dates)
        if len(gaps):
            missing_by_ticker[ticker] = len(gaps)
    report["tickers_with_missing_dates"] = len(missing_by_ticker)
    report["total_missing_date_slots"] = sum(missing_by_ticker.values())
    report["missing_detail"] = missing_by_ticker

    # 3. Compute returns within the equity panel (never across the merge)
    df = df.sort_values(["ticker", "date"])
    df["ret"] = df.groupby("ticker")["adjClose"].pct_change()

    # 4. Outlier screen: flag |return| > 20%; keep all — genuine extreme moves
    outlier_mask = df["ret"].abs() > 0.20
    report["outlier_count_gt20pct"] = int(outlier_mask.sum())
    report["outlier_detail"] = df.loc[outlier_mask, ["ticker", "date", "ret"]].copy()

    return df, report


# ---------------------------------------------------------------------------
# Crypto
# ---------------------------------------------------------------------------

def load_clean_crypto() -> tuple[pd.DataFrame, dict]:
    """Load crypto prices (365-day calendar) and run Station 1 integrity checks.

    Returns (clean_df, report). The df includes a 'ret' column computed within
    the full crypto calendar before any equity-calendar merge.
    """
    df = data_access.load_crypto_prices()
    report: dict = {}

    # 1. Cap at 2023-12-31 (brief: 10 rows are dated 2024-01-01)
    pre_cap = len(df)
    df = df[df["date"] <= "2023-12-31"].copy()
    report["rows_dropped_cap_2024"] = pre_cap - len(df)

    # 2. Duplicate ticker-date rows
    dups = df.duplicated(subset=["ticker", "date"]).sum()
    report["duplicate_ticker_date"] = int(dups)
    if dups:
        df = df.drop_duplicates(subset=["ticker", "date"])

    # 3. Missing-date audit against the full 365-day calendar
    full_calendar = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    missing_by_ticker = {}
    for ticker, grp in df.groupby("ticker"):
        ticker_dates = pd.DatetimeIndex(grp["date"].values)
        gaps = full_calendar.difference(ticker_dates)
        if len(gaps):
            missing_by_ticker[ticker] = len(gaps)
    report["tickers_with_missing_dates"] = len(missing_by_ticker)
    report["total_missing_date_slots"] = sum(missing_by_ticker.values())
    report["missing_detail"] = missing_by_ticker

    # 4. Compute returns within the crypto panel (before any equity merge)
    df = df.sort_values(["ticker", "date"])
    df["ret"] = df.groupby("ticker")["adjClose"].pct_change()

    # 5. Outlier screen: flag |return| > 20%; keep all
    outlier_mask = df["ret"].abs() > 0.20
    report["outlier_count_gt20pct"] = int(outlier_mask.sum())
    report["outlier_detail"] = df.loc[outlier_mask, ["ticker", "date", "ret"]].copy()

    return df, report


# ---------------------------------------------------------------------------
# News headlines
# ---------------------------------------------------------------------------

def load_clean_news() -> tuple[pd.DataFrame, dict]:
    """Load news headlines and run Station 1 integrity checks.

    Returns (clean_df, report). Timezone is normalised to tz-naive to match
    price dates.
    """
    df = data_access.load_news_headlines()
    report: dict = {}

    # 1. Normalise timezone: news dates are UTC-aware; prices are tz-naive
    df["date"] = df["date"].dt.tz_localize(None)
    report["timezone_stripped"] = True

    # 2. Exact duplicate check: ticker + date + title (NOT ticker-date alone)
    dups = df.duplicated(subset=["ticker", "date", "title"]).sum()
    report["exact_duplicates"] = int(dups)
    if dups:
        df = df.drop_duplicates(subset=["ticker", "date", "title"])
    report["rows_after_dedup"] = len(df)

    # 3. Null field summary
    report["null_counts"] = df.isnull().sum().to_dict()

    return df, report
