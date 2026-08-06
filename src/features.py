"""Station 2 - features: return features and headline text assembly.

daily_returns        — per-ticker pct_change on adjClose (long format)
build_combined_returns — left-merge crypto returns onto equity trading calendar
                         to produce the wide (date × ticker) panel the optimiser needs
assemble_headline_panel — date-align headlines to equity trading days (assembly only;
                           VADER scoring and signal lagging are in src/sentiment.py)
"""
from __future__ import annotations

import pandas as pd


def daily_returns(prices: pd.DataFrame, price_col: str = "adjClose") -> pd.DataFrame:
    """Compute daily returns per ticker from adjClose; result is long format."""
    df = prices.sort_values(["ticker", "date"]).copy()
    df["ret"] = df.groupby("ticker")[price_col].pct_change()
    return df


def build_combined_returns(
    eq_df: pd.DataFrame,
    cr_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-merge crypto returns onto the equity trading calendar.

    Both inputs must already have a 'ret' column computed within their own
    panel (i.e. call daily_returns on each BEFORE calling this function).
    The equity calendar is the left spine: weekend-only crypto moves are
    excluded because a fund trading on equity days cannot act on them.

    Returns a wide DataFrame indexed by date with one column per ticker
    (50 equities + 10 cryptos). Crypto columns keep their '-USD' suffix
    so they are easily separated downstream.
    """
    # Pivot equity returns to wide
    eq_wide = (
        eq_df.dropna(subset=["ret"])
        .pivot_table(index="date", columns="ticker", values="ret")
    )
    eq_wide.columns.name = None

    # Pivot crypto returns to wide (on the full 365-day calendar)
    cr_wide = (
        cr_df.dropna(subset=["ret"])
        .pivot_table(index="date", columns="ticker", values="ret")
    )
    cr_wide.columns.name = None

    # Left-merge: equity calendar is the spine
    combined = eq_wide.merge(cr_wide, left_index=True, right_index=True, how="left")
    combined.index.name = "date"
    return combined


def assemble_headline_panel(
    headlines: pd.DataFrame,
    equity_dates: pd.Index,
) -> pd.DataFrame:
    """Assemble headlines into a daily text panel aligned to equity trading days.

    Non-trading-day headlines (weekends, holidays) are bumped forward to the
    next equity trading day. Headlines are grouped by trading_date + ticker +
    sector, with all titles joined by ' || '. Raw text is preserved — VADER
    depends on casing, punctuation, and negation.
    """
    df = headlines.copy()

    trading_days = pd.DatetimeIndex(sorted(equity_dates)).normalize()

    def _next_trading_day(d: pd.Timestamp) -> pd.Timestamp | None:
        d = pd.Timestamp(d).normalize()
        if d in trading_days:
            return d
        future = trading_days[trading_days > d]
        return future[0] if len(future) else None

    df["trading_date"] = df["date"].apply(_next_trading_day)
    df = df.dropna(subset=["trading_date"])

    panel = (
        df.groupby(["trading_date", "ticker", "sector"], observed=True)
        .agg(
            headlines=("title", lambda x: " || ".join(x)),
            n_headlines=("title", "count"),
        )
        .reset_index()
        .sort_values(["trading_date", "ticker"])
        .reset_index(drop=True)
    )
    return panel
