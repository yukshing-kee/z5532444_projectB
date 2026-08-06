"""Station 3 - VADER sentiment scoring and sector sentiment index.

Scoring is a build step only. The deployed Streamlit app loads the
precomputed results/data/sector_sentiment_index.csv and never imports nltk.

Design choices documented here (required by brief):
  - Casing, punctuation, and negation are PRESERVED: VADER is rule-based
    and explicitly relies on all three ("NOT great" vs "great", "GREAT").
  - Each headline in a ticker-day row is scored individually (the panel
    joins them with ' || '); we average the compound scores. Averaging
    avoids long-text drift in VADER's compound formula.
  - Missing ticker-days (no headlines) are treated as 0.0 (neutral). A
    day without news carries no directional signal, and imputing anything
    else would inject look-ahead or carry-forward noise.
  - Signal is lagged by 1 trading day: the sector index on row t reflects
    headlines from day t-1, so no same-day look-ahead enters the backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ensure_vader() -> None:
    """Download the VADER lexicon once if not already cached (build step)."""
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        SentimentIntensityAnalyzer()  # triggers LookupError if lexicon missing
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def score_headlines(panel: pd.DataFrame) -> pd.DataFrame:
    """Score each ticker-day with VADER; return the mean compound score.

    Input  : assembled headline panel from features.assemble_headline_panel
             columns: trading_date, ticker, sector, headlines, n_headlines
    Output : same rows with an added 'sentiment_score' column in [-1, +1].

    Each row holds multiple headlines joined by ' || '. We split them,
    score each individually, and average — individual scoring preserves
    VADER's short-text accuracy (it is calibrated on sentence-length text).
    """
    _ensure_vader()
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    scores_out = []
    total = len(panel)
    for i, row in enumerate(panel.itertuples(index=False)):
        if i % 10_000 == 0:
            print(f"    scoring {i:,}/{total:,} ticker-days ...", flush=True)
        headlines = [h.strip() for h in str(row.headlines).split(" || ") if h.strip()]
        raw_scores = [sia.polarity_scores(h)["compound"] for h in headlines]
        scores_out.append(float(np.mean(raw_scores)) if raw_scores else 0.0)

    result = panel.copy()
    result["sentiment_score"] = scores_out
    return result


def sector_sentiment_index(
    scores: pd.DataFrame,
    equity_dates: pd.Index,
) -> pd.DataFrame:
    """Build a lagged daily sector sentiment index (wide: date x sector).

    Steps
    -----
    1. Pivot scores to wide (trading_date × ticker).
    2. Reindex to the full equity trading calendar; fill NaN with 0.0
       (neutral: no news = no directional signal).
    3. Average the 5 tickers within each sector (equal-weight).
    4. Shift forward by 1 trading day so day-t's index is based only on
       headlines from day t-1 or earlier (look-ahead safe).

    The first row after the lag is NaN (no prior-day signal) and is kept
    as NaN so downstream users can drop or fill it as they choose.

    Returns a DataFrame indexed by trading_date with one column per sector.
    """
    # Sector-to-ticker map derived from the scored panel
    universe: dict[str, list[str]] = (
        scores[["sector", "ticker"]]
        .drop_duplicates()
        .groupby("sector")["ticker"]
        .apply(list)
        .to_dict()
    )

    # Wide ticker scores on the full equity calendar
    all_dates = pd.DatetimeIndex(sorted(equity_dates))
    ticker_wide = (
        scores.pivot_table(
            index="trading_date",
            columns="ticker",
            values="sentiment_score",
            aggfunc="mean",
        )
        .reindex(all_dates)
        .fillna(0.0)  # no headlines → neutral
    )
    ticker_wide.columns.name = None
    ticker_wide.index.name = "trading_date"

    # Equal-weight average across tickers within each sector
    sector_idx = pd.DataFrame(index=all_dates)
    sector_idx.index.name = "trading_date"
    for sector, tickers in sorted(universe.items()):
        cols = [t for t in tickers if t in ticker_wide.columns]
        sector_idx[sector] = ticker_wide[cols].mean(axis=1)

    # Lag by 1 trading day — the critical look-ahead guard
    lagged = sector_idx.shift(1)
    return lagged
