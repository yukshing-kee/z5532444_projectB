"""Station 3 (extension) - sentiment tilt fusion.

Fuses the lagged sector sentiment index into equity fund weights.
The sentiment signal is already lagged by 1 trading day in
sector_sentiment_index(), so the fusion is look-ahead safe.

Design choices:
  - Tilt formula: w_tilt_i = w_i × max(0, 1 + α × sector_sentiment_i).
    max(0,.) keeps weights non-negative (long-only stays long-only).
  - After tilting, the equity portion is renormalised to its original sum
    so total portfolio allocation is unchanged.
  - Crypto tickers (suffix '-USD') are never tilted; equity sentiment does
    not apply to them (per brief).
  - Alpha = 0.5: a moderate tilt. Larger α concentrates more weight in
    high-sentiment sectors; α = 0 reproduces the base fund exactly.
  - Applied at each rebalance date only (same frequency as the base fund).
    Between rebalances the tilted weights are held constant, matching the
    base backtest's trading discipline.
  - A negative result is fully expected and still earns credit; the fusion
    is reported honestly as a before-vs-after comparison.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from src.portfolios import performance_metrics


def apply_sentiment(
    returns: pd.DataFrame,
    base_result: dict,
    sentiment_index: pd.DataFrame,
    ticker_sector_map: dict[str, str],
    alpha: float = 0.5,
    periods_per_year: int = 252,
) -> dict:
    """Apply a sector-sentiment tilt to base equity fund weights.

    Parameters
    ----------
    returns          : wide daily returns (date × ticker), same frame used in
                       the base backtest
    base_result      : output dict from portfolios.oos_backtest
    sentiment_index  : lagged sector sentiment (date × sector) from
                       sentiment.sector_sentiment_index — already shifted 1 day
    ticker_sector_map: {ticker: sector} for equity tickers
    alpha            : tilt strength (default 0.5)
    periods_per_year : for annualising performance metrics

    Returns
    -------
    dict with the same keys as oos_backtest:
        port_returns, weights, growth, metrics
    """
    base_weights_df = base_result["weights"]          # rebalance_date × ticker
    rebalance_dates = base_weights_df.index.sort_values()

    # ------------------------------------------------------------------
    # Step 1 — tilt weights at each rebalance date
    # ------------------------------------------------------------------
    tilted_rows: list[pd.Series] = []

    for date in rebalance_dates:
        w = base_weights_df.loc[date].copy()

        # Sentiment available at the open of `date` (lagged — yesterday's signal)
        if date in sentiment_index.index and not sentiment_index.loc[date].isna().all():
            sent = sentiment_index.loc[date]
        else:
            sent = pd.Series(0.0, index=sentiment_index.columns)

        # Identify active equity tickers (positive base weight, no '-USD' suffix)
        equity_tickers = [t for t in w.index if not t.endswith("-USD") and w[t] > 0]

        # Tilt equity weights by sector sentiment; floor at 0 (long-only)
        w_tilted = w.copy()
        for ticker in equity_tickers:
            sector = ticker_sector_map.get(ticker)
            s = float(sent.get(sector, 0.0)) if sector else 0.0
            w_tilted[ticker] = w[ticker] * max(0.0, 1.0 + alpha * s)

        # Renormalise equity portion to the original equity weight sum
        orig_eq_sum = w[equity_tickers].sum()
        new_eq_sum  = w_tilted[equity_tickers].sum()
        if new_eq_sum > 0 and orig_eq_sum > 0:
            scale = orig_eq_sum / new_eq_sum
            for t in equity_tickers:
                w_tilted[t] *= scale

        tilted_rows.append(w_tilted.rename(date))

    tilted_weights_df = pd.DataFrame(tilted_rows).fillna(0.0)
    tilted_weights_df.index.name = "date"

    # ------------------------------------------------------------------
    # Step 2 — re-run daily returns with tilted weights
    # Forward-fill rebalance weights across every live trading day so
    # weights are held constant between rebalances (same discipline as base).
    # ------------------------------------------------------------------
    live_dates = returns.index[returns.index >= rebalance_dates[0]]

    weights_daily = (
        tilted_weights_df
        .reindex(live_dates, method="ffill")
        .fillna(0.0)
    )

    common_cols  = weights_daily.columns.intersection(returns.columns)
    ret_aligned  = returns.loc[live_dates, common_cols].fillna(0.0)
    w_aligned    = weights_daily[common_cols]

    port_returns = (ret_aligned * w_aligned).sum(axis=1)
    port_returns.name = "sentiment_tilted"

    growth  = (1 + port_returns).cumprod()
    metrics = performance_metrics(port_returns, periods_per_year=periods_per_year)

    return {
        "port_returns": port_returns,
        "weights":      tilted_weights_df,
        "growth":       growth,
        "metrics":      metrics,
    }
