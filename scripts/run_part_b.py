"""Reproduce all Part B results. Run from the project root:

    python scripts/run_part_b.py

Stations run in order:
    1+2  ETL -> features (equity, crypto, combined, headline panel)
    3a   Portfolio backtests (combined, equity-only, crypto-only)
    3b   Sentiment index (run after sentiment.py is implemented)
    3c   Fusion (run after fusion.py is implemented)

Required outputs written here:
    results/data/fund_returns.csv
    results/data/fund_weights.csv
    results/tables/performance_metrics.csv
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.etl import load_clean_equities, load_clean_crypto, load_clean_news
from src.features import daily_returns, build_combined_returns, assemble_headline_panel
from src.portfolios import oos_backtest
from src.sentiment import score_headlines, sector_sentiment_index
from src.fusion import apply_sentiment
from src.figures import (
    plot_growth_of_1,
    plot_drawdown,
    plot_weights_over_time,
    plot_sharpe_barplot,
    plot_sentiment_index,
    plot_fusion_comparison,
)

RESULTS_DATA   = ROOT / "results" / "data"
RESULTS_TABLES = ROOT / "results" / "tables"
RESULTS_DATA.mkdir(parents=True, exist_ok=True)
RESULTS_TABLES.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crypto_wide(cr_df: pd.DataFrame) -> pd.DataFrame:
    """Crypto-only returns on the full 365-day calendar (pre-merge)."""
    return (
        cr_df.dropna(subset=["ret"])
        .pivot_table(index="date", columns="ticker", values="ret")
        .rename_axis(columns=None)
    )


def _equity_cols(combined: pd.DataFrame) -> list[str]:
    return [c for c in combined.columns if not c.endswith("-USD")]


def _crypto_cols(combined: pd.DataFrame) -> list[str]:
    return [c for c in combined.columns if c.endswith("-USD")]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Stations 1+2 — ETL and features
    # ------------------------------------------------------------------
    print("=== Station 1: ETL ===")
    eq_df, eq_rep = load_clean_equities()
    cr_df, cr_rep = load_clean_crypto()
    news_df, news_rep = load_clean_news()
    print(f"  equities {eq_df.shape}, outliers={eq_rep['outlier_count_gt20pct']}")
    print(f"  crypto   {cr_df.shape}, cap_dropped={cr_rep['rows_dropped_cap_2024']}")
    print(f"  news     {news_df.shape}, dupes_removed={news_rep['exact_duplicates']}")

    print("\n=== Station 2: Features ===")
    combined = build_combined_returns(eq_df, cr_df)
    print(f"  combined returns panel: {combined.shape}  "
          f"({len(_equity_cols(combined))} equity + {len(_crypto_cols(combined))} crypto)")

    equity_dates = eq_df["date"].unique()
    headline_panel = assemble_headline_panel(news_df, equity_dates)
    print(f"  headline panel: {headline_panel.shape}")

    # Separate return panels for single-asset funds
    equity_returns = combined[_equity_cols(combined)]
    cr_wide        = _crypto_wide(cr_df)          # 365-day calendar

    # ------------------------------------------------------------------
    # Station 3a — Portfolio backtests
    # ------------------------------------------------------------------
    print("\n=== Station 3a: Portfolio backtests ===")

    FUNDS = [
        # (fund_name,         returns_df,     method,          window, rebalance_freq, ppy)
        ("combined_min_var",  combined,        "min_variance",  252,    21,             252),
        ("combined_max_sr",   combined,        "max_sharpe",    252,    21,             252),
        ("equity_min_var",    equity_returns,  "min_variance",  252,    21,             252),
        ("equity_max_sr",     equity_returns,  "max_sharpe",    252,    21,             252),
        ("crypto_min_var",    cr_wide,         "min_variance",  365,    30,             365),
        ("crypto_max_sr",     cr_wide,         "max_sharpe",    365,    30,             365),
    ]

    all_returns: dict[str, pd.Series] = {}
    all_weights: list[pd.DataFrame]   = []
    metrics_rows: list[dict]           = []
    fund_results: dict[str, dict]      = {}   # keyed by fund_name for fusion step

    for fund_name, ret_df, method, window, rebal, ppy in FUNDS:
        print(f"  running {fund_name} ...", end=" ", flush=True)
        result = oos_backtest(
            ret_df,
            method=method,
            window=window,
            rebalance_freq=rebal,
            periods_per_year=ppy,
        )
        print(f"Sharpe={result['metrics']['sharpe']:.3f}  "
              f"AnnRet={result['metrics']['ann_return']:.1%}  "
              f"MaxDD={result['metrics']['max_drawdown']:.1%}  "
              f"first_live={result['first_live_date'].date()}")

        all_returns[fund_name] = result["port_returns"]
        fund_results[fund_name] = result

        # Tag weights with fund name for the long-format output
        w = result["weights"].copy()
        w.insert(0, "fund", fund_name)
        all_weights.append(w)

        metrics_rows.append({"fund": fund_name, **result["metrics"]})

    # ------------------------------------------------------------------
    # Save required outputs
    # ------------------------------------------------------------------
    print("\n=== Saving outputs ===")

    # results/data/fund_returns.csv — wide: date x fund
    fund_returns_df = pd.DataFrame(all_returns)
    fund_returns_df.index.name = "date"
    fund_returns_df.to_csv(RESULTS_DATA / "fund_returns.csv")
    print(f"  fund_returns.csv  {fund_returns_df.shape}")

    # results/data/fund_weights.csv — long: date, fund, ticker, weight
    weight_long_parts = []
    for w_df in all_weights:
        fund_label = w_df["fund"].iloc[0]
        ticker_cols = [c for c in w_df.columns if c != "fund"]
        melted = (
            w_df[ticker_cols]
            .reset_index()
            .melt(id_vars="date", var_name="ticker", value_name="weight")
        )
        melted.insert(1, "fund", fund_label)
        weight_long_parts.append(melted)

    fund_weights_df = pd.concat(weight_long_parts, ignore_index=True)
    fund_weights_df = fund_weights_df[fund_weights_df["weight"] > 0]  # drop zero-weight rows
    fund_weights_df.to_csv(RESULTS_DATA / "fund_weights.csv", index=False)
    print(f"  fund_weights.csv  {fund_weights_df.shape}")

    # results/tables/performance_metrics.csv
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(RESULTS_TABLES / "performance_metrics.csv", index=False)
    print(f"  performance_metrics.csv  {metrics_df.shape}")

    # ------------------------------------------------------------------
    # Station 3b — Sentiment index
    # ------------------------------------------------------------------
    print("\n=== Station 3b: Sentiment index ===")
    print("  scoring headlines with VADER ...")
    scored = score_headlines(headline_panel)
    print(f"  scored {len(scored):,} ticker-days  "
          f"mean={scored['sentiment_score'].mean():.4f}  "
          f"std={scored['sentiment_score'].std():.4f}")

    print("  building sector sentiment index (lagged 1 trading day) ...")
    sent_index = sector_sentiment_index(scored, equity_dates)
    print(f"  index shape={sent_index.shape}  "
          f"date_range={sent_index.index.min().date()} to {sent_index.index.max().date()}")

    sent_index.to_csv(RESULTS_DATA / "sector_sentiment_index.csv")
    print(f"  sector_sentiment_index.csv  {sent_index.shape}")

    # ------------------------------------------------------------------
    # Station 3c — Fusion (sentiment tilt on equity funds)
    # ------------------------------------------------------------------
    print("\n=== Station 3c: Fusion ===")

    # Ticker → sector map (equity only)
    ticker_sector_map = eq_df.groupby("ticker")["sector"].first().to_dict()

    FUSION_FUNDS = [
        ("equity_min_var", equity_returns, 252),
        ("equity_max_sr",  equity_returns, 252),
    ]

    fusion_comparison: list[dict] = []

    for base_name, ret_df, ppy in FUSION_FUNDS:
        aug_name = f"{base_name}_sent"
        print(f"  fusing {base_name} -> {aug_name} ...", end=" ", flush=True)

        aug_result = apply_sentiment(
            returns=ret_df,
            base_result=fund_results[base_name],
            sentiment_index=sent_index,
            ticker_sector_map=ticker_sector_map,
            alpha=0.5,
            periods_per_year=ppy,
        )

        print(f"Sharpe={aug_result['metrics']['sharpe']:.3f}  "
              f"AnnRet={aug_result['metrics']['ann_return']:.1%}  "
              f"MaxDD={aug_result['metrics']['max_drawdown']:.1%}")

        all_returns[aug_name] = aug_result["port_returns"]

        # Fusion comparison rows (base then augmented)
        base_m = fund_results[base_name]["metrics"]
        aug_m  = aug_result["metrics"]
        fusion_comparison.append({"fund": base_name, "type": "base",      **base_m})
        fusion_comparison.append({"fund": base_name, "type": "augmented", **aug_m})

    # Save fusion comparison table
    fusion_df = pd.DataFrame(fusion_comparison)
    fusion_df.to_csv(RESULTS_TABLES / "fusion_comparison.csv", index=False)
    print(f"\n  fusion_comparison.csv  {fusion_df.shape}")

    # Overwrite fund_returns.csv with augmented fund columns included
    fund_returns_df = pd.DataFrame(all_returns)
    fund_returns_df.index.name = "date"
    fund_returns_df.to_csv(RESULTS_DATA / "fund_returns.csv")
    print(f"  fund_returns.csv updated  {fund_returns_df.shape}")

    print("\nAll stations complete.")
    print(f"\n{'='*55}")
    print("FUSION BEFORE vs AFTER")
    print(f"{'='*55}")
    print(fusion_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Figures — all 6 required exhibits
    # ------------------------------------------------------------------
    print("\n=== Figures ===")

    fund_returns_full = pd.DataFrame(all_returns)
    fund_returns_full.index.name = "date"

    metrics_df = pd.read_csv(RESULTS_TABLES / "performance_metrics.csv")
    fund_weights_df = pd.read_csv(RESULTS_DATA / "fund_weights.csv",
                                  parse_dates=["date"])
    ticker_sector_map = eq_df.groupby("ticker")["sector"].first().to_dict()

    p = plot_growth_of_1(fund_returns_full)
    print(f"  saved {p.name}")

    p = plot_drawdown(fund_returns_full)
    print(f"  saved {p.name}")

    p = plot_weights_over_time(fund_weights_df, ticker_sector_map, equity_dates)
    print(f"  saved {p.name}")

    p = plot_sharpe_barplot(metrics_df)
    print(f"  saved {p.name}")

    p = plot_sentiment_index(sent_index)
    print(f"  saved {p.name}")

    p = plot_fusion_comparison(fund_returns_full)
    print(f"  saved {p.name}")

    print("\nDone. All required outputs saved under results/.")


if __name__ == "__main__":
    main()
