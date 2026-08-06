"""Station 3 required figures for Part B.

All figures use the shared fintools theme for Word/A4 output.
Each function saves to results/figures/ and returns the file path.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fintools.figures.theme import apply_theme, FINS_COLORS, FT_COLORS
from fintools.figures.export import save_figure

RESULTS_FIGURES = pathlib.Path(__file__).resolve().parent.parent / "results" / "figures"
SAMPLE = "2021-01-04 to 2023-12-29"


def _save(fig: plt.Figure, name: str) -> pathlib.Path:
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    path = RESULTS_FIGURES / name
    save_figure(fig, path)
    plt.close(fig)
    return path


def _drawdown(returns: pd.Series) -> pd.Series:
    cum = (1 + returns.dropna()).cumprod()
    return (cum - cum.cummax()) / cum.cummax()


# ---------------------------------------------------------------------------
# Colour maps
# ---------------------------------------------------------------------------

_FUND_STYLES = {
    # family       method         color                    linestyle
    "combined_min_var": (FINS_COLORS["navy"],    "-"),
    "combined_max_sr":  (FINS_COLORS["crimson"], "--"),
    "equity_min_var":   (FINS_COLORS["forest"],  "-"),
    "equity_max_sr":    (FINS_COLORS["gold"],    "--"),
    "crypto_min_var":   (FINS_COLORS["teal"],    "-"),
    "crypto_max_sr":    (FINS_COLORS["violet"],  "--"),
}

_FUND_LABELS = {
    "combined_min_var": "Combined Min-Var",
    "combined_max_sr":  "Combined Max-Sharpe",
    "equity_min_var":   "Equity Min-Var",
    "equity_max_sr":    "Equity Max-Sharpe",
    "crypto_min_var":   "Crypto Min-Var",
    "crypto_max_sr":    "Crypto Max-Sharpe",
}

_SECTOR_COLORS = [
    FT_COLORS["maroon"],  FT_COLORS["blue"],    FT_COLORS["teal"],
    FT_COLORS["pink"],    FT_COLORS["gold"],    FT_COLORS["purple"],
    FT_COLORS["orange"],  FT_COLORS["slate"],   FT_COLORS["brown"],
    FT_COLORS["green"],
]


# ---------------------------------------------------------------------------
# Figure 1 — Growth of $1
# ---------------------------------------------------------------------------

def plot_growth_of_1(fund_returns: pd.DataFrame) -> pathlib.Path:
    """Cumulative growth of $1 for all six base funds, 2021-2023."""
    apply_theme("paper")
    fig, ax = plt.subplots(figsize=(8.27, 4.5))

    base_funds = [f for f in _FUND_STYLES if f in fund_returns.columns]
    for fund in base_funds:
        r = fund_returns[fund].dropna()
        growth = (1 + r).cumprod()
        color, ls = _FUND_STYLES[fund]
        ax.plot(growth.index, growth, color=color, ls=ls,
                lw=1.5, label=_FUND_LABELS[fund])

    ax.axhline(1, color="#8A8F98", lw=0.8, ls=":")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    ax.legend(fontsize=7, ncol=2, framealpha=0.9)
    ax.set_title(
        f"Figure 1. Growth of $1 — six systematic funds\n"
        f"Note: walk-forward out-of-sample; rf = 0; long-only weights. "
        f"Sample: {SAMPLE}.",
        fontsize=8, loc="left",
    )
    fig.tight_layout()
    return _save(fig, "growth_of_1.png")


# ---------------------------------------------------------------------------
# Figure 2 — Drawdown
# ---------------------------------------------------------------------------

def plot_drawdown(fund_returns: pd.DataFrame) -> pathlib.Path:
    """Maximum-drawdown time series for the two equity base funds."""
    apply_theme("paper")
    pairs = [
        ("equity_min_var", FINS_COLORS["forest"],  "Equity Min-Var"),
        ("equity_max_sr",  FINS_COLORS["gold"],    "Equity Max-Sharpe"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(8.27, 5.0), sharex=True)

    for ax, (fund, color, label) in zip(axes, pairs):
        dd = _drawdown(fund_returns[fund].dropna())
        ax.fill_between(dd.index, dd * 100, 0,
                        color=color, alpha=0.25, linewidth=0)
        ax.plot(dd.index, dd * 100, color=color, lw=1.2)
        ax.set_ylabel("Drawdown (%)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_title(label, fontsize=9, loc="left")
        ax.axhline(0, color="#2F3337", lw=0.6)

    axes[-1].set_xlabel("Date")
    fig.suptitle(
        f"Figure 2. Drawdown — equity funds\n"
        f"Note: percentage decline from rolling peak. Sample: {SAMPLE}.",
        fontsize=8, x=0.01, ha="left",
    )
    fig.tight_layout()
    return _save(fig, "drawdown.png")


# ---------------------------------------------------------------------------
# Figure 3 — Portfolio weights over time
# ---------------------------------------------------------------------------

def plot_weights_over_time(
    fund_weights: pd.DataFrame,
    ticker_sector_map: dict[str, str],
    equity_dates: pd.Index,
) -> pathlib.Path:
    """Sector-level weight allocation over time: equity min-var vs max-Sharpe."""
    apply_theme("paper")

    sectors = sorted(set(ticker_sector_map.values()))
    colors = _SECTOR_COLORS[: len(sectors)]

    fig, axes = plt.subplots(1, 2, figsize=(8.27, 4.2), sharey=True, constrained_layout=False)
    funds_to_plot = [
        ("equity_min_var", "Equity Min-Var",    axes[0]),
        ("equity_max_sr",  "Equity Max-Sharpe", axes[1]),
    ]

    trading_days = pd.DatetimeIndex(sorted(equity_dates))

    for fund_name, title, ax in funds_to_plot:
        sub = fund_weights[fund_weights["fund"] == fund_name].copy()
        sub["sector"] = sub["ticker"].map(ticker_sector_map)
        sub = sub.dropna(subset=["sector"])

        sector_w = (
            sub.groupby(["date", "sector"])["weight"]
            .sum()
            .unstack(fill_value=0.0)
            .reindex(columns=sectors, fill_value=0.0)
        )
        sector_w = (
            sector_w
            .reindex(trading_days, method="ffill")
            .dropna()
        )

        ax.stackplot(
            sector_w.index,
            [sector_w[s] for s in sectors],
            labels=sectors,
            colors=colors,
            alpha=0.85,
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylim(0, 1)

        # Clean x-axis: one tick per year
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=7))

    axes[0].set_ylabel("Sector weight")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        f"Figure 3. Sector weight allocation — equity funds\n"
        f"Note: equal-weight within sector; weights held between monthly rebalances. "
        f"Sample: {SAMPLE}.",
        fontsize=8, x=0.01, ha="left",
    )
    fig.set_layout_engine("none")
    fig.subplots_adjust(bottom=0.22, top=0.82)
    return _save(fig, "weights_over_time.png")


# ---------------------------------------------------------------------------
# Figure 4 — Sharpe barplot
# ---------------------------------------------------------------------------

def plot_sharpe_barplot(metrics: pd.DataFrame) -> pathlib.Path:
    """Sharpe ratio comparison across all six base funds."""
    apply_theme("paper")

    # Only base funds (no _sent suffix)
    base = metrics[~metrics["fund"].str.endswith("_sent")].copy()

    family_order = ["combined", "equity", "crypto"]
    method_colors = {
        "min_var": FINS_COLORS["navy"],
        "max_sr":  FINS_COLORS["crimson"],
    }

    fig, ax = plt.subplots(figsize=(8.27, 3.8))

    x = np.arange(len(family_order))
    width = 0.35
    for i, (method_suffix, color) in enumerate(method_colors.items()):
        vals, errs = [], []
        for family in family_order:
            row = base[base["fund"] == f"{family}_{method_suffix}"]
            vals.append(float(row["sharpe"].iloc[0]) if len(row) else np.nan)
        label = "Min-Variance" if method_suffix == "min_var" else "Max-Sharpe"
        bars = ax.bar(x + (i - 0.5) * width, vals, width,
                      color=color, label=label, alpha=0.9, edgecolor="white")
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.015,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0, color="#2F3337", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["Combined\n(Equity + Crypto)", "Equity Only", "Crypto Only"])
    ax.set_ylabel("Sharpe Ratio (annualised, rf = 0)")
    ax.legend(fontsize=8)
    ax.set_title(
        f"Figure 4. Sharpe ratio by fund family and optimisation method\n"
        f"Note: annualised; rf = 0; 252 trading days (365 for crypto). "
        f"Sample: {SAMPLE}.",
        fontsize=8, loc="left",
    )
    fig.tight_layout()
    return _save(fig, "sharpe_barplot.png")


# ---------------------------------------------------------------------------
# Figure 5 — Sector sentiment index
# ---------------------------------------------------------------------------

def plot_sentiment_index(sent_index: pd.DataFrame) -> pathlib.Path:
    """Sector sentiment heatmap — monthly average VADER compound score, 2020-2023.

    A heatmap is clearer than 10 overlapping lines: colour shows whether a
    sector's news was net-positive (green) or net-negative (red) each month.
    """
    apply_theme("paper")

    sectors = sorted(sent_index.columns.tolist())

    # Monthly averages reduce noise without losing the trend
    monthly = sent_index[sectors].resample("ME").mean()

    data = monthly.T.values           # shape: (10 sectors × N months)
    n_months = data.shape[1]
    n_sectors = len(sectors)

    # constrained_layout=False prevents the colorbar from re-enabling the engine
    fig, ax = plt.subplots(figsize=(8.27, 3.8), constrained_layout=False)

    im = ax.pcolormesh(
        np.arange(n_months + 1),
        np.arange(n_sectors + 1),
        data,
        cmap="RdYlGn",
        vmin=-0.15,
        vmax=0.30,
    )

    # Y-axis: sector names centred in each row
    ax.set_yticks(np.arange(n_sectors) + 0.5)
    ax.set_yticklabels(sectors, fontsize=8)

    # X-axis: quarterly labels (every 3 months)
    quarter_idx = [i for i, d in enumerate(monthly.index) if d.month in (1, 4, 7, 10)]
    ax.set_xticks([i + 0.5 for i in quarter_idx])
    ax.set_xticklabels(
        [monthly.index[i].strftime("%Y-%b") for i in quarter_idx],
        rotation=45, ha="right", fontsize=7,
    )

    # Manual colorbar axes avoids constrained_layout fight
    cax = ax.inset_axes([1.01, 0.0, 0.03, 1.0])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("VADER compound (monthly avg)", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(
        "Figure 5. Sector news-sentiment index — monthly average VADER compound score\n"
        "Note: score in [−1, +1]; lagged 1 trading day (look-ahead safe); "
        "green = positive, red = negative. Sample: 2020-01 to 2023-12.",
        fontsize=8, loc="left",
    )
    fig.subplots_adjust(left=0.10, right=0.88, top=0.82, bottom=0.22)
    return _save(fig, "sentiment_index.png")


# ---------------------------------------------------------------------------
# Figure 6 — Fusion before-vs-after
# ---------------------------------------------------------------------------

def plot_fusion_comparison(fund_returns: pd.DataFrame) -> pathlib.Path:
    """Growth of $1: base vs sentiment-augmented equity funds."""
    apply_theme("paper")

    pairs = [
        ("equity_min_var", "equity_min_var_sent",
         FINS_COLORS["forest"], "Equity Min-Var"),
        ("equity_max_sr",  "equity_max_sr_sent",
         FINS_COLORS["gold"],   "Equity Max-Sharpe"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8.27, 4.0), sharey=False)

    for ax, (base_col, aug_col, color, title) in zip(axes, pairs):
        for col, ls, lbl in [
            (base_col, "-",  "Base"),
            (aug_col,  "--", "Sentiment-tilted (α=0.5)"),
        ]:
            if col in fund_returns.columns:
                r = fund_returns[col].dropna()
                growth = (1 + r).cumprod()
                ax.plot(growth.index, growth,
                        color=color, ls=ls, lw=1.5, label=lbl)

        ax.axhline(1, color="#8A8F98", lw=0.7, ls=":")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Date")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Growth of $1")
    fig.suptitle(
        "Figure 6. Sentiment fusion: base vs augmented — growth of $1\n"
        f"Note: tilt α=0.5 applied at monthly rebalances; lagged 1 day. "
        f"Sample: {SAMPLE}.",
        fontsize=8, x=0.01, ha="left",
    )
    fig.tight_layout()
    return _save(fig, "fusion_comparison.png")
