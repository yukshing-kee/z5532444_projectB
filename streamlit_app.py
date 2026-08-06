"""SteadyFunds — Systematic Multi-Asset Investment Platform.

Investor journey:
    Tab 1  Fund Comparison   — side-by-side metrics table + growth of $1
    Tab 2  Fund Fact Sheet   — per-fund charts, metrics, and current holdings
    Tab 3  Sentiment         — sector news-sentiment heatmap + fusion results
    Tab 4  Allocation        — interactive fund allocation with portfolio summary

The app reads only precomputed artifacts from results/. It never runs backtests,
imports nltk, or recomputes sentiment scores — those are build-only steps.

Run locally:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="SteadyFunds",
    page_icon="📈",
    layout="wide",
)

RESULTS_DATA   = pathlib.Path("results/data")
RESULTS_TABLES = pathlib.Path("results/tables")

# ------------------------------------------------------------------
# Cached loaders — all heavy reads happen once per session
# ------------------------------------------------------------------

@st.cache_data(ttl=86_400, show_spinner=False)
def load_fund_returns() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DATA / "fund_returns.csv",
                       index_col="date", parse_dates=True)

@st.cache_data(ttl=86_400, show_spinner=False)
def load_fund_weights() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DATA / "fund_weights.csv", parse_dates=["date"])

@st.cache_data(ttl=86_400, show_spinner=False)
def load_performance_metrics() -> pd.DataFrame:
    return pd.read_csv(RESULTS_TABLES / "performance_metrics.csv")

@st.cache_data(ttl=86_400, show_spinner=False)
def load_sentiment_index() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DATA / "sector_sentiment_index.csv",
                       index_col="trading_date", parse_dates=True)

@st.cache_data(ttl=86_400, show_spinner=False)
def load_fusion_comparison() -> pd.DataFrame:
    return pd.read_csv(RESULTS_TABLES / "fusion_comparison.csv")

# ------------------------------------------------------------------
# Fund metadata
# ------------------------------------------------------------------

FUND_LABELS = {
    "combined_min_var": "Combined Min-Variance",
    "combined_max_sr":  "Combined Max-Sharpe",
    "equity_min_var":   "Equity Min-Variance",
    "equity_max_sr":    "Equity Max-Sharpe",
    "crypto_min_var":   "Crypto Min-Variance",
    "crypto_max_sr":    "Crypto Max-Sharpe",
}

FUND_FAMILY = {
    "combined_min_var": "Combined (Equity + Crypto)",
    "combined_max_sr":  "Combined (Equity + Crypto)",
    "equity_min_var":   "Equity Only",
    "equity_max_sr":    "Equity Only",
    "crypto_min_var":   "Crypto Only",
    "crypto_max_sr":    "Crypto Only",
}

FUND_COLORS = {
    "combined_min_var": "#1F3A5F",
    "combined_max_sr":  "#B23A48",
    "equity_min_var":   "#2E7D32",
    "equity_max_sr":    "#C99700",
    "crypto_min_var":   "#007C89",
    "crypto_max_sr":    "#6B5B95",
}

BASE_FUNDS = list(FUND_LABELS.keys())

# ------------------------------------------------------------------
# Load all data upfront
# ------------------------------------------------------------------
fund_returns  = load_fund_returns()
fund_weights  = load_fund_weights()
perf_metrics  = load_performance_metrics()
sentiment_idx = load_sentiment_index()
fusion_df     = load_fusion_comparison()

# ------------------------------------------------------------------
# App header
# ------------------------------------------------------------------
st.title("📈 SteadyFunds")
st.markdown(
    "Six systematically managed funds built from 50 US equities and 10 cryptocurrencies. "
    "Funds are rebalanced monthly using walk-forward out-of-sample optimisation. "
    "Data: 2020-2023 · Risk-free rate: 0 · Transaction costs: 0."
)

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Fund Comparison",
    "📄 Fund Fact Sheet",
    "📰 Sentiment Analytics",
    "💰 Allocation",
])


# ==================================================================
# TAB 1 — Fund Comparison
# ==================================================================
with tab1:
    st.subheader("Compare all funds at a glance")
    st.caption(
        "Out-of-sample period: 2021-01-04 to 2023-12-29. "
        "Annualised with 252 trading days (equity/combined) or 365 days (crypto)."
    )

    # ── Metrics table ──────────────────────────────────────────────
    tbl = perf_metrics[perf_metrics["fund"].isin(BASE_FUNDS)].copy()
    tbl["Fund"]   = tbl["fund"].map(FUND_LABELS)
    tbl["Family"] = tbl["fund"].map(FUND_FAMILY)
    tbl = (
        tbl.rename(columns={
            "ann_return":     "Ann. Return",
            "ann_volatility": "Ann. Volatility",
            "sharpe":         "Sharpe Ratio",
            "max_drawdown":   "Max Drawdown",
        })
        .sort_values("Sharpe Ratio", ascending=False)
        .reset_index(drop=True)
    )

    st.dataframe(
        tbl[["Fund", "Family", "Ann. Return", "Ann. Volatility", "Sharpe Ratio", "Max Drawdown"]]
        .style
        .format({
            "Ann. Return":    "{:.1%}",
            "Ann. Volatility": "{:.1%}",
            "Sharpe Ratio":   "{:.2f}",
            "Max Drawdown":   "{:.1%}",
        })
        .background_gradient(subset=["Sharpe Ratio"], cmap="RdYlGn")
        .background_gradient(subset=["Max Drawdown"], cmap="RdYlGn_r"),
        use_container_width=True,
        hide_index=True,
    )

    # ── Growth of $1 chart ─────────────────────────────────────────
    st.markdown("#### Growth of $1 — all six funds")
    fig, ax = plt.subplots(figsize=(10, 4))
    for fund in BASE_FUNDS:
        if fund in fund_returns.columns:
            r = fund_returns[fund].dropna()
            growth = (1 + r).cumprod()
            ax.plot(growth.index, growth,
                    color=FUND_COLORS[fund], lw=1.6, label=FUND_LABELS[fund])
    ax.axhline(1, color="#8A8F98", lw=0.8, ls=":")
    ax.set_ylabel("Growth of $1")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ==================================================================
# TAB 2 — Fund Fact Sheet
# ==================================================================
with tab2:
    selected = st.selectbox(
        "Select a fund to view its fact sheet",
        options=BASE_FUNDS,
        format_func=lambda x: FUND_LABELS[x],
    )

    row = perf_metrics[perf_metrics["fund"] == selected].iloc[0]

    # ── Metric cards ───────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annualised Return",     f"{row['ann_return']:.1%}")
    c2.metric("Annualised Volatility", f"{row['ann_volatility']:.1%}")
    c3.metric("Sharpe Ratio (rf = 0)", f"{row['sharpe']:.2f}")
    c4.metric("Max Drawdown",          f"{row['max_drawdown']:.1%}")

    st.divider()

    r      = fund_returns[selected].dropna()
    cum    = (1 + r).cumprod()
    dd_pct = (cum - cum.cummax()) / cum.cummax() * 100
    color  = FUND_COLORS[selected]

    col_l, col_r = st.columns(2)

    # ── Growth of $1 ───────────────────────────────────────────────
    with col_l:
        st.markdown(f"**Growth of $1**")
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.plot(cum.index, cum, color=color, lw=1.6)
        ax.fill_between(cum.index, 1, cum, where=(cum >= 1),
                        color=color, alpha=0.12)
        ax.fill_between(cum.index, 1, cum, where=(cum < 1),
                        color="#B23A48", alpha=0.15)
        ax.axhline(1, color="#8A8F98", lw=0.8, ls=":")
        ax.set_ylabel("Growth of $1")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Drawdown ───────────────────────────────────────────────────
    with col_r:
        st.markdown(f"**Drawdown from peak**")
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.fill_between(dd_pct.index, dd_pct, 0, color=color, alpha=0.28)
        ax.plot(dd_pct.index, dd_pct, color=color, lw=1.3)
        ax.axhline(0, color="#2F3337", lw=0.6)
        ax.set_ylabel("Drawdown (%)")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.divider()

    # ── Current holdings ───────────────────────────────────────────
    latest_date = fund_weights[fund_weights["fund"] == selected]["date"].max()
    holdings = (
        fund_weights[
            (fund_weights["fund"] == selected) &
            (fund_weights["date"] == latest_date)
        ]
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )

    st.markdown(
        f"**Current holdings** — most recent rebalance: {latest_date.date()}"
    )
    holdings_display = holdings[["ticker", "weight"]].copy()
    holdings_display.columns = ["Ticker", "Weight"]
    st.dataframe(
        holdings_display.style.format({"Weight": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
        height=min(400, 40 + len(holdings_display) * 35),
    )


# ==================================================================
# TAB 3 — Sentiment Analytics
# ==================================================================
with tab3:
    st.subheader("News-sentiment index — equity sectors")
    st.caption(
        "VADER compound score applied to daily headlines for 50 US equities. "
        "Averaged across tickers within each sector (equal-weight). "
        "Signal lagged 1 trading day — look-ahead safe. "
        "Ticker-days with no headlines treated as neutral (0.0)."
    )

    # ── Heatmap ────────────────────────────────────────────────────
    sectors = sorted(sentiment_idx.columns.tolist())
    monthly = sentiment_idx[sectors].resample("ME").mean()
    data    = monthly.T.values
    n_m, n_s = data.shape[1], len(sectors)

    fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=False)
    fig.set_layout_engine("none")

    im = ax.pcolormesh(
        np.arange(n_m + 1), np.arange(n_s + 1),
        data, cmap="RdYlGn", vmin=-0.15, vmax=0.30,
    )
    ax.set_yticks(np.arange(n_s) + 0.5)
    ax.set_yticklabels(sectors, fontsize=9)

    q_idx = [i for i, d in enumerate(monthly.index) if d.month in (1, 4, 7, 10)]
    ax.set_xticks([i + 0.5 for i in q_idx])
    ax.set_xticklabels(
        [monthly.index[i].strftime("%Y-%b") for i in q_idx],
        rotation=45, ha="right", fontsize=8,
    )

    cax  = ax.inset_axes([1.01, 0.0, 0.025, 1.0])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("VADER compound (monthly avg)", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    fig.subplots_adjust(left=0.10, right=0.88, top=0.97, bottom=0.28)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()

    # ── Fusion before-vs-after ─────────────────────────────────────
    st.markdown("#### Sentiment fusion — before vs after")
    st.caption(
        "A sentiment tilt (α = 0.5) applied at each monthly rebalance tilts equity "
        "weights toward higher-sentiment sectors. The tilt underperforms the base "
        "in both equity funds — a negative result documented honestly."
    )

    fd = fusion_df.copy()
    fd["Fund"] = fd["fund"].map(FUND_LABELS)
    fd["Type"] = fd["type"].str.capitalize()
    fd = fd.rename(columns={
        "ann_return": "Ann. Return", "ann_volatility": "Ann. Volatility",
        "sharpe": "Sharpe", "max_drawdown": "Max Drawdown",
    })
    st.dataframe(
        fd[["Fund", "Type", "Ann. Return", "Ann. Volatility", "Sharpe", "Max Drawdown"]]
        .style.format({
            "Ann. Return": "{:.1%}", "Ann. Volatility": "{:.1%}",
            "Sharpe": "{:.2f}", "Max Drawdown": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ==================================================================
# TAB 4 — Allocation
# ==================================================================
with tab4:
    st.subheader("Build your portfolio allocation")
    st.caption(
        "Set how much to invest in each fund. "
        "Allocations do not need to sum to 100% — they are normalised automatically."
    )

    total_amount = st.number_input(
        "Total investment ($)", min_value=1_000, max_value=10_000_000,
        value=100_000, step=5_000,
    )

    st.markdown("**Set allocation (%) per fund:**")
    cols = st.columns(3)
    alloc: dict[str, int] = {}
    for i, fund in enumerate(BASE_FUNDS):
        with cols[i % 3]:
            alloc[fund] = st.slider(
                FUND_LABELS[fund],
                min_value=0, max_value=100, value=17, step=1,
                key=f"slider_{fund}",
            )

    total_pct = sum(alloc.values())

    if total_pct == 0:
        st.warning("Set at least one fund allocation above 0% to see your portfolio summary.")
        st.stop()

    # Normalise
    w_norm    = {f: v / total_pct for f, v in alloc.items()}
    active    = [f for f, v in alloc.items() if v > 0]

    pm_idx    = perf_metrics.set_index("fund")
    w_ret     = sum(w_norm[f] * pm_idx.loc[f, "ann_return"]     for f in active)
    w_vol     = sum(w_norm[f] * pm_idx.loc[f, "ann_volatility"] for f in active)
    w_sharpe  = w_ret / w_vol if w_vol > 0 else 0.0
    w_dd      = sum(w_norm[f] * pm_idx.loc[f, "max_drawdown"]   for f in active)

    if total_pct != 100:
        st.info(f"Allocations sum to {total_pct}% — normalised to 100% below.")

    st.divider()
    st.markdown("**Your portfolio at a glance:**")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Investment",           f"${total_amount:,.0f}")
    m2.metric("Weighted Ann. Return", f"{w_ret:.1%}")
    m3.metric("Weighted Volatility",  f"{w_vol:.1%}")
    m4.metric("Weighted Sharpe",      f"{w_sharpe:.2f}")
    m5.metric("Weighted Max Drawdown",f"{w_dd:.1%}")

    st.markdown("**Allocation breakdown:**")
    breakdown = pd.DataFrame([
        {
            "Fund":          FUND_LABELS[f],
            "Family":        FUND_FAMILY[f],
            "Allocation":    w_norm[f],
            "Amount ($)":    f"${total_amount * w_norm[f]:,.0f}",
        }
        for f in active
    ])
    st.dataframe(
        breakdown.style.format({"Allocation": "{:.1%}"}),
        use_container_width=True,
        hide_index=True,
    )
