# CLAUDE.md — Agent instructions for z5532444 Project B

This file records the instructions I give Claude Code when working on this
project. It is my own work and reflects what I actually asked the assistant to
do, the rules I set, and how I checked its output.

---

## Project overview

**App name:** (set in your report)  
**Goal:** Build a FinTech investment app that offers systematically managed
funds to investors. The app covers the full Data Factory Floor:

- **Station 1** — ETL: load, clean, and integrity-check the three datasets.
- **Station 2** — Features: daily returns per ticker, combined equity+crypto
  panel, assembled headline text panel.
- **Station 3** — Models: walk-forward out-of-sample portfolio backtests,
  VADER sentiment index, sentiment-tilt fusion.
- **Station 4** — App: Streamlit dashboard for the investor journey.

## Data sources

All data is loaded through `src/data_access.py` (provided helper — do not edit).
Three Parquet files in one hosted ZIP:

| Dataset | Content | Calendar |
|---|---|---|
| `equity_prices` | 50 US large-caps, daily OHLCV + adjClose + sector, 2020-2023 | ~252 days/year |
| `crypto_prices` | 10 cryptocurrencies, daily OHLCV + adjClose, 2020-2023 | 365 days/year — cap at 2023-12-31 |
| `news_headlines` | Date, ticker, sector, title, url, publisher for 50 equities | Multiple headlines per ticker-day |

**Never commit raw `.parquet` files or the data ZIP.**

---

## Folder layout

```
z5532444_projectB/
├── src/
│   ├── data_access.py   PROVIDED — do not edit
│   ├── etl.py           Station 1: load + clean (load_clean_equities, load_clean_crypto, load_clean_news)
│   ├── features.py      Station 2: daily_returns, build_combined_returns, assemble_headline_panel
│   ├── portfolios.py    Station 3: oos_backtest, performance_metrics, _min_variance, _max_sharpe
│   ├── sentiment.py     Station 3: score_headlines (VADER), sector_sentiment_index
│   ├── fusion.py        Station 3: apply_sentiment (tilt)
│   └── figures.py       All 6 required Part B exhibits
├── scripts/
│   ├── run_part_b.py    Master build script — runs all stations in order
│   └── check_handin.py  PROVIDED — run before submitting
├── results/
│   ├── data/            fund_returns.csv, fund_weights.csv, sector_sentiment_index.csv
│   ├── tables/          performance_metrics.csv, fusion_comparison.csv
│   └── figures/         growth_of_1.png, drawdown.png, weights_over_time.png,
│                        sharpe_barplot.png, sentiment_index.png, fusion_comparison.png
├── report/              report.docx (Word source) + report.pdf (submit this)
├── ai/                  Prompt logs
└── streamlit_app.py     Station 4 — deployed app entry point
```

---

## Coding rules I enforced with Claude Code

### No look-ahead in backtests
- Estimation window at step `i` uses rows `[i-window, i)` only — never row `i` or later.
- Sentiment signal is lagged by 1 trading day before any use in the backtest or fusion.
- Weights at rebalance date `t` are held and applied starting from day `t` — no peeking.

### Return computation order
- Compute `pct_change` within each panel (equity, crypto) separately before any merge.
- Left-merge crypto returns onto the equity trading calendar — never difference across the merge.

### Calendar and annualisation
- Equity and combined funds: 252 trading days per year.
- Crypto-only fund: 365 days per year.
- Crypto returns computed on the full 365-day calendar; combined panel uses equity calendar only.
- Cap crypto data at 2023-12-31 (10 rows dated 2024-01-01 exist in the raw file).

### Deduplication rules
- Equity prices: unique on ticker + date.
- Crypto prices: unique on ticker + date.
- News headlines: unique on ticker + date + title (NOT ticker-date alone).
- News dates are UTC-aware; normalise to tz-naive before any merge with price dates.

### VADER rules
- Do NOT strip casing, punctuation, or stopwords from headlines — VADER depends on all three.
- `nltk.download('vader_lexicon')` is a build step only; the deployed app must not call it.
- Score each headline in a ticker-day row individually; average compound scores.

### Optimiser numerical stability
- Annualise the covariance matrix (× 252) before passing to SLSQP to avoid tiny-scale stalls.
- Add a small diagonal regularisation (1e-6 × I) so the matrix is never singular.
- Use `ftol=1e-12, maxiter=1000` for SLSQP.

### Long-only constraint
- All portfolio weights are bounded [0, 1] and constrained to sum to 1.
- Sentiment tilt uses `max(0, 1 + α × sentiment)` to prevent any weight going negative.

### Deployed app rules
- The app reads precomputed CSVs from `results/` — it never imports `nltk` or recomputes backtests.
- Raw `.parquet` files and data ZIPs are never committed.
- Keep the app light so it runs on Streamlit Community Cloud free tier.

---

## How I check Claude's output

1. **Look-ahead check** — after any backtest function, I verify that `first_live_date`
   is at least `window` periods after the data start, and that weights at rebalance
   date `t` use only data up to `t-1`.

2. **Sanity-check metrics** — I read the printed Sharpe, AnnRet, and MaxDD after each
   backtest. Values outside plausible ranges (e.g. Sharpe > 3 or MaxDD > 0) trigger a review.

3. **Column/index alignment** — after any merge or pivot, I check `.shape`, `.columns`,
   and `.index.min()/.max()` to confirm no spurious NaN columns or date gaps.

4. **Required filenames** — I run `check_handin.py` after each major build step to
   confirm all required CSVs exist under `results/`.

5. **Figure review** — I view each saved PNG and check that axes are labelled,
   the sample period is stated, and the chart is legible before writing about it
   in the report.

6. **AI prose** — Claude drafts code; I write the economic interpretation and report
   prose in my own words. Any section drafted by Claude is rewritten before submission.

---

## Prompt log location

All prompt logs are kept in `ai/`. Each log entry records:
- The prompt I gave Claude Code
- The code or output it produced
- What I checked, what was wrong, and what I changed
