"""Station 3 - portfolio optimisation and out-of-sample walk-forward backtest.

Three fund families: combined (equity + crypto), equity-only, crypto-only.
Two optimisation methods per family: minimum-variance and maximum-Sharpe.
Risk-free rate = 0 throughout (stated assumption).
Annualisation: 252 for equity/combined (equity trading calendar),
               365 for crypto-only (365-day calendar).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def performance_metrics(daily_returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Annualised return, volatility, Sharpe (rf=0), and maximum drawdown.

    Annualised return uses the geometric mean so it compounds correctly
    over variable-length samples.
    """
    r = daily_returns.dropna()
    if len(r) == 0:
        return {k: np.nan for k in ("ann_return", "ann_volatility", "sharpe", "max_drawdown")}

    ann_return = float((1 + r).prod() ** (periods_per_year / len(r)) - 1)
    ann_vol = float(r.std() * np.sqrt(periods_per_year))
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    cumulative = (1 + r).cumprod()
    rolling_max = cumulative.cummax()
    max_drawdown = float(((cumulative - rolling_max) / rolling_max).min())

    return {
        "ann_return":     round(ann_return, 4),
        "ann_volatility": round(ann_vol, 4),
        "sharpe":         round(float(sharpe), 4) if not np.isnan(sharpe) else np.nan,
        "max_drawdown":   round(max_drawdown, 4),
    }


# ---------------------------------------------------------------------------
# Optimisers — long-only, fully invested
# ---------------------------------------------------------------------------

def _prep_cov(cov: np.ndarray) -> np.ndarray:
    """Annualise and add small diagonal regularisation for numerical stability.

    Daily return covariances are ~1e-6 to 1e-4 in magnitude, which can cause
    SLSQP to silently stall. Scaling by 252 and adding a tiny ridge keeps the
    objective well-conditioned without meaningfully shifting the weights.
    """
    scaled = cov * 252
    scaled += 1e-6 * np.eye(scaled.shape[0])
    return scaled


def _min_variance(cov: np.ndarray) -> np.ndarray:
    """Long-only minimum-variance weights (SLSQP)."""
    n = cov.shape[0]
    cov_s = _prep_cov(cov)
    result = minimize(
        lambda w: float(w @ cov_s @ w),
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    return result.x if result.success else np.ones(n) / n


def _max_sharpe(mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Long-only maximum-Sharpe (tangency) weights, rf=0 (SLSQP).

    Minimises negative Sharpe on annualised quantities so the objective
    sits on a reasonable scale (typically in [-5, 0] rather than [-0.01, 0]).
    Falls back to equal-weight when the optimiser does not converge.
    """
    n = len(mean)
    cov_s = _prep_cov(cov)
    mean_a = mean * 252  # annualised expected return

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(w @ mean_a)
        port_vol = float(np.sqrt(max(w @ cov_s @ w, 1e-12)))
        return -port_ret / port_vol

    result = minimize(
        neg_sharpe,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    return result.x if result.success else np.ones(n) / n


# ---------------------------------------------------------------------------
# Walk-forward out-of-sample backtest
# ---------------------------------------------------------------------------

def oos_backtest(
    returns: pd.DataFrame,
    method: str = "min_variance",
    window: int = 252,
    rebalance_freq: int = 21,
    periods_per_year: int = 252,
) -> dict:
    """Walk-forward out-of-sample backtest with no look-ahead bias.

    Parameters
    ----------
    returns : wide DataFrame (date × ticker) of daily returns
    method : "min_variance" or "max_sharpe"
    window : estimation window length in trading periods
    rebalance_freq : rebalance every N periods (default 21 ≈ monthly)
    periods_per_year : used for annualisation (252 equity/combined, 365 crypto)

    Returns
    -------
    dict with keys:
        port_returns     pd.Series  — daily out-of-sample portfolio returns
        weights          pd.DataFrame — weights at each rebalance (date × ticker)
        growth           pd.Series  — cumulative growth of $1
        metrics          dict       — performance metrics
        first_live_date  pd.Timestamp
    """
    dates = returns.index
    port_rets: list[tuple] = []
    weight_rows: list[pd.Series] = []
    current_weights: pd.Series | None = None

    for i in range(window, len(dates)):
        periods_live = i - window
        is_rebalance = (periods_live % rebalance_freq == 0)

        if is_rebalance or current_weights is None:
            # Estimation window: rows [i-window, i) — no look-ahead
            win_data = returns.iloc[i - window : i]

            # Drop tickers with >10% NaN in window, then drop remaining NaN rows
            valid_cols = win_data.columns[win_data.isna().mean() < 0.10]
            win_data = win_data[valid_cols].dropna()

            n = len(valid_cols)
            if n == 0:
                current_weights = pd.Series(dtype=float)
            else:
                if method == "min_variance":
                    cov = win_data.cov().values
                    w = _min_variance(cov)
                elif method == "max_sharpe":
                    mean = win_data.mean().values
                    cov = win_data.cov().values
                    w = _max_sharpe(mean, cov)
                else:
                    raise ValueError(f"Unknown method: {method!r}")

                current_weights = pd.Series(w, index=valid_cols)

            weight_rows.append(current_weights.rename(dates[i]))

        # Apply weights to this day's realised returns
        day_rets = returns.iloc[i]
        if current_weights is not None and len(current_weights) > 0:
            common = current_weights.index.intersection(day_rets.dropna().index)
            port_ret = float((current_weights[common] * day_rets[common]).sum())
        else:
            port_ret = 0.0

        port_rets.append((dates[i], port_ret))

    port_returns = pd.Series(
        [r for _, r in port_rets],
        index=pd.DatetimeIndex([d for d, _ in port_rets]),
        name=method,
    )

    weights_df = pd.DataFrame(weight_rows).fillna(0.0) if weight_rows else pd.DataFrame()
    weights_df.index.name = "date"

    growth = (1 + port_returns).cumprod()
    metrics = performance_metrics(port_returns, periods_per_year=periods_per_year)

    return {
        "port_returns":    port_returns,
        "weights":         weights_df,
        "growth":          growth,
        "metrics":         metrics,
        "first_live_date": dates[window],
    }
