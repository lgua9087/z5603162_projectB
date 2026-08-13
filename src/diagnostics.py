"""Evidence-focused diagnostics for sentiment, robustness, and product suitability."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.portfolios import performance_metrics


def _hac_t_stat(values: pd.Series, *, max_lag: int) -> float:
    """Return a Bartlett-kernel HAC t-statistic for a sample mean."""

    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    count = len(clean)
    if count < 3:
        return np.nan
    demeaned = clean - clean.mean()
    long_run_variance = float(demeaned @ demeaned / count)
    for lag in range(1, min(max_lag, count - 1) + 1):
        covariance = float(demeaned[lag:] @ demeaned[:-lag] / count)
        weight = 1.0 - lag / (max_lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    standard_error = np.sqrt(max(long_run_variance, 0.0) / count)
    return float(clean.mean() / standard_error) if standard_error > 0 else np.nan


def _forward_returns(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Compound returns from date t through t+horizon-1 on the equity calendar."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    growth = returns.add(1.0).rolling(horizon, min_periods=horizon).apply(np.prod, raw=True)
    return growth.shift(-(horizon - 1)).sub(1.0)


def sentiment_predictive_diagnostics(
    ticker_signals: pd.DataFrame,
    equity_returns: pd.DataFrame,
    *,
    horizons: Iterable[int] = (1, 5),
    minimum_cross_section: int = 10,
    minimum_signal_observations: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test lagged coverage-scaled sentiment against short-horizon future returns.

    The signal dated t is already based on information through t-1. Horizon-one
    therefore uses the return on t; longer horizons compound from t onward.
    """

    coverage_column = (
        "coverage_window" if "coverage_window" in ticker_signals.columns else "coverage_21d"
    )
    required = {"date", "ticker", "signal_z", coverage_column}
    if missing := sorted(required.difference(ticker_signals.columns)):
        raise ValueError(f"ticker_signals are missing columns: {missing}")
    signals = ticker_signals.copy()
    signals["date"] = pd.to_datetime(signals["date"])
    if "signal_observations" not in signals:
        signals["signal_observations"] = np.where(
            signals[coverage_column].gt(0), minimum_signal_observations, 0
        )
    signals = signals.loc[
        signals["signal_observations"].ge(minimum_signal_observations)
        & signals[coverage_column].gt(0)
    ].copy()
    signals["effective_signal"] = signals["signal_z"] * np.sqrt(
        signals[coverage_column].clip(0.0, 1.0)
    )

    daily_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    matrix = equity_returns.sort_index().apply(pd.to_numeric, errors="coerce")
    for horizon in horizons:
        horizon = int(horizon)
        forward = (
            _forward_returns(matrix, horizon)
            .rename_axis(index="date", columns="ticker")
            .stack(future_stack=True)
            .dropna()
            .rename("forward_return")
            .reset_index()
        )
        merged = signals.merge(forward, on=["date", "ticker"], validate="one_to_one")
        for date, group in merged.groupby("date", sort=True):
            usable = group.replace([np.inf, -np.inf], np.nan).dropna(
                subset=["effective_signal", "forward_return"]
            )
            if len(usable) < minimum_cross_section:
                continue
            ranks = usable["effective_signal"].rank(method="first", pct=True)
            high = usable.loc[ranks.gt(0.80), "forward_return"]
            low = usable.loc[ranks.le(0.20), "forward_return"]
            if high.empty or low.empty:
                continue
            daily_rows.append(
                {
                    "date": date,
                    "horizon_days": horizon,
                    "cross_section_size": len(usable),
                    "spearman_ic": usable["effective_signal"].corr(
                        usable["forward_return"], method="spearman"
                    ),
                    "high_sentiment_return": float(high.mean()),
                    "low_sentiment_return": float(low.mean()),
                    "high_minus_low_return": float(high.mean() - low.mean()),
                }
            )

        daily = pd.DataFrame(daily_rows)
        selected = daily.loc[daily["horizon_days"].eq(horizon)].copy()
        if selected.empty:
            raise ValueError(f"no usable sentiment diagnostics for {horizon}-day horizon")
        summary_rows.append(
            {
                "horizon_days": horizon,
                "dates": len(selected),
                "mean_cross_section_size": float(selected["cross_section_size"].mean()),
                "mean_spearman_ic": float(selected["spearman_ic"].mean()),
                "median_spearman_ic": float(selected["spearman_ic"].median()),
                "positive_ic_share": float(selected["spearman_ic"].gt(0).mean()),
                "ic_hac_t_stat": _hac_t_stat(
                    selected["spearman_ic"], max_lag=max(horizon - 1, 0)
                ),
                "mean_high_sentiment_return": float(
                    selected["high_sentiment_return"].mean()
                ),
                "mean_low_sentiment_return": float(selected["low_sentiment_return"].mean()),
                "mean_high_minus_low_return": float(
                    selected["high_minus_low_return"].mean()
                ),
                "spread_hac_t_stat": _hac_t_stat(
                    selected["high_minus_low_return"], max_lag=max(horizon - 1, 0)
                ),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(daily_rows)


def annual_fund_results(fund_returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate calendar-year total return, volatility, and drawdown for every fund."""

    frame = fund_returns.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year
    rows: list[dict[str, object]] = []
    for (fund_id, year), group in frame.groupby(["fund_id", "year"], sort=True):
        group = group.sort_values("date")
        returns = pd.to_numeric(group["daily_return"], errors="coerce").dropna()
        family = str(group["family"].iloc[0])
        periods = 365 if family == "Crypto" else 252
        growth = returns.add(1.0).cumprod()
        rows.append(
            {
                "fund_id": fund_id,
                "fund_name": str(group["fund_name"].iloc[0]),
                "family": family,
                "method": str(group["method"].iloc[0]),
                "year": int(year),
                "observations": len(returns),
                "calendar_return": float(growth.iloc[-1] - 1.0),
                "annualized_volatility": float(returns.std(ddof=1) * np.sqrt(periods)),
                "max_drawdown": float(growth.div(growth.cummax()).sub(1.0).min()),
            }
        )
    return pd.DataFrame(rows)


def transaction_cost_sensitivity(
    fund_returns: pd.DataFrame,
    *,
    cost_scenarios_bps: Iterable[float] = (0.0, 5.0, 10.0),
    baseline_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Reprice realised paths under a few prespecified turnover-cost assumptions."""

    if baseline_cost_bps <= 0:
        raise ValueError("baseline_cost_bps must be positive to recover realised turnover")
    rows: list[dict[str, object]] = []
    for fund_id, group in fund_returns.groupby("fund_id", sort=True):
        group = group.sort_values("date")
        family = str(group["family"].iloc[0])
        periods = 365 if family == "Crypto" else 252
        realised_turnover = group["transaction_cost"] / (baseline_cost_bps / 10_000.0)
        gross = pd.Series(
            group["gross_daily_return"].to_numpy(dtype=float),
            index=pd.to_datetime(group["date"]),
        )
        for cost_bps in cost_scenarios_bps:
            net = gross - realised_turnover.to_numpy(dtype=float) * (cost_bps / 10_000.0)
            metrics = performance_metrics(net, periods_per_year=periods)
            rows.append(
                {
                    "fund_id": fund_id,
                    "fund_name": str(group["fund_name"].iloc[0]),
                    "family": family,
                    "method": str(group["method"].iloc[0]),
                    "transaction_cost_bps": float(cost_bps),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def management_fee_sensitivity(
    fund_returns: pd.DataFrame,
    *,
    annual_fee_rates: Iterable[float] = (0.0, 0.005, 0.01),
) -> pd.DataFrame:
    """Apply annual AUM fees to returns already net of the baseline trading-cost model."""

    rows: list[dict[str, object]] = []
    for fund_id, group in fund_returns.groupby("fund_id", sort=True):
        group = group.sort_values("date")
        family = str(group["family"].iloc[0])
        periods = 365 if family == "Crypto" else 252
        base = pd.Series(
            group["daily_return"].to_numpy(dtype=float),
            index=pd.to_datetime(group["date"]),
        )
        for annual_fee in annual_fee_rates:
            annual_fee = float(annual_fee)
            if not 0.0 <= annual_fee < 1.0:
                raise ValueError("annual management fees must lie in [0, 1)")
            daily_fee_multiplier = (1.0 - annual_fee) ** (1.0 / periods)
            after_fee = base.add(1.0).mul(daily_fee_multiplier).sub(1.0)
            metrics = performance_metrics(after_fee, periods_per_year=periods)
            rows.append(
                {
                    "fund_id": fund_id,
                    "fund_name": str(group["fund_name"].iloc[0]),
                    "family": family,
                    "method": str(group["method"].iloc[0]),
                    "annual_management_fee": annual_fee,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)
