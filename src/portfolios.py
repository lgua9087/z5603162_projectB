"""Walk-forward systematic funds with explicit information timing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

METHOD_LABELS = {
    "equal_weight": "Equal Weight",
    "min_variance": "Minimum Variance",
    "max_sharpe": "Maximum Sharpe",
    "risk_parity": "Risk Parity",
}


@dataclass(frozen=True)
class BacktestResult:
    """Outputs for one separately investable fund."""

    returns: pd.DataFrame
    weights: pd.DataFrame
    metrics: dict[str, object]


def _normalize_with_cap(raw: np.ndarray, cap: float) -> np.ndarray:
    """Project nonnegative scores to a capped simplex using deterministic clipping."""

    values = np.nan_to_num(np.asarray(raw, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    if values.sum() <= 0:
        values = np.ones_like(values)
    weights = values / values.sum()
    effective_cap = max(float(cap), 1.0 / len(weights) + 1e-10)
    for _ in range(100):
        over = weights > effective_cap
        if not over.any():
            break
        excess = float((weights[over] - effective_cap).sum())
        weights[over] = effective_cap
        under = ~over
        if not under.any() or excess <= 1e-14:
            break
        room = np.maximum(effective_cap - weights[under], 0.0)
        if room.sum() <= 0:
            break
        weights[under] += excess * room / room.sum()
    return weights / weights.sum()


def _prepared_moments(
    history: pd.DataFrame,
    *,
    periods_per_year: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return shrunk annualized means and a diagonally shrunk covariance matrix."""

    numeric = history.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.loc[:, numeric.notna().mean().ge(0.80)]
    if numeric.shape[1] < 2:
        raise ValueError("at least two assets with sufficient history are required")
    numeric = numeric.fillna(numeric.mean())
    means = numeric.mean().to_numpy(dtype=float) * periods_per_year
    common_mean = float(np.mean(means))
    means = 0.50 * means + 0.50 * common_mean
    covariance = numeric.cov().to_numpy(dtype=float) * periods_per_year
    diagonal = np.diag(np.diag(covariance))
    covariance = 0.90 * covariance + 0.10 * diagonal
    covariance += np.eye(len(covariance)) * 1e-10
    return means, covariance


def optimize_weights(
    history: pd.DataFrame,
    *,
    method: str,
    periods_per_year: int,
    max_weight: float,
) -> tuple[pd.Series, bool, str]:
    """Estimate one long-only target portfolio from information in ``history`` only."""

    if method not in METHOD_LABELS:
        raise ValueError(f"unknown portfolio method: {method}")
    valid_columns = history.columns[history.notna().mean().ge(0.80)]
    clean = history.loc[:, valid_columns]
    means, covariance = _prepared_moments(clean, periods_per_year=periods_per_year)
    count = clean.shape[1]
    cap = max(max_weight, 1.0 / count + 1e-10)
    initial = _normalize_with_cap(np.ones(count), cap)
    if method == "equal_weight":
        return pd.Series(initial, index=valid_columns), True, "closed-form equal weight"

    if method == "min_variance":

        def objective(weights: np.ndarray) -> float:
            return float(weights @ covariance @ weights)

    elif method == "max_sharpe":

        def objective(weights: np.ndarray) -> float:
            volatility = float(np.sqrt(max(weights @ covariance @ weights, 1e-16)))
            return -float(weights @ means) / volatility

    else:

        def objective(weights: np.ndarray) -> float:
            portfolio_variance = max(float(weights @ covariance @ weights), 1e-16)
            contributions = weights * (covariance @ weights)
            target = portfolio_variance / count
            return float(np.square(contributions - target).sum() / portfolio_variance**2)

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, cap)] * count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 750, "ftol": 1e-12, "disp": False},
    )
    if result.success and np.isfinite(result.x).all():
        solution = _normalize_with_cap(result.x, cap)
        return pd.Series(solution, index=valid_columns), True, str(result.message)

    if method == "risk_parity":
        fallback = 1.0 / np.sqrt(np.diag(covariance))
    elif method == "max_sharpe":
        fallback = np.clip(means, 0.0, None) / np.diag(covariance)
    else:
        fallback = 1.0 / np.diag(covariance)
    solution = _normalize_with_cap(fallback, cap)
    message = f"solver fallback: {result.message}"
    return pd.Series(solution, index=valid_columns), False, message


def performance_metrics(
    daily_returns: pd.Series,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Calculate annualized return, volatility, Sharpe, CAGR, and drawdown."""

    clean = pd.to_numeric(daily_returns, errors="coerce").dropna()
    if clean.empty:
        raise ValueError("daily_returns must contain observations")
    growth = (1.0 + clean).cumprod()
    drawdown = growth.div(growth.cummax()).sub(1.0)
    annual_return = float(clean.mean() * periods_per_year)
    annual_volatility = float(clean.std(ddof=1) * np.sqrt(periods_per_year))
    elapsed_years = max(
        (pd.Timestamp(clean.index[-1]) - pd.Timestamp(clean.index[0])).days / 365.2425,
        1.0 / periods_per_year,
    )
    terminal = float(growth.iloc[-1])
    return {
        "annualized_return": annual_return,
        "annualized_volatility": annual_volatility,
        "sharpe_ratio": (
            annual_return / annual_volatility if annual_volatility > 0 else np.nan
        ),
        "max_drawdown": float(drawdown.min()),
        "cagr": terminal ** (1.0 / elapsed_years) - 1.0 if terminal > 0 else np.nan,
        "terminal_wealth": terminal,
    }


def rebalance_dates(index: pd.DatetimeIndex, *, window: int) -> pd.DatetimeIndex:
    """Select the first observed date of each month after the estimation window."""

    eligible = pd.DatetimeIndex(index[window:])
    if eligible.empty:
        raise ValueError("estimation window leaves no out-of-sample observations")
    series = pd.Series(eligible, index=eligible)
    return pd.DatetimeIndex(series.groupby(eligible.to_period("M")).first().to_numpy())


def portfolio_path_from_schedule(
    returns: pd.DataFrame,
    weight_schedule: pd.DataFrame,
    *,
    transaction_cost_bps: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply monthly targets, let holdings drift, and deduct realised turnover costs."""

    returns = returns.sort_index()
    schedule = weight_schedule.sort_index().reindex(columns=returns.columns, fill_value=0.0)
    live_dates = returns.index[returns.index >= schedule.index.min()]
    asset_returns = returns.reindex(live_dates).fillna(0.0)
    if not np.allclose(schedule.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("every target-weight row must sum to one")

    turnover = pd.Series(0.0, index=schedule.index, dtype=float)
    current_weights: np.ndarray | None = None
    wealth = 1.0
    peak = 1.0
    rows: list[dict[str, object]] = []
    for date, return_row in asset_returns.iterrows():
        rebalance_turnover = 0.0
        if date in schedule.index:
            target = schedule.loc[date].to_numpy(dtype=float)
            if current_weights is not None:
                rebalance_turnover = 0.5 * float(np.abs(target - current_weights).sum())
            current_weights = target
            turnover.loc[date] = rebalance_turnover
        if current_weights is None:
            raise RuntimeError("a target schedule is required before portfolio returns begin")

        asset_move = return_row.to_numpy(dtype=float)
        gross_return = float(current_weights @ asset_move)
        cost = rebalance_turnover * (transaction_cost_bps / 10_000.0)
        net_return = gross_return - cost
        wealth *= 1.0 + net_return
        peak = max(peak, wealth)
        rows.append(
            {
                "date": date,
                "gross_daily_return": gross_return,
                "transaction_cost": cost,
                "daily_return": net_return,
                "growth_of_1": wealth,
                "drawdown": wealth / peak - 1.0,
            }
        )

        end_values = current_weights * (1.0 + asset_move)
        end_total = float(end_values.sum())
        if not np.isfinite(end_total) or end_total <= 0:
            raise RuntimeError(f"portfolio holdings became invalid on {date:%Y-%m-%d}")
        current_weights = end_values / end_total

    path = pd.DataFrame(rows)
    return path, turnover


def oos_backtest(
    returns: pd.DataFrame,
    *,
    family: str,
    method: str,
    periods_per_year: int,
    estimation_window: int,
    max_weight: float,
    transaction_cost_bps: float = 5.0,
) -> BacktestResult:
    """Run a monthly walk-forward backtest with strictly lagged estimation data."""

    matrix = returns.sort_index().copy()
    matrix = matrix.loc[~matrix.index.duplicated(keep="first")]
    matrix = matrix.dropna(how="all")
    dates = rebalance_dates(pd.DatetimeIndex(matrix.index), window=estimation_window)
    schedules: list[pd.Series] = []
    audit_rows: list[dict[str, object]] = []
    for date in dates:
        position = int(matrix.index.get_loc(date))
        history = matrix.iloc[position - estimation_window : position]
        estimated, success, message = optimize_weights(
            history,
            method=method,
            periods_per_year=periods_per_year,
            max_weight=max_weight,
        )
        full = pd.Series(0.0, index=matrix.columns, name=date)
        full.loc[estimated.index] = estimated
        schedules.append(full)
        audit_rows.append(
            {
                "rebalance_date": date,
                "estimation_start": history.index.min(),
                "estimation_end": history.index.max(),
                "solver_success": success,
                "solver_message": message,
            }
        )

    schedule = pd.DataFrame(schedules)
    schedule.index = dates
    schedule.index.name = "rebalance_date"
    path, turnover = portfolio_path_from_schedule(
        matrix,
        schedule,
        transaction_cost_bps=transaction_cost_bps,
    )
    fund_id = f"{family.lower()}_{method}"
    fund_name = f"{family} {METHOD_LABELS[method]}"
    path = path.assign(
        fund_id=fund_id,
        fund_name=fund_name,
        family=family,
        method=METHOD_LABELS[method],
    )

    audit = pd.DataFrame(audit_rows).set_index("rebalance_date")
    long_weights = (
        schedule.reset_index()
        .melt(id_vars="rebalance_date", var_name="ticker", value_name="weight")
        .merge(audit.reset_index(), on="rebalance_date", validate="many_to_one")
    )
    long_weights = long_weights.loc[long_weights["weight"].gt(1e-10)].copy()
    long_weights["asset_class"] = np.where(
        long_weights["ticker"].str.endswith("-USD"), "Crypto", "Equity"
    )
    long_weights = long_weights.assign(
        fund_id=fund_id,
        fund_name=fund_name,
        family=family,
        method=METHOD_LABELS[method],
    )
    metric_values = performance_metrics(
        path.set_index("date")["daily_return"], periods_per_year=periods_per_year
    )
    metric_values.update(
        {
            "fund_id": fund_id,
            "fund_name": fund_name,
            "family": family,
            "method": METHOD_LABELS[method],
            "start_date": path["date"].min(),
            "end_date": path["date"].max(),
            "observations": len(path),
            "periods_per_year": periods_per_year,
            "estimation_window": estimation_window,
            "rebalance_frequency": "Monthly, first observed date",
            "average_turnover": float(turnover.iloc[1:].mean()),
            "cumulative_turnover": float(turnover.iloc[1:].sum()),
            "transaction_cost_bps": transaction_cost_bps,
            "solver_success_rate": float(audit["solver_success"].mean()),
        }
    )
    return BacktestResult(path, long_weights, metric_values)


def build_fund_suite(
    equity_returns: pd.DataFrame,
    crypto_returns: pd.DataFrame,
    combined_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build 12 equity, crypto, and combined systematic fund products."""

    specifications = (
        ("Equity", equity_returns, 252, 252, 0.15),
        ("Crypto", crypto_returns, 365, 365, 0.35),
        ("Combined", combined_returns, 252, 252, 0.12),
    )
    results: list[BacktestResult] = []
    for family, matrix, periods, window, cap in specifications:
        for method in METHOD_LABELS:
            results.append(
                oos_backtest(
                    matrix,
                    family=family,
                    method=method,
                    periods_per_year=periods,
                    estimation_window=window,
                    max_weight=cap,
                )
            )
    fund_returns = pd.concat([item.returns for item in results], ignore_index=True)
    fund_weights = pd.concat([item.weights for item in results], ignore_index=True)
    metrics = pd.DataFrame([item.metrics for item in results])
    return fund_returns, fund_weights, metrics


def estimation_imputation_summary(
    specifications: dict[str, tuple[pd.DataFrame, int]],
) -> pd.DataFrame:
    """Quantify mean-filled return cells and coverage exclusions in estimation windows."""

    rows: list[dict[str, object]] = []
    for family, (returns, window) in specifications.items():
        matrix = returns.sort_index().apply(pd.to_numeric, errors="coerce")
        dates = rebalance_dates(pd.DatetimeIndex(matrix.index), window=window)
        optimizer_cells = 0
        imputed_cells = 0
        asset_window_exclusions = 0
        eligible_counts: list[int] = []
        window_imputation_shares: list[float] = []
        for date in dates:
            position = int(matrix.index.get_loc(date))
            history = matrix.iloc[position - window : position]
            eligible = history.columns[history.notna().mean().ge(0.80)]
            clean = history.loc[:, eligible]
            missing = int(clean.isna().sum().sum())
            cells = int(clean.size)
            optimizer_cells += cells
            imputed_cells += missing
            asset_window_exclusions += int(history.shape[1] - len(eligible))
            eligible_counts.append(len(eligible))
            window_imputation_shares.append(missing / cells if cells else np.nan)
        rows.append(
            {
                "family": family,
                "rebalances": len(dates),
                "assets_total": matrix.shape[1],
                "minimum_eligible_assets": min(eligible_counts),
                "maximum_eligible_assets": max(eligible_counts),
                "asset_window_exclusions": asset_window_exclusions,
                "optimizer_cells": optimizer_cells,
                "mean_filled_cells": imputed_cells,
                "mean_fill_share": imputed_cells / optimizer_cells,
                "maximum_window_mean_fill_share": float(
                    np.nanmax(window_imputation_shares)
                ),
            }
        )
    return pd.DataFrame(rows)


def combined_target_risk_diagnostics(
    combined_returns: pd.DataFrame,
    fund_weights: pd.DataFrame,
    *,
    estimation_window: int = 252,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Calculate capital mix, covariance risk contribution, and target concentration."""

    matrix = combined_returns.sort_index().apply(pd.to_numeric, errors="coerce")
    weights = fund_weights.loc[fund_weights["family"].eq("Combined")].copy()
    weights["rebalance_date"] = pd.to_datetime(weights["rebalance_date"])
    rows: list[dict[str, object]] = []
    for (fund_id, date), group in weights.groupby(
        ["fund_id", "rebalance_date"], sort=True
    ):
        position = int(matrix.index.get_loc(date))
        history = matrix.iloc[position - estimation_window : position]
        eligible = history.columns[history.notna().mean().ge(0.80)]
        eligible_history = history.loc[:, eligible]
        missing_cells = int(eligible_history.isna().sum().sum())
        prepared = eligible_history.fillna(eligible_history.mean())
        covariance = prepared.cov().to_numpy(dtype=float) * periods_per_year
        covariance = 0.90 * covariance + 0.10 * np.diag(np.diag(covariance))
        covariance += np.eye(len(covariance)) * 1e-10
        target = (
            group.set_index("ticker")["weight"]
            .reindex(eligible, fill_value=0.0)
            .to_numpy(dtype=float)
        )
        portfolio_variance = float(target @ covariance @ target)
        if not np.isfinite(portfolio_variance) or portfolio_variance <= 0:
            raise RuntimeError(f"invalid target variance for {fund_id} on {date:%Y-%m-%d}")
        component_share = target * (covariance @ target) / portfolio_variance
        crypto = np.asarray([ticker.endswith("-USD") for ticker in eligible])
        hhi = float(np.square(target).sum())
        rows.append(
            {
                "fund_id": fund_id,
                "fund_name": str(group["fund_name"].iloc[0]),
                "method": str(group["method"].iloc[0]),
                "rebalance_date": date,
                "equity_capital_weight": float(target[~crypto].sum()),
                "crypto_capital_weight": float(target[crypto].sum()),
                "equity_risk_contribution": float(component_share[~crypto].sum()),
                "crypto_risk_contribution": float(component_share[crypto].sum()),
                "risk_contribution_sum": float(component_share.sum()),
                "ex_ante_annualized_volatility": float(np.sqrt(portfolio_variance)),
                "active_holdings": int(np.count_nonzero(target > 1e-10)),
                "effective_holdings": 1.0 / hhi,
                "top_5_weight": float(np.sort(target)[-5:].sum()),
                "hhi": hhi,
                "mean_filled_cells": missing_cells,
            }
        )
    return pd.DataFrame(rows)


def summarize_combined_risk(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Summarize monthly combined-fund target diagnostics by portfolio method."""

    rows: list[dict[str, object]] = []
    for fund_id, group in diagnostics.groupby("fund_id", sort=True):
        group = group.sort_values("rebalance_date")
        latest = group.iloc[-1]
        rows.append(
            {
                "fund_id": fund_id,
                "fund_name": str(group["fund_name"].iloc[0]),
                "method": str(group["method"].iloc[0]),
                "target_months": len(group),
                "median_crypto_capital_weight": float(
                    group["crypto_capital_weight"].median()
                ),
                "minimum_crypto_capital_weight": float(
                    group["crypto_capital_weight"].min()
                ),
                "maximum_crypto_capital_weight": float(
                    group["crypto_capital_weight"].max()
                ),
                "crypto_capital_weight_std": float(
                    group["crypto_capital_weight"].std(ddof=1)
                ),
                "median_crypto_risk_contribution": float(
                    group["crypto_risk_contribution"].median()
                ),
                "minimum_crypto_risk_contribution": float(
                    group["crypto_risk_contribution"].min()
                ),
                "maximum_crypto_risk_contribution": float(
                    group["crypto_risk_contribution"].max()
                ),
                "median_effective_holdings": float(group["effective_holdings"].median()),
                "median_top_5_weight": float(group["top_5_weight"].median()),
                "median_hhi": float(group["hhi"].median()),
                "latest_crypto_capital_weight": float(latest["crypto_capital_weight"]),
                "latest_crypto_risk_contribution": float(
                    latest["crypto_risk_contribution"]
                ),
                "latest_effective_holdings": float(latest["effective_holdings"]),
                "latest_top_5_weight": float(latest["top_5_weight"]),
            }
        )
    return pd.DataFrame(rows)
