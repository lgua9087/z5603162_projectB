"""Coverage-aware fusion of lagged ticker sentiment with equity fund weights."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolios import (
    _normalize_with_cap,
    performance_metrics,
    portfolio_path_from_schedule,
)


def apply_sentiment(
    weights: pd.DataFrame,
    sentiment: pd.DataFrame,
    *,
    tilt_strength: float = 0.30,
    max_weight: float = 0.15,
) -> pd.DataFrame:
    """Tilt dated base targets with signals already lagged by one trading day."""

    required_weights = {"rebalance_date", "ticker", "weight"}
    coverage_column = (
        "coverage_window" if "coverage_window" in sentiment.columns else "coverage_21d"
    )
    required_sentiment = {"date", "ticker", "signal_z", coverage_column}
    if missing := sorted(required_weights.difference(weights.columns)):
        raise ValueError(f"weights are missing columns: {missing}")
    if missing := sorted(required_sentiment.difference(sentiment.columns)):
        raise ValueError(f"sentiment is missing columns: {missing}")

    base = weights.pivot(index="rebalance_date", columns="ticker", values="weight")
    base = base.sort_index().fillna(0.0)
    signals = sentiment.pivot(index="date", columns="ticker", values="signal_z")
    coverage = sentiment.pivot(index="date", columns="ticker", values=coverage_column)
    signals = signals.reindex(index=base.index, columns=base.columns).fillna(0.0)
    coverage = coverage.reindex(index=base.index, columns=base.columns).fillna(0.0)

    tilted_rows: list[pd.Series] = []
    for date in base.index:
        base_row = base.loc[date].astype(float)
        effective_signal = signals.loc[date].astype(float) * np.sqrt(
            coverage.loc[date].astype(float).clip(0.0, 1.0)
        )
        multiplier = np.exp(tilt_strength * effective_signal.clip(-2.5, 2.5))
        raw = base_row.to_numpy() * multiplier.to_numpy()
        target = _normalize_with_cap(raw, max_weight)
        tilted_rows.append(pd.Series(target, index=base.columns, name=date))
    schedule = pd.DataFrame(tilted_rows)
    schedule.index.name = "rebalance_date"
    return (
        schedule.reset_index()
        .melt(id_vars="rebalance_date", var_name="ticker", value_name="weight")
        .loc[lambda frame: frame["weight"].gt(1e-10)]
        .sort_values(["rebalance_date", "ticker"])
        .reset_index(drop=True)
    )


def build_fusion_comparison(
    equity_returns: pd.DataFrame,
    base_weights: pd.DataFrame,
    ticker_signals: pd.DataFrame,
    *,
    transaction_cost_bps: float = 5.0,
    tilt_strength: float = 0.30,
    signal_lookback: int = 21,
    tilt_strategy_name: str = "Coverage-aware sentiment tilt",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare a base equity risk-parity fund with its sentiment-tilted version."""

    base_schedule = base_weights.pivot(
        index="rebalance_date", columns="ticker", values="weight"
    ).sort_index()
    tilted_long = apply_sentiment(
        base_weights,
        ticker_signals,
        tilt_strength=tilt_strength,
    )
    tilted_schedule = tilted_long.pivot(
        index="rebalance_date", columns="ticker", values="weight"
    ).sort_index()
    base_path, base_turnover = portfolio_path_from_schedule(
        equity_returns,
        base_schedule,
        transaction_cost_bps=transaction_cost_bps,
    )
    tilted_path, tilted_turnover = portfolio_path_from_schedule(
        equity_returns,
        tilted_schedule,
        transaction_cost_bps=transaction_cost_bps,
    )
    names = {
        "Base equity risk parity": base_path,
        tilt_strategy_name: tilted_path,
    }
    paths = pd.concat(
        [frame.assign(strategy=name) for name, frame in names.items()],
        ignore_index=True,
    )
    metrics_rows = []
    for name, frame, turnover in (
        ("Base equity risk parity", base_path, base_turnover),
        (tilt_strategy_name, tilted_path, tilted_turnover),
    ):
        metrics = performance_metrics(
            frame.set_index("date")["daily_return"], periods_per_year=252
        )
        gross_metrics = performance_metrics(
            frame.set_index("date")["gross_daily_return"], periods_per_year=252
        )
        metrics.update(
            {
                "strategy": name,
                "average_turnover": float(turnover.iloc[1:].mean()),
                "cumulative_turnover": float(turnover.iloc[1:].sum()),
                "transaction_cost_bps": transaction_cost_bps,
                "tilt_strength": 0.0 if name.startswith("Base") else tilt_strength,
                "signal_lookback": 0 if name.startswith("Base") else signal_lookback,
                "gross_annualized_return": gross_metrics["annualized_return"],
                "annualized_trading_cost_drag": (
                    gross_metrics["annualized_return"] - metrics["annualized_return"]
                ),
                "cumulative_transaction_cost": float(frame["transaction_cost"].sum()),
            }
        )
        metrics_rows.append(metrics)
    comparison = pd.DataFrame(metrics_rows)
    tilted_long["strategy"] = tilt_strategy_name
    return paths, comparison, tilted_long
