"""Reproduce all Project B funds, sentiment analytics, fusion, and exhibits."""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics import (  # noqa: E402
    annual_fund_results,
    management_fee_sensitivity,
    sentiment_predictive_diagnostics,
    transaction_cost_sensitivity,
)
from src.etl import (  # noqa: E402
    integrity_summary,
    load_clean_crypto,
    load_clean_equities,
    load_clean_news,
)
from src.features import (  # noqa: E402
    align_headlines_to_trading_days,
    combined_returns_on_equity_calendar,
    daily_returns,
    wide_returns,
)
from src.figures import build_all_figures  # noqa: E402
from src.fusion import build_fusion_comparison  # noqa: E402
from src.portfolios import (  # noqa: E402
    METHOD_LABELS,
    build_fund_suite,
    combined_target_risk_diagnostics,
    estimation_imputation_summary,
    summarize_combined_risk,
)
from src.sentiment import (  # noqa: E402
    finance_lexicon_audit,
    score_headlines,
    sector_sentiment_index,
    sentiment_model_comparison,
    ticker_signal_panel,
)

DATA_DIR = PROJECT_ROOT / "results" / "data"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"failed to create output: {path}")


def _validation_table(
    *,
    equities: pd.DataFrame,
    crypto: pd.DataFrame,
    news: pd.DataFrame,
    aligned_news: pd.DataFrame,
    equity_returns: pd.DataFrame,
    crypto_returns: pd.DataFrame,
    fund_returns: pd.DataFrame,
    fund_weights: pd.DataFrame,
    metrics: pd.DataFrame,
    sentiment_index: pd.DataFrame,
    ticker_signals: pd.DataFrame,
    fusion_weights: pd.DataFrame,
    combined_risk: pd.DataFrame,
    sentiment_predictive: pd.DataFrame,
    annual_results: pd.DataFrame,
    fusion_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    weight_sums = fund_weights.groupby(["fund_id", "rebalance_date"])["weight"].sum()
    method_vectors = (
        fund_weights.pivot_table(
            index=["family", "rebalance_date", "method"],
            columns="ticker",
            values="weight",
            fill_value=0.0,
        )
        .round(6)
        .reset_index()
    )
    latest_vectors = method_vectors.sort_values("rebalance_date").groupby(
        ["family", "method"], as_index=False
    ).tail(1)
    vector_columns = [
        column
        for column in latest_vectors.columns
        if column not in {"family", "rebalance_date", "method"}
    ]
    distinct_by_family = latest_vectors.groupby("family")[vector_columns].apply(
        lambda group: group.drop_duplicates().shape[0]
    )
    fusion_sums = fusion_weights.groupby("rebalance_date")["weight"].sum()
    first_equity_missing = equity_returns.groupby("ticker")["return"].nth(0).isna().all()
    first_crypto_missing = crypto_returns.groupby("ticker")["return"].nth(0).isna().all()
    rows = [
        ("Equity ticker-date keys are unique", not equities.duplicated(["ticker", "date"]).any()),
        ("Crypto ticker-date keys are unique", not crypto.duplicated(["ticker", "date"]).any()),
        (
            "News ticker-date-title keys are unique",
            not news.duplicated(["ticker", "date", "title"]).any(),
        ),
        ("First equity return is missing within ticker", bool(first_equity_missing)),
        ("First crypto return is missing within ticker", bool(first_crypto_missing)),
        (
            "Headline mapping never moves backward",
            not aligned_news["aligned_date"]
            .lt(aligned_news["original_news_date"])
            .fillna(False)
            .any(),
        ),
        ("Twelve fund products are present", fund_returns["fund_id"].nunique() == 12),
        ("Three fund families are present", metrics["family"].nunique() == 3),
        ("Four methods per family are present", len(metrics) == 3 * len(METHOD_LABELS)),
        ("Fund weights sum to one", bool(np.allclose(weight_sums, 1.0, atol=1e-8))),
        ("Fund weights are long-only", bool(fund_weights["weight"].ge(-1e-12).all())),
        (
            "Every estimation sample ends before its rebalance",
            bool(fund_weights["estimation_end"].lt(fund_weights["rebalance_date"]).all()),
        ),
        ("Portfolio methods produce distinct latest weights", bool(distinct_by_family.ge(3).all())),
        (
            "Solver success rate is at least 95%",
            bool(metrics["solver_success_rate"].ge(0.95).all()),
        ),
        ("Ten sector sentiment indices are present", sentiment_index["sector"].nunique() == 10),
        (
            "Sentiment scores remain inside minus one to one",
            bool(sentiment_index["sentiment"].dropna().between(-1.0, 1.0).all()),
        ),
        (
            "Ticker signal dates cover the usable equity-return calendar",
            ticker_signals["date"].nunique()
            == equity_returns.loc[equity_returns["return"].notna(), "date"].nunique(),
        ),
        ("Fusion weights sum to one", bool(np.allclose(fusion_sums, 1.0, atol=1e-8))),
        (
            "Combined capital weights sum to one",
            bool(
                np.allclose(
                    combined_risk["equity_capital_weight"]
                    + combined_risk["crypto_capital_weight"],
                    1.0,
                    atol=1e-8,
                )
            ),
        ),
        (
            "Combined covariance risk contributions sum to one",
            bool(np.allclose(combined_risk["risk_contribution_sum"], 1.0, atol=1e-8)),
        ),
        (
            "Sentiment prediction tests cover one and five trading days",
            set(sentiment_predictive["horizon_days"]) == {1, 5},
        ),
        (
            "Annual fund results cover every fund in 2021 through 2023",
            len(annual_results) == 12 * 3
            and set(annual_results["year"]) == {2021, 2022, 2023},
        ),
        (
            "Fusion robustness includes base case and two prespecified alternatives",
            len(fusion_sensitivity) == 4
            and int(fusion_sensitivity["is_primary_case"].sum()) == 1,
        ),
    ]
    return pd.DataFrame(rows, columns=["check", "passed"])


def _summary_json(
    *,
    metrics: pd.DataFrame,
    model_comparison: pd.DataFrame,
    sector_index: pd.DataFrame,
    fusion_comparison: pd.DataFrame,
    validation: pd.DataFrame,
    combined_risk_summary: pd.DataFrame,
    sentiment_predictive: pd.DataFrame,
    imputation: pd.DataFrame,
    fee_sensitivity: pd.DataFrame,
) -> dict[str, object]:
    best = metrics.loc[metrics["sharpe_ratio"].idxmax()]
    base = fusion_comparison.set_index("strategy").loc["Base equity risk parity"]
    tilt = fusion_comparison.set_index("strategy").loc["Coverage-aware sentiment tilt"]
    model = model_comparison.set_index("model")
    risk = combined_risk_summary.set_index("fund_id")
    predictive = sentiment_predictive.set_index("horizon_days")
    fee = fee_sensitivity.loc[
        fee_sensitivity["fund_id"].eq("combined_risk_parity")
    ].set_index("annual_management_fee")
    return {
        "product": "MarketReady Funds",
        "source_sample": "2020-01-01 to 2023-12-31",
        "fund_count": int(metrics["fund_id"].nunique()),
        "first_live_date": str(pd.to_datetime(metrics["start_date"]).min().date()),
        "last_live_date": str(pd.to_datetime(metrics["end_date"]).max().date()),
        "best_sharpe_fund": str(best["fund_name"]),
        "best_sharpe_ratio": float(best["sharpe_ratio"]),
        "best_fund_annualized_return": float(best["annualized_return"]),
        "best_fund_max_drawdown": float(best["max_drawdown"]),
        "plain_vader_neutral_share": float(model.loc["Plain VADER 3.3.2", "neutral_share"]),
        "extended_vader_neutral_share": float(
            model.loc["Finance-extended VADER 3.3.2", "neutral_share"]
        ),
        "headline_score_changed_share": float(
            model.loc["Finance-extended VADER 3.3.2", "changed_from_plain_share"]
        ),
        "mean_sector_ticker_coverage": float(sector_index["coverage_rate"].mean()),
        "fusion_sharpe_change": float(tilt["sharpe_ratio"] - base["sharpe_ratio"]),
        "fusion_return_change": float(tilt["annualized_return"] - base["annualized_return"]),
        "fusion_drawdown_change": float(tilt["max_drawdown"] - base["max_drawdown"]),
        "sentiment_one_day_ic": float(predictive.loc[1, "mean_spearman_ic"]),
        "sentiment_five_day_ic": float(predictive.loc[5, "mean_spearman_ic"]),
        "sentiment_one_day_high_low": float(
            predictive.loc[1, "mean_high_minus_low_return"]
        ),
        "sentiment_five_day_high_low": float(
            predictive.loc[5, "mean_high_minus_low_return"]
        ),
        "combined_risk_parity_median_crypto_capital": float(
            risk.loc["combined_risk_parity", "median_crypto_capital_weight"]
        ),
        "combined_risk_parity_median_crypto_risk": float(
            risk.loc["combined_risk_parity", "median_crypto_risk_contribution"]
        ),
        "combined_max_sharpe_median_effective_holdings": float(
            risk.loc["combined_max_sharpe", "median_effective_holdings"]
        ),
        "optimizer_mean_filled_cells": int(imputation["mean_filled_cells"].sum()),
        "combined_risk_parity_return_after_one_percent_fee": float(
            fee.loc[0.01, "annualized_return"]
        ),
        "validation_checks": len(validation),
        "validation_passes": int(validation["passed"].sum()),
        "vader_version": importlib.metadata.version("vaderSentiment"),
        "risk_free_rate": 0.0,
        "transaction_cost_bps_per_unit_turnover": 5.0,
    }


def main() -> None:
    for directory in (DATA_DIR, TABLE_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading and cleaning the three supplied datasets", flush=True)
    equities, equity_meta = load_clean_equities()
    crypto, crypto_meta = load_clean_crypto()
    news, news_meta = load_clean_news()
    source_integrity = integrity_summary(
        equities,
        crypto,
        news,
        equity_meta=equity_meta,
        crypto_meta=crypto_meta,
        news_meta=news_meta,
    )

    print("[2/7] Computing native-calendar and combined return panels", flush=True)
    equity_long = daily_returns(equities)
    crypto_long = daily_returns(crypto)
    equity_wide = wide_returns(equity_long).iloc[1:]
    crypto_wide = wide_returns(crypto_long).iloc[1:]
    combined_wide = combined_returns_on_equity_calendar(equity_long, crypto_long).iloc[1:]

    print("[3/7] Running 12 monthly walk-forward fund backtests", flush=True)
    fund_returns, fund_weights, metrics = build_fund_suite(
        equity_wide,
        crypto_wide,
        combined_wide,
    )
    annual_results = annual_fund_results(fund_returns)
    cost_sensitivity = transaction_cost_sensitivity(fund_returns)
    fee_sensitivity = management_fee_sensitivity(fund_returns)
    combined_risk = combined_target_risk_diagnostics(combined_wide, fund_weights)
    combined_risk_summary = summarize_combined_risk(combined_risk)
    imputation = estimation_imputation_summary(
        {
            "Equity": (equity_wide, 252),
            "Crypto": (crypto_wide, 365),
            "Combined": (combined_wide, 252),
        }
    )

    print("[4/7] Scoring untouched headlines and building sector indices", flush=True)
    aligned_news = align_headlines_to_trading_days(news, equity_wide.index)
    usable_news = aligned_news.dropna(subset=["aligned_date"]).copy()
    scored = score_headlines(usable_news)
    model_comparison = sentiment_model_comparison(scored)
    ticker_sector = equities[["ticker", "sector"]].drop_duplicates("ticker")
    sector_index = sector_sentiment_index(
        scored,
        trading_calendar=equity_wide.index,
        ticker_sector=ticker_sector,
    )
    ticker_signals = ticker_signal_panel(
        scored,
        trading_calendar=equity_wide.index,
        tickers=equity_wide.columns,
    )
    equity_live_start = pd.to_datetime(
        fund_returns.loc[fund_returns["family"].eq("Equity"), "date"]
    ).min()
    live_signals = ticker_signals.loc[
        pd.to_datetime(ticker_signals["date"]).ge(equity_live_start)
    ]
    sentiment_predictive, sentiment_predictive_daily = sentiment_predictive_diagnostics(
        live_signals,
        equity_wide.loc[equity_wide.index >= equity_live_start],
    )

    print("[5/7] Applying the lagged, coverage-aware sentiment tilt", flush=True)
    base_weights = fund_weights.loc[fund_weights["fund_id"].eq("equity_risk_parity")]
    fusion_returns, fusion_comparison, fusion_weights = build_fusion_comparison(
        equity_wide,
        base_weights,
        ticker_signals,
    )
    _, lower_tilt_comparison, _ = build_fusion_comparison(
        equity_wide,
        base_weights,
        ticker_signals,
        tilt_strength=0.15,
        tilt_strategy_name="21-day signal, 0.15 tilt",
    )
    slow_signals = ticker_signal_panel(
        scored,
        trading_calendar=equity_wide.index,
        tickers=equity_wide.columns,
        lookback=42,
        minimum_observations=5,
    )
    _, slow_tilt_comparison, _ = build_fusion_comparison(
        equity_wide,
        base_weights,
        slow_signals,
        tilt_strength=0.30,
        signal_lookback=42,
        tilt_strategy_name="42-day signal, 0.30 tilt",
    )
    fusion_sensitivity = pd.concat(
        [
            fusion_comparison.loc[
                fusion_comparison["strategy"].eq("Base equity risk parity")
            ].assign(scenario="Base equity risk parity", is_primary_case=False),
            lower_tilt_comparison.loc[
                lower_tilt_comparison["strategy"].eq("21-day signal, 0.15 tilt")
            ].assign(scenario="21-day signal, 0.15 tilt", is_primary_case=False),
            fusion_comparison.loc[
                fusion_comparison["strategy"].eq("Coverage-aware sentiment tilt")
            ].assign(
                scenario="21-day signal, 0.30 tilt (primary)",
                is_primary_case=True,
            ),
            slow_tilt_comparison.loc[
                slow_tilt_comparison["strategy"].eq("42-day signal, 0.30 tilt")
            ].assign(scenario="42-day signal, 0.30 tilt", is_primary_case=False),
        ],
        ignore_index=True,
    )

    coverage_summary = (
        sector_index.groupby("sector", observed=True)
        .agg(
            mean_ticker_coverage=("coverage_rate", "mean"),
            no_news_day_share=("ticker_count", lambda values: values.eq(0).mean()),
            mean_daily_sentiment=("sentiment", "mean"),
            sentiment_volatility=("sentiment", "std"),
            total_headlines=("headline_count", "sum"),
        )
        .reset_index()
        .sort_values("mean_ticker_coverage", ascending=False)
    )
    validation = _validation_table(
        equities=equities,
        crypto=crypto,
        news=news,
        aligned_news=aligned_news,
        equity_returns=equity_long,
        crypto_returns=crypto_long,
        fund_returns=fund_returns,
        fund_weights=fund_weights,
        metrics=metrics,
        sentiment_index=sector_index,
        ticker_signals=ticker_signals,
        fusion_weights=fusion_weights,
        combined_risk=combined_risk,
        sentiment_predictive=sentiment_predictive,
        annual_results=annual_results,
        fusion_sensitivity=fusion_sensitivity,
    )
    if not validation["passed"].all():
        failed = validation.loc[~validation["passed"], "check"].tolist()
        raise AssertionError(f"critical validation failed: {failed}")

    print("[6/7] Writing app artifacts and diagnostic tables", flush=True)
    _write_csv(fund_returns, DATA_DIR / "fund_returns.csv")
    _write_csv(fund_weights, DATA_DIR / "fund_weights.csv")
    _write_csv(sector_index, DATA_DIR / "sector_sentiment_index.csv")
    _write_csv(fusion_returns, DATA_DIR / "fusion_returns.csv")
    _write_csv(fusion_weights, DATA_DIR / "fusion_weights.csv")
    _write_csv(ticker_signals, DATA_DIR / "ticker_sentiment_signals.csv")
    _write_csv(combined_risk, DATA_DIR / "combined_risk_diagnostics.csv")
    _write_csv(
        sentiment_predictive_daily,
        DATA_DIR / "sentiment_predictive_daily.csv",
    )
    _write_csv(metrics, TABLE_DIR / "performance_metrics.csv")
    _write_csv(fusion_comparison, TABLE_DIR / "fusion_comparison.csv")
    _write_csv(model_comparison, TABLE_DIR / "sentiment_model_comparison.csv")
    _write_csv(finance_lexicon_audit(), TABLE_DIR / "finance_lexicon_audit.csv")
    _write_csv(coverage_summary, TABLE_DIR / "sentiment_coverage_summary.csv")
    _write_csv(source_integrity, TABLE_DIR / "data_integrity_summary.csv")
    _write_csv(validation, TABLE_DIR / "validation_results.csv")
    _write_csv(combined_risk_summary, TABLE_DIR / "combined_risk_summary.csv")
    _write_csv(
        sentiment_predictive,
        TABLE_DIR / "sentiment_predictive_diagnostics.csv",
    )
    _write_csv(annual_results, TABLE_DIR / "annual_fund_results.csv")
    _write_csv(cost_sensitivity, TABLE_DIR / "transaction_cost_sensitivity.csv")
    _write_csv(fee_sensitivity, TABLE_DIR / "management_fee_sensitivity.csv")
    _write_csv(fusion_sensitivity, TABLE_DIR / "fusion_parameter_sensitivity.csv")
    _write_csv(imputation, TABLE_DIR / "estimation_imputation_summary.csv")
    print("[7/7] Rebuilding required report figures and summary", flush=True)
    build_all_figures(
        fund_returns,
        fund_weights,
        metrics,
        sector_index,
        fusion_returns,
        FIGURE_DIR,
    )
    summary = _summary_json(
        metrics=metrics,
        model_comparison=model_comparison,
        sector_index=sector_index,
        fusion_comparison=fusion_comparison,
        validation=validation,
        combined_risk_summary=combined_risk_summary,
        sentiment_predictive=sentiment_predictive,
        imputation=imputation,
        fee_sensitivity=fee_sensitivity,
    )
    (DATA_DIR / "report_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"Complete: {len(metrics)} funds, {len(scored):,} scored headlines, "
        f"{int(validation['passed'].sum())}/{len(validation)} validations passed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
