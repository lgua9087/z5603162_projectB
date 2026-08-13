"""High-risk model, artifact, and app smoke tests for Project B."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app_data import (  # noqa: E402
    allocation_lookthrough,
    allocation_path,
    balanced_default_funds,
    equity_sector_exposure,
    latest_sentiment_snapshot,
    load_app_artifacts,
)
from src.diagnostics import (  # noqa: E402
    sentiment_predictive_diagnostics,
    transaction_cost_sensitivity,
)
from src.features import (  # noqa: E402
    align_headlines_to_trading_days,
    combined_returns_on_equity_calendar,
    daily_returns,
)
from src.portfolios import (  # noqa: E402
    combined_target_risk_diagnostics,
    oos_backtest,
    optimize_weights,
    performance_metrics,
    portfolio_path_from_schedule,
)
from src.sentiment import (  # noqa: E402
    compound_normalize,
    score_headlines,
    sector_sentiment_index,
    ticker_signal_panel,
)


def test_daily_returns_stay_within_ticker() -> None:
    prices = pd.DataFrame(
        {
            "ticker": ["A", "A", "B", "B"],
            "date": pd.to_datetime(["2023-01-02", "2023-01-03"] * 2),
            "adjClose": [100.0, 110.0, 50.0, 45.0],
        }
    )
    result = daily_returns(prices)
    first = result.groupby("ticker", sort=False).nth(0)["return"]
    assert first.isna().all()
    assert result.groupby("ticker", sort=False).nth(1)["return"].tolist() == pytest.approx(
        [0.10, -0.10]
    )


def test_crypto_is_differenced_before_equity_calendar_selection() -> None:
    equity = pd.DataFrame(
        {
            "ticker": ["A", "A", "A"],
            "date": pd.to_datetime(["2023-01-06", "2023-01-09", "2023-01-10"]),
            "return": [np.nan, 0.01, 0.02],
        }
    )
    crypto = pd.DataFrame(
        {
            "ticker": ["BTC-USD"] * 5,
            "date": pd.to_datetime(
                ["2023-01-06", "2023-01-07", "2023-01-08", "2023-01-09", "2023-01-10"]
            ),
            "return": [np.nan, 0.10, 0.20, 0.30, 0.40],
        }
    )
    combined = combined_returns_on_equity_calendar(equity, crypto)
    assert combined.loc[pd.Timestamp("2023-01-09"), "BTC-USD"] == pytest.approx(0.30)


def test_holdings_drift_between_monthly_rebalances() -> None:
    dates = pd.to_datetime(["2023-01-02", "2023-01-03"])
    returns = pd.DataFrame({"A": [0.10, 0.00], "B": [0.00, 0.10]}, index=dates)
    schedule = pd.DataFrame({"A": [0.50], "B": [0.50]}, index=dates[:1])
    path, turnover = portfolio_path_from_schedule(
        returns,
        schedule,
        transaction_cost_bps=5.0,
    )
    assert path.loc[0, "gross_daily_return"] == pytest.approx(0.05)
    assert path.loc[1, "gross_daily_return"] == pytest.approx(0.0476190476)
    assert turnover.iloc[0] == 0.0


def test_ticker_signal_lags_tone_and_coverage_together() -> None:
    dates = pd.bdate_range("2023-01-02", periods=6)
    rows = []
    for date in dates:
        rows.extend(
            [
                {
                    "aligned_date": date,
                    "ticker": "A",
                    "sector": "Tech",
                    "title": "gain",
                    "compound": 0.5,
                    "baseline_compound": 0.5,
                    "score_changed": False,
                },
                {
                    "aligned_date": date,
                    "ticker": "B",
                    "sector": "Tech",
                    "title": "loss",
                    "compound": -0.5,
                    "baseline_compound": -0.5,
                    "score_changed": False,
                },
            ]
        )
    signals = ticker_signal_panel(pd.DataFrame(rows), trading_calendar=dates, tickers=["A", "B"])
    early = signals.loc[signals["date"].lt(dates[-1])]
    final = signals.loc[signals["date"].eq(dates[-1])].set_index("ticker")
    assert early["signal_z"].eq(0.0).all()
    assert final.loc["A", "signal_z"] > 0
    assert final.loc["B", "signal_z"] < 0
    assert final["coverage_21d"].eq(5 / 21).all()
    assert final["signal_observations"].eq(5).all()


def test_sentiment_predictive_diagnostic_recovers_rank_signal() -> None:
    dates = pd.bdate_range("2023-01-02", periods=30)
    tickers = [f"T{index:02d}" for index in range(20)]
    scores = np.linspace(-2.0, 2.0, len(tickers))
    signals = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "signal_z": score,
                "coverage_window": 1.0,
                "coverage_21d": 1.0,
                "signal_observations": 21,
            }
            for date in dates
            for ticker, score in zip(tickers, scores, strict=True)
        ]
    )
    returns = pd.DataFrame(
        np.tile(scores * 0.001, (len(dates), 1)),
        index=dates,
        columns=tickers,
    )
    summary, daily = sentiment_predictive_diagnostics(signals, returns)
    assert set(summary["horizon_days"]) == {1, 5}
    assert summary["mean_spearman_ic"].gt(0.99).all()
    assert summary["mean_high_minus_low_return"].gt(0).all()
    assert not daily.empty


def test_headlines_map_same_or_forward_only() -> None:
    headlines = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-06", "2023-01-07", "2023-01-09"]),
            "title": ["a", "b", "c"],
            "ticker": ["A"] * 3,
            "sector": ["Tech"] * 3,
        }
    )
    calendar = pd.to_datetime(["2023-01-06", "2023-01-09"])
    aligned = align_headlines_to_trading_days(headlines, calendar)
    expected = pd.to_datetime(["2023-01-06", "2023-01-09", "2023-01-09"])
    assert aligned["aligned_date"].tolist() == list(expected)
    assert not aligned["aligned_date"].lt(aligned["original_news_date"]).any()


def _synthetic_returns(rows: int = 620) -> pd.DataFrame:
    generator = np.random.default_rng(5545)
    dates = pd.bdate_range("2020-01-02", periods=rows)
    return pd.DataFrame(
        generator.normal(
            loc=[0.0005, 0.0003, 0.0001, 0.0004],
            scale=[0.012, 0.008, 0.006, 0.010],
            size=(rows, 4),
        ),
        index=dates,
        columns=["A", "B", "C", "D"],
    )


def test_optimizer_weights_are_bounded_and_methods_differ() -> None:
    history = _synthetic_returns(300)
    vectors = []
    for method in ("equal_weight", "min_variance", "max_sharpe", "risk_parity"):
        weights, _, _ = optimize_weights(
            history,
            method=method,
            periods_per_year=252,
            max_weight=0.45,
        )
        assert weights.sum() == pytest.approx(1.0)
        assert weights.ge(0).all()
        assert weights.le(0.45 + 1e-9).all()
        vectors.append(tuple(weights.round(5)))
    assert len(set(vectors)) >= 3


def test_walk_forward_estimation_ends_before_live_return() -> None:
    result = oos_backtest(
        _synthetic_returns(),
        family="Equity",
        method="min_variance",
        periods_per_year=252,
        estimation_window=252,
        max_weight=0.45,
    )
    assert result.weights["estimation_end"].lt(result.weights["rebalance_date"]).all()
    assert result.returns["date"].min() > _synthetic_returns().index[251]


def test_combined_risk_diagnostic_reconciles_capital_and_covariance_risk() -> None:
    generator = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-02", periods=260)
    returns = pd.DataFrame(
        {
            "EQUITY": generator.normal(0, 0.01, len(dates)),
            "CRYPTO-USD": generator.normal(0, 0.03, len(dates)),
        },
        index=dates,
    )
    rebalance = dates[252]
    weights = pd.DataFrame(
        {
            "family": ["Combined", "Combined"],
            "fund_id": ["combined_equal_weight"] * 2,
            "fund_name": ["Combined Equal Weight"] * 2,
            "method": ["Equal Weight"] * 2,
            "rebalance_date": [rebalance] * 2,
            "ticker": ["EQUITY", "CRYPTO-USD"],
            "weight": [0.5, 0.5],
        }
    )
    result = combined_target_risk_diagnostics(returns, weights)
    assert result.loc[0, "equity_capital_weight"] == pytest.approx(0.5)
    assert result.loc[0, "crypto_capital_weight"] == pytest.approx(0.5)
    assert result.loc[0, "risk_contribution_sum"] == pytest.approx(1.0)
    assert result.loc[0, "crypto_risk_contribution"] > 0.5


def test_performance_metrics_match_manual_growth() -> None:
    values = pd.Series([0.10, -0.05, 0.02], index=pd.date_range("2023-01-01", periods=3))
    metrics = performance_metrics(values, periods_per_year=365)
    expected_terminal = (1.10 * 0.95 * 1.02)
    assert metrics["terminal_wealth"] == pytest.approx(expected_terminal)
    assert -1 <= metrics["max_drawdown"] <= 0


def test_compound_normalization_is_bounded_and_odd() -> None:
    assert compound_normalize(0) == 0
    assert compound_normalize(4) == pytest.approx(-compound_normalize(-4))
    assert -1 < compound_normalize(-100) < 0
    assert 0 < compound_normalize(100) < 1


def test_finance_extension_changes_a_known_coverage_gap() -> None:
    panel = pd.DataFrame(
        {
            "aligned_date": pd.to_datetime(["2023-01-03", "2023-01-03"]),
            "ticker": ["A", "B"],
            "sector": ["Tech", "Tech"],
            "title": ["Analysts issued a downgrade.", "The firm reduced its debt."],
        }
    )
    result = score_headlines(panel)
    downgrade = result.iloc[0]
    assert downgrade["compound"] < downgrade["baseline_compound"]
    assert result.iloc[1]["score_changed"] == np.False_


def test_sector_index_equal_weights_tickers_not_headlines() -> None:
    scores = pd.DataFrame(
        {
            "aligned_date": pd.to_datetime(["2023-01-03"] * 3),
            "ticker": ["A", "A", "B"],
            "sector": ["Tech"] * 3,
            "title": ["one", "two", "three"],
            "compound": [1.0, 1.0, -1.0],
            "baseline_compound": [1.0, 1.0, -1.0],
            "score_changed": [False, False, False],
        }
    )
    mapping = pd.DataFrame({"ticker": ["A", "B"], "sector": ["Tech", "Tech"]})
    index = sector_sentiment_index(
        scores,
        trading_calendar=pd.to_datetime(["2023-01-03"]),
        ticker_sector=mapping,
    )
    assert index.loc[0, "sentiment"] == pytest.approx(0.0)
    assert index.loc[0, "coverage_rate"] == pytest.approx(1.0)


def test_required_artifacts_and_allocation_contract() -> None:
    artifacts = load_app_artifacts(PROJECT_ROOT)
    assert artifacts.performance_metrics["fund_id"].nunique() == 12
    choices = artifacts.performance_metrics.head(2)["fund_id"].tolist()
    path, metrics = allocation_path(
        artifacts.fund_returns,
        {choices[0]: 60.0, choices[1]: 40.0},
    )
    assert not path.empty
    assert metrics["terminal_wealth"] > 0


def test_balanced_defaults_span_all_asset_families() -> None:
    artifacts = load_app_artifacts(PROJECT_ROOT)
    selected = balanced_default_funds(artifacts.performance_metrics, limit=3)
    families = artifacts.performance_metrics.set_index("fund_id").loc[selected, "family"]
    assert set(families) == {"Equity", "Crypto", "Combined"}


def test_latest_sentiment_snapshot_separates_tone_from_fresh_coverage() -> None:
    artifacts = load_app_artifacts(PROJECT_ROOT)
    snapshot = latest_sentiment_snapshot(artifacts.sector_sentiment)
    assert snapshot["date"] == pd.Timestamp("2023-12-29")
    assert snapshot["tone_label"] == "positive"
    assert snapshot["covered_tickers"] == 1
    assert snapshot["universe_tickers"] == 50
    assert snapshot["coverage_rate"] == pytest.approx(0.02)
    assert snapshot["coverage_label"] == "very thin"


def test_allocation_lookthrough_reveals_overlapping_assets() -> None:
    weights = pd.DataFrame(
        {
            "fund_id": ["f1", "f1", "f2", "f2"],
            "rebalance_date": pd.to_datetime(["2023-12-01"] * 4),
            "ticker": ["NVDA", "AMD", "NVDA", "BTC-USD"],
            "asset_class": ["Equity", "Equity", "Equity", "Crypto"],
            "weight": [0.5, 0.5, 0.5, 0.5],
        }
    )
    lookthrough, metrics = allocation_lookthrough(weights, {"f1": 50, "f2": 50})
    exposures = lookthrough.set_index("ticker")["lookthrough_weight"]
    assert exposures.loc["NVDA"] == pytest.approx(0.5)
    assert exposures.loc["AMD"] == pytest.approx(0.25)
    assert exposures.loc["BTC-USD"] == pytest.approx(0.25)
    assert metrics["equity_weight"] == pytest.approx(0.75)
    assert metrics["crypto_weight"] == pytest.approx(0.25)
    assert metrics["unclassified_equity_weight"] == pytest.approx(0.0)
    sectors = equity_sector_exposure(lookthrough).set_index("sector")
    assert sectors.loc["Technology", "lookthrough_weight"] == pytest.approx(0.75)


def test_all_real_equity_holdings_have_sector_lookthrough() -> None:
    artifacts = load_app_artifacts(PROJECT_ROOT)
    selected = balanced_default_funds(artifacts.performance_metrics, limit=3)
    lookthrough, metrics = allocation_lookthrough(
        artifacts.fund_weights,
        dict.fromkeys(selected, 1.0),
    )
    assert not equity_sector_exposure(lookthrough).empty
    assert metrics["unclassified_equity_weight"] == pytest.approx(0.0)


def test_transaction_cost_sensitivity_is_monotonic() -> None:
    dates = pd.bdate_range("2023-01-02", periods=5)
    frame = pd.DataFrame(
        {
            "date": dates,
            "fund_id": ["fund"] * 5,
            "fund_name": ["Fund"] * 5,
            "family": ["Equity"] * 5,
            "method": ["Rule"] * 5,
            "gross_daily_return": [0.01] * 5,
            "transaction_cost": [0.00025] * 5,
            "daily_return": [0.00975] * 5,
        }
    )
    result = transaction_cost_sensitivity(frame)
    terminal = result.sort_values("transaction_cost_bps")["terminal_wealth"]
    assert terminal.is_monotonic_decreasing


def test_deployed_app_imports_precomputed_artifacts_only() -> None:
    source = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    parsed = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(parsed)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or "" for node in ast.walk(parsed) if isinstance(node, ast.ImportFrom)
    )
    forbidden = ("nltk", "vaderSentiment", "src.data_access")
    assert not any(name.startswith(forbidden) for name in imported)


@pytest.mark.parametrize(
    "view,expected",
    [
        ("marketplace", "Compare the fund range"),
        ("factsheet", "Read a fund fact sheet"),
        ("allocation", "Build a multi-fund allocation"),
        ("sentiment", "Read sector news sentiment"),
        ("method", "Method, assumptions, and limits"),
    ],
)
def test_streamlit_investor_views(view: str, expected: str) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=60)
    app.query_params["view"] = view
    app.run()
    assert not app.exception, app.exception
    rendered = "\n".join(
        str(element.value)
        for collection in (
            app.title,
            app.header,
            app.subheader,
            app.markdown,
            app.caption,
            app.info,
            app.warning,
            app.error,
        )
        for element in collection
    )
    assert "MarketReady Funds" in rendered
    assert expected in rendered
    if view == "allocation":
        assert "Major equity sectors" in rendered
        assert "Top underlying holdings" in rendered
    if view == "sentiment":
        assert "Latest sample reading" in rendered
        assert "Fresh ticker coverage" in rendered
