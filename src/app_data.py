"""Pure data contracts and calculations for the MarketReady Funds app."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FAMILY_DISPLAY_ORDER = ("Equity", "Combined", "Crypto")
EQUITY_SECTOR_BY_TICKER = {
    "NVDA": "Technology",
    "AMD": "Technology",
    "INTC": "Technology",
    "QCOM": "Technology",
    "ADBE": "Technology",
    "GS": "Financials",
    "MS": "Financials",
    "WFC": "Financials",
    "V": "Financials",
    "USB": "Financials",
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "SLB": "Energy",
    "OXY": "Energy",
    "DIS": "Consumer",
    "WMT": "Consumer",
    "NKE": "Consumer",
    "SBUX": "Consumer",
    "KO": "Consumer",
    "GE": "Industrials",
    "BA": "Industrials",
    "CAT": "Industrials",
    "UPS": "Industrials",
    "MMM": "Industrials",
    "MRK": "Healthcare",
    "ABBV": "Healthcare",
    "AMGN": "Healthcare",
    "GILD": "Healthcare",
    "ABT": "Healthcare",
    "T": "Comm/Telecom",
    "CMCSA": "Comm/Telecom",
    "TMUS": "Comm/Telecom",
    "EA": "Comm/Telecom",
    "TTWO": "Comm/Telecom",
    "SHW": "Materials",
    "NEM": "Materials",
    "DOW": "Materials",
    "NUE": "Materials",
    "DD": "Materials",
    "NEE": "Utilities",
    "DUK": "Utilities",
    "SO": "Utilities",
    "D": "Utilities",
    "AEP": "Utilities",
    "AMT": "Real Estate",
    "O": "Real Estate",
    "PLD": "Real Estate",
    "CCI": "Real Estate",
    "PSA": "Real Estate",
}


class ArtifactError(RuntimeError):
    """Raised when a required precomputed app artifact is unavailable or invalid."""


@dataclass(frozen=True)
class AppArtifacts:
    """Validated precomputed inputs used by the deployed app."""

    fund_returns: pd.DataFrame
    fund_weights: pd.DataFrame
    performance_metrics: pd.DataFrame
    sector_sentiment: pd.DataFrame
    fusion_comparison: pd.DataFrame
    fusion_returns: pd.DataFrame
    sentiment_model_comparison: pd.DataFrame
    combined_risk: pd.DataFrame
    summary: dict[str, object]


def _read_csv(path: Path, *, dates: tuple[str, ...] = ()) -> pd.DataFrame:
    if not path.is_file():
        raise ArtifactError(f"Required artifact is missing: {path.relative_to(path.parents[2])}")
    frame = pd.read_csv(path)
    for column in dates:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def _require_columns(frame: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ArtifactError(f"{name} is missing required columns: {missing}")


def load_app_artifacts(project_root: str | Path | None = None) -> AppArtifacts:
    """Load and validate the committed results used by Streamlit."""

    root = PROJECT_ROOT if project_root is None else Path(project_root)
    data_dir = root / "results" / "data"
    table_dir = root / "results" / "tables"

    fund_returns = _read_csv(data_dir / "fund_returns.csv", dates=("date",))
    fund_weights = _read_csv(
        data_dir / "fund_weights.csv", dates=("rebalance_date",)
    )
    performance = _read_csv(
        table_dir / "performance_metrics.csv", dates=("start_date", "end_date")
    )
    sentiment = _read_csv(
        data_dir / "sector_sentiment_index.csv", dates=("date",)
    )
    fusion_comparison = _read_csv(table_dir / "fusion_comparison.csv")
    fusion_returns = _read_csv(data_dir / "fusion_returns.csv", dates=("date",))
    model_comparison = _read_csv(table_dir / "sentiment_model_comparison.csv")
    combined_risk = _read_csv(
        data_dir / "combined_risk_diagnostics.csv", dates=("rebalance_date",)
    )

    _require_columns(
        fund_returns,
        {"date", "fund_id", "fund_name", "family", "method", "daily_return"},
        name="fund_returns.csv",
    )
    _require_columns(
        fund_weights,
        {
            "rebalance_date",
            "fund_id",
            "fund_name",
            "family",
            "method",
            "ticker",
            "asset_class",
            "weight",
        },
        name="fund_weights.csv",
    )
    _require_columns(
        performance,
        {
            "fund_id",
            "fund_name",
            "family",
            "method",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
        },
        name="performance_metrics.csv",
    )
    _require_columns(
        sentiment,
        {
            "date",
            "sector",
            "sentiment",
            "sentiment_21d",
            "coverage_rate",
            "ticker_count",
            "sector_ticker_count",
            "headline_count",
        },
        name="sector_sentiment_index.csv",
    )
    _require_columns(
        fusion_returns,
        {"date", "strategy", "daily_return", "growth_of_1", "drawdown"},
        name="fusion_returns.csv",
    )
    _require_columns(
        combined_risk,
        {
            "rebalance_date",
            "fund_id",
            "crypto_capital_weight",
            "crypto_risk_contribution",
            "effective_holdings",
            "top_5_weight",
        },
        name="combined_risk_diagnostics.csv",
    )

    for frame, date_column in (
        (fund_returns, "date"),
        (fund_weights, "rebalance_date"),
        (sentiment, "date"),
        (fusion_returns, "date"),
        (combined_risk, "rebalance_date"),
    ):
        frame.dropna(subset=[date_column], inplace=True)
        frame.sort_values(date_column, inplace=True, ignore_index=True)

    summary_path = data_dir / "report_summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    return AppArtifacts(
        fund_returns=fund_returns,
        fund_weights=fund_weights,
        performance_metrics=performance,
        sector_sentiment=sentiment,
        fusion_comparison=fusion_comparison,
        fusion_returns=fusion_returns,
        sentiment_model_comparison=model_comparison,
        combined_risk=combined_risk,
        summary=summary,
    )


def balanced_default_funds(metrics: pd.DataFrame, *, limit: int = 3) -> list[str]:
    """Select the strongest available fund from each family before filling slots."""

    if limit <= 0 or metrics.empty:
        return []
    _require_columns(
        metrics,
        {"fund_id", "family", "sharpe_ratio"},
        name="performance metrics",
    )
    ranked = (
        metrics.drop_duplicates("fund_id")
        .sort_values(["sharpe_ratio", "fund_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    available_families = ranked["family"].dropna().astype(str).unique().tolist()
    family_order = [
        family for family in FAMILY_DISPLAY_ORDER if family in available_families
    ]
    family_order.extend(
        sorted(set(available_families).difference(FAMILY_DISPLAY_ORDER))
    )
    selected: list[str] = []
    for family in family_order:
        candidate = ranked.loc[ranked["family"].eq(family), "fund_id"]
        if not candidate.empty:
            selected.append(str(candidate.iloc[0]))
        if len(selected) == limit:
            return selected
    for fund_id in ranked["fund_id"].astype(str):
        if fund_id not in selected:
            selected.append(fund_id)
        if len(selected) == limit:
            break
    return selected


def latest_sentiment_snapshot(sentiment: pd.DataFrame) -> dict[str, object]:
    """Summarise latest tone and fresh-news breadth for investor-facing copy."""

    _require_columns(
        sentiment,
        {
            "date",
            "sentiment_21d",
            "ticker_count",
            "sector_ticker_count",
            "headline_count",
        },
        name="sector sentiment",
    )
    if sentiment.empty:
        raise ValueError("Sector sentiment is empty.")
    latest_date = pd.to_datetime(sentiment["date"], errors="coerce").max()
    latest = sentiment.loc[pd.to_datetime(sentiment["date"]).eq(latest_date)]
    score = float(pd.to_numeric(latest["sentiment_21d"], errors="coerce").mean())
    covered = int(pd.to_numeric(latest["ticker_count"], errors="coerce").fillna(0).sum())
    universe = int(
        pd.to_numeric(latest["sector_ticker_count"], errors="coerce").fillna(0).sum()
    )
    coverage = covered / universe if universe > 0 else float("nan")
    if score > 0.05:
        tone_label = "positive"
    elif score < -0.05:
        tone_label = "negative"
    else:
        tone_label = "near neutral"
    if coverage >= 0.75:
        coverage_label = "broad"
    elif coverage >= 0.50:
        coverage_label = "moderate"
    elif coverage >= 0.25:
        coverage_label = "limited"
    elif coverage >= 0.10:
        coverage_label = "thin"
    else:
        coverage_label = "very thin"
    return {
        "date": latest_date,
        "score_21d": score,
        "tone_label": tone_label,
        "covered_tickers": covered,
        "universe_tickers": universe,
        "coverage_rate": coverage,
        "coverage_label": coverage_label,
        "headline_count": int(
            pd.to_numeric(latest["headline_count"], errors="coerce").fillna(0).sum()
        ),
    }


def current_holdings(weights: pd.DataFrame, fund_id: str) -> pd.DataFrame:
    """Return the most recent target holdings for one fund."""

    selected = weights.loc[weights["fund_id"].eq(fund_id)].copy()
    if selected.empty:
        return selected
    latest = selected["rebalance_date"].max()
    return (
        selected.loc[selected["rebalance_date"].eq(latest)]
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )


def growth_and_drawdown(returns: pd.Series) -> pd.DataFrame:
    """Build a wealth index and drawdown path from simple daily returns."""

    clean = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    growth = (1.0 + clean).cumprod()
    return pd.DataFrame(
        {
            "daily_return": clean,
            "growth_of_1": growth,
            "drawdown": growth.div(growth.cummax()).sub(1.0),
        },
        index=returns.index,
    )


def infer_periods_per_year(dates: pd.DatetimeIndex) -> int:
    """Use 365 for seven-day series and 252 for equity-trading-day series."""

    normalized = pd.DatetimeIndex(pd.to_datetime(dates)).dropna().unique()
    return 365 if any(day.weekday() >= 5 for day in normalized) else 252


def performance_from_returns(
    returns: pd.Series,
    *,
    periods_per_year: int,
) -> dict[str, float]:
    """Calculate app-facing performance measures from daily simple returns."""

    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return {
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
            "terminal_wealth": np.nan,
        }
    path = growth_and_drawdown(clean)
    annualized_return = float(clean.mean() * periods_per_year)
    annualized_volatility = float(clean.std(ddof=1) * np.sqrt(periods_per_year))
    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": (
            annualized_return / annualized_volatility
            if annualized_volatility > 0
            else np.nan
        ),
        "max_drawdown": float(path["drawdown"].min()),
        "terminal_wealth": float(path["growth_of_1"].iloc[-1]),
    }


def allocation_path(
    fund_returns: pd.DataFrame,
    allocations: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Combine selected fund returns using normalized, constant fund allocations."""

    positive = {key: max(float(value), 0.0) for key, value in allocations.items()}
    total = sum(positive.values())
    if total <= 0:
        raise ValueError("At least one fund allocation must be positive.")
    normalized = {key: value / total for key, value in positive.items()}
    selected = fund_returns.loc[fund_returns["fund_id"].isin(normalized)].copy()
    wide = selected.pivot(index="date", columns="fund_id", values="daily_return")
    wide = wide.reindex(columns=list(normalized)).sort_index().fillna(0.0)
    portfolio_returns = wide.mul(pd.Series(normalized)).sum(axis=1)
    path = growth_and_drawdown(portfolio_returns).reset_index()
    periods = infer_periods_per_year(pd.DatetimeIndex(path["date"]))
    metrics = performance_from_returns(
        path.set_index("date")["daily_return"], periods_per_year=periods
    )
    metrics["periods_per_year"] = float(periods)
    return path, metrics


def allocation_lookthrough(
    fund_weights: pd.DataFrame,
    allocations: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Aggregate the latest fund targets into investor-level ticker exposures."""

    positive = {key: max(float(value), 0.0) for key, value in allocations.items()}
    total = sum(positive.values())
    if total <= 0:
        raise ValueError("At least one fund allocation must be positive.")
    normalized = pd.Series({key: value / total for key, value in positive.items()})
    selected = fund_weights.loc[fund_weights["fund_id"].isin(normalized.index)].copy()
    if selected.empty:
        raise ValueError("No target holdings match the selected fund allocation.")
    latest_dates = selected.groupby("fund_id")["rebalance_date"].max().rename("latest_date")
    selected = selected.merge(latest_dates, on="fund_id", validate="many_to_one")
    selected = selected.loc[selected["rebalance_date"].eq(selected["latest_date"])].copy()
    selected["fund_allocation"] = selected["fund_id"].map(normalized)
    selected["lookthrough_weight"] = selected["weight"] * selected["fund_allocation"]
    lookthrough = (
        selected.groupby(["ticker", "asset_class"], as_index=False, observed=True)[
            "lookthrough_weight"
        ]
        .sum()
        .sort_values("lookthrough_weight", ascending=False)
        .reset_index(drop=True)
    )
    lookthrough["lookthrough_weight"] /= lookthrough["lookthrough_weight"].sum()
    lookthrough["sector"] = lookthrough["ticker"].map(EQUITY_SECTOR_BY_TICKER)
    lookthrough.loc[lookthrough["asset_class"].eq("Crypto"), "sector"] = "Crypto"
    lookthrough["sector"] = lookthrough["sector"].fillna("Other equity")
    hhi = float(np.square(lookthrough["lookthrough_weight"]).sum())
    equity_weight = float(
        lookthrough.loc[
            lookthrough["asset_class"].eq("Equity"), "lookthrough_weight"
        ].sum()
    )
    crypto_weight = float(
        lookthrough.loc[
            lookthrough["asset_class"].eq("Crypto"), "lookthrough_weight"
        ].sum()
    )
    metrics = {
        "effective_holdings": 1.0 / hhi,
        "top_5_weight": float(lookthrough["lookthrough_weight"].nlargest(5).sum()),
        "equity_weight": equity_weight,
        "crypto_weight": crypto_weight,
        "unclassified_equity_weight": float(
            lookthrough.loc[
                lookthrough["sector"].eq("Other equity"), "lookthrough_weight"
            ].sum()
        ),
    }
    return lookthrough, metrics


def equity_sector_exposure(lookthrough: pd.DataFrame) -> pd.DataFrame:
    """Aggregate equity sectors as shares of the full multi-fund allocation."""

    _require_columns(
        lookthrough,
        {"asset_class", "sector", "lookthrough_weight"},
        name="allocation look-through",
    )
    equity = lookthrough.loc[lookthrough["asset_class"].eq("Equity")]
    if equity.empty:
        return pd.DataFrame(columns=["sector", "lookthrough_weight"])
    return (
        equity.groupby("sector", as_index=False, observed=True)["lookthrough_weight"]
        .sum()
        .sort_values(["lookthrough_weight", "sector"], ascending=[False, True])
        .reset_index(drop=True)
    )


def presentation_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create a compact comparison table with client-facing column labels."""

    columns = [
        "fund_name",
        "family",
        "method",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "average_turnover",
    ]
    available = [column for column in columns if column in metrics.columns]
    return metrics.loc[:, available].rename(
        columns={
            "fund_name": "Fund",
            "family": "Asset family",
            "method": "Method",
            "annualized_return": "Annualized return",
            "annualized_volatility": "Annualized volatility",
            "sharpe_ratio": "Sharpe ratio",
            "max_drawdown": "Maximum drawdown",
            "average_turnover": "Average turnover",
        }
    )


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize the visible sample for a Streamlit download button."""

    return frame.to_csv(index=False).encode("utf-8")
