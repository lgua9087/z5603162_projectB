"""VADER scoring, finance-language audit, and coverage-aware sector indices."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

FINANCE_LEXICON = {
    "hawkish": -1.0,
    "dovish": 1.0,
    "downgrade": -2.8,
    "upgrade": 2.8,
    "impairment": -2.8,
    "writedown": -2.8,
    "insolvency": -3.9,
    "default": -3.8,
    "delisting": -2.9,
    "outperform": 2.8,
    "underperform": -2.8,
    "headwind": -1.8,
    "tailwind": 1.8,
    "buyback": 1.8,
    "dilution": -2.0,
    "litigation": -2.0,
    "subpoena": -2.2,
    "restatement": -2.8,
    "oversubscribed": 2.0,
    "restructuring": -1.9,
}
FINANCE_PHRASES = {
    "guidance cut": -2.9,
    "guidance raise": 2.9,
    "earnings beat": 2.9,
    "earnings miss": -2.9,
    "material weakness": -3.0,
    "covenant breach": -3.1,
    "going concern": -3.9,
    "profit warning": -3.0,
    "record revenue": 3.0,
}
FINANCE_REASONS = {
    "hawkish": "directional for risk assets, although favourable for lenders",
    "dovish": "directional for risk assets",
    "downgrade": "analyst action that states a judgement",
    "upgrade": "analyst action that states a judgement",
    "impairment": "write-down event",
    "writedown": "write-down event",
    "insolvency": "failure event",
    "default": "failure event",
    "delisting": "failure event",
    "outperform": "relative performance judgement",
    "underperform": "relative performance judgement",
    "headwind": "figurative but consistently directional in filings",
    "tailwind": "figurative but consistently directional in filings",
    "buyback": "capital-return action generally read as favourable",
    "dilution": "shareholder-value reduction",
    "litigation": "adverse legal event",
    "subpoena": "adverse legal event",
    "restatement": "accounting failure",
    "oversubscribed": "favourable demand outcome",
    "restructuring": "corrective event usually reported as adverse",
    "guidance cut": "forward-looking downgrade by management",
    "guidance raise": "forward-looking upgrade by management",
    "earnings beat": "result above expectations",
    "earnings miss": "result below expectations",
    "material weakness": "defined control failure in filings",
    "covenant breach": "debt-agreement failure",
    "going concern": "audit warning about survival",
    "profit warning": "pre-announced shortfall",
    "record revenue": "favourable operating result",
}
BOOSTER_UPDATES = {
    "sharply": 0.293,
    "materially": 0.293,
    "steeply": 0.293,
    "modestly": -0.293,
    "marginally": -0.293,
}


def compound_normalize(value: float, alpha: float = 15.0) -> float:
    """Apply VADER's bounded compound-score normalization."""

    return value / np.sqrt(value * value + alpha)


def finance_lexicon_audit() -> pd.DataFrame:
    """Return the approved, reviewable finance additions used in this project."""

    rows = []
    for term_type, mapping in (("word", FINANCE_LEXICON), ("phrase", FINANCE_PHRASES)):
        rows.extend(
            {
                "term": term,
                "term_type": term_type,
                "valence": value,
                "decision": "approve",
                "reason": FINANCE_REASONS[term],
                "source": "Week 8 ten-rater mean plus recorded human review",
            }
            for term, value in mapping.items()
        )
    return pd.DataFrame(rows).sort_values("term").reset_index(drop=True)


def score_headlines(panel: pd.DataFrame) -> pd.DataFrame:
    """Score untouched headline text with plain and finance-extended VADER 3.3.2."""

    required = {"title", "ticker", "sector", "aligned_date"}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"headline panel is missing columns: {missing}")

    import vaderSentiment.vaderSentiment as vader

    frame = panel.copy()
    texts = frame["title"].fillna("").astype(str).tolist()
    baseline_analyzer = vader.SentimentIntensityAnalyzer()
    frame["baseline_compound"] = [
        baseline_analyzer.polarity_scores(text)["compound"] for text in texts
    ]

    saved_special_cases = dict(vader.SPECIAL_CASES)
    saved_boosters = dict(vader.BOOSTER_DICT)
    try:
        extended = vader.SentimentIntensityAnalyzer()
        extended.lexicon.update(FINANCE_LEXICON)
        head_words: dict[str, float] = {}
        for phrase, valence in FINANCE_PHRASES.items():
            head = phrase.split()[-1]
            if head not in extended.lexicon:
                head_words[head] = 0.1 if valence > 0 else -0.1
            vader.SPECIAL_CASES[phrase] = valence
        extended.lexicon.update(head_words)
        vader.BOOSTER_DICT.update(BOOSTER_UPDATES)
        frame["compound"] = [
            extended.polarity_scores(text)["compound"] for text in texts
        ]
    finally:
        vader.SPECIAL_CASES.clear()
        vader.SPECIAL_CASES.update(saved_special_cases)
        vader.BOOSTER_DICT.clear()
        vader.BOOSTER_DICT.update(saved_boosters)

    frame["score_changed"] = ~np.isclose(
        frame["compound"], frame["baseline_compound"], atol=1e-12
    )
    if not frame[["baseline_compound", "compound"]].apply(
        lambda values: values.between(-1.0, 1.0).all()
    ).all():
        raise AssertionError("VADER compound score left the range [-1, 1]")
    return frame


def sentiment_model_comparison(scores: pd.DataFrame) -> pd.DataFrame:
    """Measure coverage and distribution changes from the finance extension."""

    rows: list[dict[str, object]] = []
    for model, column in (
        ("Plain VADER 3.3.2", "baseline_compound"),
        ("Finance-extended VADER 3.3.2", "compound"),
    ):
        values = scores[column].astype(float)
        rows.append(
            {
                "model": model,
                "headlines": len(values),
                "mean_compound": float(values.mean()),
                "standard_deviation": float(values.std(ddof=1)),
                "positive_share": float(values.ge(0.05).mean()),
                "neutral_share": float(values.abs().lt(0.05).mean()),
                "negative_share": float(values.le(-0.05).mean()),
                "nonzero_share": float(values.ne(0.0).mean()),
                "changed_from_plain_share": (
                    0.0 if column == "baseline_compound" else float(scores["score_changed"].mean())
                ),
            }
        )
    return pd.DataFrame(rows)


def ticker_day_sentiment(scores: pd.DataFrame) -> pd.DataFrame:
    """Average headline scores within ticker trading days before sector aggregation."""

    return (
        scores.groupby(["aligned_date", "ticker", "sector"], observed=True)
        .agg(
            sentiment=("compound", "mean"),
            baseline_sentiment=("baseline_compound", "mean"),
            headline_count=("title", "size"),
            changed_headline_share=("score_changed", "mean"),
        )
        .reset_index()
        .sort_values(["aligned_date", "ticker"])
        .reset_index(drop=True)
    )


def sector_sentiment_index(
    scores: pd.DataFrame,
    *,
    trading_calendar: Iterable[pd.Timestamp],
    ticker_sector: pd.DataFrame,
) -> pd.DataFrame:
    """Build full-calendar sector indices, equal-weighting observed ticker-day scores."""

    ticker_daily = ticker_day_sentiment(scores)
    calendar = pd.DatetimeIndex(pd.to_datetime(list(trading_calendar))).sort_values().unique()
    sectors = sorted(ticker_sector["sector"].dropna().unique())
    grid = pd.MultiIndex.from_product([calendar, sectors], names=["date", "sector"])
    observed = (
        ticker_daily.groupby(["aligned_date", "sector"], observed=True)
        .agg(
            sentiment=("sentiment", "mean"),
            baseline_sentiment=("baseline_sentiment", "mean"),
            ticker_count=("ticker", "nunique"),
            headline_count=("headline_count", "sum"),
            changed_headline_share=("changed_headline_share", "mean"),
        )
        .rename_axis(index=["date", "sector"])
        .reindex(grid)
        .reset_index()
    )
    sector_sizes = ticker_sector.groupby("sector")["ticker"].nunique()
    observed["sector_ticker_count"] = observed["sector"].map(sector_sizes).astype(int)
    observed["ticker_count"] = observed["ticker_count"].fillna(0).astype(int)
    observed["headline_count"] = observed["headline_count"].fillna(0).astype(int)
    observed["coverage_rate"] = (
        observed["ticker_count"] / observed["sector_ticker_count"]
    )
    observed["sentiment_neutral_fill"] = (
        observed["sentiment"].fillna(0.0) * observed["coverage_rate"]
    )
    observed["sentiment_carry_3d"] = observed.groupby("sector", observed=True)[
        "sentiment"
    ].ffill(limit=3)
    observed["sentiment_21d"] = observed.groupby("sector", observed=True)[
        "sentiment"
    ].transform(lambda values: values.rolling(21, min_periods=5).mean())
    return observed.sort_values(["date", "sector"]).reset_index(drop=True)


def ticker_signal_panel(
    scores: pd.DataFrame,
    *,
    trading_calendar: Iterable[pd.Timestamp],
    tickers: Iterable[str],
    lookback: int = 21,
    minimum_observations: int = 5,
) -> pd.DataFrame:
    """Create a one-day-lagged, coverage-scaled ticker signal for portfolio fusion."""

    if lookback < 2 or not 1 <= minimum_observations <= lookback:
        raise ValueError("lookback and minimum_observations are inconsistent")

    daily = ticker_day_sentiment(scores)
    calendar = pd.DatetimeIndex(pd.to_datetime(list(trading_calendar))).sort_values().unique()
    ticker_list = sorted(set(tickers))
    pivot = daily.pivot(index="aligned_date", columns="ticker", values="sentiment")
    pivot = pivot.reindex(index=calendar, columns=ticker_list)
    rolling = pivot.rolling(lookback, min_periods=minimum_observations).mean()
    observed = pivot.notna().astype(int).rolling(lookback, min_periods=1).sum()
    coverage = observed / lookback
    lagged_signal = rolling.shift(1)
    lagged_coverage = coverage.shift(1).fillna(0.0)
    lagged_observed = observed.shift(1).fillna(0).astype(int)

    def cross_sectional_z(row: pd.Series) -> pd.Series:
        standard_deviation = row.std(ddof=0)
        if not np.isfinite(standard_deviation) or standard_deviation <= 1e-12:
            return pd.Series(0.0, index=row.index)
        return row.sub(row.mean()).div(standard_deviation).clip(-2.5, 2.5)

    z_scores = lagged_signal.apply(cross_sectional_z, axis=1).fillna(0.0)
    result = (
        z_scores.rename_axis(index="date", columns="ticker")
        .reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="signal_z")
        .merge(
            lagged_coverage.rename_axis(index="date", columns="ticker")
            .reset_index()
            .melt(id_vars="date", var_name="ticker", value_name="coverage_window"),
            on=["date", "ticker"],
            validate="one_to_one",
        )
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    result = result.merge(
        lagged_observed.rename_axis(index="date", columns="ticker")
        .reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="signal_observations"),
        on=["date", "ticker"],
        validate="one_to_one",
    )
    result["coverage_21d"] = result["coverage_window"]
    result["signal_lookback"] = lookback
    return result.sort_values(["date", "ticker"]).reset_index(drop=True)
