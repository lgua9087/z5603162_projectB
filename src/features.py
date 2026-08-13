"""Return panels and look-ahead-safe headline alignment for Project B."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def daily_returns(prices: pd.DataFrame, price_col: str = "adjClose") -> pd.DataFrame:
    """Compute simple returns within ticker on the panel's native calendar."""

    required = {"ticker", "date", price_col}
    missing = sorted(required.difference(prices.columns))
    if missing:
        raise ValueError(f"price panel is missing columns: {missing}")
    frame = prices.sort_values(["ticker", "date"]).copy()
    frame["return"] = frame.groupby("ticker", sort=False)[price_col].pct_change(
        fill_method=None
    )
    return frame


def wide_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long ticker-return panel to a sorted date-by-ticker matrix."""

    wide = returns.pivot(index="date", columns="ticker", values="return").sort_index()
    wide.columns.name = None
    return wide.apply(pd.to_numeric, errors="coerce")


def combined_returns_on_equity_calendar(
    equity_returns: pd.DataFrame,
    crypto_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Join precomputed crypto returns to equity dates without differencing a merge."""

    equity_wide = wide_returns(equity_returns)
    crypto_wide = wide_returns(crypto_returns).reindex(equity_wide.index)
    return equity_wide.join(crypto_wide, how="left").sort_index()


def align_headlines_to_trading_days(
    headlines: pd.DataFrame,
    trading_calendar: Iterable[pd.Timestamp],
) -> pd.DataFrame:
    """Map every headline to the same or next observed equity trading day."""

    calendar = (
        pd.DatetimeIndex(pd.to_datetime(list(trading_calendar)))
        .normalize()
        .sort_values()
        .unique()
    )
    if calendar.empty:
        raise ValueError("trading_calendar must not be empty")
    frame = headlines.copy()
    frame["original_news_date"] = pd.to_datetime(frame["date"]).dt.normalize()
    positions = calendar.searchsorted(frame["original_news_date"], side="left")
    valid = positions < len(calendar)
    mapped = np.full(len(frame), np.datetime64("NaT"), dtype="datetime64[ns]")
    mapped[valid] = calendar.to_numpy()[positions[valid]]
    frame["aligned_date"] = pd.to_datetime(mapped)
    frame["alignment_delay_days"] = (
        frame["aligned_date"] - frame["original_news_date"]
    ).dt.days
    return frame


def assemble_headline_panel(
    headlines: pd.DataFrame,
    trading_calendar: Iterable[pd.Timestamp],
) -> pd.DataFrame:
    """Aggregate raw, uncleaned headline wording to ticker trading days."""

    aligned = align_headlines_to_trading_days(headlines, trading_calendar)
    usable = aligned.dropna(subset=["aligned_date"]).copy()
    return (
        usable.groupby(["aligned_date", "ticker", "sector"], observed=True)
        .agg(
            headline_count=("title", "size"),
            headline_text=("title", lambda values: " || ".join(values.astype(str))),
            mean_alignment_delay_days=("alignment_delay_days", "mean"),
        )
        .reset_index()
        .sort_values(["aligned_date", "ticker"])
        .reset_index(drop=True)
    )
