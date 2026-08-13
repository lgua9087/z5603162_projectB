"""Conservative cleaning and integrity checks for Project B source data."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src import data_access

SAMPLE_END = pd.Timestamp("2023-12-31")
PRICE_KEY = ["ticker", "date"]
NEWS_KEY = ["ticker", "date", "title"]


def normalize_date(values: pd.Series) -> pd.Series:
    """Return timezone-naive, midnight-normalized dates for safe merging."""

    return (
        pd.to_datetime(values, utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
        .astype("datetime64[ns]")
    )


def _strip_strings(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = frame[column].astype("string").str.strip()


def clean_price_panel(
    raw: pd.DataFrame,
    *,
    asset_class: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalize, cap, and key-deduplicate a supplied price panel."""

    required = {
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjClose",
        "volume",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"{asset_class} price panel is missing columns: {missing}")
    frame = raw.copy()
    rows_before = len(frame)
    frame["date"] = normalize_date(frame["date"])
    _strip_strings(frame, ("ticker", "sector"))
    rows_after_cutoff = int(frame["date"].gt(SAMPLE_END).sum())
    frame = frame.loc[frame["date"].le(SAMPLE_END)].copy()
    duplicates = int(frame.duplicated(PRICE_KEY, keep="first").sum())
    frame = (
        frame.drop_duplicates(PRICE_KEY, keep="first")
        .sort_values(PRICE_KEY)
        .reset_index(drop=True)
    )
    return frame, {
        "rows_before": rows_before,
        "rows_after": len(frame),
        "rows_after_cutoff": rows_after_cutoff,
        "duplicate_rows_removed": duplicates,
    }


def clean_news_headlines(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalize headline dates and remove exact ticker-date-title duplicates."""

    required = {"ticker", "date", "sector", "title", "url", "publisher"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"news panel is missing columns: {missing}")
    frame = raw.copy()
    rows_before = len(frame)
    frame["date"] = normalize_date(frame["date"])
    _strip_strings(frame, ("ticker", "sector", "title", "url", "publisher"))
    frame["publisher"] = frame["publisher"].replace("", pd.NA)
    rows_after_cutoff = int(frame["date"].gt(SAMPLE_END).sum())
    frame = frame.loc[frame["date"].le(SAMPLE_END)].copy()
    duplicates = int(frame.duplicated(NEWS_KEY, keep="first").sum())
    frame = (
        frame.drop_duplicates(NEWS_KEY, keep="first")
        .sort_values(["date", "ticker", "title"])
        .reset_index(drop=True)
    )
    return frame, {
        "rows_before": rows_before,
        "rows_after": len(frame),
        "rows_after_cutoff": rows_after_cutoff,
        "duplicate_rows_removed": duplicates,
    }


def load_clean_equities() -> tuple[pd.DataFrame, dict[str, int]]:
    """Load and conservatively clean the supplied equity data."""

    return clean_price_panel(data_access.load_equity_prices(), asset_class="Equity")


def load_clean_crypto() -> tuple[pd.DataFrame, dict[str, int]]:
    """Load and conservatively clean the supplied crypto data."""

    return clean_price_panel(data_access.load_crypto_prices(), asset_class="Crypto")


def load_clean_news() -> tuple[pd.DataFrame, dict[str, int]]:
    """Load and conservatively clean the supplied headline data."""

    return clean_news_headlines(data_access.load_news_headlines())


def integrity_summary(
    equities: pd.DataFrame,
    crypto: pd.DataFrame,
    news: pd.DataFrame,
    *,
    equity_meta: dict[str, int],
    crypto_meta: dict[str, int],
    news_meta: dict[str, int],
) -> pd.DataFrame:
    """Summarize high-risk source assumptions carried from Part A."""

    def price_flags(frame: pd.DataFrame) -> tuple[int, int, int]:
        duplicate_count = int(frame.duplicated(PRICE_KEY).sum())
        nonpositive_price = int(
            frame[["open", "high", "low", "close", "adjClose"]].le(0).sum().sum()
        )
        ohlc = int(
            (
                frame["high"].lt(frame[["open", "close", "low"]].max(axis=1))
                | frame["low"].gt(frame[["open", "close", "high"]].min(axis=1))
            ).sum()
        )
        return duplicate_count, nonpositive_price, ohlc

    equity_flags = price_flags(equities)
    crypto_flags = price_flags(crypto)
    rows = [
        ("Equity rows", len(equities), "retained after 2023 cutoff"),
        ("Crypto rows", len(crypto), "10 rows after 2023 removed"),
        ("Unique news headlines", len(news), "ticker-date-title duplicates removed"),
        ("Equity duplicate keys", equity_flags[0], "must equal zero"),
        ("Crypto duplicate keys", crypto_flags[0], "must equal zero"),
        ("News duplicate keys", int(news.duplicated(NEWS_KEY).sum()), "must equal zero"),
        ("Nonpositive equity prices", equity_flags[1], "must equal zero"),
        ("Nonpositive crypto prices", crypto_flags[1], "must equal zero"),
        ("Equity OHLC flags", equity_flags[2], "must equal zero"),
        ("Crypto OHLC flags", crypto_flags[2], "must equal zero"),
        (
            "Rows removed after cutoff",
            equity_meta["rows_after_cutoff"]
            + crypto_meta["rows_after_cutoff"]
            + news_meta["rows_after_cutoff"],
            "mandatory sample cap",
        ),
        (
            "News duplicates removed",
            news_meta["duplicate_rows_removed"],
            "exact copied headlines only",
        ),
    ]
    return pd.DataFrame(rows, columns=["check", "count", "treatment"])
