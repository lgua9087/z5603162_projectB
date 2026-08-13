"""MarketReady Funds: precomputed systematic-fund and sentiment dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.app_charts import (
    allocation_figure,
    fact_sheet_path,
    fusion_figure,
    growth_comparison,
    holdings_bar,
    sentiment_figure,
)
from src.app_data import (
    ArtifactError,
    allocation_lookthrough,
    allocation_path,
    balanced_default_funds,
    current_holdings,
    equity_sector_exposure,
    latest_sentiment_snapshot,
    load_app_artifacts,
    presentation_metrics,
    to_csv_bytes,
)

PROJECT_ROOT = Path(__file__).resolve().parent
APP_TITLE = "MarketReady Funds"
VIEW_OPTIONS = (
    "Fund marketplace",
    "Fund fact sheet",
    "Allocation lab",
    "News sentiment",
    "Method and risks",
)
VIEW_SLUGS = {
    "marketplace": "Fund marketplace",
    "factsheet": "Fund fact sheet",
    "allocation": "Allocation lab",
    "sentiment": "News sentiment",
    "method": "Method and risks",
}
SLUG_BY_VIEW = {value: key for key, value in VIEW_SLUGS.items()}


def _configure_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root { --navy:#16324F; --teal:#2A9D8F; --gold:#D99B2B; --light:#F4F7FA; }
        .stApp { background: linear-gradient(180deg, #F7FAFC 0, #FFFFFF 20rem); }
        [data-testid="stMainBlockContainer"] { padding-top: 1.25rem; }
        h1, h2, h3 { color: var(--navy); letter-spacing: -0.02em; }
        h1 { padding-bottom: .2rem; }
        [data-testid="stMetric"] {
            background: white; border: 1px solid #DDE6EE; border-radius: 12px;
            padding: .58rem .78rem; box-shadow: 0 3px 12px rgba(22,50,79,.05);
        }
        [data-testid="stMetricLabel"] { margin-bottom: -.15rem; }
        [data-testid="stMetricValue"] { line-height: 1.05; }
        [data-testid="stSidebar"] { background: #EEF4F7; }
        .mr-hero {
            padding: .72rem 1rem; border-radius: 12px; line-height: 1.38;
            background: linear-gradient(110deg, #16324F, #234E70);
            color: white; margin: .1rem 0 .65rem 0;
        }
        .mr-hero strong { color: #F4C95D; }
        .mr-note {
            border-left: 4px solid var(--teal); background: #EFF8F6;
            padding: .7rem .9rem; border-radius: 0 10px 10px 0;
        }
        .mr-note-negative { border-left-color: #C44E52; background: #FFF1F2; }
        [data-baseweb="tag"] {
            background-color: #DCE8F1 !important;
            border: 1px solid #A6BDCC !important;
        }
        [data-baseweb="tag"] * {
            color: #16324F !important;
            fill: #16324F !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Loading verified fund results...")
def _load_results():
    return load_app_artifacts(PROJECT_ROOT)


def _seed_view() -> None:
    if "view_seeded" in st.session_state:
        return
    requested = str(st.query_params.get("view", "marketplace"))
    st.session_state["active_view"] = VIEW_SLUGS.get(requested, VIEW_OPTIONS[0])
    st.session_state["view_seeded"] = True


def _sync_view() -> None:
    st.query_params["view"] = SLUG_BY_VIEW[st.session_state["active_view"]]


def _data_health(artifacts) -> None:
    returns = artifacts.fund_returns
    sentiment = artifacts.sector_sentiment
    columns = st.columns(5)
    columns[0].metric("Fund products", f"{returns['fund_id'].nunique():,}")
    columns[1].metric("OOS start", f"{returns['date'].min():%d %b %Y}")
    columns[2].metric("OOS end", f"{returns['date'].max():%d %b %Y}")
    columns[3].metric("Sector indices", f"{sentiment['sector'].nunique():,}")
    columns[4].metric("Latest sentiment date", f"{sentiment['date'].max():%d %b %Y}")


def _metric_card_row(row: pd.Series) -> None:
    columns = st.columns(5)
    columns[0].metric("Annualized return", f"{row['annualized_return']:.1%}")
    columns[1].metric("Annualized volatility", f"{row['annualized_volatility']:.1%}")
    columns[2].metric("Sharpe ratio", f"{row['sharpe_ratio']:.2f}")
    columns[3].metric("Maximum drawdown", f"{row['max_drawdown']:.1%}")
    terminal = row.get("terminal_wealth", float("nan"))
    columns[4].metric("Ending value of $1", f"${terminal:.2f}")


def _render_marketplace(artifacts) -> None:
    st.header("Compare the fund range")
    st.write(
        "Each product is a separately investable, rules-based fund. Performance is "
        "walk-forward and out of sample; it is historical evidence, not a forecast."
    )
    metrics = artifacts.performance_metrics.sort_values("sharpe_ratio", ascending=False)
    families = ["All", *sorted(metrics["family"].unique())]
    family = st.segmented_control("Asset family", families, default="All")
    visible = metrics if family == "All" else metrics.loc[metrics["family"].eq(family)]
    default_ids = balanced_default_funds(visible, limit=min(3, len(visible)))
    selected_ids = st.multiselect(
        "Funds shown in the growth comparison",
        options=visible["fund_id"].tolist(),
        default=default_ids,
        format_func=dict(zip(metrics["fund_id"], metrics["fund_name"], strict=False)).get,
        max_selections=6,
    )
    if selected_ids:
        st.plotly_chart(
            growth_comparison(artifacts.fund_returns, selected_ids),
            width="stretch",
        )
    else:
        st.info("Select at least one fund to draw the growth comparison.")
    display = presentation_metrics(visible)
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        height=min(620, 38 * (len(display) + 1)),
        column_config={
            "Annualized return": st.column_config.NumberColumn(format="percent"),
            "Annualized volatility": st.column_config.NumberColumn(format="percent"),
            "Sharpe ratio": st.column_config.NumberColumn(format="%.2f"),
            "Maximum drawdown": st.column_config.NumberColumn(format="percent"),
            "Average turnover": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.download_button(
        "Download the visible fund comparison",
        data=to_csv_bytes(display),
        file_name="marketready_fund_comparison.csv",
        mime="text/csv",
    )


def _render_fact_sheet(artifacts) -> None:
    st.header("Read a fund fact sheet")
    st.write(
        "Inspect one fund's out-of-sample path, downside experience, and latest "
        "target holdings before considering an allocation."
    )
    metrics = artifacts.performance_metrics.sort_values(["family", "method"])
    label_map = dict(zip(metrics["fund_id"], metrics["fund_name"], strict=False))
    fund_id = st.selectbox(
        "Fund",
        metrics["fund_id"].tolist(),
        format_func=label_map.get,
        key="fact_sheet_fund",
    )
    row = metrics.loc[metrics["fund_id"].eq(fund_id)].iloc[0]
    _metric_card_row(row)
    st.plotly_chart(
        fact_sheet_path(artifacts.fund_returns, fund_id), width="stretch"
    )
    holdings = current_holdings(artifacts.fund_weights, fund_id)
    if holdings.empty:
        st.warning("No current target holdings are available for this fund.")
        return
    latest = holdings["rebalance_date"].max()
    risk = artifacts.combined_risk.loc[artifacts.combined_risk["fund_id"].eq(fund_id)]
    if not risk.empty:
        current_risk = risk.sort_values("rebalance_date").iloc[-1]
        st.subheader("Latest asset-family risk look-through")
        risk_cards = st.columns(4)
        risk_cards[0].metric(
            "Crypto capital weight", f"{current_risk['crypto_capital_weight']:.1%}"
        )
        risk_cards[1].metric(
            "Crypto covariance-risk share",
            f"{current_risk['crypto_risk_contribution']:.1%}",
        )
        risk_cards[2].metric(
            "Effective holdings", f"{current_risk['effective_holdings']:.1f}"
        )
        risk_cards[3].metric("Top-5 target weight", f"{current_risk['top_5_weight']:.1%}")
        st.caption(
            "Risk shares use the same trailing, diagonally shrunk covariance matrix as "
            "the target optimiser; they are estimates, not loss forecasts."
        )
    st.subheader(f"Current target holdings · {latest:%d %B %Y}")
    st.plotly_chart(holdings_bar(holdings), width="stretch")
    display = holdings[["ticker", "asset_class", "weight"]].rename(
        columns={"ticker": "Asset", "asset_class": "Asset class", "weight": "Weight"}
    )
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        height=min(520, 38 * (len(display) + 1)),
        column_config={"Weight": st.column_config.NumberColumn(format="percent")},
    )
    st.download_button(
        "Download current holdings",
        data=to_csv_bytes(display),
        file_name=f"{fund_id}_current_holdings.csv",
        mime="text/csv",
    )


def _render_allocation(artifacts) -> None:
    st.header("Build a multi-fund allocation")
    st.write(
        "Allocate across the fund products and inspect the historical combined path. "
        "Entered percentages are normalized to 100%; no capital is invested here."
    )
    metrics = artifacts.performance_metrics.sort_values("sharpe_ratio", ascending=False)
    label_map = dict(zip(metrics["fund_id"], metrics["fund_name"], strict=False))
    defaults = balanced_default_funds(metrics, limit=min(3, len(metrics)))
    fund_ids = st.multiselect(
        "Funds in the allocation",
        metrics["fund_id"].tolist(),
        default=defaults,
        format_func=label_map.get,
        max_selections=5,
    )
    if not fund_ids:
        st.info("Select at least one fund to construct an allocation.")
        return
    initial = 100.0 / len(fund_ids)
    allocations: dict[str, float] = {}
    columns = st.columns(min(3, len(fund_ids)))
    for index, fund_id in enumerate(fund_ids):
        with columns[index % len(columns)]:
            allocations[fund_id] = st.number_input(
                label_map[fund_id],
                min_value=0.0,
                max_value=100.0,
                value=round(initial, 1),
                step=1.0,
                format="%.1f",
                key=f"allocation_{fund_id}",
                help="Percentage before normalization.",
            )
    if sum(allocations.values()) <= 0:
        st.warning("Enter a positive allocation for at least one fund.")
        return
    path, metrics_out = allocation_path(artifacts.fund_returns, allocations)
    normalized = {key: value / sum(allocations.values()) for key, value in allocations.items()}
    st.caption(
        "Normalized allocation: "
        + " · ".join(f"{label_map[key]} {value:.1%}" for key, value in normalized.items())
    )
    cards = st.columns(5)
    cards[0].metric("Annualized return", f"{metrics_out['annualized_return']:.1%}")
    cards[1].metric("Annualized volatility", f"{metrics_out['annualized_volatility']:.1%}")
    cards[2].metric("Sharpe ratio", f"{metrics_out['sharpe_ratio']:.2f}")
    cards[3].metric("Maximum drawdown", f"{metrics_out['max_drawdown']:.1%}")
    cards[4].metric("Ending value of $1", f"${metrics_out['terminal_wealth']:.2f}")
    st.plotly_chart(allocation_figure(path), width="stretch")
    st.caption(
        "Historical simulation uses constant fund-level allocations, a zero risk-free "
        "rate, and the union of available fund valuation dates. Missing non-trading-day "
        "fund returns are treated as zero."
    )
    lookthrough, exposure_metrics = allocation_lookthrough(
        artifacts.fund_weights,
        allocations,
    )
    st.subheader("Latest look-through exposure")
    exposure_cards = st.columns(4)
    exposure_cards[0].metric(
        "Equity capital exposure", f"{exposure_metrics['equity_weight']:.1%}"
    )
    exposure_cards[1].metric(
        "Crypto capital exposure", f"{exposure_metrics['crypto_weight']:.1%}"
    )
    exposure_cards[2].metric(
        "Effective holdings", f"{exposure_metrics['effective_holdings']:.1f}"
    )
    exposure_cards[3].metric(
        "Top-5 underlying exposure", f"{exposure_metrics['top_5_weight']:.1%}"
    )
    sector_exposure = equity_sector_exposure(lookthrough)
    sector_column, holdings_column = st.columns([0.8, 1.2])
    with sector_column:
        st.markdown("**Major equity sectors**")
        if sector_exposure.empty:
            st.caption("No equity exposure in the selected allocation.")
        else:
            sector_display = sector_exposure.head(5).rename(
                columns={
                    "sector": "Equity sector",
                    "lookthrough_weight": "Share of total allocation",
                }
            )
            st.dataframe(
                sector_display,
                hide_index=True,
                width="stretch",
                height=min(265, 38 * (len(sector_display) + 1)),
                column_config={
                    "Share of total allocation": st.column_config.NumberColumn(
                        format="percent"
                    )
                },
            )
    with holdings_column:
        st.markdown("**Top underlying holdings**")
        holdings_display = lookthrough.head(8).rename(
            columns={
                "ticker": "Asset",
                "asset_class": "Asset class",
                "sector": "Sector",
                "lookthrough_weight": "Look-through weight",
            }
        )
        st.dataframe(
            holdings_display,
            hide_index=True,
            width="stretch",
            height=min(380, 38 * (len(holdings_display) + 1)),
            column_config={
                "Look-through weight": st.column_config.NumberColumn(format="percent")
            },
        )
    st.caption(
        "Capital exposures combine each selected fund's latest target. Sector weights are "
        "shares of the total allocation, not risk contributions; the holding table reveals "
        "overlap that fund-level percentages can hide."
    )
    st.download_button(
        "Download custom allocation path",
        data=to_csv_bytes(path),
        file_name="marketready_custom_allocation.csv",
        mime="text/csv",
    )


def _render_sentiment(artifacts) -> None:
    st.header("Read sector news sentiment")
    sentiment = artifacts.sector_sentiment.sort_values("date")
    snapshot = latest_sentiment_snapshot(sentiment)
    note_class = (
        "mr-note mr-note-negative"
        if snapshot["tone_label"] == "negative"
        else "mr-note"
    )
    st.markdown(
        f'<div class="{note_class}"><strong>Latest sample reading:</strong> '
        f'{snapshot["tone_label"].capitalize()} tone '
        f'({snapshot["score_21d"]:+.3f} on the 21-day sector average), but fresh '
        f'coverage was {snapshot["coverage_label"]}: {snapshot["covered_tickers"]} of '
        f'{snapshot["universe_tickers"]} equity tickers carried headlines on '
        f'{snapshot["date"]:%d %b %Y}. Treat tone as market context, not a standalone '
        "allocation signal.</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "How to read the measures: VADER scores original headline wording. Bold lines average "
        "observed sector scores over a 21-trading-day window; no-news sector-days remain "
        "missing rather than being coded as neutral. Fresh ticker coverage is the share of the "
        "50-stock equity universe with at least one headline on the latest index date."
    )
    sectors = sorted(sentiment["sector"].unique())
    defaults = sectors[: min(4, len(sectors))]
    selected = st.multiselect(
        "Equity sectors",
        sectors,
        default=defaults,
        max_selections=6,
    )
    cards = st.columns(4)
    cards[0].metric("Latest sample date", f"{snapshot['date']:%d %b %Y}")
    cards[1].metric("21-day sector-average tone", f"{snapshot['score_21d']:+.3f}")
    cards[2].metric(
        "Fresh ticker coverage (latest date)",
        f"{snapshot['covered_tickers']} of {snapshot['universe_tickers']} · "
        f"{snapshot['coverage_rate']:.1%}",
    )
    cards[3].metric("Headlines on latest date", f"{snapshot['headline_count']:,}")
    if selected:
        st.plotly_chart(sentiment_figure(sentiment, selected), width="stretch")
    else:
        st.info("Select at least one sector to draw the sentiment history.")
    st.subheader("Finance-language extension audit")
    model_table = artifacts.sentiment_model_comparison.copy()
    st.dataframe(model_table, hide_index=True, width="stretch")
    st.subheader("Sentiment fusion: before and after")
    st.plotly_chart(fusion_figure(artifacts.fusion_returns), width="stretch")
    st.dataframe(
        artifacts.fusion_comparison,
        hide_index=True,
        width="stretch",
        column_config={
            "annualized_return": st.column_config.NumberColumn(format="percent"),
            "annualized_volatility": st.column_config.NumberColumn(format="percent"),
            "sharpe_ratio": st.column_config.NumberColumn(format="%.2f"),
            "max_drawdown": st.column_config.NumberColumn(format="percent"),
        },
    )


def _render_method(artifacts) -> None:
    st.header("Method, assumptions, and limits")
    st.write(
        "The evidence follows a fixed information clock: portfolio weights use only "
        "returns observed before a rebalance, and a headline aligned to trading day t "
        "first affects a decision on trading day t+1."
    )
    st.subheader("Portfolio rules")
    st.latex(r"r_{p,t}=w_{t^-}^{\mathsf T}r_t,\qquad \sum_i w_{i,t^-}=1")
    st.latex(r"\mathrm{Sharpe}=\frac{252\,\bar r_p}{\sqrt{252}\,s(r_p)}")
    st.write(
        "Equity and combined funds use 252 observations per year. Crypto-only funds "
        "retain their seven-day calendar and use 365. Long-only target weights are "
        "estimated from a trailing window and held until the next monthly rebalance."
    )
    st.subheader("Headline model")
    st.latex(r"\mathrm{compound}=\frac{x}{\sqrt{x^2+15}}")
    st.write(
        "VADER receives the original casing, punctuation, negation, and intensifiers. "
        "Headline scores are averaged within ticker-day, then tickers are equally "
        "weighted into sector indices. A finance-language extension changes coverage; "
        "it is not claimed to prove higher classification accuracy."
    )
    st.subheader("Limits investors should retain")
    st.markdown(
        """
        - Results cover 2020-2023 and do not forecast future performance.
        - Headlines are short, noisy proxies for information and may repeat one event.
        - Fund and fusion returns deduct 5 basis points per unit of realised turnover.
          Reported results are before management fees and omit spreads and market impact.
        - Optimized portfolios depend on estimated means and covariances and can be
          unstable outside the sample.
        - The allocation lab is an educational historical simulation, not advice or an
          order-entry system.
        """
    )
    if artifacts.summary:
        with st.expander("Build provenance"):
            st.json(artifacts.summary)


def main() -> None:
    _configure_page()
    st.title(APP_TITLE)
    st.markdown(
        '<div class="mr-hero"><strong>Transparent systematic investing.</strong><br>'
        "Compare equity, crypto, and combined funds using walk-forward evidence, "
        "inspect current holdings, test a multi-fund allocation, and read sector news "
        "sentiment alongside its coverage.</div>",
        unsafe_allow_html=True,
    )
    try:
        artifacts = _load_results()
    except (ArtifactError, ValueError, OSError) as error:
        st.error("The verified result package is not available yet.")
        st.code("python scripts/run_part_b.py", language="bash")
        st.caption(str(error))
        return

    _data_health(artifacts)
    _seed_view()
    with st.sidebar:
        st.header("Investor journey")
        st.radio(
            "View",
            VIEW_OPTIONS,
            key="active_view",
            on_change=_sync_view,
        )
        st.divider()
        st.caption(
            "Precomputed results · FINS5545 hosted data · 2020-2023 source sample"
        )
        st.caption("Educational prototype — not financial advice.")

    active = st.session_state["active_view"]
    if active == "Fund marketplace":
        _render_marketplace(artifacts)
    elif active == "Fund fact sheet":
        _render_fact_sheet(artifacts)
    elif active == "Allocation lab":
        _render_allocation(artifacts)
    elif active == "News sentiment":
        _render_sentiment(artifacts)
    else:
        _render_method(artifacts)


if __name__ == "__main__":
    main()
