"""Plotly figures for the MarketReady Funds investor interface."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NAVY = "#16324F"
TEAL = "#2A9D8F"
GOLD = "#D99B2B"
RED = "#C44E52"
SLATE = "#64748B"
LIGHT = "#E8EEF4"
COLORS = [NAVY, TEAL, GOLD, "#4F6D8A", "#6B5B95", "#4A7C59", "#8B6F47", "#577590"]


def _theme(fig: go.Figure, *, height: int = 470) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 30, "r": 20, "t": 45, "b": 35},
        font={"family": "Arial, sans-serif", "color": NAVY},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "title": None,
        },
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=LIGHT, zerolinecolor=LIGHT)
    return fig


def growth_comparison(frame: pd.DataFrame, fund_ids: list[str]) -> go.Figure:
    """Compare growth of one dollar across selected funds."""

    selected = frame.loc[frame["fund_id"].isin(fund_ids)].sort_values("date")
    fig = go.Figure()
    for color, (fund_id, group) in zip(COLORS, selected.groupby("fund_id"), strict=False):
        label = str(group["fund_name"].iloc[0])
        values = (1.0 + group["daily_return"].fillna(0.0)).cumprod()
        fig.add_trace(
            go.Scatter(
                x=group["date"],
                y=values,
                mode="lines",
                name=label,
                line={"color": color, "width": 2},
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
            )
        )
    _theme(fig)
    fig.update_yaxes(title="Growth of $1", type="log", tickprefix="$", tickformat=".2f")
    fig.update_xaxes(title="Out-of-sample date", rangeslider={"visible": True})
    return fig


def fact_sheet_path(frame: pd.DataFrame, fund_id: str) -> go.Figure:
    """Show growth and drawdown for one selected fund."""

    group = frame.loc[frame["fund_id"].eq(fund_id)].sort_values("date").copy()
    growth = (1.0 + group["daily_return"].fillna(0.0)).cumprod()
    drawdown = growth.div(growth.cummax()).sub(1.0)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Growth of $1", "Drawdown from prior peak"),
        row_heights=(0.62, 0.38),
    )
    fig.add_trace(
        go.Scatter(
            x=group["date"],
            y=growth,
            mode="lines",
            line={"color": TEAL, "width": 2.5},
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
            name="Fund value",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=group["date"],
            y=drawdown,
            mode="lines",
            fill="tozeroy",
            line={"color": RED, "width": 1.5},
            fillcolor="rgba(196,78,82,0.20)",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1%}<extra></extra>",
            name="Drawdown",
        ),
        row=2,
        col=1,
    )
    _theme(fig, height=610)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(tickprefix="$", tickformat=".2f", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title="Out-of-sample date", row=2, col=1)
    return fig


def holdings_bar(holdings: pd.DataFrame) -> go.Figure:
    """Show the latest target holdings as a horizontal bar chart."""

    ordered = holdings.sort_values("weight", ascending=True)
    colors = [TEAL if item == "Equity" else GOLD for item in ordered["asset_class"]]
    fig = go.Figure(
        go.Bar(
            x=ordered["weight"],
            y=ordered["ticker"],
            orientation="h",
            marker_color=colors,
            text=ordered["weight"].map(lambda value: f"{value:.1%}"),
            textposition="outside",
            hovertemplate="%{y}<br>Target weight: %{x:.2%}<extra></extra>",
        )
    )
    _theme(fig, height=max(400, 24 * len(ordered)))
    fig.update_layout(showlegend=False, margin={"l": 50, "r": 55, "t": 20, "b": 35})
    fig.update_xaxes(title="Target portfolio weight", tickformat=".0%")
    fig.update_yaxes(title=None)
    return fig


def allocation_figure(path: pd.DataFrame) -> go.Figure:
    """Display a custom allocation's wealth and drawdown paths."""

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Custom allocation growth", "Custom allocation drawdown"),
        row_heights=(0.62, 0.38),
    )
    fig.add_trace(
        go.Scatter(
            x=path["date"],
            y=path["growth_of_1"],
            mode="lines",
            line={"color": NAVY, "width": 2.5},
            name="Growth of $1",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=path["date"],
            y=path["drawdown"],
            mode="lines",
            fill="tozeroy",
            line={"color": RED, "width": 1.5},
            fillcolor="rgba(196,78,82,0.20)",
            name="Drawdown",
        ),
        row=2,
        col=1,
    )
    _theme(fig, height=600)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(tickprefix="$", tickformat=".2f", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title="Out-of-sample date", row=2, col=1)
    return fig


def sentiment_figure(frame: pd.DataFrame, sectors: list[str]) -> go.Figure:
    """Plot daily and 21-day sector sentiment for selected sectors."""

    selected = frame.loc[frame["sector"].isin(sectors)].sort_values("date")
    fig = go.Figure()
    for color, (sector, group) in zip(COLORS, selected.groupby("sector"), strict=False):
        fig.add_trace(
            go.Scatter(
                x=group["date"],
                y=group["sentiment"],
                mode="lines",
                name=f"{sector} daily",
                line={"color": color, "width": 0.7},
                opacity=0.18,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:+.3f}<extra></extra>",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=group["date"],
                y=group["sentiment_21d"],
                mode="lines",
                name=f"{sector} 21-day average",
                line={"color": color, "width": 2.3},
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:+.3f}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_color=SLATE, line_width=1, line_dash="dot")
    _theme(fig, height=520)
    fig.update_yaxes(title="VADER compound score", range=[-1, 1], tickformat="+.1f")
    fig.update_xaxes(title="Aligned equity trading date", rangeslider={"visible": True})
    return fig


def fusion_figure(frame: pd.DataFrame) -> go.Figure:
    """Compare base and sentiment-tilted strategy wealth."""

    fig = go.Figure()
    for color, (strategy, group) in zip((NAVY, TEAL), frame.groupby("strategy"), strict=False):
        fig.add_trace(
            go.Scatter(
                x=group["date"],
                y=group["growth_of_1"],
                mode="lines",
                name=str(strategy),
                line={"color": color, "width": 2.5},
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
            )
        )
    _theme(fig)
    fig.update_yaxes(title="Growth of $1", tickprefix="$", tickformat=".2f")
    fig.update_xaxes(title="Out-of-sample date", rangeslider={"visible": True})
    return fig
