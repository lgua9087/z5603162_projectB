"""Report-ready figure suite for MarketReady Funds."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

NAVY = "#16324F"
TEAL = "#2A9D8F"
GOLD = "#D99B2B"
RED = "#C44E52"
SLATE = "#64748B"
LIGHT = "#E8EEF4"
COLORS = [NAVY, TEAL, GOLD, RED, "#6B5B95", "#4A7C59", "#B56576", "#577590"]
SOURCE = "Source: FINS5545 provided data; author calculations. Source sample: 2020-2023."


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "axes.edgecolor": "#AAB7C4",
            "axes.labelcolor": NAVY,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "figure.facecolor": "white",
            "font.family": "DejaVu Sans",
            "grid.color": LIGHT,
            "grid.linewidth": 0.8,
            "savefig.bbox": "tight",
            "savefig.dpi": 220,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
        }
    )


def _finish(
    fig: plt.Figure,
    output_dir: Path,
    filename: str,
    *,
    title: str,
    subtitle: str,
) -> Path:
    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=15, weight="bold", color=NAVY)
    fig.text(0.01, 0.947, subtitle, ha="left", va="top", fontsize=9.5, color=SLATE)
    fig.text(0.01, 0.012, SOURCE, ha="left", va="bottom", fontsize=7.5, color=SLATE)
    fig.tight_layout(rect=(0, 0.055, 1, 0.90))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_growth_of_dollar(fund_returns: pd.DataFrame, output_dir: Path) -> Path:
    """Compare four portfolio methods inside each asset family."""

    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5), sharey=False)
    for axis, family in zip(axes, ("Equity", "Crypto", "Combined"), strict=True):
        selected = fund_returns.loc[fund_returns["family"].eq(family)].sort_values("date")
        for color, (method, group) in zip(COLORS, selected.groupby("method"), strict=False):
            axis.plot(
                group["date"],
                group["growth_of_1"],
                label=method,
                color=color,
                linewidth=1.6,
            )
        axis.set_title(f"{family} funds", loc="left", fontsize=11, weight="bold")
        axis.set_yscale("log")
        currency = FuncFormatter(lambda value, _: f"${value:,.2f}")
        axis.yaxis.set_major_formatter(currency)
        axis.yaxis.set_minor_formatter(currency)
        axis.xaxis.set_major_locator(mdates.YearLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.grid(axis="y")
        axis.set_xlabel("Out-of-sample date")
        axis.legend(frameon=False, fontsize=7.5, loc="best")
    axes[0].set_ylabel("Growth of $1, log scale")
    return _finish(
        fig,
        output_dir,
        "growth_of_dollar.png",
        title="Portfolio method matters, but the asset family dominates the range",
        subtitle=(
            "Net growth after 5 bp per unit of target turnover; walk-forward monthly "
            "rebalancing with trailing information only"
        ),
    )


def plot_drawdowns(
    fund_returns: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Show the highest-Sharpe fund in each family through its drawdowns."""

    _apply_style()
    leaders = metrics.loc[metrics.groupby("family")["sharpe_ratio"].idxmax()]
    fig, axes = plt.subplots(3, 1, figsize=(9.4, 7.2), sharex=False)
    for axis, (_, leader), color in zip(axes, leaders.iterrows(), COLORS, strict=False):
        group = fund_returns.loc[fund_returns["fund_id"].eq(leader["fund_id"])].sort_values(
            "date"
        )
        axis.fill_between(group["date"], group["drawdown"], 0, color=color, alpha=0.24)
        axis.plot(group["date"], group["drawdown"], color=color, linewidth=1.2)
        axis.set_title(str(leader["fund_name"]), loc="left", fontsize=10, weight="bold")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.xaxis.set_major_locator(mdates.YearLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.grid(axis="y")
    axes[1].set_ylabel("Drawdown from prior wealth peak")
    axes[-1].set_xlabel("Out-of-sample date")
    return _finish(
        fig,
        output_dir,
        "drawdown.png",
        title="The best Sharpe ratio in each family still contains material losses",
        subtitle=(
            "Drawdown paths for the highest net-Sharpe fund within equity, crypto, "
            "and combined products"
        ),
    )


def plot_weights_over_time(fund_weights: pd.DataFrame, output_dir: Path) -> Path:
    """Compare combined-fund equity and crypto allocations across four methods."""

    _apply_style()
    combined = fund_weights.loc[fund_weights["family"].eq("Combined")]
    grouped = (
        combined.groupby(["method", "rebalance_date", "asset_class"], observed=True)["weight"]
        .sum()
        .reset_index()
    )
    methods = list(grouped["method"].drop_duplicates())
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex=True, sharey=True)
    for axis, method in zip(axes.flat, methods, strict=False):
        data = grouped.loc[grouped["method"].eq(method)]
        for asset_class, color in (("Equity", TEAL), ("Crypto", GOLD)):
            part = data.loc[data["asset_class"].eq(asset_class)]
            axis.plot(
                part["rebalance_date"],
                part["weight"],
                color=color,
                linewidth=1.8,
                label=asset_class,
            )
        axis.set_title(method, loc="left", fontsize=10, weight="bold")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.xaxis.set_major_locator(mdates.YearLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.grid(axis="y")
        axis.legend(frameon=False, fontsize=8, loc="best")
    axes[0, 0].set_ylabel("Target asset-family weight")
    axes[1, 0].set_ylabel("Target asset-family weight")
    axes[1, 0].set_xlabel("Rebalance date")
    axes[1, 1].set_xlabel("Rebalance date")
    return _finish(
        fig,
        output_dir,
        "weights_over_time.png",
        title="Optimization changes the equity/crypto mix through time",
        subtitle="Target asset-family weights for the four combined funds at monthly rebalances",
    )


def plot_sharpe(metrics: pd.DataFrame, output_dir: Path) -> Path:
    """Rank all fund products by net out-of-sample Sharpe ratio."""

    _apply_style()
    ordered = metrics.sort_values("sharpe_ratio")
    color_map = {"Equity": TEAL, "Crypto": GOLD, "Combined": NAVY}
    fig, axis = plt.subplots(figsize=(9.2, 6.2))
    bars = axis.barh(
        ordered["fund_name"],
        ordered["sharpe_ratio"],
        color=ordered["family"].map(color_map),
        alpha=0.92,
    )
    axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    axis.axvline(0, color=SLATE, linewidth=0.8)
    axis.set_xlabel("Annualized Sharpe ratio, 0% risk-free rate")
    axis.set_ylabel("")
    axis.grid(axis="x")
    axis.margins(x=0.10)
    return _finish(
        fig,
        output_dir,
        "sharpe_barplot.png",
        title="Risk-adjusted outcomes vary across both methods and asset families",
        subtitle="Net walk-forward Sharpe ratios after 5 bp per unit of target turnover",
    )


def plot_sector_sentiment(index: pd.DataFrame, output_dir: Path) -> Path:
    """Separate high- and low-coverage sectors so the sentiment paths remain readable."""

    _apply_style()
    coverage = index.groupby("sector", observed=True)["coverage_rate"].mean().sort_values(
        ascending=False
    )
    groups = (coverage.index[:5], coverage.index[5:])
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True, sharey=True)
    for axis, sectors, title in zip(
        axes,
        groups,
        ("Five higher-coverage sectors", "Five lower-coverage sectors"),
        strict=True,
    ):
        for color, sector in zip(COLORS, sectors, strict=False):
            part = index.loc[index["sector"].eq(sector)].sort_values("date")
            axis.plot(
                part["date"],
                part["sentiment_21d"],
                color=color,
                linewidth=1.25,
                label=f"{sector} ({coverage[sector]:.0%} coverage)",
                alpha=0.9,
            )
        axis.axhline(0, color=SLATE, linewidth=0.8, linestyle="--")
        axis.set_title(title, loc="left", fontsize=10, weight="bold")
        axis.legend(frameon=False, ncol=2, fontsize=7.5, loc="best")
        axis.grid(axis="y")
        axis.set_ylim(-0.45, 0.45)
    axes[0].set_ylabel("21-day VADER compound")
    axes[1].set_ylabel("21-day VADER compound")
    axes[1].set_xlabel("Aligned equity trading date")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return _finish(
        fig,
        output_dir,
        "sector_sentiment_index.png",
        title="Sector sentiment must be read beside unequal news coverage",
        subtitle=(
            "Finance-extended VADER; ticker-day means are equally weighted within sector; "
            "bold paths are 21-trading-day averages"
        ),
    )


def plot_fusion(fusion_returns: pd.DataFrame, output_dir: Path) -> Path:
    """Compare the base equity fund with the coverage-aware sentiment tilt."""

    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    for color, (strategy, group) in zip(
        (NAVY, TEAL), fusion_returns.groupby("strategy"), strict=False
    ):
        ordered = group.sort_values("date")
        axes[0].plot(
            ordered["date"],
            ordered["growth_of_1"],
            color=color,
            linewidth=2,
            label=strategy,
        )
        axes[1].plot(
            ordered["date"],
            ordered["drawdown"],
            color=color,
            linewidth=1.5,
            label=strategy,
        )
    axes[0].set_ylabel("Growth of $1")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.2f}"))
    axes[1].set_ylabel("Drawdown from prior peak")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    for axis in axes:
        axis.set_xlabel("Out-of-sample date")
        axis.xaxis.set_major_locator(mdates.YearLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.grid(axis="y")
        axis.legend(frameon=False, fontsize=8, loc="best")
    return _finish(
        fig,
        output_dir,
        "fusion_comparison.png",
        title="The sentiment tilt changes both return and drawdown, not only holdings",
        subtitle=(
            "Base equity risk parity versus a one-day-lagged, 21-day, coverage-aware "
            "ticker sentiment tilt; both include turnover costs"
        ),
    )


def build_all_figures(
    fund_returns: pd.DataFrame,
    fund_weights: pd.DataFrame,
    metrics: pd.DataFrame,
    sentiment_index: pd.DataFrame,
    fusion_returns: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Generate every required Part B analytical figure."""

    destination = Path(output_dir)
    return [
        plot_growth_of_dollar(fund_returns, destination),
        plot_drawdowns(fund_returns, metrics, destination),
        plot_weights_over_time(fund_weights, destination),
        plot_sharpe(metrics, destination),
        plot_sector_sentiment(sentiment_index, destination),
        plot_fusion(fusion_returns, destination),
    ]
