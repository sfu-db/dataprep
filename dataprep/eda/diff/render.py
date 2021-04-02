"""
This module implements the visualization for the plot_diff function.
"""  # pylint: disable=too-many-lines
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from bokeh.models import (
    HoverTool,
)
from bokeh.plotting import Figure
from bokeh.transform import dodge

from ..configs import Config
from ..dtypes import Continuous, DateTime, Nominal, is_dtype
from ..intermediate import Intermediate
from ..palette import CATEGORY10
from ..utils import tweak_figure, _format_axis, _format_bin_intervals

__all__ = ["render_diff"]


def _format_values(key: str, value: List[Any]) -> List[str]:
    for i, _ in enumerate(value):
        if not isinstance(value[i], (int, float)):
            # if value is a time
            value[i] = str(value[i])
            continue

        if "Memory" in key:
            # for memory usage
            ind = 0
            unit = dict(enumerate(["B", "KB", "MB", "GB", "TB"], 0))
            while value[i] > 1024:
                value[i] /= 1024
                ind += 1
            value[i] = f"{value[i]:.1f} {unit[ind]}"
            continue

        if (value[i] * 10) % 10 == 0:
            # if value is int but in a float form with 0 at last digit
            val = int(value[i])
            if abs(val) >= 1000000:
                val = f"{val:.5g}"  # type: ignore
        elif abs(value[i]) >= 1000000 or abs(value[i]) < 0.001:  # type: ignore
            val = f"{value[i]:.5g}"  # type: ignore
        elif abs(value[i]) >= 1:  # type: ignore
            # eliminate trailing zeros
            pre_value = float(f"{value[i]:.4f}")
            val = int(pre_value) if (pre_value * 10) % 10 == 0 else pre_value  # type: ignore
        elif 0.001 <= abs(value[i]) < 1:  # type: ignore
            val = f"{value[i]:.4g}"  # type: ignore
        else:
            val = str(value[i])  # type: ignore

        if "%" in key:
            # for percentage, only use digits before notation sign for extreme small number
            val = f"{float(val):.1%}"  # type: ignore
        value[i] = str(val)
        continue
    return value


def bar_viz(
    df: List[pd.DataFrame],
    ttl_grps: int,
    nrows: List[int],
    col: str,
    yscale: str,
    plot_width: int,
    plot_height: int,
    show_yticks: bool,
    orig: List[str],
    df_labels: List[str],
    baseline: int,
) -> Figure:
    """
    Render a bar chart
    """
    # pylint: disable=too-many-arguments, too-many-locals
    if len(df) > 1:
        for i, _ in enumerate(df):
            df[i] = df[i].reindex(index=df[baseline].index, fill_value=0).to_frame()

    tooltips = [
        (col, "@index"),
        ("Count", f"@{{{col}}}"),
        ("Percent", "@pct{0.2f}%"),
        ("Source", "@orig"),
    ]

    if show_yticks:
        if len(df[baseline]) > 10:
            plot_width = 28 * len(df[baseline])
    fig = Figure(
        plot_width=plot_width,
        plot_height=plot_height,
        title=col,
        toolbar_location=None,
        tooltips=tooltips,
        tools="hover",
        x_range=list(df[baseline].index),
        y_axis_type=yscale,
    )

    offset = np.linspace(-0.08 * len(df), 0.08 * len(df), len(df)) if len(df) > 1 else [0]
    for i, (nrow, data) in enumerate(zip(nrows, df)):
        data["pct"] = data[col] / nrow * 100
        data.index = [str(val) for val in data.index]
        data["orig"] = orig[i]

        fig.vbar(
            x=dodge("index", offset[i], range=fig.x_range),
            width=0.6 / len(df),
            top=col,
            bottom=0.01,
            source=data,
            fill_color=CATEGORY10[i],
            line_color=CATEGORY10[i],
        )
    tweak_figure(fig, "bar", show_yticks)

    fig.yaxis.axis_label = "Count"

    x_axis_label = ""
    if ttl_grps > len(df[baseline]):
        x_axis_label += f"Top {len(df[baseline])} of {ttl_grps} {col}"

    if orig != df_labels:
        if x_axis_label:
            x_axis_label += f", this vairable is only in {','.join(orig)}"
        else:
            x_axis_label += f"This vairable is only in {','.join(orig)}"

    fig.xaxis.axis_label = x_axis_label
    fig.xaxis.axis_label_standoff = 0

    if show_yticks and yscale == "linear":
        _format_axis(fig, 0, df[baseline].max(), "y")
    return fig


def hist_viz(
    hist: List[Tuple[np.ndarray, np.ndarray]],
    nrows: List[int],
    col: str,
    yscale: str,
    plot_width: int,
    plot_height: int,
    show_yticks: bool,
    orig: List[str],
    df_labels: List[str],
) -> Figure:
    """
    Render a histogram
    """
    # pylint: disable=too-many-arguments,too-many-locals

    tooltips = [
        ("Bin", "@intvl"),
        ("Frequency", "@freq"),
        ("Percent", "@pct{0.2f}%"),
        ("Source", "@orig"),
    ]
    fig = Figure(
        plot_height=plot_height,
        plot_width=plot_width,
        title=col,
        toolbar_location=None,
        y_axis_type=yscale,
    )

    for i, hst in enumerate(hist):
        counts, bins = hst
        if sum(counts) == 0:
            fig.rect(x=0, y=0, width=0, height=0)
            continue
        intvls = _format_bin_intervals(bins)
        df = pd.DataFrame(
            {
                "intvl": intvls,
                "left": bins[:-1],
                "right": bins[1:],
                "freq": counts,
                "pct": counts / nrows[i] * 100,
                "orig": orig[i],
            }
        )
        bottom = 0 if yscale == "linear" or df.empty else counts.min() / 2
        fig.quad(
            source=df,
            left="left",
            right="right",
            bottom=bottom,
            alpha=0.5,
            top="freq",
            fill_color=CATEGORY10[i],
            line_color=CATEGORY10[i],
        )
    hover = HoverTool(tooltips=tooltips, attachment="vertical", mode="vline")
    fig.add_tools(hover)

    tweak_figure(fig, "hist", show_yticks)
    fig.yaxis.axis_label = "Frequency"

    _format_axis(fig, df.iloc[0]["left"], df.iloc[-1]["right"], "x")

    x_axis_label = ""
    if show_yticks:
        x_axis_label += col
        if yscale == "linear":
            _format_axis(fig, 0, df["freq"].max(), "y")

    if orig != df_labels:
        if x_axis_label:
            x_axis_label += f", this vairable is only in {','.join(orig)}"
        else:
            x_axis_label += f"This vairable is only in {','.join(orig)}"
    fig.xaxis.axis_label = x_axis_label
    fig.xaxis.axis_label_standoff = 0

    return fig


def dt_line_viz(
    df: List[pd.DataFrame],
    x: str,
    timeunit: str,
    yscale: str,
    plot_width: int,
    plot_height: int,
    show_yticks: bool,
    orig: List[str],
    df_labels: List[str],
) -> Figure:
    """
    Render a line chart
    """
    # pylint: disable=too-many-arguments

    tooltips = [(timeunit, "@lbl"), ("Frequency", "@freq"), ("Source", "@orig")]
    for i, _ in enumerate(df):
        df[i]["orig"] = orig[i]

    fig = Figure(
        plot_width=plot_width,
        plot_height=plot_height,
        toolbar_location=None,
        title=x,
        tools=[],
        y_axis_type=yscale,
        x_axis_type="datetime",
    )

    for i, _ in enumerate(df):
        fig.line(
            source=df[i],
            x=x,
            y="freq",
            line_width=2,
            line_alpha=0.8,
            color=CATEGORY10[i],
        )
    hover = HoverTool(
        tooltips=tooltips,
        mode="vline",
    )
    fig.add_tools(hover)

    tweak_figure(fig, "line", show_yticks)
    if show_yticks and yscale == "linear":
        _format_axis(fig, 0, max([d["freq"].max() for d in df]), "y")

    if orig != df_labels:
        fig.xaxis.axis_label = f"This variable is only in {','.join(orig)}"
        fig.xaxis.axis_label_standoff = 0
    fig.yaxis.axis_label = "Frequency"
    return fig


def format_ov_stats(stats: Dict[str, List[Any]]) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    """
    Render statistics information for distribution grid
    """
    # pylint: disable=too-many-locals
    nrows, ncols, npresent_cells, nrows_wo_dups, mem_use, dtypes_cnt = stats.values()
    ncells = np.multiply(nrows, ncols).tolist()

    data = {
        "Number of Variables": ncols,
        "Number of Rows": nrows,
        "Missing Cells": np.subtract(ncells, npresent_cells).astype(float).tolist(),
        "Missing Cells (%)": np.subtract(1, np.divide(npresent_cells, ncells)).tolist(),
        "Duplicate Rows": np.subtract(nrows, nrows_wo_dups).tolist(),
        "Duplicate Rows (%)": np.subtract(1, np.divide(nrows_wo_dups, nrows)).tolist(),
        "Total Size in Memory": list(map(float, mem_use)),
        "Average Row Size in Memory": np.subtract(mem_use, nrows).tolist(),
    }
    return {k: _format_values(k, v) for k, v in data.items()}, dtypes_cnt  # type: ignore


def render_comparison_grid(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Create visualizations for plot(df)
    """
    # pylint: disable=too-many-locals, line-too-long
    plot_width = cfg.plot.width if cfg.plot.width is not None else 324
    plot_height = cfg.plot.height if cfg.plot.height is not None else 300
    df_labels: List[str] = cfg.diff.label  # type: ignore
    baseline: int = cfg.diff.baseline

    figs: List[Figure] = []
    nrows = itmdt["stats"]["nrows"]
    titles: List[str] = []
    for col, dtp, data, orig in itmdt["data"]:
        if is_dtype(dtp, Nominal()):
            df, ttl_grps = data
            fig = bar_viz(
                list(df),
                ttl_grps,
                nrows,
                col,
                cfg.bar.yscale,
                plot_width,
                plot_height,
                False,
                orig,
                df_labels,
                baseline if len(df) > 1 else 0,
            )
        elif is_dtype(dtp, Continuous()):
            fig = hist_viz(
                data, nrows, col, cfg.hist.yscale, plot_width, plot_height, False, orig, df_labels
            )
        elif is_dtype(dtp, DateTime()):
            df, timeunit = data
            fig = dt_line_viz(
                list(df),
                col,
                timeunit,
                cfg.line.yscale,
                plot_width,
                plot_height,
                False,
                orig,
                df_labels,
            )
        fig.frame_height = plot_height
        titles.append(fig.title.text)
        fig.title.text = ""
        figs.append(fig)

    if cfg.stats.enable:
        toggle_content = "Stats"
    else:
        toggle_content = None  # type: ignore

    return {
        "layout": figs,
        "meta": titles,
        "comparison_stats": format_ov_stats(itmdt["stats"]) if cfg.stats.enable else None,
        "container_width": plot_width * 3,
        "toggle_content": toggle_content,
        "df_labels": cfg.diff.label,
        "legend_labels": [
            {"label": label, "color": color}
            for label, color in zip(cfg.diff.label, CATEGORY10[: len(cfg.diff.label)])  # type: ignore
        ],
        "baseline": baseline,
    }


def render_diff(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Render a basic plot

    Parameters
    ----------
    itmdt
        The Intermediate containing results from the compute function.
    cfg
        Config instance
    """

    if itmdt.visual_type == "comparison_grid":
        visual_elem = render_comparison_grid(itmdt, cfg)

    return visual_elem
