"""
This module implements the visualization for the plot_diff function.
"""  # pylint: disable=too-many-lines

from typing import Any, Dict, List, Tuple, Optional

import math
import numpy as np
import pandas as pd
from bokeh.models import (
    HoverTool,
    Panel,
    FactorRange,
)
from bokeh.plotting import Figure, figure
from bokeh.transform import dodge
from bokeh.layouts import row

from ..configs import Config
from ..dtypes import Continuous, DateTime, Nominal, is_dtype
from ..intermediate import Intermediate
from ..palette import CATEGORY10, RDBU
from ..utils import tweak_figure, _format_axis, _format_bin_intervals
from ..correlation.render import create_color_mapper

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
    df_labels: List[str],
    orig: Optional[List[str]] = None,
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
                "orig": orig[i] if orig else None,
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
    if orig:
        if orig != df_labels:
            if x_axis_label:
                x_axis_label += f", this vairable is only in {','.join(orig)}"
            else:
                x_axis_label += f"This vairable is only in {','.join(orig)}"
    fig.xaxis.axis_label = x_axis_label
    fig.xaxis.axis_label_standoff = 0

    return fig


def kde_viz_figure(
    hist: List[Tuple[np.ndarray, np.ndarray]],
    kde: np.ndarray,
    col: str,
    plot_width: int,
    plot_height: int,
    cfg: Config,
) -> Figure:
    """
    Render histogram with overlayed kde
    """
    # pylint: disable=too-many-arguments, too-many-locals
    fig = Figure(
        plot_width=plot_width,
        plot_height=plot_height,
        title=col,
        toolbar_location=None,
        y_axis_type=cfg.kde.yscale,
    )
    for i, (data, kde2) in enumerate(zip(hist, kde)):
        dens, bins = data
        intvls = _format_bin_intervals(bins)
        df = pd.DataFrame(
            {
                "intvl": intvls,
                "left": bins[:-1],
                "right": bins[1:],
                "dens": dens,
            }
        )
        bottom = 0 if cfg.kde.yscale == "linear" or df.empty else df["dens"].min() / 2
        hist = fig.quad(
            source=df,
            left="left",
            right="right",
            bottom=bottom,
            top="dens",
            alpha=0.5,
            fill_color=CATEGORY10[i],
            line_color=CATEGORY10[i],
        )
        hover_hist = HoverTool(
            renderers=[hist],
            tooltips=[("Bin", "@intvl"), ("Density", "@dens")],
            mode="vline",
        )
        pts_rng = np.linspace(df.loc[0, "left"], df.loc[len(df) - 1, "right"], 1000)
        pdf = kde2(pts_rng)
        line = fig.line(x=pts_rng, y=pdf, line_color=CATEGORY10[i], line_width=2, alpha=0.5)
    hover_dist = HoverTool(renderers=[line], tooltips=[("x", "@x"), ("y", "@y")])
    fig.add_tools(hover_hist)
    fig.add_tools(hover_dist)
    tweak_figure(fig, "kde")
    fig.yaxis.axis_label = "Density"
    fig.xaxis.axis_label = col
    _format_axis(fig, df.iloc[0]["left"], df.iloc[-1]["right"], "x")
    if cfg.kde.yscale == "linear":
        _format_axis(fig, 0, max(df["dens"].max(), pdf.max()), "y")
    return fig


def kde_viz_panel(
    hist: List[Tuple[np.ndarray, np.ndarray]],
    kde: np.ndarray,
    col: str,
    plot_width: int,
    plot_height: int,
    cfg: Config,
) -> Panel:
    """
    Render histogram with overlayed kde
    """
    # pylint: disable=too-many-arguments, too-many-locals
    fig = kde_viz_figure(hist, kde, col, plot_width, plot_height, cfg)
    return Panel(child=row(fig), title="KDE Plot")


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


# pylint:disable = unused-argument
def box_viz(
    df_list: List[pd.DataFrame],
    x: str,
    plot_width: int,
    plot_height: int,
    cfg: Config,
    group_all: List[str],
) -> Panel:
    """
    Render a box plot visualization
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-statements

    width, title = 0.7, f"{x}"
    fig = figure(
        plot_width=plot_width,
        plot_height=plot_height,
        title=title,
        toolbar_location=None,
        x_range=group_all,
    )
    for i, df in enumerate(df_list):
        df["x0"], df["x1"] = df.index + 0.2, df.index + 0.8
        low = fig.segment(x0="x0", y0="lw", x1="x1", y1="lw", line_color="black", source=df)
        ltail = fig.segment(x0="grp", y0="lw", x1="grp", y1="q1", line_color="black", source=df)
        lbox = fig.vbar(
            x="grp",
            width=width,
            top="q2",
            bottom="q1",
            fill_color=CATEGORY10[i],
            line_color="black",
            source=df,
        )
        ubox = fig.vbar(
            x="grp",
            width=width,
            top="q3",
            bottom="q2",
            fill_color=CATEGORY10[i],
            line_color="black",
            source=df,
        )
        utail = fig.segment(x0="grp", y0="uw", x1="grp", y1="q3", line_color="black", source=df)
        upw = fig.segment(x0="x0", y0="uw", x1="x1", y1="uw", line_color="black", source=df)

        df.loc[df["otlrs"].isna(), "otlrs"] = pd.Series(
            [[]] * df["otlrs"].isna().sum(), dtype=np.float64
        ).values
        otlrs = [otl for otls in df["otlrs"] for otl in otls]
        if otlrs:
            gps = [grp for grp, ols in zip(df["grp"], df["otlrs"]) for _ in range(len(ols))]
            circ = fig.circle(
                x=gps,
                y=otlrs,
                size=3,
                line_color="black",
                color="black",
                fill_alpha=0.6,
            )
            fig.add_tools(
                HoverTool(
                    renderers=[circ],
                    tooltips=[("Outlier", "@y")],
                )
            )

        tooltips = [
            ("Upper Whisker", "@uw"),
            ("Upper Quartile", "@q3"),
            ("Median", "@q2"),
            ("Lower Quartile", "@q1"),
            ("Lower Whisker", "@lw"),
        ]

        fig.add_tools(
            HoverTool(
                renderers=[upw, utail, ubox, lbox, ltail, low],
                tooltips=tooltips,
            )
        )

    tweak_figure(fig, "box")
    fig.xaxis.major_tick_line_color = None
    fig.xaxis.major_label_text_font_size = "0pt"
    fig.xaxis.axis_label = None
    fig.yaxis.axis_label = x

    # pylint:disable = undefined-loop-variable
    minw = min(otlrs) if otlrs else np.nan
    maxw = max(otlrs) if otlrs else np.nan
    _format_axis(fig, min(df["lw"].min(), minw), max(df["uw"].max(), maxw), "y")

    return Panel(child=row(fig), title="Box Plot")


# pylint:disable = unused-argument
def render_correlation_single_heatmaps(
    df_list: List[Dict[str, pd.DataFrame]], col: str, plot_width: int, plot_height: int, cfg: Config
) -> List[Panel]:
    """
    Render correlation heatmaps, but with single column
    """
    # pylint:disable = too-many-locals
    corr: Dict[str, List[Any]] = {}
    group_all_x = [col + "_" + str(i + 1) for i in range(len(df_list))]
    group_all_y = df_list[0]["Pearson"]["y"].unique()
    for meth in ["Pearson", "Spearman", "KendallTau"]:
        corr[meth] = []
        for i, df in enumerate(df_list):
            df[meth]["x"] = df[meth]["x"] + "_" + str(i + 1)
            corr[meth].append(df[meth])
    tabs: List[Panel] = []
    tooltips = [("y", "@y"), ("correlation", "@correlation{1.11}")]
    for method, dfs in corr.items():
        mapper, color_bar = create_color_mapper(RDBU)

        x_range = FactorRange(*group_all_x)
        y_range = FactorRange(*group_all_y)
        fig = figure(
            x_range=x_range,
            y_range=y_range,
            plot_width=plot_width,
            plot_height=plot_height,
            x_axis_location="below",
            tools="hover",
            toolbar_location=None,
            tooltips=tooltips,
            title=" ",
        )

        tweak_figure(fig)
        for df in dfs:
            fig.rect(
                x="x",
                y="y",
                width=0.7,
                height=1,
                source=df,
                fill_color={"field": "correlation", "transform": mapper},
                line_color=None,
            )

        fig.add_layout(color_bar, "right")
        tab = Panel(child=fig, title=method)
        tabs.append(tab)
        for panel in tabs:
            panel.child.frame_width = int(plot_width * 0.9)
    return tabs


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


def format_num_stats(data: Dict[str, List[Any]]) -> Dict[str, Dict[str, List[Any]]]:
    """
    Format numerical statistics
    """
    overview = {
        "Approximate Distinct Count": [round(nuniq) for nuniq in data["nuniq"]],
        "Approximate Unique (%)": [
            nuniq / npres for nuniq, npres in zip(data["nuniq"], data["npres"])
        ],
        "Missing": [nrows - npres for nrows, npres in zip(data["nrows"], data["npres"])],
        "Missing (%)": [1 - npres / nrows for npres, nrows in zip(data["npres"], data["nrows"])],
        "Infinite": [npres - nreals for npres, nreals in zip(data["npres"], data["nreals"])],
        "Infinite (%)": [
            (npres - nreals) / nrows
            for npres, nreals, nrows in zip(data["npres"], data["nreals"], data["nrows"])
        ],
        "Memory Size": data["mem_use"],
        "Mean": data["mean"],
        "Minimum": data["min"],
        "Maximum": data["max"],
        "Zeros": data["nzero"],
        "Zeros (%)": [nzero / nrows for nzero, nrows in zip(data["nzero"], data["nrows"])],
        "Negatives": data["nneg"],
        "Negatives (%)": [nneq / nrows for nneq, nrows in zip(data["nneg"], data["nrows"])],
    }
    for qntls in data["qntls"]:
        qntls.index = np.round(qntls.index, 2)
    quantile = {
        "Minimum": data["min"],
        "5-th Percentile": [qntls.loc[0.05] for qntls in data["qntls"]],
        "Q1": [qntls.loc[0.25] for qntls in data["qntls"]],
        "Median": [qntls.loc[0.50] for qntls in data["qntls"]],
        "Q3": [qntls.loc[0.75] for qntls in data["qntls"]],
        "95-th Percentile": [qntls.loc[0.95] for qntls in data["qntls"]],
        "Maximum": data["max"],
        "Range": [max_value - min_value for max_value, min_value in zip(data["max"], data["min"])],
        "IQR": [qntls.loc[0.75] - qntls.loc[0.25] for qntls in data["qntls"]],
    }
    descriptive = {
        "Mean": data["mean"],
        "Standard Deviation": data["std"],
        "Variance": [std**2 for std in data["std"]],
        "Sum": [mean * npres for mean, npres in zip(data["mean"], data["npres"])],
        "Skewness": [float(skew) for skew in data["skew"]],
        "Kurtosis": [float(kurt) for kurt in data["kurt"]],
        "Coefficient of Variation": [
            std / mean if mean != 0 else np.nan for mean, std in zip(data["mean"], data["std"])
        ],
    }
    print(
        {
            "Overview": {k: _format_values(k, v) for k, v in overview.items()},
            "Quantile Statistics": {k: _format_values(k, v) for k, v in quantile.items()},
            "Descriptive Statistics": {k: _format_values(k, v) for k, v in descriptive.items()},
        }
    )
    return {
        "Overview": {k: _format_values(k, v) for k, v in overview.items()},
        "Quantile Statistics": {k: _format_values(k, v) for k, v in quantile.items()},
        "Descriptive Statistics": {k: _format_values(k, v) for k, v in descriptive.items()},
    }


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
        fig = None
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
            if cfg.diff.density:
                kde, dens = data
                if kde is not None and not isinstance(dens, np.integer):
                    fig = kde_viz_figure(dens, kde, col, plot_width, plot_height, cfg)
            else:
                fig = hist_viz(
                    data,
                    nrows,
                    col,
                    cfg.hist.yscale,
                    plot_width,
                    plot_height,
                    False,
                    df_labels,
                    orig,
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
        if fig is not None:
            fig.frame_height = plot_height
            titles.append(fig.title.text)
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


def render_comparison_continous(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Render for continuous variable comparison
    """
    # pylint:disable = too-many-locals
    plot_width = cfg.plot.width if cfg.plot.width is not None else 450
    plot_height = cfg.plot.height if cfg.plot.height is not None else 400
    df_labels: List[str] = cfg.diff.label  # type: ignore
    # baseline: int = cfg.diff.baseline
    tabs: List[Panel] = []
    htgs: Dict[str, List[Tuple[str, str]]] = {}
    col, data = itmdt["col"], itmdt["data"][0]
    if cfg.hist.enable:
        nrows = itmdt["stats"]["nrows"]
        fig = hist_viz(
            data["hist"], nrows, col, cfg.hist.yscale, plot_width, plot_height, False, df_labels
        )
        tabs.append(Panel(child=row(fig), title="Histogram"))
        # htgs["Histogram"] = cfg.hist.how_to_guide(plot_height, plot_width)
    if cfg.kde.enable:
        if data["kde"] is not None and (
            not math.isclose(itmdt["stats"]["min"][0], itmdt["stats"]["max"][0])
        ):
            dens, kde = data["dens"], data["kde"]
            tabs.append(kde_viz_panel(dens, kde, col, plot_width, plot_height, cfg))
            # htgs["KDE Plot"] = cfg.kde.how_to_guide(plot_height, plot_width)
    if cfg.box.enable:
        df_list = []
        group_all = []
        for i, data_box in enumerate(data["box"]):
            box_data = {
                "grp": col + str(i),
                "q1": data_box["qrtl1"],
                "q2": data_box["qrtl2"],
                "q3": data_box["qrtl3"],
                "lw": data_box["lw"],
                "uw": data_box["uw"],
                "otlrs": [data_box["otlrs"]],
            }
            df_list.append(pd.DataFrame(box_data, index=[i]))
            group_all.append(box_data["grp"])
        tabs.append(box_viz(df_list, col, plot_width, plot_height, cfg, group_all))
        # htgs["Box Plot"] = cfg.box.univar_how_to_guide(plot_height, plot_width)
    for panel in tabs:
        panel.child.children[0].frame_width = int(plot_width * 0.9)
    if cfg.correlations.enable:
        tabs = tabs + (
            render_correlation_single_heatmaps(data["corr"], col, plot_width, plot_height, cfg)
        )

    # pylint:disable=line-too-long
    legend_lables = [
        {"label": label, "color": color}
        for label, color in zip(cfg.diff.label, CATEGORY10[: len(cfg.diff.label)])  # type: ignore
    ]
    return {
        "comparison_stats": format_num_stats(itmdt["stats"]) if cfg.stats.enable else [],
        "value_table": [],
        "insights": [],
        "layout": [panel.child for panel in tabs],
        "meta": ["Stats"] + [tab.title for tab in tabs],
        "container_width": plot_width + 110,
        "how_to_guide": htgs,
        "df_labels": cfg.diff.label,
        "legend_labels": legend_lables,
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
    if itmdt.visual_type == "comparison_continuous":
        visual_elem = render_comparison_continous(itmdt, cfg)
    return visual_elem
