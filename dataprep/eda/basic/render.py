"""
    This module implements the visualization for the
    plot(df) function
"""
from math import pi
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from bokeh.models import (
    Box,
    FuncTickFormatter,
    PrintfTickFormatter,
    BasicTicker,
    HoverTool,
    LayoutDOM,
    Legend,
    LegendItem,
    Panel,
    Tabs,
    ColumnDataSource,
    LinearColorMapper,
    ColorBar,
    FactorRange,
)
from bokeh.plotting import Figure, gridplot, figure
from bokeh.transform import cumsum, linear_cmap, transform
from bokeh.util.hex import hexbin
from bokeh.palettes import viridis, Pastel1  # pylint: disable=E0611 # type: ignore

from ..intermediate import Intermediate
from ..dtypes import DType
from ..palette import PALETTE, BIPALETTE

__all__ = ["render"]


def tweak_figure(
    fig: Figure,
    ptype: Optional[str] = None,
    show_yaxis: bool = False,
    max_lbl_len: int = 15,
) -> None:
    """
    Set some common attributes for a figure
    """
    fig.axis.major_label_text_font_size = "9pt"
    fig.xaxis.major_label_orientation = pi / 3
    fig.title.text_font_size = "10pt"
    if ptype in ["bar", "pie", "hist", "kde", "qq", "box", "hex", "heatmap"]:
        fig.grid.grid_line_color = None
        fig.axis.minor_tick_line_color = None
    if ptype in ["bar", "hist"] and not show_yaxis:
        fig.yaxis.major_label_text_font_size = "0pt"
        fig.yaxis.major_tick_line_color = None
    if ptype in ["bar", "nested", "stacked", "heatmap", "box"]:
        fig.xaxis.formatter = FuncTickFormatter(
            code="""
            if (tick.length > %d) return tick.substring(0, %d-2) + '...';
            else return tick;
        """
            % (max_lbl_len, max_lbl_len)
        )
    if ptype in ["nested", "stacked"]:
        fig.y_range.start = 0
        fig.x_range.range_padding = 0.03
        fig.xgrid.grid_line_color = None


def _make_title(grp_cnt_stats: Dict[str, int], x: str, y: str) -> str:
    """
    Format the title to notify the user of sampled output
    """

    x_ttl = grp_cnt_stats["x_ttl"]
    x_show = grp_cnt_stats["x_show"]
    if "y_ttl" in grp_cnt_stats:
        y_ttl = grp_cnt_stats["y_ttl"]
        y_show = grp_cnt_stats["y_show"]

        if x_ttl > x_show and y_ttl > y_show:
            return "(top {} out of {}) {} by (top {} out of {}) {}".format(
                y_show, y_ttl, y, x_show, x_ttl, x
            )
        if y_ttl > y_show:
            return "(top {} out of {}) {} by {}".format(y_show, y_ttl, y, x)
    if x_ttl > x_show:
        return "{} by (top {} out of {}) {}".format(y, x_show, x_ttl, x)
    return f"{y} by {x}"


def _format_xaxis(fig: Figure, minv: int, maxv: int) -> None:
    """
    Format the x axis for histograms
    """  # pylint: disable=too-many-locals
    num_x_ticks = 5
    # divisor for 5 ticks (5 results in ticks that are too close together)
    divisor = 4.5
    # interval
    gap = (maxv - minv) / divisor
    # get exponent from scientific notation
    _, after = f"{gap:.0e}".split("e")
    # round to this amount
    round_to = -1 * int(after)
    # round the first x tick
    minv = np.round(minv, round_to)
    # round value between ticks
    gap = np.round(gap, round_to)

    # make the tick values
    ticks = [minv + i * gap for i in range(num_x_ticks)]
    ticks = np.round(ticks, round_to)
    ticks = [int(tick) if tick.is_integer() else tick for tick in ticks]
    fig.xaxis.ticker = ticks

    formatted_ticks = []
    for tick in ticks:  # format the tick values
        before, after = f"{tick:e}".split("e")
        if float(after) > 1e15 or abs(tick) < 1e4:
            formatted_ticks.append(str(tick))
            continue
        mod_exp = int(after) % 3
        factor = 1 if mod_exp == 0 else 10 if mod_exp == 1 else 100
        value = np.round(float(before) * factor, len(str(before)))
        value = int(value) if value.is_integer() else value
        if abs(tick) >= 1e12:
            formatted_ticks.append(str(value) + "T")
        elif abs(tick) >= 1e9:
            formatted_ticks.append(str(value) + "B")
        elif abs(tick) >= 1e6:
            formatted_ticks.append(str(value) + "M")
        elif abs(tick) >= 1e4:
            formatted_ticks.append(str(value) + "K")

    fig.xaxis.major_label_overrides = dict(zip(ticks, formatted_ticks))
    fig.xaxis.major_label_text_font_size = "10pt"
    fig.xaxis.major_label_standoff = 7
    fig.xaxis.major_label_orientation = 0
    fig.axis.major_tick_line_color = None


def bar_viz(
    df: pd.DataFrame,
    total_grps: int,
    miss_pct: float,
    col: str,
    yscale: str,
    plot_width: int,
    plot_height: int,
    show_yaxis: bool,
) -> Figure:
    """
    Render a bar chart
    """
    # pylint: disable=too-many-arguments
    title = f"{col} ({miss_pct}% missing)" if miss_pct > 0 else f"{col}"
    tooltips = [(f"{col}", "@col"), ("Count", "@cnt"), ("Percent", "@pct{0.2f}%")]
    if show_yaxis:
        if len(df) > 10:
            plot_width = 28 * len(df)
    fig = Figure(
        x_range=list(df["col"]),
        title=title,
        plot_width=plot_width,
        plot_height=plot_height,
        y_axis_type=yscale,
        tools="hover",
        toolbar_location=None,
        tooltips=tooltips,
    )
    fig.vbar(x="col", width=0.9, top="cnt", bottom=0.01, source=df)
    tweak_figure(fig, "bar", show_yaxis)
    fig.yaxis.axis_label = "Count"
    if total_grps > len(df):
        fig.xaxis.axis_label = f"Top {len(df)} of {total_grps} {col}"
        fig.xaxis.axis_label_standoff = 0
    return fig


def pie_viz(
    df: pd.DataFrame, col: str, miss_pct: float, plot_width: int, plot_height: int,
) -> Panel:
    """
    Render a pie chart
    """
    title = f"{col} ({miss_pct}% missing)" if miss_pct > 0 else f"{col}"
    tooltips = [(f"{col}", "@col"), ("Count", "@cnt"), ("Percent", "@pct{0.2f}%")]
    df["angle"] = df["cnt"] / df["cnt"].sum() * 2 * pi
    fig = Figure(
        title=title,
        plot_width=plot_width,
        plot_height=plot_height,
        tools="hover",
        toolbar_location=None,
        tooltips=tooltips,
    )
    color_list = PALETTE * (len(df) // len(PALETTE) + 1)
    df["colour"] = color_list[0 : len(df)]
    if df.iloc[-1]["cnt"] == 0:  # no "Others" group
        df = df[:-1]
    pie = fig.wedge(
        x=0,
        y=1,
        radius=0.9,
        start_angle=cumsum("angle", include_zero=True),
        end_angle=cumsum("angle"),
        line_color="white",
        fill_color="colour",
        source=df,
    )
    legend = Legend(items=[LegendItem(label=dict(field="col"), renderers=[pie])])
    legend.label_text_font_size = "8pt"
    fig.add_layout(legend, "right")
    tweak_figure(fig, "pie")
    fig.axis.major_label_text_font_size = "0pt"
    fig.axis.major_tick_line_color = None
    return Panel(child=fig, title="pie chart")


def hist_viz(
    df: pd.DataFrame,
    miss_pct: float,
    col: str,
    yscale: str,
    plot_width: int,
    plot_height: int,
    show_yaxis: bool,
) -> Figure:
    """
    Render a histogram
    """
    # pylint: disable=too-many-arguments
    title = f"{col} ({miss_pct}% missing)" if miss_pct > 0 else f"{col}"
    tooltips = [
        ("Bin", "@intervals"),
        ("Frequency", "@freq"),
        ("Percent", "@pct{0.2f}%"),
    ]
    fig = Figure(
        plot_width=plot_width,
        plot_height=plot_height,
        toolbar_location=None,
        title=title,
        tools=[],
        y_axis_type=yscale,
    )
    fig.quad(
        source=df,
        left="left",
        right="right",
        bottom=0.01,
        alpha=0.5,
        top="freq",
        fill_color="#6baed6",
    )
    hover = HoverTool(tooltips=tooltips, mode="vline",)
    fig.add_tools(hover)
    tweak_figure(fig, "hist", show_yaxis)
    fig.yaxis.axis_label = "Frequency"
    if not df.empty:
        minv = df.iloc[0]["left"]
        maxv = df.iloc[-1]["right"]
        _format_xaxis(fig, minv, maxv)

    return fig


def hist_kde_viz(
    df: pd.DataFrame,
    pts_rng: np.ndarray,
    pdf: np.ndarray,
    col: str,
    yscale: str,
    plot_width: int,
    plot_height: int,
) -> Panel:
    """
    Render histogram with overlayed kde
    """
    # pylint: disable=too-many-arguments
    fig = Figure(
        plot_width=plot_width,
        plot_height=plot_height,
        title=f"{col}",
        tools=[],
        toolbar_location=None,
        y_axis_type=yscale,
    )
    hist = fig.quad(
        source=df,
        left="left",
        right="right",
        bottom=0,
        alpha=0.5,
        top="freq",
        fill_color="#6baed6",
    )
    hover_hist = HoverTool(
        renderers=[hist],
        tooltips=[("Bin", "@intervals"), ("Density", "@freq")],
        mode="vline",
    )
    line = fig.line(pts_rng, pdf, line_color="#9467bd", line_width=2, alpha=0.5)
    hover_dist = HoverTool(renderers=[line], tooltips=[("x", "@x"), ("y", "@y")])
    fig.add_tools(hover_hist)
    fig.add_tools(hover_dist)
    tweak_figure(fig, "kde")
    fig.yaxis.axis_label = "Density"
    minv = df.iloc[0]["left"]
    maxv = df.iloc[-1]["right"]
    _format_xaxis(fig, minv, maxv)
    return Panel(child=fig, title="KDE plot")


def qqnorm_viz(
    actual_qs: np.ndarray,
    theory_qs: np.ndarray,
    col: str,
    plot_width: int,
    plot_height: int,
) -> Panel:
    """
    Render a qq plot
    """
    tooltips = [("x", "@x"), ("y", "@y")]
    fig = Figure(
        plot_width=plot_width,
        plot_height=plot_height,
        title=f"{col}",
        tools="hover",
        toolbar_location=None,
        tooltips=tooltips,
    )
    fig.circle(
        x=theory_qs, y=actual_qs, size=3, color=PALETTE[0],
    )
    all_values = np.concatenate((theory_qs, actual_qs))
    fig.line(
        x=[np.min(all_values), np.max(all_values)],
        y=[np.min(all_values), np.max(all_values)],
        color="red",
    )
    tweak_figure(fig, "qq")
    fig.xaxis.axis_label = "Normal Quantiles"
    fig.yaxis.axis_label = f"Quantiles of {col}"
    return Panel(child=fig, title="QQ normal plot")


def box_viz(
    df: pd.DataFrame,
    outx: List[str],
    outy: List[float],
    x: str,
    plot_width: int,
    plot_height: int,
    y: Optional[str] = None,
    grp_cnt_stats: Optional[Dict[str, int]] = None,
) -> Panel:
    """
    Render a box plot visualization
    """
    # pylint: disable=too-many-arguments,too-many-locals
    if y is None:
        title = f"{x}"
    else:
        if grp_cnt_stats is None:
            title = f"{y} by {x}"
        elif grp_cnt_stats["x_ttl"] > grp_cnt_stats["x_show"]:
            title = "{} by (top {} out of {}) {}".format(
                y, grp_cnt_stats["x_show"], grp_cnt_stats["x_ttl"], x
            )
        else:
            title = f"{y} by {x}"
    if grp_cnt_stats is not None:
        if grp_cnt_stats["x_show"] > 10:
            plot_width = 28 * grp_cnt_stats["x_show"]
    fig = figure(
        tools="",
        x_range=list(df["grp"]),
        toolbar_location=None,
        title=title,
        plot_width=plot_width,
        plot_height=plot_height,
    )
    utail = fig.segment(
        x0="grp", y0="uw", x1="grp", y1="q3", line_color="black", source=df
    )
    ltail = fig.segment(
        x0="grp", y0="lw", x1="grp", y1="q1", line_color="black", source=df
    )
    ubox = fig.vbar(
        x="grp",
        width=0.7,
        top="q3",
        bottom="q2",
        fill_color=PALETTE[0],
        line_color="black",
        source=df,
    )
    lbox = fig.vbar(
        x="grp",
        width=0.7,
        top="q2",
        bottom="q1",
        fill_color=PALETTE[0],
        line_color="black",
        source=df,
    )
    loww = fig.segment(
        x0="x0", y0="lw", x1="x1", y1="lw", line_color="black", source=df
    )
    upw = fig.segment(x0="x0", y0="uw", x1="x1", y1="uw", line_color="black", source=df)
    if outx:
        circ = fig.circle(
            outx, outy, size=3, line_color="black", color=PALETTE[6], fill_alpha=0.6
        )
        fig.add_tools(HoverTool(renderers=[circ], tooltips=[("Outlier", "@y")],))
    fig.add_tools(
        HoverTool(
            renderers=[upw, utail, ubox, lbox, ltail, loww],
            tooltips=[
                ("Upper Whisker", "@uw"),
                ("Upper Quartile", "@q3"),
                ("Median", "@q2"),
                ("Lower Quartile", "@q1"),
                ("Lower Whisker", "@lw"),
            ],
            point_policy="follow_mouse",
        )
    )
    tweak_figure(fig, "box")
    if y is None:
        fig.xaxis.major_tick_line_color = None
        fig.xaxis.major_label_text_font_size = "0pt"
    fig.xaxis.axis_label = x if y is not None else None
    fig.yaxis.axis_label = x if y is None else y

    return Panel(child=fig, title="box plot")


def line_viz(
    data: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]],
    x: str,
    y: str,
    yscale: str,
    plot_width: int,
    plot_height: int,
    grp_cnt_stats: Dict[str, int],
    max_lbl_len: int = 15,
) -> Panel:
    """
    Render multi-line chart
    """
    # pylint: disable=too-many-arguments,too-many-locals
    grps = list(data.keys())
    palette = PALETTE * (len(grps) // len(PALETTE) + 1)
    title = _make_title(grp_cnt_stats, x, y)

    fig = figure(
        tools=[],
        title=title,
        toolbar_location=None,
        plot_width=plot_width,
        plot_height=plot_height,
        y_axis_type=yscale,
    )

    plot_dict = dict()
    for grp, colour in zip(grps, palette):
        ticks = [
            (data[grp][1][i] + data[grp][1][i + 1]) / 2
            for i in range(len(data[grp][1]) - 1)
        ]
        grp_name = (grp[: (max_lbl_len - 1)] + "...") if len(grp) > max_lbl_len else grp

        source = ColumnDataSource(
            {"x": ticks, "y": data[grp][0], "intervals": data[grp][2]}
        )
        plot_dict[grp_name] = fig.line(x="x", y="y", source=source, color=colour)
        fig.add_tools(
            HoverTool(
                renderers=[plot_dict[grp_name]],
                tooltips=[
                    (f"{x}", f"{grp}"),
                    ("frequency", "@y"),
                    (f"{y} bin", "@intervals"),
                ],
                mode="mouse",
            )
        )
    legend = Legend(items=[(x, [plot_dict[x]]) for x in plot_dict])
    tweak_figure(fig)
    fig.add_layout(legend, "right")
    fig.yaxis.axis_label = "Frequency"
    fig.xaxis.axis_label = y

    return Panel(child=fig, title="line chart")


def scatter_viz(
    df: pd.DataFrame, x: str, y: str, spl_sz: int, plot_width: int, plot_height: int,
) -> Any:
    """
    Render a scatter plot
    """
    # pylint: disable=too-many-arguments
    title = f"{y} by {x}" if len(df) < spl_sz else f"{y} by {x} (sample size {spl_sz})"
    tooltips = [("(x,y)", f"(@{x}, @{y})")]
    fig = figure(
        tools="hover",
        title=title,
        toolbar_location=None,
        tooltips=tooltips,
        plot_width=plot_width,
        plot_height=plot_height,
    )
    fig.circle(x, y, color=PALETTE[0], size=4, name="points", source=df)
    tweak_figure(fig)
    fig.xaxis.axis_label = x
    fig.yaxis.axis_label = y
    return Panel(child=fig, title="scatter plot")


def hexbin_viz(
    df: pd.DataFrame,
    x: str,
    y: str,
    plot_width: int,
    plot_height: int,
    tile_size: Optional[float] = None,
) -> Panel:
    """
    Render a hexbin plot
    """
    # pylint: disable=too-many-arguments,too-many-locals
    xmin, xmax = df[x].min(), df[x].max()
    ymin, ymax = df[y].min(), df[y].max()
    if tile_size is None:
        tile_size = (xmax - xmin) / 25
    title = f"{y} by {x}"
    aspect_scale = (ymax - ymin) / (xmax - xmin)
    bins = hexbin(
        x=df[x],
        y=df[y],
        size=tile_size,
        orientation="flattop",
        aspect_scale=aspect_scale,
    )
    fig = figure(
        title=title,
        tools=[],
        match_aspect=False,
        background_fill_color="#f5f5f5",
        toolbar_location=None,
        plot_width=plot_width,
        plot_height=plot_height,
    )

    palette = list(reversed(viridis(256)))
    rend = fig.hex_tile(
        q="q",
        r="r",
        size=tile_size,
        line_color=None,
        source=bins,
        orientation="flattop",
        fill_color=linear_cmap(
            field_name="counts",
            palette=palette,
            low=min(bins.counts),
            high=max(bins.counts),
        ),
        aspect_scale=aspect_scale,
    )
    fig.add_tools(HoverTool(tooltips=[("Count", "@counts")], renderers=[rend],))
    mapper = LinearColorMapper(
        palette=palette, low=min(bins.counts), high=max(bins.counts)
    )
    color_bar = ColorBar(color_mapper=mapper, width=8, location=(0, 0))
    color_bar.label_standoff = 8
    fig.add_layout(color_bar, "right")
    tweak_figure(fig, "hex")
    fig.xaxis.ticker = list(np.linspace(xmin, xmax, 10))
    fig.yaxis.ticker = list(np.linspace(ymin, ymax, 10))
    fig.xaxis.axis_label = x
    fig.yaxis.axis_label = y

    return Panel(child=fig, title="hexbin plot")


def nested_viz(
    df: pd.DataFrame,
    x: str,
    y: str,
    grp_cnt_stats: Dict[str, int],
    plot_width: int,
    plot_height: int,
) -> Panel:
    """
    Render a nested bar chart
    """
    # pylint: disable=too-many-arguments
    data_source = ColumnDataSource(data=df)
    title = _make_title(grp_cnt_stats, x, y)
    plot_width = 19 * len(df) if len(df) > 50 else plot_width
    fig = figure(
        x_range=FactorRange(*df["grp_names"]),
        tools="hover",
        tooltips=[("Group", "@grp_names"), ("Count", "@cnt")],
        toolbar_location=None,
        title=title,
        plot_width=plot_width,
        plot_height=plot_height,
    )

    fig.vbar(
        x="grp_names",
        top="cnt",
        width=1,
        source=data_source,
        line_color="white",
        line_width=3,
    )
    tweak_figure(fig, "nested")
    fig.yaxis.axis_label = "Count"
    fig.xaxis.major_label_orientation = pi / 2
    return Panel(child=fig, title="nested bar chart")


def stacked_viz(
    df: pd.DataFrame,
    x: str,
    y: str,
    grp_cnt_stats: Dict[str, int],
    plot_width: int,
    plot_height: int,
) -> Panel:
    """
    Render a stacked bar chart
    """
    # pylint: disable=too-many-arguments
    title = _make_title(grp_cnt_stats, x, y)
    if grp_cnt_stats["x_show"] > 30:
        plot_width = 32 * grp_cnt_stats["x_show"]
    fig = figure(
        x_range=df["grps"],
        toolbar_location=None,
        title=title,
        tools="hover",
        tooltips=[("Group", "@grps, $name"), ("Percentage", "@$name{0.2f}%"),],
        plot_width=plot_width,
        plot_height=plot_height,
    )
    subgrps = list(df.columns)[:-1]
    palette = Pastel1[9] * (len(subgrps) // len(Pastel1) + 1)
    if "Others" in subgrps:
        colours = palette[0 : len(subgrps) - 1] + ["#636363"]
    else:
        colours = palette[0 : len(subgrps)]

    renderers = fig.vbar_stack(
        stackers=subgrps, x="grps", width=0.9, source=df, line_width=1, color=colours,
    )

    legend_it = [(subgrp, [rend]) for subgrp, rend in zip(subgrps, renderers)]
    legend = Legend(items=legend_it)
    legend.label_text_font_size = "8pt"
    fig.add_layout(legend, "right")

    tweak_figure(fig, "stacked")
    fig.yaxis.axis_label = "Percent"
    return Panel(child=fig, title="stacked bar chart")


def heatmap_viz(
    df: pd.DataFrame,
    x: str,
    y: str,
    grp_cnt_stats: Dict[str, int],
    plot_width: int,
    plot_height: int,
    max_lbl_len: int = 15,
) -> Panel:
    """
    Render a heatmap
    """
    # pylint: disable=too-many-arguments
    title = _make_title(grp_cnt_stats, x, y)

    source = ColumnDataSource(data=df)
    palette = BIPALETTE[(len(BIPALETTE) // 2 - 1) :]
    mapper = LinearColorMapper(
        palette=palette, low=df["cnt"].min() - 0.01, high=df["cnt"].max()
    )
    if grp_cnt_stats["x_show"] > 60:
        plot_width = 16 * grp_cnt_stats["x_show"]
    if grp_cnt_stats["y_show"] > 10:
        plot_height = 70 + 18 * grp_cnt_stats["y_show"]
    fig = figure(
        x_range=list(set(df["x"])),
        y_range=list(set(df["y"])),
        toolbar_location=None,
        tools=[],
        x_axis_location="below",
        title=title,
        plot_width=plot_width,
        plot_height=plot_height,
    )

    renderer = fig.rect(
        x="x",
        y="y",
        width=1,
        height=1,
        source=source,
        line_color=None,
        fill_color=transform("cnt", mapper),
    )

    color_bar = ColorBar(
        color_mapper=mapper,
        location=(0, 0),
        ticker=BasicTicker(desired_num_ticks=7),
        formatter=PrintfTickFormatter(format="%d"),
    )
    fig.add_tools(
        HoverTool(
            tooltips=[(x, "@x"), (y, "@y"), ("Count", "@cnt"),],
            mode="mouse",
            renderers=[renderer],
        )
    )
    fig.add_layout(color_bar, "right")

    tweak_figure(fig, "heatmap")
    fig.yaxis.formatter = FuncTickFormatter(
        code="""
        if (tick.length > %d) return tick.substring(0, %d-2) + '...';
        else return tick;
    """
        % (max_lbl_len, max_lbl_len)
    )
    return Panel(child=fig, title="heat map")


def render_basic(
    itmdt: Intermediate, yscale: str, plot_width: int, plot_height: int,
) -> Box:
    """
    Render plots from plot(df)
    """
    figs = list()
    for col, dtype, data in itmdt["data"]:
        if dtype == DType.Categorical:
            df, total_grps, miss_pct = data
            fig = bar_viz(
                df[:-1],
                total_grps,
                miss_pct,
                col,
                yscale,
                plot_width,
                plot_height,
                False,
            )
            figs.append(fig)
        elif dtype == DType.Numerical:
            df, miss_pct = data
            fig = hist_viz(df, miss_pct, col, yscale, plot_width, plot_height, False)
            figs.append(fig)
    return gridplot(children=figs, sizing_mode=None, toolbar_location=None, ncols=3,)


def render_basic_x_cat(
    itmdt: Intermediate, yscale: str, plot_width: int, plot_height: int
) -> Tabs:
    """
    Render plots from plot(df, x) when x is a categorical column
    """
    tabs: List[Panel] = []
    df, total_grps, miss_pct = itmdt["data"]
    fig = bar_viz(
        df[:-1],
        total_grps,
        miss_pct,
        itmdt["col"],
        yscale,
        plot_width,
        plot_height,
        True,
    )
    tabs.append(Panel(child=fig, title="bar chart"))
    tabs.append(pie_viz(df, itmdt["col"], miss_pct, plot_width, plot_height))
    tabs = Tabs(tabs=tabs)
    return tabs


def render_basic_x_num(
    itmdt: Intermediate, yscale: str, plot_width: int, plot_height: int,
) -> Tabs:
    """
    Render plots from plot(df, x) when x is a numerical column
    """
    tabs: List[Panel] = []
    df, miss_pct = itmdt["histdata"]
    fig = hist_viz(df, miss_pct, itmdt["col"], yscale, plot_width, plot_height, True)
    tabs.append(Panel(child=fig, title="histogram"))
    df, pts_rng, pdf = itmdt["kdedata"]
    tabs.append(
        hist_kde_viz(df, pts_rng, pdf, itmdt["col"], yscale, plot_width, plot_height)
    )
    actual_qs, theory_qs = itmdt["qqdata"]
    tabs.append(qqnorm_viz(actual_qs, theory_qs, itmdt["col"], plot_width, plot_height))
    df, outx, outy, _ = itmdt["boxdata"]
    tabs.append(box_viz(df, outx, outy, itmdt["col"], plot_width, plot_height))
    tabs = Tabs(tabs=tabs)
    return tabs


def render_cat_and_num_cols(
    itmdt: Intermediate, yscale: str, plot_width: int, plot_height: int,
) -> Tabs:
    """
    Render plots from plot(df, x, y) when x is a categorical column
    and y is a numerical column
    """
    tabs: List[Panel] = []
    df, outx, outy, grp_cnt_stats = itmdt["boxdata"]
    tabs.append(
        box_viz(
            df,
            outx,
            outy,
            itmdt["x"],
            plot_width,
            plot_height,
            itmdt["y"],
            grp_cnt_stats,
        )
    )
    histdict, grp_cnt_stats = itmdt["histdata"]
    tabs.append(
        line_viz(
            histdict,
            itmdt["x"],
            itmdt["y"],
            yscale,
            plot_width,
            plot_height,
            grp_cnt_stats,
        )
    )
    tabs = Tabs(tabs=tabs)
    return tabs


def render_two_num_cols(
    itmdt: Intermediate,
    plot_width: int,
    plot_height: int,
    tile_size: Optional[float] = None,
) -> Tabs:
    """
    Render plots from plot(df, x, y) when x and y are numerical columns
    """
    tabs: List[Panel] = []
    tabs.append(
        scatter_viz(
            itmdt["scatdata"],
            itmdt["x"],
            itmdt["y"],
            itmdt["spl_sz"],
            plot_width,
            plot_height,
        )
    )
    df, outx, outy, _ = itmdt["boxdata"]
    tabs.append(
        box_viz(df, outx, outy, itmdt["x"], plot_width, plot_height, itmdt["y"])
    )
    tabs.append(
        hexbin_viz(
            itmdt["hexbindata"],
            itmdt["x"],
            itmdt["y"],
            plot_width,
            plot_height,
            tile_size,
        )
    )
    tabs = Tabs(tabs=tabs)
    return tabs


def render_two_cat_cols(
    itmdt: Intermediate, plot_width: int, plot_height: int,
) -> Tabs:
    """
    Render plots from plot(df, x, y) when x and y are categorical columns
    """
    tabs: List[Panel] = []
    df, grp_cnt_stats = itmdt["nesteddata"]
    tabs.append(
        nested_viz(df, itmdt["x"], itmdt["y"], grp_cnt_stats, plot_width, plot_height)
    )
    df, grp_cnt_stats = itmdt["stackdata"]
    tabs.append(
        stacked_viz(df, itmdt["x"], itmdt["y"], grp_cnt_stats, plot_width, plot_height)
    )
    df, grp_cnt_stats = itmdt["heatmapdata"]
    tabs.append(
        heatmap_viz(df, itmdt["x"], itmdt["y"], grp_cnt_stats, plot_width, plot_height)
    )
    tabs = Tabs(tabs=tabs)
    return tabs


def render(
    itmdt: Intermediate,
    yscale: str = "linear",
    tile_size: Optional[float] = None,
    plot_width_small: int = 324,
    plot_height_small: int = 300,
    plot_width_large: int = 450,
    plot_height_large: int = 400,
    plot_width_wide: int = 972,
) -> LayoutDOM:
    """
    Render a basic plot

    Parameters
    ----------
    itmdt : Intermediate
        The Intermediate containing results from the compute function.
    yscale: str = "linear"
        The scale to show on the y axis. Can be "linear" or "log".
    tile_size : Optional[float] = None
        Size of the tile for the hexbin plot. Measured from the middle
        of a hexagon to its left or right corner.
    plot_width_small : int = 324,
        The width of the small plots
    plot_height_small: int = 300,
        The height of the small plots
    plot_width_large : int = 450,
        The width of the large plots
    plot_height_large: int = 400,
        The height of the large plots
    plot_width_large : int = 972,
        The width of the wide plots

    Returns
    -------
    LayoutDOM
        A bokeh layout domain.
    """
    # pylint: disable=too-many-arguments
    if itmdt.visual_type == "basic_grid":
        visual_elem = render_basic(itmdt, yscale, plot_width_small, plot_height_small)
    elif itmdt.visual_type == "categorical_column":
        visual_elem = render_basic_x_cat(
            itmdt, yscale, plot_width_large, plot_height_large
        )
    elif itmdt.visual_type == "numerical_column":
        visual_elem = render_basic_x_num(
            itmdt, yscale, plot_width_large, plot_height_large
        )
    elif itmdt.visual_type == "cat_and_num_cols":
        visual_elem = render_cat_and_num_cols(
            itmdt, yscale, plot_width_large, plot_height_large
        )
    elif itmdt.visual_type == "two_num_cols":
        visual_elem = render_two_num_cols(
            itmdt, plot_width_large, plot_height_large, tile_size
        )
    elif itmdt.visual_type == "two_cat_cols":
        visual_elem = render_two_cat_cols(itmdt, plot_width_wide, plot_height_small)
    return visual_elem
