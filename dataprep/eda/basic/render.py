"""
This module is for the correct rendering of bokeh plot figures.
"""
from math import pi

# pylint: disable=R0903
# pylint: disable=R0912
from typing import Any, Dict, List, Optional

import numpy as np
from bokeh.models import (
    Box,
    FuncTickFormatter,
    HoverTool,
    LayoutDOM,
    Legend,
    LegendItem,
    Panel,
    Tabs,
)
from bokeh.plotting import Figure, gridplot, show
from bokeh.transform import cumsum

from ..intermediate import Intermediate
from ..dtypes import DType
from ..palette import PALETTE


def tweak_figure(
    p: Figure, ptype: Optional[str] = None, max_label_len: int = 15
) -> None:
    p.grid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.axis.major_label_text_font_size = "9pt"
    p.xaxis.major_label_orientation = pi / 3
    p.title.text_font_size = "10pt"
    if ptype == "bar":
        p.yaxis.major_tick_line_color = None
        p.yaxis.major_label_text_font_size = "0pt"
        p.xaxis.formatter = FuncTickFormatter(
            code="""
            if (tick.length > %d) return tick.substring(0, %d-2) + '...';
            else return tick;
        """
            % (max_label_len, max_label_len)
        )
    elif ptype == "pie":
        p.axis.major_label_text_font_size = "0pt"
        p.axis.major_tick_line_color = None


def basic_x_cat_setup(data_dict: Dict[str, Any], col: str) -> Any:
    miss_perc = data_dict["miss_perc"]
    tooltips = [
        ("" + col + "", f"@{col}"),
        ("Count", "@count"),
        ("Percentage", "@percent{0.2f}%"),
    ]
    title = f"{col} ({miss_perc}% missing)" if miss_perc > 0 else f"{col}"
    return title, tooltips


def bar_viz(
    data_dict: Dict[str, Any], col: str, plot_width: int, plot_height: int,
) -> Figure:

    title, tooltips = basic_x_cat_setup(data_dict, col)
    df = data_dict["data"][:-1]
    total_groups = data_dict["total_groups"]
    p = Figure(
        x_range=list(df[col]),
        title=title,
        plot_width=plot_width,
        plot_height=plot_height,
        tools="hover",
        toolbar_location=None,
        tooltips=tooltips,
    )
    p.vbar(x=col, top="count", width=0.9, source=df)
    tweak_figure(p, "bar")
    p.yaxis.axis_label = "Count"
    if total_groups > len(df):
        p.xaxis.axis_label = f"Top {len(df)} of {total_groups} {col}"
        p.xaxis.axis_label_standoff = 0
    return p


def pie_viz(
    data_dict: Dict[str, Any], col: str, plot_width: int, plot_height: int,
) -> Panel:

    title, tooltips = basic_x_cat_setup(data_dict, col)
    df = data_dict["data"]
    df["angle"] = df["count"] / df["count"].sum() * 2 * pi
    p = Figure(
        title=title,
        plot_width=plot_width,
        plot_height=plot_height,
        tools="hover",
        toolbar_location=None,
        tooltips=tooltips,
    )
    color_list = PALETTE * (len(df) // len(PALETTE) + 1)
    df["colour"] = color_list[0 : len(df)]
    if df.iloc[-1]["count"] == 0:
        df = df[:-1]
    pie = p.wedge(
        x=0,
        y=1,
        radius=0.9,
        start_angle=cumsum("angle", include_zero=True),
        end_angle=cumsum("angle"),
        line_color="white",
        fill_color="colour",
        source=df,
    )
    legend = Legend(items=[LegendItem(label=dict(field=col), renderers=[pie])])
    legend.label_text_font_size = "8pt"
    p.add_layout(legend, "right")
    tweak_figure(p, "pie")
    return Panel(child=p, title="bar chart")


def hist_viz(
    data_dict: Dict[str, Any],
    col: str,
    yscale: str,
    plot_width: int,
    plot_height: int,
    show_y_label: bool,
) -> Figure:
    df = data_dict["hist_df"]
    miss_perc = data_dict["miss_perc"]
    if miss_perc > 0:
        title = "{} ({}% missing)".format(col, miss_perc)
    else:
        title = "{}".format(col)
    tooltips = [
        ("Bin", "[@left, @right]"),
        ("Frequency", "@freq"),
        ("Percentage", "@percent{0.2f}%"),
    ]
    p = Figure(
        plot_width=plot_width,
        plot_height=plot_height,
        toolbar_location=None,
        title=title,
        tools=[],
        y_axis_type=yscale,
    )
    p.quad(
        source=df,
        left="left",
        right="right",
        bottom=0.01,
        alpha=0.5,
        top="freq",
        fill_color="#6baed6",
    )
    hover = HoverTool(tooltips=tooltips, mode="vline",)
    p.add_tools(hover)
    tweak_figure(p)
    p.yaxis.axis_label = "Frequency"
    x_ticks = list(df["left"])
    x_ticks.append(df.iloc[-1]["right"])
    p.xaxis.ticker = x_ticks
    if not show_y_label:
        p.yaxis.major_label_text_font_size = "0pt"
        p.yaxis.major_tick_line_color = None
    return p


def hist_kde_viz(  # pylint: disable=too-many-arguments
    data_dict: Dict[str, Any], col: str, yscale: str, plot_width: int, plot_height: int,
) -> Panel:

    df = data_dict["hist_df"]
    calc_pts = data_dict["calc_pts"]
    pdf = data_dict["pdf"]

    p = Figure(
        plot_width=plot_width,
        plot_height=plot_height,
        title=f"{col}",
        tools=[],
        toolbar_location=None,
    )
    hist = p.quad(
        source=df,
        left="left",
        right="right",
        bottom=1e-7,
        alpha=0.5,
        top="freq",
        fill_color="#6baed6",
    )
    hover_hist = HoverTool(
        renderers=[hist], tooltips=[("Bin", "[@left, @right]"), ("Density", "@freq")],
    )
    line = p.line(calc_pts, pdf, line_color="#9467bd", line_width=2, alpha=0.5)
    hover_dist = HoverTool(
        renderers=[line], tooltips=[("x", "@x"), ("y", "@y")], mode="mouse"
    )
    p.add_tools(hover_hist)
    p.add_tools(hover_dist)
    tweak_figure(p)
    p.yaxis.axis_label = "Density"
    x_ticks = list(df["left"])
    x_ticks.append(df.iloc[-1]["right"])
    p.xaxis.ticker = x_ticks
    return Panel(child=p, title="KDE plot")


def qqnorm_viz(
    actual_qs: np.ndarray,
    theory_qs: np.ndarray,
    col: str,
    plot_width: int,
    plot_height: int,
) -> Panel:

    tooltips = [("x", "@x"), ("y", "@y")]
    p = Figure(
        plot_width=plot_width,
        plot_height=plot_height,
        title=f"{col}",
        tools="hover",
        toolbar_location=None,
        tooltips=tooltips,
    )
    p.circle(
        x=theory_qs, y=actual_qs, size=3, color=PALETTE[0],
    )
    all_values = np.concatenate((theory_qs, actual_qs))
    p.line(
        x=[np.min(all_values), np.max(all_values)],
        y=[np.min(all_values), np.max(all_values)],
        color="red",
    )
    tweak_figure(p)
    p.xaxis.axis_label = "Normal Quantiles"
    p.yaxis.axis_label = f"Quantiles of {col}"
    return Panel(child=p, title="QQ normal plot")


def render_basic(
    itmdt: Intermediate, yscale: str, plot_width: int, plot_height: int,
) -> Box:
    figs = list()
    for col, dtype, data in itmdt["datas"]:
        if dtype == DType.Categorical:
            fig = bar_viz(data, col, plot_width, plot_height)
            figs.append(fig)
        elif dtype == DType.Numerical:
            fig = hist_viz(data, col, yscale, plot_width, plot_height, False)
            figs.append(fig)
    return gridplot(children=figs, sizing_mode=None, toolbar_location=None, ncols=3,)


def render_basic_x_cat(itmdt: Intermediate, plot_width: int, plot_height: int) -> Tabs:
    tabs: List[Panel] = []
    fig = bar_viz(itmdt["data"], itmdt["col"], plot_width, plot_height)
    tabs.append(Panel(child=fig, title="bar chart"))
    tabs.append(pie_viz(itmdt["data"], itmdt["col"], plot_width, plot_height))
    tabs = Tabs(tabs=tabs)
    return Tabs


def render_basic_x_num(
    itmdt: Intermediate, yscale: str, plot_width: int, plot_height: int,
) -> Tabs:
    tabs: List[Panel] = []
    fig = hist_viz(
        itmdt["hist_dict"], itmdt["col"], yscale, plot_width, plot_height, True
    )
    tabs.append(Panel(child=fig, title="histogram"))
    tabs.append(
        hist_kde_viz(itmdt["kde_dict"], itmdt["col"], yscale, plot_width, plot_height)
    )
    actual_qs, theory_qs = itmdt["qqdata"]
    tabs.append(qqnorm_viz(actual_qs, theory_qs, itmdt["col"], plot_width, plot_height))
    tabs = Tabs(tabs=tabs)
    return tabs


def render(
    itmdt: Intermediate,
    yscale: str = "linear",
    tile_size: Optional[float] = None,
    plot_height_small: int = 300,
    plot_width_small: int = 324,
    plot_height_large: int = 400,
    plot_width_large: int = 450,
    plot_width_wide: int = 972,
) -> LayoutDOM:
    if itmdt.visual_type == "basic_grid":
        return render_basic(itmdt, yscale, plot_width_small, plot_height_small)
    elif itmdt.visual_type == "categorical_column":
        return render_basic_x_cat(itmdt, plot_width_large, plot_height_large)
    elif itmdt.visual_type == "numerical_column":
        return render_basic_x_num(itmdt, yscale, plot_width_large, plot_height_large)
