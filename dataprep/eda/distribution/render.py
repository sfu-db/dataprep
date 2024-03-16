"""
This module implements the visualization for the plot(df) function.
"""  # pylint: disable=too-many-lines

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import os
import math

import numpy as np
import pandas as pd
from bokeh.layouts import row
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    CustomJSHover,
    FactorRange,
    FuncTickFormatter,
    Range1d,
    HoverTool,
    LayoutDOM,
    Legend,
    LegendItem,
    LinearColorMapper,
    Panel,
    PrintfTickFormatter,
)
from bokeh.plotting import Figure, figure
from bokeh.transform import cumsum, linear_cmap, transform
from bokeh.util.hex import hexbin
from scipy.stats import norm
from wordcloud import WordCloud

from ..configs import KDE, Bar, Box, Config, Pie, QQNorm, WordFrequency
from ..dtypes_v2 import Continuous, DateTime, Nominal, GeoGraphy, SmallCardNum, GeoPoint
from ..intermediate import Intermediate
from ..palette import CATEGORY20, PASTEL1, RDBU, VIRIDIS, YlGnBu
from ..utils import tweak_figure, _format_ticks, _format_axis, _format_bin_intervals

COUNTRY_MAP_FILE = os.path.join(os.path.split(os.path.abspath(__file__))[0], "country.json")
COUNTRY_NAME_FILE = os.path.join(os.path.split(os.path.abspath(__file__))[0], "name_dict.json")
with open(COUNTRY_MAP_FILE, "r") as fp:
    MAPS = json.load(fp)
with open(COUNTRY_NAME_FILE, "r") as fp:
    NAME_DICT = json.load(fp)

__all__ = ["render"]


def _make_title(grp_cnt_stats: Dict[str, int], x: str, y: str) -> str:
    """
    Format the title to notify the user of sampled output
    """
    x_ttl, y_ttl = None, None
    if f"{x}_ttl" in grp_cnt_stats:
        x_ttl = grp_cnt_stats[f"{x}_ttl"]
        x_shw = grp_cnt_stats[f"{x}_shw"]
    if f"{y}_ttl" in grp_cnt_stats:
        y_ttl = grp_cnt_stats[f"{y}_ttl"]
        y_shw = grp_cnt_stats[f"{y}_shw"]
    if x_ttl and y_ttl:
        if x_ttl > x_shw and y_ttl > y_shw:
            return f"(top {y_shw} out of {y_ttl}) {y} by (top {x_shw} out of {x_ttl}) {x}"
    elif x_ttl:
        if x_ttl > x_shw:
            return f"{y} by (top {x_shw} out of {x_ttl}) {x}"
    elif y_ttl:
        if y_ttl > y_shw:
            return f"(top {y_shw} out of {y_ttl}) {y} by {x}"
    return f"{y} by {x}"


def _format_values(key: str, value: Any) -> str:

    if not isinstance(value, (int, float)):
        # if value is a time
        return str(value)

    if "Memory" in key:
        # for memory usage
        ind = 0
        unit = dict(enumerate(["B", "KB", "MB", "GB", "TB"], 0))
        while value > 1024:
            value /= 1024
            ind += 1
        return f"{value:.1f} {unit[ind]}"

    if (value * 10) % 10 == 0:
        # if value is int but in a float form with 0 at last digit
        value = int(value)
        if abs(value) >= 1000000:
            return f"{value:.5g}"
    elif abs(value) >= 1000000 or abs(value) < 0.001:
        value = f"{value:.5g}"
    elif abs(value) >= 1:
        # eliminate trailing zeros
        pre_value = float(f"{value:.4f}")
        value = int(pre_value) if (pre_value * 10) % 10 == 0 else pre_value
    elif 0.001 <= abs(value) < 1:
        value = f"{value:.4g}"
    else:
        value = str(value)

    if "%" in key:
        # for percentage, only use digits before notation sign for extreme small number
        value = f"{float(value):.1%}"
    return str(value)


def _empty_figure(title: str, plot_height: int, plot_width: int) -> Figure:
    # If no data to render in the heatmap, i.e. no missing values
    # we render a blank heatmap
    fig = Figure(
        x_range=[],
        y_range=[],
        plot_height=plot_height,
        plot_width=plot_width,
        title=title,
        x_axis_location="below",
        tools="hover",
        toolbar_location=None,
        background_fill_color="#fafafa",
    )

    # Add at least one renderer to fig, otherwise bokeh
    # gives us error -1000 (MISSING_RENDERERS): Plot has no renderers
    fig.rect(x=0, y=0, width=0, height=0)
    return fig


def wordcloud_viz(
    word_cnts: pd.Series,
    plot_width: int,
    plot_height: int,
) -> Panel:
    """
    Visualize the word cloud
    """  # pylint: disable=unsubscriptable-object
    ellipse_mask = np.load(f"{Path(__file__).parent.parent.parent}/assets/ellipse.npz").get("image")
    wordcloud = WordCloud(background_color="white", mask=ellipse_mask)
    wordcloud.generate_from_frequencies(word_cnts)
    wcarr = wordcloud.to_array().astype(np.uint8)

    # use image_rgba following this example
    # https://docs.bokeh.org/en/latest/docs/gallery/image_rgba.html
    img = np.empty(wcarr.shape[:2], dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((*wcarr.shape[:2], 4))
    alpha = np.full((*wcarr.shape[:2], 1), 255, dtype=np.uint8)
    view[:] = np.concatenate([wcarr, alpha], axis=2)[::-1]

    fig = figure(
        plot_width=plot_width,
        plot_height=plot_height,
        title="Word Cloud",
        toolbar_location=None,
        x_range=(0, 1),
        y_range=(0, 1),
    )
    fig.image_rgba(image=[img], x=0, y=0, dw=1, dh=1)

    fig.axis.visible = False
    fig.grid.visible = False
    return Panel(child=row(fig), title="Word Cloud")


def wordfreq_viz(
    word_cnts: pd.Series,
    nrows: int,
    plot_width: int,
    plot_height: int,
    wordfreq: WordFrequency,
) -> Figure:
    """
    Visualize the word frequency bar chart
    """
    col = word_cnts.name
    df = word_cnts.to_frame()
    df["pct"] = df[col] / nrows * 100

    tooltips = [
        ("Word", "@index"),
        ("Count", f"@{{{col}}}"),
        ("Percent", "@pct{0.2f}%"),
    ]
    fig = figure(
        plot_height=plot_height,
        plot_width=plot_width,
        title="Word Frequency",
        toolbar_location=None,
        tools="hover",
        tooltips=tooltips,
        x_range=list(df.index),
    )
    fig.vbar(
        x="index",
        top=col,
        fill_color=wordfreq.color,
        line_color=wordfreq.color,
        width=0.9,
        source=df,
    )
    fig.yaxis.axis_label = "Count"
    tweak_figure(fig, "bar", True)
    _format_axis(fig, 0, df[col].max(), "y")
    return Panel(child=row(fig), title="Word Frequency")


def bar_viz(
    df: pd.DataFrame,
    ttl_grps: int,
    nrows: int,
    col: str,
    plot_width: int,
    plot_height: int,
    show_yticks: bool,
    bar_cfg: Bar,
) -> Figure:
    """
    Render a bar chart
    """
    # pylint: disable=too-many-arguments
    df["pct"] = df[col] / nrows * 100
    df.index = [str(val) for val in df.index]

    tooltips = [(col, "@index"), ("Count", f"@{{{col}}}"), ("Percent", "@pct{0.2f}%")]
    if show_yticks:
        if len(df) > 10:
            plot_width = 28 * len(df)
    fig = Figure(
        plot_width=plot_width,
        plot_height=plot_height,
        title=col,
        toolbar_location=None,
        tooltips=tooltips,
        tools="hover",
        x_range=list(df.index),
        y_axis_type=bar_cfg.yscale,
    )
    fig.vbar(
        x="index",
        width=0.9,
        top=col,
        fill_color=bar_cfg.color,
        line_color=bar_cfg.color,
        bottom=0.01,
        source=df,
    )
    tweak_figure(fig, "bar", show_yticks)
    fig.yaxis.axis_label = "Count"
    if ttl_grps > len(df):
        fig.xaxis.axis_label = f"Top {len(df)} of {ttl_grps} {col}"
        fig.xaxis.axis_label_standoff = 0
    if show_yticks and bar_cfg.yscale == "linear":
        _format_axis(fig, 0, df[col].max(), "y")
    return fig


def pie_viz(
    df: pd.DataFrame,
    nrows: int,
    col: str,
    plot_width: int,
    plot_height: int,
    pie: Pie,
) -> Tuple[Panel, List[str]]:
    """
    Render a pie chart
    """
    # pylint: disable=too-many-arguments
    npresent = df[col].sum()
    df.index = list(df.index)  # for CategoricalIndex to normal Index
    if nrows > npresent:
        df = df.append(pd.DataFrame({col: [nrows - npresent]}, index=["Others"]))
    df["pct"] = df[col] / nrows * 100
    df["angle"] = df[col] / nrows * 2 * np.pi

    tooltips = [(col, "@index"), ("Count", f"@{{{col}}}"), ("Percent", "@pct{0.2f}%")]
    fig = Figure(
        plot_width=plot_width,
        plot_height=plot_height,
        title=col,
        toolbar_location=None,
        tools="hover",
        tooltips=tooltips,
    )
    if pie.colors is None:
        color_list = list((CATEGORY20 * (len(df) // len(CATEGORY20) + 1))[0 : len(df)])
    else:
        color_list = list(pie.colors[0 : len(df)])
    df["colour"] = color_list
    df.index = df.index.astype(str)
    df.index = df.index.map(lambda x: x[:13] + "..." if len(x) > 13 else x)

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
    legend = Legend(items=[LegendItem(label=dict(field="index"), renderers=[pie])])
    legend.label_text_font_size = "8pt"
    fig.add_layout(legend, "left")
    tweak_figure(fig, "pie")
    fig.axis.major_label_text_font_size = "0pt"
    fig.axis.major_tick_line_color = None
    return Panel(child=row(fig), title="Pie Chart"), color_list


def hist_viz(
    hist: Tuple[np.ndarray, np.ndarray],
    nrows: int,
    col: str,
    yscale: str,
    color: str,
    plot_width: int,
    plot_height: int,
    show_yticks: bool,
) -> Figure:
    """
    Render a histogram
    """
    # pylint: disable=too-many-arguments,too-many-locals
    counts, bins = hist
    if sum(counts) == 0:
        return _empty_figure(col, plot_height, plot_width)
    intvls = _format_bin_intervals(bins)
    df = pd.DataFrame(
        {
            "intvl": intvls,
            "left": bins[:-1],
            "right": bins[1:],
            "freq": counts,
            "pct": counts / nrows * 100,
        }
    )

    tooltips = [("Bin", "@intvl"), ("Frequency", "@freq"), ("Percent", "@pct{0.2f}%")]
    if yscale == "linear" or df.empty:
        bottom = 0.0
    elif yscale == "log" and df["freq"].min() == 0:  # freq >= 1 so we set 0.1 as lower bound
        bottom = 0.1
    else:
        bottom = df["freq"].min() / 2

    if yscale == "linear":
        fig = Figure(
            plot_height=plot_height,
            plot_width=plot_width,
            title=col,
            toolbar_location=None,
            y_axis_type=yscale,
        )
    else:
        fig = Figure(
            plot_height=plot_height,
            plot_width=plot_width,
            title=col,
            toolbar_location=None,
            y_axis_type=yscale,
            y_range=(bottom, df["freq"].max()),
        )

    fig.quad(
        source=df,
        left="left",
        right="right",
        bottom=bottom,
        top="freq",
        fill_color=color,
        line_color=color,
    )
    hover = HoverTool(tooltips=tooltips, mode="vline")
    fig.add_tools(hover)
    tweak_figure(fig, "hist", show_yticks)
    fig.yaxis.axis_label = "Frequency"
    _format_axis(fig, df.iloc[0]["left"], df.iloc[-1]["right"], "x")
    if show_yticks:
        fig.xaxis.axis_label = col
        if yscale == "linear":
            _format_axis(fig, 0, df["freq"].max(), "y")

    return fig


def kde_viz(
    hist: Tuple[np.ndarray, np.ndarray],
    kde: np.ndarray,
    col: str,
    plot_width: int,
    plot_height: int,
    kde_cfg: KDE,
) -> Panel:
    """
    Render histogram with overlayed kde
    """
    # pylint: disable=too-many-arguments, too-many-locals
    dens, bins = hist
    intvls = _format_bin_intervals(bins)
    df = pd.DataFrame(
        {
            "intvl": intvls,
            "left": bins[:-1],
            "right": bins[1:],
            "dens": dens,
        }
    )
    fig = Figure(
        plot_width=plot_width,
        plot_height=plot_height,
        title=col,
        toolbar_location=None,
        y_axis_type=kde_cfg.yscale,
    )
    bottom = 0 if kde_cfg.yscale == "linear" or df.empty else df["dens"].min() / 2
    hist = fig.quad(
        source=df,
        left="left",
        right="right",
        bottom=bottom,
        top="dens",
        fill_color=kde_cfg.hist_color,
        line_color=kde_cfg.hist_color,
    )
    hover_hist = HoverTool(
        renderers=[hist],
        tooltips=[("Bin", "@intvl"), ("Density", "@dens")],
        mode="vline",
    )
    pts_rng = np.linspace(df.loc[0, "left"], df.loc[len(df) - 1, "right"], 1000)
    pdf = kde(pts_rng)
    line = fig.line(x=pts_rng, y=pdf, line_color=kde_cfg.line_color, line_width=2, alpha=0.5)
    hover_dist = HoverTool(renderers=[line], tooltips=[("x", "@x"), ("y", "@y")])
    fig.add_tools(hover_hist)
    fig.add_tools(hover_dist)
    tweak_figure(fig, "kde")
    fig.yaxis.axis_label = "Density"
    fig.xaxis.axis_label = col
    _format_axis(fig, df.iloc[0]["left"], df.iloc[-1]["right"], "x")
    if kde_cfg.yscale == "linear":
        _format_axis(fig, 0, max(df["dens"].max(), pdf.max()), "y")
    return Panel(child=row(fig), title="KDE Plot")


def qqnorm_viz(
    qntls: pd.Series,
    mean: float,
    std: float,
    col: str,
    plot_width: int,
    plot_height: int,
    qqnorm: QQNorm,
) -> Panel:
    """
    Render a qq plot
    """
    # pylint: disable=too-many-arguments
    theory_qntls = norm.ppf(np.linspace(0.01, 0.99, 99), mean, std)
    tooltips = [("x", "@x"), ("y", "@y")]
    fig = Figure(
        plot_width=plot_width,
        plot_height=plot_height,
        title=col,
        tools="hover",
        toolbar_location=None,
        tooltips=tooltips,
    )
    fig.circle(
        x=theory_qntls,
        y=qntls,
        size=3,
        color=qqnorm.point_color,
    )
    vals = np.concatenate((theory_qntls, qntls))
    fig.line(x=[vals.min(), vals.max()], y=[vals.min(), vals.max()], color=qqnorm.line_color)
    tweak_figure(fig, "qq")
    fig.xaxis.axis_label = "Normal Quantiles"
    fig.yaxis.axis_label = f"Quantiles of {col}"
    _format_axis(fig, vals.min(), vals.max(), "x")
    _format_axis(fig, vals.min(), vals.max(), "y")
    return Panel(child=row(fig), title="Normal Q-Q Plot")


def box_viz(
    df: pd.DataFrame,
    x: str,
    plot_width: int,
    plot_height: int,
    box: Box,
    y: Optional[str] = None,
    ttl_grps: Optional[int] = None,
) -> Panel:
    """
    Render a box plot visualization
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-statements

    if y and ttl_grps:
        width = 0.7
        grp_cnt_stats = {f"{x}_ttl": ttl_grps, f"{x}_shw": len(df)}
        title = _make_title(grp_cnt_stats, x, y) if ttl_grps else f"{y} by {x}"
    elif y:
        width, title = 0.93, f"{y} by {x}"
        endpts = [grp.left for grp in df["grp"]] + [df["grp"][len(df) - 1].right]
        df["grp"] = df["grp"].astype(str)
    else:
        width, title = 0.7, f"{x}"
    df["x0"], df["x1"] = df.index + 0.2, df.index + 0.8

    fig = figure(
        plot_width=plot_width,
        plot_height=plot_height,
        title=title,
        toolbar_location=None,
        x_range=df["grp"],
    )
    low = fig.segment(x0="x0", y0="lw", x1="x1", y1="lw", line_color="black", source=df)
    ltail = fig.segment(x0="grp", y0="lw", x1="grp", y1="q1", line_color="black", source=df)
    lbox = fig.vbar(
        x="grp",
        width=width,
        top="q2",
        bottom="q1",
        fill_color=box.color,
        line_color="black",
        source=df,
    )
    ubox = fig.vbar(
        x="grp",
        width=width,
        top="q3",
        bottom="q2",
        fill_color=box.color,
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
    if y:
        lbl = f"{x}" if ttl_grps else "Bin"
        tooltips.insert(0, (lbl, "@grp"))
    fig.add_tools(
        HoverTool(
            renderers=[upw, utail, ubox, lbox, ltail, low],
            tooltips=tooltips,
        )
    )
    tweak_figure(fig, "box")
    if y is None:
        fig.xaxis.major_tick_line_color = None
        fig.xaxis.major_label_text_font_size = "0pt"
    fig.xaxis.axis_label = x if y is not None else None
    fig.yaxis.axis_label = x if y is None else y

    minw = min(otlrs) if otlrs else np.nan
    maxw = max(otlrs) if otlrs else np.nan
    _format_axis(fig, min(df["lw"].min(), minw), max(df["uw"].max(), maxw), "y")

    if y and not ttl_grps:  # format categorical axis tick values
        # start by rounding to the length of the largest possible number
        round_to = -len(str(max([abs(int(ept)) for ept in endpts])))
        ticks = np.round(endpts, round_to)
        nticks = len(df) // 5 + 1
        show_ticks = [ticks[i] for i in range(len(ticks)) if i % nticks == 0]
        while len(set(show_ticks)) != len(show_ticks):  # round until show ticks unique
            round_to += 1
            ticks = np.round(endpts, round_to)
            show_ticks = [ticks[i] for i in range(len(ticks)) if i % nticks == 0]
        # format the ticks
        ticks = [int(tick) if tick.is_integer() else tick for tick in ticks]
        ticks = _format_ticks(ticks)
        fig.xaxis.ticker = list(range(len(df) + 1))
        fig.xaxis.formatter = FuncTickFormatter(  # overide bokeh ticks
            args={"vals": ticks, "mod": nticks},
            code="""
                if (index % mod == 0) return vals[index];
                return "";
            """,
        )
        tweak_figure(fig, "boxnum")
        fig.xaxis.major_label_text_font_size = "10pt"

    return Panel(child=row(fig), title="Box Plot")


def latlong_viz(
    df: pd.DataFrame,
    plot_width: int,
    y: Optional[str] = None,
) -> Panel:
    """
    Render a latlong plot visualization
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
    # pylint: disable=too-many-function-args
    # title = f"{y} by {x}"

    # no_name=[]
    tooltip_1 = [("Name", "@name"), ("(Long, Lat)", "($x, $y)")]
    tooltip_2 = [(y, "@sizes_ori")]

    fig = Figure(
        plot_width=plot_width,
        plot_height=plot_width // 10 * 7,
        # tools=tools,
    )
    fig.grid.grid_line_color = None
    fig.hover.point_policy = "follow_mouse"
    fig.background_fill_color = "white"
    fig.x_range = Range1d(start=-180, end=180)
    fig.y_range = Range1d(start=-90, end=90)
    world_map = fig.patches("xs", "ys", line_color="white", source=MAPS, line_width=0.5)
    fig.add_tools(HoverTool(renderers=[world_map], tooltips=tooltip_1))
    lat = [item[0] for item in df.index]
    lon = [item[1] for item in df.index]
    minimum = min(df[y])
    maximum = max(df[y])
    num_col = (df[y] - minimum) / (maximum - minimum) * 20
    num_col = num_col.apply(lambda a: a + 5 if a > 0 else a)  # Normalization to {0, [5,25]}
    source = ColumnDataSource(data=dict(lat=lat, lon=lon, sizes=num_col, sizes_ori=df[y]))
    my_dots = fig.circle(
        x="lon", y="lat", size="sizes", fill_color="red", fill_alpha=0.8, source=source
    )
    fig.add_tools(HoverTool(renderers=[my_dots], tooltips=tooltip_2))

    return Panel(child=row(fig), title="Geo Map")


def geo_viz(
    df: pd.DataFrame,
    plot_width: int,
    y: Optional[str] = None,
) -> Panel:
    """
    Render a geo plot visualization
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
    # pylint: disable=too-many-function-args

    # title = f"{y} by {x}"

    minimum = min(df[y])
    maximum = max(df[y])

    # no_name=[]
    value = {}
    names = NAME_DICT.keys()
    for i in range(df[y].shape[0]):
        if df.index[i].lower().strip() in names:
            value[NAME_DICT[df.index[i].lower().strip()]] = df[y][i]
        # else:
        #     no_name.append(df.index[i])

    temp_list = []
    for itr in range(len(MAPS["name"])):
        temp_list.append(value.get(MAPS["fip"][itr], "unknown"))
    MAPS["value"] = temp_list

    mapper = LinearColorMapper(
        palette=YlGnBu[33:233], low=minimum, high=maximum, nan_color="#cccccc"
    )
    tools = "pan,wheel_zoom,box_zoom,reset,hover"

    fig = Figure(
        plot_width=plot_width,
        plot_height=plot_width // 10 * 7,
        tools=tools,
        tooltips=[
            ("Name", "@name"),
            (y, "@value"),
            ("(Long, Lat)", "($x, $y)"),
        ],
    )
    fig.grid.grid_line_color = None
    fig.hover.point_policy = "follow_mouse"
    fig.background_fill_color = "white"
    fig.x_range = Range1d(start=-180, end=180)
    fig.y_range = Range1d(start=-90, end=90)
    fig.patches(
        "xs",
        "ys",
        line_color="white",
        source=MAPS,
        fill_color={"field": "value", "transform": mapper},
        line_width=0.5,
    )

    color_bar = ColorBar(
        color_mapper=mapper,
        major_label_text_font_size="7px",
        ticker=BasicTicker(desired_num_ticks=11),
        formatter=PrintfTickFormatter(format="%10.2f"),
        label_standoff=6,
        border_line_color=None,
        location=(0, 0),
    )
    if minimum < maximum:
        fig.add_layout(color_bar, "right")

    return Panel(child=row(fig), title="World Map")


# this function should be removed when datetime is refactored
def box_viz_dt(
    df: pd.DataFrame,
    outx: List[str],
    outy: List[float],
    x: str,
    plot_width: int,
    plot_height: int,
    y: Optional[str] = None,
    grp_cnt_stats: Optional[Dict[str, int]] = None,
    timeunit: Optional[str] = None,
) -> Panel:
    """
    Render a box plot visualization
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
    if y:
        width = 0.7 if grp_cnt_stats else 0.93
        title = _make_title(grp_cnt_stats, x, y) if grp_cnt_stats else f"{y} by {x}"
    else:
        width = 0.7
        title = f"{x}"

    if len(df) > 10:
        plot_width = 39 * len(df)
    fig = figure(
        tools="",
        x_range=list(df["grp"]),
        toolbar_location=None,
        title=title,
        plot_width=plot_width,
        plot_height=plot_height,
    )
    utail = fig.segment(x0="grp", y0="uw", x1="grp", y1="q3", line_color="black", source=df)
    ltail = fig.segment(x0="grp", y0="lw", x1="grp", y1="q1", line_color="black", source=df)
    ubox = fig.vbar(
        x="grp",
        width=width,
        top="q3",
        bottom="q2",
        fill_color=CATEGORY20[0],
        line_color="black",
        source=df,
    )
    lbox = fig.vbar(
        x="grp",
        width=width,
        top="q2",
        bottom="q1",
        fill_color=CATEGORY20[0],
        line_color="black",
        source=df,
    )
    loww = fig.segment(x0="x0", y0="lw", x1="x1", y1="lw", line_color="black", source=df)
    upw = fig.segment(x0="x0", y0="uw", x1="x1", y1="uw", line_color="black", source=df)
    if outx:
        circ = fig.circle(  # pylint: disable=too-many-function-args
            outx, outy, size=3, line_color="black", color=CATEGORY20[6], fill_alpha=0.6
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
    if grp_cnt_stats is None and y is not None:
        lbl = timeunit if timeunit else "Bin"
        tooltips.insert(0, (lbl, "@grp"))
    fig.add_tools(
        HoverTool(
            renderers=[upw, utail, ubox, lbox, ltail, loww],
            tooltips=tooltips,
            point_policy="follow_mouse",
        )
    )
    tweak_figure(fig, "box")
    if y is None:
        fig.xaxis.major_tick_line_color = None
        fig.xaxis.major_label_text_font_size = "0pt"
    fig.xaxis.axis_label = x if y is not None else None
    fig.yaxis.axis_label = x if y is None else y

    minw = min(outy) if outy else np.nan
    maxw = max(outy) if outy else np.nan
    _format_axis(fig, min(df["lw"].min(), minw), max(df["uw"].max(), maxw), "y")

    if not grp_cnt_stats and y and not timeunit:  # format categorical axis tick values
        endpts = list(df["lb"]) + [df.iloc[len(df) - 1]["ub"]]
        # start by rounding to the length of the largest possible number
        round_to = -len(str(max([abs(int(ept)) for ept in endpts])))
        ticks = np.round(endpts, round_to)
        nticks = len(df) // 5 + 1
        show_ticks = [ticks[i] for i in range(len(ticks)) if i % nticks == 0]
        while len(set(show_ticks)) != len(show_ticks):  # round until show ticks unique
            round_to += 1
            ticks = np.round(endpts, round_to)
            show_ticks = [ticks[i] for i in range(len(ticks)) if i % nticks == 0]
        # format the ticks
        ticks = [int(tick) if tick.is_integer() else tick for tick in ticks]
        ticks = _format_ticks(ticks)
        fig.xaxis.ticker = list(range(len(df) + 1))
        fig.xaxis.formatter = FuncTickFormatter(  # overide bokeh ticks
            args={"vals": ticks, "mod": nticks},
            code="""
                if (index % mod == 0) return vals[index];
                return "";
            """,
        )
        tweak_figure(fig, "boxnum")
        fig.xaxis.major_label_text_font_size = "10pt"

    if timeunit == "Week of":
        fig.xaxis.axis_label = x + ", the week of"

    return Panel(child=row(fig), title="Box Plot")


def line_viz(
    df: pd.DataFrame,
    x: str,
    y: str,
    yscale: str,
    plot_width: int,
    plot_height: int,
    ttl_grps: int,
) -> Panel:
    """
    Render multi-line chart
    """
    # pylint: disable=too-many-arguments,too-many-locals
    palette = CATEGORY20 * (len(df) // len(CATEGORY20) + 1)
    title = _make_title({f"{x}_ttl": ttl_grps, f"{x}_shw": len(df)}, x, y)
    df.index = df.index.astype(str)

    fig = figure(
        plot_height=plot_height,
        plot_width=plot_width,
        title=title,
        toolbar_location=None,
        tools=[],
        y_axis_type=yscale,
    )

    # bin endpoints for all histograms
    bins = df[0].iloc[0][1]
    # plot the value for a histgram bin at its midpoint
    ticks = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    # format the bin intervals
    intvls = _format_bin_intervals(bins)

    lns: Dict[str, Figure] = {}
    # add the lines
    for grp, (cnts, _), color in zip(df.index, df[0], palette):
        grp_name = (grp[:14] + "...") if len(grp) > 15 else grp
        source = ColumnDataSource({"x": ticks, "y": cnts, "intvls": intvls})
        lns[grp_name] = fig.line(x="x", y="y", color=color, source=source)
        tooltips = [(f"{x}", f"{grp}"), ("Frequency", "@y"), (f"{y} bin", "@intvls")]
        fig.add_tools(HoverTool(renderers=[lns[grp_name]], tooltips=tooltips))

    fig.add_layout(Legend(items=[(x, [lns[x]]) for x in lns]), "left")
    tweak_figure(fig)
    fig.yaxis.axis_label = "Frequency"
    fig.xaxis.axis_label = y
    _format_axis(fig, bins[0], bins[-1], "x")
    if yscale == "linear":
        yvals = [val for cnts, _ in df[0] for val in cnts]
        _format_axis(fig, min(yvals), max(yvals), "y")

    return Panel(child=row(fig), title="Line Chart")


def scatter_viz(
    df: pd.DataFrame,
    x: str,
    y: str,
    sample_sr_and_name: Tuple[Union[int, float], str],
    plot_width: int,
    plot_height: int,
) -> Any:
    """
    Render a scatter plot
    """
    # pylint: disable=too-many-arguments
    if sample_sr_and_name[1] == "sample size":
        title = (
            f"{y} by {x}"
            if len(df) < sample_sr_and_name[0]
            else f"{y} by {x} (sample size {sample_sr_and_name[0]})"
        )
    elif sample_sr_and_name[1] == "sample rate":
        title = f"{y} by {x} (sample rate {sample_sr_and_name[0]})"
    else:
        raise RuntimeError("parameter name should be either 'sample size' or 'sample rate'")

    tooltips = [("(x, y)", f"(@{{{x}}}, @{{{y}}})")]
    fig = figure(
        tools="hover",
        title=title,
        toolbar_location=None,
        tooltips=tooltips,
        plot_width=plot_width,
        plot_height=plot_height,
    )
    fig.circle(x, y, color=CATEGORY20[0], source=df)  # pylint: disable=too-many-function-args
    tweak_figure(fig)
    fig.xaxis.axis_label = x
    fig.yaxis.axis_label = y
    _format_axis(fig, df[x].min(), df[x].max(), "x")
    _format_axis(fig, df[y].min(), df[y].max(), "y")
    return Panel(child=fig, title="Scatter Plot")


def hexbin_viz(
    df: pd.DataFrame,
    x: str,
    y: str,
    plot_width: int,
    plot_height: int,
    tile_size: str,
    aspect_scale: float,
) -> Panel:
    """
    Render a hexbin plot
    """
    # pylint: disable=too-many-arguments,too-many-locals
    xmin, xmax = df[x].min(), df[x].max()
    ymin, ymax = df[y].min(), df[y].max()
    if tile_size == "auto":
        tile_size = (xmax - xmin) / 25
    title = f"{y} by {x}"
    aspect_scale = (ymax - ymin) / (xmax - xmin + 1e-9)
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

    palette = list(reversed(VIRIDIS))
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
    fig.add_tools(
        HoverTool(
            tooltips=[("Count", "@counts")],
            renderers=[rend],
        )
    )
    mapper = LinearColorMapper(palette=palette, low=min(bins.counts), high=max(bins.counts))
    color_bar = ColorBar(color_mapper=mapper, width=8, location=(0, 0))
    color_bar.label_standoff = 8
    fig.add_layout(color_bar, "left")
    tweak_figure(fig, "hex")
    _format_axis(fig, df[x].min(), df[x].max(), "x")
    _format_axis(fig, df[y].min(), df[y].max(), "y")
    fig.xaxis.axis_label = x
    fig.yaxis.axis_label = y

    return Panel(child=fig, title="Hexbin Plot")


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
    df["grp_names"] = list(zip(df[x], df[y]))
    data_source = ColumnDataSource(data=df)
    title = _make_title(grp_cnt_stats, x, y)
    plot_width = 19 * len(df) if len(df) > 50 else plot_width
    fig = figure(
        plot_height=plot_height,
        plot_width=plot_width,
        title=title,
        toolbar_location=None,
        tools="hover",
        tooltips=[(f"{x}", f"@{{{x}}}"), (f"{y}", f"@{{{y}}}"), ("Count", "@cnt")],
        x_range=FactorRange(*df["grp_names"]),
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
    fig.xaxis.major_label_orientation = np.pi / 2
    _format_axis(fig, 0, df["cnt"].max(), "y")
    return Panel(child=fig, title="Nested Bar Chart")


def stacked_viz(
    df: pd.DataFrame,
    x: str,
    y: str,
    grp_cnt_stats: Dict[str, int],
    plot_width: int,
    plot_height: int,
    timeunit: Optional[str] = None,
) -> Panel:
    """
    Render a stacked bar chart
    """
    # pylint: disable=too-many-arguments,too-many-locals
    # percent
    df2 = df.div(df.sum(axis=1), axis=0) * 100
    df.columns = [f"{col}_cnt" for col in df.columns]
    # final dataframe contains percent and count
    df = pd.concat([df2, df], axis=1)

    title = _make_title(grp_cnt_stats, x, y)
    if not timeunit:
        if grp_cnt_stats[f"{x}_shw"] > 30:
            plot_width = 32 * grp_cnt_stats[f"{x}_shw"]
    else:
        if len(df) > 30:
            plot_width = 32 * len(df)

    fig = figure(
        plot_height=plot_height,
        plot_width=plot_width,
        title=title,
        toolbar_location=None,
        x_range=list(df.index),
    )
    grps = list(df2.columns)
    palette = PASTEL1 * (len(grps) // len(PASTEL1) + 1)
    if "Others" in grps:
        colours = palette[0 : len(grps) - 1] + ("#636363",)
    else:
        colours = palette[0 : len(grps)]
    source = ColumnDataSource(data=df)
    renderers = fig.vbar_stack(
        stackers=grps,
        x="index",
        width=0.9,
        source=source,
        line_width=1,
        color=colours,
    )
    grps = [(grp[:14] + "...") if len(grp) > 15 else grp for grp in grps]

    legend = Legend(items=[(grp, [rend]) for grp, rend in zip(grps, renderers)])
    legend.label_text_font_size = "8pt"
    fig.add_layout(legend, "right")

    if not timeunit:
        # include percent and count in the tooltip
        formatter = CustomJSHover(
            args=dict(source=source),
            code="""
            const cur_bar = special_vars.data_x - 0.5
            const name_cnt = special_vars.name + '_cnt'
            return source.data[name_cnt][cur_bar] + '';
        """,
        )
        for rend in renderers:
            hover = HoverTool(
                tooltips=[
                    (x, "@index"),
                    (y, "$name"),
                    ("Percentage", "@$name%"),
                    ("Count", "@{%s}{custom}" % rend.name),
                ],
                formatters={"@{%s}" % rend.name: formatter},
                renderers=[rend],
            )
            fig.add_tools(hover)
        fig.yaxis.axis_label = "Percent"
    else:
        # below is for having percent and count in the tooltip
        formatter = CustomJSHover(
            args=dict(source=source),
            code="""
            const columns = Object.keys(source.data)
            const cur_bar = special_vars.data_x - 0.5
            var ttl_bar = 0
            for (let i = 0; i < columns.length; i++) {
                if (columns[i] != 'index'){
                    ttl_bar = ttl_bar + source.data[columns[i]][cur_bar]
                }
            }
            const cur_val = source.data[special_vars.name][cur_bar]
            return (cur_val/ttl_bar * 100).toFixed(2)+'%';
        """,
        )
        for rend in renderers:
            hover = HoverTool(
                tooltips=[
                    (y, "$name"),
                    (timeunit, "@index"),
                    ("Count", "@$name"),
                    ("Percent", "@{%s}{custom}" % rend.name),
                ],
                formatters={"@{%s}" % rend.name: formatter},
                renderers=[rend],
            )
            fig.add_tools(hover)
        fig.yaxis.axis_label = "Count"
        _format_axis(fig, 0, df.sum(axis=1).max(), "y")
        fig.xaxis.axis_label = x
        if timeunit == "Week of":
            fig.xaxis.axis_label = x + ", the week of"

    tweak_figure(fig, "stacked")

    return Panel(child=fig, title="Stacked Bar Chart")


def heatmap_viz(
    df: pd.DataFrame,
    x: str,
    y: str,
    grp_cnt_stats: Dict[str, int],
    plot_width: int,
    plot_height: int,
) -> Panel:
    """
    Render a heatmap
    """
    # pylint: disable=too-many-arguments
    title = _make_title(grp_cnt_stats, x, y)

    source = ColumnDataSource(data=df)
    palette = RDBU[(len(RDBU) // 2 - 1) :]
    mapper = LinearColorMapper(palette=palette, low=df["cnt"].min() - 0.01, high=df["cnt"].max())
    if grp_cnt_stats[f"{x}_shw"] > 60:
        plot_width = 16 * grp_cnt_stats[f"{x}_shw"]
    if grp_cnt_stats[f"{y}_shw"] > 10:
        plot_height = 70 + 18 * grp_cnt_stats[f"{y}_shw"]
    fig = figure(
        x_range=sorted(list(set(df[x]))),
        y_range=sorted(list(set(df[y]))),
        toolbar_location=None,
        tools=[],
        x_axis_location="below",
        title=title,
        plot_width=plot_width,
        plot_height=plot_height,
    )

    renderer = fig.rect(
        x=x,
        y=y,
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
            tooltips=[
                (x, f"@{{{x}}}"),
                (y, f"@{{{y}}}"),
                ("Count", "@cnt"),
            ],
            mode="mouse",
            renderers=[renderer],
        )
    )
    fig.add_layout(color_bar, "right")

    tweak_figure(fig, "heatmap")
    fig.yaxis.formatter = FuncTickFormatter(
        code="""
            if (tick.length > 15) return tick.substring(0, 14) + '...';
            else return tick;
        """
    )
    return Panel(child=fig, title="Heat Map")


def dt_line_viz(
    df: pd.DataFrame,
    x: str,
    timeunit: str,
    yscale: str,
    plot_width: int,
    plot_height: int,
    show_yticks: bool,
    miss_pct: Optional[float] = None,
    y: Optional[str] = None,
) -> Figure:
    """
    Render a line chart
    """
    # pylint: disable=too-many-arguments
    agg = "freq" if miss_pct is not None else f"{df.columns[1]}"
    title_agg = agg if agg != "runningtotal" else "running total"
    if miss_pct is not None:
        title = f"{x} ({miss_pct}% missing)" if miss_pct > 0 else f"{x}"
        tooltips = [(timeunit, "@lbl"), ("Frequency", "@freq"), ("Percent", "@pct%")]
    else:
        title = f"{title_agg} of {y} by {x}"
        tooltips = [(timeunit, "@lbl"), (title_agg, f"@{df.columns[1]}")]
    fig = Figure(
        plot_width=plot_width,
        plot_height=plot_height,
        toolbar_location=None,
        title=title,
        tools=[],
        y_axis_type=yscale,
        x_axis_type="datetime",
    )
    fig.line(
        source=df,
        x=x,
        y=agg,
        line_width=2,
        line_alpha=0.8,
        color="#7e9ac8",
    )
    hover = HoverTool(
        tooltips=tooltips,
        mode="vline",
    )
    fig.add_tools(hover)

    tweak_figure(fig, "line", show_yticks)
    if show_yticks and yscale == "linear":
        _format_axis(fig, 0, df[agg].max(), "y")

    if y:
        fig.yaxis.axis_label = f"{title_agg} of {y}"
        fig.xaxis.axis_label = x
        cap_agg = title_agg.title()
        return Panel(child=fig, title=f"Line Chart ({cap_agg})")

    fig.yaxis.axis_label = "Frequency"
    return fig


def dt_multiline_viz(
    data: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]],
    x: str,
    y: str,
    timeunit: str,
    yscale: str,
    plot_width: int,
    plot_height: int,
    grp_cnt_stats: Dict[str, int],
    max_lbl_len: int = 15,
    z: Optional[str] = None,
    agg: Optional[str] = None,
) -> Panel:
    """
    Render multi-line chart
    """
    # pylint: disable=too-many-arguments,too-many-locals
    grps = list(data.keys())
    palette = CATEGORY20 * (len(grps) // len(CATEGORY20) + 1)
    if z is None:
        title = _make_title(grp_cnt_stats, x, y)
    else:
        title = f"{agg} of {_make_title(grp_cnt_stats, z, y)} over {x}"
    agg = "Frequency" if agg is None else agg

    fig = figure(
        tools=[],
        title=title,
        toolbar_location=None,
        plot_width=plot_width,
        plot_height=plot_height,
        y_axis_type=yscale,
        x_axis_type="datetime",
    )

    ymin, ymax = np.Inf, -np.Inf
    plot_dict = dict()
    for grp, colour in zip(grps, palette):
        grp_name = (grp[: (max_lbl_len - 1)] + "...") if len(grp) > max_lbl_len else grp
        source = ColumnDataSource({"x": data[grp][1], "y": data[grp][0], "lbl": data[grp][2]})
        plot_dict[grp_name] = fig.line(x="x", y="y", source=source, color=colour, line_width=1.3)
        fig.add_tools(
            HoverTool(
                renderers=[plot_dict[grp_name]],
                tooltips=[
                    (f"{y}", f"{grp}"),
                    (agg, "@y"),
                    (timeunit, "@lbl"),
                ],
                mode="mouse",
            )
        )
        ymin, ymax = min(ymin, min(data[grp][0])), max(ymax, max(data[grp][0]))

    legend = Legend(items=[(x, [plot_dict[x]]) for x in plot_dict])
    tweak_figure(fig, "line", True)
    fig.add_layout(legend, "right")
    fig.legend.click_policy = "hide"
    fig.yaxis.axis_label = f"{agg} of {y}" if z else "Frequency"
    fig.xaxis.axis_label = x
    if yscale == "linear":
        _format_axis(fig, ymin, ymax, "y")

    return Panel(child=fig, title="Line Chart")


def format_ov_stats(stats: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Render statistics information for distribution grid
    """
    # pylint: disable=too-many-locals
    nrows, ncols, npresent_cells, nrows_wo_dups, mem_use, dtypes_cnt = stats.values()
    ncells = nrows * ncols

    data = {
        "Number of Variables": ncols,
        "Number of Rows": nrows,
        "Missing Cells": float(ncells - npresent_cells),
        "Missing Cells (%)": 1 - (npresent_cells / ncells),
        "Duplicate Rows": nrows - nrows_wo_dups,
        "Duplicate Rows (%)": 1 - (nrows_wo_dups / nrows),
        "Total Size in Memory": float(mem_use),
        "Average Row Size in Memory": mem_use / nrows,
    }
    return {k: _format_values(k, v) for k, v in data.items()}, dtypes_cnt


def format_num_stats(data: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Format numerical statistics
    """
    overview = {
        "Approximate Distinct Count": data["nuniq"],
        "Approximate Unique (%)": data["nuniq"] / data["npres"],
        "Missing": data["nrows"] - data["npres"],
        "Missing (%)": 1 - (data["npres"] / data["nrows"]),
        "Infinite": (data["npres"] - data["nreals"]),
        "Infinite (%)": (data["npres"] - data["nreals"]) / data["nrows"],
        "Memory Size": data["mem_use"],
        "Mean": data["mean"],
        "Minimum": data["min"],
        "Maximum": data["max"],
        "Zeros": data["nzero"],
        "Zeros (%)": data["nzero"] / data["nrows"],
        "Negatives": data["nneg"],
        "Negatives (%)": data["nneg"] / data["nrows"],
    }
    data["qntls"].index = np.round(data["qntls"].index, 2)
    quantile = {
        "Minimum": data["min"],
        "5-th Percentile": data["qntls"].loc[0.05],
        "Q1": data["qntls"].loc[0.25],
        "Median": data["qntls"].loc[0.50],
        "Q3": data["qntls"].loc[0.75],
        "95-th Percentile": data["qntls"].loc[0.95],
        "Maximum": data["max"],
        "Range": data["max"] - data["min"],
        "IQR": data["qntls"].loc[0.75] - data["qntls"].loc[0.25],
    }
    descriptive = {
        "Mean": data["mean"],
        "Standard Deviation": data["std"],
        "Variance": data["std"] ** 2,
        "Sum": data["mean"] * data["npres"],
        "Skewness": float(data["skew"]),
        "Kurtosis": float(data["kurt"]),
        "Coefficient of Variation": data["std"] / data["mean"] if data["mean"] != 0 else np.nan,
    }

    return {
        "Overview": {k: _format_values(k, v) for k, v in overview.items()},
        "Quantile Statistics": {k: _format_values(k, v) for k, v in quantile.items()},
        "Descriptive Statistics": {k: _format_values(k, v) for k, v in descriptive.items()},
    }


def format_cat_stats(
    stats: Dict[str, Any],
    len_stats: Dict[str, Any],
    letter_stats: Dict[str, Any],
) -> Dict[str, Dict[str, str]]:
    """
    Format categorical statistics
    """
    ov_stats = {
        "Approximate Distinct Count": stats["nuniq"],
        "Approximate Unique (%)": stats["nuniq"] / stats["npres"],
        "Missing": stats["nrows"] - stats["npres"],
        "Missing (%)": 1 - stats["npres"] / stats["nrows"],
        "Memory Size": stats["mem_use"],
    }
    sampled_rows = ("1st row", "2nd row", "3rd row", "4th row", "5th row")
    smpl = dict(zip(sampled_rows, stats["first_rows"]))

    return {
        "Overview": {k: _format_values(k, v) for k, v in ov_stats.items()},
        "Length": {k: _format_values(k, v) for k, v in len_stats.items()},
        "Sample": {k: f"{v[:18]}..." if len(v) > 18 else v for k, v in smpl.items()},
        "Letter": {k: _format_values(k, v) for k, v in letter_stats.items()},
    }


def stats_viz_dt(stats: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Render statistics panel for datetime data
    """

    return {"Overview": {k: _format_values(k, v) for k, v in stats.items()}}


def render_distribution_grid(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Create visualizations for plot(df)
    """
    # pylint: disable=too-many-locals
    plot_width = cfg.plot.width if cfg.plot.width is not None else 324
    plot_height = cfg.plot.height if cfg.plot.height is not None else 300

    figs: List[Figure] = []
    htgs: Dict[str, Any] = {}
    nrows = itmdt["stats"]["nrows"]
    titles: List[str] = []
    for col, dtp, data in itmdt["data"]:
        if isinstance(dtp, (Nominal, GeoGraphy, SmallCardNum, GeoPoint)):
            df, ttl_grps = data
            fig = bar_viz(
                df,
                ttl_grps,
                nrows,
                col,
                plot_width,
                plot_height,
                False,
                cfg.bar,
            )
            htgs[col] = cfg.bar.grid_how_to_guide()
        elif isinstance(dtp, Continuous):
            fig = hist_viz(
                data, nrows, col, cfg.hist.yscale, cfg.hist.color, plot_width, plot_height, False
            )
            htgs[col] = cfg.hist.grid_how_to_guide()
        elif isinstance(dtp, DateTime):
            df, timeunit, miss_pct = data
            fig = dt_line_viz(
                df, col, timeunit, cfg.line.yscale, plot_width, plot_height, False, miss_pct
            )
        else:
            raise ValueError(f"unprocessed col:{col}, type:{dtp}")
        fig.frame_height = plot_height
        titles.append(fig.title.text)
        fig.title.text = ""
        figs.append(fig)

    if cfg.insight.enable and cfg.stats.enable:
        toggle_content = "Stats and Insights"
    elif cfg.stats.enable:
        toggle_content = "Stats"
    else:
        toggle_content = "Insights"
    return {
        "layout": figs,
        "meta": titles,
        "tabledata": format_ov_stats(itmdt["stats"]) if cfg.stats.enable else None,
        "overview_insights": itmdt["overview_insights"] if cfg.insight.enable else None,
        "column_insights": itmdt["column_insights"] if cfg.insight.enable else None,
        "container_width": plot_width * 3,
        "toggle_content": toggle_content,
        "how_to_guide": htgs,
    }


def render_cat(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Create visualizations for plot(df, Nominal)
    """
    # pylint: disable=too-many-locals,too-many-branches

    if cfg.plot.report:
        plot_width = 400
        plot_height = 350
        plot_width_bar = plot_width
        plot_height_bar = plot_height
    else:
        plot_width = cfg.plot.width if cfg.plot.width is not None else 450
        plot_height = cfg.plot.height if cfg.plot.height is not None else 400
        plot_width_bar = plot_width
        plot_height_bar = plot_height
    tabs: List[Panel] = []
    htgs: Dict[str, List[Tuple[str, str]]] = {}
    col, data = itmdt["col"], itmdt["data"]
    # overview, word length, and charater level statistcs
    if cfg.stats.enable:
        stats, len_stats, letter_stats = (
            data["stats"],
            data["len_stats"],
            data["letter_stats"],
        )
    if cfg.bar.enable:
        fig = bar_viz(
            data["bar"].to_frame(),
            data["nuniq"],
            data["nrows"],
            col,
            plot_width_bar,
            plot_height_bar,
            True,
            cfg.bar,
        )
        tabs.append(Panel(child=row(fig), title="Bar Chart"))
        htgs["Bar Chart"] = cfg.bar.how_to_guide(plot_height, plot_width)
    if cfg.pie.enable:
        fig, color_list = pie_viz(
            data["pie"].to_frame(), data["nrows"], col, plot_width, plot_height, cfg.pie
        )
        tabs.append(fig)
        htgs["Pie Chart"] = cfg.pie.how_to_guide(color_list, plot_height, plot_width)
    if cfg.wordcloud.enable:
        if data["nuniq_words_cloud"] > 0:
            tabs.append(wordcloud_viz(data["word_cnts_cloud"], plot_width, plot_height))
            htgs["Word Cloud"] = cfg.wordcloud.how_to_guide(plot_height, plot_width)
    if cfg.wordfreq.enable:
        if data["nwords_freq"] > 0:
            tabs.append(
                wordfreq_viz(
                    data["word_cnts_freq"],
                    data["nwords_freq"],
                    plot_width,
                    plot_height,
                    cfg.wordfreq,
                )
            )
            htgs["Word Frequency"] = cfg.wordfreq.how_to_guide(plot_height, plot_width)
    if cfg.wordlen.enable:
        length_dist = hist_viz(
            data["len_hist"],
            data["nrows"],
            "Word Length",
            cfg.wordlen.yscale,
            cfg.wordlen.color,
            plot_width,
            plot_height,
            True,
        )
        tabs.append(Panel(child=row(length_dist), title="Word Length"))
        htgs["Word Length"] = cfg.wordlen.how_to_guide(plot_height, plot_width)

    if cfg.value_table.enable:
        htgs["Value Table"] = cfg.value_table.how_to_guide()
        stats = data["stats"]
        value_table = _value_table(
            data["value_table"], stats["nrows"], stats["npres"], stats["nuniq"]
        )
    else:
        value_table = []

    # panel.child.children[0] is a figure
    for panel in tabs[0:]:
        panel.child.children[0].frame_width = int(plot_width * 0.9)

    if len(tabs) > 0:
        tabs[0].child.children[0].frame_width = int(plot_width_bar * 0.9)

    return {
        "tabledata": format_cat_stats(stats, len_stats, letter_stats) if cfg.stats.enable else [],
        "value_table": value_table,
        "insights": nom_insights(data, col, cfg) if cfg.insight.enable else [],
        "layout": [panel.child.children[0] for panel in tabs],
        "meta": [tab.title for tab in tabs],
        "container_width": plot_width + 110,
        "how_to_guide": htgs,
    }


def render_geo(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Create visualizations for plot(df, GeoGraphy)
    """
    # pylint: disable=too-many-locals,too-many-branches

    if cfg.plot.report:
        plot_width = 450
        plot_height = 400
        plot_width_bar = 280
        plot_height_bar = 248
    else:
        plot_width = cfg.plot.width if cfg.plot.width is not None else 450
        plot_height = cfg.plot.height if cfg.plot.height is not None else 400
        plot_width_bar = plot_width
        plot_height_bar = plot_height
    tabs: List[Panel] = []
    htgs: Dict[str, List[Tuple[str, str]]] = {}
    col, data = itmdt["col"], itmdt["data"]
    # overview, word length, and charater level statistcs
    if cfg.stats.enable:
        stats, len_stats, letter_stats = (
            data["stats"],
            data["len_stats"],
            data["letter_stats"],
        )
    if cfg.bar.enable:
        fig = bar_viz(
            data["bar"].to_frame(),
            data["nuniq"],
            data["nrows"],
            col,
            plot_width_bar,
            plot_height_bar,
            True,
            cfg.bar,
        )
        tabs.append(Panel(child=row(fig), title="Bar Chart"))
        htgs["Bar Chart"] = cfg.bar.how_to_guide(plot_height, plot_width)
    if cfg.pie.enable:
        fig, color_list = pie_viz(
            data["pie"].to_frame(), data["nrows"], col, plot_width, plot_height, cfg.pie
        )
        tabs.append(fig)
        htgs["Pie Chart"] = cfg.pie.how_to_guide(color_list, plot_height, plot_width)
    if cfg.wordcloud.enable:
        if data["nuniq_words_cloud"] > 0:
            tabs.append(wordcloud_viz(data["word_cnts_cloud"], plot_width, plot_height))
            htgs["Word Cloud"] = cfg.wordcloud.how_to_guide(plot_height, plot_width)
    if cfg.wordfreq.enable:
        if data["nwords_freq"] > 0:
            tabs.append(
                wordfreq_viz(
                    data["word_cnts_freq"],
                    data["nwords_freq"],
                    plot_width,
                    plot_height,
                    cfg.wordfreq,
                )
            )
            htgs["Word Frequency"] = cfg.wordfreq.how_to_guide(plot_height, plot_width)

    geo_df = geo_viz(data["geo"].to_frame().rename(columns={col: "count"}), plot_width, "count")
    tabs.append(geo_df)

    if cfg.value_table.enable:
        htgs["Value Table"] = cfg.value_table.how_to_guide()
        value_table = _value_table(
            data["value_table"], stats["nrows"], stats["npres"], stats["nuniq"]
        )
    else:
        value_table = []

    # panel.child.children[0] is a figure
    for panel in tabs[0:]:
        panel.child.children[0].frame_width = int(plot_width * 0.9)
    tabs[0].child.children[0].frame_width = int(plot_width_bar * 0.9)
    return {
        "tabledata": format_cat_stats(stats, len_stats, letter_stats) if cfg.stats.enable else [],
        "value_table": value_table,
        "insights": nom_insights(data, col, cfg) if cfg.insight.enable else [],
        "layout": [panel.child.children[0] for panel in tabs],
        "meta": [tab.title for tab in tabs],
        "container_width": plot_width + 110,
        "how_to_guide": htgs,
    }


def _value_table(srs: pd.Series, nrows: int, npres: int, nuniq: int) -> List[Dict[str, Any]]:
    """
    Render the rows for the frequency table.
    """

    df = srs.to_frame()
    (col,) = df.columns

    val_sum = df[col].sum()  # sum of all value counts that appear in the table
    nothers = npres - val_sum  # count of values that do not appear in the table
    nmissing = nrows - npres  # number of missing values
    df["pct"] = (df[col] / nrows * 100).round(1)

    max_freq = max(df[col].max(), nothers, nmissing)

    rows: List[Dict[str, Any]] = []
    for index, record in df.iterrows():
        rows.append(
            {
                "label": index,
                "width": record[col] / max_freq,
                "width_perc": f"{record[col] / max_freq * 100}%",
                "count": int(record[col]),
                "percentage": f"{record['pct']}%" if record["pct"] >= 0.1 else "< 0.1%",
                "n": nrows,
                "extra_class": "",
            }
        )
    if nothers > 0:
        pct = round(nothers / nrows * 100, 1)
        rows.append(
            {
                "label": f"Other values ({nuniq - len(df)})",
                "width": nothers / max_freq,
                "width_perc": f"{nothers / max_freq * 100}%",
                "count": int(nothers),
                "percentage": f"{pct}%" if pct >= 0.1 else "< 0.1%",
                "n": nrows,
                "extra_class": "other",
            }
        )
    if nmissing > 0:
        pct = round(nmissing / nrows * 100, 1)
        rows.append(
            {
                "label": "(Missing)",
                "width": nmissing / max_freq,
                "width_perc": f"{nmissing / max_freq * 100}%",
                "count": int(nmissing),
                "percentage": f"{pct}%" if pct >= 0.1 else "< 0.1%",
                "n": nrows,
                "extra_class": "missing",
            }
        )

    return rows


def nom_insights(data: Dict[str, Any], col: str, cfg: Config) -> Dict[str, List[str]]:
    """
    Format the insights for plot(df, Nominal)
    """
    # pylint: disable=too-many-branches
    # insight dictionary, with a list associated with each plot
    ins: Dict[str, List[str]] = {
        "Stats": [],
        "Bar Chart": [],
        "Pie Chart": [],
        "Word Cloud": [],
        "Word Frequency": [],
        "Word Length": [],
    }

    if cfg.stats.enable:
        if data["nuniq"] == cfg.insight.constant__threshold:
            ins["Stats"].append(f"{col} has a constant value")
        if data["nuniq"] > cfg.insight.high_cardinality__threshold:
            nuniq = data["nuniq"]
            ins["Stats"].append(f"{col} has a high cardinality: {nuniq} distinct values")
        pmiss = round((data["nrows"] - data["stats"]["npres"]) / data["nrows"] * 100, 2)
        if pmiss > cfg.insight.missing__threshold:
            nmiss = data["nrows"] - data["stats"]["npres"]
            ins["Stats"].append(f"{col} has {nmiss} ({pmiss}%) missing values")
        if data["stats"]["nuniq"] == data["stats"]["npres"]:
            ins["Stats"].append(f"{col} has all distinct values")

    if cfg.bar.enable:
        if data["chisq"][1] > cfg.insight.uniform__threshold:
            ins["Bar Chart"].append(f"{col} is relatively evenly distributed")
        factor = round(data["bar"].iloc[0] / data["bar"].iloc[1], 2) if len(data["bar"]) > 1 else 0
        if factor > cfg.insight.outstanding_no1__threshold:
            val1, val2 = data["bar"].index[0], data["bar"].index[1]
            ins["Bar Chart"].append(
                f"""The largest value ({val1}) is over {factor} times larger than the second
                largest value ({val2})"""
            )
    if cfg.pie.enable:
        if (
            data["pie"].iloc[:2].sum() / data["nrows"] > cfg.insight.attribution__threshold
            and len(data["pie"]) >= 2
        ):
            vals = ", ".join(str(data["pie"].index[i]) for i in range(2))
            ins["Pie Chart"].append(f"The top 2 categories ({vals}) take over {0.5*100}%")
    if cfg.wordcloud.enable:
        if data["nuniq_words_cloud"] > cfg.insight.high_word_cardinality__threshold:
            nwords = data["nuniq_words_cloud"]
            ins["Word Cloud"].append(f"{col} contains many words: {nwords} words")
    if cfg.wordfreq.enable:
        factor = (
            round(data["word_cnts_freq"].iloc[0] / data["word_cnts_freq"].iloc[1], 2)
            if len(data["word_cnts_freq"]) > 1
            else 0
        )
        if factor > cfg.insight.outstanding_no1_word__threshold:
            val1, val2 = (
                data["word_cnts_freq"].index[0],
                data["word_cnts_freq"].index[1],
            )
            ins["Word Frequency"].append(
                f"""The largest value ({val1}) is over {factor} times larger than the second
                largest value ({val2})"""
            )
        if data["len_stats"]["Minimum"] == data["len_stats"]["Maximum"]:
            ins["Word Frequency"].append(f"{col} has words of constant length")

    return ins


def render_num(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Create visualizations for plot(df, Continuous)
    """
    # pylint: disable=too-many-locals
    if cfg.plot.report:
        plot_width = 400
        plot_height = 350
        plot_width_hist = plot_width
        plot_height_hist = plot_height
    else:
        plot_width = cfg.plot.width if cfg.plot.width is not None else 450
        plot_height = cfg.plot.height if cfg.plot.height is not None else 400
        plot_width_hist = plot_width
        plot_height_hist = plot_height
    col, data = itmdt["col"], itmdt["data"]
    tabs: List[Panel] = []
    htgs: Dict[str, List[Tuple[str, str]]] = {}

    if cfg.hist.enable:
        fig = hist_viz(
            data["hist"],
            data["nrows"],
            col,
            cfg.hist.yscale,
            cfg.hist.color,
            plot_width_hist,
            plot_height_hist,
            True,
        )
        tabs.append(Panel(child=row(fig), title="Histogram"))
        htgs["Histogram"] = cfg.hist.how_to_guide(plot_height, plot_width)
    if cfg.kde.enable:
        # when the column is constant, we wont display kde plot
        if data["kde"] is not None and (not math.isclose(data["min"], data["max"])):
            dens, kde = data["dens"], data["kde"]
            tabs.append(kde_viz(dens, kde, col, plot_width, plot_height, cfg.kde))
            htgs["KDE Plot"] = cfg.kde.how_to_guide(plot_height, plot_width)
    if cfg.qqnorm.enable and (not math.isclose(data["min"], data["max"])):
        # when the column is constant, we wont display qq plot
        if data["qntls"].any():
            qntls, mean, std = data["qntls"], data["mean"], data["std"]
            tabs.append(qqnorm_viz(qntls, mean, std, col, plot_width, plot_height, cfg.qqnorm))
            htgs["Normal Q-Q Plot"] = cfg.qqnorm.how_to_guide(plot_height, plot_width)
    if cfg.box.enable:
        box_data = {
            "grp": col,
            "q1": data["qrtl1"],
            "q2": data["qrtl2"],
            "q3": data["qrtl3"],
            "lw": data["lw"],
            "uw": data["uw"],
            "otlrs": [data["otlrs"]],
        }
        df = pd.DataFrame(box_data, index=[0])
        tabs.append(box_viz(df, col, plot_width, plot_height, cfg.box))
        htgs["Box Plot"] = cfg.box.univar_how_to_guide(plot_height, plot_width)

    if cfg.value_table.enable:
        htgs["Value Table"] = cfg.value_table.how_to_guide()
        value_table = _value_table(data["value_table"], data["nrows"], data["npres"], data["nuniq"])
    else:
        value_table = []

    # panel.child.children[0] is a figure
    for panel in tabs[0:]:
        panel.child.children[0].frame_width = int(plot_width * 0.9)
    if cfg.hist.enable:
        tabs[0].child.children[0].frame_width = int(plot_width_hist * 0.9)
    return {
        "tabledata": format_num_stats(data) if cfg.stats.enable else [],
        "value_table": value_table,
        "insights": cont_insights(data, col, cfg) if cfg.insight.enable else [],
        "layout": [panel.child for panel in tabs],
        "meta": [tab.title for tab in tabs],
        "container_width": plot_width + 110,
        "how_to_guide": htgs,
    }


def cont_insights(data: Dict[str, Any], col: str, cfg: Config) -> Dict[str, List[str]]:
    """
    Format the insights for plot(df, Continuous)
    """
    # pylint: disable=too-many-branches
    # insight dictionary with a list associated with each plot
    ins: Dict[str, List[str]] = {
        "Stats": [],
        "Histogram": [],
        "KDE Plot": [],
        "Normal Q-Q Plot": [],
        "Box Plot": [],
    }

    if cfg.stats.enable:
        pinf = round((data["npres"] - data["nreals"]) / data["nrows"] * 100, 2)
        if pinf > cfg.insight.infinity__threshold:
            ninf = data["npres"] - data["nreals"]
            ins["Stats"].append(f"{col} has {ninf} ({pinf}%) infinite values")
        pmiss = round((data["nrows"] - data["npres"]) / data["nrows"] * 100, 2)
        if pmiss > cfg.insight.missing__threshold:
            nmiss = data["nrows"] - data["npres"]
            ins["Stats"].append(f"{col} has {nmiss} ({pmiss}%) missing values")
        pneg = round(data["nneg"] / data["nrows"] * 100, 2)
        if pneg > cfg.insight.negatives__threshold:
            nneg = data["nneg"]
            ins["Stats"].append(f"{col} has {nneg} ({pneg}%) negatives")
        pzero = round(data["nzero"] / data["nrows"] * 100, 2)
        if pzero > cfg.insight.zeros__threshold:
            nzero = data["nzero"]
            ins["Stats"].append(f"{col} has {nzero} ({pzero}%) zeros")

    if cfg.hist.enable:
        if data["norm"][1] > cfg.insight.normal__threshold:
            ins["Histogram"].append(f"{col} is normally distributed")
        if data["chisq"][1] > cfg.insight.uniform__threshold:
            ins["Histogram"].append(f"{col} is uniformly distributed")
        skw = np.round(data["skew"], 4)
        if skw >= cfg.insight.skewed__threshold:
            ins["Histogram"].append(f"{col} is skewed right (\u03B31 = {skw})")
        if skw <= -cfg.insight.skewed__threshold:
            ins["Histogram"].append(f"{col} is skewed left (\u03B31 = {skw})")

    if cfg.qqnorm.enable:
        if data["norm"][1] <= 1 - cfg.insight.normal__threshold:
            pval = data["norm"][1]
            ins["Normal Q-Q Plot"].append(f"{col} is not normally distributed (p-value {pval})")

    if cfg.box.enable:
        if data["notlrs"] > cfg.insight.outlier__threshold:
            notlrs = data["notlrs"]
            ins["Box Plot"].append(f"{col} has {notlrs} outliers")

    return ins


def render_dt(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Create visualizations for plot(df, DateTime)
    """
    if cfg.plot.report:
        plot_width = 280
        plot_height = 248
    else:
        plot_width = cfg.plot.width if cfg.plot.width is not None else 450
        plot_height = cfg.plot.height if cfg.plot.height is not None else 400

    tabs: List[Panel] = []
    if cfg.line.enable:
        df, timeunit, miss_pct = itmdt["line"]
        fig = dt_line_viz(
            df, itmdt["col"], timeunit, cfg.line.yscale, plot_width, plot_height, True, miss_pct
        )
        fig.frame_width = int(plot_width * 0.95)
        tabs.append(Panel(child=fig, title="Line Chart"))
    return {
        "tabledata": stats_viz_dt(itmdt["data"]) if cfg.stats.enable else [],
        "insights": None,
        "layout": [panel.child for panel in tabs],
        "meta": [tab.title for tab in tabs],
        "container_width": plot_width + 50,
    }


def render_cat_num(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Create visualizations for plot(df, Nominal, Continuous)
    """
    plot_width = cfg.plot.width if cfg.plot.width is not None else 450
    plot_height = cfg.plot.height if cfg.plot.height is not None else 400

    tabs: List[Panel] = []
    htgs: Dict[str, List[Tuple[str, str]]] = {}
    data, x, y = itmdt["data"], itmdt["x"], itmdt["y"]

    if cfg.box.enable:
        # box plot
        df = data["box"].to_frame().reset_index()[: 5 * cfg.box.ngroups]
        df = df.pivot(index=x, columns="level_1", values=[0]).reset_index()
        df.columns = df.columns.get_level_values(1)
        df.columns = ["grp"] + list(df.columns[1:])
        tabs.append(box_viz(df, x, plot_width, plot_height, cfg.box, y, data["ttl_grps"]))
        htgs["Box Plot"] = cfg.box.nom_cont_how_to_guide(plot_height, plot_width)

    if cfg.line.enable:
        # multiline plot
        df = data["hist"].to_frame()[: cfg.line.ngroups]
        tabs.append(
            line_viz(
                df,
                x,
                y,
                cfg.line.yscale,
                plot_width,
                plot_height,
                data["ttl_grps"],
            )
        )
        htgs["Line Chart"] = cfg.line.nom_cont_how_to_guide(plot_height, plot_width)
    for panel in tabs:
        panel.child.children[0].frame_width = int(plot_width * 0.9)
    return {
        "layout": [panel.child for panel in tabs],
        "meta": [panel.title for panel in tabs],
        "container_width": plot_width + 170,
        "how_to_guide": htgs,
    }


def render_geo_num(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Render plots from plot(df, x, y) when x is a geography column
    and y is a numerical column
    """
    plot_width = cfg.plot.width if cfg.plot.width is not None else 450
    plot_height = cfg.plot.height if cfg.plot.height is not None else 400

    tabs: List[Panel] = []
    htgs: Dict[str, List[Tuple[str, str]]] = {}
    data, x, y = itmdt["data"], itmdt["x"], itmdt["y"]

    if cfg.box.enable:
        # box plot
        df = data["box"].to_frame().reset_index()[: 5 * cfg.box.ngroups]
        df = df.pivot(index=x, columns="level_1", values=[0]).reset_index()
        df.columns = df.columns.get_level_values(1)
        df.columns = ["grp"] + list(df.columns[1:])
        tabs.append(box_viz(df, x, plot_width, plot_height, cfg.box, y, data["ttl_grps"]))
        htgs["Box Plot"] = cfg.box.nom_cont_how_to_guide(plot_height, plot_width)

    if cfg.line.enable:
        # multiline plot
        df = data["hist"].to_frame()[: cfg.line.ngroups]
        tabs.append(
            line_viz(
                df,
                x,
                y,
                cfg.line.yscale,
                plot_width,
                plot_height,
                data["ttl_grps"],
            )
        )
        htgs["Line Chart"] = cfg.line.nom_cont_how_to_guide(plot_height, plot_width)

    df = data["value"].to_frame()
    tabs.append(geo_viz(df, plot_width, y))

    for panel in tabs:
        panel.child.children[0].frame_width = int(plot_width * 0.9)
    return {
        "layout": [panel.child for panel in tabs],
        "meta": [panel.title for panel in tabs],
        "container_width": plot_width + 170,
        "how_to_guide": htgs,
    }


def render_latlong_num(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Render plots from plot(df, x, y) when x is a latlong column
    and y is a numerical column
    """
    plot_width = cfg.plot.width if cfg.plot.width is not None else 450

    tabs: List[Panel] = []
    data, y = itmdt["data"], itmdt["y"]

    df = data["value"].to_frame()

    # latlong statstic plot
    tabs.append(latlong_viz(df, plot_width, y))

    for panel in tabs:
        panel.child.children[0].frame_width = int(plot_width * 0.9)
    return {
        "layout": [panel.child for panel in tabs],
        "meta": [panel.title for panel in tabs],
        "container_width": plot_width + 170,
    }


def render_two_num(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Create visualizations for plot(df, Continuous, Continuous)
    """
    plot_width = cfg.plot.width if cfg.plot.width is not None else 450
    plot_height = cfg.plot.height if cfg.plot.height is not None else 400

    tabs: List[Panel] = []
    htgs: Dict[str, List[Tuple[str, str]]] = {}
    data, x, y = itmdt["data"], itmdt["x"], itmdt["y"]

    if cfg.scatter.enable:
        # scatter plot
        if cfg.scatter.sample_size is not None:
            sample_sr_and_name: Tuple[Union[int, float], str] = (
                cfg.scatter.sample_size,
                "sample size",
            )
        elif cfg.scatter.sample_rate is not None:
            sample_sr_and_name = (cfg.scatter.sample_rate, "sample rate")
        else:
            raise RuntimeError("In scatter plot, sample size and sample rate are both not None")
        tabs.append(
            scatter_viz(
                data["scat"],
                x,
                y,
                sample_sr_and_name,
                plot_width,
                plot_height,
            )
        )
        htgs["Scatter Plot"] = cfg.scatter.how_to_guide(plot_height, plot_width)
    tile_size = None
    if cfg.hexbin.enable:
        # hexbin plot
        x_diff = data["hex"][x].max() - data["hex"][x].min()
        tile_size = cfg.hexbin.tile_size if cfg.hexbin.tile_size != "auto" else x_diff / 25
        aspect_scale = (data["hex"][y].max() - data["hex"][y].min()) / (x_diff + 1e-9)
        tabs.append(
            hexbin_viz(
                data["hex"],
                x,
                y,
                plot_width,
                plot_height,
                tile_size,
                aspect_scale,
            )
        )
        htgs["Hexbin Plot"] = cfg.hexbin.how_to_guide(tile_size, plot_height, plot_width)
    if cfg.box.enable:
        # box plot
        df = data["box"].to_frame().reset_index()
        df = df.pivot(index="grp", columns="level_1", values=[0]).reset_index()
        df.columns = df.columns.get_level_values(1)
        df.columns = ["grp"] + list(df.columns[1:])
        tabs.append(box_viz(df, x, plot_width, plot_height, cfg.box, y))
        htgs["Box Plot"] = cfg.box.two_cont_how_to_guide(plot_height, plot_width)

    for panel in tabs:
        try:
            panel.child.frame_width = int(plot_width * 0.9)
        except AttributeError:
            panel.child.children[0].frame_width = int(plot_width * 0.9)
    return {
        "layout": [panel.child for panel in tabs],
        "meta": [panel.title for panel in tabs],
        "container_width": plot_width + 80,
        "how_to_guide": htgs,
    }


def render_two_cat(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Create visualizations for plot(df, Nominal, Nominal)
    """
    # pylint: disable=too-many-locals
    plot_width = cfg.plot.width if cfg.plot.width is not None else 972
    plot_height = cfg.plot.height if cfg.plot.height is not None else 300

    tabs: List[Panel] = []
    htgs: Dict[str, List[Tuple[str, str]]] = {}
    df = itmdt["data"].to_frame("cnt").reset_index()
    x, y = itmdt["x"], itmdt["y"]
    xgrps, ygrps = df.groupby(x)["cnt"].sum(), df.groupby(y)["cnt"].sum()

    if (
        cfg.nested.enable
        and cfg.stacked.enable
        and cfg.heatmap.enable
        and cfg.nested.ngroups == cfg.stacked.ngroups == cfg.heatmap.ngroups
        and cfg.nested.nsubgroups == cfg.stacked.nsubgroups == cfg.heatmap.nsubgroups
    ):
        # parse the dataframe to consist of the ngroups largest x groups
        # and the nsubgroups largest y groups
        df, stats = parse_grps(df, cfg.nested.ngroups, cfg.nested.nsubgroups, x, y, xgrps, ygrps)
        df_nest = df_stack = df_heat = df
        stats_nest = stats_stack = stats_heat = stats
    else:
        if cfg.nested.enable:
            df_nest, stats_nest = parse_grps(
                df, cfg.nested.ngroups, cfg.nested.nsubgroups, x, y, xgrps, ygrps
            )
        if cfg.stacked.enable:
            df_stack, stats_stack = parse_grps(
                df, cfg.stacked.ngroups, cfg.stacked.nsubgroups, x, y, xgrps, ygrps
            )
        if cfg.heatmap.enable:
            df_heat, stats_heat = parse_grps(
                df, cfg.heatmap.ngroups, cfg.heatmap.nsubgroups, x, y, xgrps, ygrps
            )

    # nested bar chart
    if cfg.nested.enable:
        tabs.append(nested_viz(df_nest, x, y, stats_nest, plot_width, plot_height))
        htgs["Nested Bar Chart"] = cfg.nested.how_to_guide(x, y, plot_height, plot_width)
    # stacked bar chart
    if cfg.stacked.enable:
        # stacked bar chart
        # wrangle the dataframe into a pivot table format
        df_stack = df_stack.pivot(index=x, columns=y, values="cnt")
        df_stack.index.name = None
        # aggregate remaining groups into "Others"
        df_stack["Others"] = xgrps - df_stack.sum(axis=1)
        if df_stack["Others"].sum() < 1e-6:
            df_stack = df_stack.drop(columns="Others")
        tabs.append(stacked_viz(df_stack, x, y, stats_stack, plot_width, plot_height))
        htgs["Stacked Bar Chart"] = cfg.stacked.how_to_guide(x, y, plot_height, plot_width)
    # heat map
    if cfg.heatmap.enable:
        tabs.append(heatmap_viz(df_heat, x, y, stats_heat, plot_width, plot_height))
        htgs["Heat Map"] = cfg.heatmap.how_to_guide(x, y, plot_height, plot_width)

    return {
        "layout": [panel.child for panel in tabs],
        "meta": [panel.title for panel in tabs],
        "container_width": plot_width,
        "how_to_guide": htgs,
    }


def parse_grps(
    df: pd.DataFrame,
    ngroups: int,
    nsubgroups: int,
    x: str,
    y: str,
    xgrps: pd.DataFrame,
    ygrps: pd.DataFrame,
) -> Any:
    """
    Parse the data for nested bar chart, stacked bar chart, heat map
    according to the given ngroups and nsubgroups
    """
    # pylint: disable=too-many-arguments

    x_lrgst = xgrps.nlargest(ngroups)
    df = df[df[x].isin(x_lrgst.index)]
    y_lrgst = ygrps.nlargest(nsubgroups)
    df = df[df[y].isin(y_lrgst.index)]
    stats = {
        f"{x}_ttl": len(xgrps),
        f"{x}_shw": len(x_lrgst),
        f"{y}_ttl": len(ygrps),
        f"{y}_shw": len(y_lrgst),
    }
    if df.empty:
        size = min(len(x_lrgst), len(y_lrgst))
        df = pd.DataFrame({x: x_lrgst[:size], y: y_lrgst[:size]})
        df["cnt"] = 0
    df[[x, y]] = df[[x, y]].astype(str)
    # final format
    df = df.pivot_table(index=y, columns=x, values="cnt", fill_value=0, aggfunc="sum")
    df = df.unstack().to_frame("cnt").reset_index()

    return df, stats


def render_dt_num(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Render plots from plot(df, x, y) when x is dt and y is num
    """
    plot_width = cfg.plot.width if cfg.plot.width is not None else 450
    plot_height = cfg.plot.height if cfg.plot.height is not None else 400

    tabs: List[Panel] = []
    if cfg.line.enable:
        linedf_agg, timeunit = itmdt["linedata_agg"]
        tabs.append(
            dt_line_viz(
                linedf_agg,
                itmdt["x"],
                timeunit,
                cfg.line.yscale,
                plot_width,
                plot_height,
                True,
                y=itmdt["y"],
            )
        )

        linedf_running_total, timeunit = itmdt["linedata_running_total"]
        tabs.append(
            dt_line_viz(
                linedf_running_total,
                itmdt["x"],
                timeunit,
                cfg.line.yscale,
                plot_width,
                plot_height,
                True,
                y=itmdt["y"],
            )
        )

    if cfg.box.enable:
        boxdf, outx, outy, timeunit = itmdt["boxdata"]
        tabs.append(
            box_viz_dt(
                boxdf,
                outx,
                outy,
                itmdt["x"],
                plot_width,
                plot_height,
                itmdt["y"],
                timeunit=timeunit,
            )
        )

    return {
        "layout": [panel.child for panel in tabs],
        "meta": [panel.title for panel in tabs],
        "container_width": plot_width + 220,
    }


def render_dt_cat(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Render plots from plot(df, x, y) when x is dt and y is num
    """
    plot_width = cfg.plot.width if cfg.plot.width is not None else 972
    plot_height = cfg.plot.height if cfg.plot.height is not None else 400

    tabs: List[Panel] = []
    if cfg.line.enable:
        data, grp_cnt_stats, timeunit = itmdt["linedata"]
        tabs.append(
            dt_multiline_viz(
                data,
                itmdt["x"],
                itmdt["y"],
                timeunit,
                cfg.line.yscale,
                plot_width,
                plot_height,
                grp_cnt_stats,
            )
        )
    if cfg.stacked.enable:
        df, grp_cnt_stats, timeunit = itmdt["stackdata"]
        tabs.append(
            stacked_viz(
                df,
                itmdt["x"],
                itmdt["y"],
                grp_cnt_stats,
                plot_width,
                plot_height,
                timeunit,
            )
        )

    return {
        "layout": [panel.child for panel in tabs],
        "meta": [panel.title for panel in tabs],
        "container_width": plot_width,
    }


def render_dt_num_cat(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Render plots from plot(df, x, y) when x is dt and y is num
    """
    plot_width = cfg.plot.width if cfg.plot.width is not None else 972
    plot_height = cfg.plot.height if cfg.plot.height is not None else 400

    tabs: List[Panel] = []
    data, grp_cnt_stats, timeunit = itmdt["data"]
    tabs.append(
        dt_multiline_viz(
            data,
            itmdt["x"],
            itmdt["y"],
            timeunit,
            cfg.line.yscale,
            plot_width,
            plot_height,
            grp_cnt_stats,
            z=itmdt["z"],
            agg=itmdt["agg"],
        )
    )
    return {
        "layout": [panel.child for panel in tabs],
        "meta": [panel.title for panel in tabs],
        "container_width": plot_width,
    }


def render(itmdt: Intermediate, cfg: Config) -> Union[LayoutDOM, Dict[str, Any]]:
    """
    Render a basic plot
    Parameters
    ----------
    itmdt
        The Intermediate containing results from the compute function.
    cfg
        Config instance
    """
    # pylint: disable = too-many-branches

    if itmdt.visual_type == "distribution_grid":
        visual_elem = render_distribution_grid(itmdt, cfg)
    elif itmdt.visual_type == "categorical_column":
        visual_elem = render_cat(itmdt, cfg)
    elif itmdt.visual_type == "geography_column":
        visual_elem = render_geo(itmdt, cfg)
    elif itmdt.visual_type == "numerical_column":
        visual_elem = render_num(itmdt, cfg)
    elif itmdt.visual_type == "datetime_column":
        visual_elem = render_dt(itmdt, cfg)
    elif itmdt.visual_type == "cat_and_num_cols":
        visual_elem = render_cat_num(itmdt, cfg)
    elif itmdt.visual_type == "geo_and_num_cols":
        visual_elem = render_geo_num(itmdt, cfg)
    elif itmdt.visual_type == "latlong_and_num_cols":
        visual_elem = render_latlong_num(itmdt, cfg)
    elif itmdt.visual_type == "two_num_cols":
        visual_elem = render_two_num(itmdt, cfg)
    elif itmdt.visual_type == "two_cat_cols":
        visual_elem = render_two_cat(itmdt, cfg)
    elif itmdt.visual_type == "dt_and_num_cols":
        visual_elem = render_dt_num(itmdt, cfg)
    elif itmdt.visual_type == "dt_and_cat_cols":
        visual_elem = render_dt_cat(itmdt, cfg)
    elif itmdt.visual_type == "dt_cat_num_cols":
        visual_elem = render_dt_num_cat(itmdt, cfg)

    return visual_elem
