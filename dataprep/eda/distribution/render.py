"""
This module implements the visualization for the plot(df) function.
"""  # pylint: disable=too-many-lines
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
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

from ..dtypes import Continuous, DateTime, Nominal, is_dtype
from ..intermediate import Intermediate
from ..palette import CATEGORY20, PASTEL1, RDBU, VIRIDIS

__all__ = ["render"]


def tweak_figure(
    fig: Figure,
    ptype: Optional[str] = None,
    show_yticks: bool = False,
    max_lbl_len: int = 15,
) -> None:
    """
    Set some common attributes for a figure
    """
    fig.axis.major_label_text_font_size = "9pt"
    fig.title.text_font_size = "10pt"
    fig.axis.minor_tick_line_color = "white"
    if ptype in ["pie", "qq", "heatmap"]:
        fig.ygrid.grid_line_color = None
    if ptype in ["bar", "pie", "hist", "kde", "qq", "heatmap", "line"]:
        fig.xgrid.grid_line_color = None
    if ptype in ["bar", "hist", "line"] and not show_yticks:
        fig.ygrid.grid_line_color = None
        fig.yaxis.major_label_text_font_size = "0pt"
        fig.yaxis.major_tick_line_color = None
    if ptype in ["bar", "nested", "stacked", "heatmap", "box"]:
        fig.xaxis.major_label_orientation = np.pi / 3
        fig.xaxis.formatter = FuncTickFormatter(
            code="""
            if (tick.length > %d) return tick.substring(0, %d-2) + '...';
            else return tick;
        """
            % (max_lbl_len, max_lbl_len)
        )
    if ptype in ["nested", "stacked", "box"]:
        fig.xgrid.grid_line_color = None
    if ptype in ["nested", "stacked"]:
        fig.y_range.start = 0
        fig.x_range.range_padding = 0.03
    if ptype in ["line", "boxnum"]:
        fig.min_border_right = 20
        fig.xaxis.major_label_standoff = 7
        fig.xaxis.major_label_orientation = 0
        fig.xaxis.major_tick_line_color = None


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


def _format_ticks(ticks: List[float]) -> List[str]:
    """
    Format the tick values
    """
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

    return formatted_ticks


def _format_axis(fig: Figure, minv: int, maxv: int, axis: str) -> None:
    """
    Format the axis ticks
    """  # pylint: disable=too-many-locals
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
    ticks = [float(minv)]
    while max(ticks) + gap < maxv:
        ticks.append(max(ticks) + gap)
    ticks = np.round(ticks, round_to)
    ticks = [int(tick) if tick.is_integer() else tick for tick in ticks]
    formatted_ticks = _format_ticks(ticks)

    if axis == "x":
        fig.xgrid.ticker = ticks
        fig.xaxis.ticker = ticks
        fig.xaxis.major_label_overrides = dict(zip(ticks, formatted_ticks))
        fig.xaxis.major_label_text_font_size = "10pt"
        fig.xaxis.major_label_standoff = 7
        # fig.xaxis.major_label_orientation = 0
        fig.xaxis.major_tick_line_color = None
    elif axis == "y":
        fig.ygrid.ticker = ticks
        fig.yaxis.ticker = ticks
        fig.yaxis.major_label_overrides = dict(zip(ticks, formatted_ticks))
        fig.yaxis.major_label_text_font_size = "10pt"
        fig.yaxis.major_label_standoff = 5


def _format_bin_intervals(bins_arr: np.ndarray) -> List[str]:
    """
    Auxillary function to format bin intervals in a histogram
    """
    bins_arr = np.round(bins_arr, 3)
    bins_arr = [int(val) if float(val).is_integer() else val for val in bins_arr]
    intervals = [f"[{bins_arr[i]}, {bins_arr[i + 1]})" for i in range(len(bins_arr) - 2)]
    intervals.append(f"[{bins_arr[-2]},{bins_arr[-1]}]")
    return intervals


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
    show_yticks: bool,
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
    fig.vbar(x="index", top=col, width=0.9, source=df)
    fig.yaxis.axis_label = "Count"
    tweak_figure(fig, "bar", show_yticks)
    _format_axis(fig, 0, df[col].max(), "y")
    return Panel(child=row(fig), title="Word Frequency")


def bar_viz(
    df: pd.DataFrame,
    ttl_grps: int,
    nrows: int,
    col: str,
    yscale: str,
    plot_width: int,
    plot_height: int,
    show_yticks: bool,
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
        y_axis_type=yscale,
    )
    fig.vbar(x="index", width=0.9, top=col, bottom=0.01, source=df)
    tweak_figure(fig, "bar", show_yticks)
    fig.yaxis.axis_label = "Count"
    if ttl_grps > len(df):
        fig.xaxis.axis_label = f"Top {len(df)} of {ttl_grps} {col}"
        fig.xaxis.axis_label_standoff = 0
    if show_yticks and yscale == "linear":
        _format_axis(fig, 0, df[col].max(), "y")
    return fig


def pie_viz(
    df: pd.DataFrame,
    nrows: int,
    col: str,
    plot_width: int,
    plot_height: int,
) -> Panel:
    """
    Render a pie chart
    """
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
    color_list = CATEGORY20 * (len(df) // len(CATEGORY20) + 1)
    df["colour"] = color_list[0 : len(df)]
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
    return Panel(child=row(fig), title="Pie Chart")


def hist_viz(
    hist: Tuple[np.ndarray, np.ndarray],
    nrows: int,
    col: str,
    yscale: str,
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
    fig = Figure(
        plot_height=plot_height,
        plot_width=plot_width,
        title=col,
        toolbar_location=None,
        y_axis_type=yscale,
    )
    bottom = 0 if yscale == "linear" or df.empty else df["freq"].min() / 2
    fig.quad(
        source=df,
        left="left",
        right="right",
        bottom=bottom,
        alpha=0.5,
        top="freq",
        fill_color="#6baed6",
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
    yscale: str,
    plot_width: int,
    plot_height: int,
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
        y_axis_type=yscale,
    )
    bottom = 0 if yscale == "linear" or df.empty else df["dens"].min() / 2
    hist = fig.quad(
        source=df,
        left="left",
        right="right",
        bottom=bottom,
        alpha=0.5,
        top="dens",
        fill_color="#6baed6",
    )
    hover_hist = HoverTool(
        renderers=[hist],
        tooltips=[("Bin", "@intvl"), ("Density", "@dens")],
        mode="vline",
    )
    pts_rng = np.linspace(df.loc[0, "left"], df.loc[len(df) - 1, "right"], 1000)
    pdf = kde(pts_rng)
    line = fig.line(x=pts_rng, y=pdf, line_color="#9467bd", line_width=2, alpha=0.5)
    hover_dist = HoverTool(renderers=[line], tooltips=[("x", "@x"), ("y", "@y")])
    fig.add_tools(hover_hist)
    fig.add_tools(hover_dist)
    tweak_figure(fig, "kde")
    fig.yaxis.axis_label = "Density"
    fig.xaxis.axis_label = col
    _format_axis(fig, df.iloc[0]["left"], df.iloc[-1]["right"], "x")
    if yscale == "linear":
        _format_axis(fig, 0, max(df["dens"].max(), pdf.max()), "y")
    return Panel(child=row(fig), title="KDE Plot")


def qqnorm_viz(
    qntls: pd.Series,
    mean: float,
    std: float,
    col: str,
    plot_width: int,
    plot_height: int,
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
        color=CATEGORY20[0],
    )
    vals = np.concatenate((theory_qntls, qntls))
    fig.line(x=[vals.min(), vals.max()], y=[vals.min(), vals.max()], color="red")
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
        x_range=list(df["grp"].astype(str)),
    )
    low = fig.segment(x0="x0", y0="lw", x1="x1", y1="lw", line_color="black", source=df)
    ltail = fig.segment(x0="grp", y0="lw", x1="grp", y1="q1", line_color="black", source=df)
    lbox = fig.vbar(
        x="grp",
        width=width,
        top="q2",
        bottom="q1",
        fill_color=CATEGORY20[0],
        line_color="black",
        source=df,
    )
    ubox = fig.vbar(
        x="grp",
        width=width,
        top="q3",
        bottom="q2",
        fill_color=CATEGORY20[0],
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
            color=CATEGORY20[6],
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
    spl_sz: int,
    plot_width: int,
    plot_height: int,
) -> Any:
    """
    Render a scatter plot
    """
    # pylint: disable=too-many-arguments
    title = f"{y} by {x}" if len(df) < spl_sz else f"{y} by {x} (sample size {spl_sz})"
    tooltips = [("(x, y)", f"(@{{{x}}}, @{{{y}}})")]
    fig = figure(  # pylint: disable=too-many-function-args
        tools="hover",
        title=title,
        toolbar_location=None,
        tooltips=tooltips,
        plot_width=plot_width,
        plot_height=plot_height,
    )
    fig.circle(  # pylint: disable=too-many-function-args
        x, y, color=CATEGORY20[0], size=4, name="points", source=df
    )
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
    _format_axis(fig, xmin, xmax, "x")
    _format_axis(fig, ymin, ymax, "y")
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
    if miss_pct is not None:
        title = f"{x} ({miss_pct}% missing)" if miss_pct > 0 else f"{x}"
        tooltips = [(timeunit, "@lbl"), ("Frequency", "@freq"), ("Percent", "@pct%")]
        agg = "freq"
    else:
        title = title = f"{df.columns[1]} of {y} by {x}"
        agg = f"{df.columns[1]}"
        tooltips = [(timeunit, "@lbl"), (agg, f"@{df.columns[1]}")]
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
        fig.yaxis.axis_label = f"{df.columns[1]} of {y}"
        fig.xaxis.axis_label = x
        return Panel(child=fig, title="Line Chart")

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
        "Distinct Count": data["nuniq"],
        "Unique (%)": data["nuniq"] / data["npres"],
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
        "Distinct Count": stats["nuniq"],
        "Unique (%)": stats["nuniq"] / stats["npres"],
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


def render_distribution_grid(
    itmdt: Intermediate, yscale: str, plot_width: int, plot_height: int
) -> Dict[str, Any]:
    """
    Render plots and dataset stats from plot(df)
    """  # pylint: disable=too-many-locals
    figs: List[Figure] = list()
    nrows = itmdt["stats"]["nrows"]
    titles: List[str] = []
    for col, dtype, data in itmdt["data"]:
        if is_dtype(dtype, Nominal()):
            df, ttl_grps = data
            fig = bar_viz(
                df,
                ttl_grps,
                nrows,
                col,
                yscale,
                plot_width,
                plot_height,
                False,
            )
        elif is_dtype(dtype, Continuous()):
            fig = hist_viz(data, nrows, col, yscale, plot_width, plot_height, False)
            figs.append(fig)
        elif is_dtype(dtype, DateTime()):
            df, timeunit, miss_pct = data
            fig = dt_line_viz(df, col, timeunit, yscale, plot_width, plot_height, False, miss_pct)
        fig.frame_height = plot_height
        titles.append(fig.title.text)
        fig.title.text = ""
        figs.append(fig)

    return {
        "layout": figs,
        "meta": titles,
        "tabledata": format_ov_stats(itmdt["stats"]),
        "overview_insights": itmdt["overview_insights"],
        "column_insights": itmdt["column_insights"],
        "container_width": plot_width * 3,
    }


def render_cat(
    itmdt: Intermediate, yscale: str, plot_width: int, plot_height: int
) -> Dict[str, Any]:
    """
    Render plots from plot(df, x) when x is a categorical column
    """
    # pylint: disable=too-many-locals
    tabs: List[Panel] = []
    col, data = itmdt["col"], itmdt["data"]
    # overview, word length, and charater level statistcs
    stats, len_stats, letter_stats = (
        data["stats"],
        data["len_stats"],
        data["letter_stats"],
    )
    # number of present (not null) rows, and total rows
    nrows, nuniq = data["nrows"], data["nuniq"]
    # bar chart and pie chart of the categorical values
    bar_data, pie = data["bar"].to_frame(), data["pie"].to_frame()
    fig = bar_viz(
        bar_data,
        nuniq,
        nrows,
        col,
        yscale,
        plot_width,
        plot_height,
        True,
    )
    tabs.append(Panel(child=row(fig), title="Bar Chart"))
    tabs.append(pie_viz(pie, nrows, col, plot_width, plot_height))
    # word counts and total number of words for the wordcloud and word frequencies bar chart
    word_cnts, nwords, = (
        data["word_cnts"],
        data["nwords"],
    )
    if nwords > 0:
        tabs.append(wordcloud_viz(word_cnts, plot_width, plot_height))
        tabs.append(wordfreq_viz(word_cnts, nwords, plot_width, plot_height, True))
    # word length histogram
    length_dist = hist_viz(
        data["len_hist"], nrows, "Word Length", yscale, plot_width, plot_height, True
    )
    tabs.append(Panel(child=row(length_dist), title="Word Length"))

    # panel.child.children[0] is a figure
    for panel in tabs:
        panel.child.children[0].frame_width = int(plot_width * 0.9)

    return {
        "tabledata": format_cat_stats(stats, len_stats, letter_stats),
        "insights": nom_insights(data, col),
        "layout": [panel.child.children[0] for panel in tabs],
        "meta": [tab.title for tab in tabs],
        "container_width": plot_width + 110,
    }


def nom_insights(data: Dict[str, Any], col: str) -> Dict[str, List[str]]:
    """
    Format the insights for plot(df, Nominal())
    """
    # pylint: disable=line-too-long
    # insight dictionary, with a list associated with each plot
    ins: Dict[str, List[str]] = {
        "Stats": [],
        "Bar Chart": [],
        "Pie Chart": [],
        "Word Cloud": [],
        "Word Frequencies": [],
        "Word Length": [],
    }

    ## if cfg.insight.constant_enable:
    if data["nuniq"] == 1:
        ins["Stats"].append(f"{col} has a constant value")

    ## if cfg.insight.high_cardinality_enable:
    if data["nuniq"] > 50:  ## cfg.insght.high_cardinality_threshold
        nuniq = data["nuniq"]
        ins["Stats"].append(f"{col} has a high cardinality: {nuniq} distinct values")

    ## if cfg.insight.missing_enable:
    pmiss = round((data["nrows"] - data["stats"]["npres"]) / data["nrows"] * 100, 2)
    if pmiss > 1:  ## cfg.insight.missing_threshold
        nmiss = data["nrows"] - data["stats"]["npres"]
        ins["Stats"].append(f"{col} has {nmiss} ({pmiss}%) missing values")

    ## if cfg.insight.constant_length_enable:
    if data["stats"]["nuniq"] == data["stats"]["npres"]:
        ins["Stats"].append(f"{col} has all distinct values")

    ## if cfg.insight.evenness_enable:
    if data["chisq"][1] > 0.999:  ## cfg.insight.uniform_threshold
        ins["Bar Chart"].append(f"{col} is relatively evenly distributed")

    ## if cfg.insight.outstanding_no1_enable
    factor = data["bar"][0] / data["bar"][1] if len(data["bar"]) > 1 else 0
    if factor > 1.5:
        val1, val2 = data["bar"].index[0], data["bar"].index[1]
        ins["Bar Chart"].append(
            f"The largest value ({val1}) is over {factor} times larger than the second largest value ({val2})"
        )

    ## if cfg.insight.attribution_enable
    if data["pie"][:2].sum() / data["nrows"] > 0.5 and len(data["pie"]) >= 2:
        vals = ", ".join(str(data["pie"].index[i]) for i in range(2))
        ins["Pie Chart"].append(f"The top 2 categories ({vals}) take over 50%")

    ## if cfg.insight.high_word_cardinlaity_enable
    if data["nwords"] > 1000:
        nwords = data["nwords"]
        ins["Word Cloud"].append(f"{col} contains many words: {nwords} words")

    ## if cfg.insight.outstanding_no1_word_enable
    factor = data["word_cnts"][0] / data["word_cnts"][1] if len(data["word_cnts"]) > 1 else 0
    if factor > 1.5:
        val1, val2 = data["word_cnts"].index[0], data["word_cnts"].index[1]
        ins["Word Frequencies"].append(
            f"The largest value ({val1}) is over {factor} times larger than the second largest value ({val2})"
        )

    ## if cfg.insight.constant_word_length_enable
    if data["len_stats"]["Minimum"] == data["len_stats"]["Maximum"]:
        ins["Word Frequencies"].append(f"{col} has words of constant length")

    return ins


def render_num(
    itmdt: Intermediate, yscale: str, plot_width: int, plot_height: int
) -> Dict[str, Any]:
    # pylint: disable=too-many-locals
    """
    Render plots from plot(df, x) when x is a numerical column
    """
    col, data = itmdt["col"], itmdt["data"]

    tabs: List[Panel] = []
    fig = hist_viz(
        data["hist"],
        data["nrows"],
        col,
        yscale,
        plot_width,
        plot_height,
        True,
    )
    tabs.append(Panel(child=row(fig), title="Histogram"))
    # kde and q-q normal
    if data["kde"] is not None:
        dens, kde = data["dens"], data["kde"]
        tabs.append(kde_viz(dens, kde, col, yscale, plot_width, plot_height))
    if data["qntls"].any():
        qntls, mean, std = data["qntls"], data["mean"], data["std"]
        tabs.append(qqnorm_viz(qntls, mean, std, col, plot_width, plot_height))

    # box plot
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
    tabs.append(box_viz(df, col, plot_width, plot_height))

    # panel.child.children[0] is a figure
    for panel in tabs:
        panel.child.children[0].frame_width = int(plot_width * 0.9)

    return {
        "tabledata": format_num_stats(data),
        "insights": cont_insights(data, col),
        "layout": [panel.child for panel in tabs],
        "meta": [tab.title for tab in tabs],
        "container_width": plot_width + 110,
    }


def cont_insights(data: Dict[str, Any], col: str) -> Dict[str, List[str]]:
    """
    Format the insights for plot(df, Continuous())
    """
    # insight dictionary with a list associated with each plot
    ins: Dict[str, List[str]] = {
        "Stats": [],
        "Histogram": [],
        "KDE Plot": [],
        "Normal Q-Q Plot": [],
        "Box Plot": [],
    }

    ## if cfg.insight.infinity_enable:
    pinf = round((data["npres"] - data["nreals"]) / data["nrows"] * 100, 2)
    if pinf > 1:  ## cfg.insight.infinity_threshold
        ninf = data["npres"] - data["nreals"]
        ins["Stats"].append(f"{col} has {ninf} ({pinf}%) infinite values")

    ## if cfg.insight.missing_enable:
    pmiss = round((data["nrows"] - data["npres"]) / data["nrows"] * 100, 2)
    if pmiss > 1:  ## cfg.insight.missing_threshold
        nmiss = data["nrows"] - data["npres"]
        ins["Stats"].append(f"{col} has {nmiss} ({pmiss}%) missing values")

    ## if cfg.insight.negatives_enable:
    pneg = round(data["nneg"] / data["nrows"] * 100, 2)
    if pneg > 1:  ## cfg.insight.negatives_threshold
        nneg = data["nneg"]
        ins["Stats"].append(f"{col} has {nneg} ({pneg}%) negatives")

    ## if cfg.insight.zeros_enable:
    pzero = round(data["nzero"] / data["nrows"] * 100, 2)
    if pzero > 5:  ## cfg.insight.zeros_threshold
        nzero = data["nzero"]
        ins["Stats"].append(f"{col} has {nzero} ({pzero}%) zeros")

    ## if cfg.insight.normal_enable:
    if data["norm"][1] > 0.99:
        ins["Histogram"].append(f"{col} is normally distributed")

    ## if cfg.insight.uniform_enable:
    if data["chisq"][1] > 0.999:  ## cfg.insight.uniform_threshold
        ins["Histogram"].append(f"{col} is uniformly distributed")

    ## if cfg.insight.skewed_enable:
    skw = np.round(data["skew"], 4)
    if skw >= 20:  ## cfg.insight.skewed_threshold
        ins["Histogram"].append(f"{col} is skewed right (\u03B31 = {skw})")
    if skw <= -20:  ## cfg.insight.skewed_threshold
        ins["Histogram"].append(f"{col} is skewed left (\u03B31 = {skw})")

    ## if cfg.insight.normal_enable:
    if data["norm"][1] <= 0.01:
        pval = data["norm"][1]
        ins["Normal Q-Q Plot"].append(f"{col} is not normally distributed (p-value {pval})")

    ## if cfg.insight.box_enable
    if data["notlrs"] > 0:
        notlrs = data["notlrs"]
        ins["Box Plot"].append(f"{col} has {notlrs} outliers")

    return ins


def render_dt(
    itmdt: Intermediate, yscale: str, plot_width: int, plot_height: int
) -> Dict[str, Any]:
    """
    Render plots from plot(df, x) when x is a numerical column
    """
    tabs: List[Panel] = []
    df, timeunit, miss_pct = itmdt["line"]
    fig = dt_line_viz(df, itmdt["col"], timeunit, yscale, plot_width, plot_height, True, miss_pct)
    fig.frame_width = int(plot_width * 0.95)
    tabs.append(Panel(child=fig, title="Line Chart"))

    return {
        "tabledata": stats_viz_dt(itmdt["data"]),
        "insights": None,
        "layout": [panel.child for panel in tabs],
        "meta": [tab.title for tab in tabs],
        "container_width": plot_width + 50,
    }


def render_cat_num(
    itmdt: Intermediate,
    yscale: str,
    plot_width: int,
    plot_height: int,
) -> Dict[str, Any]:
    """
    Render plots from plot(df, x, y) when x is a categorical column
    and y is a numerical column
    """
    tabs: List[Panel] = []
    data, x, y = itmdt["data"], itmdt["x"], itmdt["y"]

    # box plot
    df = data["box"].to_frame().reset_index()[: 5 * itmdt["ngroups"]]
    df = df.pivot(index=itmdt["x"], columns="level_1", values=[0]).reset_index()
    df.columns = df.columns.get_level_values(1)
    df.columns = ["grp"] + list(df.columns[1:])
    tabs.append(box_viz(df, x, plot_width, plot_height, y, data["ttl_grps"]))

    # multiline plot
    df = data["hist"].to_frame()[: itmdt["ngroups"]]
    tabs.append(
        line_viz(
            df,
            x,
            y,
            yscale,
            plot_width,
            plot_height,
            data["ttl_grps"],
        )
    )
    for panel in tabs:
        panel.child.children[0].frame_width = int(plot_width * 0.9)
    return {
        "layout": [panel.child for panel in tabs],
        "meta": [panel.title for panel in tabs],
        "container_width": plot_width + 170,
    }


def render_two_num(
    itmdt: Intermediate,
    plot_width: int,
    plot_height: int,
    tile_size: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Render plots from plot(df, x, y) when x and y are numerical columns
    """
    tabs: List[Panel] = []
    data = itmdt["data"]
    # scatter plot
    tabs.append(
        scatter_viz(
            data["scat"],
            itmdt["x"],
            itmdt["y"],
            itmdt["spl_sz"],
            plot_width,
            plot_height,
        )
    )
    # hexbin plot
    tabs.append(
        hexbin_viz(
            data["hex"],
            itmdt["x"],
            itmdt["y"],
            plot_width,
            plot_height,
            tile_size,
        )
    )
    # box plot
    df = data["box"].to_frame().reset_index()
    df = df.pivot(index="grp", columns="level_1", values=[0]).reset_index()
    df.columns = df.columns.get_level_values(1)
    df.columns = ["grp"] + list(df.columns[1:])
    tabs.append(box_viz(df, itmdt["x"], plot_width, plot_height, itmdt["y"]))
    for panel in tabs:
        try:
            panel.child.frame_width = int(plot_width * 0.9)
        except AttributeError:
            panel.child.children[0].frame_width = int(plot_width * 0.9)
    return {
        "layout": [panel.child for panel in tabs],
        "meta": [panel.title for panel in tabs],
        "container_width": plot_width + 80,
    }


def render_two_cat(
    itmdt: Intermediate,
    plot_width: int,
    plot_height: int,
) -> Dict[str, Any]:
    """
    Render plots from plot(df, x, y) when x and y are categorical columns
    """
    tabs: List[Panel] = []
    df = itmdt["data"].to_frame("cnt").reset_index()
    x, y = itmdt["x"], itmdt["y"]

    # parse the dataframe to consist of the ngroups largest x groups
    xgrps = df.groupby(x)["cnt"].sum()
    x_lrgst = xgrps.nlargest(itmdt["ngroups"])
    df = df[df[x].isin(x_lrgst.index)]
    stats = {f"{x}_ttl": len(xgrps), f"{x}_shw": len(x_lrgst)}

    # parse the dataframe to consist of the nsubgroups largest y groups
    ygrps = df.groupby(y)["cnt"].sum()
    y_lrgst = ygrps.nlargest(itmdt["nsubgroups"])
    df = df[df[y].isin(y_lrgst.index)]
    stats.update(zip((f"{y}_ttl", f"{y}_shw"), (len(ygrps), len(y_lrgst))))
    df[[x, y]] = df[[x, y]].astype(str)

    # final format
    df = df.pivot_table(index=y, columns=x, values="cnt", fill_value=0, aggfunc="sum")
    df = df.unstack().to_frame("cnt").reset_index()

    # nested bar chart
    tabs.append(nested_viz(df, x, y, stats, plot_width, plot_height))
    # stacked bar chart
    # wrangle the dataframe into a pivot table format
    df2 = df.pivot(index=x, columns=y, values="cnt")
    df2.index.name = None
    # aggregate remaining groups into "Others"
    df2["Others"] = xgrps - df2.sum(axis=1)
    if df2["Others"].sum() < 1e-6:
        df2 = df2.drop(columns=["Others"])
    tabs.append(stacked_viz(df2, x, y, stats, plot_width, plot_height))
    # heat map
    tabs.append(heatmap_viz(df, x, y, stats, plot_width, plot_height))

    return {
        "layout": [panel.child for panel in tabs],
        "meta": [panel.title for panel in tabs],
        "container_width": plot_width,
    }


def render_dt_num(
    itmdt: Intermediate,
    yscale: str,
    plot_width: int,
    plot_height: int,
) -> Dict[str, Any]:
    """
    Render plots from plot(df, x, y) when x is dt and y is num
    """
    tabs: List[Panel] = []
    linedf, timeunit = itmdt["linedata"]
    tabs.append(
        dt_line_viz(
            linedf,
            itmdt["x"],
            timeunit,
            yscale,
            plot_width,
            plot_height,
            True,
            y=itmdt["y"],
        )
    )
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


def render_dt_cat(
    itmdt: Intermediate,
    yscale: str,
    plot_width: int,
    plot_height: int,
) -> Dict[str, Any]:
    """
    Render plots from plot(df, x, y) when x is dt and y is num
    """
    tabs: List[Panel] = []
    data, grp_cnt_stats, timeunit = itmdt["linedata"]
    tabs.append(
        dt_multiline_viz(
            data,
            itmdt["x"],
            itmdt["y"],
            timeunit,
            yscale,
            plot_width,
            plot_height,
            grp_cnt_stats,
        )
    )
    df, grp_cnt_stats, timeunit = itmdt["stackdata"]
    tabs.append(
        stacked_viz(df, itmdt["x"], itmdt["y"], grp_cnt_stats, plot_width, plot_height, timeunit)
    )
    return {
        "layout": [panel.child for panel in tabs],
        "meta": [panel.title for panel in tabs],
        "container_width": plot_width,
    }


def render_dt_num_cat(
    itmdt: Intermediate,
    yscale: str,
    plot_width: int,
    plot_height: int,
) -> Dict[str, Any]:
    """
    Render plots from plot(df, x, y) when x is dt and y is num
    """
    tabs: List[Panel] = []
    data, grp_cnt_stats, timeunit = itmdt["data"]
    tabs.append(
        dt_multiline_viz(
            data,
            itmdt["x"],
            itmdt["y"],
            timeunit,
            yscale,
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


def render(
    itmdt: Intermediate,
    yscale: str = "linear",
    tile_size: Optional[float] = None,
    plot_width_sml: int = 324,
    plot_height_sml: int = 300,
    plot_width_lrg: int = 450,
    plot_height_lrg: int = 400,
    plot_width_wide: int = 972,
) -> Union[LayoutDOM, Dict[str, Any]]:
    """
    Render a basic plot

    Parameters
    ----------
    itmdt
        The Intermediate containing results from the compute function.
    yscale: str, default "linear"
        The scale to show on the y axis. Can be "linear" or "log".
    tile_size
        Size of the tile for the hexbin plot. Measured from the middle
        of a hexagon to its left or right corner.
    plot_width_small: int, default 324
        The width of the small plots
    plot_height_small: int, default 300
        The height of the small plots
    plot_width_large: int, default 450
        The width of the large plots
    plot_height_large: int, default 400
        The height of the large plots
    plot_width_large: int, default 972
        The width of the large plots
    plot_width_wide: int, default 972
        The width of the wide plots
    """
    # pylint: disable=too-many-arguments
    if itmdt.visual_type == "distribution_grid":
        visual_elem = render_distribution_grid(itmdt, yscale, plot_width_sml, plot_height_sml)
    elif itmdt.visual_type == "categorical_column":
        visual_elem = render_cat(itmdt, yscale, plot_width_lrg, plot_height_lrg)
    elif itmdt.visual_type == "numerical_column":
        visual_elem = render_num(itmdt, yscale, plot_width_lrg, plot_height_lrg)
    elif itmdt.visual_type == "datetime_column":
        visual_elem = render_dt(itmdt, yscale, plot_width_lrg, plot_height_lrg)
    elif itmdt.visual_type == "cat_and_num_cols":
        visual_elem = render_cat_num(itmdt, yscale, plot_width_lrg, plot_height_lrg)
    elif itmdt.visual_type == "two_num_cols":
        visual_elem = render_two_num(itmdt, plot_width_lrg, plot_height_lrg, tile_size)
    elif itmdt.visual_type == "two_cat_cols":
        visual_elem = render_two_cat(itmdt, plot_width_wide, plot_height_sml)
    elif itmdt.visual_type == "dt_and_num_cols":
        visual_elem = render_dt_num(itmdt, yscale, plot_width_lrg, plot_height_lrg)
    elif itmdt.visual_type == "dt_and_cat_cols":
        visual_elem = render_dt_cat(itmdt, yscale, plot_width_wide, plot_height_lrg)
    elif itmdt.visual_type == "dt_cat_num_cols":
        visual_elem = render_dt_num_cat(itmdt, yscale, plot_width_wide, plot_height_lrg)

    return visual_elem
