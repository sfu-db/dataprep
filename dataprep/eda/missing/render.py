"""
    This module implements the plot_missing(df, x, y) function's
    visualization part.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from bokeh.layouts import row
from bokeh.models import (
    BasicTicker,
    CategoricalColorMapper,
    ColorBar,
    ColumnDataSource,
    CustomJSHover,
    FactorRange,
    HoverTool,
    LinearColorMapper,
    NumeralTickFormatter,
    Panel,
    PrintfTickFormatter,
    Range1d,
)
from bokeh.plotting import Figure

from ...errors import UnreachableError
from ..configs import Config
from ..dtypes_v2 import Continuous, Nominal, GeoGraphy, SmallCardNum, drop_null, DateTime
from ..intermediate import ColumnMetadata, Intermediate
from ..palette import CATEGORY10, CATEGORY20, GREYS256, RDBU
from ..utils import cut_long_name, fuse_missing_perc, relocate_legend
from .compute.common import LABELS

__all__ = ["render_missing"]


def render_missing(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Render the visualizations from plot_missing
    """

    if itmdt.visual_type == "missing_impact":
        return render_missing_impact(itmdt, cfg)
    elif itmdt.visual_type == "missing_impact_1vn":
        return render_missing_impact_1vn(itmdt, cfg)
    elif itmdt.visual_type == "missing_impact_1v1":
        return render_missing_impact_1v1(itmdt, cfg)
    else:
        raise UnreachableError


def tweak_figure(fig: Figure) -> Figure:
    """
    Set some common attributes for a figure
    """
    # fig.grid.grid_line_color = None
    # fig.axis.axis_line_color = None
    fig.axis.major_tick_line_color = None
    fig.axis.major_label_text_font_size = "9pt"
    fig.axis.major_label_standoff = 0
    fig.xaxis.major_label_orientation = np.pi / 3

    return fig


def render_dist(
    df: pd.DataFrame,
    x: str,
    typ: str,
    plot_width: int,
    plot_height: int,
) -> Figure:
    """
    Render a distribution, CDF or PDF
    """
    assert typ in ["pdf", "cdf"]
    tooltips = [
        (x, "@x"),
        (typ.upper(), f"@{{{typ}}}"),
        ("Label", f"@{{{typ}}}_label"),
    ]
    y_range = Range1d(0, df[typ].max() * 1.01)
    x_range = Range1d(0, df[f"x_{typ}"].max() * 1.01)

    fig = tweak_figure(
        Figure(
            x_range=x_range,
            y_range=y_range,
            plot_width=plot_width,
            plot_height=plot_height,
            title=" ",
            tools="hover",
            toolbar_location=None,
            tooltips=tooltips,
        )
    )
    for idx, label in enumerate(LABELS):
        group = df[df[f"{typ}_label"] == label]
        fig.line(
            x=f"x_{typ}",
            y=typ,
            source=group,
            color=CATEGORY10[idx],
            legend_label=label,
        )

    relocate_legend(fig, "left")

    return fig


def render_hist(
    df: pd.DataFrame,
    x: str,
    meta: ColumnMetadata,
    plot_width: int,
    plot_height: int,
    show_legend: bool,
) -> Figure:
    """
    Render a histogram
    """
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals

    if isinstance(meta["dtype"], (Nominal, GeoGraphy, SmallCardNum, DateTime)):
        tooltips = [
            (x, "@x"),
            ("Count", "@count"),
            ("Label", "@label"),
        ]
    elif isinstance(meta["dtype"], Continuous):
        df = df.copy()
        df["repr"] = [f"[{row.lower_bound:.0f}~{row.upper_bound:.0f})" for row in df.itertuples()]

        tooltips = [
            (x, "@repr"),
            ("Frequency", "@count"),
            ("Label", "@label"),
        ]
    else:
        mtype = type(meta["dtype"])
        raise ValueError(f"unprocessed data type:{mtype}, col:{x}")

    cols = [f"{col[:12]}..." if isinstance(col, str) and len(col) > 18 else col for col in df["x"]]
    df["x"] = cols
    cmapper = CategoricalColorMapper(palette=CATEGORY10, factors=LABELS)

    if isinstance(meta["dtype"], (Nominal, GeoGraphy, SmallCardNum, DateTime)):
        radius = 0.99

        # Inputs of FactorRange() have to be sequence of strings,
        # object only contains numbers can cause errors.(Issue#98).
        df["x"] = df["x"].astype("str")
        x_range = FactorRange(*df["x"].unique())
    elif isinstance(meta["dtype"], Continuous):

        radius = df["x"][1] - df["x"][0]
        x_range = Range1d(df["x"].min() - radius, df["x"].max() + radius)
    else:
        mtype = type(meta["dtype"])
        raise ValueError(f"unprocessed data type:{mtype}, col:{x}")

    y_range = Range1d(0, df["count"].max() * 1.05)

    fig = tweak_figure(
        Figure(
            x_range=x_range,
            y_range=y_range,
            plot_width=plot_width,
            plot_height=plot_height,
            title=" " if show_legend else "",
            tools="hover",
            toolbar_location=None,
            tooltips=tooltips,
        )
    )
    if show_legend:
        fig.vbar(
            x="x",
            width=radius,
            top="count",
            source=df,
            fill_alpha=0.3,
            color={"field": "label", "transform": cmapper},
            legend_field="label",
        )

        relocate_legend(fig, "left")
    else:
        shown, total = meta["shown"], meta["total"]
        if shown != total:
            fig.xaxis.axis_label = f"Top {shown} out of {total}"
            fig.xaxis.axis_label_standoff = 0
        fig.vbar(
            x="x",
            width=radius,
            top="count",
            source=df,
            fill_alpha=0.3,
            color={"field": "label", "transform": cmapper},
        )

    return fig


def render_boxwhisker(df: pd.DataFrame, plot_width: int, plot_height: int) -> Figure:
    """
    Render a box-whisker plot
    """

    tooltips = [
        ("Upper", "@upper"),
        ("75% Quantile", "@q1"),
        ("50% Quantile", "@q2"),
        ("25% Quantile", "@q3"),
        ("Lower", "@lower"),
    ]

    fig = tweak_figure(
        Figure(
            x_range=df["label"].unique(),
            plot_width=plot_width,
            plot_height=plot_height,
            title=" ",
            tools="",
            toolbar_location=None,
            tooltips=tooltips,
        )
    )

    # stems
    fig.segment(  # pylint: disable=too-many-function-args
        "label", "q3", "label", "upper", source=df, line_color="black"
    )
    fig.segment(  # pylint: disable=too-many-function-args
        "label", "q1", "label", "lower", source=df, line_color="black"
    )

    # boxes
    fig.vbar(  # pylint: disable=too-many-function-args
        "label",
        0.7,
        "q2",
        "q3",
        source=df,
        fill_color=CATEGORY20[0],
        line_color="black",
    )
    fig.vbar(  # pylint: disable=too-many-function-args
        "label",
        0.7,
        "q2",
        "q1",
        source=df,
        fill_color=CATEGORY20[0],
        line_color="black",
    )
    # whiskers (almost-0 height rects simpler than segments)
    fig.rect(  # pylint: disable=too-many-function-args
        "label", "lower", 0.2, 0.01, source=df, line_color="black"
    )
    fig.rect(  # pylint: disable=too-many-function-args
        "label", "upper", 0.2, 0.01, source=df, line_color="black"
    )

    # # outliers
    # if not out.empty:
    #     p.circle(outx, outy, size=6, color="#F38630", fill_alpha=0.6)
    return fig


def create_color_mapper() -> Tuple[LinearColorMapper, ColorBar]:
    """
    Create a color mapper and a colorbar for spectrum
    """
    mapper = LinearColorMapper(palette=list(reversed(GREYS256)), low=0, high=1)
    colorbar = ColorBar(
        color_mapper=mapper,
        major_label_text_font_size="8pt",
        ticker=BasicTicker(),
        formatter=NumeralTickFormatter(format="0 %"),
        label_standoff=10,
        border_line_color=None,
        location=(0, 0),
    )
    return mapper, colorbar


def create_color_mapper_heatmap(
    palette: Sequence[str],
) -> Tuple[LinearColorMapper, ColorBar]:
    """
    Create a color mapper and a colorbar for heatmap
    """
    mapper = LinearColorMapper(palette=palette, low=-1, high=1)
    colorbar = ColorBar(
        color_mapper=mapper,
        major_label_text_font_size="8pt",
        ticker=BasicTicker(),
        formatter=PrintfTickFormatter(format="%.2f"),
        label_standoff=6,
        border_line_color=None,
        location=(0, 0),
    )
    return mapper, colorbar


def render_missing_impact(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Render correlation heatmaps in to tabs
    """
    plot_width = cfg.plot.width if cfg.plot.width is not None else 500
    plot_height = cfg.plot.height if cfg.plot.height is not None else 500

    tabs: List[Panel] = []
    htgs: Dict[str, List[Tuple[str, str]]] = {}

    if cfg.bar.enable:
        fig_barchart = render_bar_chart(itmdt["data_bars"], "linear", plot_width, plot_height)
        tabs.append(Panel(child=row(fig_barchart), title="Bar Chart"))
        htgs["Bar Chart"] = cfg.bar.missing_how_to_guide(plot_height, plot_width)

    if cfg.spectrum.enable:
        fig_spectrum = render_missing_spectrum(
            itmdt["data_spectrum"], itmdt["data_total_missing"], plot_width, plot_height
        )
        tabs.append(Panel(child=row(fig_spectrum), title="Spectrum"))
        htgs["Spectrum"] = cfg.spectrum.how_to_guide(plot_height, plot_width)

    if cfg.heatmap.enable:
        fig_heatmap = render_heatmaps(itmdt["data_heatmap"], plot_width, plot_height)
        tabs.append(Panel(child=row(fig_heatmap), title="Heat Map"))
        htgs["Heat Map"] = cfg.heatmap.missing_how_to_guide(plot_height, plot_width)

    # itmdt["data_dendrogram"] is None when dataframe has only one column so that won't be displayed
    if cfg.dendro.enable and itmdt["ncols"] > 1:
        fig_dendrogram = render_dendrogram(itmdt["data_dendrogram"], plot_width, plot_height)
        tabs.append(Panel(child=row(fig_dendrogram), title="Dendrogram"))
        htgs["Dendrogram"] = cfg.dendro.how_to_guide(plot_height, plot_width)

    if cfg.stats.enable:
        stat_dict = {name: itmdt["missing_stat"][name] for name in itmdt["missing_stat"]}

    return {
        "insights": itmdt["insights"],
        "tabledata": {"Missing Statistics": stat_dict} if cfg.stats.enable else {},
        "layout": [panel.child.children[0] for panel in tabs],
        "meta": [panel.title for panel in tabs],
        "container_width": max([panel.child.children[0].plot_width for panel in tabs]),
        "how_to_guide": htgs,
    }


def render_heatmaps(data: Optional[pd.DataFrame], plot_width: int, plot_height: int) -> Figure:
    """
    Render missing heatmaps in to tabs
    """
    tooltips = [("x", "@x"), ("y", "@y"), ("correlation", "@correlation{1.11}")]
    mapper, color_bar = create_color_mapper_heatmap(RDBU)

    def empty_figure() -> Figure:
        # If no data to render in the heatmap, i.e. no missing values
        # we render a blank heatmap
        fig = Figure(
            x_range=[],
            y_range=[],
            plot_width=plot_width,
            plot_height=plot_height,
            x_axis_location="below",
            title=" ",
            tools="hover",
            toolbar_location=None,
            background_fill_color="#fafafa",
        )

        # Add at least one renderer to fig, otherwise bokeh
        # gives us error -1000 (MISSING_RENDERERS): Plot has no renderers
        fig.rect(x=0, y=0, width=0, height=0)
        return fig

    if data is not None:
        df = data.copy()
        df.columns = list(map(cut_long_name, df.columns))
        df.index = list(map(cut_long_name, df.index))
        df = df.where(np.triu(np.ones(df.shape)).astype(bool)).T  # pylint: disable=no-member

        if df.size != 0:
            x_range = FactorRange(*df.columns)
            y_range = FactorRange(*reversed(df.columns))

            df = df.unstack().reset_index(name="correlation")
            df = df.rename(columns={"level_0": "x", "level_1": "y"})
            df = df[df["x"] != df["y"]]
            df = drop_null(df)

            fig = Figure(
                x_range=x_range,
                y_range=y_range,
                plot_width=plot_width,
                plot_height=plot_height,
                x_axis_location="below",
                tools="hover",
                toolbar_location=None,
                tooltips=tooltips,
                background_fill_color="#fafafa",
                title=" ",
            )

            fig.rect(
                x="x",
                y="y",
                width=1,
                height=1,
                source=df,
                fill_color={"field": "correlation", "transform": mapper},
                line_color=None,
            )
        else:
            fig = empty_figure()
    else:
        fig = empty_figure()

    tweak_figure(fig)
    fig.grid.grid_line_color = None
    fig.axis.axis_line_color = None
    fig.add_layout(color_bar, "left")
    return fig


def render_bar_chart(
    data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    yscale: str,
    plot_width: int,
    plot_height: int,
) -> Figure:
    """
    Render a bar chart for the missing and present values
    """
    pres_cnts, null_cnts, cols = data
    cols = [f"{col[:13]}...{col[len(col)-3:]}" if len(col) > 18 else col for col in cols]
    df = pd.DataFrame({"Present": pres_cnts, "Missing": null_cnts}, index=cols)

    if len(df) > 20:
        plot_width = 28 * len(df)

    fig = Figure(
        x_range=list(df.index),
        y_range=[0, df["Present"][0] + df["Missing"][0]],
        plot_width=plot_width,
        plot_height=plot_height,
        y_axis_type=yscale,
        toolbar_location=None,
        tools=[],
        title=" ",
    )

    rend = fig.vbar_stack(
        stackers=df.columns,
        x="index",
        width=0.9,
        color=[CATEGORY20[0], CATEGORY20[2]],
        source=df,
        legend_label=list(df.columns),
    )
    # hover tool with count and percent
    formatter = CustomJSHover(
        args=dict(source=ColumnDataSource(df)),
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
    for i, val in enumerate(df.columns):
        hover = HoverTool(
            tooltips=[
                ("Column", "@index"),
                (f"{val} count", "@$name"),
                (f"{val} percent", "@{%s}{custom}" % rend[i].name),
            ],
            formatters={"@{%s}" % rend[i].name: formatter},
            renderers=[rend[i]],
        )
        fig.add_tools(hover)

    fig.yaxis.axis_label = "Row Count"
    tweak_figure(fig)
    relocate_legend(fig, "left")

    return fig


def render_missing_spectrum(
    data_spectrum: pd.DataFrame,
    data_total_missing: pd.DataFrame,
    plot_width: int,
    plot_height: int,
) -> Figure:
    """
    Render the missing specturm
    """
    mapper, color_bar = create_color_mapper()
    df = data_spectrum.copy()

    df["column_with_perc"] = df["column"].apply(
        lambda c: fuse_missing_perc(cut_long_name(c), data_total_missing[c])
    )

    radius = (df["loc_end"][0] - df["loc_start"][0]) / 2

    if (df["loc_end"] - df["loc_start"]).max() <= 1:
        loc_tooltip = "@loc_start{1}"
    else:
        loc_tooltip = "@loc_start{1}~@loc_end{1}"

    tooltips = [
        ("Column", "@column"),
        ("Loc", loc_tooltip),
        ("Missing%", "@missing_rate{1%}"),
    ]

    x_range = FactorRange(*df["column_with_perc"].unique())
    minimum, maximum = df["location"].min(), df["location"].max()
    y_range = Range1d(maximum + radius, minimum - radius)
    if df["column"].nunique() > 20:
        plot_width = 28 * df["column"].nunique()

    fig = tweak_figure(
        Figure(
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
    )
    fig.xgrid.grid_line_color = None
    fig.ygrid.grid_line_color = None

    fig.rect(
        x="column_with_perc",
        y="location",
        line_width=0,
        width=0.95,
        height=radius * 2,
        source=df,
        fill_color={"field": "missing_rate", "transform": mapper},
        line_color=None,
    )
    fig.add_layout(color_bar, "left")
    return fig


def render_dendrogram(dend: Dict["str", Any], plot_width: int, plot_height: int) -> Figure:
    """
    Render a missing dendrogram.
    """
    # list of lists of dcoords and icoords from scipy.dendrogram
    xs, ys, cols = dend["icoord"], dend["dcoord"], dend["ivl"]

    # if the number of columns is greater than 20, make the plot wider
    if len(cols) > 20:
        plot_width = 28 * len(cols)

    fig = Figure(
        plot_width=plot_width,
        plot_height=plot_height,
        toolbar_location=None,
        tools="",
        title=" ",
    )

    # round the coordinates to integers, and plot the dendrogram
    xs = [[round(coord) for coord in coords] for coords in xs]
    ys = [[round(coord, 2) for coord in coords] for coords in ys]
    fig.multi_line(xs=xs, ys=ys, line_color="#8073ac")

    # extract the horizontal lines for the hover tooltip
    h_lns_x = [coords[1:3] for coords in xs]
    h_lns_y = [coords[1:3] for coords in ys]
    null_mismatch_vals = [coord[0] for coord in h_lns_y]
    source = ColumnDataSource(dict(x=h_lns_x, y=h_lns_y, n=null_mismatch_vals))
    h_lns = fig.multi_line(xs="x", ys="y", source=source, line_color="#8073ac")
    hover_pts = HoverTool(
        renderers=[h_lns],
        tooltips=[("Average distance", "@n{0.1f}")],
        line_policy="interp",
    )
    fig.add_tools(hover_pts)

    # shorten column labels if necessary, and override coordinates with column names
    cols = [f"{col[:16]}..." if len(col) > 18 else col for col in cols]
    axis_coords = list(range(5, 10 * len(cols) + 1, 10))
    axis_overrides = dict(zip(axis_coords, cols))
    fig.xaxis.ticker = axis_coords
    fig.xaxis.major_label_overrides = axis_overrides
    fig.xaxis.major_label_orientation = np.pi / 3
    fig.yaxis.axis_label = "Average Distance Between Clusters"
    fig.grid.visible = False
    return fig


def render_missing_impact_1vn(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Render the plot from `plot_missing(df, "x")`
    """
    # pylint: disable = too-many-locals
    plot_width = cfg.plot.width if cfg.plot.width is not None else 300
    plot_height = cfg.plot.height if cfg.plot.height is not None else 300

    dfs = itmdt["data"]
    x = itmdt["x"]
    meta = itmdt["meta"]
    panels = []
    htgs: Dict[str, List[Tuple[str, str]]] = {}
    titles: List[str] = []
    for col, df in dfs.items():
        title = f"Missing impact of {x} by {col}"
        fig = render_hist(df, col, meta[col], plot_width, plot_height, False)
        fig.frame_height = plot_height
        panels.append(Panel(child=fig, title=col))

        if isinstance(meta[col]["dtype"], (Nominal, GeoGraphy, SmallCardNum, DateTime)):
            htgs[title] = cfg.bar.grid_how_to_guide()
        elif isinstance(meta[col]["dtype"], Continuous):
            htgs[title] = cfg.hist.grid_how_to_guide()
        else:
            mtype = type(meta[col]["dtype"])
            raise ValueError(f"unprocessed type:{mtype}")
        titles.append(title)
    legend_colors = [CATEGORY10[count] for count in range(len(LABELS))]
    return {
        "layout": [panel.child for panel in panels],
        "container_width": plot_width * 3,
        "legend_labels": [
            {"label": label, "color": color} for label, color in zip(LABELS, legend_colors)
        ],
        "meta": titles,
        "how_to_guide": htgs,
    }


def render_missing_impact_1v1(itmdt: Intermediate, cfg: Config) -> Dict[str, Any]:
    """
    Render the plot from `plot_missing(df, "x", "y")`
    """
    plot_width = cfg.plot.width if cfg.plot.width is not None else 400
    plot_height = cfg.plot.height if cfg.plot.height is not None else 400

    x, y, meta = itmdt["x"], itmdt["y"], itmdt["meta"]
    htgs: Dict[str, List[Tuple[str, str]]] = {}

    if isinstance(meta["dtype"], Continuous):
        panels = []

        if cfg.hist.enable:
            fig = render_hist(itmdt["hist"], y, meta, plot_width, plot_height, True)
            panels.append(Panel(child=fig, title="Histogram"))
            htgs["Histogram"] = cfg.hist.how_to_guide(plot_height, plot_width)
        if cfg.pdf.enable:
            fig = render_dist(itmdt["dist"], y, "pdf", plot_width, plot_height)
            panels.append(Panel(child=fig, title="PDF"))
            htgs["PDF"] = cfg.pdf.how_to_guide(plot_height, plot_width)
        if cfg.cdf.enable:
            fig = render_dist(itmdt["dist"], y, "cdf", plot_width, plot_height)
            panels.append(Panel(child=fig, title="CDF"))
            htgs["CDF"] = cfg.cdf.how_to_guide(plot_height, plot_width)
        if cfg.box.enable:
            fig = render_boxwhisker(itmdt["box"], plot_width, plot_height)
            panels.append(Panel(child=fig, title="Box Plot"))
            htgs["Box Plot"] = cfg.box.univar_how_to_guide(plot_height, plot_width)

        for panel in panels:
            panel.child.frame_width = plot_width

        return {
            "layout": [panel.child for panel in panels],
            "meta": [panel.title for panel in panels],
            "container_width": max([panel.child.plot_width for panel in panels]),
            "how_to_guide": htgs,
        }
    elif isinstance(meta["dtype"], (Nominal, SmallCardNum, GeoGraphy, DateTime)):
        fig = render_hist(itmdt["hist"], y, meta, plot_width, plot_height, True)
        shown, total = meta["shown"], meta["total"]
        if shown != total:
            _title = f"Missing impact of {x} by ({shown} out of {total}) {y}"
        else:
            _title = f"Missing impact of {x} by {y}"
        htgs[_title] = cfg.bar.how_to_guide(plot_height, plot_width)
        return {
            "layout": [fig],
            "meta": [_title],
            "container_width": fig.plot_width,
            "how_to_guide": htgs,
        }
    else:
        mtype = type(meta["dtype"])
        raise ValueError(f"unsupported type:{mtype}")
