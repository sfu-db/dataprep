"""
    This module implements the plot_missing(df, x, y) function's
    visualization part.
"""
import math
from typing import Tuple, Union
from typing import List, Optional, Sequence
import numpy as np

import pandas as pd
from bokeh.models import (
    BasicTicker,
    CategoricalColorMapper,
    ColorBar,
    FactorRange,
    LayoutDOM,
    LinearColorMapper,
    NumeralTickFormatter,
    Panel,
    Range1d,
    Tabs,
    Title,
    PrintfTickFormatter,
    ColumnDataSource,
)

# pylint: disable=no-name-in-module
from bokeh.palettes import Category10, Greys256  # type: ignore
from bokeh.plotting import Figure

from ...errors import UnreachableError
from ..dtypes import is_dtype, Nominal, Continuous
from ..intermediate import Intermediate, ColumnMetadata
from ..utils import cut_long_name, fuse_missing_perc, relocate_legend
from .compute import LABELS
from ..palette import PALETTE
from ..palette import BIPALETTE


# pylint: enable=no-name-in-module


__all__ = ["render_missing"]


def render_missing(
    itmdt: Intermediate,
    plot_width: int = 500,
    plot_height: int = 500,
    palette: Optional[Sequence[str]] = None,
) -> LayoutDOM:
    """
    @Jinglin write here
    """
    if itmdt.visual_type == "missing_spectrum":
        return render_missing_spectrum(itmdt, plot_width, plot_height)
    elif itmdt.visual_type == "missing_impact_1vn":
        return render_missing_impact_1vn(itmdt, plot_width, plot_height)
    elif itmdt.visual_type == "missing_impact_1v1":
        return render_missing_impact_1v1(itmdt, plot_width, plot_height)
    elif itmdt.visual_type == "missing_spectrum_heatmap":
        return render_missing_heatmap(
            itmdt, plot_width, plot_height, palette or BIPALETTE
        )

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
    fig.xaxis.major_label_orientation = math.pi / 3

    return fig


def render_dist(
    df: pd.DataFrame, x: str, typ: str, plot_width: int, plot_height: int,
) -> Figure:
    """
    Render a distribution, CDF or PDF
    """
    assert typ in ["pdf", "cdf"]
    tooltips = [
        (x, "@x"),
        (typ.upper(), f"@{{{typ}}}"),
        ("Label", "@label"),
    ]
    y_range = Range1d(0, df[typ].max() * 1.01)
    x_range = Range1d(0, df["x"].max() * 1.01)

    fig = tweak_figure(
        Figure(
            x_range=x_range,
            y_range=y_range,
            plot_width=plot_width,
            plot_height=plot_height,
            tools="hover",
            toolbar_location=None,
            tooltips=tooltips,
        )
    )
    for idx, label in enumerate(LABELS):
        group = df[df["label"] == label]
        fig.line(
            x="x", y=typ, source=group, color=Category10[3][idx], legend_label=label,
        )

    relocate_legend(fig, "right")

    return fig


def render_hist(
    df: pd.DataFrame, x: str, meta: ColumnMetadata, plot_width: int, plot_height: int,
) -> Figure:
    """
    Render a histogram
    """
    if is_dtype(meta["dtype"], Nominal()):
        tooltips = [
            (x, "@x"),
            ("Count", "@count"),
            ("Label", "@label"),
        ]
    else:
        df = df.copy()
        df["repr"] = [
            f"[{row.lower_bound:.0f}~{row.upper_bound:.0f})" for row in df.itertuples()
        ]

        tooltips = [
            (x, "@repr"),
            ("Frequency", "@count"),
            ("Label", "@label"),
        ]

    cmapper = CategoricalColorMapper(palette=Category10[3], factors=LABELS)

    if is_dtype(meta["dtype"], Nominal()):
        radius = 0.99

        # Inputs of FactorRange() have to be sequence of strings,
        # object only contains numbers can cause errors.(Issue#98).
        df["x"] = df["x"].astype("str")
        x_range = FactorRange(*df["x"].unique())
    else:
        radius = df["x"][1] - df["x"][0]
        x_range = Range1d(df["x"].min() - radius, df["x"].max() + radius)

    y_range = Range1d(0, df["count"].max() * 1.05)

    fig = tweak_figure(
        Figure(
            x_range=x_range,
            y_range=y_range,
            plot_width=plot_width,
            plot_height=plot_height,
            tools="hover",
            toolbar_location=None,
            tooltips=tooltips,
        )
    )

    fig.vbar(
        x="x",
        width=radius,
        top="count",
        source=df,
        fill_alpha=0.3,
        color={"field": "label", "transform": cmapper},
        legend_field="label",
    )

    relocate_legend(fig, "right")

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
        "label", 0.7, "q2", "q3", source=df, fill_color=PALETTE[0], line_color="black",
    )
    fig.vbar(  # pylint: disable=too-many-function-args
        "label", 0.7, "q2", "q1", source=df, fill_color=PALETTE[0], line_color="black",
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
    mapper = LinearColorMapper(palette=list(reversed(Greys256)), low=0, high=1)
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


def render_missing_heatmap(
    itmdt: Intermediate, plot_width: int, plot_height: int, palette: Sequence[str]
) -> Tabs:
    """
    Render correlation heatmaps in to tabs
    """
    tabs: List[Panel] = []
    fig_barchart = render_bar_viz(itmdt, "linear", plot_width, plot_height, False)
    pan_barchart = Panel(child=fig_barchart, title="Barchart")
    tabs.append(pan_barchart)
    fig_spectrum = render_missing_spectrum(itmdt, plot_width, plot_height)
    pan_spectrum = Panel(child=fig_spectrum, title="Spectrum")
    tabs.append(pan_spectrum)
    fig_heatmap = render_heatmaps_tab(itmdt, plot_width, plot_height, palette)
    pan_heatmap = Panel(child=fig_heatmap, title="Heatmap")
    tabs.append(pan_heatmap)

    tabs = Tabs(tabs=tabs)
    return tabs


def render_heatmaps_tab(
    itmdt: Intermediate, plot_width: int, plot_height: int, palette: Sequence[str]
) -> Figure:
    """
    Render missing heatmaps in to tabs
    """
    tooltips = [("x", "@x"), ("y", "@y"), ("correlation", "@correlation{1.11}")]
    axis_range = itmdt["axis_range"]
    df = itmdt["data_heatmap"]
    df = df.where(
        np.triu(np.ones(df.shape)).astype(np.bool)  # pylint: disable=no-member
    ).T
    df = df.unstack().reset_index(name="correlation")
    df = df.rename(columns={"level_0": "x", "level_1": "y"})
    df = df[df["x"] != df["y"]]
    df = df.dropna()
    # in case of numerical column names
    df["x"] = df["x"].apply(str)
    df["y"] = df["y"].apply(str)
    mapper, color_bar = create_color_mapper_heatmap(palette)
    x_range = FactorRange(*axis_range)
    y_range = FactorRange(*reversed(axis_range))
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
    )

    tweak_figure(fig)
    fig.grid.grid_line_color = None
    fig.axis.axis_line_color = None
    fig.rect(
        x="x",
        y="y",
        width=1,
        height=1,
        source=df,
        fill_color={"field": "correlation", "transform": mapper},
        line_color=None,
    )

    fig.add_layout(color_bar, "right")

    return fig


def render_bar_viz(
    itmdt: Intermediate,
    yscale: str,
    plot_width: int,
    plot_height: int,
    show_yticks: bool,
) -> Figure:
    """
    Render a bar chart
    """
    # pylint: disable=too-many-arguments
    length = itmdt["len_data"]
    df = itmdt["data_barchart"]
    df = df.copy()
    df = df.reset_index()
    df.columns = ["col", "values"]
    df["val_per"] = df["values"] * length
    df["missing"] = length - df["val_per"]
    stack_present = df["val_per"].tolist()
    stack_missing = df["missing"].tolist()
    source = ColumnDataSource(df)
    features = source.data["col"].tolist()

    colors = [PALETTE[0], PALETTE[2]]
    value_type = ["present", "missing"]
    data = {"features": features, "present": stack_present, "missing": stack_missing}

    if show_yticks:
        if len(df) > 10:
            plot_width = 28 * len(df)

    fig = Figure(
        x_range=features,
        plot_width=plot_width,
        plot_height=plot_height,
        y_axis_type=yscale,
        toolbar_location=None,
        tooltips="$name @features: @$name",
        tools="hover",
    )

    fig.vbar_stack(
        value_type,
        x="features",
        width=0.9,
        color=colors,
        source=data,
        legend_label=value_type,
    )

    fig.legend.location = "top_right"
    fig.y_range.start = 0
    fig.x_range.range_padding = 0.1
    tweak_figure(fig)
    fig.yaxis.axis_label = "Total Count"
    relocate_legend(fig, "right")
    return fig


def render_missing_spectrum(
    itmdt: Intermediate, plot_width: int, plot_height: int
) -> Figure:
    """
    Render the missing specturm
    """
    mapper, color_bar = create_color_mapper()
    df = itmdt["data"].copy()
    df["column_with_perc"] = df["column"].apply(
        lambda c: fuse_missing_perc(cut_long_name(c), itmdt["missing_percent"][c])
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
    fig.add_layout(color_bar, "right")

    return fig


def render_missing_impact_1vn(
    itmdt: Intermediate, plot_width: int, plot_height: int,
) -> Tabs:
    """
    Render the plot from `plot_missing(df, "x")`
    """

    dfs = itmdt["data"]
    x = itmdt["x"]
    meta = itmdt["meta"]

    panels = []
    for col, df in dfs.items():
        fig = render_hist(df, col, meta[col], plot_width, plot_height)
        shown, total = meta[col]["partial"]

        if shown != total:
            fig.title = Title(
                text=f"Missing impact of {x} by ({shown} out of {total}) {col}"
            )
        else:
            fig.title = Title(text=f"Missing impact of {x} by {col}")
        panels.append(Panel(child=fig, title=col))

    tabs = Tabs(tabs=panels)
    return tabs


def render_missing_impact_1v1(
    itmdt: Intermediate, plot_width: int, plot_height: int,
) -> Union[Tabs, Figure]:
    """
    Render the plot from `plot_missing(df, "x", "y")`
    """
    x, y = itmdt["x"], itmdt["y"]
    meta = itmdt["meta"]

    if is_dtype(meta["dtype"], Continuous()):
        panels = []

        fig = render_hist(itmdt["hist"], y, meta, plot_width, plot_height)
        panels.append(Panel(child=fig, title="Histogram"))

        fig = render_dist(itmdt["dist"], y, "pdf", plot_width, plot_height)
        panels.append(Panel(child=fig, title="PDF"))

        fig = render_dist(itmdt["dist"], y, "cdf", plot_width, plot_height)
        panels.append(Panel(child=fig, title="CDF"))

        fig = render_boxwhisker(itmdt["box"], plot_width, plot_height)
        panels.append(Panel(child=fig, title="Box"))

        tabs = Tabs(tabs=panels)
        return tabs
    else:
        fig = render_hist(itmdt["hist"], y, meta, plot_width, plot_height)

        shown, total = meta["partial"]
        if shown != total:
            fig.title = Title(
                text=f"Missing impact of {x} by ({shown} out of {total}) {y}"
            )
        else:
            fig.title = Title(text=f"Missing impact of {x} by {y}")
        return fig
