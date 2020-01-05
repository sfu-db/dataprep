"""
    This module implements the plot_missing(df, x, y) function's
    visualization part.
"""
from typing import Tuple, Union

import math
import pandas as pd
from bokeh.models import (
    Panel,
    Tabs,
    FactorRange,
    Range1d,
    LinearColorMapper,
    ColorBar,
    BasicTicker,
    NumeralTickFormatter,
    CategoricalColorMapper,
    LayoutDOM,
    Title,
)

# pylint: disable=no-name-in-module
from bokeh.palettes import Greys256, Category10  # type: ignore

# pylint: enable=no-name-in-module

from bokeh.plotting import Figure

from ...errors import UnreachableError
from ..dtypes import is_categorical
from ..intermediate import Intermediate
from ..utils import relocate_legend, cut_long_name, fuse_missing_perc

__all__ = ["render_missing"]


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
    df: pd.DataFrame, typ: str, plot_width: int, plot_height: int
) -> Figure:
    """
    Render a distribution, CDF or PDF
    """
    assert typ in ["pdf", "cdf"]
    tooltips = [
        ("x", "@x"),
        (typ, f"@{typ}"),
        ("label", "@label"),
    ]

    fig = tweak_figure(
        Figure(
            y_range=Range1d(0, df[typ].max() * 1.05),
            plot_width=plot_width,
            plot_height=plot_height,
            tools="hover",
            toolbar_location=None,
            tooltips=tooltips,
        )
    )

    for (label, group), color in zip(df.groupby("label"), Category10[3]):
        fig.line(x="x", y=typ, source=group, color=color, legend_label=label)

    relocate_legend(fig, "right")

    return fig


def render_hist(df: pd.DataFrame, plot_width: int, plot_height: int) -> Figure:
    """
    Render a histogram
    """
    tooltips = [
        ("x", "@x"),
        ("count", "@count"),
        ("label", "@label"),
    ]

    cmapper = CategoricalColorMapper(
        palette=Category10[3], factors=df["label"].unique()
    )

    if is_categorical(df["x"].dtype):
        radius = 0.99
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
    fig.segment("label", "q3", "label", "upper", source=df, line_color="black")
    fig.segment("label", "q1", "label", "lower", source=df, line_color="black")

    # boxes
    fig.vbar(
        "label", 0.7, "q2", "q3", source=df, fill_color="#E08E79", line_color="black",
    )
    fig.vbar(
        "label", 0.7, "q2", "q1", source=df, fill_color="#3B8686", line_color="black",
    )
    # whiskers (almost-0 height rects simpler than segments)
    fig.rect("label", "lower", 0.2, 0.01, source=df, line_color="black")
    fig.rect("label", "upper", 0.2, 0.01, source=df, line_color="black")

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


def render_missing_spectrum(
    itmdt: Intermediate, plot_width: int, plot_height: int
) -> Figure:
    """
    Render the missing specturm
    """
    mapper, color_bar = create_color_mapper()

    df = itmdt["data"].copy()

    df["column"] = df["column"].apply(
        lambda c: fuse_missing_perc(cut_long_name(c), itmdt["missing_percent"][c])
    )

    radius = df["loc_end"][0] - df["loc_start"][0]

    tooltips = [
        ("column", "@column"),
        ("loc", "@loc_start{1}~@loc_end{1}"),
        ("missing%", "@missing_rate{1%}"),
    ]

    x_range = FactorRange(*df["column"].unique())
    minimum, maximum = df["location"].min(), df["location"].max()
    y_range = Range1d(maximum, minimum)

    fig = tweak_figure(
        Figure(
            x_range=x_range,
            y_range=y_range,
            plot_width=plot_width,
            plot_height=plot_height,
            x_axis_location="above",
            tools="hover",
            toolbar_location=None,
            tooltips=tooltips,
        )
    )

    fig.rect(
        x="column",
        y="location",
        line_width=0,
        width=0.95,
        height=radius,
        source=df,
        fill_color={"field": "missing_rate", "transform": mapper},
        line_color=None,
    )

    fig.add_layout(color_bar, "right")

    return fig


def render_missing_impact_1vn(
    itmdt: Intermediate, plot_width: int, plot_height: int
) -> Tabs:
    """
    Render the plot from `plot_missing(df, "x")`
    """

    dfs = itmdt["data"]
    x = itmdt["x"]

    panels = []
    for col, df in dfs.items():
        fig = render_hist(df, plot_width, plot_height)
        shown, total = itmdt["partial"][col]

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
    itmdt: Intermediate, plot_width: int, plot_height: int, numerical: bool = True
) -> Union[Tabs, Figure]:
    """
    Render the plot from `plot_missing(df, "x", "y")`
    """
    x, y = itmdt["x"], itmdt["y"]

    if numerical:
        panels = []

        fig = render_hist(itmdt["hist"], plot_width, plot_height)
        panels.append(Panel(child=fig, title="Histogram"))

        fig = render_dist(itmdt["dist"], "pdf", plot_width, plot_height)
        panels.append(Panel(child=fig, title="PDF"))

        fig = render_dist(itmdt["dist"], "cdf", plot_width, plot_height)
        panels.append(Panel(child=fig, title="CDF"))

        fig = render_boxwhisker(itmdt["box"], plot_width, plot_height)
        panels.append(Panel(child=fig, title="Box"))

        tabs = Tabs(tabs=panels)
        return tabs
    else:
        fig = render_hist(itmdt["hist"], plot_width, plot_height)

        shown, total = itmdt["partial"]
        if shown != total:
            fig.title = Title(
                text=f"Missing impact of {x} by ({shown} out of {total}) {y}"
            )
        else:
            fig.title = Title(text=f"Missing impact of {x} by {y}")
        return fig


def render_missing(
    itmdt: Intermediate, plot_width: int = 500, plot_height: int = 500
) -> LayoutDOM:
    """
    @Jinglin write here
    """
    if itmdt.visual_type == "missing_spectrum":
        return render_missing_spectrum(itmdt, plot_width, plot_height)
    elif itmdt.visual_type == "missing_impact_1vn":
        return render_missing_impact_1vn(itmdt, plot_width, plot_height)
    elif itmdt.visual_type == "missing_impact_1v1_numerical":
        return render_missing_impact_1v1(itmdt, plot_width, plot_height)
    elif itmdt.visual_type == "missing_impact_1v1_categorical":
        return render_missing_impact_1v1(
            itmdt, plot_width, plot_height, numerical=False
        )
    else:
        raise UnreachableError
