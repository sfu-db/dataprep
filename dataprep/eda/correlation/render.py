"""
    This module implements the visualization for
    plot_correlation(df) function
"""
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from bokeh.models import (
    BasicTicker,
    CategoricalColorMapper,
    ColorBar,
    FactorRange,
    HoverTool,
    Legend,
    LegendItem,
    LinearColorMapper,
    PrintfTickFormatter,
)
from bokeh.models.annotations import Title
from bokeh.models.widgets import Panel, Tabs
from bokeh.plotting import Figure, figure

from ..intermediate import Intermediate
from ..palette import BIPALETTE, BRG

__all__ = ["render_correlation"]


def render_correlation(
    itmdt: Intermediate,
    plot_width: int = 500,
    plot_height: int = 500,
    palette: Optional[Sequence[str]] = None,
) -> Figure:
    """
    Render a correlation plot

    Parameters
    ----------
    itmdt
    plot_width
        The width of the plot
    plot_height
        The height of the plot
    palette
        The palette to use. By default (None),
        the palette will be automatically chosen based on different visualization types.

    Returns
    -------
    Figure
        The bokeh Figure instance.
    """
    if itmdt.visual_type is None:
        visual_elem = Figure()
    elif itmdt.visual_type == "correlation_heatmaps":
        visual_elem = render_correlation_heatmaps(
            itmdt, plot_width, plot_height, palette or BIPALETTE
        )
    elif itmdt.visual_type == "correlation_single_heatmaps":
        visual_elem = render_correlation_single_heatmaps(
            itmdt, plot_width, plot_height, palette or BIPALETTE
        )
    elif itmdt.visual_type == "correlation_scatter":
        visual_elem = render_scatter(itmdt, plot_width, plot_height, palette or BRG)
    else:
        raise NotImplementedError(f"Unknown visual type {itmdt.visual_type}")

    return visual_elem


# def _vis_cross_table(intermediate: Intermediate, params: Dict[str, Any]) -> Figure:
#     """
#     :param intermediate: An object to encapsulate the
#     intermediate results.
#     :return: A figure object
#     """
#     result = intermediate.result
#     hv.extension("bokeh", logo=False)
#     cross_matrix = result["cross_table"]
#     x_cat_list = result["x_cat_list"]
#     y_cat_list = result["y_cat_list"]
#     data = []
#     for i, _ in enumerate(x_cat_list):
#         for j, _ in enumerate(y_cat_list):
#             data.append((x_cat_list[i], y_cat_list[j], cross_matrix[i, j]))
#     tooltips = [("z", "@z")]
#     hover = HoverTool(tooltips=tooltips)
#     heatmap = hv.HeatMap(data)
#     heatmap.opts(
#         tools=[hover],
#         colorbar=True,
#         width=params["width"],
#         toolbar="above",
#         title="cross_table",
#     )
#     fig = hv.render(heatmap, backend="bokeh")
#     _discard_unused_visual_elems(fig)
#     return fig

########## HeatMaps ##########
def tweak_figure(fig: Figure) -> None:
    """
    Set some common attributes for a figure
    """
    fig.grid.grid_line_color = None
    fig.axis.axis_line_color = None
    fig.axis.major_tick_line_color = None
    fig.axis.major_label_text_font_size = "9pt"
    fig.axis.major_label_standoff = 0
    fig.xaxis.major_label_orientation = math.pi / 3


def render_correlation_heatmaps(
    itmdt: Intermediate, plot_width: int, plot_height: int, palette: Sequence[str]
) -> Tabs:
    """
    Render correlation heatmaps in to tabs
    """
    tabs: List[Panel] = []
    tooltips = [("x", "@x"), ("y", "@y"), ("correlation", "@correlation{1.11}")]
    axis_range = itmdt["axis_range"]

    for method, df in itmdt["data"].items():
        # in case of numerical column names
        df = df.copy()
        df["x"] = df["x"].apply(str)
        df["y"] = df["y"].apply(str)

        mapper, color_bar = create_color_mapper(palette)
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

        tab = Panel(child=fig, title=method)
        tabs.append(tab)

    tabs = Tabs(tabs=tabs)
    return tabs


def render_correlation_single_heatmaps(
    itmdt: Intermediate, plot_width: int, plot_height: int, palette: Sequence[str]
) -> Tabs:
    """
    Render correlation heatmaps, but with single column
    """
    tabs: List[Panel] = []
    tooltips = [("y", "@y"), ("correlation", "@correlation{1.11}")]

    for method, df in itmdt["data"].items():
        mapper, color_bar = create_color_mapper(palette)

        x_range = FactorRange(*df["x"].unique())
        y_range = FactorRange(*df["y"].unique())
        fig = figure(
            x_range=x_range,
            y_range=y_range,
            plot_width=plot_width,
            plot_height=plot_height,
            x_axis_location="below",
            tools="hover",
            toolbar_location=None,
            tooltips=tooltips,
        )

        tweak_figure(fig)

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

        tab = Panel(child=fig, title=method)
        tabs.append(tab)

    tabs = Tabs(tabs=tabs)
    return tabs


def create_color_mapper(palette: Sequence[str]) -> Tuple[LinearColorMapper, ColorBar]:
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


######### Scatter #########
def render_scatter(
    itmdt: Intermediate, plot_width: int, plot_height: int, palette: Sequence[str]
) -> Figure:
    """
    Render scatter plot with a regression line and possible most influencial points
    """

    # pylint: disable=too-many-locals

    df = itmdt["data"]
    xcol, ycol, *maybe_label = df.columns

    tooltips = [(xcol, f"@{{{xcol}}}"), (ycol, f"@{{{ycol}}}")]

    fig = Figure(
        plot_width=plot_width,
        plot_height=plot_height,
        toolbar_location=None,
        title=Title(text="Scatter Plot & Regression", align="center"),
        tools=[],
        x_axis_label=xcol,
        y_axis_label=ycol,
    )

    # Scatter
    scatter = fig.scatter(x=df.columns[0], y=df.columns[1], source=df)
    if maybe_label:
        assert len(maybe_label) == 1
        mapper = CategoricalColorMapper(factors=["=", "+", "-"], palette=palette)
        scatter.glyph.fill_color = {"field": maybe_label[0], "transform": mapper}
        scatter.glyph.line_color = {"field": maybe_label[0], "transform": mapper}

    # Regression line
    coeff_a, coeff_b = itmdt["coeffs"]
    line_x = np.asarray([df.iloc[:, 0].min(), df.iloc[:, 0].max()])
    line_y = coeff_a * line_x + coeff_b
    fig.line(x=line_x, y=line_y, line_width=3)

    # Not adding the tooltips before because we only want to apply tooltip to the scatter
    hover = HoverTool(tooltips=tooltips, renderers=[scatter])
    fig.add_tools(hover)

    # Add legends
    if maybe_label:
        nidx = df.index[df[maybe_label[0]] == "-"][0]
        pidx = df.index[df[maybe_label[0]] == "+"][0]

        legend = Legend(
            items=[
                LegendItem(
                    label="Most Influential (-)", renderers=[scatter], index=nidx
                ),
                LegendItem(
                    label="Most Influential (+)", renderers=[scatter], index=pidx
                ),
            ],
            margin=0,
            padding=0,
        )

        fig.add_layout(legend, place="right")
    return fig
