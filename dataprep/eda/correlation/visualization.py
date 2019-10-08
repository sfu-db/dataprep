"""
    This module implements the visualization for
    plot_correlation(df) function
"""
import math
import sys
from typing import Any, Dict

import holoviews as hv
import numpy as np
from bokeh.models import HoverTool
from bokeh.models.annotations import Title
from bokeh.models.widgets import Panel, Tabs
from bokeh.plotting import Figure, figure

from ..common import Intermediate
from ..palette import BIPALETTE


def _discard_unused_visual_elems(fig: Figure) -> None:
    """
    :param fig: A figure object
    """
    fig.toolbar_location = None
    fig.toolbar.active_drag = None
    fig.xaxis.axis_label = ""
    fig.yaxis.axis_label = ""


def _vis_correlation_pd(  # pylint: disable=too-many-locals
    intermediate: Intermediate, params: Dict[str, Any]
) -> Tabs:
    """
    :param intermediate: An object to encapsulate the
    intermediate results.
    :return: A figure object
    """
    tab_list = []
    pd_data_frame = intermediate.raw_data["df"]
    method_list = intermediate.raw_data["method_list"]
    result = intermediate.result
    hv.extension("bokeh", logo=False)

    for method in method_list:
        corr_matrix = result["corr_" + method[0]]
        name_list = pd_data_frame.columns.values
        data = []
        for i, _ in enumerate(name_list):
            for j, _ in enumerate(name_list):
                if corr_matrix[i, j] != 0:
                    data.append((name_list[i], name_list[j], corr_matrix[i, j]))
                else:
                    data.append((name_list[i], name_list[j], np.nan))
        tooltips = [("name", "@x"), ("name", "@y"), ("correlation", "@z")]
        hover = HoverTool(tooltips=tooltips)
        heatmap = hv.HeatMap(data).redim.range(z=(-1, 1))
        heatmap.opts(
            tools=[hover],
            cmap=BIPALETTE,
            colorbar=True,
            width=params["width"],
            title="heatmap_" + method,
        )
        fig = hv.render(heatmap, backend="bokeh")
        fig.plot_width = params["plot_width"]
        fig.plot_height = params["plot_height"]
        title = Title()
        title.text = method + " correlation matrix"
        title.align = "center"
        fig.title = title
        fig.xaxis.major_label_orientation = math.pi / 4
        _discard_unused_visual_elems(fig)
        tab = Panel(child=fig, title=method)
        tab_list.append(tab)
    tabs = Tabs(tabs=tab_list)
    return tabs


def _vis_correlation_pd_x_k(  # pylint: disable=too-many-locals
    intermediate: Intermediate, params: Dict[str, Any]
) -> Tabs:
    """
    :param intermediate: An object to encapsulate the
    intermediate results.
    :return: A figure object
    """
    x_name = intermediate.raw_data["x_name"]
    result = intermediate.result
    hv.extension("bokeh", logo=False)
    data_p = []
    data_s = []
    data_k = []
    for i, _ in enumerate(result["col_p"]):
        data_p.append((x_name, result["col_p"][i], result["pearson"][i]))
    for i, _ in enumerate(result["col_s"]):
        data_s.append((x_name, result["col_s"][i], result["spearman"][i]))
    for i, _ in enumerate(result["col_k"]):
        data_k.append((x_name, result["col_k"][i], result["kendall"][i]))
    tooltips = [("name", "@x"), ("name", "@y"), ("correlation", "@z")]
    hover = HoverTool(tooltips=tooltips)
    if not data_p:
        print("Warning: The pearson correlation matrix is empty", file=sys.stderr)
    if not data_k:
        print("Warning: The kendall correlation matrix is empty", file=sys.stderr)
    if not data_s:
        print("Warning: The spearman correlation matrix is empty", file=sys.stderr)
    heatmap_p = hv.HeatMap(data_p).redim.range(z=(-1, 1))
    heatmap_p.opts(
        tools=[hover],
        cmap=BIPALETTE,
        colorbar=True,
        width=params["width"],
        toolbar="above",
    )
    heatmap_s = hv.HeatMap(data_s).redim.range(z=(-1, 1))
    heatmap_s.opts(
        tools=[hover],
        cmap=BIPALETTE,
        colorbar=True,
        width=params["width"],
        toolbar="above",
    )
    heatmap_k = hv.HeatMap(data_k).redim.range(z=(-1, 1))
    heatmap_k.opts(
        tools=[hover],
        cmap=BIPALETTE,
        colorbar=True,
        width=params["width"],
        toolbar="above",
    )
    fig_p = hv.render(heatmap_p, backend="bokeh")
    fig_s = hv.render(heatmap_s, backend="bokeh")
    fig_k = hv.render(heatmap_k, backend="bokeh")
    _discard_unused_visual_elems(fig_p)
    _discard_unused_visual_elems(fig_s)
    _discard_unused_visual_elems(fig_k)
    tab_p = Panel(child=fig_p, title="pearson")
    tab_s = Panel(child=fig_s, title="spearman")
    tab_k = Panel(child=fig_k, title="kendall")
    tabs = Tabs(tabs=[tab_p, tab_s, tab_k])
    return tabs


def _vis_correlation_pd_x_y_k(
    intermediate: Intermediate, params: Dict[str, Any]
) -> Figure:
    """
    :param intermediate: An object to encapsulate the
    intermediate results.
    :return: A figure object
    """
    data_x = intermediate.raw_data["df"][intermediate.raw_data["x_name"]].values
    result = intermediate.result
    data_x_sample = result["data_x_sample"]
    data_y_sample = result["data_y_sample"]
    tooltips = [("x", "@x"), ("y", "@y")]
    hover = HoverTool(tooltips=tooltips, names=["dec", "inc"])
    fig = figure(
        plot_width=params["plot_width"],
        plot_height=params["plot_height"],
        tools=[hover],
    )
    sample_x = np.linspace(min(data_x), max(data_x), 100)
    sample_y = result["line_a"] * sample_x + result["line_b"]
    fig.circle(
        x=data_x_sample,
        y=data_y_sample,
        size=params["size"],
        color="navy",
        alpha=params["alpha"],
        name="all",
    )
    if intermediate.raw_data["k"] is not None:
        for name, color in [("inc", "red"), ("dec", "black")]:
            if name == "inc":
                legend_name = "most influential (+)"
            else:
                legend_name = "most influential (-)"
            fig.circle(
                x=result[f"{name}_point_x"][0],
                y=result[f"{name}_point_y"][0],
                legend=legend_name,
                size=params["size"],
                color=color,
                alpha=params["alpha"],
                name=name,
            )
    fig.line(x=sample_x, y=sample_y, line_width=3)
    fig.toolbar_location = None
    fig.toolbar.active_drag = None
    title = Title()
    title.text = "scatter plot"
    title.align = "center"
    fig.title = title
    fig.xaxis.axis_label = intermediate.raw_data["x_name"]
    fig.yaxis.axis_label = intermediate.raw_data["y_name"]
    return fig


def _vis_cross_table(intermediate: Intermediate, params: Dict[str, Any]) -> Figure:
    """
    :param intermediate: An object to encapsulate the
    intermediate results.
    :return: A figure object
    """
    result = intermediate.result
    hv.extension("bokeh", logo=False)
    cross_matrix = result["cross_table"]
    x_cat_list = result["x_cat_list"]
    y_cat_list = result["y_cat_list"]
    data = []
    for i, _ in enumerate(x_cat_list):
        for j, _ in enumerate(y_cat_list):
            data.append((x_cat_list[i], y_cat_list[j], cross_matrix[i, j]))
    tooltips = [("z", "@z")]
    hover = HoverTool(tooltips=tooltips)
    heatmap = hv.HeatMap(data)
    heatmap.opts(
        tools=[hover],
        colorbar=True,
        width=params["width"],
        toolbar="above",
        title="cross_table",
    )
    fig = hv.render(heatmap, backend="bokeh")
    _discard_unused_visual_elems(fig)
    return fig
