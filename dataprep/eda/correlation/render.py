"""
    This module implements the visualization for
    plot_correlation(df) function
"""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (
    BasicTicker,
    CategoricalColorMapper,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    FactorRange,
    FuncTickFormatter,
    HoverTool,
    Legend,
    LegendItem,
    LinearColorMapper,
    PrintfTickFormatter,
    Select,
)
from bokeh.models.annotations import Title
from bokeh.models.widgets import Panel, Tabs
from bokeh.plotting import Figure, figure

from ..configs import Config
from ..intermediate import Intermediate
from ..palette import BRG, RDBU

__all__ = ["render_correlation"]


def render_correlation(itmdt: Intermediate, cfg: Config) -> Any:
    """
    Render a correlation plot

    Parameters
    ----------
    itmdt
        Intermediate computations
    cfg
        Config instance
    """
    plot_width = cfg.plot.width if cfg.plot.width is not None else 400
    plot_height = cfg.plot.height if cfg.plot.height is not None else 400

    if itmdt.visual_type is None:
        visual_elem = Figure()
    elif itmdt.visual_type == "correlation_impact":
        visual_elem = render_correlation_impact(itmdt, plot_width, plot_height, cfg)
    elif itmdt.visual_type == "correlation_heatmaps":
        visual_elem = render_correlation_heatmaps(itmdt, plot_width, plot_height)
    elif itmdt.visual_type == "correlation_single_heatmaps":
        visual_elem = render_correlation_single_heatmaps(itmdt, plot_width, plot_height, cfg)
    elif itmdt.visual_type == "correlation_scatter":
        visual_elem = render_scatter(itmdt, plot_width, plot_height, cfg)
    elif itmdt.visual_type == "correlation_crossfilter":
        visual_elem = render_crossfilter(itmdt, plot_width, plot_height, cfg)
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
    fig.xaxis.major_label_orientation = np.pi / 3
    # truncate axis tick values
    format_js = """
        if (tick.length > 15) return tick.substring(0, 13) + '...';
        else return tick;
    """
    fig.xaxis.formatter = FuncTickFormatter(code=format_js)
    fig.yaxis.formatter = FuncTickFormatter(code=format_js)


def render_correlation_impact(
    itmdt: Intermediate, plot_width: int, plot_height: int, cfg: Config
) -> Dict[str, Any]:
    """
    Render correlation heatmaps in to tabs
    """
    tabs: List[Panel] = []
    tooltips = [("x", "@x"), ("y", "@y"), ("correlation", "@correlation{1.11}")]

    for method, df in itmdt["data"].items():
        # in case of numerical column names
        df = df.copy()
        df["x"] = df["x"].apply(str)
        df["y"] = df["y"].apply(str)

        mapper, color_bar = create_color_mapper(RDBU)
        x_range = FactorRange(*itmdt["axis_range"])
        y_range = FactorRange(*reversed(itmdt["axis_range"]))
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
        fig.frame_width = plot_width
        fig.add_layout(color_bar, "left")
        tab = Panel(child=fig, title=method)
        tabs.append(tab)

    return {
        "insights": itmdt["insights"],
        "tabledata": itmdt["tabledata"],
        "layout": [panel.child for panel in tabs],
        "meta": [panel.title for panel in tabs],
        "container_width": plot_width + 150,
        "how_to_guide": corr_how_to_guides(cfg, plot_height, plot_width),
    }


def corr_how_to_guides(cfg: Config, height: int, width: int) -> Dict[str, List[Tuple[str, str]]]:
    """
    How-to guide for correlation_impact
    """
    htgs: Dict[str, List[Tuple[str, str]]] = {}

    if cfg.pearson.enable:
        htgs["Pearson"] = cfg.pearson.how_to_guide(height, width)
    if cfg.spearman.enable:
        htgs["Spearman"] = cfg.spearman.how_to_guide(height, width)
    if cfg.pearson.enable:
        htgs["KendallTau"] = cfg.kendall.how_to_guide(height, width)

    return htgs


def render_correlation_heatmaps(itmdt: Intermediate, plot_width: int, plot_height: int) -> Tabs:
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

        mapper, color_bar = create_color_mapper(RDBU)
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
    itmdt: Intermediate, plot_width: int, plot_height: int, cfg: Config
) -> Dict[str, Any]:
    """
    Render correlation heatmaps, but with single column
    """
    tabs: List[Panel] = []
    tooltips = [("y", "@y"), ("correlation", "@correlation{1.11}")]

    for method, df in itmdt["data"].items():
        mapper, color_bar = create_color_mapper(RDBU)

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
            title=" ",
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

    return {
        "tabledata": {},
        "layout": [panel.child for panel in tabs],
        "meta": [panel.title for panel in tabs],
        "container_width": plot_width,
        "how_to_guide": corr_how_to_guides(cfg, plot_height, plot_width),
    }


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
    itmdt: Intermediate, plot_width: int, plot_height: int, cfg: Config
) -> Dict[str, Any]:
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
        tools=[],
        x_axis_label=xcol,
        y_axis_label=ycol,
        title=" ",
    )

    # Scatter
    scatter = fig.scatter(x=df.columns[0], y=df.columns[1], source=df)
    if maybe_label:
        assert len(maybe_label) == 1
        mapper = CategoricalColorMapper(factors=["=", "+", "-"], palette=BRG)
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
                LegendItem(label="Most Influential (-)", renderers=[scatter], index=nidx),
                LegendItem(label="Most Influential (+)", renderers=[scatter], index=pidx),
            ],
            margin=0,
            padding=0,
        )

        fig.add_layout(legend, place="right")

    return {
        "layout": [fig],
        "meta": ["Scatter Plot & Regression Line"],
        "container_width": plot_width,
        "how_to_guide": {
            "Scatter Plot & Regression Line": cfg.scatter.how_to_guide(plot_height, plot_width)
        },
    }


######### Interactions for report #########
def render_crossfilter(
    itmdt: Intermediate, plot_width: int, plot_height: int, cfg: Config
) -> column:
    """
    Render crossfilter scatter plot with a regression line.
    """

    # pylint: disable=too-many-locals, too-many-function-args

    if cfg.interactions.cat_enable:
        all_cols = itmdt["all_cols"]
    else:
        all_cols = itmdt["num_cols"]
    scatter_df = itmdt["scatter_source"]
    # all other plots except for scatter plot, used for cat-cat and cat-num interactions.
    other_plots = itmdt["other_plots"]
    if scatter_df.empty:
        scatter_df["__x__"] = [None] * len(itmdt["scatter_source"])
        scatter_df["__y__"] = [None] * len(itmdt["scatter_source"])
    else:
        scatter_df["__x__"] = scatter_df[scatter_df.columns[0]]
        scatter_df["__y__"] = scatter_df[scatter_df.columns[0]]
    source_scatter = ColumnDataSource(scatter_df)
    source_xy_value = ColumnDataSource({"x": [scatter_df.columns[0]], "y": [scatter_df.columns[0]]})
    var_list = list(all_cols)

    xcol = source_xy_value.data["x"][0]
    ycol = source_xy_value.data["y"][0]

    tooltips = [("X-Axis: ", "@__x__"), ("Y-Axis: ", "@__y__")]
    scatter_fig = Figure(
        plot_width=plot_width,
        plot_height=plot_height,
        toolbar_location=None,
        title=Title(text="Scatter Plot", align="center"),
        tools=[],
        x_axis_label=xcol,
        y_axis_label=ycol,
    )
    scatter = scatter_fig.scatter("__x__", "__y__", source=source_scatter)

    hover = HoverTool(tooltips=tooltips, renderers=[scatter])
    scatter_fig.add_tools(hover)

    fig_all_in_one = column(scatter_fig, sizing_mode="stretch_width")

    x_select = Select(title="X-Axis", value=xcol, options=var_list, width=150)
    y_select = Select(title="Y-Axis", value=ycol, options=var_list, width=150)

    x_select.js_on_change(
        "value",
        CustomJS(
            args=dict(
                scatter=source_scatter,
                xy_value=source_xy_value,
                fig_all_in_one=fig_all_in_one,
                scatter_plot=scatter_fig,
                x_axis=scatter_fig.xaxis[0],
                other_plots=other_plots,
            ),
            code="""
        let currentSelect = this.value;
        let xyValueData = xy_value.data;
        let scatterData = scatter.data;
        xyValueData['x'][0] = currentSelect;
        xy_value.change.emit();

        const children = []
        let ycol = xyValueData['y'][0];
        let col = currentSelect + '_' + ycol
        if (col in other_plots) {
            children.push(other_plots[col])
        }
        else {
            scatterData['__x__'] = scatterData[currentSelect];
            x_axis.axis_label = currentSelect;
            scatter.change.emit();
            children.push(scatter_plot)
        }
        fig_all_in_one.children = children;        
        """,
        ),
    )
    y_select.js_on_change(
        "value",
        CustomJS(
            args=dict(
                scatter=source_scatter,
                xy_value=source_xy_value,
                fig_all_in_one=fig_all_in_one,
                scatter_plot=scatter_fig,
                y_axis=scatter_fig.yaxis[0],
                other_plots=other_plots,
            ),
            code="""
        let ycol = this.value;
        let xyValueData = xy_value.data;
        let scatterData = scatter.data;
        xyValueData['y'][0] = ycol;
        xy_value.change.emit();

        const children = []
        let xcol = xyValueData['x'][0];
        let col = xcol + '_' + ycol;
        if (col in other_plots) {
            children.push(other_plots[col])
        }
        else {
            scatterData['__y__'] = scatterData[ycol];
            y_axis.axis_label = ycol;
            scatter.change.emit();
            children.push(scatter_plot)
        }
        fig_all_in_one.children = children;        
        """,
        ),
    )

    interaction_fig = column(
        row(x_select, y_select, align="center"), fig_all_in_one, sizing_mode="stretch_width"
    )
    return interaction_fig


# ######### Interactions for report #########
# def render_crossfilter(
#     itmdt: Intermediate, plot_width: int, plot_height: int
# ) -> column:
#     """
#     Render crossfilter scatter plot with a regression line.
#     """

#     # pylint: disable=too-many-locals, too-many-function-args
#     source_scatter = ColumnDataSource(itmdt["data"])
#     source_coeffs = ColumnDataSource(itmdt["coeffs"])
#     source_xy_value = ColumnDataSource(
#         {"x": [itmdt["data"].columns[0]], "y": [itmdt["data"].columns[0]]}
#     )
#     var_list = list(itmdt["data"].columns)[0:-2]

#     xcol = source_xy_value.data["x"][0]
#     ycol = source_xy_value.data["y"][0]

#     tooltips = [("X-Axis: ", "@__x__"), ("Y-Axis: ", "@__y__")]

#     fig = Figure(
#         plot_width=plot_width,
#         plot_height=plot_height,
#         toolbar_location=None,
#         title=Title(text="Scatter Plot & Regression Line", align="center"),
#         tools=[],
#         x_axis_label=xcol,
#         y_axis_label=ycol,
#     )
#     scatter = fig.scatter("__x__", "__y__", source=source_scatter)
#     fig.line("__x__", "__y__", source=source_coeffs, line_width=3)

#     # Not adding the tooltips before because we only want to apply tooltip to the scatter
#     hover = HoverTool(tooltips=tooltips, renderers=[scatter])
#     fig.add_tools(hover)

#     x_select = Select(title="X-Axis", value=xcol, options=var_list, width=150)
#     y_select = Select(title="Y-Axis", value=ycol, options=var_list, width=150)

#     x_select.js_on_change(
#         "value",
#         CustomJS(
#             args=dict(
#                 scatter=source_scatter,
#                 coeffs=source_coeffs,
#                 xy_value=source_xy_value,
#                 x_axis=fig.xaxis[0],
#             ),
#             code="""
#         let currentSelect = this.value;
#         let xyValueData = xy_value.data;
#         let scatterData = scatter.data;
#         let coeffsData = coeffs.data;

#         xyValueData['x'][0] = currentSelect;
#         scatterData['__x__'] = scatterData[currentSelect];
#         coeffsData['__x__'] = coeffsData[`${currentSelect}${xyValueData['y'][0]}x`];
#         coeffsData['__y__'] = coeffsData[`${currentSelect}${xyValueData['y'][0]}y`];

#         x_axis.axis_label = currentSelect;
#         scatter.change.emit();
#         coeffs.change.emit();
#         xy_value.change.emit();
#         """,
#         ),
#     )
#     y_select.js_on_change(
#         "value",
#         CustomJS(
#             args=dict(
#                 scatter=source_scatter,
#                 coeffs=source_coeffs,
#                 xy_value=source_xy_value,
#                 y_axis=fig.yaxis[0],
#             ),
#             code="""
#         let currentSelect = this.value;
#         let xyValueData = xy_value.data;
#         let scatterData = scatter.data;
#         let coeffsData = coeffs.data;

#         xyValueData['y'][0] = currentSelect;
#         scatterData['__y__'] = scatterData[currentSelect];
#         coeffsData['__x__'] = coeffsData[`${xyValueData['x'][0]}${currentSelect}x`];
#         coeffsData['__y__'] = coeffsData[`${xyValueData['x'][0]}${currentSelect}y`];

#         y_axis.axis_label = currentSelect;
#         scatter.change.emit();
#         coeffs.change.emit();
#         xy_value.change.emit();
#         """,
#         ),
#     )

#     fig = column(
#         row(x_select, y_select, align="center"), fig, sizing_mode="stretch_width"
#     )
#     return fig
