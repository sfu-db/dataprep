"""
This module implements functions for plotting visualizations for two fields.
"""
# pytype: disable=import-error
import math
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from bokeh.models import ColorBar, ColumnDataSource, FactorRange, HoverTool
from bokeh.palettes import viridis  # pylint: disable=E0611
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from ..common import sample_n

TOOLS = ""


class MultiViz:
    """
    Encapsulation for multivariate vizualisations.
    """

    nested_cat: bool = False
    scatter: bool = False

    def nest_cat_viz(
        self, data: Dict[Tuple[Any, Any], int], col_x: str, col_y: str
    ) -> Any:
        """
        Nested categories plot
        :param data: the intermediates result
        :param col_x: column X
        :param col_y: column Y
        :return: Bokeh Plot Figure
        """
        x_values = list([tuple(map(str, i)) for i in data.keys()])
        counts = list(data.values())
        data_source = ColumnDataSource(data=dict(x_values=x_values, counts=counts))

        plot_figure = figure(
            x_range=FactorRange(*x_values),
            tools=TOOLS,
            toolbar_location="above",
            title="{} in {}".format(col_y, col_x),
        )

        plot_figure.vbar(
            x="x_values",
            top="counts",
            width=1,
            source=data_source,
            line_color="white",
            line_width=3,
        )
        plot_figure.y_range.start = 0
        plot_figure.x_range.range_padding = 0.2
        plot_figure.xgrid.grid_line_color = None
        plot_figure.yaxis.axis_label = "Count"
        plot_figure.xaxis.axis_label = "{} categorized in {}".format(col_x, col_y)
        plot_figure.xaxis.major_label_orientation = math.pi / 4
        self.nested_cat = True
        plot_figure.title.text_font_size = "10pt"
        return plot_figure

    def scatter_viz(  # pylint: disable=C0330, R0914
        self,
        points: List[Tuple[Any, Any]],
        col_x: str,
        col_y: str,
        tile_size: Optional[float] = None,
    ) -> Any:
        """
        Modified Hex bin scatter plot
        :param points: list of points to be plotted
        :param col_x: column X
        :param col_y: column Y
        :param tile_size: hex tile size
        :return: Bokeh Plot Figure
        """
        x_values = np.array([t[0] for t in points])
        y_values = np.array([t[1] for t in points])

        if tile_size is None:
            xmin, xmax = min(x_values), max(x_values)
            xsize = (xmax - xmin) // 20
            ymin, ymax = min(y_values), max(y_values)
            ysize = (ymax - ymin) // 20

            tile_size = max(xsize, ysize)

        plot_figure = figure(
            match_aspect=True,
            tools=TOOLS,
            title="{} v/s {}".format(col_x, col_y),
            toolbar_location=None,
            background_fill_color="#f5f5f5",
        )
        plot_figure.grid.visible = False

        cmap = list(reversed(viridis(256)))

        renderer, bins = plot_figure.hexbin(
            x_values,
            y_values,
            size=tile_size,
            hover_alpha=0.8,
            palette=cmap,
            hover_color="pink",
            line_color="white",
            aspect_scale=1,
        )

        if bins.counts.size == 0:
            max_bin_count = 0
            min_bin_count = 0
        else:
            max_bin_count = max(bins.counts)
            min_bin_count = min(bins.counts)

        color_mapper = linear_cmap("c", cmap, min_bin_count, max_bin_count)

        x_values, y_values = sample_n(x_values, 100), sample_n(y_values, 100)
        plot_figure.circle(x_values, y_values, color="white", size=3, name="points")

        plot_figure.add_tools(
            HoverTool(
                tooltips=[("Count", "@c")],
                point_policy="follow_mouse",
                renderers=[renderer],
            )
        )
        plot_figure.add_tools(
            HoverTool(
                tooltips=[("x", "$x"), ("y", "$y")], mode="mouse", names=["points"]
            )
        )

        color_bar = ColorBar(
            color_mapper=color_mapper["transform"], width=8, location=(0, 0)
        )
        plot_figure.add_layout(color_bar, "right")

        plot_figure.xaxis.axis_label = "{}".format(col_x)
        plot_figure.yaxis.axis_label = "{}".format(col_y)
        plot_figure.xaxis.major_label_orientation = math.pi / 4
        self.scatter = True
        plot_figure.title.text_font_size = "10pt"
        return plot_figure
