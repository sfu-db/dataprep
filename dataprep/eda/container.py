"""
    This module implements the Container class.
"""

import sys
import webbrowser
import random
from tempfile import NamedTemporaryFile
from typing import Any, Dict
from bokeh.io import output_notebook
from bokeh.embed import components
from bokeh.resources import INLINE
from jinja2 import Environment, PackageLoader
from ..utils import is_notebook

output_notebook(INLINE, hide_banner=True)  # for offline usage

ENV_LOADER = Environment(
    loader=PackageLoader("dataprep", "eda/templates"),
)


class Container:
    """
    This class creates a customized Container object for the plot* function.
    """

    def __init__(
        self,
        to_render: Dict[str, Any],
        visual_type: str,
    ) -> None:
        if visual_type in ("distribution_grid", "missing_impact_1vn"):
            self.context = {
                "resources": INLINE.render(),
                "components": components(to_render.get("layout")),
                "tabledata": to_render.get("tabledata"),
                "overview_insights": to_render.get("overview_insights"),
                "column_insights": to_render.get("column_insights"),
                "meta": to_render.get("meta"),
                "title": "DataPrep.EDA Report",
                "rnd": random.randint(0, 99),  # for multiple cells running in the same notebook
                "container_width": f"{to_render['fig_width']*3}px",
                "legend_labels": to_render.get("legend_labels"),
            }
            self.template_base = ENV_LOADER.get_template("grid_base.html")

        elif "_column" in visual_type or visual_type in (
            "missing_impact",
            "missing_impact_1v1",
        ):
            # todo: param management
            if to_render.get("tabledata"):
                to_render["meta"].insert(0, "Stats")
            self.context = {
                "resources": INLINE.render(),
                "tabledata": to_render.get("tabledata"),
                "insights": to_render.get("insights"),
                "components": components(to_render.get("layout")),
                "meta": to_render.get("meta"),
                "title": "DataPrep.EDA Report",
                "rnd": random.randint(100, 999),  # for multiple cells running in the same notebook
            }
            self.template_base = ENV_LOADER.get_template("tab_base.html")
        else:
            raise TypeError(f"Unsupported Visual Type: {visual_type}.")

    def save(self, filename: str) -> None:
        """
        save function
        """
        with open(filename, "w") as f:
            f.write(self.template_base.render(context=self.context))

    def _repr_html_(self) -> str:
        """
        Display itself inside a notebook
        """
        output_html = self.template_base.render(context=self.context)
        return output_html

    def show(self) -> None:
        """
        Render the report. This is useful when calling plot in a for loop.
        """
        # if not called from notebook environment, ref to show_browser function.
        if not is_notebook():
            print(
                "The plot will not show in a notebook environment, "
                "please try 'show_browser' if you want to open it in browser",
                file=sys.stderr,
            )
        try:
            from IPython.display import (  # pylint: disable=import-outside-toplevel
                HTML,
                display,
            )

            display(HTML(self._repr_html_()))
        except ImportError:
            pass

    def show_browser(self) -> None:
        """
        Open the plot in the browser. This is useful when plotting
        from terminmal or when the fig is very large in notebook.
        """

        # set delete = False to avoid early delete when user open multiple plots.
        with NamedTemporaryFile(suffix=".html", delete=False) as tmpf:
            pass
        with open(tmpf.name, "w") as file:
            file.write(self.template_base.render(context=self.context))
        webbrowser.open_new_tab(f"file://{tmpf.name}")
