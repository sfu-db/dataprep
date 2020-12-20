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

TAB_VISUAL_TYPES = {
    "missing_impact_1v1",
    "missing_impact",
    "categorical_column",
    "numerical_column",
    "datetime_column",
    "cat_and_num_cols",
    "two_num_cols",
    "two_cat_cols",
    "dt_and_num_cols",
    "dt_and_cat_cols",
    "dt_cat_num_cols",
    "correlation_impact",
    "correlation_single_heatmaps",
    "correlation_scatter",
}

GRID_VISUAL_TYPES = {"distribution_grid", "missing_impact_1vn"}


class Container:
    """
    This class creates a customized Container object for the plot* function.
    """

    def __init__(
        self,
        to_render: Dict[str, Any],
        visual_type: str,
    ) -> None:
        self.context = Context(**to_render)
        setattr(self.context, "rnd", random.randint(0, 9999))
        if visual_type in GRID_VISUAL_TYPES:
            self.template_base = ENV_LOADER.get_template("grid_base.html")
        elif visual_type in TAB_VISUAL_TYPES:
            if to_render.get("tabledata"):
                self.context.meta.insert(0, "Stats")  # type: ignore
            if visual_type == "correlation_impact":
                self.template_base = ENV_LOADER.get_template("tab_base_corr.html")
            else:
                self.template_base = ENV_LOADER.get_template("tab_base.html")
        else:
            raise TypeError(f"Unsupported Visual Type: {visual_type}.")

    def save(self, filename: str) -> None:
        """
        save function
        """
        with open(filename, "w", encoding="utf-8") as f:
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
        with open(tmpf.name, "w", encoding="utf-8") as file:
            file.write(self.template_base.render(context=self.context))
        webbrowser.open_new_tab(f"file://{tmpf.name}")


class Context:
    """
    Define the context class that stores all the parameters needed by template engine.
    The instance is read-only.
    """

    _title = "DataPrep.EDA Report"
    _resources = INLINE.render()
    _container_width = 650  # default width just in case nothing got passed in

    def __init__(self, **param: Any) -> None:
        self.title = self._title
        self.resources = self._resources
        self.container_width = self._container_width

        for attr, value in param.items():
            if attr == "layout":
                setattr(self, "components", components(value))
            else:
                setattr(self, attr, value)

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except KeyError:
            return None

    def __getattr__(self, attr: str) -> None:
        return None
