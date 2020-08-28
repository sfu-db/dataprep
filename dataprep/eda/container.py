"""
    This module implements the Container class.
"""

import sys
import webbrowser
from tempfile import NamedTemporaryFile
from typing import List, Dict, Union, Tuple
from bokeh.io import output_notebook
from bokeh.embed import components
from bokeh.models import LayoutDOM
from bokeh.resources import INLINE
from jinja2 import Environment, PackageLoader
from .utils import is_notebook

output_notebook(INLINE, hide_banner=True)  # for offline usage

ENV_LOADER = Environment(
    loader=PackageLoader("dataprep", "eda/distribution/templates"),
)


class Container:
    """
    This class creates a customized Container object for the plot(df) function.
    """

    def __init__(
        self,
        to_render: Dict[
            str,
            Union[
                List[str],
                List[LayoutDOM],
                Tuple[Dict[str, str], Dict[str, str]],
                Dict[int, List[str]],
            ],
        ],
    ) -> None:
        self.context = {
            "resources": INLINE.render(),
            "components": components(to_render["layout"]),
            "tabledata": to_render["tabledata"],
            "overview_insights": to_render["overview_insights"],
            "column_insights": to_render["column_insights"],
            "meta": to_render["meta"],
            "title": "DataPrep.EDA Report",
        }
        self.template_base = ENV_LOADER.get_template("base.html")

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
