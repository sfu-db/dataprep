"""
    This module implements the Report class.
"""

import sys
import webbrowser
from pathlib import Path
from tempfile import NamedTemporaryFile

from bokeh.io import save
from bokeh.io.notebook import load_notebook
from bokeh.embed.notebook import notebook_content
from bokeh.models import LayoutDOM
from bokeh.resources import CDN
from IPython.display import HTML, display
from jinja2 import Template

from .utils import is_notebook

INLINE_TEMPLATE = Template(
    """
{% from macros import embed %}
{% block inner_body %}
    {% block contents %}
    {% for doc in docs %}
        {{ embed(doc) if doc.elementid }}
        {% for root in doc.roots %}
        {% block root scoped %}
            {{ embed(root) | indent(10) }}
        {% endblock %}
        {% endfor %}
    {% endfor %}
    {% endblock %}
    {{ plot_script | indent(8) }}
{% endblock %}
"""
)


class Report:
    """
    This class creates a customized Report object for the plot* functions and create_report function
    """

    to_render: LayoutDOM

    def __init__(self, to_render: LayoutDOM) -> None:
        self.to_render = to_render

    def save(self, filename: str) -> None:
        """
        save function
        """
        save(
            self.to_render,
            filename=filename,
            resources=CDN,
            title="DataPrep.EDA Report",
        )

    def _repr_html_(self) -> str:
        """
        Display itself inside a notebook
        """
        # Speical case inside Google Colab
        if "google.colab" in sys.modules:
            load_notebook(hide_banner=True)
            script, div, _ = notebook_content(self.to_render)
            return f"{div}<script>{script}</script>"

        # Windows forbids us open the file twice as the result bokeh cannot
        # write to the opened temporary file.
        with NamedTemporaryFile(suffix=".html", delete=False) as tmpf:
            pass

        save(
            self.to_render,
            filename=tmpf.name,
            resources=CDN,
            template=INLINE_TEMPLATE,
            title="DataPrep.EDA Report",
        )
        with open(tmpf.name, "r") as f:
            output_html = f.read()

        # Delete the temporary file
        Path(tmpf.name).unlink()

        # Fix the bokeh: bokeh wrongly call the "waiting for bokeh to load" function
        # inside "Bokeh.safely", which causes Bokeh not found because
        # Bokeh is even not loaded!
        patched_html = output_html.replace(
            "Bokeh.safely",
            "var __dataprep_bokeh_fix = (f) => document.Bokeh === undefined ? setTimeout(f, 1000) : f(); __dataprep_bokeh_fix",  # pylint: disable=line-too-long
        )
        # embed into report template created by us here
        return patched_html

    def show(self) -> None:
        """
        Render the report. This is useful when calling plot in a for loop.
        """

        # if not call from notebook environment, ref to show_browser function.
        if not is_notebook():
            print(
                "The report is not shown in a notebook environment,"
                " please try 'show_browser' if you want to open it in browser",
                file=sys.stderr,
            )

        display(HTML(self._repr_html_()))

    def show_browser(self) -> None:
        """
        Open the report in the browser. This is useful when plotting
        from terminmal or when the fig is very large in notebook.
        """

        # set delete = False to avoid early delete when user open multiple plots.
        with NamedTemporaryFile(suffix=".html", delete=False) as tmpf:
            save(
                self.to_render,
                filename=tmpf.name,
                resources=CDN,
                title="DataPrep.EDA Report",
            )
            webbrowser.open_new_tab(f"file://{tmpf.name}")
