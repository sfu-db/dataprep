"""
    This module implements the Report class.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile

from bokeh.io import save
from bokeh.models import LayoutDOM
from bokeh.resources import CDN
from IPython.display import HTML, display
from jinja2 import Template

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
    This class creates a customized Report object for the plot* functions
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
        display(HTML(self._repr_html_()))
