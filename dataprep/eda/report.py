"""
    This module implements the Report class.
"""

from tempfile import NamedTemporaryFile
from bokeh.models import LayoutDOM
from bokeh.io import save
from bokeh.resources import CDN


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
        save(self.to_render, filename=filename, resources=CDN, title="Report")

    def _repr_html_(self) -> str:
        with NamedTemporaryFile(suffix=".html") as f:
            save(self.to_render, filename=f.name, resources=CDN, title="Report")
            output_html = f.read().decode("utf-8")

        # embed into report template created by us here
        return output_html
