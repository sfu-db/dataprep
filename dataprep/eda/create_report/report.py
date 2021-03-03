"""
    This module implements the Report class.
"""
import sys
import webbrowser
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from ...utils import is_notebook

CELL_HEIGHT_OVERRIDE = """<style>
                            div.output_scroll {
                              height: 850px;
                            }
                            div.cell-output>div:first-of-type {
                              max-height: 850px !important;
                            }
                          </style>"""


class Report:
    """
    This class creates a customized Report object for the create_report function
    """

    def __init__(self, report: str) -> None:
        self.report = report

    def _repr_html_(self) -> str:
        """
        Display report inside a notebook
        """
        return f"{CELL_HEIGHT_OVERRIDE}<div style='background-color: #fff;'>{self.report}</div>"

    def __repr__(self) -> str:
        """
        Remove object name
        """
        return ""

    def save(
        self,
        filename: Optional[str] = "report",
        to: Optional[str] = None,
    ) -> None:
        """
        Save report to current working directory.

        Parameters
        ----------
        filename: Optional[str], default 'report'
            The filename used for saving report without the extension name.
        to: Optional[str], default Path.cwd()
            The path to where the report will be saved.
        """
        # pylint: disable=invalid-name
        if to:
            path = Path(to).expanduser()
        else:
            path = Path.cwd()

        if not path.is_dir():
            raise ValueError("The second parameter is not a valid path.")

        with open(path / f"{filename}.html", "w", encoding="utf-8") as file:
            file.write(self.report)
        print(f"Report has been saved to {path}/{filename}.html!")

    def show_browser(self) -> None:
        """
        Open the report in the browser. This is useful when calling from terminmal.
        """

        with NamedTemporaryFile(suffix=".html", delete=False) as tmpf:
            pass
        with open(tmpf.name, "w", encoding="utf-8") as file:
            file.write(self.report)
        webbrowser.open(f"file://{tmpf.name}", new=2)

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
