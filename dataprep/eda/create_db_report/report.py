import sys
import webbrowser
from pathlib import Path
from typing import Optional, Any
import os


def is_notebook() -> Any:
    """
    :return: whether it is running in jupyter notebook
    """
    try:
        # pytype: disable=import-error
        from IPython import get_ipython  # pylint: disable=import-outside-toplevel

        # pytype: enable=import-error

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        return False
    except (NameError, ImportError):
        return False


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

    def __init__(self, report: str, path: str) -> None:
        self.report = report
        self.path = path

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

    def show_browser(self) -> None:
        """
        Open the report in the browser. This is useful when calling from terminal.
        """

        with open(self.path, "w", encoding="utf-8") as file:
            file.write(self.report)
        webbrowser.open(f"file://{self.path}", new=2)
