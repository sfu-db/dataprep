"""
    This module implements the create_report(df) function.
"""

from typing import Optional
from pathlib import Path
import pandas as pd
from bokeh.resources import CDN
from jinja2 import Environment, PackageLoader, select_autoescape
from .formatter import format_report

__all__ = ["create_report"]

ENV_LOADER = Environment(
    loader=PackageLoader("dataprep", "eda/create_report/templates"),
)


def create_report(
    df: pd.DataFrame,
    title: Optional[str] = "DataPrep Report",
    mode: Optional[str] = "basic",
) -> None:
    """
    This function is to generate and render element in a report object.

    Parameters
    ----------
    df
        The DataFrame for which data are calculated.
    title
        The title of the report.
    """
    context = {
        "cdn": CDN.render(),
        "title": title,
        "components": format_report(df, mode),
    }
    template_base = ENV_LOADER.get_template("base.html")
    with open(Path.cwd() / f"{title}.html", "w") as file:
        file.write(template_base.render(context=context))
