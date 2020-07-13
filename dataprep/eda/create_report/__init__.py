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
    title: Optional[str], default "DataPrep Report"
        The title and the filename of the report.
    mode: Optional[str], default "basic"
        This controls what type of report to be generated.
        Currently only the 'basic' is fully implemented.

    Examples
    --------
    >>> import pandas as pd
    >>> from dataprep.eda import create_report
    >>> iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    >>> create_report(iris)
    """
    context = {
        "cdn": CDN.render(),
        "title": title,
        "components": format_report(df, mode),
    }
    template_base = ENV_LOADER.get_template("base.html")
    with open(Path.cwd() / f"{title}.html", "w") as file:
        file.write(template_base.render(context=context))
