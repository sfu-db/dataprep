"""
    This module implements the create_report(df) function.
"""

from typing import Optional
from pathlib import Path
import pandas as pd
from bokeh.resources import INLINE
from jinja2 import Environment, PackageLoader
from .formatter import format_report
from .report import Report

__all__ = ["create_report"]

ENV_LOADER = Environment(
    loader=PackageLoader("dataprep", "eda/create_report/templates"),
)


def create_report(
    df: pd.DataFrame,
    title: Optional[str] = "DataPrep Report",
    mode: Optional[str] = "basic",
    progress: bool = True,
) -> Report:
    """
    This function is to generate and render element in a report object.

    Parameters
    ----------
    df
        The DataFrame for which data are calculated.
    title: Optional[str], default "DataPrep Report"
        The title of the report, which will be shown on the navigation bar.
    mode: Optional[str], default "basic"
        This controls what type of report to be generated.
        Currently only the 'basic' is fully implemented.
    progress
        Whether to show the progress bar.

    Examples
    --------
    >>> import pandas as pd
    >>> from dataprep.eda import create_report
    >>> df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    >>> report = create_report(df)
    >>> report # show report in notebook
    >>> report.save('My Fantastic Report') # save report to local disk
    >>> report.show_browser() # show report in the browser
    """
    context = {
        "resources": INLINE.render(),
        "title": title,
        "components": format_report(df, mode, progress),
    }
    template_base = ENV_LOADER.get_template("base.html")
    report = template_base.render(context=context)
    return Report(report)
