"""
    This module implements the create_report(df) function.
"""

import warnings
from typing import Any, Dict, List, Optional

import pandas as pd
from bokeh.resources import INLINE
from jinja2 import Environment, PackageLoader

from ..configs import Config
from .formatter import format_report
from .report import Report

__all__ = ["create_report"]

ENV_LOADER = Environment(
    loader=PackageLoader("dataprep", "eda/create_report/templates"),
)


def create_report(
    df: pd.DataFrame,
    *,
    config: Optional[Dict[str, Any]] = None,
    display: Optional[List[str]] = None,
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
    config
        A dictionary for configuring the visualizations
        E.g. config={"hist.bins": 20}
    display
        The list that contains the names of plots user wants to display,
        E.g. display =  ["bar", "hist"]
        Without user's specifications, the default is "auto"
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
    _suppress_warnings()
    cfg = Config.from_dict(display, config)
    context = {
        "resources": INLINE.render(),
        "title": title,
        "components": format_report(df, cfg, mode, progress),
    }
    template_base = ENV_LOADER.get_template("base.html")
    report = template_base.render(context=context)
    return Report(report)


def _suppress_warnings() -> None:
    """
    suppress warnings in create_report
    """
    warnings.filterwarnings(
        "ignore",
        "The default value of regex will change from True to False in a future version",
        category=FutureWarning,
    )
