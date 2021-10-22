"""
    This module implements the create_report(df) function.
"""
import warnings
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import dask.dataframe as dd
from bokeh.resources import INLINE
from jinja2 import Environment, PackageLoader

from .diff_formatter import format_diff_report
from ..configs import Config
from ..create_report.report import Report
from collections import defaultdict

__all__ = ["create_diff_report"]

ENV_LOADER = Environment(
    loader=PackageLoader("dataprep", "eda/create_report/templates"),
)


def create_diff_report(
    df: Union[List[Union[pd.DataFrame, dd.DataFrame]], Union[pd.DataFrame, dd.DataFrame]],
    config: Optional[Dict[str, Any]] = None,
    display: Optional[List[str]] = None,
    title: Optional[str] = "DataPrep Report",
    mode: Optional[str] = "basic",
    progress: bool = True,
) -> Report:
    """
    This function is to generate and render element in a report object.
    It is similar to create_report, but specifically for the difference of 2 or
    more dataframes.

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

    components_lst = format_diff_report(df, cfg, mode, progress)
    dict_stats = defaultdict(list)
    insights = []
    for comps in components_lst:
        for key, value in comps["overview"][0].items():
            if value is not None:
                dict_stats[key].append(value)
        insights.append(comps["overview_insights"])

    context = {
        "resources": INLINE.render(),
        "title": title,
        "components": components_lst,
        "stats": dict_stats,
        "insights": insights,
        "is_diff_report": True,
        "df_labels": cfg.diff.label,
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
