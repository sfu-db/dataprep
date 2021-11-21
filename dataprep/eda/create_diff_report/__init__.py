"""
    This module implements the create_diff_report([df1, df2]) function.
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
    loader=PackageLoader("dataprep", "eda/create_diff_report/templates"),
)


def create_diff_report(
    df_list: Union[List[pd.DataFrame], Dict[str, pd.DataFrame]],
    config: Optional[Dict[str, Any]] = None,
    display: Optional[List[str]] = None,
    title: Optional[str] = "DataPrep Report",
    mode: Optional[str] = "basic",
    progress: bool = True,
) -> Report:

    _suppress_warnings()
    cfg = Config.from_dict(display, config)

    components = format_diff_report(df_list, cfg, mode, progress)

    dict_stats = defaultdict(list)

    for comps in components["dfs"]:
        for key, value in comps["overview"][0].items():
            if value is not None:
                dict_stats[key].append(value)

    context = {
        "resources": INLINE.render(),
        "title": title,
        "stats": dict_stats,
        "components": components,
        "is_diff_report": True,
        "df_labels": cfg.diff.label,
        "legend_labels": components["legend_lables"]
    }

    # return context

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
