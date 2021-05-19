"""
This module implements the plot(df) function.
"""

from typing import Any, Dict, List, Optional, Union

import dask.dataframe as dd
import pandas as pd

from ..configs import Config
from ..container import Container
from ..dtypes_v2 import DTypeDef, LatLong
from ...progress_bar import ProgressBar
from .compute import compute
from .render import render

__all__ = ["plot", "compute", "render"]


def plot(
    df: Union[pd.DataFrame, dd.DataFrame],
    col1: Optional[Union[str, LatLong]] = None,
    col2: Optional[Union[str, LatLong]] = None,
    col3: Optional[str] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
    display: Optional[List[str]] = None,
    dtype: Optional[DTypeDef] = None,
    progress: bool = True,
) -> Container:
    """Generates plots for exploratory data analysis.

    If no columns are specified, the distribution of
    each coloumn is plotted. A histogram is plotted if the
    column contains numerical values, a bar chart is plotted
    if the column contains categorical values, a line chart is
    plotted if the column is of type datetime.

    If one column (x) is specified, the
    distribution of x is plotted in various ways. If x
    contains categorical values, a bar chart and pie chart are
    plotted. If x contains numerical values, a histogram,
    kernel density estimate plot, box plot, and qq plot are plotted.
    If x contains datetime values, a line chart is plotted.

    If two columns (x and y) are specified, plots depicting
    the relationship between the variables will be displayed. If
    x and y contain numerical values, a scatter plot, hexbin
    plot, and binned box plot are plotted. If one of x and y
    contain categorical values and the other contains numerical values,
    a box plot and multiline histogram are plotted. If x and y
    contain categorical vales, a nested bar chart, stacked bar chart, and
    heat map are plotted. If one of x and y contains datetime values
    and the other contains numerical values, a line chart and a box plot
    are shown. If one of x and y contains datetime values and the other
    contains categorical values, a multiline chart and a stacked box plot
    are shown.

    If x, y, and z are specified, they must be one each of type datetime,
    numerical, and categorical. A multiline chart containing an aggregate
    on the numerical column grouped by the categorical column over time is
    plotted.


    Parameters
    ----------
    df
        DataFrame from which visualizations are generated
    col1: Optional[str], default None
        A valid column name from the dataframe
    col2: Optional[str], default None
        A valid column name from the dataframe
    col3: Optional[str], default None
        A valid column name from the dataframe
    config
        A dictionary for configuring the visualizations
        E.g. config={"hist.bins": 20}
    display
        A list containing the names of the visualizations to display
        E.g. display=["Histogram"]
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous().
    progress
        Enable the progress bar.

    Examples
    --------
    >>> import pandas as pd
    >>> from dataprep.eda import *
    >>> iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    >>> plot(iris)
    >>> plot(iris, "petal_length")
    >>> plot(iris, "petal_width", "species")
    """
    cfg = Config.from_dict(display, config)

    with ProgressBar(minimum=1, disable=not progress):
        itmdt = compute(df, col1, col2, col3, cfg=cfg, dtype=dtype)

    to_render = render(itmdt, cfg)

    return Container(to_render, itmdt.visual_type, cfg)
