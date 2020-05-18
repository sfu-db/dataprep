"""
This module implements the plot(df) function.
"""

from typing import Optional, Tuple, Union, Dict

import dask.dataframe as dd
import pandas as pd
from bokeh.io import show

from .compute import compute
from .render import render
from ..report import Report
from ..dtypes import DTypeDef

__all__ = ["plot", "compute", "render"]


def plot(
    df: Union[pd.DataFrame, dd.DataFrame],
    x: Optional[str] = None,
    y: Optional[str] = None,
    z: Optional[str] = None,
    *,
    bins: int = 10,
    ngroups: int = 10,
    largest: bool = True,
    nsubgroups: int = 5,
    timeunit: str = "auto",
    agg: str = "mean",
    sample_size: int = 1000,
    value_range: Optional[Tuple[float, float]] = None,
    yscale: str = "linear",
    tile_size: Optional[float] = None,
    dtype: Optional[DTypeDef] = None,
) -> Report:
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
        Dataframe from which plots are to be generated
    x: Optional[str], default None
        A valid column name from the dataframe
    y: Optional[str], default None
        A valid column name from the dataframe
    z: Optional[str], default None
        A valid column name from the dataframe
    bins: int, default 10
        For a histogram or box plot with numerical x axis, it defines
        the number of equal-width bins to use when grouping.
    ngroups: int, default 10
        When grouping over a categorical column, it defines the
        number of groups to show in the plot. Ie, the number of
        bars to show in a bar chart.
    largest: bool, default True
        If true, when grouping over a categorical column, the groups
        with the largest count will be output. If false, the groups
        with the smallest count will be output.
    nsubgroups: int, default 5
        If x and y are categorical columns, ngroups refers to
        how many groups to show from column x, and nsubgroups refers to
        how many subgroups to show from column y in each group in column x.
    timeunit: str, default "auto"
        Defines the time unit to group values over for a datetime column.
        It can be "year", "quarter", "month", "week", "day", "hour",
        "minute", or "second". With default value "auto", it will use the
        time unit such that the resulting number of groups is closest to 15.
    agg: str, default "mean"
        Specify the aggregate to use when aggregating over a numerical
        column
    sample_size: int, default 1000
        Sample size for the scatter plot
    value_range: Optional[Tuple[float, float]], default None
        The lower and upper bounds on the range of a numerical column.
        Applies when column x is specified and column y is unspecified.
    yscale
        The scale to show on the y axis. Can be "linear" or "log".
    tile_size: Optional[float], default None
        Size of the tile for the hexbin plot. Measured from the middle
        of a hexagon to its left or right corner.
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()

    Examples
    --------
    >>> import pandas as pd
    >>> from dataprep.eda import *
    >>> iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    >>> plot(iris)
    >>> plot(iris, "petal_length", bins=20, value_range=(1,5))
    >>> plot(iris, "petal_width", "species")
    """
    # pylint: disable=too-many-locals,line-too-long

    intermediate = compute(
        df,
        x=x,
        y=y,
        z=z,
        bins=bins,
        ngroups=ngroups,
        largest=largest,
        nsubgroups=nsubgroups,
        timeunit=timeunit.lower(),
        agg=agg,
        sample_size=sample_size,
        value_range=value_range,
        dtype=dtype,
    )
    figure = render(intermediate, yscale=yscale, tile_size=tile_size)

    return Report(figure)
