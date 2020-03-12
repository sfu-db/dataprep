"""
    This module implements the plot_correlation(df) function.
"""

from typing import Any, List, Optional, Tuple, Union

import dask.dataframe as dd
import pandas as pd
from bokeh.io import show
from bokeh.plotting import Figure

from .compute import compute_correlation
from .render import render_correlation
from ..report import Report

__all__ = ["render_correlation", "compute_correlation", "plot_correlation"]


def plot_correlation(
    df: Union[pd.DataFrame, dd.DataFrame],
    x: Optional[str] = None,
    y: Optional[str] = None,
    *,
    value_range: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
) -> Figure:
    """
    This function is designed to calculate the correlation between columns
    There are three functions: plot_correlation(df), plot_correlation(df, x)
    plot_correlation(df, x, y)
    There are also some parameters such as k and value_range to satisfy your requirement

    Parameters
    ----------
    pd_data_frame: pd.DataFrame
        the pandas data_frame for which plots are calculated for each column
    x_name: str, optional
        a valid column name of the data frame
    y_name: str, optional
        a valid column name of the data frame
    value_range: list[float], optional
        range of value
    k: int, optional
        choose top-k element

    Returns
    ----------
    An object of figure or
        An object of figure and
        An intermediate representation for the plots of different columns in the data_frame.

    Examples
    ----------
    >>> from dataprep.eda.correlation.computation import plot_correlation
    >>> import pandas as pd
    >>> df = pd.read_csv("suicide-rate.csv")
    >>> plot_correlation(df)
    >>> plot_correlation(df, k=6)
    >>> plot_correlation(df, "suicides")
    >>> plot_correlation(df, "suicides", k=3)
    >>> plot_correlation(df, "suicides", value_range=[-1, 0.3])
    >>> plot_correlation(df, "suicides", value_range=[-1, 0.3], k=2)
    >>> plot_correlation(df, x_name="population", y_name="suicides_no")
    >>> plot_correlation(df, x_name="population", y_name="suicides", k=5)

    Notes
    ----------
    match (x_name, y_name, k)
        case (None, None, None) => heatmap
        case (Some, None, Some) => Top K columns for (pearson, spearman, kendall)
        case (Some, Some, _) => Scatter with regression line with/without top k outliers
        otherwise => error

    This function only supports numerical or categorical data,
    and it is better to drop None, Nan and Null value before using it
    """

    intermediate = compute_correlation(df, x=x, y=y, value_range=value_range, k=k)
    figure = render_correlation(intermediate)

    return Report(figure)
