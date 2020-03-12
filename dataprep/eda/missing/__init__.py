"""
    This module implements the plot_missing(df) function
"""

from typing import Optional, Union

import dask.dataframe as dd
import pandas as pd
from bokeh.io import show
from bokeh.models import LayoutDOM

from .compute import compute_missing
from .render import render_missing
from ..report import Report

__all__ = ["render_missing", "compute_missing", "plot_missing"]


def plot_missing(
    df: Union[pd.DataFrame, dd.DataFrame],
    x: Optional[str] = None,
    y: Optional[str] = None,
    *,
    bins: int = 30,
    ncols: int = 30,
    ndist_sample: int = 100,
) -> LayoutDOM:
    """
    This function is designed to deal with missing values
    There are three functions: plot_missing(df), plot_missing(df, x)
    plot_missing(df, x, y)

    Parameters
    ----------
    df: Union[pd.DataFrame, dd.DataFrame]
        the pandas data_frame for which plots are calculated for each column
    x_name: str, optional
        a valid column name of the data frame
    y_name: str, optional
        a valid column name of the data frame
    ncols: int, optional
        The number of columns in the figure
    bins: int
        The number of rows in the figure
    ndist_sample: int
        The number of sample points

    Returns
    ----------
    An object of figure or
        An object of figure and
        An intermediate representation for the plots of different columns in the data_frame.

    Examples
    ----------
    >>> from dataprep.eda.missing.computation import plot_missing
    >>> import pandas as pd
    >>> df = pd.read_csv("suicide-rate.csv")
    >>> plot_missing(df, "HDI_for_year")
    >>> plot_missing(df, "HDI_for_year", "population")

    Notes
    ----------
    match (x_name, y_name)
        case (Some, Some) => histogram for numerical column,
        bars for categorical column, qq-plot, box-plot, jitter plot,
        CDF, PDF
        case (Some, None) => histogram for numerical column and
        bars for categorical column
        case (None, None) => heatmap
        otherwise => error
    """
    itmdt = compute_missing(df, x, y, bins=bins, ncols=ncols, ndist_sample=ndist_sample)
    fig = render_missing(itmdt)
    return Report(fig)
