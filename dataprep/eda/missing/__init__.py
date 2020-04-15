"""
    This module implements the plot_missing(df) function
"""

from typing import Optional, Union

import dask.dataframe as dd
import pandas as pd
from bokeh.io import show

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
) -> Report:
    """
    This function is designed to deal with missing values
    There are three functions: plot_missing(df), plot_missing(df, x)
    plot_missing(df, x, y)

    Parameters
    ----------
    df
        the pandas data_frame for which plots are calculated for each column
    x
        a valid column name of the data frame
    y
        a valid column name of the data frame
    ncols
        The number of columns in the figure
    bins
        The number of rows in the figure
    ndist_sample
        The number of sample points

    Examples
    ----------
    >>> from dataprep.eda.missing.computation import plot_missing
    >>> import pandas as pd
    >>> df = pd.read_csv("suicide-rate.csv")
    >>> plot_missing(df, "HDI_for_year")
    >>> plot_missing(df, "HDI_for_year", "population")
    """
    itmdt = compute_missing(df, x, y, bins=bins, ncols=ncols, ndist_sample=ndist_sample)
    fig = render_missing(itmdt)
    return Report(fig)
