"""
    This module implements the plot_correlation(df) function.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import dask.dataframe as dd
import pandas as pd

from ..configs import Config
from ..container import Container
from ...progress_bar import ProgressBar
from .compute import compute_correlation
from .render import render_correlation

__all__ = ["render_correlation", "compute_correlation", "plot_correlation"]


def plot_correlation(
    df: Union[pd.DataFrame, dd.DataFrame],
    col1: Optional[str] = None,
    col2: Optional[str] = None,
    *,
    value_range: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    display: Optional[List[str]] = None,
    progress: bool = True,
) -> Container:
    """
    This function is designed to calculate the correlation between columns
    There are three functions: plot_correlation(df), plot_correlation(df, x)
    plot_correlation(df, x, y)
    There are also some parameters such as k and value_range to satisfy your requirement

    Parameters
    ----------
    df
        The pandas data_frame for which plots are calculated for each column.
    col1
        A valid column name of the data frame.
    col2
        A valid column name of the data frame.
    value_range
        Range of value.
    k
        Choose top-k element.
    config
        A dictionary for configuring the visualizations
        E.g. config={"scatter.sample_size": 5000}
    display
        A list containing the names of the visualizations to display
        E.g. display=["Pearson"]
    progress
        Enable the progress bar.
    Examples
    --------
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

    Note
    ----
    This function only supports numerical or categorical data,
    and it is better to drop None, Nan and Null value before using it
    """

    cfg = Config.from_dict(display, config)

    with ProgressBar(minimum=1, disable=not progress):
        itmdt = compute_correlation(df, col1, col2, cfg=cfg, value_range=value_range, k=k)
    to_render = render_correlation(itmdt, cfg)

    return Container(to_render, itmdt.visual_type, cfg)
