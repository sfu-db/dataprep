"""
    This module implements the plot_missing(df) function
"""

from typing import Any, Dict, List, Optional, Union

import dask.dataframe as dd
import pandas as pd

from ..configs import Config
from ..container import Container
from ..dtypes_v2 import DTypeDef
from ...progress_bar import ProgressBar
from .compute import compute_missing
from .render import render_missing

__all__ = ["render_missing", "compute_missing", "plot_missing"]


def plot_missing(
    df: Union[pd.DataFrame, dd.DataFrame],
    col1: Optional[str] = None,
    col2: Optional[str] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
    display: Optional[List[str]] = None,
    dtype: Optional[DTypeDef] = None,
    progress: bool = True,
) -> Container:
    """
    This function is designed to deal with missing values
    There are three functions: plot_missing(df), plot_missing(df, x)
    plot_missing(df, x, y)

    Parameters
    ----------
    df
        the pandas data_frame for which plots are calculated for each column.
    col1
        a valid column name of the data frame.
    col2
        a valid column name of the data frame.
    config
        A dictionary for configuring the visualizations.
        E.g. config={"spectrum.bins": 20}
    display
        A list containing the names of the visualizations to display
        E.g. display=["Stats", "Spectrum"]
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous().
    progress
        Enable the progress bar.

    Examples
    ----------
    >>> from dataprep.eda.missing.computation import plot_missing
    >>> import pandas as pd
    >>> df = pd.read_csv("suicide-rate.csv")
    >>> plot_missing(df, "HDI_for_year")
    >>> plot_missing(df, "HDI_for_year", "population")
    """
    cfg = Config.from_dict(display, config)

    with ProgressBar(minimum=1, disable=not progress):
        itmdt = compute_missing(df, col1, col2, cfg=cfg, dtype=dtype)

    to_render = render_missing(itmdt, cfg)

    return Container(to_render, itmdt.visual_type, cfg)
