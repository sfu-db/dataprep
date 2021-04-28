"""
    This module implements the plot_diff function.
"""

from typing import Optional, Union, List, Dict, Any
import dask.dataframe as dd
import pandas as pd

from ..configs import Config
from ..container import Container
from ..dtypes import DTypeDef
from ...progress_bar import ProgressBar
from .compute import compute_diff
from .render import render_diff

__all__ = ["plot_diff", "compute_diff", "render_diff"]


def plot_diff(
    df: Union[List[Union[pd.DataFrame, dd.DataFrame]], Union[pd.DataFrame, dd.DataFrame]],
    x: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    display: Optional[List[str]] = None,
    dtype: Optional[DTypeDef] = None,
    progress: bool = False,
) -> Container:
    """
    This function is to compute and visualize the differences between 2 or more(up to 5) datasets.

    Parameters
    ----------
    df
        The DataFrame(s) to be compared.
    x
        The column to be emphasized in the comparision.
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
        Whether to show the progress bar.

    Examples
    --------
    >>> from dataprep.datasets import load_dataset
    >>> from dataprep.eda import plot_diff
    >>> df_train = load_dataset('house_prices_train')
    >>> df_test = load_dataset('house_prices_test')
    >>> plot_diff([df_train, df_test])
    """
    # pylint: disable=too-many-arguments
    cfg = Config.from_dict(display, config)

    with ProgressBar(minimum=1, disable=not progress):
        intermediate = compute_diff(df, x=x, cfg=cfg, dtype=dtype)
    to_render = render_diff(intermediate, cfg=cfg)
    return Container(to_render, intermediate.visual_type, cfg)
