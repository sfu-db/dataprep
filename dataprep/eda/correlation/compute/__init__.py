"""This module implements the intermediates computation
for plot_correlation(df) function."""

from typing import Optional, Tuple

from ...configs import Config
from ...data_array import DataArray, DataFrame
from ...dtypes import NUMERICAL_DTYPES
from ...intermediate import Intermediate
from ...utils import to_dask
from .bivariate import _calc_bivariate
from .nullivariate import _calc_nullivariate
from .univariate import _calc_univariate

__all__ = ["compute_correlation"]


def compute_correlation(
    df: DataFrame,
    cfg: Config,
    x: Optional[str] = None,
    y: Optional[str] = None,
    *,
    value_range: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
) -> Intermediate:
    """
    Parameters
    ----------
    df
        The pandas dataframe for which plots are calculated for each column.
    cfg
        Config instance
    x
        A valid column name of the dataframe
    y
        A valid column name of the dataframe
    value_range
        If the correlation value is out of the range, don't show it.
    k
        Choose top-k element
    """
    if x and y or not x and not y:
        df = to_dask(df.select_dtypes(NUMERICAL_DTYPES))
    else:
        df = DataArray(df).select_num_columns()

    if x is None and y is None:  # pylint: disable=no-else-return
        return _calc_nullivariate(df, cfg, value_range=value_range, k=k)
    elif x is not None and y is None:
        return _calc_univariate(df, x, cfg, value_range=value_range, k=k)
    elif x is None and y is not None:
        raise ValueError("Please give the column name to x instead of y")
    elif x is not None and y is not None:
        return _calc_bivariate(df, cfg, x, y, k=k)

    raise ValueError("Not Possible")
