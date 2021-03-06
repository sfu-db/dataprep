"""This module implements the intermediates computation
for plot_correlation(df) function."""

from typing import Optional, Tuple, List, Dict, Union, Any

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
    x: Optional[str] = None,
    y: Optional[str] = None,
    *,
    cfg: Union[Config, Dict[str, Any], None] = None,
    display: Optional[List[str]] = None,
    value_range: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
) -> Intermediate:
    # pylint: disable=too-many-arguments
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
    cfg: Union[Config, Dict[str, Any], None], default None
        When a user call plot_correlation(), the created Config object will be passed to
        compute_correlation().
        When a user call compute_correlation() directly, if he/she wants to customize the output,
        cfg is a dictionary for configuring. If not, cfg is None and
        default values will be used for parameters.
    display: Optional[List[str]], default None
        A list containing the names of the visualizations to display. Only exist when
        a user call compute_correlation() directly and want to customize the output
    k
        Choose top-k element
    """
    if isinstance(cfg, dict):
        cfg = Config.from_dict(display, cfg)
    elif not cfg:
        cfg = Config()
    if x is not None and y is not None:
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
