"""This module implements the plot_missing(df) function's
calculating intermediate part
"""
from typing import Optional, cast, List, Any, Dict, Union
from warnings import catch_warnings, filterwarnings

from ...configs import Config
from ...data_array import DataArray, DataFrame
from ...dtypes import DTypeDef
from ...utils import preprocess_dataframe
from ...intermediate import Intermediate
from .bivariate import compute_missing_bivariate
from .nullivariate import compute_missing_nullivariate
from .univariate import compute_missing_univariate

__all__ = ["compute_missing"]


def compute_missing(
    df: DataFrame,
    x: Optional[str] = None,
    y: Optional[str] = None,
    *,
    cfg: Union[Config, Dict[str, Any], None] = None,
    display: Optional[List[str]] = None,
    dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    # pylint: disable=too-many-arguments

    """This function is designed to deal with missing values
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
    cfg: Union[Config, Dict[str, Any], None], default None
        When a user call plot_missing(), the created Config object will be passed to
        compute_missing().
        When a user call compute_missing() directly, if he/she wants to customize the output,
        cfg is a dictionary for configuring. If not, cfg is None and
        default values will be used for parameters.
    display: Optional[List[str]], default None
        A list containing the names of the visualizations to display. Only exist when
        a user call compute_missing() directly and want to customize the output
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()

    Examples
    --------
    >>> from dataprep.eda.missing.computation import plot_missing
    >>> import pandas as pd
    >>> df = pd.read_csv("suicide-rate.csv")
    >>> plot_missing(df, "HDI_for_year")
    >>> plot_missing(df, "HDI_for_year", "population")
    """
    df = preprocess_dataframe(df)
    df = DataArray(df)

    # pylint: disable=no-else-raise
    if isinstance(cfg, dict):
        cfg = Config.from_dict(display, cfg)
    elif not cfg:
        cfg = Config()
    if x is None and y is not None:
        raise ValueError("x cannot be None while y has value")
    elif x is not None and y is None:
        ret = compute_missing_univariate(df, x, cfg, dtype)
    elif x is not None and y is not None:
        ret = compute_missing_bivariate(df, x, y, cfg, dtype)
    else:
        # supress divide by 0 error due to heatmap
        with catch_warnings():
            filterwarnings(
                "ignore",
                "invalid value encountered in true_divide",
                category=RuntimeWarning,
            )
            ret = compute_missing_nullivariate(df, cfg)

    return cast(Intermediate, ret)
