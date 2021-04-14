"""
Computations for plot(df, ...)
"""


from typing import Optional, Union, List, Dict, Any, Tuple

import dask.dataframe as dd
import pandas as pd

from ...configs import Config
from ...dtypes import DTypeDef, string_dtype_to_object, is_dtype, GeoPoint
from ...intermediate import Intermediate
from ...utils import preprocess_dataframe
from .bivariate import compute_bivariate
from .overview import compute_overview
from .trivariate import compute_trivariate
from .univariate import compute_univariate

__all__ = ["compute"]


def compute(
    df: Union[pd.DataFrame, dd.DataFrame],
    x: Optional[str] = None,
    y: Optional[str] = None,
    z: Optional[str] = None,
    *,
    cfg: Union[Config, Dict[str, Any], None] = None,
    display: Optional[List[str]] = None,
    dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    """
    All in one compute function.

    Parameters
    ----------
    df
        DataFrame from which visualizations are generated
    cfg: Union[Config, Dict[str, Any], None], default None
        When a user call plot(), the created Config object will be passed to compute().
        When a user call compute() directly, if he/she wants to customize the output,
        cfg is a dictionary for configuring. If not, cfg is None and
        default values will be used for parameters.
    display: Optional[List[str]], default None
        A list containing the names of the visualizations to display. Only exist when
        a user call compute() directly and want to customize the output
    x: Optional[str], default None
        A valid column name from the dataframe
    y: Optional[str], default None
        A valid column name from the dataframe
    z: Optional[str], default None
        A valid column name from the dataframe
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    """
    # pylint: disable=too-many-arguments

    params, exlude, ddf = process_latlong(df, x, y, z)
    ddf = preprocess_dataframe(ddf, excluded_columns=exlude)

    if isinstance(cfg, dict):
        cfg = Config.from_dict(display, cfg)

    elif not cfg:
        cfg = Config()

    if not any(params):
        return compute_overview(ddf, cfg, dtype)

    if sum(v is None for v in params) == 2:
        x = params[0] or params[1] or params[2]
        return compute_univariate(ddf, x, cfg, dtype)

    if sum(v is None for v in params) == 1:
        x, y = (v for v in params if v is not None)
        return compute_bivariate(ddf, x, y, cfg, dtype)

    if x is not None and y is not None and z is not None:
        return compute_trivariate(ddf, x, y, z, cfg, dtype)

    raise ValueError("not possible")


def concat_latlong(df: Union[pd.DataFrame, dd.DataFrame], x: Any) -> Tuple[str, Any]:
    """
    Merge Latlong into one new column.
    """

    name = x.lat + "_&_" + x.long
    lat_long = tuple(zip(df[x.lat], df[x.long]))

    return name, lat_long


def process_latlong(
    df: pd.DataFrame,
    x: Optional[str] = None,
    y: Optional[str] = None,
    z: Optional[str] = None,
) -> Tuple[List[Optional[str]], List[str], pd.DataFrame]:
    """
    Process Latlong data tpye.
    """

    params = []
    exclude: List[str] = []
    add_df = df.copy()
    for temp in (x, y, z):
        name = temp
        if isinstance(temp, GeoPoint):
            name, lat_long = concat_latlong(df, temp)
            add_df[name] = lat_long
            exclude.append(name)
        params.append(name)

    return params, exclude, add_df
