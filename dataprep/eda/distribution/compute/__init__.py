"""
Computations for plot(df, ...)
"""

from typing import Optional, Union

import dask.dataframe as dd
import pandas as pd

from ...configs import Config
from ...dtypes import DTypeDef, string_dtype_to_object
from ...intermediate import Intermediate
from ...utils import to_dask
from .bivariate import compute_bivariate
from .overview import compute_overview
from .trivariate import compute_trivariate
from .univariate import compute_univariate

__all__ = ["compute"]


def compute(
    df: Union[pd.DataFrame, dd.DataFrame],
    cfg: Config,
    x: Optional[str] = None,
    y: Optional[str] = None,
    z: Optional[str] = None,
    dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    """
    All in one compute function.

    Parameters
    ----------
    df
        DataFrame from which visualizations are generated
    cfg
        Config instance
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

    df = to_dask(df)
    df.columns = df.columns.astype(str)
    df = string_dtype_to_object(df)

    if not any((x, y, z)):
        return compute_overview(df, cfg, dtype)

    if sum(v is None for v in (x, y, z)) == 2:
        x = x or y or z
        return compute_univariate(df, x, cfg, dtype)

    if sum(v is None for v in (x, y, z)) == 1:
        x, y = (v for v in (x, y, z) if v is not None)
        return compute_bivariate(df, x, y, cfg, dtype)

    if x is not None and y is not None and z is not None:
        return compute_trivariate(df, x, y, z, cfg, dtype)

    raise ValueError("not possible")
