"""Computations for plot(df, x, y, z)."""

from typing import Optional

import dask
import dask.dataframe as dd

from ...configs import Config
from ...dtypes_v2 import (
    Continuous,
    DateTime,
    DTypeDef,
    Nominal,
    GeoGraphy,
    SmallCardNum,
    GeoPoint,
    Union,
)
from ...intermediate import Intermediate
from ...eda_frame import EDAFrame
from ...utils import _calc_line_dt


def compute_trivariate(
    df: Union[dd.DataFrame, dd.DataFrame],
    col1: str,
    col2: str,
    col3: str,
    cfg: Config,
    dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    """Compute functions for plot(df, x, y, z).

    Parameters
    ----------
    df
        DataFrame from which visualizations are generated
    x
        A column name from the DataFrame
    y
        A column name from the DataFrame
    z
        A column name from the DataFrame
    cfg:
        Config instance
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    """
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals

    x, y, z = col1, col2, col3
    frame = EDAFrame(df[[x, y, z]], dtype)

    xtype = frame.get_eda_dtype(x)
    ytype = frame.get_eda_dtype(y)
    ztype = frame.get_eda_dtype(z)

    # Note that CategoricalTypes need to be defined case by case. Whether
    # SmallCardNum and GeoPoint treated as Categorical is depends on the function.
    # pylint: disable = invalid-name
    CategoricalTypes = (Nominal, GeoGraphy, SmallCardNum, GeoPoint)

    # Make x datetime, y: numerical, z: categorical
    if (
        isinstance(xtype, DateTime)
        and isinstance(ytype, CategoricalTypes)
        and isinstance(ztype, Continuous)
    ):
        y, z = z, y
    elif (
        isinstance(xtype, Continuous)
        and isinstance(ytype, DateTime)
        and isinstance(ztype, CategoricalTypes)
    ):
        x, y = y, x
    elif (
        isinstance(xtype, Continuous)
        and isinstance(ytype, CategoricalTypes)
        and isinstance(ztype, DateTime)
    ):
        x, y, z = z, x, y
    elif (
        isinstance(xtype, CategoricalTypes)
        and isinstance(ytype, DateTime)
        and isinstance(ztype, Continuous)
    ):
        x, y, z = y, z, x
    elif (
        isinstance(xtype, CategoricalTypes)
        and isinstance(ytype, Continuous)
        and isinstance(ztype, DateTime)
    ):
        x, z = z, x
    else:
        raise ValueError(
            "Three column types must be one each of type datetime, numerical, and categorical."
            + f" Current types:({x},{xtype}), ({y},{ytype}), ({z},{ztype})"
        )

    tmp_df = frame.frame[[x, y, z]].dropna()
    tmp_df[z] = tmp_df[z].astype(str)

    # line chart
    data = dask.compute(
        dask.delayed(_calc_line_dt)(
            df, cfg.line.unit, cfg.line.agg, cfg.line.ngroups, cfg.line.sort_descending
        )
    )
    return Intermediate(
        x=x,
        y=y,
        z=z,
        agg=cfg.line.agg,
        data=data[0],
        visual_type="dt_cat_num_cols",
    )
