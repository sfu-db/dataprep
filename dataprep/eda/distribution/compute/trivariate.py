"""Computations for plot(df, x, y, z)."""

from typing import Optional

import dask
import dask.dataframe as dd

from ...configs import Config
from ...dtypes import (
    Continuous,
    DateTime,
    DTypeDef,
    Nominal,
    GeoGraphy,
    detect_dtype,
    drop_null,
    is_dtype,
)
from ...intermediate import Intermediate
from ...utils import _calc_line_dt


def compute_trivariate(
    df: dd.DataFrame,
    x: str,
    y: str,
    z: str,
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

    xtype = detect_dtype(df[x], dtype)
    ytype = detect_dtype(df[y], dtype)
    ztype = detect_dtype(df[z], dtype)

    if (
        is_dtype(xtype, DateTime())
        and (is_dtype(ytype, Nominal()) or is_dtype(ytype, GeoGraphy()))
        and is_dtype(ztype, Continuous())
    ):
        y, z = z, y
    elif (
        is_dtype(xtype, Continuous())
        and is_dtype(ytype, DateTime())
        and (is_dtype(ztype, Nominal()) or is_dtype(ztype, GeoGraphy()))
    ):
        x, y = y, x
    elif (
        is_dtype(xtype, Continuous())
        and (is_dtype(ytype, Nominal()) or is_dtype(ytype, GeoGraphy()))
        and is_dtype(ztype, DateTime())
    ):
        x, y, z = z, x, y
    elif (
        (is_dtype(xtype, Nominal()) or is_dtype(xtype, GeoGraphy()))
        and is_dtype(ytype, DateTime())
        and is_dtype(ztype, Continuous())
    ):
        x, y, z = y, z, x
    elif (
        (is_dtype(xtype, Nominal()) or is_dtype(xtype, GeoGraphy()))
        and is_dtype(ytype, Continuous())
        and is_dtype(ztype, DateTime())
    ):
        x, z = z, x

    if not (
        is_dtype(xtype, DateTime())
        and is_dtype(ytype, Continuous())
        and (is_dtype(ztype, Nominal()) or is_dtype(ztype, GeoGraphy()))
    ):
        raise ValueError(
            "x, y, and z must be one each of type datetime, numerical, and categorical"
        )

    df = drop_null(df[[x, y, z]])
    df[z] = df[z].apply(str, meta=(z, str))

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
