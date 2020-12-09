"""Computations for plot(df, x, y, z)."""

from typing import Optional

import dask
import dask.dataframe as dd
from ...dtypes import (
    Continuous,
    DateTime,
    DTypeDef,
    Nominal,
    detect_dtype,
    drop_null,
    is_dtype,
)
from ...intermediate import Intermediate
from .common import _calc_line_dt


def compute_trivariate(
    df: dd.DataFrame,
    x: str,
    y: str,
    z: str,
    ngroups: int,
    largest: bool,
    timeunit: str,
    agg: str,
    dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    """Compute functions for plot(df, x, y, z).

    Parameters
    ----------
    df
        Dataframe from which plots are to be generated
    x
        A valid column name from the dataframe
    y
        A valid column name from the dataframe
    z
        A valid column name from the dataframe
    bins
        For a histogram or box plot with numerical x axis, it defines
        the number of equal-width bins to use when grouping.
    ngroups
        When grouping over a categorical column, it defines the
        number of groups to show in the plot. Ie, the number of
        bars to show in a bar chart.
    largest
        If true, when grouping over a categorical column, the groups
        with the largest count will be output. If false, the groups
        with the smallest count will be output.
    timeunit
        Defines the time unit to group values over for a datetime column.
        It can be "year", "quarter", "month", "week", "day", "hour",
        "minute", "second". With default value "auto", it will use the
        time unit such that the resulting number of groups is closest to 15.
    agg
        Specify the aggregate to use when aggregating over a numeric column
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

    if is_dtype(xtype, DateTime()) and is_dtype(ytype, Nominal()) and is_dtype(ztype, Continuous()):
        y, z = z, y
    elif (
        is_dtype(xtype, Continuous()) and is_dtype(ytype, DateTime()) and is_dtype(ztype, Nominal())
    ):
        x, y = y, x
    elif (
        is_dtype(xtype, Continuous()) and is_dtype(ytype, Nominal()) and is_dtype(ztype, DateTime())
    ):
        x, y, z = z, x, y
    elif (
        is_dtype(xtype, Nominal()) and is_dtype(ytype, DateTime()) and is_dtype(ztype, Continuous())
    ):
        x, y, z = y, z, x
    elif (
        is_dtype(xtype, Nominal()) and is_dtype(ytype, Continuous()) and is_dtype(ztype, DateTime())
    ):
        x, z = z, x

    if not (
        is_dtype(xtype, DateTime()) and is_dtype(ytype, Continuous()) and is_dtype(ztype, Nominal())
    ):
        raise ValueError(
            "x, y, and z must be one each of type datetime, numerical, and categorical"
        )

    df = drop_null(df[[x, y, z]])
    df[z] = df[z].apply(str, meta=(z, str))

    # line chart
    data = dask.compute(dask.delayed(_calc_line_dt)(df, timeunit, agg, ngroups, largest))
    return Intermediate(
        x=x,
        y=y,
        z=z,
        agg=agg,
        data=data[0],
        visual_type="dt_cat_num_cols",
    )
