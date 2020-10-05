"""Computations for plot(df, x, y)."""
from typing import Any, Dict, List, Optional, Tuple, Union

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

from ....errors import UnreachableError
from ...intermediate import Intermediate
from ...dtypes import (
    Continuous,
    DateTime,
    DTypeDef,
    Nominal,
    detect_dtype,
    is_dtype,
)
from .common import (
    DTMAP,
    _get_timeunit,
    _calc_line_dt,
    _calc_groups,
    _calc_box_otlrs,
    _calc_box_stats,
)


def compute_bivariate(
    df: dd.DataFrame,
    x: str,
    y: str,
    bins: int,
    ngroups: int,
    largest: bool,
    nsubgroups: int,
    timeunit: str,
    agg: str,
    sample_size: int,
    dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    """Compute functions for plot(df, x, y).

    Parameters
    ----------
    df
        Dataframe from which plots are to be generated
    x
        A valid column name from the dataframe
    y
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
    nsubgroups
        If x and y are categorical columns, ngroups refers to
        how many groups to show from column x, and nsubgroups refers to
        how many subgroups to show from column y in each group in column x.
    timeunit
        Defines the time unit to group values over for a datetime column.
        It can be "year", "quarter", "month", "week", "day", "hour",
        "minute", "second". With default value "auto", it will use the
        time unit such that the resulting number of groups is closest to 15.
    agg
        Specify the aggregate to use when aggregating over a numeric column
    sample_size
        Sample size for the scatter plot
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    """
    # pylint: disable=too-many-arguments,too-many-locals

    xtype = detect_dtype(df[x], dtype)
    ytype = detect_dtype(df[y], dtype)
    if (
        is_dtype(xtype, Nominal())
        and is_dtype(ytype, Continuous())
        or is_dtype(xtype, Continuous())
        and is_dtype(ytype, Nominal())
    ):
        x, y = (x, y) if is_dtype(xtype, Nominal()) else (y, x)
        df = df[[x, y]]
        first_rows = df.head()
        try:
            first_rows[x].apply(hash)
        except TypeError:
            df[x] = df[x].astype(str)

        (comps,) = dask.compute(nom_cont_comps(df.dropna(), bins, ngroups, largest))

        return Intermediate(x=x, y=y, data=comps, ngroups=ngroups, visual_type="cat_and_num_cols")
    elif (
        is_dtype(xtype, DateTime())
        and is_dtype(ytype, Continuous())
        or is_dtype(xtype, Continuous())
        and is_dtype(ytype, DateTime())
    ):
        x, y = (x, y) if is_dtype(xtype, DateTime()) else (y, x)
        df = df[[x, y]].dropna()
        dtnum: List[Any] = []
        # line chart
        dtnum.append(dask.delayed(_calc_line_dt)(df, timeunit, agg))
        # box plot
        dtnum.append(dask.delayed(calc_box_dt)(df, timeunit))
        dtnum = dask.compute(*dtnum)
        return Intermediate(
            x=x,
            y=y,
            linedata=dtnum[0],
            boxdata=dtnum[1],
            visual_type="dt_and_num_cols",
        )
    elif (
        is_dtype(xtype, DateTime())
        and is_dtype(ytype, Nominal())
        or is_dtype(xtype, Nominal())
        and is_dtype(ytype, DateTime())
    ):
        x, y = (x, y) if is_dtype(xtype, DateTime()) else (y, x)
        df = df[[x, y]].dropna()
        df[y] = df[y].apply(str, meta=(y, str))
        dtcat: List[Any] = []
        # line chart
        dtcat.append(dask.delayed(_calc_line_dt)(df, timeunit, ngroups=ngroups, largest=largest))
        # stacked bar chart
        dtcat.append(dask.delayed(calc_stacked_dt)(df, timeunit, ngroups, largest))
        dtcat = dask.compute(*dtcat)
        return Intermediate(
            x=x,
            y=y,
            linedata=dtcat[0],
            stackdata=dtcat[1],
            visual_type="dt_and_cat_cols",
        )
    elif is_dtype(xtype, Nominal()) and is_dtype(ytype, Nominal()):
        df = df[[x, y]]
        first_rows = df.head()
        try:
            first_rows[x].apply(hash)
        except TypeError:
            df[x] = df[x].astype(str)
        try:
            first_rows[y].apply(hash)
        except TypeError:
            df[y] = df[y].astype(str)

        (comps,) = dask.compute(df.dropna().groupby([x, y]).size())

        return Intermediate(
            x=x,
            y=y,
            data=comps,
            ngroups=ngroups,
            nsubgroups=nsubgroups,
            visual_type="two_cat_cols",
        )
    elif is_dtype(xtype, Continuous()) and is_dtype(ytype, Continuous()):
        # one partition required for apply(pd.cut) in calc_box_num
        df = df[[x, y]].dropna().repartition(npartitions=1)

        data: Dict[str, Any] = {}
        # scatter plot data
        data["scat"] = df.map_partitions(lambda x: x.sample(min(100, x.shape[0])), meta=df)
        # hexbin plot data
        data["hex"] = df
        # box plot
        data["box"] = calc_box_num(df, bins)

        (data,) = dask.compute(data)

        return Intermediate(
            x=x,
            y=y,
            data=data,
            spl_sz=sample_size,
            visual_type="two_num_cols",
        )
    else:
        raise UnreachableError


def nom_cont_comps(df: dd.DataFrame, bins: int, ngroups: int, largest: bool) -> Dict[str, Any]:
    """
    Computations for a nominal and continuous column

    Parameters
    ----------
    df
        Dask dataframe with one categorical and one numerical column
    bins
        Number of bins to use in the histogram
    ngroups
        Number of groups to show from the categorical column
    largest
        Select the largest or smallest groups
    """
    data: Dict[str, Any] = {}

    x, y = df.columns[0], df.columns[1]

    # filter the dataframe to consist of ngroup groups
    # https://stackoverflow.com/questions/46927174/filtering-grouped-df-in-dask
    cnts = df[x].value_counts(sort=False)
    data["ttl_grps"] = cnts.shape[0]
    thresh = cnts.nlargest(ngroups).min() if largest else cnts.nsmallest(ngroups).max()
    df = df[df[x].map(cnts) >= thresh] if largest else df[df[x].map(cnts) <= thresh]

    # group the data to compute a box plot and histogram for each group
    grps = df.groupby(x)[y]
    data["box"] = grps.apply(box_comps, meta="object")

    minv, maxv = df[y].min(), df[y].max()
    # TODO when are minv and maxv computed? This may not be optimal if
    # minv and maxv are computed ngroups times for each histogram
    data["hist"] = grps.apply(hist, bins, minv, maxv, meta="object")

    return data


def calc_box_num(df: dd.DataFrame, bins: int) -> dd.Series:
    """
    Box plot for a binned numerical variable

    Parameters
    ----------
    df
        dask dataframe
    bins
        number of bins to compute a box plot
    """
    x, y = df.columns[0], df.columns[1]
    # group the data into intervals
    # https://stackoverflow.com/questions/42442043/how-to-use-pandas-cut-or-equivalent-in-dask-efficiently
    df["grp"] = df[x].map_partitions(pd.cut, bins=bins, include_lowest=True)
    # TODO is this calculating the box plot stats for each group in parallel?
    # https://examples.dask.org/dataframes/02-groupby.html#Groupby-Apply
    # https://github.com/dask/dask/issues/4239
    # https://github.com/dask/dask/issues/5124
    srs = df.groupby("grp")[y].apply(box_comps, meta="object")

    return srs


def box_comps(srs: pd.Series) -> Dict[str, Union[float, np.array]]:
    """
    Box plot computations

    Parameters
    ----------
    srs
        pandas series
    """
    data: Dict[str, Any] = {}

    # quartiles
    data.update(zip(("q1", "q2", "q3"), srs.quantile([0.25, 0.5, 0.75])))
    iqr = data["q3"] - data["q1"]
    # inliers
    srs_iqr = srs[srs.between(data["q1"] - 1.5 * iqr, data["q3"] + 1.5 * iqr)]
    data["lw"], data["uw"] = srs_iqr.min(), srs_iqr.max()
    # outliers
    otlrs = srs[~srs.between(data["q1"] - 1.5 * iqr, data["q3"] + 1.5 * iqr)]
    # randomly sample at most 100 outliers
    data["otlrs"] = otlrs.sample(min(100, otlrs.shape[0])).values

    return data


def hist(srs: pd.Series, bins: int, minv: float, maxv: float) -> Any:
    """
    Compute a histogram on a given series

    Parameters
    ----------
    srs
        pandas Series of values for the histogram
    bins
        number of bins
    minv
        lowest bin endpoint
    maxv
        highest bin endpoint
    """

    return np.histogram(srs, bins=bins, range=[minv, maxv])


def calc_box_dt(df: dd.DataFrame, unit: str) -> Tuple[pd.DataFrame, List[str], List[float], str]:
    """
    Calculate a box plot with date on the x axis.
    Parameters
    ----------
    df
        A dataframe with one datetime and one numerical column
    unit
        The unit of time over which to group the values
    """

    x, y = df.columns[0], df.columns[1]  # time column
    unit = _get_timeunit(df[x].min(), df[x].max(), 10) if unit == "auto" else unit
    if unit not in DTMAP.keys():
        raise ValueError
    grps = df.groupby(pd.Grouper(key=x, freq=DTMAP[unit][0]))  # time groups
    # box plot for the values in each time group
    df = pd.concat(
        [_calc_box_stats(g[1][y], g[0], True) for g in grps],
        axis=1,
    )
    df = df.append(
        pd.Series(
            {c: i + 1 for i, c in enumerate(df.columns)},
            name="x",
        )
    ).T
    # If grouping by week, make the label for the week the beginning Sunday
    df.index = df.index - pd.to_timedelta(6, unit="d") if unit == "week" else df.index
    df.index.name = "grp"
    df = df.reset_index()
    df["grp"] = df["grp"].dt.to_period("S").dt.strftime(DTMAP[unit][2])
    df["x0"], df["x1"] = df["x"] - 0.8, df["x"] - 0.2  # width of whiskers for plotting
    outx, outy = _calc_box_otlrs(df)

    return df, outx, outy, DTMAP[unit][3]


def calc_stacked_dt(
    df: dd.DataFrame,
    unit: str,
    ngroups: int,
    largest: bool,
) -> Tuple[pd.DataFrame, Dict[str, int], str]:
    """
    Calculate a stacked bar chart with date on the x axis
    Parameters
    ----------
    df
        A dataframe with one datetime and one categorical column
    unit
        The unit of time over which to group the values
    ngroups
        Number of groups for the categorical column
    largest
        Use the largest or smallest groups in the categorical column
    """
    # pylint: disable=too-many-locals

    x, y = df.columns[0], df.columns[1]  # time column
    unit = _get_timeunit(df[x].min(), df[x].max(), 10) if unit == "auto" else unit
    if unit not in DTMAP.keys():
        raise ValueError

    # get the largest groups
    df_grps, grp_cnt_stats, _ = _calc_groups(df, y, ngroups, largest)
    grouper = (pd.Grouper(key=x, freq=DTMAP[unit][0]),)  # time grouper
    # pivot table of counts with date groups as index and categorical values as column names
    dfr = pd.pivot_table(
        df_grps,
        index=grouper,
        columns=y,
        aggfunc=len,
        fill_value=0,
    ).rename_axis(None)

    # if more than ngroups categorical values, aggregate the smallest groups into "Others"
    if grp_cnt_stats[f"{y}_ttl"] > grp_cnt_stats[f"{y}_shw"]:
        grp_cnts = df.groupby(pd.Grouper(key=x, freq=DTMAP[unit][0])).size()
        dfr["Others"] = grp_cnts - dfr.sum(axis=1)

    dfr.index = (  # If grouping by week, make the label for the week the beginning Sunday
        dfr.index - pd.to_timedelta(6, unit="d") if unit == "week" else dfr.index
    )
    dfr.index = dfr.index.to_period("S").strftime(DTMAP[unit][2])  # format labels

    return dfr, grp_cnt_stats, DTMAP[unit][3]
