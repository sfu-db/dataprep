"""
This module implements the intermediates computation for plot(df) function.
"""  # pylint: disable=too-many-lines
from sys import stderr
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm, kurtosis, skew, median_absolute_deviation

from ...errors import UnreachableError
from ..dtypes import (
    is_dtype,
    detect_dtype,
    DType,
    Nominal,
    Continuous,
    DateTime,
    DTypeDef,
)
from ..intermediate import Intermediate
from ..utils import to_dask

__all__ = ["compute"]

# Dictionary for mapping the time unit to its formatting. Each entry is of the
# form unit:(unit code for pd.Grouper freq parameter, pandas to_period strftime
# formatting for line charts, pandas to_period strftime formatting for box plot,
# label format).
DTMAP = {
    "year": ("Y", "%Y", "%Y", "Year"),
    "quarter": ("Q", "Q%q %Y", "Q%q %Y", "Quarter"),
    "month": ("M", "%B %Y", "%b %Y", "Month"),
    "week": ("W-SAT", "%d %B, %Y", "%d %b, %Y", "Week of"),
    "day": ("D", "%d %B, %Y", "%d %b, %Y", "Date"),
    "hour": ("H", "%d %B, %Y, %I %p", "%d %b, %Y, %I %p", "Hour"),
    "minute": ("T", "%d %B, %Y, %I:%M %p", "%d %b, %Y, %I:%M %p", "Minute"),
    "second": ("S", "%d %B, %Y, %I:%M:%S %p", "%d %b, %Y, %I:%M:%S %p", "Second"),
}


def compute(
    df: Union[pd.DataFrame, dd.DataFrame],
    x: Optional[str] = None,
    y: Optional[str] = None,
    z: Optional[str] = None,
    *,
    bins: int = 10,
    ngroups: int = 10,
    largest: bool = True,
    nsubgroups: int = 5,
    timeunit: str = "auto",
    agg: str = "mean",
    sample_size: int = 1000,
    value_range: Optional[Tuple[float, float]] = None,
    dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    """
    Parameters
    ----------
    df
        Dataframe from which plots are to be generated
    x: Optional[str], default None
        A valid column name from the dataframe
    y: Optional[str], default None
        A valid column name from the dataframe
    z: Optional[str], default None
        A valid column name from the dataframe
    bins: int, default 10
        For a histogram or box plot with numerical x axis, it defines
        the number of equal-width bins to use when grouping.
    ngroups: int, default 10
        When grouping over a categorical column, it defines the
        number of groups to show in the plot. Ie, the number of
        bars to show in a bar chart.
    largest: bool, default True
        If true, when grouping over a categorical column, the groups
        with the largest count will be output. If false, the groups
        with the smallest count will be output.
    nsubgroups: int, default 5
        If x and y are categorical columns, ngroups refers to
        how many groups to show from column x, and nsubgroups refers to
        how many subgroups to show from column y in each group in column x.
    timeunit: str, default "auto"
        Defines the time unit to group values over for a datetime column.
        It can be "year", "quarter", "month", "week", "day", "hour",
        "minute", "second". With default value "auto", it will use the
        time unit such that the resulting number of groups is closest to 15.
    agg: str, default "mean"
        Specify the aggregate to use when aggregating over a numeric column
    sample_size: int, default 1000
        Sample size for the scatter plot
    value_range: Optional[Tuple[float, float]], default None
        The lower and upper bounds on the range of a numerical column.
        Applies when column x is specified and column y is unspecified.
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    """

    df = to_dask(df)

    if not any((x, y, z)):
        return compute_overview(df, bins, ngroups, largest, timeunit, dtype)

    if sum(v is None for v in (x, y, z)) == 2:
        col: str = cast(str, x or y or z)
        return compute_univariate(
            df, col, bins, ngroups, largest, timeunit, value_range, dtype
        )

    if sum(v is None for v in (x, y, z)) == 1:
        x, y = (v for v in (x, y, z) if v is not None)
        return compute_bivariate(
            df,
            x,
            y,
            bins,
            ngroups,
            largest,
            nsubgroups,
            timeunit,
            agg,
            sample_size,
            dtype,
        )

    if x is not None and y is not None and z is not None:
        return compute_trivariate(df, x, y, z, ngroups, largest, timeunit, agg, dtype)

    return Intermediate()


def compute_overview(
    df: dd.DataFrame,
    bins: int,
    ngroups: int,
    largest: bool,
    timeunit: str,
    dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    # pylint: disable=too-many-arguments
    """
    Compute functions for plot(df)
    Parameters
    ----------
    df
        Dataframe from which plots are to be generated
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
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    """

    datas: List[Any] = []
    counter = {"Categorical": 0, "Numerical": 0, "Datetime": 0}
    col_names_dtypes: List[Tuple[str, DType]] = []
    for column in df.columns:
        column_dtype = detect_dtype(df[column], dtype)
        if is_dtype(column_dtype, Nominal()):
            # bar chart
            datas.append(dask.delayed(calc_bar_pie)(df[column], ngroups, largest))
            col_names_dtypes.append((column, Nominal()))
            counter["Categorical"] += 1
        elif is_dtype(column_dtype, Continuous()):
            # histogram
            datas.append(dask.delayed(calc_hist)(df[column], bins))
            col_names_dtypes.append((column, Continuous()))
            counter["Numerical"] += 1
        elif is_dtype(column_dtype, DateTime()):
            datas.append(dask.delayed(calc_line_dt)(df[[column]], timeunit))
            col_names_dtypes.append((column, DateTime()))
            counter["Datetime"] += 1
        else:
            raise UnreachableError
    datas.append(dask.delayed(calc_stats)(df, counter))
    datas = dask.compute(*datas)
    data = [(col, dtp, dat) for (col, dtp), dat in zip(col_names_dtypes, datas[:-1])]
    return Intermediate(data=data, statsdata=datas[-1], visual_type="basic_grid")


def compute_univariate(
    df: dd.DataFrame,
    x: str,
    bins: int,
    ngroups: int,
    largest: bool,
    timeunit: str,
    value_range: Optional[Tuple[float, float]],
    dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    """
    Compute functions for plot(df, x)
    Parameters
    ----------
    df
        Dataframe from which plots are to be generated
    x
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
    value_range
        The lower and upper bounds on the range of a numerical column.
        Applies when column x is specified and column y is unspecified.
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    """
    # pylint: disable=too-many-locals, too-many-arguments

    col_dtype = detect_dtype(df[x], dtype)
    if is_dtype(col_dtype, Nominal()):
        # data for bar and pie charts
        data_cat: List[Any] = []
        data_cat.append(dask.delayed(calc_bar_pie)(df[x], ngroups, largest))
        # stats
        data_cat.append(dask.delayed(calc_stats_cat)(df[x]))
        data, statsdata_cat = dask.compute(*data_cat)
        return Intermediate(
            col=x, data=data, statsdata=statsdata_cat, visual_type="categorical_column",
        )
    elif is_dtype(col_dtype, Continuous()):
        if value_range is not None:
            if (
                (value_range[0] <= np.nanmax(df[x]))
                and (value_range[1] >= np.nanmin(df[x]))
                and (value_range[0] < value_range[1])
            ):
                df = df[df[x].between(value_range[0], value_range[1])]
            else:
                print("Invalid range of values for this column", file=stderr)
        data_num: List[Any] = []
        # qq plot
        qqdata = calc_qqnorm(df[x].dropna())
        # kde plot
        kdedata = calc_hist_kde(df[x].dropna().values, bins)
        # box plot
        boxdata = calc_box(df[[x]].dropna(), bins, dtype=dtype)
        # histogram
        data_num.append(dask.delayed(calc_hist)(df[x], bins))
        # stats
        data_num.append(
            dask.delayed(calc_stats_num)(
                df[x],
                mean=qqdata[2],
                std=qqdata[3],
                min=kdedata[3],
                max=kdedata[4],
                quantile=qqdata[0],
            )
        )
        histdata, statsdata_num = dask.compute(*data_num)
        return Intermediate(
            col=x,
            histdata=histdata,
            kdedata=kdedata,
            qqdata=qqdata,
            boxdata=boxdata,
            statsdata=statsdata_num,
            visual_type="numerical_column",
        )
    elif is_dtype(col_dtype, DateTime()):
        data_dt: List[Any] = []
        # line chart
        data_dt.append(dask.delayed(calc_line_dt)(df[[x]], timeunit))
        # stats
        data_dt.append(dask.delayed(calc_stats_dt)(df[x]))
        data, statsdata_dt = dask.compute(*data_dt)
        return Intermediate(
            col=x, data=data, statsdata=statsdata_dt, visual_type="datetime_column",
        )
    else:
        raise UnreachableError


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
    """
    Compute functions for plot(df, x, y)
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
        df = df[[x, y]].dropna()
        df[x] = df[x].apply(str, meta=(x, str))
        # box plot per group
        boxdata = calc_box(df, bins, ngroups, largest, dtype)
        # histogram per group
        hisdata = calc_hist_by_group(df, bins, ngroups, largest)
        return Intermediate(
            x=x, y=y, boxdata=boxdata, histdata=hisdata, visual_type="cat_and_num_cols",
        )
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
        dtnum.append(dask.delayed(calc_line_dt)(df, timeunit, agg))
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
        dtcat.append(
            dask.delayed(calc_line_dt)(df, timeunit, ngroups=ngroups, largest=largest)
        )
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
        df = df[[x, y]].dropna()
        df[x] = df[x].apply(str, meta=(x, str))
        df[y] = df[y].apply(str, meta=(y, str))
        # nested bar chart
        nesteddata = calc_nested(df, ngroups, nsubgroups)
        # stacked bar chart
        stackdata = calc_stacked(df, ngroups, nsubgroups)
        # heat map
        heatmapdata = calc_heatmap(df, ngroups, nsubgroups)
        return Intermediate(
            x=x,
            y=y,
            nesteddata=nesteddata,
            stackdata=stackdata,
            heatmapdata=heatmapdata,
            visual_type="two_cat_cols",
        )
    elif is_dtype(xtype, Continuous()) and is_dtype(ytype, Continuous()):
        df = df[[x, y]].dropna()
        # scatter plot
        scatdata = calc_scatter(df, sample_size)
        # hexbin plot
        hexbindata = df.compute()
        # box plot
        boxdata = calc_box(df, bins)
        return Intermediate(
            x=x,
            y=y,
            scatdata=scatdata,
            boxdata=boxdata,
            hexbindata=hexbindata,
            spl_sz=sample_size,
            visual_type="two_num_cols",
        )
    else:
        raise UnreachableError


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
    """
    Compute functions for plot(df, x, y, z)
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

    if (
        is_dtype(xtype, DateTime())
        and is_dtype(ytype, Nominal())
        and is_dtype(ztype, Continuous())
    ):
        y, z = z, y
    elif (
        is_dtype(xtype, Continuous())
        and is_dtype(ytype, DateTime())
        and is_dtype(ztype, Nominal())
    ):
        x, y = y, x
    elif (
        is_dtype(xtype, Continuous())
        and is_dtype(ytype, Nominal())
        and is_dtype(ztype, DateTime())
    ):
        x, y, z = z, x, y
    elif (
        is_dtype(xtype, Nominal())
        and is_dtype(ytype, DateTime())
        and is_dtype(ztype, Continuous())
    ):
        x, y, z = y, z, x
    elif (
        is_dtype(xtype, Nominal())
        and is_dtype(ytype, Continuous())
        and is_dtype(ztype, DateTime())
    ):
        x, z = z, x
    assert (
        is_dtype(xtype, DateTime())
        and is_dtype(ytype, Continuous())
        and is_dtype(ztype, Nominal())
    ), "x, y, and z must be one each of type datetime, numerical, and categorical"
    df = df[[x, y, z]].dropna()
    df[z] = df[z].apply(str, meta=(z, str))

    # line chart
    data = dask.compute(dask.delayed(calc_line_dt)(df, timeunit, agg, ngroups, largest))
    return Intermediate(
        x=x, y=y, z=z, agg=agg, data=data[0], visual_type="dt_cat_num_cols",
    )


def calc_line_dt(
    df: dd.DataFrame,
    unit: str,
    agg: Optional[str] = None,
    ngroups: Optional[int] = None,
    largest: Optional[bool] = None,
) -> Union[
    Tuple[pd.DataFrame, Dict[str, int], str],
    Tuple[pd.DataFrame, str, float],
    Tuple[pd.DataFrame, str],
]:
    """
    Calculate a line or multiline chart with date on the x axis. If df contains
    one datetime column, it will make a line chart of the frequency of values. If
    df contains a datetime and categorical column, it will compute the frequency
    of each categorical value in each time group. If df contains a datetime and
    numerical column, it will compute the aggregate of the numerical column grouped
    by the time groups. If df contains a datetime, categorical, and numerical column,
    it will compute the aggregate of the numerical column for values in the categorical
    column grouped by time.
    Parameters
    ----------
    df
        A dataframe
    unit
        The unit of time over which to group the values
    agg
        Aggregate to use for the numerical column
    ngroups
        Number of groups for the categorical column
    largest
        Use the largest or smallest groups in the categorical column
    """
    # pylint: disable=too-many-locals

    x = df.columns[0]  # time column
    unit = _get_timeunit(df[x].min(), df[x].max(), 100) if unit == "auto" else unit
    if unit not in DTMAP.keys():
        raise ValueError
    grouper = pd.Grouper(key=x, freq=DTMAP[unit][0])  # for grouping the time values

    # multiline charts
    if ngroups and largest:
        hist_dict: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = dict()
        hist_lst: List[Tuple[np.ndarray, np.ndarray, List[str]]] = list()
        agg = (
            "freq" if agg is None else agg
        )  # default agg if unspecified for notational concision

        # categorical column for grouping over, each resulting group is a line in the chart
        grpby_col = df.columns[1] if len(df.columns) == 2 else df.columns[2]
        df, grp_cnt_stats, largest_grps = _calc_groups(df, grpby_col, ngroups, largest)
        groups = df.groupby([grpby_col])

        for grp in largest_grps:
            srs = groups.get_group(grp)
            # calculate the frequencies or aggregate value in each time group
            if len(df.columns) == 3:
                dfr = srs.groupby(grouper)[df.columns[1]].agg(agg).reset_index()
            else:
                dfr = srs[x].to_frame().groupby(grouper).size().reset_index()
            dfr.columns = [x, agg]
            # if grouping by week, make the label for the week the beginning Sunday
            dfr[x] = dfr[x] - pd.to_timedelta(6, unit="d") if unit == "week" else dfr[x]
            # format the label
            dfr["lbl"] = dfr[x].dt.to_period("S").dt.strftime(DTMAP[unit][1])
            hist_lst.append((list(dfr[agg]), list(dfr[x]), list(dfr["lbl"])))
        hist_lst = dask.compute(*hist_lst)
        for elem in zip(largest_grps, hist_lst):
            hist_dict[elem[0]] = elem[1]
        return hist_dict, grp_cnt_stats, DTMAP[unit][3]

    # single line charts
    if agg is None:  # frequency of datetime column
        miss_pct = round(df[x].isna().sum() / len(df) * 100, 1)
        dfr = df.dropna().groupby(grouper).size().reset_index()
        dfr.columns = [x, "freq"]
        dfr["pct"] = dfr["freq"] / len(df) * 100
    else:  # aggregate over a second column
        dfr = df.groupby(grouper)[df.columns[1]].agg(agg).reset_index()
        dfr.columns = [x, agg]
    dfr[x] = dfr[x] - pd.to_timedelta(6, unit="d") if unit == "week" else dfr[x]
    dfr["lbl"] = dfr[x].dt.to_period("S").dt.strftime(DTMAP[unit][1])

    return (dfr, DTMAP[unit][3], miss_pct) if agg is None else (dfr, DTMAP[unit][3])


def calc_box_dt(
    df: dd.DataFrame, unit: str
) -> Tuple[pd.DataFrame, List[str], List[float], str]:
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
    df = pd.concat([_calc_box_stats(g[1][y], g[0], True) for g in grps], axis=1,)
    df = df.append(pd.Series({c: i + 1 for i, c in enumerate(df.columns)}, name="x",)).T
    # If grouping by week, make the label for the week the beginning Sunday
    df.index = df.index - pd.to_timedelta(6, unit="d") if unit == "week" else df.index
    df.index.name = "grp"
    df = df.reset_index()
    df["grp"] = df["grp"].dt.to_period("S").dt.strftime(DTMAP[unit][2])
    df["x0"], df["x1"] = df["x"] - 0.8, df["x"] - 0.2  # width of whiskers for plotting
    outx, outy = _calc_box_otlrs(df)

    return df, outx, outy, DTMAP[unit][3]


def calc_stacked_dt(
    df: dd.DataFrame, unit: str, ngroups: int, largest: bool,
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
        df_grps, index=grouper, columns=y, aggfunc=len, fill_value=0,
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


def calc_bar_pie(
    srs: dd.Series, ngroups: int, largest: bool
) -> Tuple[pd.DataFrame, int, float]:
    """
    Calculates the group counts given a series.
    Parameters
    ----------
    srs
        One categorical column
    ngroups
        Number of groups to return
    largest
        If true, show the groups with the largest count,
        else show the groups with the smallest count
    Returns
    -------
    Tuple[pd.DataFrame, float]
        A dataframe of the group counts, the total count of groups,
        and the percent of missing values
    """
    miss_pct = round(srs.isna().sum() / len(srs) * 100, 1)
    try:
        grp_srs = srs.groupby(srs).size()
    except TypeError:
        srs = srs.astype(str)
        grp_srs = srs.groupby(srs).size()
    # select largest or smallest groups
    smp_srs = grp_srs.nlargest(n=ngroups) if largest else grp_srs.nsmallest(n=ngroups)
    df = smp_srs.to_frame().rename(columns={srs.name: "cnt"}).reset_index()
    # add a row containing the sum of the other groups
    other_cnt = len(srs) - df["cnt"].sum()
    df = df.append(pd.DataFrame({srs.name: ["Others"], "cnt": [other_cnt]}))
    # add a column containing the percent of count in each group
    df["pct"] = df["cnt"] / len(srs) * 100
    df.columns = ["col", "cnt", "pct"]
    df["col"] = df["col"].astype(str)  # needed when numeric is cast as categorical
    return df, len(grp_srs), miss_pct


def calc_hist(srs: dd.Series, bins: int,) -> Tuple[pd.DataFrame, float]:
    """
    Calculate a histogram over a given series.
    Parameters
    ----------
    srs
        One numerical column over which to compute the histogram
    bins
        Number of bins to use in the histogram
    Returns
    -------
    Tuple[pd.DataFrame, float]:
        The histogram in a dataframe and the percent of missing values
    """
    miss_pct = round(srs.isna().sum() / len(srs) * 100, 1)
    data = srs.dropna().values
    if len(data) == 0:  # all values in column are missing
        return pd.DataFrame({"left": [], "right": [], "freq": []}), miss_pct
    hist_arr, bins_arr = np.histogram(data, range=[data.min(), data.max()], bins=bins)
    intvls = _format_bin_intervals(bins_arr)
    hist_df = pd.DataFrame(
        {
            "intvls": intvls,
            "left": bins_arr[:-1],
            "right": bins_arr[1:],
            "freq": hist_arr,
            "pct": hist_arr / len(srs) * 100,
        }
    )
    return hist_df, miss_pct


def calc_hist_kde(
    data: da.Array, bins: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, float, float]:
    """
    Calculate a density histogram and its corresponding kernel density
    estimate over a given series. The kernel is guassian.
    Parameters
    ----------
    data
        One numerical column over which to compute the histogram and kde
    bins
        Number of bins to use in the histogram
    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray, np.ndarray]
        The histogram in a dataframe, range of points for the kde,
        and the kde calculated at the specified points
    """
    minv, maxv = dask.compute(data.min(), data.max())
    hist_arr, bins_arr = da.histogram(data, range=[minv, maxv], bins=bins, density=True)
    hist_arr = hist_arr.compute()
    intervals = _format_bin_intervals(bins_arr)
    hist_df = pd.DataFrame(
        {
            "intervals": intervals,
            "left": bins_arr[:-1],
            "right": bins_arr[1:],
            "freq": hist_arr,
        }
    )
    pts_rng = np.linspace(minv, maxv, 1000)
    if minv == maxv:
        pdf = np.zeros(1000, dtype=np.float64)
    else:
        pdf = gaussian_kde(data.compute())(pts_rng)
    return hist_df, pts_rng, pdf, minv, maxv


def calc_qqnorm(srs: dd.Series) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Calculate QQ plot given a series.
    Parameters
    ----------
    srs
        One numerical column from which to compute the quantiles
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of (actual quantiles, theoretical quantiles)
    """
    q_range = np.linspace(0.01, 0.99, 100)
    actual_qs, mean, std = dask.compute(srs.quantile(q_range), srs.mean(), srs.std())
    theory_qs = np.sort(np.asarray(norm.ppf(q_range, mean, std)))
    return actual_qs, theory_qs, mean, std


def calc_box(
    df: dd.DataFrame,
    bins: int,
    ngroups: int = 10,
    largest: bool = True,
    dtype: Optional[DTypeDef] = None,
) -> Tuple[pd.DataFrame, List[str], List[float], Optional[Dict[str, int]]]:
    """
    Compute a box plot over either
        1) the values in one column
        2) the values corresponding to groups in another column
        3) the values corresponding to binning another column
    Parameters
    ----------
    df
        Dataframe with one or two columns
    bins
        Number of bins to use if df has two numerical columns
    ngroups
        Number of groups to show if df has a categorical and numerical column
    largest
        When calculating a box plot per group, select the largest or smallest groups
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    Returns
    -------
    Tuple[pd.DataFrame, List[str], List[float], Dict[str, int]]
        The box plot statistics in a dataframe, a list of the outlier
        groups and another list of the outlier values, a dictionary
        logging the sampled group output
    """
    # pylint: disable=too-many-locals
    grp_cnt_stats = None  # to inform the user of sampled output

    x = df.columns[0]
    if len(df.columns) == 1:
        df = _calc_box_stats(df[x], x)
    else:
        y = df.columns[1]
        if is_dtype(detect_dtype(df[x], dtype), Continuous()) and is_dtype(
            detect_dtype(df[y], dtype), Continuous()
        ):
            minv, maxv, cnt = dask.compute(df[x].min(), df[x].max(), df[x].nunique())
            bins = cnt if cnt < bins else bins
            endpts = np.linspace(minv, maxv, num=bins + 1)
            # calculate a box plot over each bin
            df = dd.concat(
                [
                    _calc_box_stats(
                        df[(df[x] >= endpts[i]) & (df[x] < endpts[i + 1])][y],
                        f"[{endpts[i]},{endpts[i + 1]})",
                    )
                    if i != len(endpts) - 2
                    else _calc_box_stats(
                        df[(df[x] >= endpts[i]) & (df[x] <= endpts[i + 1])][y],
                        f"[{endpts[i]},{endpts[i + 1]}]",
                    )
                    for i in range(len(endpts) - 1)
                ],
                axis=1,
            ).compute()
            endpts_df = pd.DataFrame(
                [endpts[:-1], endpts[1:]], ["lb", "ub"], df.columns
            )
            df = pd.concat([df, endpts_df], axis=0)
        else:
            df, grp_cnt_stats, largest_grps = _calc_groups(df, x, ngroups, largest)
            # calculate a box plot over each group
            df = dd.concat(
                [_calc_box_stats(df[df[x] == grp][y], grp) for grp in largest_grps],
                axis=1,
            ).compute()

    df = df.append(pd.Series({c: i + 1 for i, c in enumerate(df.columns)}, name="x",)).T
    df.index.name = "grp"
    df = df.reset_index()
    df["x0"], df["x1"] = df["x"] - 0.8, df["x"] - 0.2  # width of whiskers for plotting
    outx, outy = _calc_box_otlrs(df)

    return df, outx, outy, grp_cnt_stats


def calc_hist_by_group(
    df: dd.DataFrame, bins: int, ngroups: int, largest: bool
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Compute a histogram over the values corresponding to the groups in another column
    Parameters
    ----------
    df
        Dataframe with one categorical and one numerical column
    bins
        Number of bins to use in the histogram
    ngroups
        Number of groups to show from the categorical column
    largest
        Select the largest or smallest groups
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        The histograms in a dataframe and a dictionary
        logging the sampled group output
    """
    # pylint: disable=too-many-locals

    hist_dict: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = dict()
    hist_lst: List[Tuple[np.ndarray, np.ndarray, List[str]]] = list()
    df, grp_cnt_stats, largest_grps = _calc_groups(df, df.columns[0], ngroups, largest)

    # create a histogram for each group
    groups = df.groupby([df.columns[0]])
    minv, maxv = dask.compute(df[df.columns[1]].min(), df[df.columns[1]].max())
    for grp in largest_grps:
        grp_srs = groups.get_group(grp)[df.columns[1]]
        hist_arr, bins_arr = da.histogram(grp_srs, range=[minv, maxv], bins=bins)
        intervals = _format_bin_intervals(bins_arr)
        hist_lst.append((hist_arr, bins_arr, intervals))

    hist_lst = dask.compute(*hist_lst)

    for elem in zip(largest_grps, hist_lst):
        hist_dict[elem[0]] = elem[1]

    return hist_dict, grp_cnt_stats


def calc_scatter(df: dd.DataFrame, sample_size: int) -> pd.DataFrame:
    """
    Extracts the points to use in a scatter plot
    Parameters
    ----------
    df
        Dataframe with two numerical columns
    sample_size
        the number of points to randomly sample in the scatter plot
    Returns
    -------
    pd.DataFrame
        A dataframe containing the scatter points
    """
    if len(df) > sample_size:
        df = df.sample(frac=sample_size / len(df))
    return df.compute()


def calc_nested(
    df: dd.DataFrame, ngroups: int, nsubgroups: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Calculate a nested bar chart of the counts of two columns
    Parameters
    ----------
    df
        Dataframe with two categorical columns
    ngroups
        Number of groups to show from the first column
    nsubgroups
        Number of subgroups (from the second column) to show in each group
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        The bar chart counts in a dataframe and a dictionary
        logging the sampled group output
    """
    x, y = df.columns[0], df.columns[1]
    df, grp_cnt_stats, _ = _calc_groups(df, x, ngroups)

    df2 = df.groupby([x, y]).size().reset_index()
    max_subcol_cnt = df2.groupby(x).size().max().compute()
    df2.columns = [x, y, "cnt"]
    df_res = (
        df2.groupby(x)[[y, "cnt"]]
        .apply(
            lambda x: x.nlargest(n=nsubgroups, columns="cnt"),
            meta=({y: "f8", "cnt": "i8"}),
        )
        .reset_index()
        .compute()
    )
    df_res["grp_names"] = list(zip(df_res[x], df_res[y]))
    df_res = df_res.drop([x, "level_1", y], axis=1)
    grp_cnt_stats[f"{y}_ttl"] = max_subcol_cnt
    grp_cnt_stats[f"{y}_shw"] = min(max_subcol_cnt, nsubgroups)

    return df_res, grp_cnt_stats


def calc_stacked(
    df: dd.DataFrame, ngroups: int, nsubgroups: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Calculate a stacked bar chart of the counts of two columns
    Parameters
    ----------
    df
        two categorical columns
    ngroups
        number of groups to show from the first column
    nsubgroups
        number of subgroups (from the second column) to show in each group
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        The bar chart counts in a dataframe and a dictionary
        logging the sampled group output
    """
    x, y = df.columns[0], df.columns[1]
    df, grp_cnt_stats, largest_grps = _calc_groups(df, x, ngroups)

    fin_df = pd.DataFrame()
    for grp in largest_grps:
        df_grp = df[df[x] == grp]
        df_res = df_grp.groupby(y).size().nlargest(n=nsubgroups) / len(df_grp) * 100
        df_res = df_res.to_frame().compute().T
        df_res.columns = list(df_res.columns)
        df_res["Others"] = 100 - df_res.sum(axis=1)
        fin_df = fin_df.append(df_res, sort=False)

    fin_df = fin_df.fillna(value=0)
    others = fin_df.pop("Others")
    if others.sum() > 1e-4:
        fin_df["Others"] = others
    fin_df.index = list(largest_grps)
    return fin_df, grp_cnt_stats


def calc_heatmap(
    df: dd.DataFrame, ngroups: int, nsubgroups: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Calculate a heatmap of the counts of two columns
    Parameters
    ----------
    df
        Dataframe with two categorical columns
    ngroups
        Number of groups to show from the first column
    nsubgroups
        Number of subgroups (from the second column) to show in each group
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        The heatmap counts in a dataframe and a dictionary
        logging the sampled group output
    """
    x, y = df.columns[0], df.columns[1]
    df, grp_cnt_stats, _ = _calc_groups(df, x, ngroups)

    srs = df.groupby(y).size()
    srs_lrgst = srs.nlargest(n=nsubgroups)
    largest_subgrps = list(srs_lrgst.index.compute())
    df = df[df[y].isin(largest_subgrps)]

    df_res = df.groupby([x, y]).size().reset_index().compute()
    df_res.columns = ["x", "y", "cnt"]
    df_res = pd.pivot_table(
        df_res, index=["x", "y"], values="cnt", fill_value=0, aggfunc=np.sum,
    ).reset_index()

    grp_cnt_stats[f"{y}_ttl"] = len(srs.index.compute())
    grp_cnt_stats[f"{y}_shw"] = len(largest_subgrps)

    return df_res, grp_cnt_stats


def calc_stats_num(
    srs: dd.Series, **kwargs: Any,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Calculate stats from a numerical column
    Parameters
    ----------
    srs
        a numerical column
    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]
        Dictionaries that contain Overview, Quantiles and Descriptives
    """
    size = len(srs)  # include nan
    count = srs.count()  # exclude nan
    uniq_count = srs.nunique()
    inf_count = ((srs == np.inf) | (srs == -np.inf)).sum()
    non_zero_count = np.count_nonzero(srs)
    overview_dict = {
        "Distinct Count": uniq_count,
        "Unique (%)": uniq_count / count,
        "Missing": size - count,
        "Missing (%)": 1 - (count / size),
        "Infinite": inf_count,
        "Infinite (%)": inf_count / size,
        "Mean": kwargs["mean"],
        "Minimum": kwargs["min"],
        "Maximum": kwargs["max"],
        "Zeros": size - non_zero_count,
        "Zeros (%)": 1 - non_zero_count / size,
        "Memory Size": srs.memory_usage(),
    }
    quantiles_dict = {
        "Minimum": kwargs["min"],
        "5-th Percentile": kwargs["quantile"].iloc[5],
        "Q1": kwargs["quantile"].iloc[25],
        "Median": kwargs["quantile"].iloc[50],
        "Q3": kwargs["quantile"].iloc[75],
        "95-th Percentile": kwargs["quantile"].iloc[95],
        "Maximum": kwargs["max"],
        "Range": kwargs["max"] - kwargs["min"],
        "IQR": kwargs["quantile"].iloc[75] - kwargs["quantile"].iloc[25],
    }
    descriptives_dict = {
        "Standard Deviation": kwargs["std"],
        "Coefficient of Variation": kwargs["std"] / kwargs["mean"],
        "Kurtosis": float(kurtosis(srs, nan_policy="omit")),
        "Mean": kwargs["mean"],
        "Median Absolute Deviation": float(
            median_absolute_deviation(srs, nan_policy="omit")
        ),
        "Skewness": float(skew(srs, nan_policy="omit")),
        "Sum": kwargs["mean"] * count,
        "Variance": kwargs["std"] ** 2,
    }
    return (
        {k: _format_values(k, v) for k, v in overview_dict.items()},
        {k: _format_values(k, v) for k, v in quantiles_dict.items()},
        {k: _format_values(k, v) for k, v in descriptives_dict.items()},
    )


def calc_stats_cat(srs: dd.Series) -> Dict[str, str]:
    """
    Calculate stats from a categorical column
    Parameters
    ----------
    srs
        a categorical column
    Returns
    -------
    Dict[str, str]
        Dictionary that contains Overview
    """
    size = len(srs)  # include nan
    count = srs.count()  # exclude nan
    uniq_count = srs.nunique()
    overview_dict = {
        "Distinct Count": uniq_count,
        "Unique (%)": uniq_count / count,
        "Missing": size - count,
        "Missing (%)": 1 - (count / size),
        "Memory Size": srs.memory_usage(),
    }

    return {k: _format_values(k, v) for k, v in overview_dict.items()}


def calc_stats_dt(srs: dd.Series) -> Dict[str, str]:
    """
    Calculate stats from a datetime column
    Parameters
    ----------
    srs
        a datetime column
    Returns
    -------
    Dict[str, str]
        Dictionary that contains Overview
    """
    size = len(srs)  # include nan
    count = srs.count()  # exclude nan
    uniq_count = srs.nunique()
    overview_dict = {
        "Distinct Count": uniq_count,
        "Unique (%)": uniq_count / count,
        "Missing": size - count,
        "Missing (%)": 1 - (count / size),
        "Memory Size": srs.memory_usage(),
        "Minimum": srs.min(),
        "Maximum": srs.max(),
    }

    return {k: _format_values(k, v) for k, v in overview_dict.items()}


def calc_stats(
    df: dd.DataFrame, counter: Dict[str, int]
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Calculate stats from a DataFrame

    Parameters
    ----------
    df
        a DataFrame
    counter
        a dictionary that contains count for each type
    Returns
    -------
    Tuple[Dict[str, str], Dict[str, int]]
        Dictionary that contains Overview and Variable Types
    """
    dim = df.shape
    total_cell = dim[0] * dim[1]
    nonan_cell = df.count().sum()
    memory_usage = float(df.memory_usage().sum())
    try:  # for unhashable data types
        dup_rows = len(df.drop_duplicates())
    except TypeError:
        df = df.astype(str)
        dup_rows = len(df.drop_duplicates())
    overview_dict = {
        "Number of Variables": dim[1],
        "Number of Observations": dim[0],
        "Missing Cells": float(total_cell - nonan_cell),
        "Missing Cells (%)": 1 - (nonan_cell / total_cell),
        "Duplicate Rows": dim[0] - dup_rows,
        "Duplicate Rows (%)": 1 - (dup_rows / dim[0]),
        "Total Size in Memory": memory_usage,
        "Average Record Size in Memory": memory_usage / dim[0],
    }

    return (
        {k: _format_values(k, v) for k, v in overview_dict.items()},
        counter,
    )


def _calc_box_stats(grp_srs: dd.Series, grp: str, dlyd: bool = False) -> pd.DataFrame:
    """
    Auxiliary function to calculate the Tukey box plot statistics
    dlyd is for if this function is called when dask is computing in parallel (dask.delayed)
    """
    stats: Dict[str, Any] = dict()

    try:  # this is a bad fix for the problem of when there is no data passed to this function
        if dlyd:
            qntls = np.round(grp_srs.quantile([0.25, 0.50, 0.75]), 3)
        else:
            qntls = np.round(grp_srs.quantile([0.25, 0.50, 0.75]).compute(), 3)
        stats["q1"], stats["q2"], stats["q3"] = qntls[0.25], qntls[0.50], qntls[0.75]
    except ValueError:
        stats["q1"], stats["q2"], stats["q3"] = np.nan, np.nan, np.nan

    iqr = stats["q3"] - stats["q1"]
    stats["lw"] = grp_srs[grp_srs >= stats["q1"] - 1.5 * iqr].min()
    stats["uw"] = grp_srs[grp_srs <= stats["q3"] + 1.5 * iqr].max()
    if not dlyd:
        stats["lw"], stats["uw"] = dask.compute(stats["lw"], stats["uw"])

    otlrs = grp_srs[(grp_srs < stats["lw"]) | (grp_srs > stats["uw"])]
    if len(otlrs) > 100:  # sample 100 outliers
        otlrs = otlrs.sample(frac=100 / len(otlrs))
    stats["otlrs"] = list(otlrs) if dlyd else list(otlrs.compute())

    return pd.DataFrame({grp: stats})


def _calc_box_otlrs(df: dd.DataFrame) -> Tuple[List[str], List[float]]:
    """
    Calculate the outliers for a box plot
    """
    outx: List[str] = []  # list for the outlier groups
    outy: List[float] = []  # list for the outlier values
    for ind in df.index:
        otlrs = df.loc[ind]["otlrs"]
        outx = outx + [df.loc[ind]["grp"]] * len(otlrs)
        outy = outy + otlrs

    return outx, outy


def _calc_groups(
    df: dd.DataFrame, x: str, ngroups: int, largest: bool = True
) -> Tuple[dd.DataFrame, Dict[str, int], List[str]]:
    """
    Auxillary function to parse the dataframe to consist of only the
    groups with the largest counts
    """

    # group count statistics to inform the user of the sampled output
    grp_cnt_stats: Dict[str, int] = dict()

    srs = df.groupby(x).size()
    srs_lrgst = srs.nlargest(n=ngroups) if largest else srs.nsmallest(n=ngroups)
    try:
        largest_grps = list(srs_lrgst.index.compute())
        grp_cnt_stats[f"{x}_ttl"] = len(srs.index.compute())
    except AttributeError:
        largest_grps = list(srs_lrgst.index)
        grp_cnt_stats[f"{x}_ttl"] = len(srs.index)

    df = df[df[x].isin(largest_grps)]
    grp_cnt_stats[f"{x}_shw"] = len(largest_grps)

    return df, grp_cnt_stats, largest_grps


def _format_bin_intervals(bins_arr: np.ndarray) -> List[str]:
    """
    Auxillary function to format bin intervals in a histogram
    """
    bins_arr = np.round(bins_arr, 3)
    bins_arr = [int(val) if val.is_integer() else val for val in bins_arr]
    intervals = [
        f"[{bins_arr[i]}, {bins_arr[i + 1]})" for i in range(len(bins_arr) - 2)
    ]
    intervals.append(f"[{bins_arr[-2]},{bins_arr[-1]}]")
    return intervals


def _get_timeunit(min_time: pd.Timestamp, max_time: pd.Timestamp, dflt: int) -> str:
    """
    Auxillary function to find an appropriate time unit. Will find the
    time unit such that the number of time units are closest to dflt.
    """
    dt_secs = {
        "year": 60 * 60 * 24 * 365,
        "quarter": 60 * 60 * 24 * 91,
        "month": 60 * 60 * 24 * 30,
        "week": 60 * 60 * 24 * 7,
        "day": 60 * 60 * 24,
        "hour": 60 * 60,
        "minute": 60,
        "second": 1,
    }

    time_rng_secs = (max_time - min_time).total_seconds()
    prev_bin_cnt, prev_unit = 0, "year"
    for unit, secs_in_unit in dt_secs.items():
        cur_bin_cnt = time_rng_secs / secs_in_unit
        if abs(prev_bin_cnt - dflt) < abs(cur_bin_cnt - dflt):
            return prev_unit
        prev_bin_cnt = cur_bin_cnt
        prev_unit = unit

    return prev_unit


def _format_values(key: str, value: Any) -> str:
    if not isinstance(value, (int, float)):
        # if value is a time
        return str(value)
    if "Memory" in key:
        # for memory usage
        ind = 0
        unit = dict(enumerate(["B", "KB", "MB", "GB", "TB"], 0))
        while value > 1024:
            value /= 1024
            ind += 1
        return f"{value:.1f} {unit[ind]}"

    if (value * 10) % 10 == 0:
        # if value is int but in a float form with 0 at last digit
        value = int(value)
        if abs(value) >= 1000000:
            return f"{value:.5g}"
    elif abs(value) >= 1000000 or abs(value) < 0.001:
        value = f"{value:.5g}"
    elif abs(value) >= 1:
        # eliminate trailing zeros
        pre_value = float(f"{value:.4f}")
        value = int(pre_value) if (pre_value * 10) % 10 == 0 else pre_value
    elif 0.001 <= abs(value) < 1:
        value = f"{value:.4g}"
    else:
        value = str(value)

    if "%" in key:
        # for percentage, only use digits before notation sign for extreme small number
        value = f"{float(value):.1%}"
    return str(value)
