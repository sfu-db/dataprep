"""Computations for plot(df, x, y)."""
from typing import Any, Dict, List, Optional, Tuple

import dask
import dask.array as da
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
    drop_null,
    is_dtype,
)
from ...utils import get_intervals
from .common import (
    DTMAP,
    _get_timeunit,
    calc_box,
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
        df = drop_null(df[[x, y]])
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
        df = drop_null(df[[x, y]])
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
        df = drop_null(df[[x, y]])
        df[y] = df[y].apply(str, meta=(y, str))
        dtcat: List[Any] = []
        # line chart
        dtcat.append(
            dask.delayed(_calc_line_dt)(df, timeunit, ngroups=ngroups, largest=largest)
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
        df = drop_null(df[[x, y]])
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
        df = drop_null(df[[x, y]])
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
        frmtd_bins = get_intervals(minv, maxv, bins)
        hist_arr, bins_arr = da.histogram(grp_srs, bins=frmtd_bins)
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


def _format_bin_intervals(bins_arr: np.ndarray) -> List[str]:
    """
    Auxillary function to format bin intervals in a histogram
    """
    bins_arr = np.round(bins_arr, 3)
    bins_arr = [int(val) if float(val).is_integer() else val for val in bins_arr]
    intervals = [
        f"[{bins_arr[i]}, {bins_arr[i + 1]})" for i in range(len(bins_arr) - 2)
    ]
    intervals.append(f"[{bins_arr[-2]},{bins_arr[-1]}]")
    return intervals
