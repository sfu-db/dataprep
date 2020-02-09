"""
    This module implements the intermediates computation
    for plot(df) function.
"""
from sys import stderr
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm

from ...errors import UnreachableError
from ..dtypes import DType, is_categorical, is_numerical
from ..intermediate import Intermediate
from ..utils import to_dask

__all__ = ["compute"]


def compute(
    df: Union[pd.DataFrame, dd.DataFrame],
    x: Optional[str] = None,
    y: Optional[str] = None,
    *,
    bins: int = 10,
    ngroups: int = 10,
    largest: bool = True,
    nsubgroups: int = 5,
    bandwidth: float = 1.5,
    sample_size: int = 1000,
    value_range: Optional[Tuple[float, float]] = None,
) -> Intermediate:
    """
    Parameters:
    ----------
    df : Union[pd.DataFrame, dd.DataFrame]
        Dataframe from which plots are to be generated.
    x : str, optional, default None
        A valid column name from the dataframe.
    y : str, optional, default None
        A valid column name from the dataframe.
    bins : int, default 10
        For a histogram or box plot with numerical x axis, it defines
        the number of equal-width bins to use when grouping.
    ngroups : int, default 10
        When grouping over a categorical column, it defines the
        number of groups to show in the plot. Ie, the number of
        bars to show in a bar chart.
    largest : bool, default True
        If true, when grouping over a categorical column, the groups
        with the largest count will be output. If false, the groups
        with the smallest count will be output.
    nsubgroups : int
        If x and y are categorical columns, ngroups refers to
        how many groups to show from column x, and nsubgroups refers to
        how many subgroups to show from column y in each group in column x.
    bandwidth : float, default 1.5
        Bandwidth for the kernel density estimation.
    sample_size : int, default 1000
        Sample size for the scatter plot.
    value_range : (float, float), optional, default None
        The lower and upper bounds on the range of a numerical column.
        Applies when column x is specified and column y is unspecified.

    Returns
    -------
    Intermediate
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-return-statements,too-many-statements
    # pylint: disable=no-else-return

    df = to_dask(df)
    orig_df_len = len(df)

    if x is None and y is None:
        datas: List[Any] = []
        col_names_dtypes: List[Tuple[str, DType]] = []
        for column in df.columns:
            if is_categorical(df[column].dtype):
                # bar chart
                datas.append(dask.delayed(calc_bar_pie)(df[column], ngroups, largest))
                col_names_dtypes.append((column, DType.Categorical))
            elif is_numerical(df[column].dtype):
                # histogram
                datas.append(dask.delayed(calc_hist)(df[column], bins, orig_df_len))
                col_names_dtypes.append((column, DType.Numerical))
            else:
                raise UnreachableError
        datas = dask.compute(*datas)
        data = [(col, dtp, dat) for (col, dtp), dat in zip(col_names_dtypes, datas)]
        return Intermediate(data=data, visual_type="basic_grid")

    elif (x is None) != (y is None):
        col: str = cast(str, x or y)
        if is_categorical(df[col].dtype):
            # data for bar and pie charts
            data = dask.compute(dask.delayed(calc_bar_pie)(df[col], ngroups, largest))
            return Intermediate(col=col, data=data[0], visual_type="categorical_column")
        elif is_numerical(df[col].dtype):
            if value_range is not None:
                if (
                    (value_range[0] <= np.nanmax(df[x]))
                    and (value_range[1] >= np.nanmin(df[x]))
                    and (value_range[0] < value_range[1])
                ):
                    df = df[df[col].between(value_range[0], value_range[1])]
                else:
                    print("Invalid range of values for this column", file=stderr)
            # qq plot
            qqdata = calc_qqnorm(df[col].dropna())
            # histogram
            histdata = dask.compute(dask.delayed(calc_hist)(df[col], bins, orig_df_len))
            # kde plot
            kdedata = calc_hist_kde(df[col].dropna().values, bins, bandwidth)
            # box plot
            boxdata = calc_box(df[[col]].dropna(), bins)
            return Intermediate(
                col=col,
                histdata=histdata[0],
                kdedata=kdedata,
                qqdata=qqdata,
                boxdata=boxdata,
                visual_type="numerical_column",
            )
        else:
            raise UnreachableError

    if x is not None and y is not None:
        xdtype, ydtype = df[x].dtype, df[y].dtype

        if (
            is_categorical(xdtype)
            and is_numerical(ydtype)
            or is_numerical(xdtype)
            and is_categorical(ydtype)
        ):
            x, y = (x, y) if is_categorical(df[x].dtype) else (y, x)
            df[x] = df[x].apply(str, meta=(x, str))
            # box plot per group
            boxdata = calc_box(df[[x, y]].dropna(), bins, ngroups, largest)
            # histogram per group
            hisdata = calc_hist_by_group(df[[x, y]].dropna(), bins, ngroups, largest)
            return Intermediate(
                x=x,
                y=y,
                boxdata=boxdata,
                histdata=hisdata,
                visual_type="cat_and_num_cols",
            )
        elif is_categorical(xdtype) and is_categorical(ydtype):
            df[x] = df[x].apply(str, meta=(x, str))
            df[y] = df[y].apply(str, meta=(y, str))
            # nested bar chart
            nesteddata = calc_nested(df[[x, y]].dropna(), ngroups, nsubgroups)
            # stacked bar chart
            stackdata = calc_stacked(df[[x, y]].dropna(), ngroups, nsubgroups)
            # heat map
            heatmapdata = calc_heatmap(df[[x, y]].dropna(), ngroups, nsubgroups)
            return Intermediate(
                x=x,
                y=y,
                nesteddata=nesteddata,
                stackdata=stackdata,
                heatmapdata=heatmapdata,
                visual_type="two_cat_cols",
            )
        elif is_numerical(xdtype) and is_numerical(ydtype):
            # scatter plot
            scatdata = calc_scatter(df[[x, y]].dropna(), sample_size)
            # hexbin plot
            hexbindata = df[[x, y]].dropna().compute()
            # box plot
            boxdata = calc_box(df[[x, y]].dropna(), bins)
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
    return Intermediate()


def calc_bar_pie(
    srs: dd.Series, ngroups: int, largest: bool
) -> Tuple[pd.DataFrame, int, float]:
    """
    Calculates the group counts given a series.

    Parameters
    ----------
    srs : dd.Series
        one categorical column
    ngroups : int
        number of groups to return
    largest : bool
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


def calc_hist(
    srs: dd.Series, bins: int, orig_df_len: int
) -> Tuple[pd.DataFrame, float]:
    """
    Calculate a histogram over a given series.

    Parameters
    ----------
    srs : dd.Series
        one numerical column over which to compute the histogram
    bins : int
        number of bins to use in the histogram
    orig_df_len : int
        length of the original dataframe

    Returns
    -------
    Tuple[pd.DataFrame, float]:
        The histogram in a dataframe and the percent of missing values
    """
    miss_pct = round(srs.isna().sum() / len(srs) * 100, 1)
    data = srs.dropna().values
    if len(data) == 0:  # all values in column are missing
        return pd.DataFrame({"left": [], "right": [], "freq": []}), miss_pct
    minv, maxv = data.min(), data.max()
    hist_arr, bins_arr = np.histogram(data, range=[minv, maxv], bins=bins)
    intervals = _format_bin_intervals(bins_arr)
    hist_df = pd.DataFrame(
        {
            "intervals": intervals,
            "left": bins_arr[:-1],
            "right": bins_arr[1:],
            "freq": hist_arr,
            "pct": hist_arr / orig_df_len * 100,
        }
    )
    return hist_df, miss_pct


def calc_hist_kde(
    data: da.Array, bins: int, bandwidth: float
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Calculate a density histogram and its corresponding kernel density
    estimate over a given series. The kernel is guassian.

    Parameters
    ----------
    data: da.Array
        one numerical column over which to compute the histogram and kde
    bins : int
        number of bins to use in the histogram
    bandwidth: float
        bandwidth for the kde

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
    pdf = gaussian_kde(data.compute(), bw_method=bandwidth)(pts_rng)
    return hist_df, pts_rng, pdf


def calc_qqnorm(srs: dd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate QQ plot given a series.

    Parameters
    ----------
    srs : dd.Series
        one numerical column from which to compute the quantiles

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of (actual quantiles, theoretical quantiles)
    """
    q_range = np.linspace(0.01, 0.99, 100)
    actual_qs, mean, std = dask.compute(srs.quantile(q_range), srs.mean(), srs.std())
    theory_qs = np.sort(np.asarray(norm.ppf(q_range, mean, std)))
    return actual_qs, theory_qs


def calc_box(
    df: dd.DataFrame, bins: int, ngroups: int = 10, largest: bool = True
) -> Tuple[pd.DataFrame, List[str], List[float], Optional[Dict[str, int]]]:
    """
    Compute a box plot over either
        1) the values in one column
        2) the values corresponding to groups in another column
        3) the values corresponding to binning another column

    Parameters
    ----------
    df : dd.DataFrame
        dask dataframe with one or two columns
    bins : int
        number of bins to use if df has two numerical columns
    ngroups : int
        number of groups to show if df has a categorical and numerical column
    largest: bool
        when calculating a box plot per group, select the largest or smallest groups

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
        if is_numerical(df[x].dtype) and is_numerical(df[y].dtype):
            minv, maxv, cnt = dask.compute(df[x].min(), df[x].max(), df[x].nunique())
            if cnt < bins:
                bins = cnt - 1
            endpts = np.linspace(minv, maxv, num=bins + 1)
            # calculate a box plot over each bin
            df = dd.concat(
                [
                    _calc_box_stats(
                        df[(df[x] >= endpts[i]) & (df[x] < endpts[i + 1])][y],
                        f"[{endpts[i]},{endpts[i+1]})",
                    )
                    if i != len(endpts) - 2
                    else _calc_box_stats(
                        df[(df[x] >= endpts[i]) & (df[x] <= endpts[i + 1])][y],
                        f"[{endpts[i]},{endpts[i+1]}]",
                    )
                    for i in range(len(endpts) - 1)
                ],
                axis=1,
            ).compute()
        else:
            df, grp_cnt_stats, largest_grps = _calc_groups(df, ngroups, largest)
            # calculate a box plot over each group
            df = dd.concat(
                [_calc_box_stats(df[df[x] == grp][y], grp) for grp in largest_grps],
                axis=1,
            ).compute()

    df = df.append(pd.Series({c: i + 1 for i, c in enumerate(df.columns)}, name="x",)).T
    df.index.name = "grp"
    df = df.reset_index()
    df["x0"], df["x1"] = df["x"] - 0.8, df["x"] - 0.2  # width of whiskers for plotting

    outx: List[str] = []  # list for the outlier groups
    outy: List[float] = []  # list for the outlier values
    for ind in df.index:
        otlrs = df.loc[ind]["otlrs"]
        outx = outx + [df.loc[ind]["grp"]] * len(otlrs)
        outy = outy + otlrs

    return df, outx, outy, grp_cnt_stats


def calc_hist_by_group(
    df: dd.DataFrame, bins: int, ngroups: int, largest: bool
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Compute a histogram over the values corresponding to the groups in another column

    Parameters
    ----------
    df : dd.DataFrame
        dask dataframe with one categorical and one numerical column
    bins : int
        number of bins to use in the histogram
    ngroups : int
        number of groups to show from the categorical column
    largest: bool
        select the largest or smallest groups

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        The histograms in a dataframe and a dictionary
        logging the sampled group output
    """
    # pylint: disable=too-many-locals

    hist_dict: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = dict()
    hist_lst: List[Tuple[np.ndarray, np.ndarray, List[str]]] = list()
    df, grp_cnt_stats, largest_grps = _calc_groups(df, ngroups, largest)

    # create a histogram for each group
    for grp in largest_grps:
        grp_srs = df.groupby([df.columns[0]]).get_group(grp)[df.columns[1]]
        minv, maxv = dask.compute(grp_srs.min(), grp_srs.max())
        hist_arr, bins_arr = da.histogram(grp_srs, range=[minv, maxv], bins=bins)
        intervals = _format_bin_intervals(bins_arr)
        hist_lst.append((hist_arr, bins_arr, intervals))

    hist_lst = dask.compute(*hist_lst)

    for elem in zip(largest_grps, hist_lst):
        hist_dict[elem[0]] = elem[1]

    return hist_dict, grp_cnt_stats


def calc_scatter(df: dd.DataFrame, sample_size: int) -> pd.DataFrame:
    """
    TO-DO: WARNING: For very large amount of points, implement Heat Map.
    Extracts the points to use in a scatter plot

    Parameters
    ----------
    df : dd.DataFrame
        two numerical columns
    sample_size : int
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
    df : dd.DataFrame
        two categorical columns
    ngroups : int
        number of groups to show from the first column
    nsubgroups : int
        number of subgroups (from the second column) to show in each group

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        The bar chart counts in a dataframe and a dictionary
        logging the sampled group output
    """
    df, grp_cnt_stats, _ = _calc_groups(df, ngroups)
    x, y = df.columns[0], df.columns[1]

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
    grp_cnt_stats["y_ttl"] = max_subcol_cnt
    grp_cnt_stats["y_show"] = min(max_subcol_cnt, nsubgroups)

    return df_res, grp_cnt_stats


def calc_stacked(
    df: dd.DataFrame, ngroups: int, nsubgroups: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Calculate a stacked bar chart of the counts of two columns

    Parameters
    ----------
    df : dd.DataFrame
        two categorical columns
    ngroups : int
        number of groups to show from the first column
    nsubgroups : int
        number of subgroups (from the second column) to show in each group

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        The bar chart counts in a dataframe and a dictionary
        logging the sampled group output
    """
    df, grp_cnt_stats, largest_grps = _calc_groups(df, ngroups)
    x, y = df.columns[0], df.columns[1]

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
    fin_df["grps"] = list(largest_grps)
    return fin_df, grp_cnt_stats


def calc_heatmap(
    df: dd.DataFrame, ngroups: int, nsubgroups: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Calculate a heatmap of the counts of two columns

    Parameters
    ----------
    df : dd.DataFrame
        two categorical columns
    ngroups : int
        number of groups to show from the first column
    nsubgroups : int
        number of subgroups (from the second column) to show in each group

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        The heatmap counts in a dataframe and a dictionary
        logging the sampled group output
    """
    df, grp_cnt_stats, _ = _calc_groups(df, ngroups)
    x, y = df.columns[0], df.columns[1]

    srs = df.groupby(y).size()
    srs_lrgst = srs.nlargest(n=nsubgroups)
    largest_subgrps = list(srs_lrgst.index.compute())
    df = df[df[y].isin(largest_subgrps)]

    df_res = df.groupby([x, y]).size().reset_index().compute()
    df_res.columns = ["x", "y", "cnt"]
    df_res = pd.pivot_table(
        df_res, index=["x", "y"], values="cnt", fill_value=0, aggfunc=np.sum,
    ).reset_index()

    grp_cnt_stats["y_ttl"] = len(srs.index.compute())
    grp_cnt_stats["y_show"] = len(largest_subgrps)

    return df_res, grp_cnt_stats


def _calc_box_stats(grp_srs: dd.Series, grp: str) -> pd.DataFrame:
    """
    Auxiliary function to calculate the Tukey box plot statistics

    Parameters
    ----------
    grp_srs: dd.Series
        one numerical column
    grp: str
        Name of the group of the corresponding series values

    Returns
    -------
    pd.DataFrame
        A dataframe containing box plot statistics
    """
    stats: Dict[str, Any] = dict()

    try:  # this is a bad fix for the problem of when there is no data passed to this function
        qntls = np.round(grp_srs.quantile([0.25, 0.50, 0.75]).compute(), 3)
        stats["q1"], stats["q2"], stats["q3"] = qntls[0.25], qntls[0.50], qntls[0.75]
    except ValueError:
        stats["q1"], stats["q2"], stats["q3"] = np.nan, np.nan, np.nan

    iqr = stats["q3"] - stats["q1"]
    stats["lw"] = grp_srs[grp_srs >= stats["q1"] - 1.5 * iqr].min()
    stats["uw"] = grp_srs[grp_srs <= stats["q3"] + 1.5 * iqr].max()
    stats["lw"], stats["uw"] = dask.compute(stats["lw"], stats["uw"])

    otlrs = grp_srs[(grp_srs < stats["lw"]) | (grp_srs > stats["uw"])]
    if len(otlrs) > 100:  # sample 100 outliers
        otlrs = otlrs.sample(frac=100 / len(otlrs))
    stats["otlrs"] = list(otlrs.compute())

    return pd.DataFrame({grp: stats})


def _calc_groups(
    df: dd.DataFrame, ngroups: int, largest: bool = True
) -> Tuple[dd.DataFrame, Dict[str, int], List[str]]:
    """
    Auxillary function to parse the dataframe to consist
    of only the groups with the largest counts

    Parameters
    ----------
    df: dd.DataFrame
        two columns, the first column is categorical
    ngroups: int
        the number of groups with the largest counts to isolate
    largest: bool
        find largest or smallest groups

    Returns
    -------
        The parsed dataframe, a dictionary logging the sampled groups,
        and a list of the sampled groups
    """

    # group count statistics to inform the user of the sampled output
    grp_cnt_stats: Dict[str, int] = dict()

    srs = df.groupby(df.columns[0]).size()
    srs_lrgst = srs.nlargest(n=ngroups) if largest else srs.nsmallest(n=ngroups)
    largest_grps = list(srs_lrgst.index.compute())
    df = df[df[df.columns[0]].isin(largest_grps)]

    grp_cnt_stats["x_ttl"] = len(srs.index.compute())
    grp_cnt_stats["x_show"] = len(largest_grps)

    return df, grp_cnt_stats, largest_grps


def _format_bin_intervals(bins_arr: np.ndarray) -> List[str]:
    """
    Auxillary function to format bin intervals in a histogram

    Parameters
    ----------
    bins_arr: np.ndarray
        Bin endpoints to format into intervals

    Returns
    -------
        List of formatted bin intervals
    """
    bins_arr = np.round(bins_arr, 3)
    intervals = [f"[{bins_arr[i]},{bins_arr[i+1]})" for i in range(len(bins_arr) - 2)]
    intervals.append(f"[{bins_arr[-2]},{bins_arr[-1]}]")
    return intervals
