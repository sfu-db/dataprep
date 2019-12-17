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
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-return-statements
    # pylint: disable=no-else-return

    df = to_dask(df)
    orig_df_len = len(df)

    if x is None and y is None:
        datas: List[Tuple[str, DType, Any]] = []
        for column in df.columns:
            dtype = df[column].dtype

            if is_categorical(dtype):
                bardata = calc_bar_pie(df[column], ngroups, largest)
                datas.append((column, DType.Categorical, bardata))
            elif is_numerical(dtype):
                histdata = calc_hist(df[column], orig_df_len, bins)
                datas.append((column, DType.Numerical, histdata))
            else:
                raise UnreachableError

        return Intermediate(data=datas, visual_type="basic_grid")
    elif (x is None) != (y is None):
        col: str = cast(str, x or y)
        if is_categorical(df[col].dtype):
            # data for bar and pie charts
            data = calc_bar_pie(df[col], ngroups, largest)
            return Intermediate(col=col, data=data, visual_type="categorical_column")

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
            histdata = calc_hist(df[col], orig_df_len, bins)
            # kde plot
            kdedata = calc_hist_kde(df[col].dropna().values, bins, bandwidth)
            # box plot
            boxdata = calc_box(df[[col]].dropna(), bins)
            return Intermediate(
                col=col,
                histdata=histdata,
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
            # box plot per group
            boxdata = calc_box(df[[x, y]].dropna(), bins)
            # histogram per group
            hisdata = calc_hist_by_group(df[[x, y]].dropna(), bins)
            return Intermediate(
                x=x,
                y=y,
                boxdata=boxdata,
                histdata=hisdata,
                visual_type="cat_and_num_cols",
            )
        if is_categorical(xdtype) and is_categorical(ydtype):
            # nested bar chart
            nesteddata = calc_nested(df[[x, y]].dropna())
            # stacked bar chart
            stackdata = calc_stacked(df[[x, y]].dropna())
            # heat map
            heatmapdata = calc_heatmap(df[[x, y]].dropna())
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
    miss_pct = round(srs.isna().sum().compute() / len(srs) * 100, 1)
    grp_srs = srs.groupby(srs).size()
    # select largest or smallest groups
    if largest:
        df = grp_srs.nlargest(n=ngroups).to_frame()
    else:
        df = grp_srs.nsmallest(n=ngroups).to_frame()
    df.columns = ["cnt"]
    # create a row containing the sum of the other groups
    other_cnt = len(srs) - df["cnt"].sum().compute()
    df2 = pd.DataFrame({srs.name: ["Others"], "cnt": [other_cnt]})
    df = df.reset_index().append(to_dask(df2))
    df["pct"] = df["cnt"] / len(srs) * 100
    return df.compute(), len(grp_srs), miss_pct


def calc_hist(
    srs: dd.Series, orig_df_len: int, bins: int
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
    miss_pct = round(srs.isna().sum().compute() / len(srs) * 100, 1)

    data = srs.dropna().values
    minv, maxv = data.min().compute(), data.max().compute()

    hist_array, bins_array = da.histogram(data, range=[minv, maxv], bins=bins)
    hist_array = hist_array.compute()
    hist_df = pd.DataFrame(
        {
            "left": bins_array[:-1],
            "right": bins_array[1:],
            "freq": hist_array,
            "pct": hist_array / orig_df_len * 100,
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
    minv, maxv = data.min().compute(), data.max().compute()
    hist_arr, bins_arr = da.histogram(data, range=[minv, maxv], bins=bins, density=True)
    hist_arr = hist_arr.compute()
    hist_df = pd.DataFrame(
        {"left": bins_arr[:-1], "right": bins_arr[1:], "freq": hist_arr}
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
    actual_qs = srs.quantile(q_range).compute()
    mean, std = srs.mean().compute(), srs.std().compute()
    theory_qs = np.sort(np.asarray(norm.ppf(q_range, mean, std)))
    return actual_qs, theory_qs


def calc_box(
    df: dd.DataFrame, bins: int, ngroups: int = 10
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
        number of groups to show if df has a categorical and
        numerical column

    Returns
    -------
    Tuple[pd.DataFrame, List[str], List[float], Dict[str, int]]
        The box plot statistics in a dataframe, a list of the outlier
        groups and another list of the outlier values, a dictionary
        logging the sampled group output
    """
    grp_cnt_stats = None  # to inform the user of sampled output

    x = df.columns[0]
    if len(df.columns) == 1:
        df = _calc_box_stats(df[x], x)
    else:
        y = df.columns[1]
        if is_numerical(df[x].dtype) and is_numerical(df[y].dtype):
            minv, maxv = df[x].min().compute(), df[x].max().compute()
            if df[x].nunique().compute() < bins:
                bins = df[x].nunique().compute() - 1
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
            df, grp_cnt_stats, largest_grps = _calc_groups(df, ngroups)
            # calculate a box plot over each group
            df = dd.concat(
                [_calc_box_stats(df[df[x] == grp][y], grp) for grp in largest_grps],
                axis=1,
            ).compute()

    df = df.append(pd.Series({c: i + 1 for i, c in enumerate(df.columns)}, name="x",)).T
    df.index.name = "grp"
    df["x0"], df["x1"] = df["x"] - 0.8, df["x"] - 0.2  # width of whiskers for plotting

    outx: List[str] = []  # list for the outlier groups
    outy: List[float] = []  # list for the outlier values
    for grp in df.index:
        otlrs = df.loc[grp]["otlrs"]
        outx = outx + [grp] * len(otlrs)
        outy = outy + otlrs

    return df, outx, outy, grp_cnt_stats


def calc_hist_by_group(
    df: dd.DataFrame, bins: int, ngroups: int = 10
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

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        The histograms in a dataframe and a dictionary
        logging the sampled group output
    """

    hist_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = dict()
    hist_lst: List[Any] = list()
    df, grp_cnt_stats, largest_grps = _calc_groups(df, ngroups)

    # create a histogram for each group
    for grp in largest_grps:
        grp_series = df.groupby([df.columns[0]]).get_group(grp)[df.columns[1]]
        minv, maxv = grp_series.min().compute(), grp_series.max().compute()
        hist = da.histogram(grp_series, range=[minv, maxv], bins=bins)
        hist_lst.append(hist)

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
    df: dd.DataFrame, ngroups: int = 5, nsubgroups: int = 8,
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
    df: dd.DataFrame, ngroups: int = 20, nsubgroups: int = 5,
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
        df_res["Others"] = 100 - df_res.sum(axis=1)
        fin_df = fin_df.append(df_res, sort=False)

    fin_df = fin_df.fillna(value=0)
    others = fin_df.pop("Others")
    if others.sum() > 1e-4:
        fin_df["Others"] = others
    fin_df["grps"] = largest_grps
    return fin_df, grp_cnt_stats


def calc_heatmap(
    df: dd.DataFrame, ngroups: int = 70, nsubgroups: int = 10,
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

    df_res = df.groupby([x, y]).size().reset_index()
    df_res.columns = [x, y, "cnt"]

    grp_cnt_stats["y_ttl"] = len(srs.index.compute())
    grp_cnt_stats["y_show"] = len(largest_subgrps)

    return df_res.compute(), grp_cnt_stats


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
    stats["lw"] = grp_srs[grp_srs >= stats["q1"] - 1.5 * iqr].min().compute()
    stats["uw"] = grp_srs[grp_srs <= stats["q3"] + 1.5 * iqr].max().compute()

    otlrs = grp_srs[(grp_srs < stats["lw"]) | (grp_srs > stats["uw"])]
    if len(otlrs) > 100:  # sample 100 outliers
        otlrs = otlrs.sample(frac=100 / len(otlrs))
    stats["otlrs"] = list(otlrs.compute())

    return pd.DataFrame({grp: stats})


def _calc_groups(
    df: dd.DataFrame, ngroups: int
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

    Returns
    -------
        The parsed dataframe, a dictionary logging the sampled groups,
        and a list of the sampled groups
    """

    # group count statistics to inform the user of the sampled output
    grp_cnt_stats = dict()

    srs = df.groupby(df.columns[0]).size()
    srs_lrgst = srs.nlargest(n=ngroups)
    largest_grps = list(srs_lrgst.index.compute())
    df = df[df[df.columns[0]].isin(largest_grps)]

    grp_cnt_stats["x_ttl"] = len(srs.index.compute())
    grp_cnt_stats["x_show"] = len(largest_grps)

    return df, grp_cnt_stats, list(map(str, largest_grps))
