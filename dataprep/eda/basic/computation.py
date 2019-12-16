"""
    This module implements the plot(df) function.
"""
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple, cast

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import probscale
import scipy.stats as sm

from ..utils import DataType, get_type, sample_n
from ..intermediate import Intermediate
from .render import Render

DEFAULT_PARTITIONS = 1

# Type aliasing
StringList = List[str]
LOGGER = logging.getLogger(__name__)


def __calc_box_stats(grp_series: dask.dataframe.core.Series) -> Dict[str, Any]:
    """
    This is an auxiliary function to _calc_box for computation.
    :param grp_series: a series type of numerical data.
    :return: a stats dict containg computed results.
    """
    stats: Dict[str, Any] = dict()

    (grp_series,) = dask.compute(grp_series)
    quantiles = grp_series.quantile([0.25, 0.50, 0.75], interpolation="midpoint")
    stats["tf"], stats["fy"], stats["sf"] = (
        np.round(quantiles[0.25], 2),
        np.round(quantiles[0.50], 2),
        np.round(quantiles[0.75], 2),
    )
    stats["iqr"] = stats["sf"] - stats["tf"]
    stats["lw"] = grp_series[grp_series >= stats["tf"] - 1.5 * stats["iqr"]].min()
    stats["uw"] = grp_series[grp_series <= stats["sf"] + 1.5 * stats["iqr"]].max()

    if len(grp_series) == 1:
        val = grp_series.reset_index().iloc[0, 1]
        stats["max"], stats["min"], stats["max_outlier"] = (val,) * 3
        stats["outliers"] = [val]
    else:
        bounded = grp_series[
            (grp_series >= (stats["tf"] - 1.5 * stats["iqr"]))
            & (grp_series <= (stats["sf"] + 1.5 * stats["iqr"]))
        ]
        min_value, max_value = bounded.min(), bounded.max()

        outliers = list(
            grp_series[
                (grp_series < (stats["tf"] - 1.5 * stats["iqr"]))
                | (grp_series > (stats["sf"] + 1.5 * stats["iqr"]))
            ].round(2)
        )

        max_outlier = np.nan if not outliers else max(outliers)

        stats["min"] = 0 if np.equal(min_value, np.inf) else np.round(min_value, 2)
        stats["max"] = 0 if np.equal(max_value, -np.inf) else np.round(max_value, 2)
        stats["max_outlier"] = max_outlier
        stats["outliers"] = list(outliers)
    return stats


def __calc_groups(
    dataframe: dd.DataFrame,
    col_x: str,
    num_x_cats: int,
    col_y: Optional[str] = None,
    num_y_cats: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Auxiliary function to group the dataframe along specified columns.

    Parameters
    ----------
    dataframe : Dask DataFrame
        Dataframe to be grouped over by specified columns.

    col_x : str
        The name of a column in the input data frame.

    num_x_cats : int
        Number of groups with the highest count kept after grouping
        the dataframe over column col_x. If the total number of groups
        in col_x after grouping is less than num_x_cats, keep all groups.

    col_y : str, optional
        The name of a column in the input data frame.

    num_y_cats : int, optional
        Number of groups with the highest count kept after grouping
        the dataframe over column col_y. If the total number of groups
        in col_y after grouping is less than num_y_cats, keep all groups.
    """

    # group count statistics to inform user of the sampled output
    grp_cnt_stats = dict()

    # remove missing values
    if col_y is not None:
        df = dataframe.dropna(subset=[col_x, col_y])
    else:
        df = dataframe.dropna(subset=[col_x])

    # group by col_x and compute the count in each group
    grp = df.groupby(col_x)
    grp_series = dask.compute(grp[col_x].count())[0]
    # parse dataset to consist of only largest categories
    num_cats = min(len(grp_series), num_x_cats)
    largest_cats = list(grp_series.nlargest(n=num_cats).keys())
    df = df[df[col_x].isin(largest_cats)]

    if len(grp_series) > num_x_cats:
        grp_cnt_stats["x_total"] = len(grp_series)
    else:
        grp_cnt_stats["x_total"] = num_x_cats
    grp_cnt_stats["x_show"] = num_x_cats

    if col_y is None:
        return {"df": df, "grp_cnt_stats": grp_cnt_stats}

    if num_y_cats is not None:
        # group by col_y and compute the count in each group
        grp = df.groupby(col_y)
        grp_series = dask.compute(grp[col_y].count())[0]
        # parse dataset to consist of only largest categories
        num_cats = min(len(grp_series), num_y_cats)
        largest_cats = list(grp_series.nlargest(n=num_cats).keys())
        df = df[df[col_y].isin(largest_cats)]

        if len(grp_series) > num_y_cats:
            grp_cnt_stats["y_total"] = len(grp_series)
        else:
            grp_cnt_stats["y_total"] = num_y_cats
        grp_cnt_stats["y_show"] = num_y_cats

    # group by col_x and col_y and compute the count in each group
    grp_object = df.groupby([col_x, col_y])
    grp_series = dask.compute(grp_object[col_x].count())[0]

    return {
        "grp_series": grp_series,
        "categories": largest_cats,
        "grp_cnt_stats": grp_cnt_stats,
    }


def _calc_box(
    dataframe: dd.DataFrame,
    col_x: str,
    bins: int,
    col_y: Optional[str] = None,
    num_cats: int = 10,
) -> Intermediate:
    # pylint: disable=too-many-locals
    """
    Returns intermediate stats of the box plot
    of columns col_x and col_y.

    PARAMETERS
    __________
    dataframe: the input dataframe
    col_x : a valid column name of the dataframe
    bins : number of bins to group by on x axis if numerical x
    col_y : a valid column name of the dataframe
    num_cats : number of categories to show for categorical variable

    RETURNS
    __________
    a (column_name: data) dict storing the intermediate results
    """
    res: Dict[str, Any] = dict()
    grp_cnt_stats = None  # for if we group over categorical variable

    # remove missing values
    df = dataframe.dropna(subset=[col_x])
    if col_y is not None:
        df = df.dropna(subset=[col_y])

    if col_y is None:
        col_series = df[col_x]
        res[str(col_x)] = __calc_box_stats(col_series)
    elif (
        get_type(dataframe[col_x]) == DataType.TYPE_NUM
        and get_type(dataframe[col_y]) == DataType.TYPE_NUM
    ):
        df = df[[col_x, col_y]].compute()
        bin_endpoints = np.linspace(df[col_x].min(), df[col_x].max(), num=bins + 1)
        if np.issubdtype(df[col_x], np.int64):
            bin_endpoints = [int(val) for val in bin_endpoints]
        else:
            bin_endpoints = [np.round(val, 3) for val in bin_endpoints]

        # create a box plot for each bin
        for i in range(len(bin_endpoints) - 1):
            df_subset = df[df[col_x].between(bin_endpoints[i], bin_endpoints[i + 1])]
            grp_series = dd.from_pandas(df_subset, npartitions=1)[col_y]
            group = "({},{})".format(str(bin_endpoints[i]), str(bin_endpoints[i + 1]))
            res[str(group)] = __calc_box_stats(grp_series)
    else:
        col_x, col_y = (
            (col_x, col_y)
            if (get_type(dataframe[col_x]) == DataType.TYPE_CAT)
            else (col_y, col_x)
        )
        grouped_data = __calc_groups(df, col_x, num_cats)
        df = grouped_data["df"]
        grp_cnt_stats = grouped_data["grp_cnt_stats"]

        # create box plot for each category
        for group in dask.compute(df[col_x].unique())[0]:
            grp_series = df.groupby([col_x]).get_group(group)[col_y]
            res[str(group)] = __calc_box_stats(grp_series)
    raw_data = {"df": dataframe, "col_x": col_x, "col_y": col_y}
    result = {"box_plot": res, "grp_cnt_stats": grp_cnt_stats}
    return Intermediate(result, raw_data)


def _calc_nested(
    dataframe: dd.DataFrame,
    col_x: str,
    col_y: str,
    num_x_cats: int = 5,
    num_y_cats: int = 10,
) -> Intermediate:
    # pylint: disable=too-many-locals
    """ Returns intermediate stats of the nested bar chart
            of columns col_x and col_y.

    PARAMETERS
    __________
    dataframe: the input dataframe
    col_x : a valid column name of the dataframe
    col_y : a valid column name of the dataframe
    num_x_cats : number of categories to show for x variable
    num_y_cats : number of categories to show for y variable

    RETURNS
    __________
    a (column_name: data) dict storing the intermediate results
    """

    grouped_data = __calc_groups(dataframe, col_x, num_x_cats, col_y)
    grp_series = grouped_data["grp_series"]
    largest_cats = grouped_data["categories"]
    grp_cnt_stats = grouped_data["grp_cnt_stats"]

    # create the final dataframe with only the largest subcategories
    most_sub_cats = 0
    final_series = pd.Series([])
    for category in largest_cats:
        largest_cats = list(set(grp_series[category].nlargest(n=num_y_cats).keys()))
        isolate = [(category, x) for x in largest_cats]
        final_series = final_series.append(
            grp_series[grp_series.index.isin(isolate)].sort_values(ascending=False)
        )
        if len(grp_series[category]) > most_sub_cats:
            most_sub_cats = len(grp_series[category])

    if most_sub_cats > num_y_cats:
        grp_cnt_stats["y_total"] = most_sub_cats
    else:
        grp_cnt_stats["y_total"] = num_y_cats
    grp_cnt_stats["y_show"] = num_y_cats

    raw_data = {"df": dataframe, "col_x": col_x, "col_y": col_y}
    result = {"nested_bar_chart": dict(final_series), "grp_cnt_stats": grp_cnt_stats}
    return Intermediate(result, raw_data)


def _calc_stacked(
    dataframe: dd.DataFrame,
    col_x: str,
    col_y: str,
    num_x_cats: int = 20,
    num_y_cats: int = 5,
) -> Intermediate:
    # pylint: disable=too-many-locals
    """ Returns intermediate stats of the stacked bar chart
            of columns col_x and col_y.

    PARAMETERS
    __________
    dataframe: the input dataframe
    col_x : a valid column name of the dataframe
    col_y : a valid column name of the dataframe
    num_x_cats : number of categories to show for x variable
    num_y_cats : number of categories to show for y variable

    RETURNS
    __________
    a (column_name: data) dict storing the intermediate results
    """
    grouped_data = __calc_groups(dataframe, col_x, num_x_cats, col_y)
    grp_series = grouped_data["grp_series"]
    largest_cats = grouped_data["categories"]

    # lists to create final dataframe in format for a stacked bar chart
    index1: List[Any] = []  # group values that serve as first index of resulting df
    index2: List[Any] = []  # group values that serve as second index of resulting df
    values: List[Any] = []  # corresponding values to the groups in index1 and index2

    for category in largest_cats:
        # find largest subcategories in each category
        num_cats = min(len(grp_series[category]), num_y_cats)
        largest_sub_cats = list(grp_series[category].nlargest(n=num_cats).keys())
        largest_indices = [(category, x) for x in largest_sub_cats]
        result = grp_series[grp_series.index.isin(largest_indices)].sort_values(
            ascending=False
        )
        # if more than max_sub_cats categories, aggregate other categories as 'Others'
        if len(grp_series[category].keys()) > num_y_cats:
            sum_series = grp_series[category][
                ~grp_series[category].index.isin(largest_sub_cats)
            ]
            others = pd.Series([sum_series.sum()], index=[(category, "Others")])
            result = result.append(others)
            largest_sub_cats = largest_sub_cats + ["Others"]

        index1 = index1 + [category] * len(largest_sub_cats)
        index2 = index2 + largest_sub_cats
        values = values + [val / np.sum(list(result)) * 100 for val in list(result)]
    index1 = list(map(str, index1))
    index2 = list(map(str, index2))

    # create formatted dictionary for stacked bar charts
    final_series = pd.Series(values, index=[index1, index2])
    final_data_dict: Dict[Any, Any] = {}
    sub_categories = list(set(final_series.index.get_level_values(1)))

    if "Others" in sub_categories:
        # Move 'Others' to the end of the list so it appears on top in the chart
        sub_categories.remove("Others")
        sub_categories.append("Others")

    for sub_val in sub_categories:
        final_data_dict[sub_val] = []
        for x_val in largest_cats:
            if (x_val, sub_val) in final_series.index:
                final_data_dict[sub_val].append(final_series[x_val, sub_val])
            else:
                final_data_dict[sub_val].append(0)
    final_data_dict["x_categories"] = list(map(str, largest_cats))
    raw_data = {"df": dataframe, "col_x": col_x, "col_y": col_y}
    result = {
        "stacked_bar_chart": final_data_dict,
        "sub_categories": sub_categories,
        "grp_cnt_stats": grouped_data["grp_cnt_stats"],
    }
    return Intermediate(result, raw_data)


def _calc_heat_map(
    dataframe: dd.DataFrame,
    col_x: str,
    col_y: str,
    num_x_cats: int = 70,
    num_y_cats: int = 10,
) -> Intermediate:
    # pylint: disable=too-many-locals
    """ Returns intermediate stats of the nested column plot
            of columns col_x and col_y.

    PARAMETERS
    __________
    dataframe: the input dataframe
    col_x : a valid column name of the dataframe
    col_y : a valid column name of the dataframe
    num_x_cats : number of categories to show for x variable
    num_y_cats : number of categories to show for y variable

    RETURNS
    __________
    a (column_name: data) dict storing the intermediate results
    """
    grouped_data = __calc_groups(dataframe, col_x, num_x_cats, col_y, num_y_cats)
    grp_series = grouped_data["grp_series"]

    grp_df = grp_series.to_frame().unstack(fill_value=0).stack()
    grp_df.columns = ["total"]
    grp_df = grp_df.reset_index(level=[col_x, col_y])
    grp_df[col_x] = grp_df[col_x].astype(str)
    grp_df[col_y] = grp_df[col_y].astype(str)

    raw_data = {"df": dataframe, "col_x": col_x, "col_y": col_y}
    result = {"heat_map": grp_df, "grp_cnt_stats": grouped_data["grp_cnt_stats"]}
    return Intermediate(result, raw_data)


def _calc_scatter(
    dataframe: dd.DataFrame, col_x: str, col_y: str, plot_type: str
) -> Intermediate:
    """
        TO-DO: WARNING: For very large amount of points, implement Heat Map.
        Returns intermediate stats of the scattered plot
        of columns col_x and col_y.

        PARAMETERS
        __________
        dataframe: the input dataframe
        col_x : a valid column name of the dataframe
        col_y : a valid column name of the dataframe
        plot_type : scatter_plot or hexbin_plot

        RETURNS
        __________
        a (column_name: data) dict storing the intermediate results
    """
    series_x = dask.compute(dataframe[col_x])[0].dropna()
    series_y = dask.compute(dataframe[col_y])[0].dropna()

    res = list()
    for each in zip(series_x, series_y):
        res.append((round(each[0], 2), round(each[1], 2)))

    raw_data = {"df": dataframe, "col_x": col_x, "col_y": col_y}
    result = {plot_type: res}
    return Intermediate(result, raw_data)


def _calc_pie(dataframe: dd.DataFrame, col_x: str) -> Intermediate:
    """ Returns a dict {category: category_count} for the
        categorical column given as the second argument

    Parameters
    __________
    dataframe : the input pandas dataframe
    col : the str column of dataframe for which count needs to be calculated

    Returns
    __________
    dict : A dict of (category : count) for the input col
    """
    grp_object = dataframe.groupby(col_x)[col_x].count()
    raw_data = {"df": dataframe, "col_x": col_x, "col_y": None}
    result_dict = dict(dask.compute(grp_object)[0])
    # result_dict.update({"NULL": dask.compute(dataframe[col_x].isna().sum())[0]})
    result = {"pie_chart": result_dict}
    return Intermediate(result, raw_data)


def _calc_bar(dataframe: dd.DataFrame, col_x: str) -> Intermediate:
    """ Returns a dict {category: category_count} for the
        categorical column given as the second argument

    Parameters
    __________
    dataframe : the input pandas dataframe
    col : the str column of dataframe for which count needs to be calculated

    Returns
    __________
    dict : A dict of (category : count) for the input col
    """
    grp_object = dask.compute(dataframe.groupby([col_x])[col_x].count())[0]
    miss_vals = dask.compute(dataframe[col_x].isna().sum())[0]
    raw_data = {"df": dataframe, "col_x": col_x, "col_y": None}
    result = {"bar_chart": dict(grp_object), "missing": [miss_vals]}
    return Intermediate(result=result, raw_data=raw_data)


def _calc_hist_by_group(
    dataframe: dd.DataFrame, col_x: str, col_y: str, bins: int, num_x_cats: int = 10
) -> Intermediate:
    # pylint: disable=too-many-locals
    """Returns the histogram array for the continuous
        distribution of values in the column given as the second argument
    _TODO write test
    Parameters
    __________
    dataframe : the input pandas dataframe
    col : the str column of dataframe for which hist array needs to be
    calculated
    bins : number of bins to use in the histogram
    num_x_cats : number of categories to show for categorical variable

    Returns
    __________
    np.array : An array of values representing histogram for the input col
    """
    col_cat, col_num = (
        (col_x, col_y)
        if (get_type(dataframe[col_x]) == DataType.TYPE_CAT)
        else (col_y, col_x)
    )

    grp_hist: Dict[str, Tuple[Any, Any]] = dict()
    hist_interm: List[Any] = list()
    grp_name_list: List[str] = list()

    grouped_data = __calc_groups(dataframe, col_cat, num_x_cats)
    df = grouped_data["df"].dropna(subset=[col_num])

    # create a histogram for each category
    for group in dask.compute(df[col_cat].unique())[0]:
        grp_series = df.groupby([col_cat]).get_group(group)[col_num]
        minv = dask.compute(grp_series.min())[0]
        maxv = dask.compute(grp_series.max())[0]
        hist = da.histogram(grp_series, range=[minv, maxv], bins=bins)
        hist_interm.append(hist)
        grp_name_list.append(str(group))

    (hist_interm,) = dask.compute(hist_interm)

    for zipped_element in zip(grp_name_list, hist_interm):
        grp_hist[zipped_element[0]] = zipped_element[1]

    return Intermediate(
        {"line_chart": grp_hist, "grp_cnt_stats": grouped_data["grp_cnt_stats"]},
        {"df": dataframe, "col_x": col_cat, "col_y": col_num, "bins": bins},
    )


def _calc_hist(
    dataframe: dd.DataFrame, col_x: str, orig_df_len: int, show_y_label: bool, bins: int
) -> Intermediate:
    """Returns the histogram array for the continuous
        distribution of values in the column given as the second argument

    Parameters
    __________
    dataframe : the input pandas dataframe
    col : the str column of dataframe for which hist array needs to be
    calculated
    orig_df_len : length of the original dataframe
    show_y_label : show y_axis_labels in visualization if True
    bins : number of bins to use in the histogram

    Returns
    _______
    np.array : An array of values representing histogram for the input col
    """
    raw_data = {"df": dataframe, "col_x": col_x, "col_y": None, "bins": bins}
    if dask.compute(dataframe[col_x].size)[0] == 0:
        return Intermediate({"histogram": (list(), list())}, raw_data)

    (minv,) = dask.compute(dataframe[col_x].min())
    (maxv,) = dask.compute(dataframe[col_x].max())

    dframe = dataframe[col_x].dropna().values

    if isinstance(dframe, dask.array.core.Array):
        hist_array, bins_array = da.histogram(dframe, range=[minv, maxv], bins=bins)
        (hist_array,) = dask.compute(hist_array)
    elif isinstance(dframe, np.ndarray):
        (dframe,) = dask.compute(dframe)
        minv = 0 if np.isnan(dframe.min()) else dframe.min()
        maxv = 0 if np.isnan(dframe.max()) else dframe.max()
        hist_array, bins_array = np.histogram(dframe, bins=bins, range=[minv, maxv])

    if dask.compute(np.issubdtype(dataframe[col_x], np.int64))[0]:
        bins_temp = [int(x) for x in np.ceil(bins_array)]
        if len(bins_temp) != len(set(bins_temp)):
            bins_array = [round(x, 2) for x in bins_array]
        else:
            bins_array = bins_temp
    else:
        bins_array = [round(x, 2) for x in bins_array]

    miss_vals = dask.compute(dataframe[col_x].isna().sum())[0]

    return Intermediate(
        {
            "histogram": (hist_array, bins_array),
            "missing": [miss_vals],
            "orig_df_len": orig_df_len,
            "show_y_label": show_y_label,
        },
        raw_data,
    )


def _calc_qqnorm(df: dd.DataFrame, col_x: str) -> Intermediate:
    """
    Calculates points of the QQ plot of the given column of the data frame.
    :param dataframe - the input dataframe
    :param col - the input column of the dataframe
    :return: calculated quantiles
    """
    dask_series = df[col_x].dropna()
    position, y_points = probscale.plot_pos(dask.compute(dask_series)[0])
    mean, std = dask_series.mean().compute(), dask_series.std().compute()
    theory_ys = np.sort(np.asarray(sm.norm.ppf(position, mean, std)))
    theory_ys = sample_n(theory_ys, 100)

    actual_ys = np.sort(np.asarray(y_points))
    actual_ys = sample_n(actual_ys, 100)
    result_dict = dict(theory=theory_ys, sample=actual_ys)
    return Intermediate(
        {"qqnorm_plot": result_dict}, {"df": df, "col_x": col_x, "col_y": None}
    )


def _calc_hist_kde(dataframe: dd.DataFrame, col_x: str) -> Intermediate:
    """
    Returns numpy ndarray representation of the column series
    :param dataframe: the input dataframe
    :param col_x: the input column
    :return: numpy ndarray
    """
    raw_data = {"df": dataframe, "col_x": col_x, "col_y": None}
    # hist = _calc_hist(dataframe, col_x)
    return Intermediate(
        {"kde_plot": np.array(dask.compute(dataframe[col_x])[0])}, raw_data
    )


def plot_df(
    data_frame: dd.DataFrame,
    bins: int,
    force_cat: Optional[StringList] = None,
    force_num: Optional[StringList] = None,
) -> List[Intermediate]:
    """
    Supporting funtion to the main plot function
    :param data_frame: dask dataframe
    :param bins: number of bins to show in the histogram
    :param force_cat: list of categorical columns defined explicitly
    :param force_num: list of numerical columns defined explicitly
    :return:
    """
    col_list = list()
    dask_result: List[Any] = list()
    for col in data_frame.columns:
        if dask.compute(data_frame[col].count())[0] == 0:
            col_list.append(col)
            dask_result.append(Intermediate(dict(), {"col_x": col}))

        elif get_type(data_frame[col]) == DataType.TYPE_CAT or (
            force_cat is not None and col in force_cat
        ):
            # print("df bar", type(data_frame))
            cnt_series = dask.delayed(_calc_bar)(data_frame, col)
            dask_result.append(cnt_series)
            col_list.append(col)

        elif get_type(data_frame[col]) == DataType.TYPE_NUM or (
            force_num is not None and col in force_num
        ):
            # print("df, hist", type(data_frame))
            hist = dask.delayed(_calc_hist)(
                data_frame, col, len(data_frame), False, bins
            )
            dask_result.append(hist)
            col_list.append(col)

    computed_res: List[Intermediate] = dask.compute(dask_result)[0]
    return computed_res


def plot(
    pd_data_frame: pd.DataFrame,
    col_x: Optional[str] = None,
    col_y: Optional[str] = None,
    force_cat: Optional[StringList] = None,
    force_num: Optional[StringList] = None,
    bins: int = 10,
    value_range: Optional[Tuple[float, float]] = None,
    **kwrgs: Any,
) -> List[Intermediate]:
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    """Generates plots for exploratory data analysis.

    If col_x and col_y are unspecified, the distribution of
    each coloumn is plotted. A histogram is plotted if the
    column contains numerical values, and a bar chart is plotted
    if the column contains categorical values.

    If col_x is specified and col_y is unspecified, the
    distribution of col_x is plotted in various ways. If col_x
    contains categorical values, a bar chart and pie chart are
    plotted. If col_x contains numerical values, a histogram,
    kernel density estimate plot, box plot, and qq plot are plotted.

    If col_x and col_y are specified, plots depicting
    the relationship between the variables will be displayed. If
    col_x and col_y contain numerical values, a scatter plot, hexbin
    plot, and binned box plot are plotted. If one of col_x and col_y
    contain categorical values and the other contains numerical values,
    a box plot and multi-line histogram are plotted. If col_X and col_y
    contain categorical vales, a nested bar chart, stacked bar chart, and
    heat map are plotted.

    Parameters:
    ----------
    pd_data_frame : pandas DataFrame
        The input data frame from which charts or plots are to
        be generated.

    col_x : str, optional
        The name of a column in the input data frame.

    col_y : str, optional
        The name of a column in the input data frame.

    force_cat : List[str], optional
        A list of column names which forces these columns to be treated
        as having categorical values.

    force_num : List[str], optional
        A list of column names which forces these columns to be treated
        as having numerical values.

    bins : int, optional, default 10
        For a histogram or box plot with numerical x axis, it defines
        the number of equal-width bins to use when grouping.

    value_range : (float, float), optional
        The lower and upper bounds on the histogram plotted when
        col_x is numerical, and col_y is unspecified.

    Examples
    --------
    >>> import pandas as pd
    >>> from dataprep.eda import *
    >>> iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    >>> plot(iris)
    >>> plot(iris, "petal_length", bins=20, value_range=(1,5))
    >>> plot(iris, "petal_width", "species")
    """

    data_frame: dd.DataFrame = dd.from_pandas(
        pd_data_frame, npartitions=DEFAULT_PARTITIONS
    )

    list_of_intermediates: List[Intermediate] = list()
    orig_df_len = len(data_frame)

    if (col_x is None and col_y is not None) or (col_x is not None and col_y is None):

        target_col: str = cast(str, col_x if col_y is None else col_y)
        dask_result: List[Intermediate] = list()

        if dask.compute(data_frame[target_col].count())[0] == 0:
            dask_result.append(Intermediate(dict(), {"col_x": target_col}))

        elif get_type(data_frame[target_col]) == DataType.TYPE_CAT or (
            force_cat is not None and target_col in force_cat
        ):
            # BAR_CHART
            dask_result.append(_calc_bar(data_frame, target_col))

            # PIE_CHART
            dask_result.append(_calc_pie(data_frame, target_col))

        elif get_type(data_frame[target_col]) == DataType.TYPE_NUM or (
            force_num is not None and target_col in force_num
        ):
            if value_range is not None:
                if (
                    (value_range[0] <= np.nanmax(data_frame[col_x]))
                    and (value_range[1] >= np.nanmin(data_frame[col_x]))
                    and (value_range[0] < value_range[1])
                ):
                    data_frame = data_frame[
                        (data_frame[col_x] >= value_range[0])
                        & (data_frame[col_x] <= value_range[1])
                    ]
                else:
                    print("Invalid range of values for this column", file=sys.stderr)
                    return dask_result
            # HISTOGRAM
            dask_result.append(
                _calc_hist(data_frame, target_col, orig_df_len, True, bins)
            )

            # KDE_PLOT
            dask_result.append(_calc_hist_kde(data_frame, target_col))

            # BOX_PLOT
            dask_result.append(_calc_box(data_frame, target_col, bins))

            # QQ-NORM
            dask_result.append(_calc_qqnorm(data_frame, target_col))
        Render.vizualise(Render(**kwrgs), dask_result, True)
        return dask_result  # if kwrgs.get("return_result") else None

    if col_x is not None and col_y is not None:
        if force_cat is not None and col_x in force_cat:
            type_x = DataType.TYPE_CAT
        elif force_num is not None and col_x in force_num:
            type_x = DataType.TYPE_NUM
        else:
            type_x = get_type(data_frame[col_x])
        if force_cat is not None and col_y in force_cat:
            type_y = DataType.TYPE_CAT
        elif force_num is not None and col_y in force_num:
            type_y = DataType.TYPE_NUM
        else:
            type_y = get_type(data_frame[col_y])

        temp_result: List[Intermediate] = list()
        try:
            if (
                type_y == DataType.TYPE_CAT
                and type_x == DataType.TYPE_NUM
                or type_y == DataType.TYPE_NUM
                and type_x == DataType.TYPE_CAT
            ):
                # BOX_PER_GROUP
                temp_result.append(_calc_box(data_frame, col_x, bins, col_y))
                # HISTOGRAM_PER_GROUP
                temp_result.append(_calc_hist_by_group(data_frame, col_x, col_y, bins))

            elif type_x == DataType.TYPE_CAT and type_y == DataType.TYPE_CAT:
                # NESTED_BAR_CHART
                temp_result.append(_calc_nested(data_frame, col_x, col_y))
                # STACKED_BAR_CHART
                temp_result.append(_calc_stacked(data_frame, col_x, col_y))
                # HEAT_MAP
                temp_result.append(_calc_heat_map(data_frame, col_x, col_y))

            elif type_x == DataType.TYPE_NUM and type_y == DataType.TYPE_NUM:
                # SCATTER_PLOT
                temp_result.append(
                    _calc_scatter(data_frame, col_x, col_y, "scatter_plot")
                )
                # HEXBIN_PLOT
                temp_result.append(
                    _calc_scatter(data_frame, col_x, col_y, "hexbin_plot")
                )
                # BOX_PLOT
                temp_result.append(_calc_box(data_frame, col_x, bins, col_y))
            else:
                pass
                # WARNING: _TODO
            Render.vizualise(Render(**kwrgs), temp_result, True)
            return temp_result  # if kwrgs.get("return_result") else None

        except NotImplementedError as error:  # _TODO
            LOGGER.info("Plot could not be obtained due to : %s", error)

    if col_x is None and col_y is None:
        intermediates = plot_df(data_frame, bins, force_cat, force_num)
        Render.vizualise(Render(**kwrgs), intermediates)
        return intermediates

    return list_of_intermediates
