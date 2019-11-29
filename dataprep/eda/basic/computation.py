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

from ...utils import DataType, get_type
from ..common import Intermediate, sample_n
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
    stats["lw"] = grp_series[grp_series > stats["tf"] - 1.5 * stats["iqr"]].min()
    stats["uw"] = grp_series[grp_series < stats["sf"] + 1.5 * stats["iqr"]].max()

    if len(grp_series) == 1:
        val = grp_series.reset_index().iloc[0, 1]
        stats["max"], stats["min"], stats["max_outlier"] = (val,) * 3
        stats["outliers"] = list(val)
    else:
        bounded = grp_series[
            (grp_series > (stats["tf"] - 1.5 * stats["iqr"]))
            & (grp_series < (stats["sf"] + 1.5 * stats["iqr"]))
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


def _calc_box(
    dataframe: dd.DataFrame, col_x: str, col_y: Optional[str] = None
) -> Intermediate:
    """
    Returns intermediate stats of the box plot
    of columns col_x and col_y.

    PARAMETERS
    __________
    dataframe: the input dataframe
    col_x : a valid column name of the dataframe
    col_y : a valid column name of the dataframe

    RETURNS
    __________
    a (column_name: data) dict storing the intermediate results
    """
    res: Dict[str, Any] = dict()
    cat_col, num_col = (
        (col_x, col_y)
        if (get_type(dataframe[col_x]) == DataType.TYPE_CAT)
        else (col_y, col_x)
    )

    if col_y is None:
        col_series = dataframe[col_x]
        res[col_x] = __calc_box_stats(col_series)
    else:
        for group in dask.compute(dataframe[cat_col].unique())[0]:
            grp_series = dataframe.groupby([cat_col]).get_group(group)[num_col]
            res[group] = __calc_box_stats(grp_series)
    raw_data = {"df": dataframe, "col_x": cat_col, "col_y": num_col}
    result = {"box_plot": res}
    return Intermediate(result, raw_data)


def _calc_statcked(dataframe: dd.DataFrame, col_x: str, col_y: str) -> Intermediate:
    """ Returns intermediate stats of the stacked column plot
            of columns col_x and col_y.

    PARAMETERS
    __________
    dataframe: the input dataframe
    col_x : a valid column name of the dataframe
    col_y : a valid column name of the dataframe


    RETURNS
    __________
    a (column_name: data) dict storing the intermediate results
    """
    grp_object = dataframe.groupby([col_x, col_y])
    grp_series = dask.compute(grp_object[col_x].count())[0]
    raw_data = {"df": dataframe, "col_x": col_x, "col_y": col_y}
    result = {"stacked_column_plot": dict(grp_series)}
    return Intermediate(result, raw_data)


def _calc_scatter(dataframe: dd.DataFrame, col_x: str, col_y: str) -> Intermediate:
    """
        TO-DO: WARNING: For very large amount of points, implement Heat Map.
        Returns intermediate stats of the scattered plot
        of columns col_x and col_y.

        PARAMETERS
        __________
        dataframe: the input dataframe
        col_x : a valid column name of the dataframe
        col_y : a valid column name of the dataframe


        RETURNS
        __________
        a (column_name: data) dict storing the intermediate results
    """
    series_x = dask.compute(dataframe[col_x])[0]
    series_y = dask.compute(dataframe[col_y])[0]

    res = list()
    for each in zip(series_x, series_y):
        res.append((round(each[0], 2), round(each[1], 2)))

    raw_data = {"df": dataframe, "col_x": col_x, "col_y": col_y}
    result = {"scatter_plot": res}
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
    result = {"pie_plot": result_dict}
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
    result = {"bar_plot": dict(grp_object), "missing": [miss_vals]}
    return Intermediate(result, raw_data)


def _calc_hist_by_group(
    dataframe: dd.DataFrame, col_x: str, col_y: str, bins: int
) -> Intermediate:
    """Returns the histogram array for the continuous
        distribution of values in the column given as the second argument
    _TODO write test
    Parameters
    __________
    dataframe : the input pandas dataframe
    col : the str column of dataframe for which hist array needs to be
    calculated
    bins : number of bins to use in the histogram

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

    for group in dask.compute(dataframe[col_cat].unique())[0]:
        grp_series = dataframe.groupby([col_cat]).get_group(group)[col_num]
        minv = dask.compute(grp_series.min())[0]
        maxv = dask.compute(grp_series.max())[0]
        hist = da.histogram(grp_series, range=[minv, maxv], bins=bins)
        hist_interm.append(hist)
        grp_name_list.append(group)

    (hist_interm,) = dask.compute(hist_interm)

    for zipped_element in zip(grp_name_list, hist_interm):
        grp_hist[zipped_element[0]] = zipped_element[1]

    return Intermediate(
        {"histogram": grp_hist, "missing": [0]},
        {"df": dataframe, "col_x": col_x, "col_y": col_y, "bins": bins},
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
    __________
    np.array : An array of values representing histogram for the input col
    """
    raw_data = {"df": dataframe, "col_x": col_x, "col_y": None, "bins": bins}
    if dask.compute(dataframe[col_x].size)[0] == 0:
        return Intermediate({"histogram": (list(), list())}, raw_data)

    (minv,) = dask.compute(dataframe[col_x].min())
    (maxv,) = dask.compute(dataframe[col_x].max())

    dframe = dataframe[col_x].dropna().values
    hist_array = None
    bins_array = None
    if isinstance(dframe, dask.array.core.Array):
        hist_array, bins_array = da.histogram(dframe, range=[minv, maxv], bins=bins)
        (hist_array,) = dask.compute(hist_array)
    elif isinstance(dframe, np.ndarray):
        (dframe,) = dask.compute(dframe)
        minv = 0 if np.isnan(dframe.min()) else dframe.min()
        maxv = 0 if np.isnan(dframe.max()) else dframe.max()
        hist_array, bins_array = np.histogram(dframe, bins=bins, range=[minv, maxv])
    bins_array = cast(Any, bins_array)

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
    """
    Returns an intermediate representation for the plots of
        different columns in the data_frame.

    Parameters
    data_frame: the pandas data_frame for which plots are calculated for each
    column.
    col_x : A column in the data_frame.
    col_y : A column in the data_frame.
    force_cat: the list of columns which have to considered of type "TYPE_CAT"
    force_num: the list of columns which have to considered of type "TYPE_NUM"
    bins: the number of bins to show for the plot(df, x) histogram
    value_range: the range of values to plot in the plot(df, x) histogram
    kwargs : TO-DO

    Returns
    __________
    dict : A (column: [array/dict]) dict to encapsulate the
    intermediate results.

    for column in pd_data_frame:
        values = list()
        for x in pd_data_frame[column]:
            if type(x)=='str':
                values.append(re.escape(x))
            else:
                values.append(x)
        pd_data_frame[column] = values
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
            # BAR_PLOT
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
            dask_result.append(_calc_box(data_frame, target_col))

            # QQ-NORM
            dask_result.append(_calc_qqnorm(data_frame, target_col))
        Render.vizualise(Render(**kwrgs), dask_result, True)
        return dask_result  # if kwrgs.get("return_result") else None

    if col_x is not None and col_y is not None:
        type_x = get_type(data_frame[col_x])
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
                temp_result.append(_calc_box(data_frame, col_x, col_y))
                # HISTOGRAM_PER_GROUP
                temp_result.append(_calc_hist_by_group(data_frame, col_x, col_y, bins))

            elif type_x == DataType.TYPE_CAT and type_y == DataType.TYPE_CAT:
                temp_result.append(_calc_statcked(data_frame, col_x, col_y))

            elif type_x == DataType.TYPE_NUM and type_y == DataType.TYPE_NUM:
                temp_result.append(_calc_scatter(data_frame, col_x, col_y))
            else:
                pass
                # WARNING: _TODO
            Render.vizualise(Render(**kwrgs), temp_result)
            return temp_result  # if kwrgs.get("return_result") else None

        except NotImplementedError as error:  # _TODO
            LOGGER.info("Plot could not be obtained due to : %s", error)

    if col_x is None and col_y is None:
        Render.vizualise(
            Render(**kwrgs), plot_df(data_frame, bins, force_cat, force_num)
        )
        return plot_df(
            data_frame, bins, force_cat, force_num
        )  # if kwrgs.get("return_result") else None

    return list_of_intermediates
