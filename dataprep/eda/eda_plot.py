"""
    This module implements the plot(df) function.
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

from .__init__ import LOGGER, DEFAULT_PARTITIONS


class DataType(Enum):
    TYPE_NUM = 1
    TYPE_CAT = 2
    TYPE_UNSUP = 3


DEFAULT_RANGE = [x for x in range(1, 101)]
# Type aliasing
StringList = List[str]


def _calc_box(
    dataframe: pd.DataFrame,
    col_x: str,
    col_y: str
) -> Dict[str, Dict[Any, Any]]:
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
    cat_col, num_col = (
        col_x, col_y) if (get_type(dataframe[col_x]) == DataType.TYPE_CAT) else (col_y, col_x)

    dask_df = dd.from_pandas(dataframe, npartitions=DEFAULT_PARTITIONS)
    res: Dict[str, Dict[str, Any]] = dict()
    stats: Dict[str, Any] = dict()

    for group in dask_df[cat_col].unique().compute():
        stats = {}
        grp_series = dask_df.groupby(cat_col).get_group(group)[num_col]
        quantiles = grp_series.quantile([.25, .50, .75]).compute()
        stats["25%"], stats["50%"], stats["75%"] = quantiles[.25], quantiles[.50], quantiles[.75]
        stats["iqr"] = stats["75%"] - stats["25%"]

        outliers = list()
        grp_series = grp_series.compute()
        if len(grp_series) == 1:
            stats["min"] = grp_series.reset_index().iloc[0, 1]
            stats["max"] = stats["min"]
        else:
            min_value, max_value = np.inf, -np.inf

            for value in grp_series:
                if (stats["25%"] - 1.5 * stats["iqr"]) < value < (stats["75%"] + 1.5 * stats["iqr"]):  # data is in the bound
                    min_value = min(value, min_value)
                    max_value = max(value, max_value)
                else:  # otherwise, outliers
                    outliers.append(value)

            stats["min"] = min_value
            stats["max"] = max_value
        stats["outliers"] = outliers

        res[group] = stats

    res, = dask.compute(res)

    if not res:
        return {"box_plot": res}
    return dict()


def _calc_statcked(
    dataframe: pd.DataFrame,
    col_x: str,
    col_y: str
) -> Dict[str, Dict[Tuple[Any, Any], int]]:
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

    dask_df = dask.dataframe.from_pandas(
        dataframe, npartitions=DEFAULT_PARTITIONS)
    grp_object = dask_df.groupby([col_x, col_y])

    grp_series = grp_object.count().compute().iloc[:, 0]
    # print (grp_series)
    if not grp_series:
        return {"stacked_column_plot": dict(grp_series)}
    return dict()


def _calc_scatter(
        dataframe: pd.DataFrame,
        col_x: str,
        col_y: str
) -> Dict[str, Dict[Union[int, float], Union[int, float]]]:
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
    dask_df = dask.dataframe.from_pandas(
        dataframe, npartitions=DEFAULT_PARTITIONS)
    series_x = dask_df[col_x].compute()
    series_y = dask_df[col_y].compute()

    res = set()
    for each in zip(series_x, series_y):
        res.add(each)

    if not res:
        return {"scatter_plot": dict(res)}
    return dict()


def _calc_pie(dataframe: pd.DataFrame, col: str) -> Dict[str, Dict[str, int]]:
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
    dask_df = dd.from_pandas(dataframe, npartitions=DEFAULT_PARTITIONS)
    grp_object = (dask_df.groupby(col)[col].count() / dask_df[col].size) * 100

    if not grp_object:
        return {"pie_plot": dict(grp_object.compute())}
    return dict()


def _calc_bar(dataframe: pd.DataFrame, col: str) -> Dict[str, Dict[str, int]]:
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
    dask_df = dd.from_pandas(dataframe, npartitions=DEFAULT_PARTITIONS)
    grp_object = dask_df.groupby(col)[col].count()

    if not grp_object:
        return {"bar_plot": dict(grp_object.compute())}
    return dict()


def _calc_hist_by_group(
        dataframe: pd.DataFrame,
        col_x: str,
        col_y: str,
        nbins: int = 10) -> Dict[str, Dict[str, Tuple[Any, Any]]]:
    """Returns the histogram array for the continuous
        distribution of values in the column given as the second argument
    TODO write test
    Parameters
    __________
    dataframe : the input pandas dataframe
    col : the str column of dataframe for which hist array needs to be
    calculated

    Returns
    __________
    np.array : An array of values representing histogram for the input col
    """
    col_cat, col_num = (col_x, col_y) if (get_type(dataframe[col_x]) == DataType.TYPE_CAT) \
                                        else (col_y, col_x)
    dask_df = dd.from_pandas(dataframe, npartitions=DEFAULT_PARTITIONS)

    grp_hist: Dict[str, Tuple[Any, Any]] = dict()
    for group in dask_df[col_cat].unique().compute():
        grp_series = dask_df.groupby(col_cat).get_group(group)[col_num]
        minv = grp_series.min().compute()
        maxv = grp_series.max().compute()
        dframe = dd.from_array(grp_series).dropna()
        hist_array, bins = da.histogram(dframe.values, range=[ minv, maxv], bins=nbins)
        grp_hist[group] = (hist_array, bins)

    return {"histogram": grp_hist}


def _calc_hist(
        dataframe: pd.DataFrame,
        col: str,
        nbins: int = 10) -> Dict[str, List]:
    """Returns the histogram array for the continuous
        distribution of values in the column given as the second argument

    Parameters
    __________
    dataframe : the input pandas dataframe
    col : the str column of dataframe for which hist array needs to be
    calculated

    Returns
    __________
    np.array : An array of values representing histogram for the input col
    """
    minv = dataframe[col].min()
    maxv = dataframe[col].max()
    dframe = dd.from_array(dataframe[col]).dropna()
    hist_array, = da.histogram(dframe.values, range=[minv, maxv], bins=nbins)

    if not hist_array:
        return {'histogram': hist_array.compute()}
    return dict()


def _calc_qqnorm(
        dataframe: pd.DataFrame,
        col: str,
        qrange: Optional[list] = DEFAULT_RANGE) -> Dict[str, List[Tuple[float, float]]]:
    """
        Calculates points of the QQ plot of the given column of the data frame.
        :param dataframe - the input dataframe
        :param col - the input column of the dataframe
        :param qrange - the list of quantiles to be calculated. By default, all the percentiles are
        calculated.
    """
    points = list()
    dask_df = dd.from_pandas(dataframe, npartitions=DEFAULT_PARTITIONS)
    dask_series = dask_df[col]
    try:
        size_ = dask_series.size.compute()
        np.random.seed(0)
        normal_points = np.sort(np.random.standard_normal(size=(1, size_)))
        x_points = np.percentile(normal_points, q=qrange)
        y_points = dask_series.compute().sort_values().quantile([
            x / 100 for x in qrange])
        for point in zip(x_points, y_points):
            points.append(point)
    except TypeError:
        # TODO
        pass

    if not points:
        return {"qq_norm_plot": points}
    return dict()


def get_type(data: pd.Series) -> DataType:
    """ Returns the type of the input data.
        Identified types are according to the DataType Enumeration.

    Parameter
    __________
    The data for which the type needs to be identified.

    Returns
    __________
    str representing the type of the data.
    """

    col_type = DataType.TYPE_UNSUP
    try:
        if pd.api.types.is_bool_dtype(data):
            col_type = DataType.TYPE_CAT
        elif pd.api.types.is_numeric_dtype(data) and data.count() == 2:
            col_type = DataType.TYPE_CAT
        elif pd.api.types.is_numeric_dtype(data):
            col_type = DataType.TYPE_NUM
        else:
            col_type = DataType.TYPE_CAT
    except NotImplementedError as error:  # TO-DO
        LOGGER.info("Type cannot be determined due to : %s", error)

    return col_type


def plot_df(
        data_frame: pd.DataFrame,
        force_cat: Optional[StringList] = None,
        force_num: Optional[StringList] = None
) -> Dict[str, List]:
    """
    Supporting funtion to the main plot function
    :param data_frame: pandas dataframe
    :param force_cat: list of categorical columns defined explicitly
    :param force_num: list of numerical columns defined explicitly
    :return:
    """
    col_list = list()
    dask_result = list()

    for col in data_frame.columns:
        if data_frame[col].count() == 0:
            col_list.append(col)
            dask_result.append([])
            continue

        elif get_type(data_frame[col]) == DataType.TYPE_CAT or (
                force_cat is not None and col in force_cat):
            cnt_series = dask.delayed(_calc_bar)(data_frame, col)
            dask_result.append(cnt_series)
            col_list.append(col)

        elif get_type(data_frame[col]) == DataType.TYPE_NUM or (
                force_num is not None and col in force_num):
            hist = dask.delayed(_calc_hist)(data_frame, col)
            dask_result.append(hist)
            col_list.append(col)

    column_dict = dict()
    computed_res, = dask.compute(dask_result)

    for each in zip(col_list, computed_res):
        column_dict[each[0]] = each[1]

    return column_dict


def plot(
    data_frame: pd.DataFrame,
    col_x: str = None,
    col_y: Optional[str] = None,
    force_cat: Optional[StringList] = None,
    force_num: Optional[StringList] = None
) -> Union[List, Dict[str, Any]]:
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
    kwargs : TO-DO

    Returns
    __________
    dict : A (column: [array/dict]) dict to encapsulate the
    intermediate results.
    """
    result = dict()

    if col_x is None and col_y is None:
        result = plot_df(data_frame, force_cat, force_num)

    if (col_x is None and col_y is not None) or (
            col_x is not None and col_y is None):

        target_col = col_x if col_y is None else col_y
        dask_result = list()

        if data_frame[target_col].count() == 0:
            dask_result.append([])

        elif get_type(data_frame[target_col]) == DataType.TYPE_CAT or (
                force_cat is not None and target_col in force_cat):
            # BAR_PLOT
            dask_result.append(dask.delayed(_calc_bar)(data_frame, target_col))
            # PIE_CHART
            dask_result.append(dask.delayed(_calc_pie)(data_frame, target_col))

        elif get_type(data_frame[target_col]) == DataType.TYPE_NUM or (
                force_num is not None and target_col in force_num):
            # HISTOGRAM
            dask_result.append(dask.delayed(_calc_hist)(data_frame, target_col))
            # BOX_PLOT
            dask_result.append(dask.delayed(_calc_bar)(data_frame, target_col))
            # QQ-NORM
            dask_result.append(dask.delayed(_calc_qqnorm)(data_frame, target_col))

        column_dict = {target_col: dask.compute(dask_result)}
        result = column_dict

    elif col_x is not None and col_y is not None:
        type_x = get_type(data_frame[col_x])
        type_y = get_type(data_frame[col_y])
        dask_result = dict()

        try:
            if type_y == DataType.TYPE_CAT and type_x == DataType.TYPE_NUM or \
                    type_y == DataType.TYPE_NUM and type_x == DataType.TYPE_CAT:
                # BOX_PER_GROUP
                dask_result.update(_calc_box(data_frame, col_x, col_y))
                # HISTOGRAM_PER_GROUP
                #dask_result.append(_calc_hist_by_group(data_frame, col_x, col_y))

            elif type_x == DataType.TYPE_CAT and type_y == DataType.TYPE_CAT:
                dask_result.update(_calc_statcked(data_frame, col_x, col_y))

            elif type_x == DataType.TYPE_NUM and type_y == DataType.TYPE_NUM:
                dask_result.update(_calc_scatter(data_frame, col_x, col_y))
            else:
                pass
                # WARNING: TODO
            result, = dask.compute(dask_result)
        except NotImplementedError as error:  # TODO
            LOGGER.info("Plot could not be obtained due to : %s", error)
            result = {'Error': str(error)}
    else:
        pass
        # TODO to be added

    return result
