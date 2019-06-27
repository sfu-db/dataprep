"""
    This module implements the plot(df) function.
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.stats import kendalltau


from .__init__ import LOGGER, DEFAULT_PARTITIONS


class DataType(Enum):
    """
        Enumeration for storing the different types of data possible in a column
    """
    TYPE_NUM = 1
    TYPE_CAT = 2
    TYPE_UNSUP = 3


# Type aliasing
StringList = List[str]


def _calc_box_stats(grp_series: Any) -> Dict[str, Any]:
    stats: Dict[str, Any] = dict()
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
            if (stats["25%"] - 1.5 * stats["iqr"]) < value < (
                    stats["75%"] + 1.5 * stats["iqr"]):  # data is in the bound
                min_value = min(value, min_value)
                max_value = max(value, max_value)
            else:  # otherwise, outliers
                outliers.append(value)

        stats["min"] = min_value
        stats["max"] = max_value
    stats["outliers"] = outliers
    return stats


def _calc_box(
        dataframe: dd.DataFrame,
        col_x: str,
        col_y: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
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
        col_x, col_y) if (get_type(dataframe[col_x]) == DataType.TYPE_CAT) else (col_y, col_x)

    if col_y is None:
        col_series = dataframe[col_x]
        res = _calc_box_stats(col_series)
    else:
        for group in dataframe[cat_col].unique().compute():
            grp_series = dataframe.groupby(cat_col).get_group(group)[num_col]
            res[group] = _calc_box_stats(grp_series)

    return {"box_plot": res}


def _calc_statcked(
        dataframe: dd.DataFrame,
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
    grp_object = dataframe.groupby([col_x, col_y])

    grp_series = grp_object.count().compute().iloc[:, 0]
    # print (grp_series)
    return {"stacked_column_plot": dict(grp_series)}


def _calc_scatter(
        dataframe: dd.DataFrame,
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
    series_x = dataframe[col_x].compute()
    series_y = dataframe[col_y].compute()

    res = set()
    for each in zip(series_x, series_y):
        res.add(each)

    return {"scatter_plot": dict(res)}


def _calc_pie(dataframe: dd.DataFrame, col: str) -> Dict[str, Dict[str, float]]:
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
    grp_object = (dataframe.groupby(col)[col].count() / dataframe[col].size) * 100
    return {"pie_plot": dict(grp_object.compute())}


def _calc_bar(dataframe: dd.DataFrame, col: str) -> Dict[str, Dict[str, int]]:
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
    grp_object = dataframe.groupby(col)[col].count()
    return {"bar_plot": dict(grp_object)}


def _calc_hist_by_group(
        dataframe: dd.DataFrame,
        col_x: str,
        col_y: str,
        nbins: int = 10) -> Dict[str, Dict[str, Tuple[Any, Any]]]:
    """Returns the histogram array for the continuous
        distribution of values in the column given as the second argument
    _TODO write test
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

    grp_hist: Dict[str, Tuple[Any, Any]] = dict()
    hist_interm: List[Any] = list()
    grp_name_list: List[str] = list()

    for group in dataframe[col_cat].unique().compute():
        grp_series = dataframe.groupby(col_cat).get_group(group)[col_num]
        minv = grp_series.min().compute()
        maxv = grp_series.max().compute()
        hist = da.histogram(grp_series, range=[minv, maxv], bins=nbins)
        hist_interm.append(hist)
        grp_name_list.append(group)

    hist_interm, = dask.compute(hist_interm)

    for zipped_element in zip(grp_name_list, hist_interm):
        grp_hist[zipped_element[0]] = zipped_element[1]

    return {"histogram": grp_hist}


def _calc_hist(
        dataframe: dd.DataFrame,
        col: str,
        nbins: int = 10) -> Dict[str, Tuple[List[Union[int, float]], List[Union[int, float]]]]:
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
    hist_array, bins = da.histogram(dframe.values, range=[minv, maxv], bins=nbins)
    hist_array = hist_array.compute()

    if not hist_array.size == 0:
        return {'histogram': (hist_array, bins)}
    return {'histogram': (list(), list())}


def _calc_qqnorm(
        dataframe: dd.DataFrame,
        col: str,
        qrange: Optional[List[int]] = None) -> Dict[str, List[Tuple[float, float]]]:
    """
        Calculates points of the QQ plot of the given column of the data frame.
        :param dataframe - the input dataframe
        :param col - the input column of the dataframe
        :param qrange - the list of quantiles to be calculated. By default, all the percentiles are
        calculated.
    """
    points = list()
    if qrange is None:
        qrange = list(range(1, 101))

    dask_series = dataframe[col]
    try:
        size_ = dask_series.size.compute()
        np.random.seed(0)
        normal_points = np.sort(np.random.standard_normal(size=(size_, )))
        x_points = np.percentile(normal_points, q=qrange)
        y_points = dask_series.compute().sort_values().quantile([x / 100 for x in qrange])
        for point in zip(x_points, y_points):
            points.append(point)
    except TypeError:
        # _TODO
        pass

    if points:
        return {"qq_norm_plot": points}
    return {"qq_norm_plot": list()}


def _calc_kendall(
        data_a: np.ndarray, data_b: np.ndarray) -> Any:
    """
    :param data_a: the input numpy array
    :param data_b: the input numpy array
    :return: A float value which indicates the
    correlation of two numpy array
    """
    kendallta, _ = kendalltau(data_a, data_b)
    return kendallta


def _value_to_rank(array: np.ndarray) -> pd.Series:
    """
    :param array: the input numpy array
    :return: the output numpy array
    """
    array_ranks = pd.Series(array).rank()
    return array_ranks.values


def get_type(data: dd.Series) -> DataType:
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
        elif pd.api.types.is_numeric_dtype(data) and data.dropna().unique().size.compute() == 2:
            col_type = DataType.TYPE_CAT
        elif pd.api.types.is_numeric_dtype(data):
            col_type = DataType.TYPE_NUM
        else:
            col_type = DataType.TYPE_CAT
    except NotImplementedError as error:  # TO-DO
        LOGGER.info("Type cannot be determined due to : %s", error)

    return col_type


def plot_df(
        data_frame: dd.DataFrame,
        force_cat: Optional[StringList] = None,
        force_num: Optional[StringList] = None
) -> Dict[str, Union[Dict[str, Union[List[Any], Dict[Any, Any]]], Tuple[Any], List[Any],
                     Dict[Any, Any]]]:
    """
    Supporting funtion to the main plot function
    :param data_frame: dask dataframe
    :param force_cat: list of categorical columns defined explicitly
    :param force_num: list of numerical columns defined explicitly
    :return:
    """
    col_list = list()
    dask_result: List[Any] = list()

    for col in data_frame.columns:
        if data_frame[col].count().compute() == 0:
            col_list.append(col)
            dask_result.append(data_frame[col])

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
        pd_data_frame: pd.DataFrame,
        col_x: Optional[str] = None,
        col_y: Optional[str] = None,
        force_cat: Optional[StringList] = None,
        force_num: Optional[StringList] = None
) -> Dict[str, Union[Dict[str, Union[List[Any], Dict[Any, Any]]], Tuple[Any], List[Any],
                     Dict[Any, Any]]]:
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
    data_frame: dd.DataFrame = dd.from_pandas(pd_data_frame, npartitions=DEFAULT_PARTITIONS)

    result: Dict[str, Union[Dict[str, Union[List[Any], Dict[Any, Any]]], Tuple[Any], List[Any],
                            Dict[Any, Any]]] = dict()

    if col_x is None and col_y is None:
        result = plot_df(data_frame, force_cat, force_num)

    elif (col_x is None and col_y is not None) or (col_x is not None and col_y is None):

        target_col: str = cast(str, col_x if col_y is None else col_y)
        dask_result: List[Any] = list()

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
        temp_dask_result: Dict[str, Any] = dict()

        try:
            if type_y == DataType.TYPE_CAT and type_x == DataType.TYPE_NUM or \
                    type_y == DataType.TYPE_NUM and type_x == DataType.TYPE_CAT:
                # BOX_PER_GROUP
                temp_dask_result.update(_calc_box(data_frame, col_x, col_y))
                # HISTOGRAM_PER_GROUP
                temp_dask_result.update(_calc_hist_by_group(data_frame, col_x, col_y))

            elif type_x == DataType.TYPE_CAT and type_y == DataType.TYPE_CAT:
                temp_dask_result.update(_calc_statcked(data_frame, col_x, col_y))

            elif type_x == DataType.TYPE_NUM and type_y == DataType.TYPE_NUM:
                temp_dask_result.update(_calc_scatter(data_frame, col_x, col_y))
            else:
                pass
                # WARNING: _TODO
            result, = dask.compute(temp_dask_result)
        except NotImplementedError as error:  # _TODO
            LOGGER.info("Plot could not be obtained due to : %s", error)
    else:
        pass
        # _TODO to be added

    return result


def plot_correlation_pd(
        pd_data_frame: pd.DataFrame,
        method: str = 'pearson') -> np.ndarray:
    """

    :param pd_data_frame: the pandas data_frame for which plots
    are calculated for each column.
    :param method: Three method we can use to calculate
    the correlation matrix
    :return: A correlation matrix
    """
    if method == 'pearson':
        cal_matrix = pd_data_frame.values.T
        cov_xy = np.cov(cal_matrix)
        std_xy = np.sqrt(np.diag(cov_xy))
        result = cov_xy / std_xy[:, None] / std_xy[None, :]
    elif method == 'spearman':
        cal_matrix = pd_data_frame.values.T
        matrix_row, _ = np.shape(cal_matrix)
        for i in range(matrix_row):
            cal_matrix[i, :] = _value_to_rank(cal_matrix[i, :])
        cov_xy = np.cov(cal_matrix)
        std_xy = np.sqrt(np.diag(cov_xy))
        result = cov_xy / std_xy[:, None] / std_xy[None, :]
    elif method == 'kendall':
        cal_matrix = pd_data_frame.values.T
        matrix_row, _ = np.shape(cal_matrix)
        result = np.ones(shape=(matrix_row, matrix_row))
        result_list = []
        for i in range(matrix_row):
            for j in range(i + 1, matrix_row):
                tmp = dask.delayed(_calc_kendall)(
                    cal_matrix[i, :], cal_matrix[j, :])
                result_list.append(tmp)
        result_comp = dask.compute(*result_list)
        idx = 0
        for i in range(matrix_row):
            for j in range(i + 1, matrix_row):
                result[i][j] = result_comp[idx]
                result[j][i] = result[i][j]
                idx = idx + 1
    else:
        raise ValueError("Method Error")

    return result


def plot_correlation_pd_k(
        pd_data_frame: pd.DataFrame,
        k: int = 0,
        method: str = 'pearson') -> np.ndarray:
    """
    :param pd_data_frame: the pandas data_frame for which plots
    are calculated for each column.
    :param k: choose top-k
    :param method: Three method we can use to calculate
    the correlation matrix
    :return: A correlation matrix
    """
    result = plot_correlation_pd(pd_data_frame=pd_data_frame, method=method)
    matrix_row, _ = np.shape(result)
    result_re = np.reshape(np.triu(result),
                           (matrix_row * matrix_row,))
    idx = np.argsort(result_re)
    mask = np.zeros(shape=(matrix_row * matrix_row,))
    mask[idx[:k]] = 1
    mask[idx[-k:]] = 1
    result = np.multiply(result_re, mask)
    result = np.reshape(result,
                        (matrix_row, matrix_row))
    result += result.T - np.diag(result.diagonal())
    return result


def plot_correlation_pd_x_k(  # pylint: disable=too-many-locals
        pd_data_frame: pd.DataFrame,
        data_x: np.ndarray = None,
        k: int = 0) -> np.ndarray:
    """
    :param pd_data_frame: the pandas data_frame for which plots
    are calculated for each column.
    :param data_x: a valid column name of the dataframe which
    has been translated to numpy array
    :param k: choose top-k
    :return: A correlation matrix
    """
    cal_matrix = pd_data_frame.values.T
    cal_matrix = np.vstack((data_x, cal_matrix))

    cov_xy = np.cov(cal_matrix)
    std_xy = np.sqrt(np.diag(cov_xy))
    result_p = cov_xy / std_xy[:, None] / std_xy[None, :]
    row_p = result_p[0, :]
    row_p[0] = -1
    idx_p = np.argsort(row_p)

    matrix_row, _ = np.shape(cal_matrix)
    for i in range(matrix_row):
        cal_matrix[i, :] = _value_to_rank(cal_matrix[i, :])
    cov_xy = np.cov(cal_matrix)
    std_xy = np.sqrt(np.diag(cov_xy))
    result_s = cov_xy / std_xy[:, None] / std_xy[None, :]
    row_s = result_s[0, :]
    row_s[0] = -1
    idx_s = np.argsort(row_s)

    matrix_row, _ = np.shape(cal_matrix)
    result_k = np.ones(shape=(matrix_row, matrix_row))
    result_list = []
    for i in range(1, matrix_row):
        tmp = dask.delayed(_calc_kendall)(
            cal_matrix[0, :], cal_matrix[i, :])
        result_list.append(tmp)
    result_comp = dask.compute(*result_list)
    idx = 0
    for i in range(1, matrix_row):
        result_k[0][i] = result_comp[idx]
        result_k[i][0] = result_k[0][i]
        idx = idx + 1
    row_k = result_k[0, :]
    row_k[0] = -1
    idx_k = np.argsort(row_k)

    result = np.stack((sorted(row_p[idx_p[-k:]], reverse=True),
                       sorted(row_s[idx_s[-k:]], reverse=True),
                       sorted(row_k[idx_k[-k:]], reverse=True)))
    return result


def plot_correlation(
        pd_data_frame: pd.DataFrame,
        data_x: np.ndarray = None,
        data_y: np.ndarray = None,
        k: int = 0,
        method: str = 'pearson') -> np.ndarray:
    """
    :param pd_data_frame: the pandas data_frame for which plots are calculated for each
    column.
    :param data_x: a valid column name of the dataframe which has been translated
    to numpy array
    :param data_y: a valid column name of the dataframe which has been translated
    to numpy array
    :param k: choose top-k element
    :param method: Three method we can use to calculate the correlation matrix
    :return: A correlation matrix
    """
    if data_x is not None and data_y is not None and k != 0:
        pass
    elif data_x is not None and k != 0:
        result = plot_correlation_pd_x_k(pd_data_frame=pd_data_frame,
                                         data_x=data_x, k=k)
    elif k != 0:
        result = plot_correlation_pd_k(pd_data_frame=pd_data_frame,
                                       k=k, method=method)
    else:
        result = plot_correlation_pd(pd_data_frame=pd_data_frame, method=method)

    return result
