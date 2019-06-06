"""
    This module implements the plot(df) function.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.delayed import Delayed

LOGGER = logging.getLogger(__name__)


# Type aliasing
StringList = List[str]


def calc_box(
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
    cat_col, num_col = (col_x, col_y) if (get_type(dataframe[col_x]) == "TYPE_CAT") else (col_y, col_x)

    dask_df = dd.from_pandas(dataframe, npartitions=1)
    res = dict()
    stats: Dict[str, Delayed] = dict()

    for group in dask_df[cat_col].unique().compute():
        stats = {}
        grp_series = dask_df.groupby(cat_col).get_group(group)[num_col]
        quantiles = grp_series.quantile([.25, .50, .75]).compute()
        stats["25%"], stats["50%"], stats["75%"] = \
            dask.delayed(quantiles[.25]), dask.delayed(quantiles[.50]), dask.delayed(quantiles[.75])
        stats["iqr"] = stats["75%"] - stats["25%"]

        outliers = list()
        grp_series = grp_series.compute()
        if len(grp_series) == 1:
            stats["min"] = grp_series.reset_index().iloc[0, 1]
            stats["max"] = stats["min"]
        else:
            min_value, max_value = np.inf, -np.inf

            p25 = stats["25%"].compute()
            p75 = stats["75%"].compute()
            iqr = stats["iqr"].compute()

            for value in grp_series:
                if p25 - 1.5 * iqr < value < p75 + 1.5 * iqr:  # data is in the bound
                    min_value = min(value, min_value)
                    max_value = max(value, max_value)
                else:  # otherwise, outliers
                    outliers.append(value)

            stats["min"] = min_value
            stats["max"] = max_value
        stats["outliers"] = outliers

        res[group] = stats

    res, = dask.compute(res)
    return res


def calc_statcked(
    dataframe: pd.DataFrame,
    col_x: str,
    col_y: str
) -> Dict[Tuple[Any, Any], int]:
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

    dask_df = dask.dataframe.from_pandas(dataframe, npartitions=1)
    grp_object = dask_df.groupby([col_x, col_y])

    grp_series = grp_object.count().compute().iloc[:, 0]
    # print (grp_series)
    return dict(grp_series)


def calc_scatter(
        dataframe: pd.DataFrame,
        col_x: str,
        col_y: str
) -> Dict[Union[int, float], Union[int, float]]:
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
    dask_df = dask.dataframe.from_pandas(dataframe, npartitions=1)
    series_x = dask_df[col_x].compute()
    series_y = dask_df[col_y].compute()

    res = set()
    for each in zip(series_x, series_y):
        res.add(each)

    return dict(res)


def calc_count(dataframe: pd.DataFrame, col: str) -> Dict[str, int]:
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
    dask_df = dd.from_pandas(dataframe, npartitions=1)
    grp_object = dask_df.groupby(col)[col].count()
    return dict(grp_object.compute())


def calc_hist(
    dataframe: pd.DataFrame,
    col: str,
    nbins: int = 10
) -> List[float]:
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
    hist_array, _ = da.histogram(dframe.values, range=[minv, maxv], bins=nbins)
    return hist_array.compute()


def get_type(data: pd.Series) -> str:
    """ Returns the type of the input data.
        Identified types are:
        "TYPE_CAT" - if data is categorical.
        "TYPE_NUM" - if data is numeric.
        "TYPE_UNSUP" - type not supported.
         TO-DO

    Parameter
    __________
    The data for which the type needs to be identified.

    Returns
    __________
    str representing the type of the data.
    """

    col_type = "TYPE_UNSUP"
    try:
        if pd.api.types.is_bool_dtype(data):
            col_type = "TYPE_CAT"
        elif pd.api.types.is_numeric_dtype(data) and data.count() == 2:
            col_type = "TYPE_CAT"
        elif pd.api.types.is_numeric_dtype(data):
            col_type = "TYPE_NUM"
        else:
            col_type = "TYPE_CAT"
    except NotImplementedError as error:  # TO-DO
        LOGGER.info("Type cannot be determined due to : %s", error)

    return col_type


def plot_df(
        data_frame: pd.DataFrame,
        force_cat: Optional[StringList] = None,
        force_num: Optional[StringList] = None
) -> dict:
    """
        Supporting funtion to the main plot function
    :param data_frame: pandas dataframe
    :param force_cat: list of categorical columns defined explicitly
    :param force_num: list of numerical columns defined explicitly
    :return:
    """
    col_list = []
    dask_result = list()

    for col in data_frame.columns:
        if data_frame[col].count() == 0:
            col_list.append(col)
            dask_result.append([])
            continue

        elif get_type(data_frame[col]) == "TYPE_CAT" or (
                force_cat is not None and col in force_cat):
            cnt_series = dask.delayed(calc_count)(data_frame, col)
            dask_result.append(cnt_series)
            col_list.append(col)

        elif get_type(data_frame[col]) == "TYPE_NUM" or (
                force_num is not None and col in force_num):
            hist = dask.delayed(calc_hist)(data_frame, col)
            dask_result.append(hist)
            col_list.append(col)

    column_dict = dict()
    computed_res, = dask.compute(dask_result)

    for each in zip(col_list, computed_res):
        column_dict[each[0]] = each[1]

    return column_dict


def plot(
    data_frame: pd.DataFrame,
    col_x: Optional[str] = None,
    col_y: Optional[str] = None,
    force_cat: Optional[StringList] = None,
    force_num: Optional[StringList] = None
) -> dict:
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

    elif col_x is None and col_y is not None or col_x is not None and col_y is None:

        target_col = col_x if col_y is None else col_y
        dask_result = list()

        if data_frame[target_col].count() == 0:
            dask_result.append([])

        elif get_type(data_frame[target_col]) == "TYPE_CAT" or (
                force_cat is not None and target_col in force_cat):
            dask_result.append(dask.delayed(calc_count)(data_frame, target_col))

        elif get_type(data_frame[target_col]) == "TYPE_NUM" or (
                force_num is not None and target_col in force_num):
            dask_result.append(dask.delayed(calc_hist)(data_frame, target_col))

        column_dict = {target_col: dask.compute(dask_result)}
        result = column_dict

    elif col_x is not None and col_y is not None:
        type_x = get_type(data_frame[col_x])
        type_y = get_type(data_frame[col_y])

        try:
            if type_y == "TYPE_CAT" and type_x == "TYPE_NUM" or \
                    type_y == "TYPE_NUM" and type_x == "TYPE_CAT":
                result = calc_box(data_frame, col_x, col_y)

            elif type_x == "TYPE_CAT" and type_y == "TYPE_CAT":
                result = calc_statcked(data_frame, col_x, col_y)

            elif type_x == "TYPE_NUM" and type_y == "TYPE_NUM":
                result = calc_scatter(data_frame, col_x, col_y)
            else:
                pass
                # WARNING: TODO
        except NotImplementedError as error:  # TODO
            LOGGER.info("Plot could not be obtained due to : %s", error)
            result = dict()
    else:
        pass
        # TODO to be added

    return result
