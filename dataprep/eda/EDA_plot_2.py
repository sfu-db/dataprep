import logging
from typing import Dict, Union

import dask
import pandas as pd

from .EDA_plot import get_type


def calc_box(dataframe: pd.DataFrame, col_x: str, col_y: str) \
        -> Dict[str, dict]:
    """ Returns intermediate stats of the box plot
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
    cat_col, num_col = (col_x, col_y) if (
            get_type(dataframe[col_x]) == 'TYPE_CAT') else (col_y, col_x)

    dask_df = dask.dataframe.from_pandas(dataframe, npartitions=1)
    grp_object = dask_df.groupby(cat_col)
    # print (cat_col)
    groups = list(dask_df[cat_col].unique().compute())
    res = dict()

    for group in groups:
        stats = dict()
        grp_series = grp_object.get_group(group)[num_col]
        quantiles = grp_series.quantile([.25, .50, .75]).compute()
        stats['25%'], stats['50%'], stats['75%'] = dask.delayed(
            quantiles[.25]), dask.delayed(quantiles[.50]), dask.delayed(
            quantiles[.75])
        stats['iqr'] = stats['75%'] - stats['25%']
        outliers = list()
        grp_series = grp_series.compute()
        if len(grp_series) == 1:
            stats['min'] = grp_series.reset_index().iloc[0, 1]
            stats['max'] = stats['min']
        else:
            for i in grp_series.index:
                if (grp_series[i] < stats['25%'].compute() - (
                        1.5 * stats['iqr'].compute())) or (
                        grp_series[i] > stats['75%'].compute() + (
                        1.5 * stats['iqr'].compute())):
                    outliers.append(grp_series[i])
                    grp_series.drop(index=i, inplace=True)
            stats['min'] = grp_series.min()
            stats['max'] = grp_series.max()

        stats['outliers'] = outliers

        res[group] = stats

    res, = dask.compute(res)
    return res


def calc_statcked(dataframe: pd.DataFrame, col_x: str, col_y: str) \
        -> Dict[tuple, int]:
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


def calc_scatter(dataframe: pd.DataFrame, col_x: str, col_y: str) \
        -> Dict[Union[int, float], Union[int, float]]:
    """ 
    TODO: WARNING: For very large amount of points, implement Heat Map.

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


logging.basicConfig(level=logging.DEBUG)


def plot(dataframe: pd.DataFrame, col_x: str, col_y: str) -> Dict[str, dict]:
    """ Returns intermediate stats of the bi-variate plots 
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

    type_x = get_type(dataframe[col_x])
    type_y = get_type(dataframe[col_y])
    result = None

    try:
        if type_y == 'TYPE_CAT' and type_x == 'TYPE_NUM' or type_y == 'TYPE_NUM' and type_x == 'TYPE_CAT':
            result = calc_box(dataframe, col_x, col_y)

        elif type_x == 'TYPE_CAT' and type_y == 'TYPE_CAT':
            result = calc_statcked(dataframe, col_x, col_y)

        elif type_x == 'TYPE_NUM' and type_y == 'TYPE_NUM':
            result = calc_scatter(dataframe, col_x, col_y)
        else:
            pass
            # WARNING: TODO
    except Exception as e:
        logging.debug('Failed to plot due to ' + str(e))
        result = dict()

    return result
