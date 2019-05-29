import logging
from typing import Dict, List, Union, Optional

# noinspection PyUnresolvedReferences
import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd


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
    x = dask_df.groupby(col)[col].count()
    return dict(x.compute())


def calc_hist(dataframe: pd.DataFrame, col: str, nbins: int = 10) \
        -> List[float]:
    """Returns the histogram array for the continuous
        distribution of values in the column given as the second argument

    Parameters
    __________
    dataframe : the input pandas dataframe
    col : the str column of dataframe for which hist array needs to be calculated

    Returns
    __________
    np.array : An array of values representing histogram for the input col 

        """
    minv = dataframe[col].min()
    maxv = dataframe[col].max()
    dframe = dd.from_array(dataframe[col]).dropna()
    h, _ = da.histogram(dframe.values, range=[minv, maxv], bins=nbins)
    return h


def get_type(data: pd.Series) -> str:
    """ Returns the type of the input data.
        Identified types are:
        'TYPE_CAT' - if data is categorical.
        'TYPE_NUM' - if data is numeric.
        'TYPE_UNSUP' - type not supported.
         #TODO

    Parameter
    __________
    The data for which the type needs to be identified.

    Returns
    __________
    str representing the type of the data.
    """

    col_type = None
    try:
        if pd.api.types.is_bool_dtype(data):
            col_type = 'TYPE_CAT'
        elif pd.api.types.is_numeric_dtype(data) and data.count() == 2:
            col_type = 'TYPE_CAT'
        elif pd.api.types.is_numeric_dtype(data):
            col_type = 'TYPE_NUM'
        else:
            col_type = 'TYPE_CAT'
    except Exception as e:
        logging.error("An Error occured: " + str(e))
        col_type = 'TYPE_UNSUP'

    return col_type


# Type aliasing
String_List = List[str]


def plot(df: pd.DataFrame, force_cat: Optional[String_List] = None,
         force_num: Optional[String_List] = None) \
        -> Dict[str, Union[np.array, dict]]:
    """ Returns an intermediate representation for the plots of
        different columns in the dataframe.

    Parameters
    dataframe: the pandas dataframe for which plots are calculated for each column.
    force_cat: the list of columns which have to considered of type 'TYPE_CAT'
    force_num: the list of columns which have to considered of type 'TYPE_NUM'

    Returns
    __________
    dict : A (column: [array/dict]) dict to encapsulate the intermediate results.
    """
    result = list()

    debug = []

    for i, col in enumerate(df.columns):
        if df[col].count() == 0:
            debug.append(col)
            result.append([])
            continue

        elif get_type(df[col]) == 'TYPE_CAT' or (
                force_cat is not None and col in force_cat):
            cnt_series = dask.delayed(calc_count)(df, col)
            result.append(cnt_series)
            debug.append(col)

        elif get_type(df[col]) == 'TYPE_NUM' or (
                force_num is not None and col in force_num):
            hist = dask.delayed(calc_hist)(df, col)
            result.append(hist)
            debug.append(col)

    computed_res, = dask.compute(result)
    column_dict = dict()

    for each in zip(debug, computed_res):
        column_dict[each[0]] = each[1]

    return column_dict
