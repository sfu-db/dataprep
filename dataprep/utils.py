"""
    This module implements the utils function.
"""
from enum import Enum
from typing import Any

import pandas as pd
import dask.dataframe as dd
from IPython import get_ipython


from .__init__ import LOGGER


class DataType(Enum):
    """
        Enumeration for storing the different types of data possible in a column
    """
    TYPE_NUM = 1
    TYPE_CAT = 2
    TYPE_UNSUP = 3


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


def is_notebook() -> Any:
    """
    :return: whether it is running in jupyter notebook
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        return False
    except NameError:
        return False
