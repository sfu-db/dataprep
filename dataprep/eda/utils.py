"""
    This module implements the utils function.
"""
import logging
import random
import string
from enum import Enum
from math import ceil
from typing import Any, Union

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from bokeh.models import Legend
from bokeh.plotting import Figure

LOGGER = logging.getLogger(__name__)

# TODO: Remove old stuffs
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
        elif (
            pd.api.types.is_numeric_dtype(data)
            and dask.compute(data.dropna().unique().size) == 2
        ):
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
        # pytype: disable=import-error
        from IPython import get_ipython  # pylint: disable=import-outside-toplevel

        # pytype: enable=import-error

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        return False
    except (NameError, ImportError):
        return False


def rand_str(str_length: int = 20) -> str:
    """
    Generate a random string
    """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(str_length))


def to_dask(df: Union[pd.DataFrame, dd.DataFrame]) -> dd.DataFrame:
    """
    Convert a dataframe to a dask dataframe.
    """
    if isinstance(df, dd.DataFrame):
        return df

    df_size = df.memory_usage(deep=True).sum()
    npartitions = ceil(df_size / 128 / 1024 / 1024)
    return dd.from_pandas(df, npartitions=npartitions)


def sample_n(arr: np.ndarray, n: int) -> np.ndarray:  # pylint: disable=C0103
    """
    Sample n values uniformly from the range of the `arr`,
    not from the distribution of `arr`'s elems.
    """
    if len(arr) <= n:
        return arr

    subsel = np.linspace(0, len(arr) - 1, n)
    subsel = np.floor(subsel).astype(int)
    return arr[subsel]


def relocate_legend(fig: Figure, loc: str) -> Figure:
    """
    Relocate legend(s) from center to `loc`
    """
    remains = []
    targets = []
    for layout in fig.center:
        if isinstance(layout, Legend):
            targets.append(layout)
        else:
            remains.append(layout)
    fig.center = remains
    for layout in targets:
        fig.add_layout(layout, loc)

    return fig


def cut_long_name(name: str, max_len: int = 12) -> str:
    """
    If the name is longer than `max_len`,
    cut it to `max_len` length and append "..."
    """
    if len(name) <= max_len:
        return name
    return f"{name[:max_len]}..."


def fuse_missing_perc(name: str, perc: float) -> str:
    """
    Append (x.y%) to the name if `perc` is not 0
    """
    if perc == 0:
        return name

    return f"{name} ({perc:.1%})"
