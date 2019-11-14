"""
    This module implements the utils function.
"""
import logging
import random
import string
from enum import Enum, auto
from math import ceil
from typing import Any, Union

import dask
import dask.dataframe as dd
import pandas as pd


LOGGER = logging.getLogger(__name__)

DataType = None
get_type = lambda x: x


def is_notebook() -> Any:
    """
    :return: whether it is running in jupyter notebook
    """
    try:
        from IPython import get_ipython  # pylint: disable=import-outside-toplevel

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        return False
    except (NameError, ImportError):
        return False


def _rand_str(str_length: int = 20) -> Any:
    """
    :param str_length: The length of random string
    :return: A generated random string
    """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(str_length))


def to_dask(df: Union[pd.DataFrame, dd.DataFrame]) -> dd.DataFrame:
    if isinstance(df, dd.DataFrame):
        return df

    df_size = df.memory_usage(deep=True).sum()
    npartitions = ceil(df_size / 128 / 1024 / 1024)
    return dd.from_pandas(df, npartitions=npartitions)
