"""
Module for auxiliary type detection functions
"""

from enum import Enum, auto
from typing import Any

import numpy as np
import pandas as pd

CATEGORICAL_NUMPY_DTYPES = [np.bool, np.object]
CATEGORICAL_PANDAS_DTYPES = [pd.CategoricalDtype, pd.PeriodDtype]
CATEGORICAL_DTYPES = CATEGORICAL_NUMPY_DTYPES + CATEGORICAL_PANDAS_DTYPES

NUMERICAL_NUMPY_DTYPES = [np.number, np.datetime64]
NUMERICAL_PANDAS_DTYPES = [pd.DatetimeTZDtype]
NUMERICAL_DTYPES = NUMERICAL_NUMPY_DTYPES + NUMERICAL_PANDAS_DTYPES


class DType(Enum):
    """
    Possible dtypes for a column, currently we only support categorical and numerical dtypes.
    """

    Categorical = auto()
    Numerical = auto()


def is_categorical(dtype: Any) -> bool:
    """
    Given a type, return if that type is a categorical type
    """

    if is_numerical(dtype):
        return False

    if isinstance(dtype, np.dtype):
        dtype = dtype.type

        return any(issubclass(dtype, c) for c in CATEGORICAL_NUMPY_DTYPES)
    else:
        return any(isinstance(dtype, c) for c in CATEGORICAL_PANDAS_DTYPES)


def is_numerical(dtype: Any) -> bool:
    """
    Given a type, return if that type is a numerical type
    """
    if isinstance(dtype, np.dtype):
        dtype = dtype.type
        return any(issubclass(dtype, c) for c in NUMERICAL_NUMPY_DTYPES)
    else:
        return any(isinstance(dtype, c) for c in NUMERICAL_PANDAS_DTYPES)


def is_pandas_categorical(dtype: Any) -> bool:
    """
    Detect if a dtype is categorical and from pandas.
    """
    return any(isinstance(dtype, c) for c in CATEGORICAL_PANDAS_DTYPES)
