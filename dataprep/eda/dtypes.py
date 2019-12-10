"""
Module for auxiliary type detection functions
"""

from typing import Any

import numpy as np
import pandas as pd

CATEGORICAL_DTYPES = [pd.CategoricalDtype, np.bool, np.object, pd.PeriodDtype]
NUMERICAL_DTYPES = [np.number, np.datetime64, pd.DatetimeTZDtype]


def is_categorical(dtype: Any) -> bool:
    """
    Given a type, return if that type is a categorical type
    """
    if isinstance(dtype, np.dtype):
        dtype = dtype.type

    return not is_numerical(dtype) and any(
        issubclass(dtype, c) for c in CATEGORICAL_DTYPES
    )


def is_numerical(dtype: Any) -> bool:
    """
    Given a type, return if that type is a numerical type
    """
    if isinstance(dtype, np.dtype):
        dtype = dtype.type

    return any(issubclass(dtype, c) for c in NUMERICAL_DTYPES)
