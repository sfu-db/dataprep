import pandas as pd
import numpy as np
from typing import Any

CATEGORICAL_DTYPES = [pd.CategoricalDtype, np.bool, np.object, pd.PeriodDtype]
NUMERICAL_DTYPES = [np.number, np.datetime64, pd.DatetimeTZDtype]


def is_categorical(dtype: Any) -> bool:
    if isinstance(dtype, np.dtype):
        dtype = dtype.type

    return not is_numerical(dtype) and any(
        issubclass(dtype, c) for c in CATEGORICAL_DTYPES
    )


def is_numerical(dtype: Any) -> bool:
    if isinstance(dtype, np.dtype):
        dtype = dtype.type

    return any(issubclass(dtype, c) for c in NUMERICAL_DTYPES)
