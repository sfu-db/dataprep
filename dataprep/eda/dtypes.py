"""
In this module lives the type tree.
"""


from typing import Any, Dict, Optional, Union, Type


import numpy as np
import pandas as pd
import dask.dataframe as dd

from ..errors import UnreachableError

CATEGORICAL_NUMPY_DTYPES = [np.bool, np.object]
CATEGORICAL_PANDAS_DTYPES = [pd.CategoricalDtype, pd.PeriodDtype]
CATEGORICAL_DTYPES = CATEGORICAL_NUMPY_DTYPES + CATEGORICAL_PANDAS_DTYPES

NUMERICAL_NUMPY_DTYPES = [np.number]
NUMERICAL_DTYPES = NUMERICAL_NUMPY_DTYPES

DATETIME_NUMPY_DTYPES = [np.datetime64]
DATETIME_PANDAS_DTYPES = [pd.DatetimeTZDtype]
DATETIME_DTYPES = DATETIME_NUMPY_DTYPES + DATETIME_PANDAS_DTYPES


class DType:
    """
    Root of Type Tree
    """


############## Syntactic DTypes ##############
class Categorical(DType):
    """
    Type Categorical
    """


class Nominal(Categorical):
    """
    Type Nominal, Subtype of Categorical
    """


class Ordinal(Categorical):
    """
    Type Ordinal, Subtype of Categorical
    """


class Numerical(DType):
    """
    Type Numerical
    """


class Continuous(Numerical):
    """
    Type Continuous, Subtype of Numerical
    """


class Discrete(Numerical):
    """
    Type Discrete, Subtype of Numerical
    """


############## Semantic DTypes ##############


class DateTime(Numerical):
    """
    Type DateTime, Subtype of Numerical
    """


class Text(Nominal):
    """
    Type Text, Subtype of Nominal
    """


############## End of the Type Tree ##############

DTypeOrStr = Union[DType, Type[DType], str, None]
DTypeDict = Union[Dict[str, Union[DType, Type[DType], str]], None]
DTypeDef = Union[Dict[str, Union[DType, Type[DType], str]], DType, Type[DType], None]


def detect_dtype(col: dd.Series, known_dtype: Optional[DTypeDef] = None,) -> DType:
    """
    Given a column, detect its type or transform its type according to users' specification

    Parameters
    ----------
    col: dask.datafram.Series
        A dataframe column
    known_dtype: Optional[Union[Dict[str, Union[DType, str]], DType]], default None
        A dictionary or single DType given by users to specify the types for designated columns or
        all columns. E.g.  known_dtype = {"a": Continuous, "b": "Nominal"} or
        known_dtype = {"a": Continuous(), "b": "nominal"} or
        known_dtype = Continuous() or known_dtype = "Continuous" or known_dtype = Continuous()
    """
    if not known_dtype:
        return detect_without_known(col)

    if isinstance(known_dtype, dict):
        if col.name in known_dtype:
            dtype = normalize_dtype(known_dtype[col.name])
            return map_dtype(dtype)

    elif isinstance(normalize_dtype(known_dtype), DType):
        return map_dtype(normalize_dtype(known_dtype))

    return detect_without_known(col)


def map_dtype(dtype: DType) -> DType:
    """
    Currently, we want to keep our Type System flattened.
    We will map Categorical() to Nominal() and Numerical() to Continuous()
    """
    if (
        isinstance(dtype, Categorical) is True
        and isinstance(dtype, Ordinal) is False
        and isinstance(dtype, Nominal) is False
    ):
        return Nominal()
    elif (
        isinstance(dtype, Numerical) is True
        and isinstance(dtype, Continuous) is False
        and isinstance(dtype, Discrete) is False
    ):
        return Continuous()
    else:
        return dtype


def detect_without_known(col: dd.Series) -> DType:
    """
    This function detects dtypes of column when users didn't specify.
    """
    if is_nominal(col.dtype):
        return Nominal()

    elif is_continuous(col.dtype):
        return Continuous()

    elif is_datetime(col.dtype):
        return DateTime()
    else:
        raise UnreachableError


def is_dtype(dtype1: DType, dtype2: DType) -> bool:
    """
    This function detects if dtype2 is dtype1.
    """
    return isinstance(dtype1, dtype2.__class__)


def normalize_dtype(dtype_repr: Any) -> DType:
    """
    This function normalizes a dtype repr.
    """
    normalized: DType
    str_dic = {
        "Categorical": Categorical,
        "Ordinal": Ordinal,
        "Nominal": Nominal,
        "Numerical": Numerical,
        "Continuous": Continuous,
        "Discrete": Discrete,
        "DateTime": DateTime,
        "Text": Text,
    }
    for str_dtype, dtype in str_dic.items():
        if isinstance(dtype_repr, str):
            if dtype_repr.lower() == str_dtype.lower():
                normalized = dtype()
                break

        elif isinstance(dtype_repr, dtype):
            normalized = dtype_repr
            break

        elif dtype_repr == dtype:
            normalized = dtype()
            break

    return normalized


def is_nominal(dtype: Any) -> bool:
    """
    Given a type, return if that type is a nominal type
    """

    if is_continuous(dtype) or is_datetime(dtype):
        return False

    if isinstance(dtype, np.dtype):
        dtype = dtype.type

        return any(issubclass(dtype, c) for c in CATEGORICAL_NUMPY_DTYPES)
    else:
        return any(isinstance(dtype, c) for c in CATEGORICAL_PANDAS_DTYPES)


def is_continuous(dtype: Any) -> bool:
    """
    Given a type, return if that type is a continuous type
    """
    dtype = dtype.type
    return any(issubclass(dtype, c) for c in NUMERICAL_NUMPY_DTYPES)


def is_datetime(dtype: Any) -> bool:
    """
    Given a type, return if that type is a datetime type
    """
    if isinstance(dtype, np.dtype):
        dtype = dtype.type
        return any(issubclass(dtype, c) for c in DATETIME_NUMPY_DTYPES)
    else:
        return any(isinstance(dtype, c) for c in DATETIME_PANDAS_DTYPES)


def is_pandas_categorical(dtype: Any) -> bool:
    """
    Detect if a dtype is categorical and from pandas.
    """
    return any(isinstance(dtype, c) for c in CATEGORICAL_PANDAS_DTYPES)
