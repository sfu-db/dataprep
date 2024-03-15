"""
In this module lives the type tree.
"""

from typing import Any, Dict, Optional, Type, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
from ..clean import validate_country, validate_lat_long

STRING_PANDAS_DTYPES = [pd.StringDtype]
STRING_DTYPES = STRING_PANDAS_DTYPES

CATEGORICAL_NUMPY_DTYPES = [bool, object]
CATEGORICAL_PANDAS_DTYPES = [pd.CategoricalDtype, pd.PeriodDtype]
CATEGORICAL_DTYPES = CATEGORICAL_NUMPY_DTYPES + CATEGORICAL_PANDAS_DTYPES + STRING_DTYPES

NUMERICAL_NUMPY_DTYPES = [np.number]
NUMERICAL_DTYPES = NUMERICAL_NUMPY_DTYPES

DATETIME_NUMPY_DTYPES = [np.datetime64]
DATETIME_PANDAS_DTYPES = [pd.DatetimeTZDtype]
DATETIME_DTYPES = DATETIME_NUMPY_DTYPES + DATETIME_PANDAS_DTYPES

NULL_VALUES = {
    float("NaN"),
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "n/a",
    "nan",
    "null",
    "",
}


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


class SmallCardNum(Numerical):
    """
    Numerical column with small cardinality (distinct values)
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


class GeoGraphy(Categorical):
    """
    Type GeoGraphy, Subtype of Categorical
    """


class GeoPoint(DType):
    """
    Type GeoPoint
    """


class LatLong(GeoPoint):
    """
    Type LatLong, Tuple
    """

    def __init__(self, lat_col: str, long_col: str) -> None:
        self.lat = lat_col
        self.long = long_col


############## End of the Type Tree ##############

DTypeOrStr = Union[DType, Type[DType], str, None]
DTypeDict = Union[Dict[str, Union[DType, Type[DType], str]], None]
DTypeDef = Union[Dict[str, Union[DType, Type[DType], str]], DType, Type[DType], None]


def detect_dtype(
    col: Union[dd.Series, pd.Series],
    head: pd.Series,
    known_dtype: Optional[DTypeDef] = None,
) -> DType:
    """
    Given a column, detect its type or transform its type according to users' specification

    Parameters
    ----------
    col: dask.datafram.Series or pd.Series
        A dataframe column
    head: pd.Series
        The first n rows of col. Used for type inference.
    known_dtype: Optional[Union[Dict[str, Union[DType, str]], DType]], default None
        A dictionary or single DType given by users to specify the types for designated columns or
        all columns. E.g.  known_dtype = {"a": Continuous, "b": "Nominal"} or
        known_dtype = {"a": Continuous(), "b": "nominal"} or
        known_dtype = Continuous() or known_dtype = "Continuous" or known_dtype = Continuous()
    detect_small_distinct: bool, default True
        Whether to detect numerical columns with small distinct values as categorical column.
    """

    if not known_dtype:
        return detect_without_known(col, head)

    if isinstance(known_dtype, dict):
        if col.name in known_dtype:
            dtype = normalize_dtype(known_dtype[col.name])
            return map_dtype(dtype)

    elif isinstance(normalize_dtype(known_dtype), DType):
        return map_dtype(normalize_dtype(known_dtype))

    return detect_without_known(col, head)


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


def detect_without_known(col: Union[dd.Series, pd.Series], head: pd.Series) -> DType:
    # pylint: disable=too-many-return-statements
    """
    This function detects dtypes of column when users didn't specify.
    """

    if is_continuous(col.dtype):
        # detect as categorical if distinct value is small
        if isinstance(col, dd.Series):
            nuniques = col.nunique_approx().compute()
        elif isinstance(col, pd.Series):
            nuniques = col.nunique()
        else:
            raise TypeError(f"unprocessed column type:{type(col)}")
        if nuniques < 10:
            return SmallCardNum()
        else:
            return Continuous()

    elif is_datetime(col.dtype):
        return DateTime()
    elif is_geography(head):
        return GeoGraphy()
    elif is_geopoint(head):
        return GeoPoint()
    else:
        return Nominal()


def is_dtype(dtype1: Any, dtype2: DType) -> bool:
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


def is_geography(head: pd.Series) -> bool:
    """
    Given a column, return if its type is a geography type
    """
    geo_ratio: float = np.sum(validate_country(head)) / head.shape[0]
    return geo_ratio > 0.8


def is_geopoint(head: pd.Series) -> bool:
    """
    Given a column, return if its type is a geopoint type
    """
    lat_long = pd.Series(head, dtype="string")
    lat_long_ratio: float = np.sum(validate_lat_long(lat_long)) / lat_long.shape[0]
    return lat_long_ratio > 0.8


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


def string_dtype_to_object(df: dd.DataFrame) -> dd.DataFrame:
    """
    Convert string dtype to object dtype
    """
    for col in df.columns:
        if any(isinstance(df[col].dtype, c) for c in STRING_DTYPES):
            df[col] = df[col].astype(object)

    return df


def drop_null(
    var: Union[dd.Series, pd.DataFrame, dd.DataFrame]
) -> Union[pd.Series, dd.Series, pd.DataFrame, dd.DataFrame]:
    """
    Drop the null values (specified in NULL_VALUES) from a series or DataFrame
    """

    if isinstance(var, (pd.Series, dd.Series)):
        if is_datetime(var.dtype):
            return var.dropna()
        return var[~var.isin(NULL_VALUES)]

    elif isinstance(var, (pd.DataFrame, dd.DataFrame)):
        df = var
        for values in df.columns:
            if is_datetime(df[values].dtype):
                df = df.dropna(subset=[values])
            else:
                df = df[~df[values].isin(NULL_VALUES)]
        return df

    raise ValueError("Input should be a Pandas/Dask Dataframe or Series")
