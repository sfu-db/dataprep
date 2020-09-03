"""Defines DataArray."""

from math import ceil
from typing import Any, List, Tuple, Union, cast

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

from .dtypes import NUMERICAL_DTYPES

DataFrame = Union[pd.DataFrame, dd.DataFrame, "DataArray"]


class DataArray:
    """DataArray provides an abstraction over dask DataFrame
    and dask Array. The reason is that sometimes some algorithms
    only works on the Array and not the DataFrame. However,
    the cost for getting the array from a dask DataFrame is
    non trivial. Instead of computing the array from a dask
    DataFrame again and again, it would be better do that once.

    Other reasons to have a separate DataArray abstraction includes
    converting the column names to string without modifying the
    DataFrame from user, and preprocessings like dropna and type detection.

    Parameters
    ----------
    df
        The DataFrame
    value_length
        Whether to compute the lengths of the array.
        This triggers a read on the data thus expensive if the passed in df
        is a dask DataFrame.
        If a pandas DataFrame passed in, lengths will always be compute.
    repartition
        Whether to repartition the DataFrame into 128M chunks.
    """

    _ddf: dd.DataFrame
    _values: da.Array
    _columns: pd.Index

    def __init__(
        self, df: DataFrame, value_length: bool = False, repartition: bool = True,
    ) -> None:

        if isinstance(df, dd.DataFrame):
            is_pandas = False
            self._ddf = df
        elif isinstance(df, DataArray):
            self._ddf = df._ddf
            self._values = df._values
            self._columns = df._columns
            return
        elif isinstance(df, pd.DataFrame):
            is_pandas = True
            if repartition:
                df_size = df.memory_usage(deep=True).sum()
                npartitions = ceil(df_size / 128 / 1024 / 1024)
                self._ddf = dd.from_pandas(df, npartitions=npartitions)
            else:
                self._ddf = dd.from_pandas(df)
        else:
            raise ValueError(f"{type(df)} not supported")

        if value_length or is_pandas:
            self._values = self._ddf.to_dask_array(lengths=True)
        else:
            self._values = self._ddf.to_dask_array()

        self._columns = self._ddf.columns.astype(str)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the data"""
        return cast(Tuple[int, ...], self._values.shape)

    @property
    def columns(self) -> pd.Index:
        """Return the columns of the DataFrame."""
        return self._columns

    @property
    def values(self) -> da.Array:
        """Return the array representation of the data."""
        return self._values

    @property
    def frame(self) -> dd.DataFrame:
        """Return the underlying dataframe."""
        return self._ddf

    def compute_length(self) -> None:
        """Compute the length of values inplace."""
        not_computed = any(np.isnan(shape) for shape in self.shape)
        if not_computed:
            self._values = self._values.compute_chunk_sizes()

    def select_dtypes(self, include: List[Any]) -> "DataArray":
        """Return a new DataArray with designated dtype columns."""
        subdf = self._ddf.select_dtypes(include)  # pylint: disable=W0212
        cidx = [self.columns.get_loc(col) for col in subdf.columns]
        df = DataArray(subdf)
        df._values = self.values[:, cidx]  # pylint: disable=W0212
        return df

    def select_num_columns(self) -> "DataArray":
        """Return a new DataArray with numerical dtype columns."""
        df = self.select_dtypes(NUMERICAL_DTYPES)
        df._values = df._values.astype(np.float)  # pylint: disable=W0212
        return df
