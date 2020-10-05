"""Defines DataArray."""

from functools import reduce
from math import ceil
from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

from .dtypes import CATEGORICAL_PANDAS_DTYPES, NUMERICAL_DTYPES

DataFrame = Union[pd.DataFrame, dd.DataFrame, "DataArray"]


# class DataArray:
#     """DataArray provides an abstraction over dask DataFrame
#     and dask Array. The reason is that sometimes some algorithms
#     only works on the Array and not the DataFrame. However,
#     the cost for getting the array from a dask DataFrame is
#     non trivial. Instead of computing the array from a dask
#     DataFrame again and again, it would be better do that once.

#     Other reasons to have a separate DataArray abstraction includes
#     converting the column names to string without modifying the
#     DataFrame from user, and preprocessings like dropna and type detection.

#     Parameters
#     ----------
#     df
#         The DataFrame
#     value_length
#         Whether to compute the lengths of the array.
#         This triggers a read on the data thus expensive if the passed in df
#         is a dask DataFrame.
#         If a pandas DataFrame passed in, lengths will always be compute.
#     repartition
#         Whether to repartition the DataFrame into 128M chunks.
#     """

#     _ddf: dd.DataFrame
#     _values: Dict[str, da.Array]
#     _nulls: da.Array
#     _columns: pd.Index

#     def __init__(
#         self, df: DataFrame, value_length: bool = False, repartition: bool = True,
#     ) -> None:

#         if isinstance(df, dd.DataFrame):
#             is_pandas = False
#             self._ddf = df
#         elif isinstance(df, DataArray):
#             self._ddf = df._ddf
#             self._values = df._values
#             self._columns = df._columns
#             return
#         elif isinstance(df, pd.DataFrame):
#             is_pandas = True
#             if repartition:
#                 df_size = df.memory_usage(deep=True).sum()
#                 npartitions = ceil(df_size / 128 / 1024 / 1024)
#                 self._ddf = dd.from_pandas(df, npartitions=npartitions)
#             else:
#                 self._ddf = dd.from_pandas(df)
#         else:
#             raise ValueError(f"{type(df)} not supported")

#         self._columns = self._ddf.columns.astype(str)

#         self._values = {}
#         for col in self._ddf.columns:
#             if isinstance(self._ddf[col].dtype, pd.CategoricalDtype):
#                 self._values[col] = self._ddf[col].astype(str).to_dask_array()
#             else:
#                 self._values[col] = self._ddf[col].to_dask_array()

#         if value_length or is_pandas:
#             self.compute_length()
#         else:
#             self._derive_nulls()

#     @property
#     def columns(self) -> pd.Index:
#         """Return the columns of the DataFrame."""
#         return self._columns

#     @property
#     def nulls(self) -> da.Array:
#         """Return the nullity array of the data."""
#         return self._nulls

#     @property
#     def shape(self) -> Tuple[int, int]:
#         """Return the shape of the data"""
#         return self._values[self.columns[0]].shape[0], len(self.columns)

#     @property
#     def values(self) -> da.Array:
#         """Return the array representation of the data."""
#         return da.concatenate(
#             [self._values[col][:, None] for col in self.columns],
#             axis=1,
#             allow_unknown_chunksizes=True,
#         )

#     @property
#     def frame(self) -> dd.DataFrame:
#         """Return the underlying dataframe."""
#         return self._ddf

#     def compute_length(self) -> None:
#         """Compute the length of values inplace."""

#         not_computed = np.isnan(self.shape[0])
#         if not_computed:
#             # Compute the chunk size for the first column,
#             # then apply that info to all other columns.
#             # Since all the column data are coming from a same dataframe,
#             # we are still good here
#             col = self._values[self.columns[0]]
#             col.compute_chunk_sizes()

#             # Here we use dask private API to set chunks
#             for _, val in self._values.items():
#                 val._chunks = col.chunks

#             self._derive_nulls()

#     def select_dtypes(self, include: List[Any]) -> "DataArray":
#         """Return a new DataArray with designated dtype columns."""
#         subdf = self._ddf.select_dtypes(include)  # pylint: disable=W0212
#         cidx = [self.columns.get_loc(col) for col in subdf.columns]
#         df = DataArray(subdf)
#         df._values = {col: self._values[col] for col in df.columns}
#         df._nulls = self.nulls[:, cidx]  # pylint: disable=W0212
#         return df

#     def select_num_columns(self) -> "DataArray":
#         """Return a new DataArray with numerical dtype columns."""
#         df = self.select_dtypes(NUMERICAL_DTYPES)
#         return df

#     def _derive_nulls(self) -> None:
#         # no need to use dict for nulls since it is uniformly a boolean array
#         self._nulls = da.concatenate(
#             [da.isnull(self._values[col])[:, None] for col in self._ddf.columns],
#             axis=1,
#             allow_unknown_chunksizes=True,
#         )


class DataArray:
    """DataArray provides an abstraction over dask DataFrame
    and dask Array. The reason is that sometimes some algorithms
    only works on the Array and not the DataFrame. However,
    the cost for getting the array from a dask DataFrame (with known length)
    is non trivial. Instead of computing the array from a dask
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
    _nulls: Union[da.Array, np.ndarray]
    _columns: pd.Index
    _head: Optional[pd.DataFrame] = None

    def __init__(
        self,
        df: DataFrame,
        value_length: bool = False,
        repartition: bool = True,
    ) -> None:
        if isinstance(df, (dd.Series, pd.Series)):
            df = df.to_frame()

        # numpy does not understand pandas types
        cat_cols = [
            col
            for col, dtype in df.dtypes.iteritems()
            for dtype_ in CATEGORICAL_PANDAS_DTYPES
            if isinstance(dtype, dtype_)
        ]

        if isinstance(df, dd.DataFrame):
            is_pandas = False
            if cat_cols and df.shape[1] != 0:
                df = df.astype({col: np.object for col in cat_cols})
            self._ddf = df
        elif isinstance(df, DataArray):
            self._ddf = df._ddf
            self._values = df._values
            self._columns = df._columns
            return
        elif isinstance(df, pd.DataFrame):
            is_pandas = True
            df = df.astype({col: np.object for col in cat_cols})
            if repartition:
                df_size = df.memory_usage(deep=True).sum()
                npartitions = ceil(df_size / 128 / 1024 / 1024)
                self._ddf = dd.from_pandas(df, npartitions=npartitions)
            else:
                self._ddf = dd.from_pandas(df, chunksize=-1)
        else:
            raise ValueError(f"{type(df)} not supported")

        self._columns = self._ddf.columns.astype(str)

        if value_length or is_pandas:
            self._values = self._ddf.to_dask_array(lengths=True)
        else:
            self._values = self._ddf.to_dask_array()
        self._nulls = self.frame.isnull().to_dask_array()
        self._nulls._chunks = self.values.chunks

    @property
    def columns(self) -> pd.Index:
        """Return the columns of the DataFrame."""
        return self._columns

    @property
    def dtypes(self) -> pd.Series:
        """Returns the dtypes of the DataFrame."""
        return self._ddf.dtypes

    @property
    def nulls(self) -> da.Array:
        """Return the nullity array of the data."""
        return self._nulls

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the data"""
        return cast(Tuple[int, int], self.values.shape)

    @property
    def values(self) -> da.Array:
        """Return the array representation of the data."""
        return self._values

    @property
    def frame(self) -> dd.DataFrame:
        """Return the underlying dataframe."""
        return self._ddf

    @property
    def head(self) -> pd.DataFrame:
        """Return the head of the DataFrame, if not exist, read it."""
        if self._head is None:
            self._head = self.frame.head()
        return self._head

    def compute(self, type: str = "lengths") -> None:  # pylint: disable=redefined-builtin
        """Compute the lengths or materialize the null values inplace.

        Parameters
        ----------
        type
            Can be lengths or nulls. lengths will compute the array chunk sizes and nulls
            will compute and materialize the null values as well as the lengths of the chunks.

        """

        if type == "lengths":
            not_computed = np.isnan(self.shape[0])
            if not_computed:
                self._values = self.frame.to_dask_array(lengths=True)
                self._nulls = self.frame.isnull().to_dask_array()
                self._nulls._chunks = self.values.chunks
        elif type == "nulls":
            x = self.nulls
            # Copied from compute_chunk_sizes
            # pylint: disable=invalid-name
            chunk_shapes = x.map_blocks(
                _get_chunk_shape,
                dtype=int,
                chunks=tuple(len(c) * (1,) for c in x.chunks) + ((x.ndim,),),
                new_axis=x.ndim,
            )

            c = []
            for i in range(x.ndim):
                s = x.ndim * [0] + [i]
                s[i] = slice(None)
                s = tuple(s)

                c.append(tuple(chunk_shapes[s]))

            chunks_, nulls = dask.compute(tuple(c), self.nulls)

            chunks = tuple([tuple([int(chunk) for chunk in chunks]) for chunks in chunks_])
            # pylint: enable=invalid-name
            self._nulls = nulls
            self._values._chunks = chunks
        else:
            raise ValueError(f"{type} not supported.")

    def select_dtypes(self, include: List[Any]) -> "DataArray":
        """Return a new DataArray with designated dtype columns."""
        subdf = self._ddf.select_dtypes(include)  # pylint: disable=W0212
        return self[subdf.columns]

    def select_num_columns(self) -> "DataArray":
        """Return a new DataArray with numerical dtype columns."""
        df = self.select_dtypes(NUMERICAL_DTYPES)
        return df

    def __getitem__(self, indexer: Union[Sequence[str], str]) -> "DataArray":
        """Return a new DataArray select by column names."""
        if isinstance(indexer, str):
            indexer = [indexer]

        subdf = self._ddf[indexer]  # pylint: disable=W0212
        cidx = [self.columns.get_loc(col) for col in subdf.columns]
        df = DataArray(subdf)
        df._values = self.values[:, cidx]  # pylint: disable=W0212

        if df.shape[1] != 0:
            # coerce the array to it's minimal type
            dtype = reduce(np.promote_types, df.dtypes.values)
            if df._values.dtype != dtype:
                df._values = df._values.astype(dtype)

        df._nulls = self.nulls[:, cidx]  # pylint: disable=W0212
        if self._head is not None:
            df._head = self.head[subdf.columns]  # pylint: disable=W0212
        return df


def _get_chunk_shape(arr: np.ndarray) -> np.ndarray:
    """Given an (x,y,...) N-d array, returns (1,1,...,N) N+1-d array"""
    shape = np.asarray(arr.shape, dtype=int)
    return shape[len(shape) * (None,) + (slice(None),)]
