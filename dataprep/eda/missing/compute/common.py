"""Common parts for compute missing."""
from typing import Optional, Tuple

import dask.array as da
import dask.dataframe as dd

from ...dtypes import (
    Continuous,
    DTypeDef,
    Nominal,
    detect_dtype,
    is_dtype,
)

LABELS = ["Orignal data", "After drop missing values"]


def uni_histogram(
    srs: dd.Series,
    bins: int,
    dtype: Optional[DTypeDef] = None,
) -> Tuple[da.Array, ...]:
    """Calculate "histogram" for both numerical and categorical."""

    if is_dtype(detect_dtype(srs, dtype), Continuous()):

        counts, edges = da.histogram(srs, bins, range=[srs.min(), srs.max()])
        centers = (edges[:-1] + edges[1:]) / 2

        return counts, centers, edges

    elif is_dtype(detect_dtype(srs, dtype), Nominal()):
        # Dask array's unique is way slower than the values_counts on Series
        # See https://github.com/dask/dask/issues/2851
        # centers, counts = da.unique(arr, return_counts=True)

        value_counts = srs.value_counts()

        counts = value_counts.to_dask_array()
        centers = value_counts.index.to_dask_array()

        return (counts, centers)
    else:
        raise ValueError(f"Unsupported dtype {srs.dtype}")


def histogram(
    arr: da.Array,
    bins: Optional[int] = None,
    return_edges: bool = True,
    range: Optional[Tuple[int, int]] = None,  # pylint: disable=redefined-builtin
    dtype: Optional[DTypeDef] = None,
) -> Tuple[da.Array, ...]:
    """Calculate "histogram" for both numerical and categorical."""
    if len(arr.shape) != 1:
        raise ValueError("Histogram only supports 1-d array.")

    if is_dtype(detect_dtype(arr, dtype), Continuous()):
        if range is not None:
            minimum, maximum = range
        else:
            minimum, maximum = arr.min(axis=0), arr.max(axis=0)

        if bins is None:
            raise ValueError("num_bins cannot be None if calculating numerical histograms.")

        counts, edges = da.histogram(arr, bins, range=[minimum, maximum])
        centers = (edges[:-1] + edges[1:]) / 2

        if not return_edges:
            return counts, centers
        return counts, centers, edges
    elif is_dtype(detect_dtype(arr, dtype), Nominal()):
        # Dask array's unique is way slower than the values_counts on Series
        # See https://github.com/dask/dask/issues/2851
        # centers, counts = da.unique(arr, return_counts=True)

        srs = dd.from_dask_array(arr)
        value_counts = srs.value_counts()

        counts = value_counts.to_dask_array()
        centers = value_counts.index.to_dask_array()

        return (counts, centers)
    else:
        raise ValueError(f"Unsupported dtype {arr.dtype}")
