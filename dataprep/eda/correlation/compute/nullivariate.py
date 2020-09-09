"""Implementations of correlations.

Currently this boils down to pandas' implementation."""

from functools import partial
from typing import Dict, Optional, Tuple

import dask
import dask.array as da
import numpy as np
import pandas as pd

from ...data_array import DataArray
from ...intermediate import Intermediate
from .common import CorrelationMethod


def _calc_nullivariate(
    df: DataArray,
    *,
    value_range: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
) -> Intermediate:

    if value_range is not None and k is not None:
        raise ValueError("value_range and k cannot be present in both")

    cordx, cordy, corrs = correlation_nxn(df)

    # The computations below is not expensive (scales with # of columns)
    # So we do them in pandas

    (corrs,) = dask.compute(corrs)

    dfs = {}
    for method, corr in corrs.items():
        ndf = pd.DataFrame(
            {
                "x": df.columns[cordx],
                "y": df.columns[cordy],
                "correlation": corr.ravel(),
            }
        )
        ndf = ndf[cordy > cordx]  # Retain only lower triangle (w/o diag)

        if k is not None:
            thresh = ndf["correlation"].abs().nlargest(k).iloc[-1]
            ndf = ndf[(ndf["correlation"] >= thresh) | (ndf["correlation"] <= -thresh)]
        elif value_range is not None:
            mask = (value_range[0] <= ndf["correlation"]) & (
                ndf["correlation"] <= value_range[1]
            )
            ndf = ndf[mask]

        dfs[method.name] = ndf

    return Intermediate(
        data=dfs,
        axis_range=list(df.columns.unique()),
        visual_type="correlation_heatmaps",
    )


def correlation_nxn(
    df: DataArray,
) -> Tuple[np.ndarray, np.ndarray, Dict[CorrelationMethod, da.Array]]:
    """
    Calculation of a n x n correlation matrix for n columns

    Returns
    -------
        The long format of the correlations
    """

    ncols = len(df.columns)
    cordx, cordy = np.meshgrid(range(ncols), range(ncols))
    cordx, cordy = cordy.ravel(), cordx.ravel()

    corrs = {
        CorrelationMethod.Pearson: _pearson_nxn(df),
        CorrelationMethod.Spearman: _spearman_nxn(df),
        CorrelationMethod.KendallTau: _kendall_tau_nxn(df),
    }

    return cordx, cordy, corrs


def _pearson_nxn(df: DataArray) -> da.Array:
    """Calculate column-wise pearson correlation."""
    return (
        df.frame.repartition(npartitions=1)
        .map_partitions(partial(pd.DataFrame.corr, method="pearson"))
        .to_dask_array()
    )


def _spearman_nxn(df: DataArray) -> da.Array:
    """Calculate column-wise spearman correlation."""
    return (
        df.frame.repartition(npartitions=1)
        .map_partitions(partial(pd.DataFrame.corr, method="spearman"))
        .to_dask_array()
    )


def _kendall_tau_nxn(df: DataArray) -> da.Array:
    """Calculate column-wise kendalltau correlation."""
    return (
        df.frame.repartition(npartitions=1)
        .map_partitions(partial(pd.DataFrame.corr, method="kendall"))
        .to_dask_array()
    )


## The code below is the correlation algorithms for array. Since we don't have
## block-wise algorithms for spearman and kendalltal, it might be more suitable
## to just use the pandas version of correlation.
## The correlations from pandas use double for-loops but they write them in cython
## and they are super fast already.
#
# def _pearson_nxn(data: da.Array) -> da.Array:
#     """Calculate column-wise pearson correlation."""

#     mean = data.mean(axis=0)[None, :]
#     dem = data - mean

#     num = dem.T @ dem

#     std = data.std(axis=0, keepdims=True)
#     dom = data.shape[0] * (std * std.T)

#     correl = num / dom

#     return correl


# def _spearman_nxn(array: da.Array) -> da.Array:
#     rank_array = (
#         array.rechunk((-1, None))  #! TODO: avoid this
#         .map_blocks(partial(rankdata, axis=0))
#         .rechunk("auto")
#     )
#     return _pearson_nxn(rank_array)


# def _kendall_tau_nxn(array: da.Array) -> da.Array:
#     """Kendal Tau correlation outputs an n x n correlation matrix for n columns."""

#     _, ncols = array.shape

#     corrmat = []
#     for _ in range(ncols):
#         corrmat.append([float("nan")] * ncols)

#     for i in range(ncols):
#         corrmat[i][i] = 1.0

#     for i in range(ncols):
#         for j in range(i + 1, ncols):

#             tmp = kendalltau(array[:, i], array[:, j])

#             corrmat[j][i] = corrmat[i][j] = da.from_delayed(
#                 tmp, shape=(), dtype=np.float
#             )

#     return da.stack(corrmat)
