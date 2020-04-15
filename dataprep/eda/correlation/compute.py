"""
    This module implements the intermediates computation
    for plot_correlation(df) function.
"""
import sys
from enum import Enum, auto
from operator import itruediv
from typing import Dict, Optional, Sequence, Tuple, Union

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from ...errors import UnreachableError
from ..dtypes import NUMERICAL_DTYPES
from ..intermediate import Intermediate
from ..utils import to_dask

__all__ = ["compute_correlation"]


class CorrelationMethod(Enum):
    """
    Supported correlation methods
    """

    Pearson = auto()
    Spearman = auto()
    KendallTau = auto()


def compute_correlation(
    df: Union[pd.DataFrame, dd.DataFrame],
    x: Optional[str] = None,
    y: Optional[str] = None,
    *,
    value_range: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
) -> Intermediate:
    """
    Parameters
    ----------
    df
        The pandas dataframe for which plots are calculated for each column.
    x
        A valid column name of the dataframe
    y
        A valid column name of the dataframe
    value_range
        If the correlation value is out of the range, don't show it.
    k
        Choose top-k element
    """
    # pylint: disable=too-many-locals,too-many-statements,too-many-branches

    df = to_dask(df)
    df.columns = [str(e) for e in df.columns]  # convert column names to string

    if x is None and y is None:  # pylint: disable=no-else-return
        assert (value_range is None) or (
            k is None
        ), "value_range and k cannot be present in both"

        df = df.select_dtypes(NUMERICAL_DTYPES)
        if len(df.columns) == 0:
            return Intermediate(visual_type=None)

        data = df.to_dask_array()
        # TODO Can we remove this? Without the computing, data has unknown rows so da.cov will fail.
        data.compute_chunk_sizes()

        cordx, cordy, corrs = correlation_nxn(data)
        cordx, cordy = dd.from_dask_array(cordx), dd.from_dask_array(cordy)
        columns = df.columns

        dfs = {}
        for method, corr in corrs.items():
            df = dd.concat([cordx, cordy, dd.from_dask_array(corr)], axis=1)
            df.columns = ["x", "y", "correlation"]
            df = df[df["y"] > df["x"]]  # Retain only lower triangle (w/o diag)

            if k is not None:
                thresh = df["correlation"].abs().nlargest(k).compute().iloc[-1]
                df = df[(df["correlation"] >= thresh) | (df["correlation"] <= -thresh)]
            elif value_range is not None:
                mask = (value_range[0] <= df["correlation"]) & (
                    df["correlation"] <= value_range[1]
                )
                df = df[mask]

            # Translate int x,y coordinates to categorical labels
            # Hint the return type of the function to dask through param "meta"
            df["x"] = df["x"].apply(lambda e: columns[e], meta=("x", np.object))
            df["y"] = df["y"].apply(lambda e: columns[e], meta=("y", np.object))
            dfs[method.name] = df.compute()

        return Intermediate(
            data=dfs,
            axis_range=list(columns.unique()),
            visual_type="correlation_heatmaps",
        )
    elif x is not None and y is None:
        df = df.select_dtypes(NUMERICAL_DTYPES)
        if len(df.columns) == 0:
            return Intermediate(visual_type=None)
        assert x in df.columns, f"{x} not in numerical column names"

        columns = df.columns[df.columns != x]
        xarr = df[x].to_dask_array().compute_chunk_sizes()
        data = df[columns].to_dask_array().compute_chunk_sizes()

        funcs = [pearson_1xn, spearman_1xn, kendall_tau_1xn]

        dfs = {}
        for meth, func in zip(CorrelationMethod, funcs):
            indices, corrs = func(xarr, data, value_range=value_range, k=k)
            if len(indices) == 0:
                print(
                    f"Correlation for {meth.name} is empty, try to broaden the value_range.",
                    file=sys.stderr,
                )
            df = pd.DataFrame(
                {
                    "x": np.full(len(indices), x),
                    "y": columns[indices],
                    "correlation": corrs,
                }
            )
            dfs[meth.name] = df

        return Intermediate(data=dfs, visual_type="correlation_single_heatmaps")
    elif x is None and y is not None:
        raise ValueError("Please give the value to x instead of y")
    elif x is not None and y is not None:
        assert value_range is None
        assert x in df.columns, f"{x} not in columns names"
        assert y in df.columns, f"{y} not in columns names"
        df = df.select_dtypes(NUMERICAL_DTYPES)
        if x in df.columns and y in df.columns:
            coeffs, df, influences = scatter_with_regression(
                df[x].values.compute_chunk_sizes(),
                df[y].values.compute_chunk_sizes(),
                k=k,
                sample_size=1000,
            )
            result = {
                "coeffs": dd.compute(coeffs)[0],
                "data": df.rename(columns={"x": x, "y": y}).compute(),
            }

            assert (influences is None) == (k is None)

            if influences is not None and k is not None:
                infidx = np.argsort(influences)
                labels = np.full(len(influences), "=")
                # pylint: disable=invalid-unary-operand-type
                labels[infidx[-k:]] = "-"  # type: ignore
                # pylint: enable=invalid-unary-operand-type
                labels[infidx[:k]] = "+"
                result["data"]["influence"] = labels

            return Intermediate(**result, visual_type="correlation_scatter")
        else:
            return Intermediate(visual_type=None)

    raise UnreachableError


def scatter_with_regression(
    xarr: da.Array, yarr: da.Array, sample_size: int, k: Optional[int] = None
) -> Tuple[Tuple[float, float], dd.DataFrame, Optional[np.ndarray]]:
    """
    Calculate pearson correlation on 2 given arrays.

    Parameters
    ----------
    xarr : da.Array
    yarr : da.Array
    sample_size : int
    k : Optional[int] = None
        Highlight k points which influence pearson correlation most

    Returns
    -------
    Intermediate
    """
    if k == 0:
        raise ValueError("k should be larger than 0")

    mask = ~(da.isnan(xarr) | da.isnan(yarr))
    xarr = da.from_array(np.array(xarr)[mask])
    yarr = da.from_array(np.array(yarr)[mask])
    xarrp1 = da.vstack([xarr, da.ones_like(xarr)]).T
    xarrp1 = xarrp1.rechunk((xarrp1.chunks[0], -1))
    (coeffa, coeffb), _, _, _ = da.linalg.lstsq(xarrp1, yarr)

    if sample_size < len(xarr):
        samplesel = np.random.choice(len(xarr), int(sample_size))
        xarr = xarr[samplesel]
        yarr = yarr[samplesel]

    df = dd.concat([dd.from_dask_array(arr) for arr in [xarr, yarr]], axis=1)
    df.columns = ["x", "y"]

    if k is None:
        return (coeffa, coeffb), df, None

    influences = pearson_influence(xarr, yarr)
    return (coeffa, coeffb), df, influences


def pearson_influence(xarr: da.Array, yarr: da.Array) -> da.Array:
    """
    Calculating the influence for deleting a point on the pearson correlation
    """
    assert (
        xarr.shape == yarr.shape
    ), f"The shape of xarr and yarr should be same, got {xarr.shape}, {yarr.shape}"

    # Fast calculating the influence for removing one element on the correlation
    n = len(xarr)

    x2, y2 = da.square(xarr), da.square(yarr)
    xy = xarr * yarr

    # The influence is vectorized on xarr and yarr, so we need to repeat all the sums for n times

    xsum = da.ones(n) * da.sum(xarr)
    ysum = da.ones(n) * da.sum(yarr)
    xysum = da.ones(n) * da.sum(xy)
    x2sum = da.ones(n) * da.sum(x2)
    y2sum = da.ones(n) * da.sum(y2)

    # Note: in we multiply (n-1)^2 to both denominator and numerator to avoid divisions.
    numerator = (n - 1) * (xysum - xy) - (xsum - xarr) * (ysum - yarr)

    varx = (n - 1) * (x2sum - x2) - da.square(xsum - xarr)
    vary = (n - 1) * (y2sum - y2) - da.square(ysum - yarr)
    denominator = da.sqrt(varx * vary)

    return da.map_blocks(itruediv, numerator, denominator, dtype=numerator.dtype)


def correlation_nxn(
    data: da.Array, columns: Optional[Sequence[str]] = None
) -> Tuple[da.Array, da.Array, Dict[CorrelationMethod, da.Array]]:
    """
    Calculation of a n x n correlation matrix for n columns
    """
    _, ncols = data.shape
    cordx, cordy = da.meshgrid(range(ncols), range(ncols))
    cordx, cordy = cordy.ravel(), cordx.ravel()

    corrs = {
        CorrelationMethod.Pearson: pearson_nxn(data).ravel(),
        CorrelationMethod.Spearman: spearman_nxn(data).ravel(),
        CorrelationMethod.KendallTau: kendall_tau_nxn(data).ravel(),
    }

    if columns is not None:
        # The number of columns usually is not too large
        cordx = da.from_array(columns[cordx.compute()], chunks=1)
        cordy = da.from_array(columns[cordy.compute()], chunks=1)

    return cordx, cordy, corrs


def pearson_nxn(data: da.Array) -> da.Array:
    """
    Pearson correlation calculation of a n x n correlation matrix for n columns
    """
    _, ncols = data.shape

    corrmat = np.zeros(shape=(ncols, ncols))
    corr_list = []
    for i in range(ncols):
        for j in range(i + 1, ncols):
            mask = ~(da.isnan(data[:, i]) | da.isnan(data[:, j]))
            tmp = dask.delayed(lambda a, b: np.corrcoef(a, b)[0, 1])(
                data[:, i][mask], data[:, j][mask]
            )
            corr_list.append(tmp)
    corr_comp = dask.compute(*corr_list)  # TODO avoid explicitly compute
    idx = 0
    for i in range(ncols):  # TODO: Optimize by using numpy api
        for j in range(i + 1, ncols):
            corrmat[i][j] = corr_comp[idx]
            idx = idx + 1

    corrmat2 = corrmat + corrmat.T
    np.fill_diagonal(corrmat2, 1)
    corrmat = da.from_array(corrmat2)
    return corrmat


def spearman_nxn(data: da.Array) -> da.Array:
    """
    Spearman correlation calculation of a n x n correlation matrix for n columns
    """
    _, ncols = data.shape
    data = data.compute()  # TODO: How to compute rank distributedly?

    ranks = np.empty_like(data)
    for j in range(ncols):
        ranks[:, j] = pd.Series(data[:, j]).rank()
    ranks = da.from_array(ranks)
    corrmat = pearson_nxn(ranks)
    return corrmat


def kendall_tau_nxn(data: da.Array) -> da.Array:
    """
    Kendal Tau correlation calculation of a n x n correlation matrix for n columns
    """
    _, ncols = data.shape

    corrmat = np.zeros(shape=(ncols, ncols))
    corr_list = []
    for i in range(ncols):
        for j in range(i + 1, ncols):
            mask = ~(da.isnan(data[:, i]) | da.isnan(data[:, j]))
            tmp = dask.delayed(lambda a, b: kendalltau(a, b).correlation)(
                data[:, i][mask], data[:, j][mask]
            )
            corr_list.append(tmp)
    corr_comp = dask.compute(*corr_list)  # TODO avoid explicitly compute
    idx = 0
    for i in range(ncols):  # TODO: Optimize by using numpy api
        for j in range(i + 1, ncols):
            corrmat[i][j] = corr_comp[idx]
            idx = idx + 1

    corrmat2 = corrmat + corrmat.T
    np.fill_diagonal(corrmat2, 1)
    corrmat = da.from_array(corrmat2)

    return corrmat


def pearson_1xn(
    x: da.Array,
    data: da.Array,
    value_range: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    x : da.Array
    data : da.Array
    value_range : Optional[Tuple[float, float]] = None
    k : Optional[int] = None
    """
    _, ncols = data.shape

    corrs = []
    for j in range(ncols):
        mask = ~(da.isnan(x) | da.isnan(data[:, j]))
        _, (corr, _) = da.corrcoef(np.array(x)[mask], np.array(data[:, j])[mask])
        corrs.append(corr)

    (corrs,) = da.compute(corrs)
    corrs = np.asarray(corrs)

    return corr_filter(corrs, value_range, k)


def spearman_1xn(
    x: da.Array,
    data: da.Array,
    value_range: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    x : da.Array
    data : da.Array
    value_range : Optional[Tuple[float, float]] = None
    k : Optional[int] = None
    """

    _, ncols = data.shape
    data = data.compute()  # TODO: How to compute rank distributedly?

    ranks = np.empty_like(data)
    for j in range(ncols):
        ranks[:, j] = pd.Series(data[:, j]).rank()
    ranks = da.from_array(ranks)
    xrank = pd.Series(x.compute()).rank()
    xrank = da.from_array(xrank)

    return pearson_1xn(xrank, ranks, value_range, k)


def kendall_tau_1xn(
    x: da.Array,
    data: da.Array,
    value_range: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    x : da.Array
    data : da.Array
    value_range : Optional[Tuple[float, float]] = None
    k : Optional[int] = None
    """

    _, ncols = data.shape

    corrs = []
    for j in range(ncols):
        mask = ~(da.isnan(x) | da.isnan(data[:, j]))
        corr = dask.delayed(lambda a, b: kendalltau(a, b)[0])(
            np.array(x)[mask], np.array(data[:, j])[mask]
        )
        corrs.append(corr)

    (corrs,) = da.compute(corrs)
    corrs = np.asarray(corrs)
    return corr_filter(corrs, value_range, k)


def corr_filter(
    corrs: np.ndarray,
    value_range: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter correlation values by k and value_range
    """

    assert (value_range is None) or (
        k is None
    ), "value_range and k cannot be present in both"

    if k is not None:
        sorted_idx = np.argsort(corrs)
        sorted_corrs = corrs[sorted_idx]
        # pylint: disable=invalid-unary-operand-type
        return (sorted_idx[-k:], corrs[sorted_idx[-k:]])
        # pylint: enable=invalid-unary-operand-type

    sorted_idx = np.argsort(corrs)
    sorted_corrs = corrs[sorted_idx]

    if value_range is not None:
        start, end = value_range
        istart = np.searchsorted(sorted_corrs, start)
        iend = np.searchsorted(sorted_corrs, end, side="right")
        return sorted_idx[istart:iend], sorted_corrs[istart:iend]
    return sorted_idx, sorted_corrs
