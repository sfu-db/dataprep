"""
    This module implements the intermediates computation
    for plot_correlation(df) function.
"""
import sys
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from bokeh.io import show
from bokeh.models.widgets import Tabs
from bokeh.plotting import Figure
from scipy.stats import kendalltau

from ...errors import UnreachableError
from ...utils import DataType, get_type, to_dask
from ..common import Intermediate
from ..dtypes import CATEGORICAL_DTYPES, NUMERICAL_DTYPES, is_categorical, is_numerical

__all__ = ["compute_correlation"]


class CorrelationMethod(Enum):
    Pearson = auto()
    Spearman = auto()
    KendallTau = auto()


def compute_correlation(  # pylint: disable=too-many-arguments
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
    df : pd.DataFrame
        The pandas dataframe for which plots are calculated for each column.
    x : Optional[str]
        A valid column name of the dataframe
    y : Optional[str]
        A valid column name of the dataframe
    value_range : Optional[Tuple[float, float]] = None
        If the correlation value is out of the range, don't show it.
    k : Optional[int]
        Choose top-k element

    Returns
    -------
    Intermediate

    Note
    ----
    match (x, y, k)
        case (None, None, _) => heatmap
        case (Some, None, _) => Top K columns for (pearson, spearman, kendall)
        case (Some, Some, _) => Scatter with regression line with/without top k outliers
        otherwise => error
    """

    df = to_dask(df)
    df.columns = [str(e) for e in df.columns] # convert column names to string

    if x is None and y is None:
        assert value_range is None
        df = df.select_dtypes(NUMERICAL_DTYPES)
        assert len(df.columns) != 0, f"No numerical columns found"

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
            if k is not None:
                df = df[df["y"] < df["x"]]  # Retain only upper triangle (w/o diag)
                thresh = df["correlation"].abs().nlargest(k).compute().iloc[-1]
                df = df[(df["correlation"] >= thresh) | (df["correlation"] <= -thresh)]

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
        assert len(df.columns) != 0, f"No numerical columns found"
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

        xdtype = df[x].dtype
        ydtype = df[y].dtype
        if is_categorical(xdtype) and is_categorical(ydtype):
            raise NotImplementedError

            # intermediate = _cross_table(df=df, x=x, y=y)
            # return intermediate
        elif is_numerical(xdtype) and is_numerical(ydtype):
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

            if influences is not None:
                infidx = np.argsort(influences)
                labels = np.full(len(influences), "=")
                labels[infidx[-k:]] = "-"
                labels[infidx[:k]] = "+"
                result["data"]["influence"] = labels
            return Intermediate(**result, visual_type="correlation_scatter")
        else:
            raise ValueError(
                "Cannot calculate the correlation between two different dtype column"
            )
    else:
        raise UnreachableError


def scatter_with_regression(
    xarr: da.Array, yarr: da.Array, sample_size: int, k: Optional[int] = None,
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

    _, (corr, _) = da.corrcoef(xarr, yarr)

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

    influences = np.zeros(len(xarr))
    mask = np.ones(len(xarr), dtype=bool)

    # TODO: Optimize, since some part of the coeffs can be reused.
    for i in range(len(xarr)):
        mask[i] = False
        _, (corrlo1, _) = np.corrcoef(xarr[mask], yarr[mask])
        influences[i] = corr - corrlo1
        mask[i] = True

    return (coeffa, coeffb), df, influences


def correlation_nxn(
    data: da.Array, columns: Optional[Sequence[str]] = None
) -> Tuple[da.Array, da.Array, Dict[CorrelationMethod, da.Array]]:
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
    cov = da.cov(data.T)
    stderr = da.sqrt(da.diag(cov))
    corrmat = cov / stderr[:, None] / stderr[None, :]
    return corrmat


def spearman_nxn(data: da.Array) -> da.Array:

    _, ncols = data.shape
    data = data.compute()  # TODO: How to compute rank distributedly?

    ranks = np.empty_like(data)
    for j in range(ncols):
        ranks[:, j] = pd.Series(data[:, j]).rank()
    ranks = da.from_array(ranks)
    corrmat = pearson_nxn(ranks)
    return corrmat


def kendall_tau_nxn(data: da.Array) -> da.Array:

    _, ncols = data.shape

    corrmat = np.zeros(shape=(ncols, ncols))
    corr_list = []
    for i in range(ncols):
        for j in range(i + 1, ncols):
            tmp = dask.delayed(lambda a, b: kendalltau(a, b).correlation)(data[:, i], data[:, j])
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
        _, (corr, _) = da.corrcoef(x, data[:, j])
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
        corr = dask.delayed(lambda a, b: kendalltau(a, b)[0])(x, data[:, j])
        corrs.append(corr)

    (corrs,) = da.compute(corrs)
    corrs = np.asarray(corrs)
    return corr_filter(corrs, value_range, k)


def corr_filter(
    corrs: np.ndarray,
    value_range: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    assert (value_range is None) or (
        k is None
    ), "value_range and k cannot be present in both"

    if k is not None:
        sorted_idx = np.argsort(corrs)
        sorted_corrs = corrs[sorted_idx]
        return sorted_idx[-k:], corrs[sorted_idx[-k:]]
    else:
        sorted_idx = np.argsort(corrs)
        sorted_corrs = corrs[sorted_idx]

        if value_range is not None:
            start, end = value_range
            istart = np.searchsorted(sorted_corrs, start)
            iend = np.searchsorted(sorted_corrs, end, side="right")
            return sorted_idx[istart:iend], sorted_corrs[istart:iend]
        return sorted_idx, sorted_corrs
