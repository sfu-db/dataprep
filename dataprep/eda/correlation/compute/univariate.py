"""This module implements the intermediates computation
for plot_correlation(df) function."""

import sys
from typing import Optional, Tuple

import dask
import dask.array as da
import numpy as np
import pandas as pd

from ...data_array import DataArray
from ...intermediate import Intermediate
from .common import CorrelationMethod, kendalltau, nanrankdata, corrcoef


def _calc_univariate(
    df: DataArray,
    x: Optional[str] = None,
    *,
    value_range: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
) -> Intermediate:
    """
    Parameters
    ----------
    df
        The dataframe for which plots are calculated.
    x
        A valid column name of the dataframe.
    value_range
        If the correlation value is out of the range, don't show it.
    k
        Choose top-k element
    """

    if len(df.columns) == 0:
        return Intermediate(visual_type=None)

    if x not in df.columns:
        raise ValueError(f"{x} not in numerical column names")

    df.compute()
    columns = df.columns[df.columns != x]
    xarr = df.values[:, df.columns == x]
    data = df.values[:, df.columns != x]

    funcs = [_pearson_1xn, _spearman_1xn, _kendall_tau_1xn]

    dfs = {}
    (computed,) = dask.compute(
        {meth: func(xarr, data) for meth, func in zip(CorrelationMethod, funcs)}
    )

    for meth, corrs in computed.items():
        indices, corrs = _corr_filter(corrs, value_range, k)
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


def _pearson_1xn(x: da.Array, data: da.Array) -> da.Array:
    _, ncols = data.shape

    fused = da.concatenate([data, x], axis=1)
    mask = ~da.isnan(fused)

    corrs = []
    for j in range(ncols):
        xy = fused[:, [-1, j]]
        mask_ = mask[:, -1] & mask[:, j]
        xy = xy[mask_]
        corr = da.from_delayed(corrcoef(xy), dtype=np.float, shape=())
        # not usable because xy has unknown rows due to the null filter
        # _, (corr, _) = da.corrcoef(xy, rowvar=False)
        corrs.append(corr)

    return da.stack(corrs)


def _spearman_1xn(x: da.Array, data: da.Array) -> da.Array:
    xrank = da.from_delayed(nanrankdata(x), dtype=np.float, shape=x.shape)
    ranks = da.from_delayed(nanrankdata(data), dtype=np.float, shape=data.shape)

    return _pearson_1xn(xrank, ranks)


def _kendall_tau_1xn(x: da.Array, data: da.Array) -> da.Array:
    _, ncols = data.shape

    datamask = da.isnan(data)
    xmask = da.isnan(x)[:, 0]

    corrs = []
    for j in range(ncols):
        y = data[:, [j]]

        mask = ~(xmask | datamask[:, j])
        corr = da.from_delayed(kendalltau(x[mask], y[mask]), dtype=np.float, shape=())
        corrs.append(corr)

    return da.stack(corrs)


def _corr_filter(
    corrs: np.ndarray,
    value_range: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter correlation values by k and value_range
    """

    if (value_range is not None) and (k is not None):
        raise ValueError("value_range and k cannot be present in both")

    if k is not None:
        sorted_idx = np.argsort(corrs)
        sorted_corrs = corrs[sorted_idx]
        # pylint: disable=invalid-unary-operand-type
        return (sorted_idx[-k:], corrs[sorted_idx[-k:]])
        # pylint: enable=invalid-unary-operand-type

    sorted_idx = np.argsort(corrs)
    sorted_idx = np.roll(sorted_idx, np.count_nonzero(np.isnan(corrs)))
    sorted_corrs = corrs[sorted_idx]

    if value_range is not None:
        start, end = value_range
        istart = np.searchsorted(sorted_corrs, start)
        iend = np.searchsorted(sorted_corrs, end, side="right")
        return sorted_idx[istart:iend], sorted_corrs[istart:iend]
    return sorted_idx, sorted_corrs
