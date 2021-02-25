"""This module implements the intermediates computation
for plot_correlation(df) function."""

import sys
from typing import Optional, Tuple

import dask
import dask.array as da
import numpy as np
import pandas as pd

from ...data_array import DataArray, DataFrame
from ...intermediate import Intermediate
from .common import CorrelationMethod, kendalltau, nanrankdata, corrcoef


def _calc_cat_col(
    cat_df: DataArray,
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

    if len(cat_df.columns) == 0:
        return Intermediate(visual_type=None)

    cat_df = cat_df
    cat_df.compute()
    columns = cat_df.columns[cat_df.columns != x]
    xarr = cat_df.values[:, cat_df.num_columns == x]
    data = cat_df.values[:, cat_df.columns != x]

    cmp_dict = {}
    cmp_dict[CorrelationMethod.CramerV] = _cramerv_1xn(xarr, data)

    (computed,) = dask.compute(cmp_dict)

    dfs = {}
    for meth, corrs in computed.items():
        indices, corrs = _corr_filter(corrs, value_range, k)
        if len(indices) == 0:
            print(
                f"Correlation for {meth.name} is empty, try to broaden the value_range.",
                file=sys.stderr,
            )
        cat_df = pd.DataFrame(
            {"x": np.full(len(indices), x), "y": columns[indices], "correlation": corrs,}
        )
        dfs[meth.name] = cat_df

    return Intermediate(data=dfs, visual_type="correlation_single_heatmaps")



def _calc_num_col(
    num_df: DataArray,
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

    if len(num_df.columns) == 0:
        return Intermediate(visual_type=None)

    num_df.compute()
    num_columns = num_df.columns[num_df.columns != x]
    num_df_xarr = num_df.values[:, num_df.num_columns == x]
    num_df_data = num_df.values[:, num_df.columns != x]

    cmp_dict = {}
    cmp_dict[CorrelationMethod.Pearson] = _pearson_1xn(xarr, data)
    cmp_dict[CorrelationMethod.Spearman] = _spearman_1xn(xarr, data)
    cmp_dict[CorrelationMethod.KendallTau] = _kendall_tau_1xn(xarr, data)

    (computed,) = dask.compute(cmp_dict)

    dfs = {}
    for meth, corrs in computed.items():
        indices, corrs = _corr_filter(corrs, value_range, k)
        if len(indices) == 0:
            print(
                f"Correlation for {meth.name} is empty, try to broaden the value_range.",
                file=sys.stderr,
            )
        num_df = pd.DataFrame(
            {"x": np.full(len(indices), x), "y": num_columns[indices], "correlation": corrs,}
        )
        dfs[meth.name] = num_df

    return Intermediate(data=dfs, visual_type="correlation_single_heatmaps")



def _calc_univariate(
    df: DataFrame,
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

    if x not in df.columns:
        raise ValueError(f"{x} not in column names")

    num_df = DataArray(df).select_num_columns()
    cat_df = DataArray(df).select_cat_columns()

    if x in num_df.columns:
        return _calc_num_col(num_df, x, *, value_range, k)
    elif x in cat_df.columns:
        return _calc_cat_col(cat_df, x, *, value_range, k)

    


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


def _cramerv_1xn(x: da.Array, data: da.Array) -> da.Array:
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
    corrs: np.ndarray, value_range: Optional[Tuple[float, float]] = None, k: Optional[int] = None,
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
