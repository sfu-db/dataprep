"""Implementations of correlations.

Currently this boils down to pandas' implementation."""

from functools import partial
from typing import Dict, Optional, Tuple, List, Any

import dask
import dask.array as da
import numpy as np
import pandas as pd

from ...data_array import DataArray
from ...intermediate import Intermediate
from .common import CorrelationMethod
from ...utils import cut_long_name


def _calc_nullivariate(
    df: DataArray,
    *,
    value_range: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
) -> Intermediate:
    # pylint: disable=too-many-statements,too-many-locals,too-many-branches

    most_show = 6  # the most number of column/row to show in "insight"
    # longest = 5  # the longest length of word to show in "insight"

    if value_range is not None and k is not None:
        raise ValueError("value_range and k cannot be present in both")

    cordx, cordy, corrs = correlation_nxn(df)

    # The computations below is not expensive (scales with # of columns)
    # So we do them in pandas

    (corrs,) = dask.compute(corrs)
    pearson_corr, spearman_corr, kendalltau_corr = corrs.values()

    pearson_pos_max, pearson_neg_max, pearson_mean, pearson_pos_cols, pearson_neg_cols = most_corr(
        pearson_corr
    )
    (
        spearman_pos_max,
        spearman_neg_max,
        spearman_mean,
        spearman_pos_cols,
        spearman_neg_cols,
    ) = most_corr(spearman_corr)
    (
        kendalltau_pos_max,
        kendalltau_neg_max,
        kendalltau_mean,
        kendalltau_pos_cols,
        kendalltau_neg_cols,
    ) = most_corr(kendalltau_corr)
    pearson_min, pearson_cols = least_corr(pearson_corr)
    spearman_min, spearman_cols = least_corr(spearman_corr)
    kendalltau_min, kendalltau_cols = least_corr(kendalltau_corr)

    p_p_corr = create_string("positive", pearson_pos_cols, most_show, df)
    s_p_corr = create_string("positive", spearman_pos_cols, most_show, df)
    k_p_corr = create_string("positive", kendalltau_pos_cols, most_show, df)
    p_n_corr = create_string("negative", pearson_neg_cols, most_show, df)
    s_n_corr = create_string("negative", spearman_neg_cols, most_show, df)
    k_n_corr = create_string("negative", kendalltau_neg_cols, most_show, df)
    p_corr = create_string("least", pearson_cols, most_show, df)
    s_corr = create_string("least", spearman_cols, most_show, df)
    k_corr = create_string("least", kendalltau_cols, most_show, df)

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
            mask = (value_range[0] <= ndf["correlation"]) & (ndf["correlation"] <= value_range[1])
            ndf = ndf[mask]

        dfs[method.name] = ndf

    return Intermediate(
        data=dfs,
        axis_range=list(df.columns.unique()),
        visual_type="correlation_impact",
        tabledata={
            "Highest Positive Correlation": {
                "Pearson": pearson_pos_max,
                "Spearman": spearman_pos_max,
                "KendallTau": kendalltau_pos_max,
            },
            "Highest Negative Correlation": {
                "Pearson": pearson_neg_max,
                "Spearman": spearman_neg_max,
                "KendallTau": kendalltau_neg_max,
            },
            "Lowest Correlation": {
                "Pearson": pearson_min,
                "Spearman": spearman_min,
                "KendallTau": kendalltau_min,
            },
            "Mean Correlation": {
                "Pearson": pearson_mean,
                "Spearman": spearman_mean,
                "KendallTau": kendalltau_mean,
            },
        },
        insights={
            "Pearson": [p_p_corr, p_n_corr, p_corr],
            "Spearman": [s_p_corr, s_n_corr, s_corr],
            "KendallTau": [k_p_corr, k_n_corr, k_corr],
        },
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


def most_corr(corrs: np.ndarray) -> Tuple[float, float, float, List[Any], List[Any]]:
    """Find the most correlated columns."""
    positive_col_set = set()
    negative_col_set = set()
    corrs_copy = corrs
    for i in range(corrs_copy.shape[0]):
        corrs_copy[i, i] = 0
    mean = corrs_copy.mean()
    p_maximum = corrs_copy.max()
    n_maximum = (-corrs_copy).max()

    if p_maximum != 0:
        p_col1, p_col2 = np.where(corrs_copy == p_maximum)
    else:
        p_col1, p_col2 = [], []
    if n_maximum != 0:
        n_col1, n_col2 = np.where(corrs_copy == -n_maximum)
    else:
        n_col1, n_col2 = [], []

    for i, _ in enumerate(p_col1):
        if p_col1[i] < p_col2[i]:
            positive_col_set.add((p_col1[i], p_col2[i]))
        elif p_col1[i] > p_col2[i]:
            positive_col_set.add((p_col2[i], p_col1[i]))
    for i, _ in enumerate(n_col1):
        if n_col1[i] < n_col2[i]:
            negative_col_set.add((n_col1[i], n_col2[i]))
        elif n_col1[i] > n_col2[i]:
            negative_col_set.add((n_col2[i], n_col1[i]))

    return (
        round(p_maximum, 3),
        round(-n_maximum, 3),
        round(mean, 3),
        list(positive_col_set),
        list(negative_col_set),
    )


def least_corr(corrs: np.ndarray) -> Tuple[float, List[Any]]:
    """Find the least correlated columns."""
    col_set = set()
    corrs_copy = corrs
    for i in range(corrs_copy.shape[0]):
        corrs_copy[i, i] = 2
    minimum = abs(corrs_copy).min()
    col1, col2 = np.where(corrs_copy == minimum)

    for i, _ in enumerate(col1):
        if col1[i] < col2[i]:
            col_set.add((col1[i], col2[i]))
        elif col1[i] > col2[i]:
            col_set.add((col2[i], col1[i]))

    return round(minimum, 3), list(col_set)


def create_string(flag: str, source: List[Any], most_show: int, df: DataArray) -> str:
    """Create the output string"""
    suffix = "" if len(source) <= most_show else ", ..."
    if flag == "positive":
        prefix = "Most positive correlated: "
        temp = "Most positive correlated: None"
    elif flag == "negative":
        prefix = "Most negative correlated: "
        temp = "Most negative correlated: None"
    elif flag == "least":
        prefix = "Least correlated: "
        temp = "Least correlated: None"

    if source != []:
        out = (
            prefix
            + ", ".join(
                "(" + cut_long_name(df.columns[e[0]]) + ", " + cut_long_name(df.columns[e[1]]) + ")"
                for e in source[:most_show]
            )
            + suffix
        )
    else:
        out = temp

    return out


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
