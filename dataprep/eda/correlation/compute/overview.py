"""Implementations of correlations.

Currently this boils down to pandas' implementation."""

from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import dask
import dask.array as da
import numpy as np
import pandas as pd

from ...configs import Config
from ...data_array import DataArray, DataFrame
from ...intermediate import Intermediate
from ...utils import cut_long_name
from .common import CorrelationMethod


def _calc_overview(
    df: DataFrame,
    cfg: Config,
    *,
    value_range: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
) -> Intermediate:
    # pylint: disable=too-many-statements,too-many-locals,too-many-branches

    most_show = 6  # the most number of column/row to show in "insight"

    if value_range is not None and k is not None:
        raise ValueError("value_range and k cannot be present in both")
    dadf = DataArray(df)

    # num_df: df of num columns. Used for numerical correlation such as pearson.
    num_df = dadf.select_num_columns()

    # cordx, cordy are used to locate a cell in correlation matrix.
    num_cordx, num_cordy = _get_cord(len(num_df.columns))

    # The below variables are dict since some methods are applied to numerical columns
    # and some methods are applied to categorical columns.
    # columns: used column names
    # cordx, cordy: used to locate a cell in correlation matrix.
    # corrs: correlation matrix.
    method2columns: Dict[CorrelationMethod, str] = {}
    method2cordx: Dict[CorrelationMethod, np.ndarray] = {}
    method2cordy: Dict[CorrelationMethod, np.ndarray] = {}
    method2corrs: Dict[CorrelationMethod, da.Array] = {}

    if cfg.pearson.enable or cfg.stats.enable:
        method2columns[CorrelationMethod.Pearson] = num_df.columns
        method2cordx[CorrelationMethod.Pearson] = num_cordx
        method2cordy[CorrelationMethod.Pearson] = num_cordy
        method2corrs[CorrelationMethod.Pearson] = _pearson_nxn(num_df)
    if cfg.spearman.enable or cfg.stats.enable:
        method2columns[CorrelationMethod.Spearman] = num_df.columns
        method2cordx[CorrelationMethod.Spearman] = num_cordx
        method2cordy[CorrelationMethod.Spearman] = num_cordy
        method2corrs[CorrelationMethod.Spearman] = _spearman_nxn(num_df)
    if cfg.kendall.enable or cfg.stats.enable:
        method2columns[CorrelationMethod.KendallTau] = num_df.columns
        method2cordx[CorrelationMethod.KendallTau] = num_cordx
        method2cordy[CorrelationMethod.KendallTau] = num_cordy
        method2corrs[CorrelationMethod.KendallTau] = _kendall_tau_nxn(num_df)

    # The computations below is not expensive (scales with # of columns)
    # So we do them in pandas

    (method2corrs,) = dask.compute(method2corrs)

    # compute stat information such as hightest correlation value and columns.
    if cfg.stats.enable or cfg.insight.enable:
        positive_max_corr_value = {}
        negative_max_corr_value = {}
        mean_corr_value = {}
        positive_max_corr_cols = {}
        negative_max_corr_cols = {}
        min_corr_value = {}
        min_corr_cols = {}
        for method in method2corrs.keys():
            (
                positive_max_corr_value[method.value],
                negative_max_corr_value[method.value],
                mean_corr_value[method.value],
                positive_max_corr_cols[method.value],
                negative_max_corr_cols[method.value],
            ) = most_corr(method2corrs[method])
            min_corr_value[method.value], min_corr_cols[method.value] = least_corr(
                method2corrs[method]
            )

    # create stat table
    if cfg.stats.enable:
        tabledata = {
            "Highest Positive Correlation": positive_max_corr_value,
            "Highest Negative Correlation": negative_max_corr_value,
            "Lowest Correlation": min_corr_value,
            "Mean Correlation": mean_corr_value,
        }

    # create insight. E.g., most correlated columns.
    if cfg.insight.enable:
        insights: Dict[str, List[Any]] = {}
        for method in method2corrs.keys():
            pos_str = create_string(
                "positive", positive_max_corr_cols[method.value], most_show, num_df
            )
            neg_str = create_string(
                "negative", negative_max_corr_cols[method.value], most_show, num_df
            )
            least_str = create_string("least", min_corr_cols[method.value], most_show, num_df)
            insights[method.value] = [pos_str, neg_str, least_str]

    dfs = {}
    for method, corr in method2corrs.items():
        if (  # pylint: disable=too-many-boolean-expressions
            method == CorrelationMethod.Pearson
            and not cfg.pearson.enable
            or method == CorrelationMethod.Spearman
            and not cfg.spearman.enable
            or method == CorrelationMethod.KendallTau
            and not cfg.kendall.enable
        ):
            continue
        cordx = method2cordx[method]
        cordy = method2cordy[method]
        columns = method2columns[method]

        # create correlation df from correlation matrix.
        ndf = pd.DataFrame(
            {
                "x": columns[cordx],
                "y": columns[cordy],
                "correlation": corr.ravel(),
            }
        )
        ndf = ndf[cordy > cordx]  # Retain only lower triangle (w/o diag)

        # filter correlation df by top-k or value_range.
        if k is not None:
            thresh = ndf["correlation"].abs().nlargest(k).iloc[-1]
            ndf = ndf[(ndf["correlation"] >= thresh) | (ndf["correlation"] <= -thresh)]
        elif value_range is not None:
            mask = (value_range[0] <= ndf["correlation"]) & (ndf["correlation"] <= value_range[1])
            ndf = ndf[mask]

        dfs[method.name] = ndf

    return Intermediate(
        data=dfs,
        axis_range=list(num_df.columns.unique()),
        visual_type="correlation_impact",
        tabledata=tabledata if cfg.stats.enable else {},
        insights=insights if cfg.insight.enable else {},
    )


def _get_cord(ncols: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the coordinate of the correlation matrix. (cordx[i], cordy[i]) represents
    a cell in the correlation matrix.

    Returns
    -------
        cordx: the x axis, e.g., [0 0 0 1 1 1 2 2 2]
        cordy: the y axis, e.g., [0 1 2 0 1 2 0 1 2]
    """
    cordx, cordy = np.meshgrid(range(ncols), range(ncols))
    cordx, cordy = cordy.ravel(), cordx.ravel()
    return cordx, cordy


def correlation_nxn(
    df: DataArray, cfg: Config
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

    corrs: Dict[CorrelationMethod, da.Array] = {}

    if cfg.pearson.enable or cfg.stats.enable:
        corrs[CorrelationMethod.Pearson] = _pearson_nxn(df)
    if cfg.spearman.enable or cfg.stats.enable:
        corrs[CorrelationMethod.Spearman] = _spearman_nxn(df)
    if cfg.kendall.enable or cfg.stats.enable:
        corrs[CorrelationMethod.KendallTau] = _kendall_tau_nxn(df)

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
