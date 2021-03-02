"""Implementations of correlations.

Currently this boils down to pandas' implementation."""

from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import dask
import dask.array as da
import numpy as np
import pandas as pd
from scipy import stats


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
    # pylint: disable=too-many-statements,too-many-locals,too-many-branches,too-many-boolean-expressions

    most_show = 6  # the most number of column/row to show in "insight"

    if value_range is not None and k is not None:
        raise ValueError("value_range and k cannot be present in both")
    dadf = DataArray(df)

    # num_df: df of num columns. Used for numerical correlation such as pearson.
    num_df = dadf.select_num_columns()
    # cat_df: df of cat columns. Used for categorical correlation such as Cramer's V.
    cat_df = dadf.select_cat_columns()

    # whether data contains numerical column and categorical column.
    contains_num = len(num_df.columns) > 0
    contains_cat = len(cat_df.columns) > 0

    # cordx, cordy are used to locate a cell in correlation matrix.
    if contains_num:
        num_cordx, num_cordy = _get_cord(len(num_df.columns))
    if contains_cat:
        cat_cordx, cat_cordy = _get_cord(len(cat_df.columns))

    # The below variables are dict since some methods are applied to numerical columns
    # and some methods are applied to categorical columns.
    # columns: used column names
    # cordx, cordy: used to locate a cell in correlation matrix.
    # corrs: correlation matrix.
    method2columns: Dict[CorrelationMethod, pd.Index] = {}
    method2cordx: Dict[CorrelationMethod, np.ndarray] = {}
    method2cordy: Dict[CorrelationMethod, np.ndarray] = {}
    method2corrs: Dict[CorrelationMethod, da.Array] = {}

    if contains_num and (cfg.pearson.enable or cfg.stats.enable):
        method2columns[CorrelationMethod.Pearson] = num_df.columns
        method2cordx[CorrelationMethod.Pearson] = num_cordx
        method2cordy[CorrelationMethod.Pearson] = num_cordy
        method2corrs[CorrelationMethod.Pearson] = _pearson_nxn(num_df)
    if contains_num and (cfg.spearman.enable or cfg.stats.enable):
        method2columns[CorrelationMethod.Spearman] = num_df.columns
        method2cordx[CorrelationMethod.Spearman] = num_cordx
        method2cordy[CorrelationMethod.Spearman] = num_cordy
        method2corrs[CorrelationMethod.Spearman] = _spearman_nxn(num_df)
    if contains_num and (cfg.kendall.enable or cfg.stats.enable):
        method2columns[CorrelationMethod.KendallTau] = num_df.columns
        method2cordx[CorrelationMethod.KendallTau] = num_cordx
        method2cordy[CorrelationMethod.KendallTau] = num_cordy
        method2corrs[CorrelationMethod.KendallTau] = _kendall_tau_nxn(num_df)
    if contains_cat and (cfg.cramerv.enable or cfg.stats.enable):
        method2columns[CorrelationMethod.CramerV] = cat_df.columns
        method2cordx[CorrelationMethod.CramerV] = cat_cordx
        method2cordy[CorrelationMethod.CramerV] = cat_cordy
        method2corrs[CorrelationMethod.CramerV] = _cramerv_nxn(cat_df)

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
        for method in method2corrs:
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
        for method in method2corrs:
            pos_str = create_string(
                "positive", positive_max_corr_cols[method.value], most_show, method2columns[method]
            )
            neg_str = create_string(
                "negative", negative_max_corr_cols[method.value], most_show, method2columns[method]
            )
            least_str = create_string(
                "least", min_corr_cols[method.value], most_show, method2columns[method]
            )
            insights[method.value] = [pos_str, neg_str, least_str]

    dfs = {}
    axis_ranges = {}
    for method, corr in method2corrs.items():
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
        if len(ndf) > 0 and k is not None:
            thresh = ndf["correlation"].abs().nlargest(k).iloc[-1]
            ndf = ndf[(ndf["correlation"] >= thresh) | (ndf["correlation"] <= -thresh)]
        elif value_range is not None:
            mask = (value_range[0] <= ndf["correlation"]) & (ndf["correlation"] <= value_range[1])
            ndf = ndf[mask]

        dfs[method.value] = ndf
        axis_ranges[method.value] = list(columns.unique())

    return Intermediate(
        data=dfs,
        axis_range=axis_ranges,
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
    """
    Calculate column-wise pearson correlation.
    """
    return (
        df.frame.repartition(npartitions=1)
        .map_partitions(partial(pd.DataFrame.corr, method="pearson"))
        .to_dask_array()
    )


def _spearman_nxn(df: DataArray) -> da.Array:
    """
    Calculate column-wise spearman correlation.
    """
    return (
        df.frame.repartition(npartitions=1)
        .map_partitions(partial(pd.DataFrame.corr, method="spearman"))
        .to_dask_array()
    )


def _kendall_tau_nxn(df: DataArray) -> da.Array:
    """
    Calculate column-wise kendalltau correlation.
    """
    return (
        df.frame.repartition(npartitions=1)
        .map_partitions(partial(pd.DataFrame.corr, method="kendall"))
        .to_dask_array()
    )


def _cramerv_nxn(df: DataArray) -> da.Array:
    """
    Calculate column-wise Cramer'V correlation for categorical column.
    Input df should only contain categorical column.
    """
    return df.frame.repartition(npartitions=1).map_partitions(_calc_cramerv).to_dask_array()


def _calc_cramerv_pair(col1: pd.Series, col2: pd.Series) -> float:
    """
    Calculate the Cramer'V correlation for a pair of columns.
    Input columns should be categorical columns.
    """
    confusion_matrix = pd.crosstab(col1, col2)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    shape0 = confusion_matrix.shape[0]
    shape1 = confusion_matrix.shape[1] if len(confusion_matrix.shape) > 1 else 1

    with np.errstate(divide="ignore", invalid="ignore"):
        phi2corr = max(0.0, phi2 - ((shape1 - 1.0) * (shape0 - 1.0)) / (n - 1.0))
        rcorr = shape0 - ((shape0 - 1.0) ** 2.0) / (n - 1.0)
        kcorr = shape1 - ((shape1 - 1.0) ** 2.0) / (n - 1.0)
        rkcorr = min((kcorr - 1.0), (rcorr - 1.0))
        if rkcorr == 0.0:
            corr = 1.0
        else:
            corr = np.sqrt(phi2corr / rkcorr)
    return corr


def _calc_cramerv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Cramer'V correlation matrix.
    Input df should contain only categorical column.
    """
    cols = df.columns
    idx = cols.copy()
    ncols = len(cols)
    correl = np.empty((ncols, ncols), dtype=float)
    for i in range(ncols):
        for j in range(ncols):
            if i > j:
                continue
            value = _calc_cramerv_pair(df[cols[i]], df[cols[j]])
            correl[i, j] = value
            correl[j, i] = value
    return pd.DataFrame(correl, index=idx, columns=cols)


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


def create_string(flag: str, source: List[Any], most_show: int, columns: pd.Index) -> str:
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
                "(" + cut_long_name(columns[e[0]]) + ", " + cut_long_name(columns[e[1]]) + ")"
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
