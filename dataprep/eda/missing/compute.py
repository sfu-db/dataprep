"""
    This module implements the plot_missing(df) function's
    calculating intermediate part
"""
from typing import Optional, Tuple, Union, List

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.stats import rv_histogram

from ...errors import UnreachableError
from ..utils import to_dask
from ..dtypes import (
    is_dtype,
    detect_dtype,
    is_pandas_categorical,
    Continuous,
    Nominal,
    DTypeDef,
)
from ..intermediate import Intermediate, ColumnsMetadata

__all__ = ["compute_missing"]

LABELS = ["Origin", "DropMissing"]


def histogram(
    srs: dd.Series,
    bins: Optional[int] = None,
    return_edges: bool = True,
    range: Optional[Tuple[int, int]] = None,  # pylint: disable=redefined-builtin
    dtype: Optional[DTypeDef] = None,
) -> Union[Tuple[da.Array, da.Array], Tuple[da.Array, da.Array, da.Array]]:
    """
    Calculate histogram for both numerical and categorical
    """

    if is_dtype(detect_dtype(srs, dtype), Continuous()):
        if range is not None:
            minimum, maximum = range
        else:
            minimum, maximum = srs.min(axis=0), srs.max(axis=0)
        minimum, maximum = dask.compute(minimum, maximum)

        assert (
            bins is not None
        ), "num_bins cannot be None if calculating numerical histograms"

        counts, edges = da.histogram(
            srs.to_dask_array(), bins, range=[minimum, maximum]
        )
        centers = (edges[:-1] + edges[1:]) / 2

        if not return_edges:
            return counts, centers
        return counts, centers, edges
    elif is_dtype(detect_dtype(srs, dtype), Nominal()):
        value_counts = srs.value_counts()
        counts = value_counts.to_dask_array()

        # Dask array dones't understand the pandas dtypes such as categorical type.
        # We convert these types into str before calling into `to_dask_array`.

        if is_pandas_categorical(value_counts.index.dtype):
            centers = value_counts.index.astype("str").to_dask_array()
        else:
            centers = value_counts.index.to_dask_array()
        return (counts, centers)
    else:
        raise UnreachableError()


def missing_perc_blockwise(block: np.ndarray) -> np.ndarray:
    """
    Compute the missing percentage in a block
    """
    return block.sum(axis=0, keepdims=True) / len(block)


def missing_spectrum(df: dd.DataFrame, bins: int, ncols: int) -> Intermediate:
    """
    Calculate a missing spectrum for each column
    """
    # pylint: disable=too-many-locals
    num_bins = min(bins, len(df) - 1)

    df = df.iloc[:, :ncols]
    cols = df.columns[:ncols]
    ncols = len(cols)
    nrows = len(df)
    chunk_size = len(df) // num_bins
    data = df.isnull().to_dask_array()
    data.compute_chunk_sizes()
    data = data.rechunk((chunk_size, None))

    (notnull_counts,) = dd.compute(data.sum(axis=0) / data.shape[0])
    missing_percent = {col: notnull_counts[idx] for idx, col in enumerate(cols)}

    missing_percs = data.map_blocks(missing_perc_blockwise, dtype=float).compute()
    locs0 = np.arange(len(missing_percs)) * chunk_size
    locs1 = np.minimum(locs0 + chunk_size, nrows)
    locs_middle = locs0 + chunk_size / 2

    df = pd.DataFrame(
        {
            "column": np.repeat(cols.values, len(missing_percs)),
            "location": np.tile(locs_middle, ncols),
            "missing_rate": missing_percs.T.ravel(),
            "loc_start": np.tile(locs0, ncols),
            "loc_end": np.tile(locs1, ncols),
        }
    )
    return Intermediate(
        data=df, missing_percent=missing_percent, visual_type="missing_spectrum"
    )


def missing_impact_1vn(  # pylint: disable=too-many-locals
    df: dd.DataFrame, x: str, bins: int, dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    """
    Calculate the distribution change on other columns when
    the missing values in x is dropped.
    """
    df0 = df
    df1 = df.dropna(subset=[x])
    cols = [col for col in df.columns if col != x]

    hists = {}
    hists_restore_dtype = {}

    for col in cols:
        range = None  # pylint: disable=redefined-builtin
        if is_dtype(detect_dtype(df0[col], dtype), Continuous()):
            range = (df0[col].min(axis=0), df0[col].max(axis=0))

        hists[col] = [
            histogram(df[col], dtype=dtype, bins=bins, return_edges=True, range=range)
            for df in [df0, df1]
        ]

        # In some cases(Issue#98), dd.compute() can change the features dtypes and cause error.
        # So we need to restore features dtypes after dd.compute().
        centers_dtypes = (hists[col][0][1].dtype, hists[col][1][1].dtype)
        (hists,) = dd.compute(hists)
        dict_value = []

        # Here we do not reassign to the "hists" variable as
        # dd.compute() can change variables' types and cause error to mypy test in CircleCI .
        # Instead, we assign to a new variable hists_restore_dtype.
        for i in [0, 1]:
            intermediate = list(hists[col][i])
            intermediate[1] = intermediate[1].astype(centers_dtypes[i])
            dict_value.append(tuple(intermediate))
        hists_restore_dtype[col] = dict_value

    dfs = {}

    meta = ColumnsMetadata()

    for col, hists_ in hists_restore_dtype.items():
        counts, xs, *edges = zip(*hists_)

        labels = np.repeat(LABELS, [len(x) for x in xs])

        data = {
            "x": np.concatenate(xs),
            "count": np.concatenate(counts),
            "label": labels,
        }

        if edges:
            lower_bound: List[float] = []
            upper_bound: List[float] = []

            for edge in edges[0]:
                lower_bound.extend(edge[:-1])
                upper_bound.extend(edge[1:])

            data["lower_bound"] = lower_bound
            data["upper_bound"] = upper_bound

        df = pd.DataFrame(data)

        # If the cardinality of a categorical column is too large,
        # we show the top `num_bins` values, sorted by their count before drop
        if len(counts[0]) > bins and is_dtype(detect_dtype(df0[col], dtype), Nominal()):
            sortidx = np.argsort(-counts[0])
            selected_xs = xs[0][sortidx[:bins]]
            df = df[df["x"].isin(selected_xs)]
            meta[col, "partial"] = (bins, len(counts[0]))
        else:
            meta[col, "partial"] = (len(counts[0]), len(counts[0]))
        meta[col, "dtype"] = detect_dtype(df0[col], dtype)
        dfs[col] = df

    return Intermediate(data=dfs, x=x, meta=meta, visual_type="missing_impact_1vn")


def missing_impact_1v1(  # pylint: disable=too-many-locals
    df: dd.DataFrame,
    x: str,
    y: str,
    bins: int,
    ndist_sample: int,
    dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    # pylint: disable=too-many-arguments
    """
    Calculate the distribution change on another column y when
    the missing values in x is dropped.
    """

    df0 = df[[x, y]]
    df1 = df.dropna(subset=[x])

    srs0, srs1 = df0[y], df1[y]
    minimum, maximum = srs0.min(), srs0.max()

    hists = [
        histogram(srs, dtype=dtype, bins=bins, return_edges=True)
        for srs in [srs0, srs1]
    ]
    hists = da.compute(*hists)

    meta = ColumnsMetadata()
    meta["y", "dtype"] = detect_dtype(df[y], dtype)

    if is_dtype(detect_dtype(df[y], dtype), Continuous()):
        dists = [rv_histogram((hist[0], hist[2])) for hist in hists]  # type: ignore
        xs = np.linspace(minimum, maximum, ndist_sample)

        pdfs = [dist.pdf(xs) for dist in dists]
        cdfs = [dist.cdf(xs) for dist in dists]

        distdf = pd.DataFrame(
            {
                "x": np.tile(xs, 2),
                "pdf": np.concatenate(pdfs),
                "cdf": np.concatenate(cdfs),
                "label": np.repeat(LABELS, ndist_sample),
            }
        )

        counts, xs, edges = zip(*hists)

        lower_bounds: List[float] = []
        upper_bounds: List[float] = []

        for edge in edges:
            lower_bounds.extend(edge[:-1])
            upper_bounds.extend(edge[1:])

        histdf = pd.DataFrame(
            {
                "x": np.concatenate(xs),
                "count": np.concatenate(counts),
                "label": np.repeat(LABELS, [len(count) for count in counts]),
                "lower_bound": lower_bounds,
                "upper_bound": upper_bounds,
            }
        )

        quantiles = [
            [srs.quantile(q) for q in [0, 0.25, 0.5, 0.75, 1]] for srs in [srs0, srs1]
        ]
        quantiles = dd.compute(*quantiles)

        boxdf = pd.DataFrame(quantiles)
        boxdf.columns = ["min", "q1", "q2", "q3", "max"]

        iqr = boxdf["q3"] - boxdf["q1"]
        boxdf["upper"] = np.minimum(boxdf["q3"] + 1.5 * iqr, boxdf["max"])
        boxdf["lower"] = np.maximum(boxdf["q3"] - 1.5 * iqr, boxdf["min"])
        boxdf["label"] = LABELS

        itmdt = Intermediate(
            dist=distdf,
            hist=histdf,
            box=boxdf,
            meta=meta["y"],
            x=x,
            y=y,
            visual_type="missing_impact_1v1",
        )
        return itmdt
    else:

        counts, xs = zip(*hists)

        df = pd.DataFrame(
            {
                "x": np.concatenate(xs, axis=0),
                "count": np.concatenate(counts, axis=0),
                "label": np.repeat(LABELS, [len(count) for count in counts]),
            }
        )

        # If the cardinality of a categorical column is too large,
        # we show the top `num_bins` values, sorted by their count before drop
        if len(counts[0]) > bins:
            sortidx = np.argsort(-counts[0])
            selected_xs = xs[0][sortidx[:bins]]
            df = df[df["x"].isin(selected_xs)]
            partial = (bins, len(counts[0]))
        else:
            partial = (len(counts[0]), len(counts[0]))

        meta["y", "partial"] = partial

        itmdt = Intermediate(
            hist=df, x=x, y=y, meta=meta["y"], visual_type="missing_impact_1v1",
        )
        return itmdt


def compute_missing(
    # pylint: disable=too-many-arguments
    df: Union[pd.DataFrame, dd.DataFrame],
    x: Optional[str] = None,
    y: Optional[str] = None,
    *,
    bins: int = 30,
    ncols: int = 30,
    ndist_sample: int = 100,
    dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    """
    This function is designed to deal with missing values
    There are three functions: plot_missing(df), plot_missing(df, x)
    plot_missing(df, x, y)

    Parameters
    ----------
    df
        the pandas data_frame for which plots are calculated for each column
    x
        a valid column name of the data frame
    y
        a valid column name of the data frame
    ncols
        The number of columns in the figure
    bins
        The number of rows in the figure
    ndist_sample
        The number of sample points
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    Examples
    ----------
    >>> from dataprep.eda.missing.computation import plot_missing
    >>> import pandas as pd
    >>> df = pd.read_csv("suicide-rate.csv")
    >>> plot_missing(df, "HDI_for_year")
    >>> plot_missing(df, "HDI_for_year", "population")
    """

    df = to_dask(df)

    # pylint: disable=no-else-raise
    if x is None and y is not None:
        raise ValueError("x cannot be None while y has value")
    elif x is not None and y is None:
        return missing_impact_1vn(df, dtype=dtype, x=x, bins=bins)
    elif x is not None and y is not None:
        return missing_impact_1v1(
            df, dtype=dtype, x=x, y=y, bins=bins, ndist_sample=ndist_sample
        )
    else:
        return missing_spectrum(df, bins=bins, ncols=ncols)
