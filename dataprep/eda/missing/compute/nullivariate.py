"""This module implements the plot_missing(df) function's
calculating intermediate part
"""
from typing import Any, Callable, Dict, Generator, Optional, Tuple, List

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import delayed
from scipy.cluster import hierarchy

from ...data_array import DataArray
from ...intermediate import Intermediate
from ...staged import staged


def _compute_missing_nullivariate(df: DataArray, bins: int) -> Generator[Any, Any, Intermediate]:
    """Calculate the data for visualizing the plot_missing(df).
    This contains the missing spectrum, missing bar chart and missing heatmap."""
    # pylint: disable=too-many-locals

    most_show = 5  # the most number of column/row to show in "insight"
    longest = 5  # the longest length of word to show in "insight"

    df.compute()

    nullity = df.nulls
    null_cnts = nullity.sum(axis=0)
    nrows = df.shape[0]
    ncols = df.shape[1]
    null_perc = null_cnts / nrows
    miss_perc = nullity.sum() / (nrows * ncols)
    avg_row = nullity.sum() / nrows
    avg_col = nullity.sum() / ncols

    tasks = (
        missing_spectrum(df, bins=bins),
        null_perc,
        missing_bars(null_cnts, df.columns.values, nrows),
        missing_heatmap(df),
        missing_dendrogram(df),
        nullity.sum(),
        missing_col_cnt(df),
        missing_row_cnt(df),
        missing_most_col(df),
        missing_most_row(df),
        miss_perc,
        avg_row,
        avg_col,
    )

    ### Lazy Region End
    (
        spectrum,
        null_perc,
        bars,
        heatmap,
        dendrogram,
        cnt,
        col_cnt,
        row_cnt,
        most_col,
        most_row,
        miss_perc,
        avg_row,
        avg_col,
    ) = yield tasks
    ### Eager Region Begin

    sel = ~((null_perc == 0) | (null_perc == 1))
    heatmap = pd.DataFrame(
        data=heatmap[:, sel][sel, :], columns=df.columns[sel], index=df.columns[sel]
    )

    suffix_col = "" if most_col[0] <= most_show else ", ..."
    suffix_row = "" if most_row[0] <= most_show else ", ..."

    top_miss_col = (
        str(most_col[0])
        + "-col(s) "
        + str(
            "("
            + ", ".join(abbr(df.columns[e], longest) for e in most_col[2][:most_show])
            + suffix_col
            + ")"
        )
    )

    top_miss_row = (
        str(most_row[0])
        + "-row(s) "
        + str("(" + ", ".join(str(e) for e in most_row[2][:most_show]) + suffix_row + ")")
    )

    return Intermediate(
        data_total_missing={col: null_perc[idx] for idx, col in enumerate(df.columns)},
        data_spectrum=pd.DataFrame(spectrum),
        data_bars=bars,
        data_heatmap=heatmap,
        data_dendrogram=dendrogram,
        visual_type="missing_impact",
        missing_stat={
            "Missing Cells": cnt,
            "Missing Cells (%)": str(round(miss_perc * 100, 1)) + "%",
            "Missing Columns": col_cnt,
            "Missing Rows": row_cnt,
            "Avg Missing Cells per Column": round(avg_col, 2),
            "Avg Missing Cells per Row": round(avg_row, 2),
        },
        insights={
            "Bar Chart": [
                top_miss_col
                + " contain the most missing values with rate "
                + str(round(most_col[1] * 100, 1))
                + "%",
                top_miss_row
                + " contain the most missing columns with rate "
                + str(round(most_row[1] * 100, 1))
                + "%",
            ]
        },
    )


# Not using decorator here because jupyter autoreload does not support it.
compute_missing_nullivariate = staged(_compute_missing_nullivariate)  # pylint: disable=invalid-name


def missing_perc_blockwise(bin_size: int) -> Callable[[np.ndarray], np.ndarray]:
    """Compute the missing percentage in a block."""

    def imp(block: np.ndarray) -> np.ndarray:
        nbins = block.shape[0] // bin_size

        sep = nbins * bin_size
        block1 = block[:sep].reshape((bin_size, nbins, *block.shape[1:]))
        ret = block1.sum(axis=0) / bin_size

        # remaining data that cannot be fit into a single bin
        if block.shape[0] != sep:
            ret_remainder = block[sep:].sum(axis=0, keepdims=True) / (block.shape[0] - sep)
            ret = np.concatenate([ret, ret_remainder], axis=0)

        return ret

    return imp


def missing_spectrum(  # pylint: disable=too-many-locals
    df: DataArray, bins: int
) -> Dict[str, da.Array]:
    """Calculate a missing spectrum for each column."""

    nrows, ncols = df.shape
    data = df.nulls

    num_bins = min(bins, nrows - 1)
    bin_size = nrows // num_bins
    chunk_size = min(1024 * 1024 * 128, nrows * ncols)  # max 1024 x 1024 x 128 Bytes bool values
    nbins_per_chunk = max(chunk_size // (bin_size * data.shape[1]), 1)

    chunk_size = nbins_per_chunk * bin_size

    data = data.rechunk((chunk_size, None))

    sep = nrows // chunk_size * chunk_size

    spectrum_missing_percs = data[:sep].map_blocks(
        missing_perc_blockwise(bin_size),
        chunks=(nbins_per_chunk, *data.chunksize[1:]),
        dtype=float,
    )

    # calculation for the last chunk
    if sep != nrows:
        spectrum_missing_percs_remain = data[sep:].map_blocks(
            missing_perc_blockwise(bin_size),
            chunks=(int(np.ceil((nrows - sep) / bin_size)), *data.shape[1:]),
            dtype=float,
        )
        spectrum_missing_percs = da.concatenate(
            [spectrum_missing_percs, spectrum_missing_percs_remain], axis=0
        )

    num_bins = spectrum_missing_percs.shape[0]

    locs0 = da.arange(num_bins) * bin_size
    locs1 = da.minimum(locs0 + bin_size, nrows)
    locs_middle = locs0 + bin_size / 2

    return {
        "column": da.repeat(da.from_array(df.columns.values, (1,)), num_bins),
        "location": da.tile(locs_middle, ncols),
        "missing_rate": spectrum_missing_percs.T.ravel().rechunk(locs_middle.shape[0]),
        "loc_start": da.tile(locs0, ncols),
        "loc_end": da.tile(locs1, ncols),
    }


def missing_bars(
    null_cnts: da.Array, cols: np.ndarray, nrows: dd.core.Scalar
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate a bar chart visualization of nullity correlation
    in the given DataFrame."""
    return nrows - null_cnts, null_cnts, cols


def missing_heatmap(df: DataArray) -> Optional[pd.DataFrame]:
    """Calculate a heatmap visualization of nullity correlation
    in the given DataFrame."""

    return da.corrcoef(df.nulls, rowvar=False)


def missing_dendrogram(df: DataArray) -> Any:
    """Calculate a missing values dendrogram."""
    # Link the hierarchical output matrix, figure out orientation, construct base dendrogram.
    linkage_matrix = delayed(hierarchy.linkage)(df.nulls.T, "average")

    dendrogram = delayed(hierarchy.dendrogram)(
        Z=linkage_matrix,
        orientation="bottom",
        labels=df.columns,
        distance_sort="descending",
        no_plot=True,
    )

    return dendrogram


def missing_col_cnt(df: DataArray) -> Any:
    """Calculate how many columns contain missing values."""
    nulls = df.nulls
    rst = nulls.sum(0)
    rst = rst[rst > 0]

    return (rst > 0).sum()


def missing_row_cnt(df: DataArray) -> Any:
    """Calculate how many rows contain missing values."""
    nulls = df.nulls
    rst = nulls.sum(1)
    rst = rst[rst > 0]

    return (rst > 0).sum()


def missing_most_col(df: DataArray) -> Tuple[int, float, List[Any]]:
    """Find which column has the most number of missing values.

    Parameters
    ----------
    df
        the DataArray data_frame

    Outputs
    -------
    cnt
        the count of columns having the most missing values
    rate
        the highest rate of missing values in one column
    rst
        a list of column indices with highest missing rate
    """
    nulls = df.nulls
    col_sum = nulls.sum(axis=0)
    maximum = col_sum.max()
    rate = maximum / df.shape[0]
    cnt = (col_sum == maximum).sum()
    rst = da.where(col_sum == maximum)[0]

    return cnt, rate, rst


def missing_most_row(df: DataArray) -> Tuple[int, float, List[Any]]:
    """Find which row has the most number of missing values.

    Parameters
    ----------
    df
        the DataArray data_frame

    Outputs
    -------
    cnt
        the count of rows having the most missing values
    rate
        the highest rate of missing values in one row
    rst
        a list of row indices with highest missing rate
    """
    nulls = df.nulls
    row_sum = nulls.sum(axis=1)
    maximum = row_sum.max()
    rate = maximum / df.shape[1]
    cnt = (row_sum == maximum).sum()
    rst = da.where(row_sum == maximum)[0]

    return cnt, rate, rst


def abbr(name: str, longest: int) -> str:
    """Cut the name if it is too long."""
    if len(name) > longest:
        return str(name[0:longest] + "...")
    else:
        return name
