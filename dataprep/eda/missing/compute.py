"""
    This module implements the plot_missing(df) function's
    calculating intermediate part
"""
from typing import Optional, Tuple, Union, Dict

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.stats import rv_histogram

from ...errors import UnreachableError
from ..utils import to_dask
from ..intermediate import Intermediate
from ..dtypes import is_categorical, is_numerical

__all__ = ["compute_missing"]


def histogram(
    srs: dd.Series,
    num_bins: Optional[int] = None,
    return_edges: bool = True,
    range: Optional[Tuple[int, int]] = None,  # pylint: disable=redefined-builtin
) -> Tuple[da.Array, da.Array]:
    """
    Calculate histogram for both numerical and categorical
    """

    if is_numerical(srs.dtype):
        if range is not None:
            minimum, maximum = range
        else:
            minimum, maximum = srs.min(axis=0), srs.max(axis=0)

        assert (
            num_bins is not None
        ), "num_bins cannot be None if calculating numerical histograms"

        counts, edges = da.histogram(
            srs.to_dask_array(), num_bins, range=[minimum, maximum]
        )
        if not return_edges:
            centers = (edges[:-1] + edges[1:]) / 2
            return counts, centers
        return counts, edges
    elif is_categorical(srs.dtype):
        value_counts = srs.value_counts()

        counts = value_counts.to_dask_array()
        centers = value_counts.index.to_dask_array()
        return (counts, centers)
    else:
        raise UnreachableError()


def missing_spectrum(df: dd.DataFrame, num_bins: int, num_cols: int) -> Intermediate:
    """
    Calculate a missing spectrum for each column
    """
    # pylint: disable=too-many-locals

    df = df.iloc[:, :num_cols]
    cols = df.columns[:num_cols]

    data = df.isnull().to_dask_array()
    data.compute_chunk_sizes()

    (notnull_counts,) = dd.compute(data.sum(axis=0) / data.shape[0])
    missing_percent = {col: notnull_counts[idx] for idx, col in enumerate(cols)}

    locs = da.arange(data.shape[0])

    hists = []
    for j in range(data.shape[1]):
        mask = data[:, j]
        counts, edges = da.histogram(
            locs[mask], bins=num_bins, range=(0, data.shape[0] - 1)
        )
        centers = (edges[:-1] + edges[1:]) / 2
        radius = np.average(edges[1:] - edges[:-1])
        hist = np.stack([centers, counts / radius, edges[:-1], edges[1:]]).T
        hists.append(hist)
    hists = da.compute(*hists)

    data = np.concatenate(hists, axis=0)
    labels = np.repeat(cols, [len(hist) for hist in hists])
    data = np.concatenate([labels[:, None], data], axis=1)
    df = pd.DataFrame(
        data=data,
        columns=["column", "location", "missing_rate", "loc_start", "loc_end"],
    )
    return Intermediate(
        data=df, missing_percent=missing_percent, visual_type="missing_spectrum"
    )


def missing_impact_1vn(  # pylint: disable=too-many-locals
    df: dd.DataFrame, x: str, num_bins: int
) -> Intermediate:
    """
    Calculate the distribution change on other columns when
    the missing values in x is dropped.
    """
    df0 = df
    df1 = df.dropna(subset=[x])
    cols = [col for col in df.columns if col != x]

    hists = {}

    for col in cols:
        range = None  # pylint: disable=redefined-builtin
        if is_numerical(df0[col].dtype):
            range = (df0[col].min(axis=0), df0[col].max(axis=0))

        hists[col] = [
            histogram(df[col], num_bins=num_bins, return_edges=False, range=range)
            for df in [df0, df1]
        ]
    (hists,) = dd.compute(hists)

    dfs = {}
    # partial stores total number of bins
    # and how many columns are shown for each column
    partial: Dict[str, Optional[Tuple[int, int]]] = {}
    for col, hists_ in hists.items():
        counts, xs = zip(*hists_)
        labels = np.repeat(["Origin", "DropMissing"], [len(x) for x in xs])

        df = pd.DataFrame(
            {"x": np.concatenate(xs), "count": np.concatenate(counts), "label": labels}
        )

        # If the cardinality of a categorical column is too large,
        # we show the top `num_bins` values, sorted by their count before drop
        if len(counts[0]) > num_bins and is_categorical(df0[col].dtype):
            sortidx = np.argsort(-counts[0])
            selected_xs = xs[0][sortidx[:num_bins]]
            df = df[df["x"].isin(selected_xs)]
            partial[col] = (num_bins, len(counts[0]))
        else:
            partial[col] = (len(counts[0]), len(counts[0]))
        dfs[col] = df

    return Intermediate(
        data=dfs, x=x, partial=partial, visual_type="missing_impact_1vn"
    )


def missing_impact_1v1(  # pylint: disable=too-many-locals
    df: dd.DataFrame, x: str, y: str, num_bins: int, num_dist_sample: int
) -> Intermediate:
    """
    Calculate the distribution change on another column y when
    the missing values in x is dropped.
    """

    df0 = df[[x, y]]
    df1 = df.dropna(subset=[x])

    srs0, srs1 = df0[y], df1[y]
    minimum, maximum = srs0.min(), srs0.max()

    hists = [histogram(srs, num_bins=num_bins) for srs in [srs0, srs1]]
    hists = da.compute(*hists)

    if is_numerical(df[y].dtype):
        dists = [rv_histogram(hist) for hist in hists]
        xs = np.linspace(minimum, maximum, num_dist_sample)

        pdfs = [dist.pdf(xs) for dist in dists]
        cdfs = [dist.cdf(xs) for dist in dists]

        distdf = pd.DataFrame(
            {
                "x": np.tile(xs, 2),
                "pdf": np.concatenate(pdfs),
                "cdf": np.concatenate(cdfs),
                "label": np.repeat(["Origin", "DropMissing"], num_dist_sample),
            }
        )

        counts, edges = zip(*hists)
        xs = [(edge[1:] + edge[:-1]) / 2 for edge in edges]

        histdf = pd.DataFrame(
            {
                "x": np.concatenate(xs),
                "count": np.concatenate(counts),
                "label": np.repeat(
                    ["Origin", "DropMissing"], [len(count) for count in counts]
                ),
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
        boxdf["label"] = ["Origin", "DropMissing"]

        itmdt = Intermediate(
            dist=distdf,
            hist=histdf,
            box=boxdf,
            x=x,
            y=y,
            visual_type="missing_impact_1v1_numerical",
        )
        return itmdt
    else:

        counts, xs = zip(*hists)

        df = pd.DataFrame(
            {
                "x": np.concatenate(xs, axis=0),
                "count": np.concatenate(counts, axis=0),
                "label": np.repeat(
                    ["Origin", "DropMissing"], [len(count) for count in counts]
                ),
            }
        )

        # If the cardinality of a categorical column is too large,
        # we show the top `num_bins` values, sorted by their count before drop
        if len(counts[0]) > num_bins:
            sortidx = np.argsort(-counts[0])
            selected_xs = xs[0][sortidx[:num_bins]]
            df = df[df["x"].isin(selected_xs)]
            partial = (num_bins, len(counts[0]))
        else:
            partial = (len(counts[0]), len(counts[0]))

        itmdt = Intermediate(
            hist=df,
            x=x,
            y=y,
            partial=partial,
            visual_type="missing_impact_1v1_categorical",
        )
        return itmdt


def compute_missing(
    # pylint: disable=too-many-arguments
    df: Union[pd.DataFrame, dd.DataFrame],
    x: Optional[str] = None,
    y: Optional[str] = None,
    *,
    num_bins: int = 30,
    num_cols: int = 30,
) -> Intermediate:
    """
    This function is designed to deal with missing values
    There are three functions: plot_missing(df), plot_missing(df, x)
    plot_missing(df, x, y)

    Parameters
    ----------
    pd_data_frame: pd.DataFrame
        the pandas data_frame for which plots are calculated for each column
    x_name: str, optional
        a valid column name of the data frame
    y_name: str, optional
        a valid column name of the data frame
    num_cols: int, optional
        The number of columns in the figure
    bins_num: int
        The number of rows in the figure
    return_intermediate: bool
        whether show intermediate results to users

    Returns
    ----------
    An object of figure or
        An object of figure and
        An intermediate representation for the plots of different columns in the data_frame.

    Examples
    ----------
    >>> from dataprep.eda.missing.computation import plot_missing
    >>> import pandas as pd
    >>> df = pd.read_csv("suicide-rate.csv")
    >>> plot_missing(df, "HDI_for_year")
    >>> plot_missing(df, "HDI_for_year", "population")

    Notes
    ----------
    match (x_name, y_name)
        case (Some, Some) => histogram for numerical column,
        bars for categorical column, qq-plot, box-plot, jitter plot,
        CDF, PDF
        case (Some, None) => histogram for numerical column and
        bars for categorical column
        case (None, None) => heatmap
        otherwise => error
    """

    df = to_dask(df)

    # pylint: disable=no-else-raise
    if x is None and y is not None:
        raise ValueError("x cannot be None while y has value")
    elif x is not None and y is None:
        return missing_impact_1vn(df, x=x, num_bins=num_bins)
    elif x is not None and y is not None:
        return missing_impact_1v1(df, x=x, y=y, num_bins=num_bins, num_dist_sample=100)
    else:
        return missing_spectrum(df, num_bins=num_bins, num_cols=num_cols)
