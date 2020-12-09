"""This module implements the plot_missing(df) function's
calculating intermediate part."""

from typing import List, Optional, Generator, Any

import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.stats import rv_histogram

from ...data_array import DataArray
from ...dtypes import (
    Continuous,
    DTypeDef,
    detect_dtype,
    is_dtype,
)
from ...intermediate import ColumnsMetadata, Intermediate
from ...staged import staged
from .common import LABELS, histogram


def _compute_missing_bivariate(  # pylint: disable=too-many-locals
    df: DataArray,
    x: str,
    y: str,
    bins: int,
    ndist_sample: int,
    dtype: Optional[DTypeDef] = None,
) -> Generator[Any, Any, Intermediate]:
    # pylint: disable=too-many-arguments
    """Calculate the distribution change on another column y when
    the missing values in x is dropped."""

    xloc = df.columns.get_loc(x)
    yloc = df.columns.get_loc(y)

    col0 = df.values[~df.nulls[:, yloc], yloc].astype(df.dtypes[y])
    col1 = df.values[~(df.nulls[:, xloc] | df.nulls[:, yloc]), yloc].astype(df.dtypes[y])

    minimum, maximum = col0.min(), col0.max()

    hists = [histogram(col, dtype=dtype, bins=bins, return_edges=True) for col in [col0, col1]]

    quantiles = None
    if is_dtype(detect_dtype(df.frame[y], dtype), Continuous()):
        quantiles = [
            dd.from_dask_array(col).quantile([0, 0.25, 0.5, 0.75, 1]) for col in [col0, col1]
        ]

    ### Lazy Region Finished
    minimum, maximum, hists, quantiles = yield (minimum, maximum, hists, quantiles)
    ### Eager region Begin

    meta = ColumnsMetadata()
    meta["y", "dtype"] = detect_dtype(df.frame[y], dtype)

    if is_dtype(detect_dtype(df.frame[y], dtype), Continuous()):
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

        df_ret = pd.DataFrame(
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
            df_ret = df_ret[df_ret["x"].isin(selected_xs)]
            partial = (bins, len(counts[0]))
        else:
            partial = (len(counts[0]), len(counts[0]))

        meta["y", "partial"] = partial

        itmdt = Intermediate(
            hist=df_ret,
            x=x,
            y=y,
            meta=meta["y"],
            visual_type="missing_impact_1v1",
        )
        return itmdt


# Not using decorator here because jupyter autoreload does not support it.
compute_missing_bivariate = staged(_compute_missing_bivariate)  # pylint: disable=invalid-name
