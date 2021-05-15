"""This module implements the plot_missing(df) function's
calculating intermediate part."""

from typing import Any, Generator, List

import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.stats import rv_histogram

from ...configs import Config
from ...eda_frame import EDAFrame
from ...dtypes_v2 import Continuous, Nominal, GeoGraphy, SmallCardNum, DateTime
from ...intermediate import ColumnsMetadata, Intermediate
from ...staged import staged
from .common import LABELS, histogram


def _compute_missing_bivariate(  # pylint: disable=too-many-locals,too-many-statements, too-many-branches
    df: EDAFrame,
    x: str,
    y: str,
    cfg: Config,
) -> Generator[Any, Any, Intermediate]:
    """Calculate the distribution change on another column y when
    the missing values in x is dropped."""

    y_dtype = df.get_eda_dtype(y)
    # dataframe with all rows where column x is null removed
    ddf = df.frame[~df.frame[x].isna()]
    if isinstance(y_dtype, (SmallCardNum, DateTime)):
        col0 = df.frame[y].dropna().astype(str).values  # series from original dataframe
        col1 = ddf[y].dropna().astype(str).values  # series with null rows from col x removed
    elif isinstance(y_dtype, (GeoGraphy, Nominal, Continuous)):
        # Geograph, Nominal should be transformed to str when constructing edaframe.
        # Here we do not need to transform them again.
        col0 = df.frame[y].dropna().values
        col1 = ddf[y].dropna().values
    else:
        raise ValueError(f"unprocessed type:{y_dtype}")

    minimum, maximum = col0.min(), col0.max()
    bins = (
        cfg.bar.bars
        if isinstance(y_dtype, (Nominal, GeoGraphy, SmallCardNum, DateTime))
        else cfg.hist.bins
    )

    hists = [
        histogram(col, eda_dtype=y_dtype, bins=bins, return_edges=True) for col in [col0, col1]
    ]

    quantiles = None
    if isinstance(y_dtype, Continuous) and cfg.box.enable:
        quantiles = [
            dd.from_dask_array(col).quantile([0, 0.25, 0.5, 0.75, 1]) for col in [col0, col1]
        ]

    ### Lazy Region Finished
    minimum, maximum, hists, quantiles = yield (minimum, maximum, hists, quantiles)
    ### Eager region Begin

    meta = ColumnsMetadata()
    meta["y", "dtype"] = y_dtype

    if isinstance(y_dtype, Continuous):

        if cfg.pdf.enable or cfg.cdf.enable:
            dists = [rv_histogram((hist[0], hist[2])) for hist in hists]  # type: ignore
            distdf = pd.DataFrame({})

        if cfg.pdf.enable:
            xs_pdf = np.linspace(minimum, maximum, cfg.pdf.sample_size)
            pdfs = [dist.pdf(xs_pdf) for dist in dists]
            distdf["x_pdf"] = np.tile(xs_pdf, 2)
            distdf["pdf"] = np.concatenate(pdfs)
            distdf["pdf_label"] = np.repeat(LABELS, cfg.pdf.sample_size)

        if cfg.cdf.enable:
            xs_cdf = np.linspace(minimum, maximum, cfg.cdf.sample_size)
            cdfs = [dist.cdf(xs_cdf) for dist in dists]
            distdf["x_cdf"] = np.tile(xs_cdf, 2)
            distdf["cdf"] = np.concatenate(cdfs)
            distdf["cdf_label"] = np.repeat(LABELS, cfg.cdf.sample_size)

        if cfg.hist.enable:
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

        if cfg.box.enable:
            boxdf = pd.DataFrame(quantiles)
            boxdf.columns = ["min", "q1", "q2", "q3", "max"]

            iqr = boxdf["q3"] - boxdf["q1"]
            boxdf["upper"] = np.minimum(boxdf["q3"] + 1.5 * iqr, boxdf["max"])
            boxdf["lower"] = np.maximum(boxdf["q3"] - 1.5 * iqr, boxdf["min"])
            boxdf["label"] = LABELS

        itmdt = Intermediate(
            dist=distdf if cfg.pdf.enable or cfg.cdf.enable else pd.DataFrame({}),
            hist=histdf if cfg.hist.enable else pd.DataFrame({}),
            box=boxdf if cfg.box.enable else pd.DataFrame({}),
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
            shown = bins
        else:
            shown = len(counts[0])

        meta["y", "shown"] = shown
        meta["y", "total"] = len(counts[0])
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
