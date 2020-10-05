"""This module implements the plot_missing(df) function's
calculating intermediate part
"""
from typing import Any, Generator, List, Optional

import numpy as np
import pandas as pd

from ...data_array import DataArray
from ...dtypes import (
    DTypeDef,
    Nominal,
    detect_dtype,
    is_dtype,
)
from ...intermediate import ColumnsMetadata, Intermediate
from ...staged import staged
from .common import LABELS, uni_histogram


def _compute_missing_univariate(  # pylint: disable=too-many-locals
    df: DataArray,
    x: str,
    bins: int,
    dtype: Optional[DTypeDef] = None,
) -> Generator[Any, Any, Intermediate]:
    """Calculate the distribution change on other columns when
    the missing values in x is dropped."""

    # dataframe with all rows where column x is null removed
    ddf = df.frame[~df.frame[x].isna()]

    hists = {}

    for col in df.columns:
        if col == x:
            continue

        srs0 = df.frame[col].dropna()  # series from original dataframe
        srs1 = ddf[col].dropna()  # series with null rows from col x removed

        hists[col] = [uni_histogram(srs, bins=bins, dtype=dtype) for srs in [srs0, srs1]]

    ### Lazy Region End
    hists = yield hists
    ### Eager Region Begin

    dfs = {}

    meta = ColumnsMetadata()

    for col_name, hists_ in hists.items():
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

        ret_df = pd.DataFrame(data)

        # If the cardinality of a categorical column is too large,
        # we show the top `num_bins` values, sorted by their count before drop
        if len(counts[0]) > bins and is_dtype(detect_dtype(df.frame[col_name], dtype), Nominal()):
            sortidx = np.argsort(-counts[0])
            selected_xs = xs[0][sortidx[:bins]]
            ret_df = ret_df[ret_df["x"].isin(selected_xs)]
            meta[col_name, "partial"] = (bins, len(counts[0]))
        else:
            meta[col_name, "partial"] = (len(counts[0]), len(counts[0]))
        meta[col_name, "dtype"] = detect_dtype(df.frame[col_name], dtype)
        dfs[col_name] = ret_df

    return Intermediate(data=dfs, x=x, meta=meta, visual_type="missing_impact_1vn")


# Not using decorator here because jupyter autoreload does not support it.
compute_missing_univariate = staged(_compute_missing_univariate)  # pylint: disable=invalid-name
