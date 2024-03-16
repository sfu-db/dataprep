"""This module implements the plot_missing(df, x) function's
calculating intermediate part
"""

from typing import Any, Generator, List

import numpy as np
import pandas as pd

from ...configs import Config
from ...eda_frame import EDAFrame
from ...dtypes_v2 import Continuous, Nominal, GeoGraphy, SmallCardNum, DateTime
from ...intermediate import ColumnsMetadata, Intermediate
from ...staged import staged
from .common import LABELS, uni_histogram


def _compute_missing_univariate(  # pylint: disable=too-many-locals
    df: EDAFrame,
    x: str,
    cfg: Config,
) -> Generator[Any, Any, Intermediate]:
    """Calculate the distribution change on other columns when
    the missing values in x is dropped."""
    # pylint: disable = too-many-boolean-expressions

    # dataframe with all rows where column x is null removed
    ddf = df.frame[~df.frame[x].isna()]

    hists = {}

    for col in df.columns:
        col_dtype = df.get_eda_dtype(col)
        if (
            col == x
            or (
                isinstance(col_dtype, (Nominal, GeoGraphy, SmallCardNum, DateTime))
                and not cfg.bar.enable
            )
            or (isinstance(col_dtype, Continuous) and not cfg.hist.enable)
        ):
            continue

        if isinstance(col_dtype, (SmallCardNum, DateTime)):
            srs0 = df.frame[col].dropna().astype(str)  # series from original dataframe
            srs1 = ddf[col].dropna().astype(str)  # series with null rows from col x removed
        elif isinstance(col_dtype, (GeoGraphy, Nominal, Continuous)):
            # Geograph, Nominal should be transformed to str when constructing edaframe.
            # Here we do not need to transform them again.
            srs0 = df.frame[col].dropna()
            srs1 = ddf[col].dropna()
        else:
            raise ValueError(f"unprocessed type:{col_dtype}")

        hists[col] = [uni_histogram(srs, col_dtype, cfg) for srs in [srs0, srs1]]

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
        col_dtype = df.get_eda_dtype(col_name)
        if len(counts[0]) > cfg.bar.bars and (
            isinstance(col_dtype, (Nominal, GeoGraphy, SmallCardNum, DateTime))
        ):
            sortidx = np.argsort(-counts[0])
            selected_xs = xs[0][sortidx[: cfg.bar.bars]]
            ret_df = ret_df[ret_df["x"].isin(selected_xs)]
            meta[col_name, "shown"] = cfg.bar.bars
        else:
            meta[col_name, "shown"] = len(counts[0])
        meta[col_name, "total"] = len(counts[0])
        meta[col_name, "dtype"] = col_dtype
        dfs[col_name] = ret_df

    return Intermediate(data=dfs, x=x, meta=meta, visual_type="missing_impact_1vn")


# Not using decorator here because jupyter autoreload does not support it.
compute_missing_univariate = staged(_compute_missing_univariate)  # pylint: disable=invalid-name
