"""This module implements the formatting
for create_report(df) function."""

from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import catch_warnings, filterwarnings

import dask
import dask.dataframe as dd
import pandas as pd
from bokeh.embed import components
from bokeh.models import Title
from bokeh.plotting import Figure

from ..correlation import render_correlation
from ..correlation.compute.nullivariate import correlation_nxn
from ..data_array import DataArray
from ..distribution import render
from ..distribution.compute.common import _calc_line_dt
from ..distribution.compute.overview import calc_stats
from ..distribution.compute.univariate import cont_comps, nom_comps, calc_stats_dt
from ..distribution.render import format_cat_stats, format_num_stats, format_ov_stats, stats_viz_dt
from ..dtypes import (
    CATEGORICAL_DTYPES,
    Continuous,
    DateTime,
    Nominal,
    detect_dtype,
    is_dtype,
    string_dtype_to_object,
)
from ..intermediate import Intermediate
from ..missing import render_missing
from ..missing.compute.nullivariate import compute_missing_nullivariate
from ..progress_bar import ProgressBar
from ..utils import to_dask


def format_report(
    df: Union[pd.DataFrame, dd.DataFrame], mode: Optional[str], progress: bool = True
) -> Dict[str, Any]:
    """
    Format the data and figures needed by report

    Parameters
    ----------
    df
        The DataFrame for which data are calculated.
    mode
        This controls what type of report to be generated.
        Currently only the 'basic' is fully implemented.
    progress
        Whether to show the progress bar.

    Returns
    -------
    Dict[str, Any]
        A dictionary in which formatted data will be stored.
        This variable acts like an API in passing data to the template engine.
    """
    # pylint: disable=too-many-locals,too-many-statements
    with ProgressBar(minimum=1, disable=not progress):
        df = to_dask(df)
        df = string_dtype_to_object(df)
        if mode == "basic":
            comps = format_basic(df)
        # elif mode == "full":
        #     comps = format_full(df)
        # elif mode == "minimal":
        #     comps = format_mini(df)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return comps


def format_basic(df: dd.DataFrame) -> Dict[str, Any]:
    # pylint: disable=too-many-statements
    """
    Format basic version.

    Parameters
    ----------
    df
        The DataFrame for which data are calculated.

    Returns
    -------
    Dict[str, Any]
        A dictionary in which formatted data is stored.
        This variable acts like an API in passing data to the template engine.
    """
    # pylint: disable=too-many-locals
    # aggregate all computations
    data, completions = basic_computations(df)

    with catch_warnings():
        filterwarnings(
            "ignore",
            "invalid value encountered in true_divide",
            category=RuntimeWarning,
        )
        (data,) = dask.compute(data)

    # results dictionary
    res: Dict[str, Any] = {}

    # overview
    data["ov"].pop("ks_tests")
    res["overview"] = format_ov_stats(data["ov"])

    # variables
    res["variables"] = {}
    for col in df.columns:
        stats: Any = None  # needed for pylint
        if is_dtype(detect_dtype(df[col]), Continuous()):
            itmdt = Intermediate(col=col, data=data[col], visual_type="numerical_column")
            stats = format_num_stats(data[col])
        elif is_dtype(detect_dtype(df[col]), Nominal()):
            itmdt = Intermediate(col=col, data=data[col], visual_type="categorical_column")
            stats = format_cat_stats(
                data[col]["stats"], data[col]["len_stats"], data[col]["letter_stats"]
            )
        elif is_dtype(detect_dtype(df[col]), DateTime()):
            itmdt = Intermediate(
                col=col,
                data=data[col]["stats"],
                line=data[col]["line"],
                visual_type="datetime_column",
            )
            stats = stats_viz_dt(data[col]["stats"])
        rndrd = render(itmdt, plot_height_lrg=250, plot_width_lrg=280)["layout"]
        figs: List[Figure] = []
        for tab in rndrd:
            try:
                fig = tab.children[0]
            except AttributeError:
                fig = tab
            # fig.title = Title(text=tab.title, align="center")
            figs.append(fig)
        res["variables"][col] = {
            "tabledata": stats,
            "plots": components(figs),
            "col_type": itmdt.visual_type.replace("_column", ""),
        }

    if len(data["num_cols"]) > 0:
        # interactions
        res["has_interaction"] = True
        itmdt = Intermediate(data=data["scat"], visual_type="correlation_crossfilter")
        rndrd = render_correlation(itmdt)
        rndrd.sizing_mode = "stretch_width"
        res["interactions"] = components(rndrd)

        # correlations
        res["has_correlation"] = True
        dfs: Dict[str, pd.DataFrame] = {}
        for method, corr in data["corrs"].items():
            ndf = pd.DataFrame(
                {
                    "x": data["num_cols"][data["cordx"]],
                    "y": data["num_cols"][data["cordy"]],
                    "correlation": corr.ravel(),
                }
            )
            dfs[method.name] = ndf[data["cordy"] > data["cordx"]]
        itmdt = Intermediate(
            data=dfs,
            axis_range=list(data["num_cols"]),
            visual_type="correlation_heatmaps",
        )
        rndrd = render_correlation(itmdt)
        figs.clear()
        for tab in rndrd.tabs:
            fig = tab.child
            fig.sizing_mode = "stretch_width"
            fig.title = Title(text=tab.title, align="center", text_font_size="20px")
            figs.append(fig)
        res["correlations"] = components(figs)
    else:
        res["has_interaction"], res["has_correlation"] = False, False

    # missing
    res["has_missing"] = True
    itmdt = completions["miss"](data["miss"])

    rndrd = render_missing(itmdt)
    figs.clear()
    for fig in rndrd["layout"]:
        fig.sizing_mode = "stretch_width"
        fig.title = Title(
            text=rndrd["meta"][rndrd["layout"].index(fig)], align="center", text_font_size="20px"
        )
        figs.append(fig)
    res["missing"] = components(figs)

    return res


def basic_computations(df: dd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Computations for the basic version.

    Parameters
    ----------
    df
        The DataFrame for which data are calculated.
    """
    data: Dict[str, Any] = {}
    df = DataArray(df)

    df_num = df.select_num_columns()
    data["num_cols"] = df_num.columns
    first_rows = df.select_dtypes(CATEGORICAL_DTYPES).head

    # variables
    for col in df.columns:
        if is_dtype(detect_dtype(df.frame[col]), Continuous()):
            data[col] = cont_comps(df.frame[col], 20)
        elif is_dtype(detect_dtype(df.frame[col]), Nominal()):
            # cast the column as string type if it contains a mutable type
            try:
                first_rows[col].apply(hash)
            except TypeError:
                df.frame[col] = df.frame[col].astype(str)
            data[col] = nom_comps(
                df.frame[col], first_rows[col], 10, True, 10, 20, True, False, False
            )
        elif is_dtype(detect_dtype(df.frame[col]), DateTime()):
            data[col] = {}
            data[col]["stats"] = calc_stats_dt(df.frame[col])
            data[col]["line"] = dask.delayed(_calc_line_dt)(df.frame[[col]], "auto")

    # overview
    data["ov"] = calc_stats(df.frame, None)
    # interactions
    data["scat"] = df_num.frame.map_partitions(
        lambda x: x.sample(min(1000, x.shape[0])), meta=df_num.frame
    )
    # correlations
    data.update(zip(("cordx", "cordy", "corrs"), correlation_nxn(df_num)))
    # missing values
    (delayed, completion,) = compute_missing_nullivariate(  # pylint: disable=unexpected-keyword-arg
        df, 30, _staged=True
    )
    data["miss"] = delayed
    completions = {"miss": completion}

    return data, completions
