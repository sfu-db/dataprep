"""
    This module implements the formatting
    for create_report(df) function.
"""
from typing import Any, Dict, List, Optional, Union

import dask
import dask.dataframe as dd
import pandas as pd
from bokeh.embed import components
from bokeh.models import Title
from bokeh.plotting import Figure

from ..correlation import render_correlation
from ..correlation.compute.nullivariate import correlation_nxn_frame
from ..distribution import render
from ..distribution.compute.overview import calc_stats
from ..distribution.compute.univariate import cont_comps, nom_comps
from ..distribution.render import format_cat_stats, format_num_stats, format_ov_stats
from ..dtypes import (
    CATEGORICAL_DTYPES,
    NUMERICAL_DTYPES,  # DateTime,
    Continuous,
    Nominal,
    detect_dtype,
    is_dtype,
)
from ..intermediate import Intermediate
from ..missing import render_missing
from ..missing.compute import dlyd_missing_comps
from ..progress_bar import ProgressBar
from ..utils import to_dask


def format_report(
    df: Union[pd.DataFrame, dd.DataFrame], mode: Optional[str]
) -> Dict[str, Any]:
    """
    Format the data and figures needed by report

    Parameters
    ----------
    df
        The DataFrame for which data are calculated.
    mode: Optional[str]
        This controls what type of report to be generated.
        Currently only the 'basic' is fully implemented.

    Returns
    -------
    Dict[str, Any]
        A dictionary in which formatted data will be stored.
        This variable acts like an API in passing data to the template engine.
    """
    # pylint: disable=too-many-locals,too-many-statements
    with ProgressBar(minimum=1):
        df = to_dask(df)
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
    data = basic_computations(df)

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
            itmdt = Intermediate(
                col=col, data=data[col], visual_type="numerical_column"
            )
            rndrd = render(itmdt, plot_height_lrg=250, plot_width_lrg=280)
            stats = format_num_stats(data[col])
        elif is_dtype(detect_dtype(df[col]), Nominal()):
            itmdt = Intermediate(
                col=col, data=data[col], visual_type="categorical_column"
            )
            rndrd = render(itmdt, plot_height_lrg=250, plot_width_lrg=280)
            stats = format_cat_stats(
                data[col]["stats"], data[col]["len_stats"], data[col]["letter_stats"]
            )
        figs: List[Figure] = []
        for tab in rndrd.tabs[1:]:
            fig = tab.child.children[0]
            fig.title = Title(text=tab.title, align="center")
            figs.append(fig)
        res["variables"][col] = {
            "tabledata": stats,
            "plots": components(figs),
            "col_type": itmdt.visual_type.replace("_column", ""),
        }

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
        data=dfs, axis_range=list(data["num_cols"]), visual_type="correlation_heatmaps",
    )
    rndrd = render_correlation(itmdt)
    figs.clear()
    for tab in rndrd.tabs:
        fig = tab.child
        fig.sizing_mode = "stretch_width"
        fig.title = Title(text=tab.title, align="center", text_font_size="20px")
        figs.append(fig)
    res["correlations"] = components(figs)

    # missing
    res["has_missing"] = True
    spectrum, null_perc, bars, heatmap, dendrogram = data["miss"]
    itmdt = Intermediate(
        data_total_missing={col: null_perc[i] for i, col in enumerate(df.columns)},
        data_spectrum=spectrum,
        data_bars=bars,
        data_heatmap=heatmap,
        data_dendrogram=dendrogram,
        visual_type="missing_impact",
    )
    rndrd = render_missing(itmdt)
    figs.clear()
    for tab in rndrd.tabs:
        fig = tab.child.children[0]
        fig.sizing_mode = "stretch_width"
        fig.title = Title(text=tab.title, align="center", text_font_size="20px")
        figs.append(fig)
    res["missing"] = components(figs)

    return res


def basic_computations(df: dd.DataFrame) -> Dict[str, Any]:
    """
    Computations for the basic version

    Parameters
    ----------
    df
        The DataFrame for which data are calculated
    """
    data: Dict[str, Any] = {}

    df_num = df.select_dtypes(NUMERICAL_DTYPES)
    data["num_cols"] = df_num.columns
    first_rows = df.select_dtypes(CATEGORICAL_DTYPES).head()

    # overview
    data["ov"] = calc_stats(df, None)
    # # variables
    for col in df.columns:
        if is_dtype(detect_dtype(df[col]), Continuous()):
            data[col] = cont_comps(df[col], 20)
        elif is_dtype(detect_dtype(df[col]), Nominal()):
            data[col] = nom_comps(
                df[col], first_rows[col], 10, True, 10, 20, True, False, False
            )
    # interactions
    data["scat"] = df_num.map_partitions(
        lambda x: x.sample(min(1000, x.shape[0])), meta=df_num
    )
    # correlations
    data.update(zip(("cordx", "cordy", "corrs"), correlation_nxn_frame(df_num)))
    # missing values
    data["miss"] = dlyd_missing_comps(df, 30)

    return data
