"""This module implements the formatting
for create_report(df) function."""  # pylint: disable=line-too-long,

from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import catch_warnings, filterwarnings

import dask
import dask.dataframe as dd
import pandas as pd
from bokeh.embed import components
from bokeh.models import Title
from bokeh.plotting import Figure

from ..configs import Config
from ..correlation import render_correlation
from ..correlation.compute.overview import correlation_nxn
from ..data_array import DataArray
from ..distribution import render
from ..distribution.compute.common import _calc_line_dt
from ..distribution.compute.overview import calc_stats
from ..distribution.compute.univariate import calc_stats_dt, cont_comps, nom_comps
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
    df: Union[pd.DataFrame, dd.DataFrame],
    cfg: Config,
    mode: Optional[str],
    progress: bool = True,
) -> Dict[str, Any]:
    """
    Format the data and figures needed by report

    Parameters
    ----------
    df
        The DataFrame for which data are calculated.
    cfg
        The config instance
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
    with ProgressBar(minimum=1, disable=not progress):
        df = to_dask(df)
        df = string_dtype_to_object(df)
        if mode == "basic":
            comps = format_basic(df, cfg)
        # elif mode == "full":
        #     comps = format_full(df)
        # elif mode == "minimal":
        #     comps = format_mini(df)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return comps


def format_basic(df: dd.DataFrame, cfg: Config) -> Dict[str, Any]:
    """
    Format basic version.

    Parameters
    ----------
    df
        The DataFrame for which data are calculated.
    cfg
        The config dict user passed in. E.g. config =  {"hist.bins": 20}
        Without user's specifications, the default is "auto"
    Returns
    -------
    Dict[str, Any]
        A dictionary in which formatted data is stored.
        This variable acts like an API in passing data to the template engine.
    """
    # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    # aggregate all computations
    setattr(getattr(cfg, "plot"), "report", True)
    if cfg.missingvalues.enable:
        data, completions = basic_computations(df, cfg)
    else:
        data = basic_computations(df, cfg)

    with catch_warnings():
        filterwarnings(
            "ignore",
            "invalid value encountered in true_divide",
            category=RuntimeWarning,
        )
        filterwarnings(
            "ignore",
            "overflow encountered in long_scalars",
            category=RuntimeWarning,
        )
        (data,) = dask.compute(data)

    # results dictionary
    res: Dict[str, Any] = {}
    # figure list
    figs: List[Figure] = []
    # overview
    if cfg.overview.enable:
        data["ov"].pop("ks_tests")
        res["overview"] = format_ov_stats(data["ov"])
        res["has_overview"] = True
    else:
        res["has_overview"] = False

    # variables
    if cfg.variables.enable:
        res["variables"] = {}
        res["has_variables"] = True
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
            rndrd = render(itmdt, cfg)["layout"]
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
    else:
        res["has_variables"] = False

    if len(data["num_cols"]) > 0:
        # interactions
        if cfg.interactions.enable:
            res["has_interaction"] = True
            itmdt = Intermediate(data=data["scat"], visual_type="correlation_crossfilter")
            rndrd = render_correlation(itmdt, cfg)
            rndrd.sizing_mode = "stretch_width"
            res["interactions"] = components(rndrd)

        # correlations
        if cfg.correlations.enable:
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
            rndrd = render_correlation(itmdt, cfg)
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
    if cfg.missingvalues.enable:
        res["has_missing"] = True
        itmdt = completions["miss"](data["miss"])

        rndrd = render_missing(itmdt, cfg)
        figs.clear()
        for fig in rndrd["layout"]:
            fig.sizing_mode = "stretch_width"
            fig.title = Title(
                text=rndrd["meta"][rndrd["layout"].index(fig)],
                align="center",
                text_font_size="20px",
            )
            figs.append(fig)
        res["missing"] = components(figs)

    return res


def basic_computations(
    df: dd.DataFrame, cfg: Config
) -> Union[Tuple[Dict[str, Any], Dict[str, Any]], Any]:
    """Computations for the basic version.

    Parameters
    ----------
    df
        The DataFrame for which data are calculated.
    cfg
        The config dict user passed in. E.g. config =  {"hist.bins": 20}
        Without user's specifications, the default is "auto"
    """
    data: Dict[str, Any] = {}
    df = DataArray(df)

    df_num = df.select_num_columns()
    data["num_cols"] = df_num.columns
    first_rows = df.select_dtypes(CATEGORICAL_DTYPES).head
    # variables
    if cfg.variables.enable:
        for col in df.columns:
            if is_dtype(detect_dtype(df.frame[col]), Continuous()):
                data[col] = cont_comps(df.frame[col], cfg)
            elif is_dtype(detect_dtype(df.frame[col]), Nominal()):
                # Since it will throw error if column is object while some cells are
                # numerical, we transform column to string first.
                df.frame[col] = df.frame[col].astype(str)
                data[col] = nom_comps(df.frame[col], first_rows[col], cfg)
            elif is_dtype(detect_dtype(df.frame[col]), DateTime()):
                data[col] = {}
                data[col]["stats"] = calc_stats_dt(df.frame[col])
                data[col]["line"] = dask.delayed(_calc_line_dt)(df.frame[[col]], "auto")
    # overview
    if cfg.overview.enable:
        data["ov"] = calc_stats(df.frame.astype(str), cfg, None)
    # interactions
    if cfg.interactions.enable:
        data["scat"] = df_num.frame.map_partitions(
            lambda x: x.sample(min(1000, x.shape[0])), meta=df_num.frame
        )
    # correlations
    if cfg.correlations.enable:
        data.update(zip(("cordx", "cordy", "corrs"), correlation_nxn(df_num, cfg)))
    # missing values
    if cfg.missingvalues.enable:
        (
            delayed,
            completion,
        ) = compute_missing_nullivariate(  # pylint: disable=unexpected-keyword-arg
            df, cfg, _staged=True
        )
        data["miss"] = delayed
        completions = {"miss": completion}
        return data, completions

    else:
        return data
