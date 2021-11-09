"""This module implements the formatting
for create_report(df) function."""  # pylint: disable=line-too-long,

import sys
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import catch_warnings, filterwarnings

import dask
import dask.dataframe as dd
from ...errors import DataprepError
import pandas as pd
from bokeh.embed import components
from bokeh.plotting import Figure
from ..diff import compute_diff
from ..configs import Config
from ..correlation import render_correlation
from ..correlation.compute.overview import correlation_nxn
from ..distribution import render
from ..utils import _calc_line_dt
from ..distribution.compute.overview import calc_stats
from ..distribution.compute.univariate import calc_stats_dt, cont_comps, nom_comps
from ..distribution.render import format_cat_stats, format_num_stats, format_ov_stats, stats_viz_dt
from ..distribution.compute.overview import (
    _nom_calcs,
    _cont_calcs,
    _format_nom_ins,
    _format_cont_ins,
    _format_ov_ins,
    _insight_pagination,
)
from ..dtypes_v2 import (
    Continuous,
    DateTime,
    Nominal,
    GeoGraphy,
    GeoPoint,
    SmallCardNum,
)
from ..eda_frame import EDAFrame
from ..intermediate import Intermediate
from ..missing import render_missing
from ..missing.compute.nullivariate import compute_missing_nullivariate
from ...progress_bar import ProgressBar

from ..diff import render_diff, compute_diff



# pylint: disable=E1133


def format_diff_report(
    dfs: Union[List[Union[pd.DataFrame, dd.DataFrame]], Union[pd.DataFrame, dd.DataFrame]],
    cfg: Config,
    mode: Optional[str],
    progress: bool = True,
) -> Dict[str, Any]:
    """
    Format the data and figures needed by report

    Parameters
    ----------
    dfs
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
    if isinstance(dfs, list):

        if not cfg.diff.label:
            cfg.diff.label = [f"dfs{i+1}" for i in range(len(dfs))]

        if len(dfs) < 2:
            raise DataprepError("create_plot_diff needs at least 2 DataFrames.")
        if len(dfs) > 5:
            raise DataprepError("Too many DataFrames, max: 5.")

    elif isinstance(dfs, dict):

        if not cfg.diff.label:
            cfg.diff.label = list(dfs.keys())

        dfs = list(dfs.values())

    computations = []
    with ProgressBar(minimum=1, disable=not progress):
        if mode == "basic":
            for i, x in enumerate(dfs):
                edaframe = EDAFrame(x)
                computations.append(format_basic(edaframe, cfg, dfs, i))
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return computations


def _format_variables(df: EDAFrame, cfg: Config, data: Dict[str, Any], dfs: Union[List[Union[pd.DataFrame, dd.DataFrame]], Union[pd.DataFrame, dd.DataFrame]]) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    # variables
    if not cfg.variables.enable:
        res["has_variables"] = False
        return res

    res["variables"] = {}
    res["has_variables"] = True
    for col in df.columns:
        try:
            stats: Any = None  # needed for pylint
            dtp = df.get_eda_dtype(col)
            if isinstance(dtp, Continuous):
                itmdt = Intermediate(col=col, data=data[col], visual_type="numerical_column")
                stats = format_num_stats(data[col])
            elif type(dtp) in [Nominal, SmallCardNum, GeoGraphy, GeoPoint]:
                itmdt = Intermediate(col=col, data=data[col], visual_type="categorical_column")
                stats = format_cat_stats(
                    data[col]["stats"], data[col]["len_stats"], data[col]["letter_stats"]
                )
            elif isinstance(dtp, DateTime):
                itmdt = Intermediate(
                    col=col,
                    data=data[col]["stats"],
                    line=data[col]["line"],
                    visual_type="datetime_column",
                )
                stats = stats_viz_dt(data[col]["stats"])
            else:
                raise RuntimeError(f"the type of column {col} is unknown: {type(dtp)}")

            rndrd = render(itmdt, cfg)
            layout = rndrd["layout"]
            figs_var: List[Figure] = []
            for tab in layout:
                try:
                    fig = tab.children[0]
                except AttributeError:
                    fig = tab
                # fig.title = Title(text=tab.title, align="center")
                figs_var.append(fig)
            comp = components(figs_var)
            insight_keys = list(rndrd["insights"].keys())[2:] if rndrd["insights"] else []
            res["variables"][col] = {
                "tabledata": stats,
                "plots": comp,
                "col_type": itmdt.visual_type.replace("_column", ""),
                "tab_name": rndrd["meta"],
                "plots_tab": zip(comp[1][1:], rndrd["meta"][1:], insight_keys),
                "insights_tab": rndrd["insights"],
            }

        except:
            print(f"error happended in column:{col}", file=sys.stderr)
            raise

    return res

def _format_plots(
    dfs: Union[List[Union[pd.DataFrame, dd.DataFrame]], Union[pd.DataFrame, dd.DataFrame]],
    cfg: Config
) -> Dict[str, Any]:

    itmdt = compute_diff(dfs, cfg=cfg)
    return render_diff(itmdt, cfg=cfg)


def _format_overview(data: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    """Format of Overview section"""
    # results dictionary
    res: Dict[str, Any] = {}
    # overview
    if cfg.overview.enable:
        # insight
        all_ins = _format_ov_ins(data["ov"], cfg)
        for col, dtp, dat in data["insights"]:
            if isinstance(dtp, Continuous):
                ins = _format_cont_ins(col, dat, data["ov"]["nrows"], cfg)[1]
            elif type(dtp) in [Nominal, SmallCardNum, GeoGraphy, GeoPoint]:
                ins = _format_nom_ins(col, dat, data["ov"]["nrows"], cfg)[1]
            else:
                continue
            all_ins += ins
        res["overview_insights"] = _insight_pagination(all_ins)
        res["overview"] = format_ov_stats(data["ov"])
        res["has_overview"] = True
    else:
        res["has_overview"] = False
    return res


def format_basic(dfs: List[EDAFrame], 
cfg: Config, 
df_list: Union[List[Union[pd.DataFrame, dd.DataFrame]], 
Union[pd.DataFrame, dd.DataFrame]],
index: int) -> Dict[str, Any]:
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
    # data, completions = basic_computations(dfs, cfg)
    data, _ = basic_computations(dfs, cfg)
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

    res_overview = _format_overview(data, cfg)
    res_variables = _format_variables(dfs, cfg, data, df_list)
    res_plots = None
    if index == 0:
        res_plots = _format_plots(cfg=cfg, dfs=df_list)
    # res_interaction = _format_interaction(data, cfg)
    # res_correlations = _format_correlation(data, cfg)
    # res_missing = _format_missing(data, cfg, completions, df.shape[1])
    # res = {**res_overview, **res_variables, **res_interaction, **res_correlations, **res_missing}
    if res_plots:
        res = {**res_overview, **res_variables, **res_plots}
        return res
    res = {**res_overview, **res_variables}

    return res


def _compute_variables(df: EDAFrame, cfg: Config) -> Dict[str, Any]:
    """Computation of Variables section."""
    data: Dict[str, Any] = {}
    # variables
    if cfg.variables.enable:
        for col in df.columns:
            try:
                dtype = df.get_eda_dtype(col)
                # Since it will throw error if a numerical column is all-nan,
                # we transform it to categorical column.
                # We also transform to categorical for small cardinality numerical column.
                if df.get_missing_cnt(col) == df.shape[0]:
                    srs = df.get_col_as_str(col, na_as_str=True)
                    data[col] = nom_comps(srs, cfg)
                elif isinstance(dtype, (Nominal, GeoGraphy, GeoPoint)):
                    data[col] = nom_comps(df.frame[col], cfg)
                elif isinstance(dtype, SmallCardNum):
                    srs = df.get_col_as_str(col, na_as_str=False)
                    data[col] = nom_comps(srs, cfg)
                elif isinstance(dtype, Continuous):
                    data[col] = cont_comps(df.frame[col], cfg)
                elif isinstance(dtype, DateTime):
                    data[col] = {}
                    data[col]["stats"] = calc_stats_dt(df.frame[col])
                    data[col]["line"] = dask.delayed(_calc_line_dt)(df.frame[[col]], "auto")
                else:
                    raise ValueError(f"unprocessed type in column{col}:{dtype}")
            except:
                print(f"error happended in column:{col}", file=sys.stderr)
                raise
    return data


def _compute_overview(df: EDAFrame, cfg: Config) -> Dict[str, Any]:
    """Computation of Overview section."""
    data: Dict[str, Any] = {}
    # overview
    if cfg.overview.enable:
        data["ov"] = calc_stats(df, cfg)
        data["insights"] = []
        for col in df.columns:
            col_dtype = df.get_eda_dtype(col)

            # when srs is full NA, the column type will be float
            # and need to be transformed into str.
            if df.get_missing_cnt(col) == df.shape[0]:
                srs = df.get_col_as_str(col, na_as_str=False).dropna().astype(str)
                data["insights"].append((col, Nominal(), _nom_calcs(srs, cfg)))
            elif isinstance(col_dtype, Continuous):
                data["insights"].append(
                    (col, Continuous(), _cont_calcs(df.frame[col].dropna(), cfg))
                )
            elif isinstance(col_dtype, (Nominal, GeoGraphy, GeoPoint, SmallCardNum)):
                srs = df.get_col_as_str(col, na_as_str=False).dropna()
                data["insights"].append((col, Nominal(), _nom_calcs(srs, cfg)))
            elif isinstance(col_dtype, DateTime):
                data["insights"].append(
                    (col, DateTime(), dask.delayed(_calc_line_dt)(df.frame[[col]], cfg.line.unit))
                )
            else:
                raise RuntimeError(f"unprocessed data type: col:{col}, dtype: {type(col_dtype)}")
    return data


def basic_computations(
    df: EDAFrame, cfg: Config
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Computations for the basic version.

    Parameters
    ----------
    df
        The DataFrame for which data are calculated.
    df_num
        The DataFrame of numerical column (used for correlation). It is seperated from df since
        the small distinct value numerical column in df is regarded as categorical column, and
        will transform to str then used for other plots. But they should be regarded as numerical
        column in df_num and used in correlation. This is a temporary fix, in the future we should treat
        those small distinct value numerical columns as ordinary in both correlation plots and other plots.
    cfg
        The config dict user passed in. E.g. config =  {"hist.bins": 20}
        Without user's specifications, the default is "auto"
    """  # pylint: disable=too-many-branches

    variables_data = _compute_variables(df, cfg)
    overview_data = _compute_overview(df, cfg)
    data: Dict[str, Any] = {**variables_data, **overview_data}

    df_num = df.select_num_columns()
    data["num_cols"] = df_num.columns
    # interactions
    if cfg.interactions.enable:
        if cfg.scatter.sample_size is not None:
            sample_func = lambda x: x.sample(n=min(cfg.scatter.sample_size, x.shape[0]))
        else:
            sample_func = lambda x: x.sample(frac=cfg.scatter.sample_rate)
        data["scat"] = df_num.frame.map_partitions(
            sample_func,
            meta=df_num.frame,
        )

    # correlations
    if cfg.correlations.enable:
        data.update(zip(("cordx", "cordy", "corrs"), correlation_nxn(df_num, cfg)))

    # missing values
    completions = None
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
