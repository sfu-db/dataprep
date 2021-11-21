"""This module implements the formatting
for create_report(df) function."""  # pylint: disable=line-too-long,

import sys
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import catch_warnings, filterwarnings

import dask
import dask.dataframe as dd
import dask.array as da
from ..utils import to_dask
from ...errors import DataprepError
import pandas as pd
from bokeh.embed import components
from bokeh.plotting import Figure
from ..diff import compute_diff
from ..configs import Config
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

from collections import OrderedDict
from ..dtypes import DType, DTypeDef, is_dtype, detect_dtype, Continuous as Continuous_v1, Nominal as Nominal_v1, DateTime as DateTime_v1
from ..eda_frame import EDAFrame
from ..intermediate import Intermediate
from ..diff.compute.multiple_df import _is_all_int
from ..missing.compute.nullivariate import compute_missing_nullivariate
from ...progress_bar import ProgressBar

from ..diff.compute.multiple_df import _cont_calcs as diff_cont_calcs, _calc_line_dt as diff_calc_line_dt, _nom_calcs as diff_nom_calcs, calc_stats as diff_calc_stats
from ..diff.compute.multiple_df import Srs, Dfs
from ..diff import compute_diff, render_diff
from ..diff.render import hist_viz, bar_viz
from ..palette import CATEGORY10


def format_diff_report(
    df_list: Union[List[pd.DataFrame], Dict[str, pd.DataFrame]],
    cfg: Config,
    mode: Optional[str],
    progress: bool = True,
) -> Dict[str, Any]:
    """
    Format the data and figures needed by create_diff_report

    Parameters
    ----------
    df_list
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
        A dictionary of results
    """
    if isinstance(df_list, list):

        if not cfg.diff.label:
            cfg.diff.label = [f"df{i+1}" for i in range(len(df_list))]

        if len(df_list) < 2:
            raise DataprepError("create_plot_diff needs at least 2 DataFrames.")
        if len(df_list) > 5:
            raise DataprepError("Too many DataFrames, max: 5.")

    elif isinstance(df_list, dict):

        if not cfg.diff.label:
            cfg.diff.label = list(df_list.keys())

        df_list = list(df_list.values())

    with ProgressBar(minimum=1, disable=not progress):
        if mode == "basic":
            report = format_basic(df_list, cfg)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return report

def format_basic(
df_list: List[pd.DataFrame],
cfg: Config
) -> Dict[str, Any]:
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
    final_results: Dict[str, Any] = {"dfs": []}
    delayed_results: List[Any] = []

    for df in df_list:
        df = EDAFrame(df)
        setattr(getattr(cfg, "plot"), "report", True)
        # data, completions = basic_computations(df, cfg)
        # data = dask.delayed(basic_computations)(df, cfg)
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
            # data = dask.compute(data)
            delayed_results.append(data)
    
    figs_var: List[Figure] = []
    res_plots = dask.delayed(_format_plots)(cfg=cfg, df_list=df_list)
    d = {}

    d['v1'] = (delayed_results)
    d['v2'] = res_plots

    test = dask.compute(d)
    
    # computations_results, res_plots = dask.compute(*delayed_results, res_plots)
    
    computations_results, res_plots = test[0]['v1'], test[0]['v2']
    for df, data in zip(df_list, computations_results):
        res_overview = _format_overview(data, cfg)
        res_variables = _format_variables(EDAFrame(df), cfg, data, df_list)
        res = {**res_overview, **res_variables}
        final_results["dfs"].append(res)
        

    
    layout = res_plots["layout"]

    for tab in layout:
        try:
            fig = tab.children[0]
        except AttributeError:
            fig = tab
        figs_var.append(fig)
    # plots = {str(k): v for k, v in enumerate(figs_var)}
    

    plots = components(figs_var)
    final_results["graphs"] = plots

    final_results["legend_lables"] = [
        {"label": label, "color": color}
        for label, color in zip(cfg.diff.label, CATEGORY10[: len(cfg.diff.label)])
    ]

    return final_results

# @dask.delayed(nout=2)
def basic_computations(
    df: EDAFrame, cfg: Config
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Computations for the basic version.

    Parameters
    ----------
    df
        The DataFrame for which data are calculated.
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
    # completions = None
    # if cfg.missingvalues.enable:
    #     (
    #         delayed,
    #         completion,
    #     ) = compute_missing_nullivariate(  # pylint: disable=unexpected-keyword-arg
    #         df, cfg, _staged=True
    #     )
    #     data["miss"] = delayed
    #     completions = {"miss": completion}

    return data

def compute_plot_data(
    df_list: List[dd.DataFrame], cfg: Config, dtype: Optional[DTypeDef]
) -> Intermediate:
    """
    Compute function for plot_diff([df...])

    Parameters
    ----------
    dfs
        Dataframe sequence to be compared.
    cfg
        Config instance
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    """
    # pylint: disable=too-many-branches, too-many-locals

    dfs = Dfs(df_list)
    dfs_cols = dfs.columns.apply("to_list").data

    labeled_cols = dict(zip(cfg.diff.label, dfs_cols))
    baseline: int = cfg.diff.baseline
    data: List[Any] = []
    aligned_dfs = dd.concat(df_list, axis=1)

    # OrderedDict for keeping the order
    uniq_cols = list(OrderedDict.fromkeys(sum(dfs_cols, [])))

    for col in uniq_cols:
        srs = Srs(aligned_dfs[col])
        col_dtype = srs.self_map(detect_dtype, known_dtype=dtype)
        if len(col_dtype) > 1:
            col_dtype = col_dtype[baseline]
        else:
            col_dtype = col_dtype[0]

        orig = [src for src, seq in labeled_cols.items() if col in seq]

        if is_dtype(col_dtype, Continuous_v1()):
            data.append((col, Continuous_v1(), diff_cont_calcs(srs.apply("dropna"), cfg), orig))
        elif is_dtype(col_dtype, Nominal_v1()):
            # When concating dfs, NA may be introduced (e.g., dfs with different rows),
            # making the int column becomes float. Hence we check whether the col should be
            # int after drop NA. If so, we will round column before transform it to str.
            is_int = _is_all_int(df_list, col)
            if is_int:
                norm_srs = srs.apply("dropna").apply(
                    "apply", lambda x: str(round(x)), meta=(col, "object")
                )
            else:
                norm_srs = srs.apply("dropna").apply("astype", "str")

            data.append((col, Nominal_v1(), diff_nom_calcs(norm_srs, cfg), orig))
        # elif is_dtype(col_dtype, DateTime_v1()):
        #     data.append(
        #         (col, DateTime_v1(), dask.delayed(diff_calc_line_dt)(srs, col, cfg.line.unit), orig)
        #     )
        

    stats = diff_calc_stats(dfs, cfg, dtype)
    data, stats = dask.compute(data, stats)
    plot_data: List[Tuple[str, DTypeDef, Any, List[str]]] = []

    for col, dtp, datum, orig in data:
        if is_dtype(dtp, Continuous_v1()):
            if cfg.hist.enable:
                plot_data.append((col, dtp, datum["hist"], orig))
        elif is_dtype(dtp, Nominal_v1()):
            if cfg.bar.enable:
                plot_data.append((col, dtp, (datum["bar"], datum["nuniq"]), orig))
        elif is_dtype(dtp, DateTime_v1()):
            plot_data.append((col, dtp, dask.compute(*datum), orig))  # workaround
    
    return Intermediate(data=plot_data, stats=stats, visual_type="comparison_grid")

# def compute_plot_data(
#     df_list: List[dd.DataFrame], cfg: Config, dtype: Optional[DTypeDef]
# ) -> Intermediate:
#     """
#     Compute function for plot_diff([df...])

#     Parameters
#     ----------
#     dfs
#         Dataframe sequence to be compared.
#     cfg
#         Config instance
#     """
#     # pylint: disable=too-many-branches, too-many-locals

#     dfs = Dfs(df_list)
#     dfs_cols = dfs.columns.apply("to_list").data

#     labeled_cols = dict(zip(cfg.diff.label, dfs_cols))
#     baseline: int = cfg.diff.baseline
#     data: List[Any] = []
#     aligned_dfs = dd.concat(df_list, axis=1)

#     # OrderedDict for keeping the order
#     uniq_cols = list(OrderedDict.fromkeys(sum(dfs_cols, [])))

#     for col in uniq_cols:
#         srs = Srs(aligned_dfs[col])
#         col_dtype = srs.self_map(detect_dtype, known_dtype=dtype)
#         if len(col_dtype) > 1:
#             col_dtype = col_dtype[baseline]
#         else:
#             col_dtype = col_dtype[0]

#         orig = [src for src, seq in labeled_cols.items() if col in seq]

#         if is_dtype(col_dtype, Continuous()):
#             data.append((col, Continuous(), diff_cont_calcs(srs.apply("dropna"), cfg), orig))
#         elif is_dtype(col_dtype, Nominal()):
#             # When concating dfs, NA may be introduced (e.g., dfs with different rows),
#             # making the int column becomes float. Hence we check whether the col should be
#             # int after drop NA. If so, we will round column before transform it to str.
#             is_int = _is_all_int(df_list, col)
#             if is_int:
#                 norm_srs = srs.apply("dropna").apply(
#                     "apply", lambda x: str(round(x)), meta=(col, "object")
#                 )
#             else:
#                 norm_srs = srs.apply("dropna").apply("astype", "str")

#             data.append((col, Nominal(), diff_nom_calcs(norm_srs, cfg), orig))
#         elif is_dtype(col_dtype, DateTime()) and cfg.line.enable:
#             data.append(
#                 (col, DateTime(), dask.delayed(diff_calc_line_dt)(srs, col, cfg.line.unit), orig)
#             )

#     data = dask.compute(data)
#     plot_data: List[Tuple[str, DTypeDef, Any, List[str]]] = []

#     for col, dtp, datum, orig in data:
#         if is_dtype(dtp, Continuous()):
#             if cfg.hist.enable:
#                 plot_data.append((col, dtp, datum["hist"], orig))
#         elif is_dtype(dtp, Nominal()):
#             if cfg.bar.enable:
#                 plot_data.append((col, dtp, (datum["bar"], datum["nuniq"]), orig))
#         elif is_dtype(dtp, DateTime()):
#             plot_data.append((col, dtp, dask.compute(*datum), orig))  # workaround
#     return Intermediate(data=plot_data, visual_type="comparison_grid")

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
                # elif isinstance(dtype, DateTime):
                #     data[col] = {}
                #     data[col]["stats"] = calc_stats_dt(df.frame[col])
                #     data[col]["line"] = dask.delayed(_calc_line_dt)(df.frame[[col]], "auto")
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
            # elif isinstance(col_dtype, DateTime):
            #     data["insights"].append(
            #         (col, DateTime(), dask.delayed(_calc_line_dt)(df.frame[[col]], cfg.line.unit))
            #     )
            else:
                raise RuntimeError(f"unprocessed data type: col:{col}, dtype: {type(col_dtype)}")
    return data


def _format_variables(df: EDAFrame, cfg: Config, data: Dict[str, Any], dfs: Union[List[pd.DataFrame], Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
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

            res["variables"][col] = {
                "tabledata": stats,
                "col_type": itmdt.visual_type.replace("_column", ""),
            }

        except:
            print(f"error happended in column:{col}", file=sys.stderr)
            raise

    return res

def _format_plots(
    df_list: Union[List[pd.DataFrame], Dict[str, pd.DataFrame]],
    cfg: Config
) -> Dict[str, Any]:
    df_list = list(map(to_dask, df_list))
    for i, _ in enumerate(df_list):
        df_list[i].columns = df_list[i].columns.astype(str)

    itmdt = compute_plot_data(df_list=df_list, cfg=cfg, dtype=None)
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