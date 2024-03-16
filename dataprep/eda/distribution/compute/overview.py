"""Computations for plot(df)"""

from itertools import combinations
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union, DefaultDict

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.array.stats import chisquare

from ...configs import Config
from ...dtypes_v2 import (
    Continuous,
    DateTime,
    SmallCardNum,
    GeoPoint,
    DType,
    DTypeDef,
    Nominal,
    GeoGraphy,
)
from ...eda_frame import EDAFrame
from ...utils import _calc_line_dt, ks_2samp, normaltest, skewtest
from ...intermediate import Intermediate


def compute_overview(
    df: Union[pd.DataFrame, dd.DataFrame], cfg: Config, dtype: Optional[DTypeDef]
) -> Intermediate:
    """
    Compute functions for plot(df)

    Parameters
    ----------
    df
        DataFrame from which visualizations are generated
    cfg
        Config instance
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    """
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-locals

    frame = EDAFrame(df, dtype=dtype)

    data: List[Tuple[str, DType, Any]] = []
    for col in frame.columns:
        col_dtype = frame.get_eda_dtype(col)
        if isinstance(col_dtype, Continuous) and (cfg.hist.enable or cfg.insight.enable):
            data.append((col, col_dtype, _cont_calcs(frame.frame[col].dropna(), cfg)))
        elif isinstance(col_dtype, (Nominal, GeoGraphy, GeoPoint, SmallCardNum)) and (
            cfg.bar.enable or cfg.insight.enable
        ):
            srs = frame.get_col_as_str(col).dropna()
            # when srs is full NA, the column type will be float
            # and need to be transformed into str.
            if frame.get_missing_cnt(col) == frame.shape[0]:
                srs = srs.astype(str)
            data.append((col, col_dtype, _nom_calcs(srs, cfg)))
        elif isinstance(col_dtype, DateTime) and (cfg.line.enable or cfg.insight.enable):
            data.append(
                (col, col_dtype, dask.delayed(_calc_line_dt)(frame.frame[[col]], cfg.line.unit))
            )

    ov_stats = calc_stats(frame, cfg)  # overview statistics
    data, ov_stats = dask.compute(data, ov_stats)

    # extract the plotting data, and detect and format the insights
    plot_data: List[Tuple[str, DType, Any]] = []
    col_insights: Dict[str, List[str]] = {}
    all_ins = _format_ov_ins(ov_stats, cfg) if cfg.insight.enable else []

    for col, dtp, dat in data:
        if isinstance(dtp, Continuous):
            if cfg.insight.enable:
                col_ins, ins = _format_cont_ins(col, dat, ov_stats["nrows"], cfg)
            if cfg.hist.enable:
                plot_data.append((col, dtp, dat["hist"]))
        elif isinstance(dtp, (Nominal, GeoGraphy, SmallCardNum, GeoPoint)):
            if cfg.insight.enable:
                col_ins, ins = _format_nom_ins(col, dat, ov_stats["nrows"], cfg)
            if cfg.bar.enable:
                plot_data.append((col, dtp, (dat["bar"].to_frame(), dat["nuniq"])))
        elif isinstance(dtp, DateTime):
            plot_data.append((col, dtp, dat))
            continue

        if cfg.insight.enable:
            if col_ins:
                col_insights[col] = col_ins
            all_ins += ins
    return Intermediate(
        data=plot_data,
        stats=ov_stats,
        column_insights=col_insights,
        overview_insights=_insight_pagination(all_ins),
        visual_type="distribution_grid",
    )


def _cont_calcs(srs: dd.Series, cfg: Config) -> Dict[str, Any]:
    """
    Computations for a continuous column in plot(df)
    """
    # dictionary of data for the histogram and related insights
    data: Dict[str, Any] = {}

    if cfg.insight.enable:
        data["npres"] = srs.shape[0]  # number of present (not null) values

    # drop infinite values
    srs = srs[~srs.isin({np.inf, -np.inf})]

    # histogram
    data["hist"] = da.histogram(srs, bins=cfg.hist.bins, range=(srs.min(), srs.max()))

    if cfg.insight.enable:
        data["chisq"] = chisquare(data["hist"][0])
        data["norm"] = normaltest(data["hist"][0])
        data["skew"] = skewtest(data["hist"][0])
        data["nneg"] = (srs < 0).sum()  # number of negative values
        data["nuniq"] = srs.nunique_approx()  # number of unique values
        data["nzero"] = (srs == 0).sum()  # number of zeros
        data["nreals"] = srs.shape[0]  # number of non-inf values
    return data


def _nom_calcs(srs: dd.Series, cfg: Config) -> Dict[str, Any]:
    """
    Computations for a nominal column in plot(df). Assume srs is string column.
    """
    # dictionary of data for the bar chart and related insights
    data: Dict[str, Any] = {}

    # value counts for barchart and uniformity insight
    grps = srs.value_counts(sort=False)

    if cfg.bar.enable:
        # select the largest or smallest groups
        data["bar"] = (
            grps.nlargest(cfg.bar.bars) if cfg.bar.sort_descending else grps.nsmallest(cfg.bar.bars)
        )
        data["nuniq"] = grps.shape[0]

    if cfg.insight.enable:
        data["chisq"] = chisquare(grps.values)  # chi-squared test for uniformity
        data["nuniq"] = grps.shape[0]  # number of unique values
        data["npres"] = grps.sum()  # number of present (not null) values
        data["min_len"] = srs.str.len().min()
        data["max_len"] = srs.str.len().max()

    return data


def _get_dtype_cnts_and_num_cols(
    frame: EDAFrame,
) -> Tuple[Dict[str, int], List[str]]:
    """
    Get the count of each dtype in a dataframe
    """

    dtype_cnts: DefaultDict[str, int] = defaultdict(int)
    num_cols: List[str] = []
    for col in frame.columns:
        col_dtype = frame.get_eda_dtype(col)
        if isinstance(col_dtype, (Nominal, SmallCardNum)):
            dtype_cnts["Categorical"] += 1
        elif isinstance(col_dtype, Continuous):
            dtype_cnts["Numerical"] += 1
            num_cols.append(col)
        elif isinstance(col_dtype, DateTime):
            dtype_cnts["DateTime"] += 1
        elif isinstance(col_dtype, GeoGraphy):
            dtype_cnts["GeoGraphy"] += 1
        elif isinstance(col_dtype, GeoPoint):
            dtype_cnts["GeoPoint"] += 1
        else:
            raise NotImplementedError(f"col:{col}, type:{col_dtype}")
    return dtype_cnts, num_cols


def calc_stats(frame: EDAFrame, cfg: Config) -> Dict[str, Any]:
    """
    Calculate the statistics for plot(df)
    """
    stats: Dict[str, Any] = {"nrows": frame.shape[0]}

    if cfg.stats.enable or cfg.insight.enable:
        dtype_cnts, num_cols = _get_dtype_cnts_and_num_cols(frame)

    df: dd.DataFrame = frame.frame
    if cfg.stats.enable:
        stats["ncols"] = df.shape[1]
        stats["npresent_cells"] = df.count().sum()
        stats["nrows_wo_dups"] = df.drop_duplicates().shape[0]
        stats["mem_use"] = df.memory_usage(deep=True).sum()
        stats["dtype_cnts"] = dtype_cnts

    if not cfg.stats.enable and cfg.insight.enable:
        stats["nrows_wo_dups"] = df.drop_duplicates().shape[0]

    if cfg.insight.enable:
        # compute distribution similarity on a data sample
        df_smp = df.map_partitions(lambda x: x.sample(min(1000, x.shape[0])), meta=df)
        stats["ks_tests"] = []
        for col1, col2 in combinations(num_cols, 2):
            stats["ks_tests"].append((col1, col2, ks_2samp(df_smp[col1], df_smp[col2])[1]))

    return stats


def _format_ov_ins(data: Dict[str, Any], cfg: Config) -> List[Dict[str, str]]:
    """
    Determine and format the overview insights for plot(df)
    """
    # pylint: disable=line-too-long
    # list of insights
    ins: List[Dict[str, str]] = []

    pdup = round((1 - data["nrows_wo_dups"] / data["nrows"]) * 100, 2)
    if pdup > cfg.insight.duplicates__threshold:
        ndup = data["nrows"] - data["nrows_wo_dups"]
        ins.append({"Duplicates": f"Dataset has {ndup} ({pdup}%) duplicate rows"})

    for *cols, test_result in data.get("ks_tests", []):
        if test_result > cfg.insight.similar_distribution__threshold:
            msg = f"/*start*/{cols[0]}/*end*/ and /*start*/{cols[1]}/*end*/ have similar distributions"
            ins.append({"Similar Distribution": msg})

    data.pop("ks_tests", None)

    return ins


def _format_cont_ins(col: str, data: Dict[str, Any], nrows: int, cfg: Config) -> Any:
    """
    Determine and format the insights for a continuous column
    """
    # list of insights
    ins: List[Dict[str, str]] = []

    if data["chisq"][1] > cfg.insight.uniform__threshold:
        ins.append({"Uniform": f"/*start*/{col}/*end*/ is uniformly distributed"})

    pmiss = round((1 - (data["npres"] / nrows)) * 100, 2)
    if pmiss > cfg.insight.missing__threshold:
        nmiss = nrows - data["npres"]
        ins.append({"Missing": f"/*start*/{col}/*end*/ has {nmiss} ({pmiss}%) missing values"})

    if data["skew"][1] < cfg.insight.skewed__threshold:
        ins.append({"Skewed": f"/*start*/{col}/*end*/ is skewed"})

    pinf = round((data["npres"] - data["nreals"]) / nrows * 100, 2)
    if pinf >= cfg.insight.infinity__threshold:
        ninf = data["npres"] - data["nreals"]
        ins.append({"Infinity": f"/*start*/{col}/*end*/ has {ninf} ({pinf}%) infinite values"})

    pzero = round(data["nzero"] / nrows * 100, 2)
    if pzero > cfg.insight.zeros__threshold:
        nzero = data["nzero"]
        ins.append({"Zeros": f"/*start*/{col}/*end*/ has {nzero} ({pzero}%) zeros"})

    pneg = round(data["nneg"] / nrows * 100, 2)
    if pneg > cfg.insight.negatives__threshold:
        nneg = data["nneg"]
        ins.append({"Negatives": f"/*start*/{col}/*end*/ has {nneg} ({pneg}%) negatives"})

    if data["norm"][1] > cfg.insight.normal__threshold:
        ins.append({"Normal": f"/*start*/{col}/*end*/ is normally distributed"})

    # list of insight messages
    ins_msg_list = [list(insight.values())[0] for insight in ins]

    return ins_msg_list, ins


def _format_nom_ins(col: str, data: Dict[str, Any], nrows: int, cfg: Config) -> Any:
    """
    Determine and format the insights for a nominal column
    """
    # list of insights
    ins: List[Dict[str, str]] = []

    if data["chisq"][1] > cfg.insight.uniform__threshold:
        ins.append({"Uniform": f"/*start*/{col}/*end*/ is uniformly distributed"})

    pmiss = round((1 - (data["npres"] / nrows)) * 100, 2)
    if pmiss > cfg.insight.missing__threshold:
        nmiss = nrows - data["npres"]
        ins.append({"Missing": f"/*start*/{col}/*end*/ has {nmiss} ({pmiss}%) missing values"})

    if data["nuniq"] > cfg.insight.high_cardinality__threshold:
        uniq = data["nuniq"]
        msg = f"/*start*/{col}/*end*/ has a high cardinality: {uniq} distinct values"
        ins.append({"High Cardinality": msg})

    if data["nuniq"] == cfg.insight.constant__threshold:
        val = data["bar"].index[0]
        ins.append({"Constant": f'/*start*/{col}/*end*/ has constant value "{val}"'})

    if data["min_len"] == data["max_len"]:
        length = data["min_len"]
        ins.append({"Constant Length": f"/*start*/{col}/*end*/ has constant length {length}"})

    if data["nuniq"] == data["npres"]:
        ins.append({"Unique": f"/*start*/{col}/*end*/ has all distinct values"})

    # list of insight messages
    ins_msg_list = [list(ins.values())[0] for ins in ins]

    return ins_msg_list, ins


def _insight_pagination(ins: List[Dict[str, str]]) -> Dict[int, List[Dict[str, str]]]:
    """
    Set the insight display order and paginate the insights
    """
    ins_order = [
        "Uniform",
        "Similar Distribution",
        "Missing",
        "Skewed",
        "Infinity",
        "Duplicates",
        "Normal",
        "High Cardinality",
        "Constant",
        "Constant Length",
        "Unique",
        "Negatives",
        "Zeros",
    ]
    # sort the insights based on the list ins_order
    ins.sort(key=lambda x: ins_order.index(list(x.keys())[0]))
    # paginate the sorted insights
    page_count = int(np.ceil(len(ins) / 10))
    paginated_ins: Dict[int, List[Dict[str, str]]] = dict()
    for i in range(1, page_count + 1):
        paginated_ins[i] = ins[(i - 1) * 10 : i * 10]

    return paginated_ins
