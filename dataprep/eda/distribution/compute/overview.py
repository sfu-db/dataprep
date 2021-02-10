"""Computations for plot(df)"""

from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.array.stats import chisquare

from ...configs import Config
from ...dtypes import (
    Continuous,
    DateTime,
    DType,
    DTypeDef,
    Nominal,
    GeoGraphy,
    detect_dtype,
    get_dtype_cnts_and_num_cols,
    is_dtype,
)
from ...utils import _calc_line_dt, ks_2samp, normaltest, skewtest
from ...intermediate import Intermediate


def compute_overview(df: dd.DataFrame, cfg: Config, dtype: Optional[DTypeDef]) -> Intermediate:
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

    if cfg.bar.enable or cfg.insight.enable:
        # extract the first rows to check if a column contains a mutable type
        head: pd.DataFrame = df.head()  # head triggers a (small) data read

    data: List[Tuple[str, DType, Any]] = []
    for col in df.columns:
        col_dtype = detect_dtype(df[col], dtype)
        if is_dtype(col_dtype, Continuous()) and (cfg.hist.enable or cfg.insight.enable):
            data.append((col, Continuous(), _cont_calcs(df[col].dropna(), cfg)))
        elif is_dtype(col_dtype, Nominal()) and (cfg.bar.enable or cfg.insight.enable):
            # Since it will throw error if column is object while some cells are
            # numerical, we transform column to string first.
            df[col] = df[col].astype(str)
            data.append((col, Nominal(), _nom_calcs(df[col].dropna(), head[col], cfg)))
        elif is_dtype(col_dtype, GeoGraphy()) and (cfg.bar.enable or cfg.insight.enable):
            # cast the column as string type if it contains a mutable type
            try:
                head[col].apply(hash)
            except TypeError:
                df[col] = df[col].astype(str)
            data.append((col, GeoGraphy(), _nom_calcs(df[col].dropna(), head[col], cfg)))
        elif is_dtype(col_dtype, DateTime()) and (cfg.line.enable or cfg.insight.enable):
            data.append((col, DateTime(), dask.delayed(_calc_line_dt)(df[[col]], cfg.line.unit)))

    ov_stats = calc_stats(df, cfg, dtype)  # overview statistics
    data, ov_stats = dask.compute(data, ov_stats)

    # extract the plotting data, and detect and format the insights
    plot_data: List[Tuple[str, DType, Any]] = []
    col_insights: Dict[str, List[str]] = {}
    all_ins = _format_ov_ins(ov_stats, cfg) if cfg.insight.enable else []

    for col, dtp, dat in data:
        if is_dtype(dtp, Continuous()):
            if cfg.insight.enable:
                col_ins, ins = _format_cont_ins(col, dat, ov_stats["nrows"], cfg)
            if cfg.hist.enable:
                plot_data.append((col, dtp, dat["hist"]))
        elif is_dtype(dtp, Nominal()) or is_dtype(dtp, GeoGraphy()):
            if cfg.insight.enable:
                col_ins, ins = _format_nom_ins(col, dat, ov_stats["nrows"], cfg)
            if cfg.bar.enable:
                plot_data.append((col, dtp, (dat["bar"].to_frame(), dat["nuniq"])))
        elif is_dtype(dtp, DateTime()):
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

    return data


def _nom_calcs(srs: dd.Series, head: pd.Series, cfg: Config) -> Dict[str, Any]:
    """
    Computations for a nominal column in plot(df)
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
        if not head.apply(lambda x: isinstance(x, str)).all():
            srs = srs.astype(str)  # srs must be a string to compute the value lengths
        data["min_len"], data["max_len"] = srs.str.len().min(), srs.str.len().max()

    return data


def calc_stats(df: dd.DataFrame, cfg: Config, dtype: Optional[DTypeDef]) -> Dict[str, Any]:
    """
    Calculate the statistics for plot(df)
    """
    stats = {"nrows": df.shape[0]}

    if cfg.stats.enable or cfg.insight.enable:
        dtype_cnts, num_cols = get_dtype_cnts_and_num_cols(df, dtype)

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
    # list of insights
    ins: List[Dict[str, str]] = []

    pdup = round((1 - data["nrows_wo_dups"] / data["nrows"]) * 100, 2)
    if pdup > cfg.insight.duplicates__threshold:
        ndup = data["nrows"] - data["nrows_wo_dups"]
        ins.append({"Duplicates": f"Dataset has {ndup} ({pdup}%) duplicate rows"})

    for (*cols, test_result) in data.get("ks_tests", []):
        if test_result > cfg.insight.similar_distribution__threshold:
            msg = f"/*{cols[0]}*/ and /*{cols[1]}*/ have similar distributions"
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
        ins.append({"Uniform": f"/*{col}*/ is uniformly distributed"})

    pmiss = round((1 - (data["npres"] / nrows)) * 100, 2)
    if pmiss > cfg.insight.missing__threshold:
        nmiss = nrows - data["npres"]
        ins.append({"Missing": f"/*{col}*/ has {nmiss} ({pmiss}%) missing values"})

    if data["skew"][1] < cfg.insight.skewed__threshold:
        ins.append({"Skewed": f"/*{col}*/ is skewed"})

    pinf = round((nrows - data["npres"]) / nrows * 100, 2)
    if pinf >= cfg.insight.infinity__threshold:
        ninf = nrows - data["npres"]
        ins.append({"Infinity": f"/*{col}*/ has {ninf} ({pinf}%) infinite values"})

    pzero = round(data["nzero"] / nrows * 100, 2)
    if pzero > cfg.insight.zeros__threshold:
        nzero = data["nzero"]
        ins.append({"Zeros": f"/*{col}*/ has {nzero} ({pzero}%) zeros"})

    pneg = round(data["nneg"] / nrows * 100, 2)
    if pneg > cfg.insight.negatives__threshold:
        nneg = data["nneg"]
        ins.append({"Negatives": f"/*{col}*/ has {nneg} ({pneg}%) negatives"})

    if data["norm"][1] > cfg.insight.normal__threshold:
        ins.append({"Normal": f"/*{col}*/ is normally distributed"})

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
        ins.append({"Uniform": f"/*{col}*/ is uniformly distributed"})

    pmiss = round((1 - (data["npres"] / nrows)) * 100, 2)
    if pmiss > cfg.insight.missing__threshold:
        nmiss = nrows - data["npres"]
        ins.append({"Missing": f"/*{col}*/ has {nmiss} ({pmiss}%) missing values"})

    if data["nuniq"] > cfg.insight.high_cardinality__threshold:
        uniq = data["nuniq"]
        msg = f"/*{col}*/ has a high cardinality: {uniq} distinct values"
        ins.append({"High Cardinality": msg})

    if data["nuniq"] == cfg.insight.constant__threshold:
        val = data["bar"].index[0]
        ins.append({"Constant": f'/*{col}*/ has constant value "{val}"'})

    if data["min_len"] == data["max_len"]:
        length = data["min_len"]
        ins.append({"Constant Length": f"/*{col}*/ has constant length {length}"})

    if data["nuniq"] == data["npres"]:
        ins.append({"Unique": f"/*{col}*/ has all distinct values"})

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
