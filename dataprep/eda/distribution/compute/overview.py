"""Computations for plot(df) function."""

from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.array.stats import chisquare

from ....errors import UnreachableError
from ...dtypes import (
    Continuous,
    DateTime,
    DType,
    DTypeDef,
    Nominal,
    detect_dtype,
    get_dtype_cnts_and_num_cols,
    is_dtype,
)
from ...intermediate import Intermediate
from .common import _calc_line_dt, ks_2samp, normaltest, skewtest


def compute_overview(
    df: dd.DataFrame,
    bins: int,
    ngroups: int,
    largest: bool,
    timeunit: str,
    dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches

    """
    Compute functions for plot(df)
    Parameters
    ----------
    df
        Dataframe from which plots are to be generated
    bins
        For a histogram or box plot with numerical x axis, it defines
        the number of equal-width bins to use when grouping.
    ngroups
        When grouping over a categorical column, it defines the
        number of groups to show in the plot. Ie, the number of
        bars to show in a bar chart.
    largest
        If true, when grouping over a categorical column, the groups
        with the largest count will be output. If false, the groups
        with the smallest count will be output.
    timeunit
        Defines the time unit to group values over for a datetime column.
        It can be "year", "quarter", "month", "week", "day", "hour",
        "minute", "second". With default value "auto", it will use the
        time unit such that the resulting number of groups is closest to 15.
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    """
    # extract the first rows for checking if a column contains a mutable type
    first_rows: pd.DataFrame = df.head()  # dd.DataFrame.head triggers a (small) data read

    datas: List[Any] = []
    col_names_dtypes: List[Tuple[str, DType]] = []

    for col in df.columns:
        srs = df[col]
        col_dtype = detect_dtype(srs, dtype)
        if is_dtype(col_dtype, Nominal()):
            ## if cfg.barchart_enable or cfg.any_insights("barchart"):
            # cast the column as string type if it contains a mutable type
            try:
                first_rows[col].apply(hash)
            except TypeError:
                srs = df[col] = srs.astype(str)
            datas.append(calc_nom_col(srs.dropna(), first_rows[col], ngroups, largest))
            col_names_dtypes.append((col, Nominal()))
        elif is_dtype(col_dtype, Continuous()):
            ## if cfg.hist_enable or cfg.any_insights("hist"):
            datas.append(calc_cont_col(srs.dropna(), bins))
            col_names_dtypes.append((col, Continuous()))
        elif is_dtype(col_dtype, DateTime()):
            datas.append(dask.delayed(_calc_line_dt)(df[[col]], timeunit))
            col_names_dtypes.append((col, DateTime()))
        else:
            raise UnreachableError

    ov_stats = calc_stats(df, dtype)
    datas, ov_stats = dask.compute(datas, ov_stats)

    # extract the plotting data, and detect and format the insights
    plot_data: List[Any] = []
    col_insights: Dict[str, List[str]] = {}
    ov_insights = format_overview(ov_stats)
    nrows = ov_stats["nrows"]
    for (col, dtp), dat in zip(col_names_dtypes, datas):
        if is_dtype(dtp, Continuous()):
            hist, col_ins, ov_ins = format_cont(col, dat, nrows)
            if hist:
                plot_data.append((col, dtp, hist))
        elif is_dtype(dtp, Nominal()):
            bardata, col_ins, ov_ins = format_nom(col, dat, nrows)
            if bardata:
                plot_data.append((col, dtp, bardata))
        elif is_dtype(dtp, DateTime()):
            plot_data.append((col, dtp, dat))
            continue
        if col_ins:
            col_insights[col] = col_ins
        ov_insights += ov_ins

    return Intermediate(
        data=plot_data,
        stats=ov_stats,
        column_insights=col_insights,
        overview_insights=_insight_pagination(ov_insights),
        ov_insights=ov_insights,
        visual_type="distribution_grid",
    )


## def calc_cont_col(srs: dd.Series, cfg: Config)
def calc_cont_col(srs: dd.Series, bins: int) -> Dict[str, Any]:
    """
    Computations for a numerical column in plot(df)

    Parameters
    ----------
    srs
        srs over which to compute the barchart and insights
    bins
        number of bins in the bar chart
    """
    # dictionary of data for the histogram and related insights
    data: Dict[str, Any] = {}

    ## if cfg.insight.missing_enable:
    data["npres"] = srs.shape[0]

    ## if cfg.insight.infinity_enable:
    is_inf_srs = srs.isin({np.inf, -np.inf})
    data["ninf"] = is_inf_srs.sum()

    # remove infinite values
    srs = srs[~is_inf_srs]

    ## if cfg.hist_enable or config.insight.uniform_enable or cfg.insight.normal_enable:
    ## bins = cfg.hist_bins
    data["hist"] = da.histogram(srs, bins=bins, range=[srs.min(), srs.max()])

    ## if cfg.insight.uniform_enable:
    data["chisq"] = chisquare(data["hist"][0])

    ## if cfg.insight.normal_enable
    data["norm"] = normaltest(data["hist"][0])

    ## if cfg.insight.negative_enable:
    data["nneg"] = (srs < 0).sum()

    ## if cfg.insight.skew_enabled:
    data["skew"] = skewtest(data["hist"][0])

    ## if cfg.insight.unique_enabled:
    data["nuniq"] = srs.nunique()

    ## if cfg.insight.zero_enabled:
    data["nzero"] = (srs == 0).sum()

    return data


## def calc_nom_col(srs: dd.Series, first_rows: pd.Series, cfg: Config)
def calc_nom_col(
    srs: dd.Series, first_rows: pd.Series, ngroups: int, largest: bool
) -> Dict[str, Any]:
    """
    Computations for a categorical column in plot(df)

    Parameters
    ----------
    srs
        srs over which to compute the barchart and insights
    first_rows
        first rows of the dataset read into memory
    ngroups
        number of groups to show in the barchart
    largest
        whether to show the largest or smallest groups
    """
    # dictionary of data for the bar chart and related insights
    data = {}

    ## if cfg.barchart_enable or cfg.insight.uniform_enable:
    grps = srs.value_counts(sort=False)

    ##    if cfg.barchart_enable:
    ##       nbars = cfg.barchart_nbars
    ##       largest = cfg.barchart_largest
    # select the largest or smallest groups
    data["bar"] = grps.nlargest(ngroups) if largest else grps.nsmallest(ngroups)

    ##    if cfg.insight.uniform_enable:
    # compute a chi-squared test on the frequency distribution
    data["chisq"] = chisquare(grps.values)

    ##    if cfg.barchart_enable or cfg.insight.unique_enable:
    # total number of groups
    data["nuniq"] = grps.shape[0]

    ##    if cfg.insight.missing_enable:
    # number of present (not null) values
    data["npres"] = grps.sum()

    ## if cfg.insight.unique_enable and not cfg.barchart_enable:
    ## data["nuniq"] = srs.nunique()

    ## if cfg.insight.missing_enable and not cfg.barchart_enable:
    ## data["npresent"] = srs.shape[0]

    ## if cfg.insight.constant_length_enable:
    if not first_rows.apply(lambda x: isinstance(x, str)).all():
        srs = srs.astype(str)  # srs must be a string to compute the value lengths
    lengths = srs.str.len()
    data["min_len"], data["max_len"] = lengths.min(), lengths.max()

    return data


## def calc_stats(srs: dd.Series, dtype_cnts: Dict[str, int], num_cols: List[str], cfg: Config)
def calc_stats(df: dd.DataFrame, dtype: Optional[DTypeDef]) -> Dict[str, Any]:
    """
    Calculate the statistics for plot(df) from a DataFrame

    Parameters
    ----------
    df
        a DataFrame
    dtype
        str or DType or dict of str or dict of DType
    """

    stats = {"nrows": df.shape[0]}

    ## if cfg.stats_enable
    dtype_cnts, num_cols = get_dtype_cnts_and_num_cols(df, dtype)
    stats["ncols"] = df.shape[1]
    stats["npresent_cells"] = df.count().sum()
    stats["nrows_wo_dups"] = df.drop_duplicates().shape[0]
    stats["mem_use"] = df.memory_usage(deep=True).sum()
    stats["dtype_cnts"] = dtype_cnts

    ## if cfg.insight.duplicates_enable and not cfg.stats_enable
    ## stats["nrows_wo_dups"] = df.drop_duplicates().shape[0]

    ## if cfg.insight.similar_distribution_enable
    # compute distribution similarity on a data sample
    df_smp = df.map_partitions(lambda x: x.sample(min(1000, x.shape[0])), meta=df)
    stats["ks_tests"] = []
    for col1, col2 in list(combinations(num_cols, 2)):
        stats["ks_tests"].append((col1, col2, ks_2samp(df_smp[col1], df_smp[col2])[1] > 0.05))

    return stats


## def format_overview(data: Dict[str, Any], cfg: Config)
def format_overview(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Determine and format the overview statistics and insights from plot(df)

    Parameters
    ----------
    data
        dictionary with overview statistics
    """
    # list of insights
    ins: List[Dict[str, str]] = []

    ## if cfg.insight.duplicates_enable
    pdup = round((1 - data["nrows_wo_dups"] / data["nrows"]) * 100, 2)
    if pdup > 1:  ## if cfg.insight.duplicates_threshold
        ndup = data["nrows"] - data["nrows_wo_dups"]
        ins.append({"Duplicates": f"Dataset has {ndup} ({pdup}%) duplicate rows"})

    ## if cfg.insight.similar_distribution_enable
    for (*cols, test_result) in data.get("ks_tests", []):
        if test_result:
            msg = f"{cols[0]} and {cols[1]} have similar distributions"
            ins.append({"Similar Distribution": msg})

    data.pop("ks_tests", None)

    return ins


## def format_cont(col: str, data: Dict[str, Any], nrows: int, cfg: Config)
def format_cont(col: str, data: Dict[str, Any], nrows: int) -> Any:
    """
    Determine and format the insights for a numerical column

    Parameters
    ----------
    col
        the column associated with the insights
    data
        dictionary with overview statistics
    nrows
        number of rows in the dataset
    """
    # list of insights
    ins: List[Dict[str, str]] = []

    ## if cfg.insight.uniform_enable:
    if data["chisq"][1] > 0.999:  ## cfg.insight.uniform_threshold
        ins.append({"Uniform": f"{col} is uniformly distributed"})

    ## if cfg.insight.missing_enable:
    pmiss = round((1 - (data["npres"] / nrows)) * 100, 2)
    if pmiss > 1:  ## cfg.insight.missing_threshold
        nmiss = nrows - data["npres"]
        ins.append({"Missing": f"{col} has {nmiss} ({pmiss}%) missing values"})

    ## if cfg.insight.skewed_enable:
    if data["skew"][1] < 1e-5:  ## cfg.insight.skewed_threshold
        ins.append({"Skewed": f"{col} is skewed"})

    ## if cfg.insight.infinity_enable:
    pinf = round(data["ninf"] / nrows * 100, 2)
    if pinf >= 1:  ## cfg.insight.infinity_threshold
        ninf = data["ninf"]
        ins.append({"Infinity": f"{col} has {ninf} ({pinf}%) infinite values"})

    ## if cfg.insight.zeros_enable:
    pzero = round(data["nzero"] / nrows * 100, 2)
    if pzero > 5:  ## cfg.insight.zeros_threshold
        nzero = data["nzero"]
        ins.append({"Zeros": f"{col} has {nzero} ({pzero}%) zeros"})

    ## if cfg.insight.negatives_enable:
    pneg = round(data["nneg"] / nrows * 100, 2)
    if pneg > 1:  ## cfg.insight.negatives_threshold
        nneg = data["nneg"]
        ins.append({"Negatives": f"{col} has {nneg} ({pneg}%) negatives"})

    ## if cfg.insight.normal_enable:
    if data["norm"][1] > 0.99:
        ins.append({"Normal": f"{col} is normally distributed"})

    hist = data["hist"]  ## if cfg.hist_enable else None
    # list of insight messages
    ins_msg_list = [list(insight.values())[0] for insight in ins]

    return hist, ins_msg_list, ins


## def format_nom(col: str, data: Dict[str, Any], nrows: int, cfg: Config)
def format_nom(col: str, data: Dict[str, Any], nrows: int) -> Any:
    """
    Determine and format the insights for a categorical column

    Parameters
    ----------
    col
        the column associated with the insights
    data
        dictionary with overview statistics
    nrows
        number of rows in the dataset
    """
    # list of insights
    ins: List[Dict[str, str]] = []

    ## if cfg.insight.uniform_enable:
    if data["chisq"][1] > 0.999:  ## cfg.insight.uniform_threshold
        ins.append({"Uniform": f"{col} is uniformly distributed"})

    ## if cfg.insight.missing_enable:
    pmiss = round((1 - (data["npres"] / nrows)) * 100, 2)
    if pmiss > 1:  ## cfg.insight.missing_threshold
        nmiss = nrows - data["npres"]
        ins.append({"Missing": f"{col} has {nmiss} ({pmiss}%) missing values"})

    ## if cfg.insight.high_cardinality_enable:
    if data["nuniq"] > 50:  ## cfg.insght.high_cardinality_threshold
        uniq = data["nuniq"]
        msg = f"{col} has a high cardinality: {uniq} distinct values"
        ins.append({"High Cardinality": msg})

    ## if cfg.insight.constant_enable:
    if data["nuniq"] == 1:
        val = data["bar"].index[0]
        ins.append({"Constant": f'{col} has constant value "{val}"'})

    ## if cfg.insight.constant_length_enable:
    if data["min_len"] == data["max_len"]:
        length = data["min_len"]
        ins.append({"Constant Length": f"{col} has constant length {length}"})

    ## if cfg.insight.constant_length_enable:
    if data["nuniq"] == data["npres"]:
        ins.append({"Unique": f"{col} has all distinct values"})

    bardata = (
        data["bar"].to_frame(),
        data["nuniq"],
    )  ## if cfg.barchart.enable else None
    # list of insight messages
    ins_msg_list = [list(ins.values())[0] for ins in ins]

    return bardata, ins_msg_list, ins


def _insight_pagination(ins: List[Dict[str, str]]) -> Dict[int, List[Dict[str, str]]]:
    """
    Set the insight display order and paginate the insights
    Parameters
    ----------
    olddc
        a dict contains all insights for overview section
    Returns
    -------
    Dict[int, List[Dict[str, str]]]
        paginated dict
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
    paginated_ins: Dict[int, List[Dict[str, str]]] = {}
    for i in range(1, page_count + 1):
        paginated_ins[i] = ins[(i - 1) * 10 : i * 10]

    return paginated_ins
