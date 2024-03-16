"""Miscellaneous functions
"""

import logging
from math import ceil
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from collections import Counter


import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pandas._libs.missing as libmissing
from bokeh.models import Legend, FuncTickFormatter
from bokeh.plotting import Figure
from scipy.stats import gaussian_kde as gaussian_kde_
from scipy.stats import ks_2samp as ks_2samp_
from scipy.stats import normaltest as normaltest_
from scipy.stats import skewtest as skewtest_
from .dtypes import (
    drop_null,
    Nominal,
    detect_dtype,
    is_dtype,
)


LOGGER = logging.getLogger(__name__)


def to_dask(df: Union[pd.DataFrame, dd.DataFrame]) -> dd.DataFrame:
    """Convert a dataframe to a dask dataframe."""
    if isinstance(df, dd.DataFrame):
        return df
    elif isinstance(df, dd.Series):
        return df.to_frame()

    if isinstance(df, pd.Series):
        df = df.to_frame()

    df_size = df.memory_usage(deep=True).sum()
    npartitions = ceil(df_size / 128 / 1024 / 1024)  # 128 MB partition size
    return dd.from_pandas(df, npartitions=npartitions)


def preprocess_dataframe(
    org_df: Union[pd.DataFrame, dd.DataFrame],
    used_columns: Optional[Union[List[str], List[object]]] = None,
    excluded_columns: Optional[Union[List[str], List[object]]] = None,
    detect_small_distinct: bool = True,
) -> dd.DataFrame:
    """
    Make a dask dataframe with only used_columns.
    This function will do the following:
        1. keep only used_columns.
        2. transform column name to string (avoid object column name) and rename
        duplicate column names in form of {col}_{id}.
        3. reset index
        4. transform object column to string column (note that obj column can contain
        cells from different type).
        5. transform to dask dataframe if input is pandas dataframe.
    Parameters
    ----------------
    org_df: dataframe
        the original dataframe
    used_columns: optional list[str], default None
        used columns in org_df
    excluded_columns: optional list[str], default None
        excluded columns from used_columns, mainly used for geo point data processing.
    detect_small_distinct: bool, default True
        whether to detect numerical columns with small distinct values as categorical column.
    """
    if used_columns is None:
        df = org_df.copy()
    else:
        # Process the case when used_columns are string column name,
        # but org_df column name is object.
        used_columns_set = set(used_columns)
        used_cols_obj = set()
        for col in org_df.columns:
            if str(col) in used_columns_set or col in used_columns_set:
                used_cols_obj.add(col)
        df = org_df[used_cols_obj]

    columns = list(df.columns)

    # Resolve duplicate names in columns.
    # Duplicate names will be renamed as col_{id}.
    column_count = Counter(columns)
    current_id: Dict[Any, int] = {}
    for i, col in enumerate(columns):
        if column_count[col] > 1:
            current_id[col] = current_id.get(col, 0) + 1
            new_col_name = f"{col}_{current_id[col]}"
        else:
            new_col_name = f"{col}"
        columns[i] = new_col_name

    df.columns = columns
    df = df.reset_index(drop=True)
    df = to_dask(df)

    # Since an object column could contains multiple types
    # in different cells. transform non-na values in object column to string.

    # Function `_notna2str` transforms an obj to str if it is not NA.
    # The check for NA is similar to pd.isna, but will treat a list obj as
    # a scalar and return a single boolean, rather than a list of booleans.
    # Otherwise when a cell is tuple or list it will throw an error.
    _notna2str = lambda obj: obj if libmissing.checknull(obj) else str(obj)
    for col in df.columns:
        col_dtype = detect_dtype(df[col], detect_small_distinct=detect_small_distinct)
        if (is_dtype(col_dtype, Nominal())) and (
            (excluded_columns is None) or (col not in excluded_columns)
        ):
            df[col] = df[col].apply(_notna2str, meta=(col, "object"))
    return df


def sample_n(arr: np.ndarray, n: int) -> np.ndarray:  # pylint: disable=C0103
    """Sample n values uniformly from the range of the `arr`,
    not from the distribution of `arr`'s elems."""

    if len(arr) <= n:
        return arr

    subsel = np.linspace(0, len(arr) - 1, n)
    subsel = np.floor(subsel).astype(int)
    return arr[subsel]


def relocate_legend(fig: Figure, loc: str) -> Figure:
    """Relocate legend(s) from center to `loc`."""
    remains = []
    targets = []
    for layout in fig.center:
        if isinstance(layout, Legend):
            targets.append(layout)
        else:
            remains.append(layout)
    fig.center = remains
    for layout in targets:
        fig.add_layout(layout, loc)

    return fig


def cut_long_name(name: str, max_len: int = 18) -> str:
    """If the name is longer than `max_len`,
    cut it to `max_len` length and append "..."""

    # Bug 136 Fixed
    name = str(name)
    cut_name = f"{name[:13]}...{name[len(name)-3:]}" if len(name) > max_len else name
    return cut_name


def fuse_missing_perc(name: str, perc: float) -> str:
    """Append (x.y%) to the name if `perc` is not 0."""
    if perc == 0:
        return name

    return f"{name} ({perc:.1%})"


# Dictionary for mapping the time unit to its formatting. Each entry is of the
# form unit:(unit code for pd.Grouper freq parameter, pandas to_period strftime
# formatting for line charts, pandas to_period strftime formatting for box plot,
# label format).
DTMAP = {
    "year": ("Y", "%Y", "%Y", "Year"),
    "quarter": ("Q", "Q%q %Y", "Q%q %Y", "Quarter"),
    "month": ("M", "%B %Y", "%b %Y", "Month"),
    "week": ("W-SAT", "%d %B, %Y", "%d %b, %Y", "Week of"),
    "day": ("D", "%d %B, %Y", "%d %b, %Y", "Date"),
    "hour": ("H", "%d %B, %Y, %I %p", "%d %b, %Y, %I %p", "Hour"),
    "minute": ("T", "%d %B, %Y, %I:%M %p", "%d %b, %Y, %I:%M %p", "Minute"),
    "second": ("S", "%d %B, %Y, %I:%M:%S %p", "%d %b, %Y, %I:%M:%S %p", "Second"),
}


def _get_timeunit(min_time: pd.Timestamp, max_time: pd.Timestamp, dflt: int) -> str:
    """Auxillary function to find an appropriate time unit. Will find the
    time unit such that the number of time units are closest to dflt."""

    dt_secs = {
        "year": 60 * 60 * 24 * 365,
        "quarter": 60 * 60 * 24 * 91,
        "month": 60 * 60 * 24 * 30,
        "week": 60 * 60 * 24 * 7,
        "day": 60 * 60 * 24,
        "hour": 60 * 60,
        "minute": 60,
        "second": 1,
    }

    time_rng_secs = (max_time - min_time).total_seconds()
    prev_bin_cnt, prev_unit = 0, "year"
    for unit, secs_in_unit in dt_secs.items():
        cur_bin_cnt = time_rng_secs / secs_in_unit
        if abs(prev_bin_cnt - dflt) < abs(cur_bin_cnt - dflt):
            return prev_unit
        prev_bin_cnt = cur_bin_cnt
        prev_unit = unit

    return prev_unit


def _calc_box_stats(grp_srs: dd.Series, grp: str, dlyd: bool = False) -> pd.DataFrame:
    """
    Auxiliary function to calculate the Tukey box plot statistics
    dlyd is for if this function is called when dask is computing in parallel (dask.delayed)
    """
    stats: Dict[str, Any] = dict()

    try:  # this is a bad fix for the problem of when there is no data passed to this function
        if dlyd:
            qntls = np.round(grp_srs.quantile([0.25, 0.50, 0.75]), 3)
        else:
            qntls = np.round(grp_srs.quantile([0.25, 0.50, 0.75]).compute(), 3)
        stats["q1"], stats["q2"], stats["q3"] = qntls[0.25], qntls[0.50], qntls[0.75]
    except ValueError:
        stats["q1"], stats["q2"], stats["q3"] = np.nan, np.nan, np.nan

    iqr = stats["q3"] - stats["q1"]
    stats["lw"] = grp_srs[grp_srs >= stats["q1"] - 1.5 * iqr].min()
    stats["uw"] = grp_srs[grp_srs <= stats["q3"] + 1.5 * iqr].max()
    if not dlyd:
        stats["lw"], stats["uw"] = dask.compute(stats["lw"], stats["uw"])

    otlrs = grp_srs[(grp_srs < stats["lw"]) | (grp_srs > stats["uw"])]
    if len(otlrs) > 100:  # sample 100 outliers
        otlrs = otlrs.sample(frac=100 / len(otlrs))
    stats["otlrs"] = list(otlrs) if dlyd else list(otlrs.compute())

    return pd.DataFrame({grp: stats})


def _calc_box_otlrs(df: dd.DataFrame) -> Tuple[List[str], List[float]]:
    """
    Calculate the outliers for a box plot
    """
    outx: List[str] = []  # list for the outlier groups
    outy: List[float] = []  # list for the outlier values
    for ind in df.index:
        otlrs = df.loc[ind]["otlrs"]
        outx = outx + [df.loc[ind]["grp"]] * len(otlrs)
        outy = outy + otlrs

    return outx, outy


def _calc_line_dt(
    df: dd.DataFrame,
    unit: str,
    agg: Optional[str] = None,
    ngroups: Optional[int] = None,
    largest: Optional[bool] = None,
) -> Union[
    Tuple[pd.DataFrame, Dict[str, int], str],
    Tuple[pd.DataFrame, str, float],
    Tuple[pd.DataFrame, str],
]:
    """
    Calculate a line or multiline chart with date on the x axis. If df contains
    one datetime column, it will make a line chart of the frequency of values. If
    df contains a datetime and categorical column, it will compute the frequency
    of each categorical value in each time group. If df contains a datetime and
    numerical column, it will compute the aggregate of the numerical column grouped
    by the time groups. If df contains a datetime, categorical, and numerical column,
    it will compute the aggregate of the numerical column for values in the categorical
    column grouped by time.
    Parameters
    ----------
    df
        A dataframe
    unit
        The unit of time over which to group the values
    agg
        Aggregate to use for the numerical column
    ngroups
        Number of groups for the categorical column
    largest
        Use the largest or smallest groups in the categorical column
    """
    # pylint: disable=too-many-locals

    x = df.columns[0]  # time column
    unit = _get_timeunit(df[x].min(), df[x].max(), 100) if unit == "auto" else unit
    if unit not in DTMAP.keys():
        raise ValueError
    grouper = pd.Grouper(key=x, freq=DTMAP[unit][0])  # for grouping the time values

    # multiline charts
    if ngroups and largest:
        hist_dict: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = dict()
        hist_lst: List[Tuple[np.ndarray, np.ndarray, List[str]]] = list()
        agg = "freq" if agg is None else agg  # default agg if unspecified for notational concision

        # categorical column for grouping over, each resulting group is a line in the chart
        grpby_col = df.columns[1] if len(df.columns) == 2 else df.columns[2]
        df, grp_cnt_stats, largest_grps = _calc_groups(df, grpby_col, ngroups, largest)
        groups = df.groupby([grpby_col])

        for grp in largest_grps:
            srs = groups.get_group(grp)
            # calculate the frequencies or aggregate value in each time group
            if len(df.columns) == 3:
                dfr = srs.groupby(grouper)[df.columns[1]].agg(agg).reset_index()
            else:
                dfr = srs[x].to_frame().groupby(grouper).size().reset_index()
            dfr.columns = [x, agg]
            # if grouping by week, make the label for the week the beginning Sunday
            dfr[x] = dfr[x] - pd.to_timedelta(6, unit="d") if unit == "week" else dfr[x]
            # format the label
            dfr["lbl"] = dfr[x].dt.to_period("S").dt.strftime(DTMAP[unit][1])
            hist_lst.append((list(dfr[agg]), list(dfr[x]), list(dfr["lbl"])))
        hist_lst = dask.compute(*hist_lst)
        for elem in zip(largest_grps, hist_lst):
            hist_dict[elem[0]] = elem[1]
        return hist_dict, grp_cnt_stats, DTMAP[unit][3]

    # single line charts
    if agg is None:  # frequency of datetime column
        miss_pct = round(df[x].isna().sum() / len(df) * 100, 1)
        dfr = drop_null(df).groupby(grouper).size().reset_index()
        dfr.columns = [x, "freq"]
        dfr["pct"] = dfr["freq"] / len(df) * 100
    else:  # aggregate over a second column
        dfr = df.groupby(grouper)[df.columns[1]].agg(agg).reset_index()
        dfr.columns = [x, agg]
    dfr[x] = dfr[x] - pd.to_timedelta(6, unit="d") if unit == "week" else dfr[x]
    dfr["lbl"] = dfr[x].dt.to_period("S").dt.strftime(DTMAP[unit][1])

    return (dfr, DTMAP[unit][3], miss_pct) if agg is None else (dfr, DTMAP[unit][3])


def _calc_running_total_dt(
    df: dd.DataFrame,
    unit: str,
) -> Union[Tuple[pd.DataFrame, str],]:
    """
    Calculate a running total line for a df two columns: a datetime column and numerical column.
    Parameters
    ----------
    df
        A dataframe
    unit
        The unit of time over which to group the values
    """
    res_df, time_prefix = _calc_line_dt(df, unit, agg="sum")
    res_df["sum"] = res_df["sum"].cumsum()
    res_df.rename(columns={"sum": "runningtotal"}, inplace=True)
    return res_df, time_prefix


def _calc_groups(
    df: dd.DataFrame, x: str, ngroups: int, largest: bool = True
) -> Tuple[dd.DataFrame, Dict[str, int], List[str]]:
    """Auxillary function to parse the dataframe to consist of only the
    groups with the largest counts.
    """

    # group count statistics to inform the user of the sampled output
    grp_cnt_stats: Dict[str, int] = dict()

    srs = df.groupby(x).size()
    srs_lrgst = srs.nlargest(n=ngroups) if largest else srs.nsmallest(n=ngroups)
    try:
        largest_grps = list(srs_lrgst.index.compute())
        grp_cnt_stats[f"{x}_ttl"] = len(srs.index.compute())
    except AttributeError:
        largest_grps = list(srs_lrgst.index)
        grp_cnt_stats[f"{x}_ttl"] = len(srs.index)

    df = df[df[x].isin(largest_grps)]
    grp_cnt_stats[f"{x}_shw"] = len(largest_grps)

    return df, grp_cnt_stats, largest_grps


@dask.delayed(name="scipy-normaltest", pure=True, nout=2)  # pylint: disable=no-value-for-parameter
def normaltest(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Delayed version of scipy normaltest. Due to the dask version will
    trigger a compute."""
    return cast(Tuple[np.ndarray, np.ndarray], normaltest_(arr))


@dask.delayed(name="scipy-ks_2samp", pure=True, nout=2)  # pylint: disable=no-value-for-parameter
def ks_2samp(data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
    """Delayed version of scipy ks_2samp."""
    return cast(Tuple[float, float], ks_2samp_(data1, data2))


@dask.delayed(  # pylint: disable=no-value-for-parameter
    name="scipy-gaussian_kde", pure=True, nout=2
)
def gaussian_kde(arr: np.ndarray) -> Tuple[float, float]:
    """Delayed version of scipy gaussian_kde."""
    return cast(Tuple[np.ndarray, np.ndarray], gaussian_kde_(arr))


@dask.delayed(name="scipy-skewtest", pure=True, nout=2)  # pylint: disable=no-value-for-parameter
def skewtest(arr: np.ndarray) -> Tuple[float, float]:
    """Delayed version of scipy skewtest."""
    return cast(Tuple[float, float], skewtest_(arr))


def tweak_figure(
    fig: Figure,
    ptype: Optional[str] = None,
    show_yticks: bool = False,
    max_lbl_len: int = 15,
) -> None:
    """
    Set some common attributes for a figure
    """
    fig.axis.major_label_text_font_size = "9pt"
    fig.title.text_font_size = "10pt"
    fig.axis.minor_tick_line_color = "white"
    if ptype in ["pie", "qq", "heatmap"]:
        fig.ygrid.grid_line_color = None
    if ptype in ["bar", "pie", "hist", "kde", "qq", "heatmap", "line"]:
        fig.xgrid.grid_line_color = None
    if ptype in ["bar", "hist", "line"] and not show_yticks:
        fig.ygrid.grid_line_color = None
        fig.yaxis.major_label_text_font_size = "0pt"
        fig.yaxis.major_tick_line_color = None
    if ptype in ["bar", "nested", "stacked", "heatmap", "box"]:
        fig.xaxis.major_label_orientation = np.pi / 3
        fig.xaxis.formatter = FuncTickFormatter(
            code="""
            if (tick.length > %d) return tick.substring(0, %d-2) + '...';
            else return tick;
        """
            % (max_lbl_len, max_lbl_len)
        )
    if ptype in ["nested", "stacked", "box"]:
        fig.xgrid.grid_line_color = None
    if ptype in ["nested", "stacked"]:
        fig.y_range.start = 0
        fig.x_range.range_padding = 0.03
    if ptype in ["line", "boxnum"]:
        fig.min_border_right = 20
        fig.xaxis.major_label_standoff = 7
        fig.xaxis.major_label_orientation = 0
        fig.xaxis.major_tick_line_color = None


def _format_ticks(ticks: List[float]) -> List[str]:
    """
    Format the tick values
    """
    formatted_ticks = []
    for tick in ticks:  # format the tick values
        before, after = f"{tick:e}".split("e")
        if float(after) > 1e15 or abs(tick) < 1e4:
            formatted_ticks.append(str(tick))
            continue
        mod_exp = int(after) % 3
        factor = 1 if mod_exp == 0 else 10 if mod_exp == 1 else 100
        value = np.round(float(before) * factor, len(str(before)))
        value = int(value) if value.is_integer() else value
        if abs(tick) >= 1e12:
            formatted_ticks.append(str(value) + "T")
        elif abs(tick) >= 1e9:
            formatted_ticks.append(str(value) + "B")
        elif abs(tick) >= 1e6:
            formatted_ticks.append(str(value) + "M")
        elif abs(tick) >= 1e4:
            formatted_ticks.append(str(value) + "K")

    return formatted_ticks


def _format_axis(fig: Figure, minv: int, maxv: int, axis: str) -> None:
    """
    Format the axis ticks
    """  # pylint: disable=too-many-locals

    # divisor for 5 ticks (5 results in ticks that are too close together)
    divisor = 4.5
    # interval
    if np.isinf(minv) or np.isinf(maxv):
        gap = 1.0
    else:
        gap = (maxv - minv) / divisor
    # get exponent from scientific notation
    _, after = f"{gap:.0e}".split("e")
    # round to this amount
    round_to = -1 * int(after)
    # round the first x tick
    minv = np.round(minv, round_to)
    # round value between ticks
    gap = np.round(gap, round_to)

    # make the tick values
    ticks = [float(minv)]
    if not np.isinf(maxv):
        while max(ticks) + gap < maxv:
            ticks.append(max(ticks) + gap)
    ticks = np.round(ticks, round_to)
    ticks = [int(tick) if tick.is_integer() else tick for tick in ticks]
    formatted_ticks = _format_ticks(ticks)

    if axis == "x":
        fig.xgrid.ticker = ticks
        fig.xaxis.ticker = ticks
        fig.xaxis.major_label_overrides = dict(zip(ticks, formatted_ticks))
        fig.xaxis.major_label_text_font_size = "10pt"
        fig.xaxis.major_label_standoff = 7
        # fig.xaxis.major_label_orientation = 0
        fig.xaxis.major_tick_line_color = None
    elif axis == "y":
        fig.ygrid.ticker = ticks
        fig.yaxis.ticker = ticks
        fig.yaxis.major_label_overrides = dict(zip(ticks, formatted_ticks))
        fig.yaxis.major_label_text_font_size = "10pt"
        fig.yaxis.major_label_standoff = 5


def _format_bin_intervals(bins_arr: np.ndarray) -> List[str]:
    """
    Auxillary function to format bin intervals in a histogram
    """
    bins_arr = np.round(bins_arr, 3)
    bins_arr = [int(val) if float(val).is_integer() else val for val in bins_arr]
    intervals = [f"[{bins_arr[i]}, {bins_arr[i + 1]})" for i in range(len(bins_arr) - 2)]
    intervals.append(f"[{bins_arr[-2]},{bins_arr[-1]}]")
    return intervals
