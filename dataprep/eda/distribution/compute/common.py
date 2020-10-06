"""Common types and functionalities for compute(...)."""

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde as gaussian_kde_
from scipy.stats import ks_2samp as ks_2samp_
from scipy.stats import normaltest as normaltest_
from scipy.stats import skewtest as skewtest_

from ...dtypes import drop_null

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
