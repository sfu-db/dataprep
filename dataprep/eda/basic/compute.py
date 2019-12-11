"""
    This module implements the plot(df) function.
"""
from sys import stderr
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm

from ...utils import to_dask
from ..common import Intermediate
from ..dtypes import is_categorical, is_numerical, DType
from ...errors import UnreachableError

__all__ = ["compute"]


def compute(
    df: Union[pd.DataFrame, dd.DataFrame],
    x: Optional[str] = None,
    y: Optional[str] = None,
    *,
    value_range: Optional[Tuple[float, float]] = None,
    bars: int = 10,
    bins: int = 10,
    bandwidth: int = 1.5,
    reverse: bool = True,
) -> Intermediate:
    # pylint: disable=too-many-arguments

    df = to_dask(df)
    orig_df_len = len(df)

    if x is None and y is None:
        datas: List[Tuple[str, DType, Any]] = []
        for col in df.columns:
            dtype = df[col].dtype

            if is_categorical(dtype):
                data = calc_bar_pie(df, col, bars, reverse)
                datas.append((col, DType.Categorical, data))
            elif is_numerical(dtype):
                data = calc_hist(df, col, orig_df_len, bins)
                datas.append((col, DType.Numerical, data))
            else:
                raise UnreachableError

        return Intermediate(data=datas, visual_type="basic_grid")
    elif (x is None) != (y is None):
        target_col: str = cast(str, x or y)
        if is_categorical(df[target_col].dtype):
            # data for bar and pie charts
            data = calc_bar_pie(df, target_col, bars, reverse)
            return Intermediate(
                col=target_col, data=data, visual_type="categorical_column",
            )

        elif is_numerical(df[target_col].dtype):
            parse_dataframe(df, target_col, value_range)
            # histogram
            hist_dict = calc_hist(df, target_col, orig_df_len, bins)
            # kde plot
            kde_dict = calc_hist_kde(df, target_col, bins, bandwidth)
            # TODO box plot
            # result.append(_calc_box(df, target_col, bins))
            # qq plot
            qqdata = calc_qqnorm(df[target_col].dropna())
            return Intermediate(
                col=target_col,
                hist_dict=hist_dict,
                kde_dict=kde_dict,
                qqdata=qqdata,
                visual_type="numerical_column",
            )

    # TODO plot(df,x,y)
    # if x is not None and y is not None:
    #     xdtype, ydtype = df[x].dtype, df[y].dtype

    #     if is_categorical(xdtype) and is_numerical(ydtype) or is_numerical(xdtype) and is_categorical(ydtype):
    #         # box plot per group
    #         result.append(_calc_box(df, x, bins, y))
    #         # histogram per group
    #         result.append(calc_hist_by_group(df, x, y, bins))
    #     if is_categorical(xdtype) and is_categorical(ydtype):
    #         # nested bar chart
    #         result.append(_calc_nested(df, x, y))
    #         # stacked bar chart
    #         result.append(_calc_stacked(df, x, y))
    #         # heat map
    #         result.append(_calc_heat_map(df, x, y))
    #     elif is_numerical(xdtype) and is_numerical(ydtype):
    #         # scatter plot
    #         result.append(_calc_scatter(df, x, y, "scatter_plot"))
    #         # hexbin plot
    #         result.append(_calc_scatter(df, x, y, "hexbin_plot"))
    #         # box plot
    #         result.append(_calc_box(df, x, bins, y))
    #     else:
    #         raise ValueError("Invalid data types")
    #     return result


def calc_bar_pie(df: dd.DataFrame, x: str, bars: int, reverse: bool) -> Dict[str, Any]:
    """ Groups the dataframe over the specified column

    Parameters
    __________
    df : the input dask dataframe
    x : the str column of dataframe for which count needs to be calculated
    bars : number of groups with the largest count to return

    Returns
    __________
    dataframe containing the counts in each group
    """
    miss_perc = round(df[x].isna().sum().compute() / len(df) * 100, 1)
    len_df = len(df)
    series = df.groupby(x)[x].count()
    # select largest or smallest groups
    if reverse:
        df = series.nlargest(n=bars).to_frame()
    else:
        df = series.nsmallest(n=bars).to_frame()
    df.columns = ["count"]
    # create a row containing the sum of the other groups
    other_count = len_df - df["count"].sum().compute()
    df2 = pd.DataFrame({x: ["Others"], "count": [other_count]})
    df = df.reset_index().append(to_dask(df2))
    df["percent"] = df["count"] / len_df * 100
    return {"data": df.compute(), "total_groups": len(series), "miss_perc": miss_perc}


def calc_hist(df: dd.DataFrame, x: str, orig_df_len: int, bins: int) -> pd.DataFrame:
    """Returns the histogram array for the continuous
        distribution of values in the column given as the second argument

    Parameters
    __________
    dataframe : the input pandas dataframe
    col : the str column of dataframe for which hist array needs to be
    calculated
    orig_df_len : length of the original dataframe
    show_y_label : show y_axis_labels in visualization if True
    bins : number of bins to use in the histogram

    Returns
    _______
    np.array : An array of values representing histogram for the input col
    """
    # TODO ask about missing values on parsed df
    miss_perc = round(df[x].isna().sum().compute() / len(df) * 100, 1)

    data = df[x].dropna().values
    minv = df[x].min().compute()
    maxv = df[x].max().compute()

    hist_array, bins_array = da.histogram(data, range=[minv, maxv], bins=bins)
    hist_array = hist_array.compute()
    bins_array = format_numbers(df, x, bins_array)
    hist_df = pd.DataFrame(
        {
            "left": bins_array[:-1],
            "right": bins_array[1:],
            "freq": hist_array,
            "percent": hist_array / orig_df_len * 100,
        }
    )
    return {"hist_df": hist_df, "miss_perc": miss_perc}


def calc_hist_kde(
    df: dd.DataFrame, x: str, bins: int, bandwidth: float
) -> Dict[str, Any]:

    data = df[x].dropna().values
    minv = df[x].min().compute()
    maxv = df[x].max().compute()

    hist_array, bins_array = da.histogram(
        data, range=[minv, maxv], bins=bins, density=True
    )
    hist_array = hist_array.compute()
    bins_array = format_numbers(df, x, bins_array)
    hist_df = pd.DataFrame(
        {"left": bins_array[:-1], "right": bins_array[1:], "freq": hist_array}
    )

    calc_pts = np.linspace(minv, maxv, 1000)
    pdf = gaussian_kde(data.compute(), bw_method=bandwidth)(calc_pts)
    return {"hist_df": hist_df, "pdf": pdf, "calc_pts": calc_pts}


def calc_qqnorm(srs: dd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate QQ plot given a series.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of (actual quantiles, theoretical quantiles)
    """
    q_range = np.linspace(0.01, 0.99, 100)
    actual_qs = srs.quantile(q=q_range).compute()
    mean, std = srs.mean().compute(), srs.std().compute()
    theory_qs = np.sort(np.asarray(norm.ppf(q_range, mean, std)))
    return actual_qs, theory_qs


def parse_dataframe(df: dd.DataFrame, x: str, value_range: Tuple[float, float]) -> None:

    if value_range is not None:
        if (
            (value_range[0] <= np.nanmax(df[x]))
            and (value_range[1] >= np.nanmin(df[x]))
            and (value_range[0] < value_range[1])
        ):
            df = df[df[x] >= value_range[0] & df[x] <= value_range[1]]
        else:
            print("Invalid range of values for this column", file=stderr)


def format_numbers(df: dd.DataFrame, x: str, bins_array: List[float]) -> List[float]:
    if np.issubdtype(df[x], np.int64):
        bins_temp = [int(x) for x in np.ceil(bins_array)]
        if len(bins_temp) != len(set(bins_temp)):
            bins_array = [round(x, 2) for x in bins_array]
        else:
            bins_array = bins_temp
    else:
        bins_array = [round(x, 2) for x in bins_array]
    return bins_array
