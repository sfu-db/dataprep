# pylint: disable=unused-import
# type: ignore
"""Computations for plot_diff([df1, df2, ..., dfn])."""

from collections import UserList, OrderedDict
from typing import Any, Callable, Dict, List, Tuple, Union, Optional

import math
import pandas as pd
import numpy as np
import dask
import dask.array as da
import dask.dataframe as dd
from pandas.api.types import is_integer_dtype
from ...utils import DTMAP, _get_timeunit, to_dask
from ...intermediate import Intermediate
from ...dtypes import (
    Nominal,
    Continuous,
    DateTime,
    detect_dtype,
    get_dtype_cnts_and_num_cols,
    is_dtype,
    drop_null,
    DTypeDef,
)
from ...utils import gaussian_kde
from ...configs import Config


class Dfs(UserList):
    """
    This class implements a sequence of DataFrames
    """

    # pylint: disable=too-many-ancestors
    def __init__(self, dfs: List[dd.DataFrame]) -> None:
        super().__init__(dfs)

    def __getattr__(self, attr: str) -> UserList:
        output = []
        for df in self.data:
            output.append(getattr(df, attr))
        return Dfs(output)

    def apply(self, method: str, *params: Optional[Any], **kwargs: Optional[Any]) -> UserList:
        """
        Apply the same method for all elements in the list.
        """
        output = []
        for df in self.data:
            output.append(getattr(df, method)(*params, **kwargs))

        return Dfs(output)

    def getidx(self, ind: Union[str, int]) -> List[Any]:
        """
        Get the specified index for all elements in the list.
        """
        output = []
        for data in self.data:
            output.append(data[ind])
        return output

    def self_map(self, func: Callable[[dd.Series], Any], **kwargs: Any) -> List[Any]:
        """
        Map the data to the given function.
        """
        return [func(df, **kwargs) for df in self.data]


class Srs(UserList):
    """
    This class **separates** the columns with the same name into individual series.
    """

    # pylint: disable=too-many-ancestors, eval-used, too-many-locals
    def __init__(self, srs: Union[dd.DataFrame, List[Any]], agg: bool = False) -> None:
        super().__init__()
        if agg:
            self.data = srs
        else:
            if len(srs.shape) > 1:
                self.data: List[dd.Series] = [srs.iloc[:, loc] for loc in range(srs.shape[1])]
            else:
                self.data: List[dd.Series] = [srs]

    def __getattr__(self, attr: str) -> UserList:
        output = []
        for srs in self.data:
            output.append(getattr(srs, attr))
        return Srs(output, agg=True)

    def apply(self, method: str, *params: Optional[Any], **kwargs: Optional[Any]) -> UserList:
        """
        Apply the same method for all elements in the list.
        """
        output = []
        for srs in self.data:
            output.append(getattr(srs, method)(*params, **kwargs))

        return Srs(output, agg=True)

    def getidx(self, ind: Union[str, int]) -> List[Any]:
        """
        Get the specified index for all elements in the list.
        """
        output = []
        for data in self.data:
            output.append(data[ind])

        return output

    def getmask(
        self, mask: Union[List[dd.Series], UserList], inverse: bool = False
    ) -> List[dd.Series]:
        """
        Return rows based on a boolean mask.
        """
        output = []
        for data, cond in zip(self.data, mask):
            if inverse:
                output.append(data[~cond])
            else:
                output.append(data[cond])

        return output

    def self_map(
        self,
        func: Callable[[dd.Series], Any],
        condition: Optional[List[bool]] = None,
        **kwargs: Any,
    ) -> List[Any]:
        """
        Map the data to the given function.
        """
        if condition:
            rslt = []
            for cond, data in zip(condition, self.data):
                if not cond:
                    rslt.append(func(data, **kwargs))
                else:
                    rslt.append(None)
            return rslt
        return [func(srs, **kwargs) for srs in self.data]


def compare_multiple_df(
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

        if is_dtype(col_dtype, Continuous()) and cfg.hist.enable:
            data.append((col, Continuous(), _cont_calcs(srs.apply("dropna"), cfg), orig))
        elif is_dtype(col_dtype, Nominal()) and cfg.bar.enable:
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

            data.append((col, Nominal(), _nom_calcs(norm_srs, cfg), orig))
        elif is_dtype(col_dtype, DateTime()) and cfg.line.enable:
            data.append(
                (col, DateTime(), dask.delayed(_calc_line_dt)(srs, col, cfg.line.unit), orig)
            )

    stats = calc_stats(dfs, cfg, dtype)
    stats, data = dask.compute(stats, data)
    plot_data: List[Tuple[str, DTypeDef, Any, List[str]]] = []

    for col, dtp, datum, orig in data:
        if is_dtype(dtp, Continuous()):
            if cfg.diff.density:
                plot_data.append((col, dtp, (datum["kde"], datum["dens"]), orig))
            elif cfg.hist.enable:
                plot_data.append((col, dtp, datum["hist"], orig))
        elif is_dtype(dtp, Nominal()):
            if cfg.bar.enable:
                plot_data.append((col, dtp, (datum["bar"], datum["nuniq"]), orig))
        elif is_dtype(dtp, DateTime()):
            plot_data.append((col, dtp, dask.compute(*datum), orig))  # workaround
    return Intermediate(data=plot_data, stats=stats, visual_type="comparison_grid")


def _is_all_int(df_list: List[Union[dd.DataFrame, pd.DataFrame]], col: str) -> bool:
    """
    Check whether the col in all dataframes are all integer type.
    """
    for df in df_list:
        if col in df.columns:
            srs = df[col]
            if isinstance(srs, (dd.DataFrame, pd.DataFrame)):
                for dtype in srs.dtypes:
                    if not is_integer_dtype(dtype):
                        return False
            elif isinstance(srs, (dd.Series, pd.Series)):
                if not is_integer_dtype(srs.dtype):
                    return False
            else:
                raise ValueError(f"unprocessed type of data:{type(srs)}")
    return True


def calc_stats(dfs: Dfs, cfg: Config, dtype: Optional[DTypeDef]) -> Dict[str, List[str]]:
    """
    Calculate the statistics for plot_diff([df1, df2, ..., dfn])

    Params
    ------
    dfs
        DataFrames to be compared
    """
    stats: Dict[str, List[Any]] = {"nrows": dfs.shape.getidx(0)}
    dtype_cnts = []
    num_cols = []
    if cfg.stats.enable:
        for df in dfs:
            temp = get_dtype_cnts_and_num_cols(df, dtype=dtype)
            dtype_cnts.append(temp[0])
            num_cols.append(temp[1])

        stats["ncols"] = dfs.shape.getidx(1)
        stats["npresent_cells"] = dfs.apply("count").apply("sum").data
        stats["nrows_wo_dups"] = dfs.apply("drop_duplicates").shape.getidx(0)
        stats["mem_use"] = dfs.apply("memory_usage", "deep=True").apply("sum").data
        stats["dtype_cnts"] = dtype_cnts

    return stats


def _cont_calcs(srs: Srs, cfg: Config) -> Dict[str, List[Any]]:
    """
    Computations for a continuous column in plot_diff([df1, df2, ..., dfn])
    """

    data: Dict[str, List[Any]] = {}

    # drop infinite values
    mask = srs.apply("isin", {np.inf, -np.inf})
    srs = Srs(srs.getmask(mask, inverse=True), agg=True)
    min_max = srs.apply(
        "map_partitions", lambda x: pd.Series([x.max(), x.min()]), meta=pd.Series([], dtype=float)
    ).data
    min_max_comp = []
    if cfg.diff.density:
        for min_max_value in dask.compute(min_max)[0]:
            min_max_comp.append(math.isclose(min_max_value.min(), min_max_value.max()))
    min_max = dd.concat(min_max).repartition(npartitions=1)

    # histogram
    data["hist"] = srs.self_map(
        da.histogram, bins=cfg.hist.bins, range=(min_max.min(), min_max.max())
    )

    # compute the density histogram
    if cfg.diff.density:
        data["dens"] = srs.self_map(
            da.histogram,
            condition=min_max_comp,
            bins=cfg.kde.bins,
            range=(min_max.min(), min_max.max()),
            density=True,
        )
        # gaussian kernel density estimate
        data["kde"] = []
        sample_data = dask.compute(
            srs.apply(
                "map_partitions",
                lambda x: x.sample(min(1000, x.shape[0])),
                meta=pd.Series([], dtype=float),
            ).data
        )
        for ind in range(len(sample_data[0])):
            data["kde"].append(gaussian_kde(sample_data[0][ind]))

    return data


def _nom_calcs(srs: Srs, cfg: Config) -> Dict[str, List[Any]]:
    """
    Computations for a nominal column in plot_diff([df1, df2, ..., dfn])
    """
    # pylint: disable=bare-except, too-many-nested-blocks

    # dictionary of data for the bar chart and related insights
    data: Dict[str, List[Any]] = {}
    baseline: int = cfg.diff.baseline

    # value counts for barchart and uniformity insight

    if cfg.bar.enable:
        if len(srs) > 1:
            grps = srs.apply("value_counts", "sort=False").data
            # select the largest or smallest groups
            grps[baseline] = (
                grps[baseline].nlargest(cfg.bar.bars)
                if cfg.bar.sort_descending
                else grps[baseline].nsmallest(cfg.bar.bars)
            )
            data["bar"] = grps
            data["nuniq"] = grps[baseline].shape[0]
        else:  # singular column
            grps = srs.apply("value_counts", "sort=False").data[0]
            ngrp = (
                grps.nlargest(cfg.bar.bars)
                if cfg.bar.sort_descending
                else grps.nsmallest(cfg.bar.bars)
            )
            data["bar"] = [ngrp.to_frame()]
            data["nuniq"] = grps.shape[0]
    return data


def _dask_group_by_time_series(dd: dd.DataFrame, key: str, freq: str) -> dd.DataFrame:
    df = dd.compute()
    gb_df = df.groupby(pd.Grouper(key=key, freq=freq)).size().reset_index()
    return to_dask(gb_df)


def _calc_line_dt(srs: Srs, x: str, unit: str) -> Tuple[List[pd.DataFrame], str]:
    """
    Calculate a line or multiline chart with date on the x axis. If df contains
    one datetime column, it will make a line chart of the frequency of values.

    Parameters
    ----------
    df
        A dataframe
    x
        The column name
    unit
        The unit of time over which to group the values
    """
    # pylint: disable=too-many-locals
    unit_range = dask.compute(*(srs.apply("min").data, srs.apply("max").data))
    unit = _get_timeunit(min(unit_range[0]), min(unit_range[1]), 100) if unit == "auto" else unit
    dfs = Dfs(srs.apply("to_frame").self_map(drop_null))
    if unit not in DTMAP.keys():
        raise ValueError
    dfr = dfs.self_map(_dask_group_by_time_series, key=x, freq=DTMAP[unit][0])
    for df in dfr:
        df.columns = [x, "freq"]
        df["pct"] = df["freq"] / len(df) * 100

        df[x] = df[x] - pd.to_timedelta(6, unit="d") if unit == "week" else df[x]
        df["lbl"] = df[x].dt.to_period("S").dt.strftime(DTMAP[unit][1])

    return (dfr, DTMAP[unit][3])
