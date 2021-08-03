# pylint: disable=unused-import
# type: ignore
"""Computations for plot_diff([df1, df2, ..., dfn]，x)."""
from collections import UserList
from typing import Any, Callable, Dict, List, Union, Optional

import sys
import math
import pandas as pd
import numpy as np
import dask
import dask.array as da
import dask.dataframe as dd

from dask.array.stats import kurtosis, skew
from ...utils import gaussian_kde
from ...intermediate import Intermediate
from ...dtypes import (
    Continuous,
    detect_dtype,
    is_dtype,
)
from ...configs import Config
from ...distribution.compute.univariate import _calc_box
from ...correlation.compute.univariate import (
    _pearson_1xn,
    _spearman_1xn,
    _kendall_tau_1xn,
    _corr_filter,
)
from ...correlation.compute.common import CorrelationMethod
from ...eda_frame import EDAFrame


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
        multi_args: Optional[Any] = None,
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
        elif multi_args:
            rslt = []
            for args, data in zip(multi_args, self.data):
                rslt.append(func(data, args, **kwargs))
            return rslt
        else:
            return [func(srs, **kwargs) for srs in self.data]


def compare_multiple_col(
    df_list: List[dd.DataFrame],
    x: str,
    cfg: Config,
) -> Intermediate:
    """
    Compute function for plot_diff([df...],x)

    Parameters
    ----------
    df_list
        Dataframe sequence to be compared.
    x
        Name of the column to be compared
    cfg
        Config instance
    """
    aligned_dfs = dd.concat(df_list, axis=1)
    baseline: int = cfg.diff.baseline
    srs = Srs(aligned_dfs[x])
    data: List[Any] = []
    col_dtype = srs.self_map(detect_dtype)
    if len(col_dtype) > 1:
        col_dtype = col_dtype[baseline]
    else:
        col_dtype = col_dtype[0]

    if is_dtype(col_dtype, Continuous()):
        data.append((_cont_calcs(srs.apply("dropna"), cfg, df_list, x)))
        stats = calc_stats_cont(srs, cfg)
        stats, data = dask.compute(stats, data)

        return Intermediate(col=x, data=data, stats=stats, visual_type="comparison_continuous")
    else:
        return Intermediate()


def _cont_calcs(srs: Srs, cfg: Config, df_list: List[dd.DataFrame], x: str) -> Dict[str, List[Any]]:
    """
    Computations for a continuous column in plot_diff([df...],x)
    """

    # pylint:disable = too-many-branches, too-many-locals

    data: Dict[str, List[Any]] = {}

    # drop infinite values
    mask = srs.apply("isin", {np.inf, -np.inf})
    srs = Srs(srs.getmask(mask, inverse=True), agg=True)
    min_max = srs.apply(
        "map_partitions", lambda x: pd.Series([x.max(), x.min()]), meta=pd.Series([], dtype=float)
    ).data
    if cfg.kde.enable:
        min_max_comp = []
        for min_max_value in dask.compute(min_max)[0]:
            min_max_comp.append(math.isclose(min_max_value.min(), min_max_value.max()))
    min_max = dd.concat(min_max).repartition(npartitions=1)

    # histogram
    if cfg.hist.enable:
        data["hist"] = srs.self_map(
            da.histogram, bins=cfg.hist.bins, range=(min_max.min(), min_max.max())
        )
    # compute the density histogram
    if cfg.kde.enable:
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
    if cfg.box.enable:
        qntls = srs.apply("quantile", [0.25, 0.5, 0.75]).data
        data["box"] = srs.self_map(_calc_box, multi_args=qntls, cfg=cfg)
    if cfg.correlations.enable:
        data["corr"] = []
        for df in df_list:
            df = EDAFrame(df)
            num_df = df.select_num_columns()
            columns = num_df.columns[num_df.columns != x]
            xarr = num_df.values[:, num_df.columns == x]
            data_corr = num_df.values[:, num_df.columns != x]
            funcs = []
            if cfg.pearson.enable:
                funcs.append(_pearson_1xn)
            if cfg.spearman.enable:
                funcs.append(_spearman_1xn)
            if cfg.kendall.enable:
                funcs.append(_kendall_tau_1xn)
            dfs = {}
            (computed,) = dask.compute(
                {meth: func(xarr, data_corr) for meth, func in zip(CorrelationMethod, funcs)}
            )
            for meth, corrs in computed.items():
                indices, corrs = _corr_filter(
                    corrs, cfg.correlations.value_range, cfg.correlations.k
                )
                if len(indices) == 0:
                    print(
                        f"Correlation for {meth.name} is empty, try to broaden the value_range.",
                        file=sys.stderr,
                    )
                num_df = pd.DataFrame(
                    {
                        "x": np.full(len(indices), x),
                        "y": columns[indices],
                        "correlation": corrs,
                    }
                )
                dfs[meth.name] = num_df
            data["corr"].append(dfs)
    if cfg.value_table.enable:
        data["value_table"] = (
            srs.apply("value_counts", "sort=False").apply("nlargest", cfg.value_table.ngroups).data
        )
    return data


def calc_stats_cont(srs: Srs, cfg: Config) -> Dict[str, List[str]]:
    """
    Calculate the statistics for plot_diff([df1, df2, ..., dfn],x)

    Params
    ------
    dfs
        DataFrames to be compared
    """
    stats: Dict[str, List[Any]] = {"nrows": srs.shape.getidx(0)}
    if cfg.stats.enable:
        srs = srs.apply("dropna")
        stats["npres"] = srs.shape.getidx(0)  # number of present (not null) values
        # remove infinite values
        mask = srs.apply("isin", {np.inf, -np.inf})
        srs = Srs(srs.getmask(mask, inverse=True), agg=True)
        stats["qntls"] = srs.apply("quantile", [0.05, 0.25, 0.5, 0.75, 0.95]).data
        stats["skew"] = srs.self_map(skew)
        stats["mean"] = srs.apply("mean").data
        stats["std"] = srs.apply("std").data
        stats["min"] = srs.apply("min").data
        stats["max"] = srs.apply("max").data
        stats["nreals"] = srs.shape.getidx(0)
        stats["nzero"] = srs.apply("apply", lambda x: x == 0).apply("sum").data
        stats["nneg"] = srs.apply("apply", lambda x: x < 0).apply("sum").data
        stats["kurt"] = srs.self_map(kurtosis)
        stats["mem_use"] = srs.apply("memory_usage", "deep=True").data
        stats["nuniq"] = srs.apply("nunique_approx").data
    return stats
