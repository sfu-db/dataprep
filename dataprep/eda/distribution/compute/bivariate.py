"""Computations for plot(df, x, y)."""
from typing import Any, Dict, List, Optional, Tuple, Union

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

from ....errors import UnreachableError
from ...configs import Config
from ...dtypes import (
    Continuous,
    DateTime,
    DTypeDef,
    Nominal,
    GeoGraphy,
    GeoPoint,
    detect_dtype,
    is_dtype,
)
from ...intermediate import Intermediate
from ...utils import (
    DTMAP,
    _calc_box_otlrs,
    _calc_box_stats,
    _calc_groups,
    _calc_line_dt,
    _get_timeunit,
)


def compute_bivariate(
    df: dd.DataFrame,
    x: str,
    y: str,
    cfg: Config,
    dtype: Optional[DTypeDef],
) -> Intermediate:
    """Compute functions for plot(df, x, y).

    Parameters
    ----------
    df
        DataFrame from which visualizations are generated
    x
        A column name from the DataFrame
    y
        A column name from the DataFrame
    cfg
        Config instance
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    """
    # pylint: disable=too-many-branches,too-many-statements,too-many-boolean-expressions
    # pylint: disable = too-many-return-statements
    xtype, ytype = detect_dtype(df[x], dtype), detect_dtype(df[y], dtype)

    if (is_dtype(xtype, Nominal()) and is_dtype(ytype, Continuous())) or (
        is_dtype(xtype, Continuous()) and is_dtype(ytype, Nominal())
    ):
        x, y = (x, y) if is_dtype(xtype, Nominal()) else (y, x)
        df = df[[x, y]]
        # Since it will throw error if column is object while some cells are
        # numerical, we transform column to string first.
        df[x] = df[x].astype(str)

        (comps,) = dask.compute(_nom_cont_comps(df.dropna(), cfg))

        return Intermediate(
            x=x,
            y=y,
            data=comps,
            visual_type="cat_and_num_cols",
        )
    elif (is_dtype(xtype, DateTime()) and is_dtype(ytype, Continuous())) or (
        is_dtype(xtype, Continuous()) and is_dtype(ytype, DateTime())
    ):
        x, y = (x, y) if is_dtype(xtype, DateTime()) else (y, x)
        df = df[[x, y]].dropna()
        dtnum: List[Any] = []
        # line chart
        if cfg.line.enable:
            dtnum.append(dask.delayed(_calc_line_dt)(df, cfg.line.unit, cfg.line.agg))
        # box plot
        if cfg.box.enable:
            dtnum.append(dask.delayed(_calc_box_dt)(df, cfg.box.unit))

        dtnum = dask.compute(*dtnum)

        if len(dtnum) == 2:
            linedata = dtnum[0]
            boxdata = dtnum[1]
        elif cfg.line.enable:
            linedata = dtnum[0]
            boxdata = []
        else:
            boxdata = dtnum[0]
            linedata = []

        return Intermediate(
            x=x,
            y=y,
            linedata=linedata,
            boxdata=boxdata,
            visual_type="dt_and_num_cols",
        )
    elif (is_dtype(xtype, DateTime()) and is_dtype(ytype, Nominal())) or (
        is_dtype(xtype, Nominal()) and is_dtype(ytype, DateTime())
    ):
        x, y = (x, y) if is_dtype(xtype, DateTime()) else (y, x)
        df = df[[x, y]].dropna()
        df[y] = df[y].apply(str, meta=(y, str))
        dtcat: List[Any] = []
        if cfg.line.enable:
            # line chart
            dtcat.append(
                dask.delayed(_calc_line_dt)(
                    df,
                    cfg.line.unit,
                    ngroups=cfg.line.ngroups,
                    largest=cfg.line.sort_descending,
                )
            )
        if cfg.stacked.enable:
            # stacked bar chart
            dtcat.append(
                dask.delayed(_calc_stacked_dt)(
                    df,
                    cfg.stacked.unit,
                    cfg.stacked.ngroups,
                    cfg.stacked.sort_descending,
                )
            )
        dtcat = dask.compute(*dtcat)

        if len(dtcat) == 2:
            linedata = dtcat[0]
            stackdata = dtcat[1]
        elif cfg.line.enable:
            linedata = dtcat[0]
            stackdata = []
        else:
            stackdata = dtcat[0]
            linedata = []

        return Intermediate(
            x=x,
            y=y,
            linedata=linedata,
            stackdata=stackdata,
            visual_type="dt_and_cat_cols",
        )
    elif (is_dtype(xtype, GeoGraphy()) and is_dtype(ytype, Continuous())) or (
        is_dtype(xtype, Continuous()) and is_dtype(ytype, GeoGraphy())
    ):
        x, y = (x, y) if is_dtype(xtype, GeoGraphy()) else (y, x)
        df = df[[x, y]]
        first_rows = df.head()
        try:
            first_rows[x].apply(hash)
        except TypeError:
            df[x] = df[x].astype(str)

        (comps,) = dask.compute(geo_cont_comps(df.dropna(), cfg))

        return Intermediate(x=x, y=y, data=comps, visual_type="geo_and_num_cols")
    elif (is_dtype(xtype, GeoPoint()) and is_dtype(ytype, Continuous())) or (
        is_dtype(xtype, Continuous()) and is_dtype(ytype, GeoPoint())
    ):
        x, y = (x, y) if is_dtype(xtype, GeoPoint()) else (y, x)
        df = df[[x, y]]
        first_rows = df.head()
        try:
            first_rows[x].apply(hash)
        except TypeError:
            df[x] = df[x].astype(str)

        (comps,) = dask.compute(geop_cont_comps(df.dropna()))

        return Intermediate(x=x, y=y, data=comps, visual_type="latlong_and_num_cols")
    elif (
        is_dtype(xtype, Nominal()) or is_dtype(xtype, GeoGraphy()) or is_dtype(xtype, GeoPoint())
    ) and (
        is_dtype(ytype, Nominal()) or is_dtype(ytype, GeoGraphy()) or is_dtype(ytype, GeoPoint())
    ):
        df = df[[x, y]]
        # Since it will throw error if column is object while some cells are
        # numerical, we transform column to string first.
        df[x] = df[x].astype(str)
        df[y] = df[y].astype(str)

        if is_dtype(xtype, GeoPoint()):
            df[x] = df[x].astype(str)
        if is_dtype(ytype, GeoPoint()):
            df[y] = df[y].astype(str)

        (comps,) = dask.compute(df.dropna().groupby([x, y]).size())
        return Intermediate(
            x=x,
            y=y,
            data=comps,
            visual_type="two_cat_cols",
        )
    elif is_dtype(xtype, Continuous()) and is_dtype(ytype, Continuous()):
        # one partition required for apply(pd.cut) in _calc_box_cont
        df = df[[x, y]].dropna().repartition(npartitions=1)

        data: Dict[str, Any] = {}
        if cfg.scatter.enable:
            # scatter plot data
            data["scat"] = df.map_partitions(
                lambda x: x.sample(min(cfg.scatter.sample_size, x.shape[0])), meta=df
            )
        if cfg.hexbin.enable:
            # hexbin plot data
            data["hex"] = df
        if cfg.box.enable:
            # box plot
            data["box"] = _calc_box_cont(df, cfg)

        (data,) = dask.compute(data)

        return Intermediate(
            x=x,
            y=y,
            data=data,
            visual_type="two_num_cols",
        )
    else:
        raise UnreachableError


def _nom_cont_comps(df: dd.DataFrame, cfg: Config) -> Dict[str, Any]:
    """
    Computations for a nominal and continuous column
    """
    data: Dict[str, Any] = {}
    x, y = df.columns

    # filter the dataframe to consist of ngroup groups
    # https://stackoverflow.com/questions/46927174/filtering-grouped-df-in-dask
    cnts = df[x].value_counts(sort=False)
    data["ttl_grps"] = cnts.shape[0]

    if cfg.box.enable:
        if cfg.box.sort_descending:
            thresh = cnts.nlargest(cfg.box.ngroups).min()
            df_box = df[df[x].map(cnts) >= thresh]
        else:
            thresh = cnts.nsmallest(cfg.box.ngroups).max()
            df_box = df[df[x].map(cnts) <= thresh]
        grps_box = df_box.groupby(x)[y]
        data["box"] = grps_box.apply(_box_comps, meta=object)
        if (
            cfg.line.enable
            and cfg.box.sort_descending == cfg.line.sort_descending
            and cfg.box.ngroups == cfg.line.ngroups
        ):
            data["hist"] = grps_box.apply(
                _hist, cfg.line.bins, df_box[y].min(), df_box[y].max(), meta=object
            )
    if cfg.line.enable and (
        cfg.box.sort_descending != cfg.line.sort_descending or cfg.box.ngroups != cfg.line.ngroups
    ):
        if cfg.line.sort_descending:
            thresh = cnts.nlargest(cfg.line.ngroups).min()
            df_hist = df[df[x].map(cnts) >= thresh]
        else:
            thresh = cnts.nsmallest(cfg.line.ngroups).max()
            df_hist = df[df[x].map(cnts) <= thresh]
        grps_hist = df_hist.groupby(x)[y]
        data["hist"] = grps_hist.apply(
            _hist, cfg.line.bins, df_hist[y].min(), df_hist[y].max(), meta=object
        )

    return data


def geo_cont_comps(df: dd.DataFrame, cfg: Config) -> Dict[str, Any]:
    """
    Computations for a geography and continuous column

    Parameters
    ----------
    df
        Dask dataframe with one geography and one numerical column
    """
    data: Dict[str, Any] = {}
    x, y = df.columns
    cnts = df[x].value_counts(sort=False)
    data["ttl_grps"] = cnts.shape[0]

    if cfg.box.enable:
        if cfg.box.sort_descending:
            thresh = cnts.nlargest(cfg.box.ngroups).min()
            df_box = df[df[x].map(cnts) >= thresh]
        else:
            thresh = cnts.nsmallest(cfg.box.ngroups).max()
            df_box = df[df[x].map(cnts) <= thresh]
        grps_box = df_box.groupby(x)[y]
        data["box"] = grps_box.apply(_box_comps, meta=object)
        if (
            cfg.line.enable
            and cfg.box.sort_descending == cfg.line.sort_descending
            and cfg.box.ngroups == cfg.line.ngroups
        ):
            data["hist"] = grps_box.apply(
                _hist, cfg.line.bins, df_box[y].min(), df_box[y].max(), meta=object
            )
    if cfg.line.enable and (
        cfg.box.sort_descending != cfg.line.sort_descending or cfg.box.ngroups != cfg.line.ngroups
    ):
        if cfg.line.sort_descending:
            thresh = cnts.nlargest(cfg.line.ngroups).min()
            df_hist = df[df[x].map(cnts) >= thresh]
        else:
            thresh = cnts.nsmallest(cfg.line.ngroups).max()
            df_hist = df[df[x].map(cnts) <= thresh]
        grps_hist = df_hist.groupby(x)[y]
        data["hist"] = grps_hist.apply(
            _hist, cfg.line.bins, df_hist[y].min(), df_hist[y].max(), meta=object
        )
    # group the data to compute the mean
    data["value"] = df.groupby(x)[y].mean()

    return data


def geop_cont_comps(df: dd.DataFrame) -> Dict[str, Any]:
    """
    Computations for a geography and continuous column

    Parameters
    ----------
    df
        Dask dataframe with one geography and one numerical column
    """
    data: Dict[str, Any] = {}
    x, y = df.columns
    # group the data to compute the mean
    data["value"] = df.groupby(x)[y].mean()

    return data


def _calc_box_cont(df: dd.DataFrame, cfg: Config) -> dd.Series:
    """
    Box plot for a binned continuous variable
    """
    x, y = df.columns
    # group the data into intervals
    # https://stackoverflow.com/questions/42442043/how-to-use-pandas-cut-or-equivalent-in-dask-efficiently
    df["grp"] = df[x].map_partitions(pd.cut, bins=cfg.box.bins, include_lowest=True)
    # TODO is this calculating the box plot stats for each group in parallel?
    # https://examples.dask.org/dataframes/02-groupby.html#Groupby-Apply
    # https://github.com/dask/dask/issues/4239
    # https://github.com/dask/dask/issues/5124
    srs = df.groupby("grp")[y].apply(_box_comps, meta=object)

    return srs


def _box_comps(srs: pd.Series) -> Dict[str, Union[float, np.array]]:
    """
    Box plot computations
    """
    data: Dict[str, Any] = {}

    # quartiles
    data.update(zip(("q1", "q2", "q3"), srs.quantile([0.25, 0.5, 0.75])))
    iqr = data["q3"] - data["q1"]
    # inliers
    srs_iqr = srs[srs.between(data["q1"] - 1.5 * iqr, data["q3"] + 1.5 * iqr)]
    data["lw"], data["uw"] = srs_iqr.min(), srs_iqr.max()
    # outliers
    otlrs = srs[~srs.between(data["q1"] - 1.5 * iqr, data["q3"] + 1.5 * iqr)]
    # randomly sample at most 100 outliers
    data["otlrs"] = otlrs.sample(min(100, otlrs.shape[0])).values

    return data


def _hist(srs: pd.Series, bins: int, minv: float, maxv: float) -> Any:
    """
    Compute a histogram on a given series
    """

    return np.histogram(srs, bins=bins, range=(minv, maxv))


def _calc_box_dt(df: dd.DataFrame, unit: str) -> Tuple[pd.DataFrame, List[str], List[float], str]:
    """
    Calculate a box plot with date on the x axis.
    Parameters
    ----------
    df
        A dataframe with one datetime and one numerical column
    unit
        The unit of time over which to group the values
    """

    x, y = df.columns[0], df.columns[1]  # time column
    unit = _get_timeunit(df[x].min(), df[x].max(), 10) if unit == "auto" else unit
    if unit not in DTMAP.keys():
        raise ValueError
    grps = df.groupby(pd.Grouper(key=x, freq=DTMAP[unit][0]))  # time groups
    # box plot for the values in each time group
    df = pd.concat(
        [_calc_box_stats(g[1][y], g[0], True) for g in grps],
        axis=1,
    )
    df = df.append(
        pd.Series(
            {c: i + 1 for i, c in enumerate(df.columns)},
            name="x",
        )
    ).T
    # If grouping by week, make the label for the week the beginning Sunday
    df.index = df.index - pd.to_timedelta(6, unit="d") if unit == "week" else df.index
    df.index.name = "grp"
    df = df.reset_index()
    df["grp"] = df["grp"].dt.to_period("S").dt.strftime(DTMAP[unit][2])
    df["x0"], df["x1"] = df["x"] - 0.8, df["x"] - 0.2  # width of whiskers for plotting
    outx, outy = _calc_box_otlrs(df)

    return df, outx, outy, DTMAP[unit][3]


def _calc_stacked_dt(
    df: dd.DataFrame,
    unit: str,
    ngroups: int,
    largest: bool,
) -> Tuple[pd.DataFrame, Dict[str, int], str]:
    """
    Calculate a stacked bar chart with date on the x axis
    Parameters
    ----------
    df
        A dataframe with one datetime and one categorical column
    unit
        The unit of time over which to group the values
    ngroups
        Number of groups for the categorical column
    largest
        Use the largest or smallest groups in the categorical column
    """
    # pylint: disable=too-many-locals

    x, y = df.columns[0], df.columns[1]  # time column
    unit = _get_timeunit(df[x].min(), df[x].max(), 10) if unit == "auto" else unit
    if unit not in DTMAP.keys():
        raise ValueError

    # get the largest groups
    df_grps, grp_cnt_stats, _ = _calc_groups(df, y, ngroups, largest)
    grouper = (pd.Grouper(key=x, freq=DTMAP[unit][0]),)  # time grouper
    # pivot table of counts with date groups as index and categorical values as column names
    dfr = pd.pivot_table(
        df_grps,
        index=grouper,
        columns=y,
        aggfunc=len,
        fill_value=0,
    ).rename_axis(None)

    # if more than ngroups categorical values, aggregate the smallest groups into "Others"
    if grp_cnt_stats[f"{y}_ttl"] > grp_cnt_stats[f"{y}_shw"]:
        grp_cnts = df.groupby(pd.Grouper(key=x, freq=DTMAP[unit][0])).size()
        dfr["Others"] = grp_cnts - dfr.sum(axis=1)

    dfr.index = (  # If grouping by week, make the label for the week the beginning Sunday
        dfr.index - pd.to_timedelta(6, unit="d") if unit == "week" else dfr.index
    )
    dfr.index = dfr.index.to_period("S").strftime(DTMAP[unit][2])  # format labels

    return dfr, grp_cnt_stats, DTMAP[unit][3]
