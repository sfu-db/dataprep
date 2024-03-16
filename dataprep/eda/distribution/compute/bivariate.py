"""Computations for plot(df, x, y)."""

from typing import Any, Dict, List, Optional, Tuple, Union

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

from ....errors import UnreachableError
from ...configs import Config
from ...dtypes_v2 import (
    Continuous,
    DateTime,
    DType,
    DTypeDef,
    Nominal,
    GeoGraphy,
    GeoPoint,
    SmallCardNum,
    is_dtype,
    LatLong,
)
from ...intermediate import Intermediate
from ...utils import (
    DTMAP,
    _calc_box_otlrs,
    _calc_box_stats,
    _calc_groups,
    _calc_line_dt,
    _calc_running_total_dt,
    _get_timeunit,
)
from .common import gen_new_df_with_used_cols
from ...eda_frame import EDAFrame


def _check_type_combination(
    instance_tuple: Tuple[DType, DType],
    type_tuple: Tuple[
        Union[type, Tuple[Union[type, Tuple[Any, ...]], ...]],
        Union[type, Tuple[Union[type, Tuple[Any, ...]], ...]],
    ],
) -> bool:
    """Check whether two instance types is instance of types in type_tuples"""

    xtype, ytype = instance_tuple[0], instance_tuple[1]
    type1, type2 = type_tuple[0], type_tuple[1]
    return (isinstance(xtype, type1) and isinstance(ytype, type2)) or (
        isinstance(xtype, type2) and isinstance(ytype, type1)
    )


def compute_bivariate(
    df: Union[pd.DataFrame, dd.DataFrame],
    col1: Union[str, LatLong],
    col2: Union[str, LatLong],
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
    # pylint: disable = too-many-locals

    if col1 is None or col2 is None:
        raise ValueError(f"Input column name should not be None. Input columns: {col1}, y={col2}.")

    new_col_names, ndf = gen_new_df_with_used_cols(df, col1, col2, z=None)
    x = new_col_names[col1]
    y = new_col_names[col2]
    if x is None or y is None:
        raise ValueError

    frame = EDAFrame(ndf, dtype)

    xtype = frame.get_eda_dtype(x)
    ytype = frame.get_eda_dtype(y)

    if _check_type_combination((xtype, ytype), ((Nominal, SmallCardNum), Continuous)):
        x, y = (x, y) if isinstance(xtype, (Nominal, SmallCardNum)) else (y, x)
        tmp_df = frame.frame[[x, y]].dropna()
        # Note that NA is droped, so transform to str will not introduce NA in viz.
        if isinstance(xtype, SmallCardNum):
            tmp_df[x] = tmp_df[x].astype(str)
        (comps,) = dask.compute(_nom_cont_comps(tmp_df, cfg))

        return Intermediate(
            x=x,
            y=y,
            data=comps,
            visual_type="cat_and_num_cols",
        )
    elif _check_type_combination((xtype, ytype), (DateTime, Continuous)):
        x, y = (x, y) if isinstance(xtype, DateTime) else (y, x)
        tmp_df = frame.frame[[x, y]].dropna()
        dtnum: Dict[Str, Any] = {}
        # line chart
        if cfg.line.enable:
            dtnum["linedata_agg"] = dask.delayed(_calc_line_dt)(
                tmp_df, cfg.line.unit, agg=cfg.line.agg
            )
            dtnum["linedata_running_total"] = dask.delayed(_calc_running_total_dt)(
                tmp_df, cfg.line.unit
            )
        # box plot
        if cfg.box.enable:
            dtnum["boxdata"] = dask.delayed(_calc_box_dt)(tmp_df, cfg.box.unit)

        dtnum = dask.compute(dtnum)[0]
        return Intermediate(
            x=x,
            y=y,
            linedata_agg=dtnum.get("linedata_agg", []),
            linedata_running_total=dtnum.get("linedata_running_total", []),
            boxdata=dtnum.get("boxdata", []),
            visual_type="dt_and_num_cols",
        )
    elif _check_type_combination((xtype, ytype), (DateTime, (Nominal, SmallCardNum))):
        x, y = (x, y) if is_dtype(xtype, DateTime()) else (y, x)
        tmp_df = frame.frame[[x, y]].dropna()
        if isinstance(ytype, SmallCardNum):
            tmp_df[y] = tmp_df[y].astype(str)
        dtcat: List[Any] = []
        if cfg.line.enable:
            # line chart
            dtcat.append(
                dask.delayed(_calc_line_dt)(
                    tmp_df,
                    cfg.line.unit,
                    ngroups=cfg.line.ngroups,
                    largest=cfg.line.sort_descending,
                )
            )
        if cfg.stacked.enable:
            # stacked bar chart
            dtcat.append(
                dask.delayed(_calc_stacked_dt)(
                    tmp_df,
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

    elif _check_type_combination((xtype, ytype), (GeoGraphy, Continuous)):
        x, y = (x, y) if isinstance(xtype, GeoGraphy) else (y, x)
        tmp_df = frame.frame[[x, y]]
        (comps,) = dask.compute(geo_cont_comps(tmp_df.dropna(), cfg))
        return Intermediate(x=x, y=y, data=comps, visual_type="geo_and_num_cols")

    elif _check_type_combination((xtype, ytype), (GeoPoint, Continuous)):
        x, y = (x, y) if isinstance(xtype, GeoPoint) else (y, x)
        tmp_df = frame.frame[[x, y]].dropna()
        (comps,) = dask.compute(geop_cont_comps(tmp_df))
        return Intermediate(x=x, y=y, data=comps, visual_type="latlong_and_num_cols")

    elif isinstance(xtype, (Nominal, SmallCardNum, GeoGraphy, GeoPoint)) and isinstance(
        ytype, (Nominal, SmallCardNum, GeoGraphy, GeoPoint)
    ):
        tmp_df = frame.frame[[x, y]].dropna()
        if isinstance(xtype, (SmallCardNum, GeoPoint)):
            tmp_df[x] = tmp_df[x].astype(str)
        if isinstance(ytype, (SmallCardNum, GeoPoint)):
            tmp_df[y] = tmp_df[y].astype(str)

        (comps,) = dask.compute(tmp_df.groupby([x, y]).size())
        return Intermediate(
            x=x,
            y=y,
            data=comps,
            visual_type="two_cat_cols",
        )
    elif isinstance(xtype, Continuous) and isinstance(ytype, Continuous):
        # one partition required for apply(pd.cut) in _calc_box_cont
        tmp_df = frame.frame[[x, y]].dropna().repartition(npartitions=1)

        data: Dict[str, Any] = {}
        if cfg.scatter.enable:
            # scatter plot data
            if cfg.scatter.sample_size is not None:
                sample_func = lambda x: x.sample(n=min(cfg.scatter.sample_size, x.shape[0]))
            else:
                sample_func = lambda x: x.sample(frac=cfg.scatter.sample_rate)
            data["scat"] = tmp_df.map_partitions(sample_func, meta=tmp_df)
        if cfg.hexbin.enable:
            # hexbin plot data
            data["hex"] = tmp_df
        if cfg.box.enable:
            # box plot
            data["box"] = _calc_box_cont(tmp_df, cfg)

        (data,) = dask.compute(data)

        return Intermediate(
            x=x,
            y=y,
            data=data,
            visual_type="two_num_cols",
        )
    else:
        raise UnreachableError(
            "Unprocessed type. x_name:{x}, x_type:{xtype}, y_name:{y}, y_type:{ytype}"
        )


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
