"""
This module implements the intermediates computation for plot(df) function.
"""  # pylint: disable=too-many-lines
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union, cast

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.array.stats import kurtosis, skew
from nltk.stem import PorterStemmer, WordNetLemmatizer
from scipy.stats import gaussian_kde

from ...assets.english_stopwords import english_stopwords
from ...errors import UnreachableError
from ..dtypes import (
    Continuous,
    DateTime,
    DType,
    DTypeDef,
    Nominal,
    detect_dtype,
    drop_null,
    is_dtype,
)
from ..intermediate import Intermediate
from ..utils import to_dask

__all__ = ["compute"]

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


def compute(
    df: Union[pd.DataFrame, dd.DataFrame],
    x: Optional[str] = None,
    y: Optional[str] = None,
    z: Optional[str] = None,
    *,
    bins: int = 10,
    ngroups: int = 10,
    largest: bool = True,
    nsubgroups: int = 5,
    timeunit: str = "auto",
    agg: str = "mean",
    sample_size: int = 1000,
    top_words: int = 30,
    stopword: bool = True,
    lemmatize: bool = False,
    stem: bool = False,
    value_range: Optional[Tuple[float, float]] = None,
    dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    """
    Parameters
    ----------
    df
        Dataframe from which plots are to be generated
    x: Optional[str], default None
        A valid column name from the dataframe
    y: Optional[str], default None
        A valid column name from the dataframe
    z: Optional[str], default None
        A valid column name from the dataframe
    bins: int, default 10
        For a histogram or box plot with numerical x axis, it defines
        the number of equal-width bins to use when grouping.
    ngroups: int, default 10
        When grouping over a categorical column, it defines the
        number of groups to show in the plot. Ie, the number of
        bars to show in a bar chart.
    largest: bool, default True
        If true, when grouping over a categorical column, the groups
        with the largest count will be output. If false, the groups
        with the smallest count will be output.
    nsubgroups: int, default 5
        If x and y are categorical columns, ngroups refers to
        how many groups to show from column x, and nsubgroups refers to
        how many subgroups to show from column y in each group in column x.
    timeunit: str, default "auto"
        Defines the time unit to group values over for a datetime column.
        It can be "year", "quarter", "month", "week", "day", "hour",
        "minute", "second". With default value "auto", it will use the
        time unit such that the resulting number of groups is closest to 15.
    agg: str, default "mean"
        Specify the aggregate to use when aggregating over a numeric column
    sample_size: int, default 1000
        Sample size for the scatter plot
    top_words: int, default 30
        Specify the amount of words to show in the wordcloud and
        word frequency bar chart
    stopword: bool, default True
        Eliminate the stopwords in the text data for plotting wordcloud and
        word frequency bar chart
    lemmatize: bool, default False
        Lemmatize the words in the text data for plotting wordcloud and
        word frequency bar chart
    stem: bool, default False
        Apply Potter Stem on the text data for plotting wordcloud and
        word frequency bar chart
    value_range: Optional[Tuple[float, float]], default None
        The lower and upper bounds on the range of a numerical column.
        Applies when column x is specified and column y is unspecified.
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    """  # pylint: disable=too-many-locals

    df = to_dask(df)

    if not any((x, y, z)):
        return compute_overview(df, bins, ngroups, largest, timeunit, dtype)

    if sum(v is None for v in (x, y, z)) == 2:
        col: str = cast(str, x or y or z)
        return compute_univariate(
            df,
            col,
            bins,
            ngroups,
            largest,
            timeunit,
            top_words,
            stopword,
            lemmatize,
            stem,
            value_range,
            dtype,
        )

    if sum(v is None for v in (x, y, z)) == 1:
        x, y = (v for v in (x, y, z) if v is not None)
        return compute_bivariate(
            df,
            x,
            y,
            bins,
            ngroups,
            largest,
            nsubgroups,
            timeunit,
            agg,
            sample_size,
            dtype,
        )

    if x is not None and y is not None and z is not None:
        return compute_trivariate(df, x, y, z, ngroups, largest, timeunit, agg, dtype)

    return Intermediate()


def compute_overview(
    df: dd.DataFrame,
    bins: int,
    ngroups: int,
    largest: bool,
    timeunit: str,
    dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    # pylint: disable=too-many-arguments,too-many-locals
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
    dtype_cnts: DefaultDict[str, int] = defaultdict(int)
    col_names_dtypes: List[Tuple[str, DType]] = []
    for column in df.columns:
        srs = df[column]
        column_dtype = detect_dtype(srs, dtype)

        if is_dtype(column_dtype, Nominal()):
            # cast the column as string type if it contains a mutable type
            try:
                first_rows[column].apply(hash)
            except TypeError:
                srs = df[column] = srs.dropna().astype(str)
            # bar chart
            datas.append(calc_bar(srs, ngroups, largest))
            col_names_dtypes.append((column, Nominal()))
            dtype_cnts["Categorical"] += 1
        elif is_dtype(column_dtype, Continuous()):
            # histogram
            hist = da.histogram(drop_null(srs), bins=bins, range=[srs.min(), srs.max()])
            datas.append(hist)
            col_names_dtypes.append((column, Continuous()))
            dtype_cnts["Numerical"] += 1
        elif is_dtype(column_dtype, DateTime()):
            datas.append(dask.delayed(calc_line_dt)(df[[column]], timeunit))
            col_names_dtypes.append((column, DateTime()))
            dtype_cnts["DateTime"] += 1
        else:
            raise UnreachableError

    stats = calc_stats(df, dtype_cnts)
    datas, stats = dask.compute(datas, stats)
    data = [(col, dtp, dat) for (col, dtp), dat in zip(col_names_dtypes, datas)]
    return Intermediate(data=data, stats=stats, visual_type="distribution_grid",)


def compute_univariate(
    df: dd.DataFrame,
    x: str,
    bins: int,
    ngroups: int,
    largest: bool,
    timeunit: str,
    top_words: int,
    stopword: bool = True,
    lemmatize: bool = False,
    stem: bool = False,
    value_range: Optional[Tuple[float, float]] = None,
    dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    """
    Compute functions for plot(df, x)
    Parameters
    ----------
    df
        Dataframe from which plots are to be generated
    x
        A valid column name from the dataframe
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
    top_words: int, default 30
        Specify the amount of words to show in the wordcloud and
        word frequency bar chart
    stopword: bool, default True
        Eliminate the stopwords in the text data for plotting wordcloud and
        word frequency bar chart
    lemmatize: bool, default False
        Lemmatize the words in the text data for plotting wordcloud and
        word frequency bar chart
    stem: bool, default False
        Apply Potter Stem on the text data for plotting wordcloud and
        word frequency bar chart
    value_range
        The lower and upper bounds on the range of a numerical column.
        Applies when column x is specified and column y is unspecified.
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    """
    # pylint: disable=too-many-locals, too-many-arguments

    col_dtype = detect_dtype(df[x], dtype)
    if is_dtype(col_dtype, Nominal()):
        # extract the column
        df_x = df[x]
        # calculate the total rows
        nrows = df_x.shape[0]
        # cast the column as string type if it contains a mutable type
        if df_x.head().apply(lambda x: hasattr(x, "__hash__")).any():
            # drop_null() will not work if the column conatains a mutable type
            df_x = df_x.dropna().astype(str)

        # drop null values
        df_x = drop_null(df_x)

        # calc_word_freq() returns the frequency of words (for the word cloud and word
        # frequency bar chart) and the total number of words
        word_data = calc_word_freq(df_x, top_words, stopword, lemmatize, stem)

        # calc_cat_stats() computes all the categorical stats including the length
        # histogram. calc_bar_pie() does the calculations for the bar and pie charts
        # NOTE this dictionary could be returned to create_report without
        # calling the subsequent compute
        cat_data = {
            "stats": calc_cat_stats(df_x, nrows, bins),
            "bar_pie": calc_bar_pie(df_x, ngroups, largest),
            "word_data": word_data,
        }
        cat_data = dask.compute(cat_data)[0]

        return Intermediate(
            col=x,
            stats=cat_data["stats"],
            bar_pie=cat_data["bar_pie"],
            word_data=cat_data["word_data"],
            visual_type="categorical_column",
        )
    elif is_dtype(col_dtype, Continuous()):

        # calculate the total number of rows then drop the missing values
        nrows = df.shape[0]
        df_x = drop_null(df[x])

        if value_range is not None:
            df_x = df_x[df_x.between(*value_range)]

        # TODO perhaps we should not use to_dask() on the entire
        # initial dataframe and instead only use the column of data
        # df_x = df_x.repartition(partition_size="100MB")

        # calculate numerical statistics and extract the min and max
        num_stats = calc_num_stats(df_x, nrows)
        minv, maxv = num_stats["min"], num_stats["max"]

        # NOTE this dictionary could be returned to create_report without
        # calling the subsequent compute
        num_data = {
            "hist": da.histogram(df_x, bins=bins, range=[minv, maxv]),
            "kde": calc_kde(df_x, bins, minv, maxv),
            "box_data": calc_box_new(df_x, num_stats["qntls"]),
            "stats": num_stats,
        }
        num_data = dask.compute(num_data)[0]

        return Intermediate(
            col=x,
            hist=num_data["hist"],
            kde=num_data["kde"],
            box_data=num_data["box_data"],
            stats=num_data["stats"],
            visual_type="numerical_column",
        )
    elif is_dtype(col_dtype, DateTime()):
        data_dt: List[Any] = []
        # line chart
        data_dt.append(dask.delayed(calc_line_dt)(df[[x]], timeunit))
        # stats
        data_dt.append(dask.delayed(calc_stats_dt)(df[x]))
        data, statsdata_dt = dask.compute(*data_dt)
        return Intermediate(
            col=x, data=data, stats=statsdata_dt, visual_type="datetime_column",
        )
    else:
        raise UnreachableError


def compute_bivariate(
    df: dd.DataFrame,
    x: str,
    y: str,
    bins: int,
    ngroups: int,
    largest: bool,
    nsubgroups: int,
    timeunit: str,
    agg: str,
    sample_size: int,
    dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    """
    Compute functions for plot(df, x, y)
    Parameters
    ----------
    df
        Dataframe from which plots are to be generated
    x
        A valid column name from the dataframe
    y
        A valid column name from the dataframe
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
    nsubgroups
        If x and y are categorical columns, ngroups refers to
        how many groups to show from column x, and nsubgroups refers to
        how many subgroups to show from column y in each group in column x.
    timeunit
        Defines the time unit to group values over for a datetime column.
        It can be "year", "quarter", "month", "week", "day", "hour",
        "minute", "second". With default value "auto", it will use the
        time unit such that the resulting number of groups is closest to 15.
    agg
        Specify the aggregate to use when aggregating over a numeric column
    sample_size
        Sample size for the scatter plot
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    """
    # pylint: disable=too-many-arguments,too-many-locals

    xtype = detect_dtype(df[x], dtype)
    ytype = detect_dtype(df[y], dtype)
    if (
        is_dtype(xtype, Nominal())
        and is_dtype(ytype, Continuous())
        or is_dtype(xtype, Continuous())
        and is_dtype(ytype, Nominal())
    ):
        x, y = (x, y) if is_dtype(xtype, Nominal()) else (y, x)
        df = drop_null(df[[x, y]])
        df[x] = df[x].apply(str, meta=(x, str))
        # box plot per group
        boxdata = calc_box(df, bins, ngroups, largest, dtype)
        # histogram per group
        hisdata = calc_hist_by_group(df, bins, ngroups, largest)
        return Intermediate(
            x=x, y=y, boxdata=boxdata, histdata=hisdata, visual_type="cat_and_num_cols",
        )
    elif (
        is_dtype(xtype, DateTime())
        and is_dtype(ytype, Continuous())
        or is_dtype(xtype, Continuous())
        and is_dtype(ytype, DateTime())
    ):
        x, y = (x, y) if is_dtype(xtype, DateTime()) else (y, x)
        df = drop_null(df[[x, y]])
        dtnum: List[Any] = []
        # line chart
        dtnum.append(dask.delayed(calc_line_dt)(df, timeunit, agg))
        # box plot
        dtnum.append(dask.delayed(calc_box_dt)(df, timeunit))
        dtnum = dask.compute(*dtnum)
        return Intermediate(
            x=x,
            y=y,
            linedata=dtnum[0],
            boxdata=dtnum[1],
            visual_type="dt_and_num_cols",
        )
    elif (
        is_dtype(xtype, DateTime())
        and is_dtype(ytype, Nominal())
        or is_dtype(xtype, Nominal())
        and is_dtype(ytype, DateTime())
    ):
        x, y = (x, y) if is_dtype(xtype, DateTime()) else (y, x)
        df = drop_null(df[[x, y]])
        df[y] = df[y].apply(str, meta=(y, str))
        dtcat: List[Any] = []
        # line chart
        dtcat.append(
            dask.delayed(calc_line_dt)(df, timeunit, ngroups=ngroups, largest=largest)
        )
        # stacked bar chart
        dtcat.append(dask.delayed(calc_stacked_dt)(df, timeunit, ngroups, largest))
        dtcat = dask.compute(*dtcat)
        return Intermediate(
            x=x,
            y=y,
            linedata=dtcat[0],
            stackdata=dtcat[1],
            visual_type="dt_and_cat_cols",
        )
    elif is_dtype(xtype, Nominal()) and is_dtype(ytype, Nominal()):
        df = drop_null(df[[x, y]])
        df[x] = df[x].apply(str, meta=(x, str))
        df[y] = df[y].apply(str, meta=(y, str))
        # nested bar chart
        nesteddata = calc_nested(df, ngroups, nsubgroups)
        # stacked bar chart
        stackdata = calc_stacked(df, ngroups, nsubgroups)
        # heat map
        heatmapdata = calc_heatmap(df, ngroups, nsubgroups)
        return Intermediate(
            x=x,
            y=y,
            nesteddata=nesteddata,
            stackdata=stackdata,
            heatmapdata=heatmapdata,
            visual_type="two_cat_cols",
        )
    elif is_dtype(xtype, Continuous()) and is_dtype(ytype, Continuous()):
        df = drop_null(df[[x, y]])
        # scatter plot
        scatdata = calc_scatter(df, sample_size)
        # hexbin plot
        hexbindata = df.compute()
        # box plot
        boxdata = calc_box(df, bins)
        return Intermediate(
            x=x,
            y=y,
            scatdata=scatdata,
            boxdata=boxdata,
            hexbindata=hexbindata,
            spl_sz=sample_size,
            visual_type="two_num_cols",
        )
    else:
        raise UnreachableError


def compute_trivariate(
    df: dd.DataFrame,
    x: str,
    y: str,
    z: str,
    ngroups: int,
    largest: bool,
    timeunit: str,
    agg: str,
    dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    """
    Compute functions for plot(df, x, y, z)
    Parameters
    ----------
    df
        Dataframe from which plots are to be generated
    x
        A valid column name from the dataframe
    y
        A valid column name from the dataframe
    z
        A valid column name from the dataframe
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
    agg
        Specify the aggregate to use when aggregating over a numeric column
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    """
    # pylint: disable=too-many-arguments

    xtype = detect_dtype(df[x], dtype)
    ytype = detect_dtype(df[y], dtype)
    ztype = detect_dtype(df[z], dtype)

    if (
        is_dtype(xtype, DateTime())
        and is_dtype(ytype, Nominal())
        and is_dtype(ztype, Continuous())
    ):
        y, z = z, y
    elif (
        is_dtype(xtype, Continuous())
        and is_dtype(ytype, DateTime())
        and is_dtype(ztype, Nominal())
    ):
        x, y = y, x
    elif (
        is_dtype(xtype, Continuous())
        and is_dtype(ytype, Nominal())
        and is_dtype(ztype, DateTime())
    ):
        x, y, z = z, x, y
    elif (
        is_dtype(xtype, Nominal())
        and is_dtype(ytype, DateTime())
        and is_dtype(ztype, Continuous())
    ):
        x, y, z = y, z, x
    elif (
        is_dtype(xtype, Nominal())
        and is_dtype(ytype, Continuous())
        and is_dtype(ztype, DateTime())
    ):
        x, z = z, x
    assert (
        is_dtype(xtype, DateTime())
        and is_dtype(ytype, Continuous())
        and is_dtype(ztype, Nominal())
    ), "x, y, and z must be one each of type datetime, numerical, and categorical"
    df = drop_null(df[[x, y, z]])
    df[z] = df[z].apply(str, meta=(z, str))

    # line chart
    data = dask.compute(dask.delayed(calc_line_dt)(df, timeunit, agg, ngroups, largest))
    return Intermediate(
        x=x, y=y, z=z, agg=agg, data=data[0], visual_type="dt_cat_num_cols",
    )


def calc_line_dt(
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
        agg = (
            "freq" if agg is None else agg
        )  # default agg if unspecified for notational concision

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


def calc_box_dt(
    df: dd.DataFrame, unit: str
) -> Tuple[pd.DataFrame, List[str], List[float], str]:
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
    df = pd.concat([_calc_box_stats(g[1][y], g[0], True) for g in grps], axis=1,)
    df = df.append(pd.Series({c: i + 1 for i, c in enumerate(df.columns)}, name="x",)).T
    # If grouping by week, make the label for the week the beginning Sunday
    df.index = df.index - pd.to_timedelta(6, unit="d") if unit == "week" else df.index
    df.index.name = "grp"
    df = df.reset_index()
    df["grp"] = df["grp"].dt.to_period("S").dt.strftime(DTMAP[unit][2])
    df["x0"], df["x1"] = df["x"] - 0.8, df["x"] - 0.2  # width of whiskers for plotting
    outx, outy = _calc_box_otlrs(df)

    return df, outx, outy, DTMAP[unit][3]


def calc_stacked_dt(
    df: dd.DataFrame, unit: str, ngroups: int, largest: bool,
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
        df_grps, index=grouper, columns=y, aggfunc=len, fill_value=0,
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


def calc_bar(
    srs: dd.Series, ngroups: int, largest: bool
) -> Tuple[dd.DataFrame, dd.core.Scalar, dd.core.Scalar]:
    """
    Calculates the counts of categorical values, the total number of
    categorical values, and the number of non-null cells required
    for a bar chart in plot(df).

    Parameters
    ----------
    srs
        One categorical column
    ngroups
        Number of groups to return
    largest
        If true, show the groups with the largest count,
        else show the groups with the smallest count
    """
    # drop null values
    srs_present = drop_null(srs)
    # number of present (not null) values
    npresent = srs_present.shape[0]
    # counts of unique values in the series
    grps = srs_present.value_counts(sort=False)
    # total number of groups
    ttl_grps = grps.shape[0]
    # select the largest or smallest groups
    fnl_grp_cnts = grps.nlargest(ngroups) if largest else grps.nsmallest(ngroups)

    return fnl_grp_cnts.to_frame(), ttl_grps, npresent


def calc_bar_pie(
    srs: dd.Series, ngroups: int, largest: bool
) -> Tuple[dd.DataFrame, dd.core.Scalar]:
    """
    Calculates the counts of categorical values and the total number of
    categorical values required for the bar and pie charts in plot(df, x).

    Parameters
    ----------
    srs
        One categorical column
    ngroups
        Number of groups to return
    largest
        If true, show the groups with the largest count,
        else show the groups with the smallest count
    """
    # counts of unique values in the series
    grps = srs.value_counts(sort=False)
    # total number of groups
    ttl_grps = grps.shape[0]
    # select the largest or smallest groups
    fnl_grp_cnts = grps.nlargest(ngroups) if largest else grps.nsmallest(ngroups)

    return fnl_grp_cnts.to_frame(), ttl_grps


def calc_word_freq(
    srs: dd.Series,
    top_words: int = 30,
    stopword: bool = True,
    lemmatize: bool = False,
    stem: bool = False,
) -> Tuple[dd.Series, dd.core.Scalar]:
    """
    Parse a categorical column of text data into words, and then
    compute the frequency distribution of words and the total
    number of words.

    Parameters
    ----------
    srs
        One categorical column
    top_words
        Number of highest frequency words to show in the
        wordcloud and word frequency bar chart
    stopword
        If True, remove stop words, else keep them
    lemmatize
        If True, lemmatize the words before computing
        the word frequencies, else don't
    stem
        If True, extract the stem of the words before
        computing the word frequencies, else don't
    """
    # pylint: disable=unnecessary-lambda
    if stopword:
        # use a regex to replace stop words with empty string
        srs = srs.str.replace(r"\b(?:{})\b".format("|".join(english_stopwords)), "")
    # replace all non-alphanumeric characters with an empty string, and convert to lowercase
    srs = srs.str.replace(r"[^\w+ ]", "").str.lower()

    # split each string on whitespace into words then apply "explode()" to "stack" all
    # the words into a series
    # NOTE this is slow. One possibly better solution: after .split(), count the words
    # immediately rather than create a new series with .explode() and apply
    # .value_counts()
    srs = srs.str.split().explode()

    # lemmatize and stem
    if lemmatize or stem:
        srs = srs.dropna()
    if lemmatize:
        lem = WordNetLemmatizer()
        srs = srs.apply(lambda x: lem.lemmatize(x), meta=(srs.name, "object"))
    if stem:
        porter = PorterStemmer()
        srs = srs.apply(lambda x: porter.stem(x), meta=(srs.name, "object"))

    # counts of words, excludes null values
    word_cnts = srs.value_counts(sort=False)
    # total number of words
    nwords = word_cnts.sum()
    # words with the highest frequency
    fnl_word_cnts = word_cnts.nlargest(n=top_words)

    return fnl_word_cnts, nwords


def calc_kde(
    srs: dd.Series, bins: int, minv: float, maxv: float,
) -> Tuple[Tuple[da.core.Array, da.core.Array], np.ndarray]:
    """
    Calculate a density histogram and its corresponding kernel density
    estimate over a given series. The kernel is Gaussian.
    Parameters
    ----------
    data
        One numerical column over which to compute the histogram and kde
    bins
        Number of bins to use in the histogram
    """
    # compute the density histogram
    hist = da.histogram(srs, bins=bins, range=[minv, maxv], density=True)
    # probability density function for the series
    # NOTE gaussian_kde triggers a .compute()
    try:
        kde = gaussian_kde(
            srs.map_partitions(lambda x: x.sample(min(1000, x.shape[0])), meta=srs)
        )
    except np.linalg.LinAlgError:
        kde = None

    return hist, kde


def calc_box_new(srs: dd.Series, qntls: dd.Series) -> Dict[str, Any]:
    """
    Calculate the data required for a box plot
    Parameters
    ----------
    srs
        One numerical column from which to compute the box plot data
    qntls
        Quantiles from the normal Q-Q plot
    """
    # box plot stats
    # inter-quartile range
    # TODO figure out how to extract a scalar from a Dask series without using a function like sum()
    qrtl1 = qntls.loc[0.25].sum()
    qrtl3 = qntls.loc[0.75].sum()
    iqr = qrtl3 - qrtl1
    srs_iqr = srs[srs.between(qrtl1 - 1.5 * iqr, qrtl3 + 1.5 * iqr)]
    # outliers
    otlrs = srs[~srs.between(qrtl1 - 1.5 * iqr, qrtl3 + 1.5 * iqr)]
    # randomly sample at most 100 outliers from each partition without replacement
    otlrs = otlrs.map_partitions(lambda x: x.sample(min(100, x.shape[0])), meta=otlrs)

    box_data = {
        "grp": srs.name,
        "q1": qrtl1,
        "q2": qntls.loc[0.5].sum(),
        "q3": qrtl3,
        "lw": srs_iqr.min(),
        "uw": srs_iqr.max(),
        "otlrs": otlrs.values,
        "x": 1,  # x, x0, and x1 are for plotting the box plot with bokeh
        "x0": 0.2,
        "x1": 0.8,
    }

    return box_data


def calc_stats(
    df: dd.DataFrame, dtype_cnts: Dict[str, int]
) -> Dict[str, Union[int, dd.core.Scalar, Dict[str, int]]]:
    """
    Calculate the statistics for plot(df) from a DataFrame

    Parameters
    ----------
    df
        a DataFrame
    dtype_cnts
        a dictionary that contains the count for each type
    """
    stats = {
        "nrows": df.shape[0],
        "ncols": df.shape[1],
        "npresent_cells": df.count().sum(),
        "nrows_wo_dups": df.drop_duplicates().shape[0],
        "mem_use": df.memory_usage(deep=True).sum(),
        "dtype_cnts": dtype_cnts,
    }

    return stats


def calc_num_stats(srs: dd.Series, nrows: dd.core.Scalar,) -> Dict[str, Any]:
    """
    Calculate statistics for a numerical column
    Parameters
    ----------
    srs
        a numerical column
    nrows
        number of rows in the column before dropping null values
    """
    stats = {
        "nrows": nrows,
        "npresent": srs.shape[0],
        "nunique": srs.nunique(),
        "ninfinite": ((srs == np.inf) | (srs == -np.inf)).sum(),
        "nzero": (srs == 0).sum(),
        "min": srs.min(),
        "max": srs.max(),
        "qntls": srs.quantile(np.linspace(0.01, 0.99, 99)),
        "mean": srs.mean(),
        "std": srs.std(),
        "skew": skew(srs),
        "kurt": kurtosis(srs),
        "mem_use": srs.memory_usage(),
    }

    return stats


def calc_cat_stats(
    srs: dd.Series, nrows: int, bins: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Calculate stats for a categorical column
    Parameters
    ----------
    srs
        a categorical column
    nrows
        number of rows before dropping null values
    bins
        number of bins for the category length frequency histogram
    """
    # overview stats
    stats = {
        "nrows": nrows,
        "npresent": srs.shape[0],
        "nunique": srs.nunique(),
        "mem_use": srs.memory_usage(),
        "first_rows": srs.loc[:4],
    }
    # length stats
    lengths = srs.str.len()
    minv, maxv = lengths.min(), lengths.max()
    hist = da.histogram(lengths.values, bins=bins, range=[minv, maxv])
    length_stats = {
        "Mean": lengths.mean(),
        "Median": lengths.quantile(0.5),
        "Minimum": minv,
        "Maximum": maxv,
        "hist": hist,
    }
    # letter stats
    letter_stats = {
        "Count": srs.str.count(r"[a-zA-Z]").sum(),
        "Lowercase Letter": srs.str.count(r"[a-z]").sum(),
        "Space Separator": srs.str.count(r"[ ]").sum(),
        "Uppercase Letter": srs.str.count(r"[A-Z]").sum(),
        "Dash Punctuation": srs.str.count(r"[-]").sum(),
        "Decimal Number": srs.str.count(r"[0-9]").sum(),
    }

    return stats, length_stats, letter_stats


def calc_box(
    df: dd.DataFrame,
    bins: int,
    ngroups: int = 10,
    largest: bool = True,
    dtype: Optional[DTypeDef] = None,
) -> Tuple[pd.DataFrame, List[str], List[float], Optional[Dict[str, int]]]:
    """
    Compute a box plot over either
        1) the values in one column
        2) the values corresponding to groups in another column
        3) the values corresponding to binning another column
    Parameters
    ----------
    df
        Dataframe with one or two columns
    bins
        Number of bins to use if df has two numerical columns
    ngroups
        Number of groups to show if df has a categorical and numerical column
    largest
        When calculating a box plot per group, select the largest or smallest groups
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    Returns
    -------
    Tuple[pd.DataFrame, List[str], List[float], Dict[str, int]]
        The box plot statistics in a dataframe, a list of the outlier
        groups and another list of the outlier values, a dictionary
        logging the sampled group output
    """
    # pylint: disable=too-many-locals
    grp_cnt_stats = None  # to inform the user of sampled output

    x = df.columns[0]
    if len(df.columns) == 1:
        df = _calc_box_stats(df[x], x)
    else:
        y = df.columns[1]
        if is_dtype(detect_dtype(df[x], dtype), Continuous()) and is_dtype(
            detect_dtype(df[y], dtype), Continuous()
        ):
            minv, maxv, cnt = dask.compute(df[x].min(), df[x].max(), df[x].nunique())
            bins = cnt if cnt < bins else bins
            endpts = np.linspace(minv, maxv, num=bins + 1)
            # calculate a box plot over each bin
            df = dd.concat(
                [
                    _calc_box_stats(
                        df[(df[x] >= endpts[i]) & (df[x] < endpts[i + 1])][y],
                        f"[{endpts[i]},{endpts[i + 1]})",
                    )
                    if i != len(endpts) - 2
                    else _calc_box_stats(
                        df[(df[x] >= endpts[i]) & (df[x] <= endpts[i + 1])][y],
                        f"[{endpts[i]},{endpts[i + 1]}]",
                    )
                    for i in range(len(endpts) - 1)
                ],
                axis=1,
            ).compute()
            endpts_df = pd.DataFrame(
                [endpts[:-1], endpts[1:]], ["lb", "ub"], df.columns
            )
            df = pd.concat([df, endpts_df], axis=0)
        else:
            df, grp_cnt_stats, largest_grps = _calc_groups(df, x, ngroups, largest)
            # calculate a box plot over each group
            df = dd.concat(
                [_calc_box_stats(df[df[x] == grp][y], grp) for grp in largest_grps],
                axis=1,
            ).compute()

    df = df.append(pd.Series({c: i + 1 for i, c in enumerate(df.columns)}, name="x",)).T
    df.index.name = "grp"
    df = df.reset_index()
    df["x0"], df["x1"] = df["x"] - 0.8, df["x"] - 0.2  # width of whiskers for plotting
    outx, outy = _calc_box_otlrs(df)

    return df, outx, outy, grp_cnt_stats


def calc_hist_by_group(
    df: dd.DataFrame, bins: int, ngroups: int, largest: bool
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Compute a histogram over the values corresponding to the groups in another column
    Parameters
    ----------
    df
        Dataframe with one categorical and one numerical column
    bins
        Number of bins to use in the histogram
    ngroups
        Number of groups to show from the categorical column
    largest
        Select the largest or smallest groups
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        The histograms in a dataframe and a dictionary
        logging the sampled group output
    """
    # pylint: disable=too-many-locals

    hist_dict: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = dict()
    hist_lst: List[Tuple[np.ndarray, np.ndarray, List[str]]] = list()
    df, grp_cnt_stats, largest_grps = _calc_groups(df, df.columns[0], ngroups, largest)

    # create a histogram for each group
    groups = df.groupby([df.columns[0]])
    minv, maxv = dask.compute(df[df.columns[1]].min(), df[df.columns[1]].max())
    for grp in largest_grps:
        grp_srs = groups.get_group(grp)[df.columns[1]]
        hist_arr, bins_arr = da.histogram(grp_srs, range=[minv, maxv], bins=bins)
        intervals = _format_bin_intervals(bins_arr)
        hist_lst.append((hist_arr, bins_arr, intervals))

    hist_lst = dask.compute(*hist_lst)

    for elem in zip(largest_grps, hist_lst):
        hist_dict[elem[0]] = elem[1]

    return hist_dict, grp_cnt_stats


def calc_scatter(df: dd.DataFrame, sample_size: int) -> pd.DataFrame:
    """
    Extracts the points to use in a scatter plot
    Parameters
    ----------
    df
        Dataframe with two numerical columns
    sample_size
        the number of points to randomly sample in the scatter plot
    Returns
    -------
    pd.DataFrame
        A dataframe containing the scatter points
    """
    if len(df) > sample_size:
        df = df.sample(frac=sample_size / len(df))
    return df.compute()


def calc_nested(
    df: dd.DataFrame, ngroups: int, nsubgroups: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Calculate a nested bar chart of the counts of two columns
    Parameters
    ----------
    df
        Dataframe with two categorical columns
    ngroups
        Number of groups to show from the first column
    nsubgroups
        Number of subgroups (from the second column) to show in each group
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        The bar chart counts in a dataframe and a dictionary
        logging the sampled group output
    """
    x, y = df.columns[0], df.columns[1]
    df, grp_cnt_stats, _ = _calc_groups(df, x, ngroups)

    df2 = df.groupby([x, y]).size().reset_index()
    max_subcol_cnt = df2.groupby(x).size().max().compute()
    df2.columns = [x, y, "cnt"]
    df_res = (
        df2.groupby(x)[[y, "cnt"]]
        .apply(
            lambda x: x.nlargest(n=nsubgroups, columns="cnt"),
            meta=({y: "f8", "cnt": "i8"}),
        )
        .reset_index()
        .compute()
    )
    df_res["grp_names"] = list(zip(df_res[x], df_res[y]))
    df_res = df_res.drop([x, "level_1", y], axis=1)
    grp_cnt_stats[f"{y}_ttl"] = max_subcol_cnt
    grp_cnt_stats[f"{y}_shw"] = min(max_subcol_cnt, nsubgroups)

    return df_res, grp_cnt_stats


def calc_stacked(
    df: dd.DataFrame, ngroups: int, nsubgroups: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Calculate a stacked bar chart of the counts of two columns
    Parameters
    ----------
    df
        two categorical columns
    ngroups
        number of groups to show from the first column
    nsubgroups
        number of subgroups (from the second column) to show in each group
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        The bar chart counts in a dataframe and a dictionary
        logging the sampled group output
    """
    x, y = df.columns[0], df.columns[1]
    df, grp_cnt_stats, largest_grps = _calc_groups(df, x, ngroups)

    fin_df = pd.DataFrame()
    for grp in largest_grps:
        df_grp = df[df[x] == grp]
        df_res = df_grp.groupby(y).size().nlargest(n=nsubgroups) / len(df_grp) * 100
        df_res = df_res.to_frame().compute().T
        df_res.columns = list(df_res.columns)
        df_res["Others"] = 100 - df_res.sum(axis=1)
        fin_df = fin_df.append(df_res, sort=False)

    fin_df = fin_df.fillna(value=0)
    others = fin_df.pop("Others")
    if others.sum() > 1e-4:
        fin_df["Others"] = others
    fin_df.index = list(largest_grps)
    return fin_df, grp_cnt_stats


def calc_heatmap(
    df: dd.DataFrame, ngroups: int, nsubgroups: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Calculate a heatmap of the counts of two columns
    Parameters
    ----------
    df
        Dataframe with two categorical columns
    ngroups
        Number of groups to show from the first column
    nsubgroups
        Number of subgroups (from the second column) to show in each group
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        The heatmap counts in a dataframe and a dictionary
        logging the sampled group output
    """
    x, y = df.columns[0], df.columns[1]
    df, grp_cnt_stats, _ = _calc_groups(df, x, ngroups)

    srs = df.groupby(y).size()
    srs_lrgst = srs.nlargest(n=nsubgroups)
    largest_subgrps = list(srs_lrgst.index.compute())
    df = df[df[y].isin(largest_subgrps)]

    df_res = df.groupby([x, y]).size().reset_index().compute()
    df_res.columns = ["x", "y", "cnt"]
    df_res = pd.pivot_table(
        df_res, index=["x", "y"], values="cnt", fill_value=0, aggfunc=np.sum,
    ).reset_index()

    grp_cnt_stats[f"{y}_ttl"] = len(srs.index.compute())
    grp_cnt_stats[f"{y}_shw"] = len(largest_subgrps)

    return df_res, grp_cnt_stats


def calc_stats_dt(srs: dd.Series) -> Dict[str, str]:
    """
    Calculate stats from a datetime column
    Parameters
    ----------
    srs
        a datetime column
    Returns
    -------
    Dict[str, str]
        Dictionary that contains Overview
    """
    size = len(srs)  # include nan
    count = srs.count()  # exclude nan
    uniq_count = srs.nunique()
    overview_dict = {
        "Distinct Count": uniq_count,
        "Unique (%)": uniq_count / count,
        "Missing": size - count,
        "Missing (%)": 1 - (count / size),
        "Memory Size": srs.memory_usage(),
        "Minimum": srs.min(),
        "Maximum": srs.max(),
    }

    return overview_dict


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


def _calc_groups(
    df: dd.DataFrame, x: str, ngroups: int, largest: bool = True
) -> Tuple[dd.DataFrame, Dict[str, int], List[str]]:
    """
    Auxillary function to parse the dataframe to consist of only the
    groups with the largest counts
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


def _format_bin_intervals(bins_arr: np.ndarray) -> List[str]:
    """
    Auxillary function to format bin intervals in a histogram
    """
    bins_arr = np.round(bins_arr, 3)
    bins_arr = [int(val) if float(val).is_integer() else val for val in bins_arr]
    intervals = [
        f"[{bins_arr[i]}, {bins_arr[i + 1]})" for i in range(len(bins_arr) - 2)
    ]
    intervals.append(f"[{bins_arr[-2]},{bins_arr[-1]}]")
    return intervals


def _get_timeunit(min_time: pd.Timestamp, max_time: pd.Timestamp, dflt: int) -> str:
    """
    Auxillary function to find an appropriate time unit. Will find the
    time unit such that the number of time units are closest to dflt.
    """
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
